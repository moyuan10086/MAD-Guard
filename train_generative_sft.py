import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_npu  # type: ignore
except ImportError:
    torch_npu = None
from PIL import Image
from torch.utils.data import DataLoader, Dataset
try:
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
        get_cosine_schedule_with_warmup
    )
    from peft import LoraConfig, get_peft_model
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None
    get_cosine_schedule_with_warmup = None
    LoraConfig = None
    get_peft_model = None


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL with Generative SFT on Ascend 910C")
    parser.add_argument("--model_path", type=str, default="/opt/models/qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="/opt/datasets/FakeClue/train.json")
    parser.add_argument("--image_dir", type=str, default="/opt/datasets/FakeClue/train")
    parser.add_argument("--output_dir", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/generative_sft")
    parser.add_argument("--device_id", type=int, default=1, help="NPU ID (default: 1)")
    parser.add_argument("--max_samples", type=int, default=2400, help="Number of training samples (-1 for all)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=5)
    return parser.parse_args()


class FakeClueGenerativeDataset(Dataset):
    def __init__(self, data_path: str, image_dir: str, max_samples: int = 2400):
        print(f"Loading annotations from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.image_dir = Path(image_dir)
        valid_items = []
        for item in raw_data:
            img_rel = item.get("image")
            img_full = self.image_dir / img_rel
            if img_full.exists():
                valid_items.append(item)

        print(f"Found {len(valid_items)} valid samples with existing image files.")

        if 0 < max_samples < len(valid_items):
            # Stratified balanced sampling across (label, cate) matching MAD-Guard exactly
            random.seed(42)
            groups = {}
            for s in valid_items:
                key = (s.get("label", 0), s.get("cate", "unknown"))
                groups.setdefault(key, []).append(s)

            per_group = max(1, max_samples // len(groups))
            sampled = []
            for k, g_items in groups.items():
                k_sample = min(len(g_items), per_group)
                sampled.extend(random.sample(g_items, k_sample))

            if len(sampled) < max_samples:
                sampled_set = set(id(x) for x in sampled)
                remaining = [x for x in valid_items if id(x) not in sampled_set]
                extra = min(len(remaining), max_samples - len(sampled))
                sampled.extend(random.sample(remaining, extra))

            random.shuffle(sampled)
            self.samples = sampled[:max_samples]
        else:
            self.samples = valid_items

        fake_cnt = sum(1 for x in self.samples if x["label"] == 0)
        real_cnt = sum(1 for x in self.samples if x["label"] == 1)
        print(f"Loaded {len(self.samples)} samples. Distribution: Fake={fake_cnt} ({fake_cnt/len(self.samples)*100:.1f}%), Real={real_cnt} ({real_cnt/len(self.samples)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = str(self.image_dir / item["image"])
        image = Image.open(img_path).convert("RGB")

        # In FakeClue: label 0 is fake, label 1 is real
        # Target token string: "REAL" if label==1 else "FAKE"
        target_str = "REAL" if item.get("label", 0) == 1 else "FAKE"
        prompt_text = "Is this image REAL or FAKE? Output ONLY the word REAL or FAKE.\nAnswer:"

        return {
            "image": image,
            "prompt_text": prompt_text,
            "target_str": target_str,
            "label": item.get("label", 0)
        }


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        full_texts = []
        prompt_texts = []

        for b in batch:
            # Full dialogue (user + assistant)
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": b["image"]},
                        {"type": "text", "text": b["prompt_text"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": b["target_str"]}
                    ]
                }
            ]
            full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            full_texts.append(full_text)

            # Prompt only (for masking loss)
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": b["image"]},
                        {"type": "text", "text": b["prompt_text"]}
                    ]
                }
            ]
            prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_texts.append(prompt_text)

        inputs = processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        prompt_inputs = processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        labels = inputs.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask prompt prefix so loss is ONLY computed on the assistant target tokens
        for i in range(len(batch)):
            p_len = prompt_inputs.input_ids[i].shape[0]
            mask_len = min(p_len, labels.shape[1])
            labels[i, :mask_len] = -100

        inputs["labels"] = labels
        return inputs
    return collate_fn


def main():
    args = parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    device = "npu:0"

    print("=" * 70)
    print("🚀 Qwen3-VL-8B Generative SFT Training on Ascend 910C")
    print(f"   Target NPU Device: {args.device_id} (mapped to {device})")
    print(f"   Model Backbone: {args.model_path}")
    print(f"   Dataset: {args.data_path} (samples={args.max_samples})")
    print(f"   Batch size: {args.batch_size} x Accum {args.grad_accum} = Effective {args.batch_size * args.grad_accum}")
    print(f"   Learning Rate: {args.lr:.2e}")
    print(f"   LoRA Config: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"   Output Directory: {args.output_dir}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[2/5] Initializing Qwen3VLForConditionalGeneration...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print(f"Model initialized in {time.time() - t0:.2f}s")
    model.print_trainable_parameters()

    print("[3/5] Loading FakeClue dataset...")
    dataset = FakeClueGenerativeDataset(args.data_path, args.image_dir, max_samples=args.max_samples)
    collate_fn = make_collate_fn(processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    total_steps = (len(dataloader) // args.grad_accum) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=max(1, total_steps)
    )

    print(f"[4/5] Training Configuration:")
    print(f"   Steps per epoch: {len(dataloader) // args.grad_accum}")
    print(f"   Total optimizer steps: {total_steps}")
    print(f"   Warmup steps: {max(1, int(total_steps * 0.1))}")

    print(f"[5/5] Starting Generative SFT Training Loop...")
    global_step = 0
    start_train_time = time.time()

    for epoch in range(args.epochs):
        print(f"\n======== Epoch {epoch+1}/{args.epochs} ========")
        model.train()
        accum_loss = 0.0
        step_samples = 0
        step_start_time = time.time()

        for step, batch in enumerate(dataloader):
            input_labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass without passing labels directly to avoid HuggingFace allocating full L x V float logits
            outputs = model(**batch)
            logits = outputs.logits  # (B, L, V)
            
            # Efficient selective loss: only compute cross-entropy on target positions (labels != -100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_labels[..., 1:].contiguous()
            
            active_mask = (shift_labels != -100)
            if active_mask.any():
                active_logits = shift_logits[active_mask]  # (N_target_tokens, V)
                active_labels = shift_labels[active_mask]  # (N_target_tokens)
                loss = F.cross_entropy(active_logits, active_labels) / args.grad_accum
                loss.backward()
                accum_loss += loss.item() * args.grad_accum
            else:
                loss = None

            step_samples += args.batch_size

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    elapsed = time.time() - step_start_time
                    speed = step_samples / elapsed if elapsed > 0 else 0.0
                    cur_lr = lr_scheduler.get_last_lr()[0]
                    print(
                        f"Step {global_step:4d}/{total_steps:4d} | "
                        f"Loss: {accum_loss / args.log_steps:.4f} | "
                        f"LR: {cur_lr:.2e} | "
                        f"Speed: {speed:.1f} samples/s",
                        flush=True
                    )
                    accum_loss = 0.0
                    step_samples = 0
                    step_start_time = time.time()

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"[*] Saving checkpoint to {ckpt_dir}...")
                    model.save_pretrained(ckpt_dir)
                    processor.save_pretrained(ckpt_dir)
                    print(f"[✓] Checkpoint saved successfully.")

    final_dir = os.path.join(args.output_dir, "final_adapter")
    print(f"\n[✓] Training complete! Saving final adapter to {final_dir}...")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    total_time = time.time() - start_train_time
    print(f"Total training time: {total_time:.2f}s ({total_time / 60:.1f} min)")

    meta = {
        "model": "Qwen3-VL-8B Generative SFT",
        "dataset_samples": args.max_samples,
        "epochs": args.epochs,
        "total_steps": global_step,
        "total_time_seconds": round(total_time, 2),
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "device": f"Ascend 910C (NPU {args.device_id})"
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

