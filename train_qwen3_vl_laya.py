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
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup  # type: ignore
except ImportError:
    AutoProcessor = None
    get_cosine_schedule_with_warmup = None

# Import local architecture
from qwen3_vl_laya_model import Qwen3VLLayaModel, CATE2ID, CATE_LIST


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL with Laya Decision Heads on Ascend 910C")
    parser.add_argument("--model_path", type=str, default="/opt/models/qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="/opt/datasets/FakeClue/train.json")
    parser.add_argument("--image_dir", type=str, default="/opt/datasets/FakeClue/train")
    parser.add_argument("--output_dir", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head")
    parser.add_argument("--device_id", type=int, default=1, help="NPU ID (default: 1)")
    parser.add_argument("--max_samples", type=int, default=2400, help="Number of training samples (-1 for all)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--log_steps", type=int, default=5)
    return parser.parse_args()


class FakeClueLayaDataset(Dataset):
    def __init__(self, data_path: str, image_dir: str, max_samples: int = -1):
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
            # Stratified balanced sampling across (label, cate)
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
        # In Safety Audit: 1.0 is Risk/Fake, 0.0 is Safe/Real
        is_fake = 1.0 if item.get("label", 0) == 0 else 0.0
        act_label = 1 if item.get("label", 0) == 0 else 0  # 1: intercept, 0: pass

        cate = item.get("cate", "deepfake")
        cate_id = CATE2ID.get(cate, 0)

        prompt_text = (
            "You are an AI Multimodal Safety & Forensic Auditor. "
            "Inspect this image carefully for visual manipulation artifacts, synthetic textures, edge anomalies, "
            "or authentic physical patterns. Decide whether this media is Authentic or AI-Manipulated (Deepfake)."
        )

        return {
            "image": image,
            "text": prompt_text,
            "fake_label": is_fake,
            "cate_id": cate_id,
            "act_label": act_label
        }


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        texts = []
        for b in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": b["image"]},
                        {"type": "text", "text": b["text"]}
                    ]
                }
            ]
            full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(full_text)

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        fake_labels = torch.tensor([b["fake_label"] for b in batch], dtype=torch.float32)
        cate_labels = torch.tensor([b["cate_id"] for b in batch], dtype=torch.long)
        act_labels = torch.tensor([b["act_label"] for b in batch], dtype=torch.long)

        return {
            "inputs": inputs,
            "fake_labels": fake_labels,
            "cate_labels": cate_labels,
            "act_labels": act_labels
        }
    return collate_fn


def main():
    args = parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    device = "npu:0"

    print("=" * 70)
    print("🚀 Qwen3-VL + Laya Calibrated Decision Head Training on Ascend 910C")
    print(f"   Target NPU Device: {args.device_id} (mapped to {device})")
    print(f"   Model Backbone: {args.model_path}")
    print(f"   Dataset: {args.data_path} (samples={args.max_samples})")
    print(f"   Batch size: {args.batch_size} x Accum {args.grad_accum} = Effective {args.batch_size * args.grad_accum}")
    print(f"   Backbone LR: {args.lr_backbone:.2e} | Laya Head LR: {args.lr_head:.2e}")
    print(f"   Output Directory: {args.output_dir}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[2/5] Initializing Qwen3VLLayaModel...")
    t0 = time.time()
    model = Qwen3VLLayaModel(
        base_model_path=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_categories=len(CATE_LIST),
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    print(f"Model initialized in {time.time() - t0:.2f}s")
    model.backbone.print_trainable_parameters()

    # Move decision head to NPU device
    model.decision_head.to(device)

    print("[3/5] Loading FakeClue dataset...")
    dataset = FakeClueLayaDataset(args.data_path, args.image_dir, max_samples=args.max_samples)
    collate_fn = make_collate_fn(processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    # Differential learning rate setup (Laya paradigm)
    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.decision_head.named_parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 0.01},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 0.01}
    ])

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

    print("\n[5/5] Starting Multi-Task Decision Training Loop...")
    model.train()
    global_step = 0
    accum_bce = 0.0
    accum_ce = 0.0
    accum_act = 0.0
    accum_loss = 0.0
    correct_act = 0
    total_samples = 0
    start_time = time.time()
    step_start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n======== Epoch {epoch}/{args.epochs} ========")
        for step, batch in enumerate(dataloader):
            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            fake_labels = batch["fake_labels"].to(device)
            cate_labels = batch["cate_labels"].to(device)
            act_labels = batch["act_labels"].to(device)

            out = model(**inputs)

            loss_noul = F.binary_cross_entropy_with_logits(out["noul_logit"], fake_labels)
            loss_choice = F.cross_entropy(out["choice_logits"], cate_labels)
            loss_act = F.cross_entropy(out["act_logits"], act_labels)

            total_step_loss = loss_noul + 0.5 * loss_choice + 0.5 * loss_act
            loss = total_step_loss / args.grad_accum
            loss.backward()

            accum_bce += loss_noul.item()
            accum_ce += loss_choice.item()
            accum_act += loss_act.item()
            accum_loss += total_step_loss.item()

            preds_act = out["act_logits"].argmax(dim=-1)
            correct_act += (preds_act == act_labels).sum().item()
            total_samples += len(act_labels)

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_steps == 0:
                    dt = time.time() - step_start
                    step_start = time.time()
                    speed = (args.batch_size * args.grad_accum * args.log_steps) / max(0.001, dt)
                    avg_loss = accum_loss / (args.log_steps * args.grad_accum)
                    avg_bce = accum_bce / (args.log_steps * args.grad_accum)
                    avg_ce = accum_ce / (args.log_steps * args.grad_accum)
                    avg_act = accum_act / (args.log_steps * args.grad_accum)
                    acc = (correct_act / max(1, total_samples)) * 100.0
                    temp_val = float(model.decision_head.temperature.data.item())

                    print(
                        f"Step {global_step:4d}/{total_steps:4d} | "
                        f"Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, CE: {avg_ce:.4f}, Act: {avg_act:.4f}) | "
                        f"Audit Acc: {acc:.1f}% | "
                        f"Temp: {temp_val:.2f} | "
                        f"Speed: {speed:.1f} samples/s"
                    )
                    accum_bce = 0.0
                    accum_ce = 0.0
                    accum_act = 0.0
                    accum_loss = 0.0
                    correct_act = 0
                    total_samples = 0

                if global_step % args.save_steps == 0 or global_step == total_steps:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"[*] Saving checkpoint to {ckpt_dir}...")
                    model.save_laya_model(ckpt_dir, processor=processor)

    final_dir = os.path.join(args.output_dir, "final_model")
    print(f"\n✅ Fine-tuning complete! Saving final model to {final_dir}...")
    model.save_laya_model(final_dir, processor=processor)
    print(f"Total training time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
