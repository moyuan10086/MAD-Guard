import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
try:
    import torch_npu  # type: ignore
except ImportError:
    torch_npu = None
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Generative SFT model vs Baselines on Ascend 910C")
    parser.add_argument("--base_model_path", type=str, default="/opt/models/qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lora_adapter_path", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/generative_sft/final_adapter")
    parser.add_argument("--data_path", type=str, default="/opt/datasets/FakeClue/test.json")
    parser.add_argument("--image_dir", type=str, default="/opt/datasets/FakeClue/test")
    parser.add_argument("--output_file", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/generative_sft/generative_sft_eval_results.json")
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--benchmark", type=str, default="genimage", choices=["all", "genimage", "ff++", "chameleon", "doc", "satellite"])
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="-1 for full benchmark")
    return parser.parse_args()


def compute_ece(probs, labels, n_bins=15):
    probs = np.array(probs)
    labels = np.array(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def main():
    args = parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    device = "npu:0"

    print("=" * 70)
    print("🔬 Evaluating Generative SFT Baseline on Ascend 910C")
    print(f"   Visible NPU: {args.device_id} (logical: {device})")
    print(f"   Base Model: {args.base_model_path}")
    print(f"   LoRA Adapter: {args.lora_adapter_path}")
    print(f"   Benchmark Target: {args.benchmark}")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(args.base_model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    if args.lora_adapter_path and os.path.exists(args.lora_adapter_path):
        print(f"Loading Generative SFT LoRA adapter from {args.lora_adapter_path}...")
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
    model.eval()

    tokenizer = processor.tokenizer
    real_tokens = [tokenizer.encode(w, add_special_tokens=False)[-1] for w in ["REAL", " REAL", "real", " real"]]
    fake_tokens = [tokenizer.encode(w, add_special_tokens=False)[-1] for w in ["FAKE", " FAKE", "fake", " fake"]]

    image_dir = Path(args.image_dir)
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_test = json.load(f)

    if args.benchmark == "genimage":
        samples = [x for x in all_test if x.get("image", "").startswith("genimage/")]
    elif args.benchmark == "ff++":
        samples = [x for x in all_test if x.get("image", "").startswith("ff++/") or x.get("image", "").startswith("FaceForensics/")]
    elif args.benchmark == "chameleon":
        samples = [x for x in all_test if x.get("image", "").startswith("chameleon/")]
    elif args.benchmark == "doc":
        samples = [x for x in all_test if x.get("image", "").startswith("doc/")]
    elif args.benchmark == "satellite":
        samples = [x for x in all_test if x.get("image", "").startswith("satellite/")]
    else:
        samples = all_test

    if 0 < args.max_eval_samples < len(samples):
        samples = samples[:args.max_eval_samples]

    print(f"Evaluating {len(samples)} samples on benchmark '{args.benchmark}'...")

    # Warmup
    for w_item in samples[:2]:
        w_img = Image.open(image_dir / w_item["image"]).convert("RGB")
        w_msgs = [{"role": "user", "content": [{"type": "image", "image": w_img}, {"type": "text", "text": "Is this image REAL or FAKE? Output ONLY the word REAL or FAKE.\nAnswer:"}]}]
        w_prompt = processor.apply_chat_template(w_msgs, tokenize=False, add_generation_prompt=True)
        w_inp = processor(text=[w_prompt], images=w_img, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.generate(**w_inp, max_new_tokens=4, do_sample=False)
    if torch_npu:
        torch_npu.npu.synchronize()

    latencies_total = []
    latencies_ttft = []
    latencies_decode = []
    preds_text = []
    preds_logit = []
    p_fakes = []
    ground_truths = []
    gen_texts = []

    start_eval_time = time.time()
    for idx, item in enumerate(samples):
        img_path = image_dir / item["image"]
        label = item["label"]  # 0 is fake, 1 is real
        gt_is_fake = 1 if label == 0 else 0
        ground_truths.append(gt_is_fake)

        img = Image.open(img_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Is this image REAL or FAKE? Output ONLY the word REAL or FAKE.\nAnswer:"}]}]
        prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[prompt_text], images=img, return_tensors="pt").to(device)

        if torch_npu:
            torch_npu.npu.synchronize()
        t_start = time.time()

        with torch.no_grad():
            out_prefill = model(**inp)
            if torch_npu:
                torch_npu.npu.synchronize()
            t_prefill = time.time()

            logits = out_prefill.logits[0, -1, :]
            l_r = max(logits[t].item() for t in real_tokens)
            l_f = max(logits[t].item() for t in fake_tokens)
            exp_r, exp_f = np.exp(l_r - max(l_r, l_f)), np.exp(l_f - max(l_r, l_f))
            p_fake = exp_f / (exp_r + exp_f)
            p_fakes.append(p_fake)
            pred_logit = 1 if p_fake >= 0.5 else 0
            preds_logit.append(pred_logit)

            gen_out = model.generate(**inp, max_new_tokens=4, do_sample=False)
            if torch_npu:
                torch_npu.npu.synchronize()
            t_end = time.time()

        ttft_ms = (t_prefill - t_start) * 1000.0
        total_ms = (t_end - t_start) * 1000.0
        dec_ms = total_ms - ttft_ms

        latencies_ttft.append(ttft_ms)
        latencies_total.append(total_ms)
        latencies_decode.append(dec_ms)

        out_tokens = gen_out[0][inp.input_ids.shape[1]:]
        gen_text = processor.decode(out_tokens, skip_special_tokens=True).strip().upper()
        if len(gen_texts) < 50:
            gen_texts.append(gen_text)

        if "FAKE" in gen_text:
            preds_text.append(1)
        elif "REAL" in gen_text:
            preds_text.append(0)
        else:
            preds_text.append(pred_logit)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            cur_acc = np.mean(np.array(preds_text) == np.array(ground_truths[:len(preds_text)]))
            print(f"[{idx+1}/{len(samples)}] Acc(Text): {cur_acc*100:.2f}% | Latency: {np.mean(latencies_total):.1f}ms", flush=True)

    y_true = np.array(ground_truths)
    y_pred = np.array(preds_text)
    y_score = np.array(p_fakes)

    acc = float(np.mean(y_true == y_pred))
    acc_logit = float(np.mean(y_true == np.array(preds_logit)))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos > 0 and n_neg > 0:
        ranks = np.argsort(np.argsort(y_score)) + 1
        pos_rank_sum = np.sum(ranks[y_true == 1])
        roc_auc = float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    else:
        roc_auc = 1.0 if (n_pos == len(y_true) or n_neg == len(y_true)) else 0.5

    ece = compute_ece(y_score, y_true)

    # Memory usage
    peak_vram_mb = 0.0
    if torch_npu:
        peak_vram_mb = float(torch_npu.npu.max_memory_allocated() / (1024 * 1024))

    result = {
        "benchmark": args.benchmark,
        "model": "Qwen3-VL-8B Supervised AR-SFT",
        "num_samples": len(samples),
        "num_fake": int(n_pos),
        "num_real": int(n_neg),
        "accuracy_text": round(acc, 4),
        "accuracy_logit": round(acc_logit, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "ece": round(ece, 4),
        "mean_latency_ms": round(float(np.mean(latencies_total)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies_total, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_total, 95)), 2),
        "mean_ttft_ms": round(float(np.mean(latencies_ttft)), 2),
        "mean_decode_latency_ms": round(float(np.mean(latencies_decode)), 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "sample_generated_texts": gen_texts[:10]
    }

    print("\n" + "=" * 65)
    print("🏆 FINAL EVALUATION RESULT SUMMARY:")
    print("=" * 65)
    print(json.dumps(result, indent=2))
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

