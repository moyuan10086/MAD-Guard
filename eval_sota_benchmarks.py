import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from qwen3_vl_laya_model import Qwen3VLLayaModel, CATE2ID, ID2CATE, CATE_LIST


def calc_metrics(y_true, y_pred, y_score):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_score = np.array(y_score, dtype=float)

    total = len(y_true)
    if total == 0:
        return {}

    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos > 0 and n_neg > 0:
        ranks = np.argsort(np.argsort(y_score)) + 1
        pos_rank_sum = np.sum(ranks[y_true == 1])
        roc_auc = float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    else:
        roc_auc = 1.0 if n_pos == total or n_neg == total else 0.5

    return {
        "num_samples": total,
        "num_fake": int(n_pos),
        "num_real": int(n_neg),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4)
    }


class SOTABenchmarkDataset(Dataset):
    def __init__(self, data_path: str, image_dir: str):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.image_dir = Path(image_dir)
        self.samples = []
        for item in raw_data:
            img_rel = item.get("image", "")
            prefix = img_rel.split("/")[0] if "/" in img_rel else "unknown"

            img_full = self.image_dir / img_rel
            if img_full.exists():
                item_copy = dict(item)
                item_copy["benchmark"] = prefix
                self.samples.append(item_copy)

        print(f"Loaded {len(self.samples)} valid samples across all SOTA benchmarks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = str(self.image_dir / item["image"])
        image = Image.open(img_path).convert("RGB")

        # FakeClue: 0=fake, 1=real -> Audit: 1=fake, 0=real
        is_fake = 1 if item.get("label", 0) == 0 else 0
        cate = item.get("cate", "deepfake")
        cate_id = CATE2ID.get(cate, 0)

        prompt_text = (
            "You are an AI Multimodal Safety & Forensic Auditor. "
            "Inspect this image carefully for visual manipulation artifacts, synthetic textures, edge anomalies, "
            "or authentic physical patterns. Decide whether this media is Authentic or AI-Manipulated (Deepfake)."
        )

        return {
            "image": image,
            "image_path": item["image"],
            "benchmark": item.get("benchmark", "unknown"),
            "text": prompt_text,
            "fake_label": is_fake,
            "cate_id": cate_id,
            "cate": cate
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
        return {
            "inputs": inputs,
            "fake_labels": [b["fake_label"] for b in batch],
            "benchmarks": [b["benchmark"] for b in batch],
            "cates": [b["cate"] for b in batch],
            "image_paths": [b["image_path"] for b in batch]
        }
    return collate_fn


def evaluate_dataset(model, loader, device):
    benchmark_records = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_score": []})
    all_y_true = []
    all_y_pred = []
    all_y_score = []
    latencies = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            bs = len(batch["fake_labels"])

            torch.npu.synchronize()
            t0 = time.time()
            out = model(**inputs)
            torch.npu.synchronize()
            dt_ms = (time.time() - t0) * 1000.0 / bs

            probs = out["fake_prob"].cpu().numpy()

            for i in range(bs):
                prob = float(probs[i])
                pred = 1 if prob > 0.5 else 0
                true_label = int(batch["fake_labels"][i])
                bmark = batch["benchmarks"][i]

                benchmark_records[bmark]["y_true"].append(true_label)
                benchmark_records[bmark]["y_pred"].append(pred)
                benchmark_records[bmark]["y_score"].append(prob)

                all_y_true.append(true_label)
                all_y_pred.append(pred)
                all_y_score.append(prob)
                latencies.append(dt_ms)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                torch.npu.empty_cache()
                print(f"Evaluated {(batch_idx + 1) * loader.batch_size}/{len(loader.dataset)} samples | Latency: {dt_ms:.1f} ms/sample")

    summary_per_bmark = {}
    for bmark, data in benchmark_records.items():
        summary_per_bmark[bmark] = calc_metrics(data["y_true"], data["y_pred"], data["y_score"])

    overall_metrics = calc_metrics(all_y_true, all_y_pred, all_y_score)
    overall_metrics["avg_latency_ms"] = round(float(np.mean(latencies)), 2)

    return overall_metrics, summary_per_bmark


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Standard SOTA Benchmarks (FF++, GenImage, etc.)")
    parser.add_argument("--base_model_path", type=str, default="/opt/models/qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--checkpoint_dir", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head/final_model")
    parser.add_argument("--test_data_path", type=str, default="/opt/datasets/FakeClue/test.json")
    parser.add_argument("--test_image_dir", type=str, default="/opt/datasets/FakeClue/test")
    parser.add_argument("--output_file", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head/sota_benchmark_comparison.json")
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    device = "npu:0"

    print("=" * 70)
    print("🏆 Evaluating Qwen3-VL + Laya against SOTA Deepfake Benchmarks on Ascend 910C")
    print(f"   Base model: {args.base_model_path}")
    print(f"   Checkpoint: {args.checkpoint_dir}")
    print("=" * 70)

    # Cap max_pixels to 512*512 to prevent giant newspaper scans from overflowing memory
    processor = AutoProcessor.from_pretrained(
        args.base_model_path,
        min_pixels=256 * 256,
        max_pixels=512 * 512
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen3VLLayaModel.load_laya_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device=device
    )
    model.eval()

    test_dataset = SOTABenchmarkDataset(args.test_data_path, args.test_image_dir)
    collate_fn = make_collate_fn(processor)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Running full evaluation on all {len(test_dataset)} test samples (Batch Size = {args.batch_size})...")
    overall, per_bmark = evaluate_dataset(model, loader, device)

    result = {
        "model": "Qwen3-VL-8B + Laya Calibrated Decision Head",
        "device": "Huawei Ascend 910C",
        "overall": overall,
        "per_benchmark": per_bmark
    }

    print("\n" + "=" * 70)
    print("📊 SOTA BENCHMARK EVALUATION RESULTS (ALL 5,000 SAMPLES):")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved results to: {args.output_file}")


if __name__ == "__main__":
    main()
