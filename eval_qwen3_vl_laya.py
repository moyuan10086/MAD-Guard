import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
try:
    import torch_npu  # type: ignore
except ImportError:
    torch_npu = None
from PIL import Image
from torch.utils.data import DataLoader, Dataset
try:
    from transformers import AutoProcessor  # type: ignore
except ImportError:
    AutoProcessor = None

from qwen3_vl_laya_model import Qwen3VLLayaModel, CATE2ID, ID2CATE, CATE_LIST


def calc_metrics(y_true, y_pred, y_score):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_score = np.array(y_score, dtype=float)

    total = len(y_true)
    acc = np.mean(y_true == y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROC AUC via Mann-Whitney U test rank calculation
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos > 0 and n_neg > 0:
        ranks = np.argsort(np.argsort(y_score)) + 1
        pos_rank_sum = np.sum(ranks[y_true == 1])
        roc_auc = float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    else:
        roc_auc = 1.0 if n_pos == total or n_neg == total else 0.5

    # PR AUC (Average Precision)
    if n_pos > 0:
        sorted_idx = np.argsort(-y_score)
        y_sorted = y_true[sorted_idx]
        cum_pos = np.cumsum(y_sorted == 1)
        cum_total = np.arange(1, total + 1)
        precisions = cum_pos / cum_total
        pr_auc = float(np.sum(precisions[y_sorted == 1]) / n_pos)
    else:
        pr_auc = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL + Laya Calibrated Decision Model on FakeClue")
    parser.add_argument("--base_model_path", type=str, default="/opt/models/qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--checkpoint_dir", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head/final_model")
    parser.add_argument("--test_data_path", type=str, default="/opt/datasets/FakeClue/test.json")
    parser.add_argument("--test_image_dir", type=str, default="/opt/datasets/FakeClue/test")
    parser.add_argument("--output_file", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head/test_benchmark_results.jsonl")
    parser.add_argument("--summary_file", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head/test_summary.json")
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=1000, help="Test subset (-1 for all 5,000)")
    return parser.parse_args()


class FakeClueTestDataset(Dataset):
    def __init__(self, data_path: str, image_dir: str, max_samples: int = -1):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.image_dir = Path(image_dir)
        valid_items = []
        for item in raw_data:
            img_rel = item.get("image")
            img_full = self.image_dir / img_rel
            if img_full.exists():
                valid_items.append(item)

        if 0 < max_samples < len(valid_items):
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
        print(f"Loaded {len(self.samples)} test samples. Fake={fake_cnt} ({fake_cnt/len(self.samples)*100:.1f}%), Real={real_cnt} ({real_cnt/len(self.samples)*100:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = str(self.image_dir / item["image"])
        image = Image.open(img_path).convert("RGB")

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
            "cate_ids": [b["cate_id"] for b in batch],
            "cates": [b["cate"] for b in batch],
            "image_paths": [b["image_path"] for b in batch]
        }
    return collate_fn


def main():
    args = parse_args()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    device = "npu:0"

    print("=" * 70)
    print("📊 Evaluating Qwen3-VL + Laya Calibrated Decision Model on Ascend 910C")
    print(f"   Base model: {args.base_model_path}")
    print(f"   Checkpoint: {args.checkpoint_dir}")
    print(f"   Test Set: {args.test_data_path} (max_samples={args.max_samples})")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(args.base_model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading fine-tuned model...")
    model = Qwen3VLLayaModel.load_laya_model(
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device=device
    )
    model.eval()

    test_dataset = FakeClueTestDataset(args.test_data_path, args.test_image_dir, max_samples=args.max_samples)
    collate_fn = make_collate_fn(processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    y_trues = []
    y_scores = []
    y_preds = []
    cate_trues = []
    cate_preds = []
    latencies = []
    results = []

    print(f"Running non-autoregressive decision inference on {len(test_dataset)} samples...")
    t_start_all = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            bs = len(batch["fake_labels"])

            torch.npu.synchronize()
            t0 = time.time()
            out = model(**inputs)
            torch.npu.synchronize()
            dt_batch = (time.time() - t0) * 1000.0
            lat_per_sample = dt_batch / bs

            probs = out["fake_prob"].cpu().numpy()
            act_preds = out["act_logits"].argmax(dim=-1).cpu().numpy()
            pred_cates = out["choice_logits"].argmax(dim=-1).cpu().numpy()

            for i in range(bs):
                prob = float(probs[i])
                pred_act = int(act_preds[i])
                true_label = int(batch["fake_labels"][i])
                true_cate = batch["cates"][i]
                pred_cate = ID2CATE.get(int(pred_cates[i]), "unknown")

                y_trues.append(true_label)
                y_scores.append(prob)
                y_preds.append(pred_act)
                cate_trues.append(true_cate)
                cate_preds.append(pred_cate)
                latencies.append(lat_per_sample)

                results.append({
                    "image": batch["image_paths"][i],
                    "y_true": true_label,
                    "y_pred": pred_act,
                    "calibrated_fake_prob": round(prob, 4),
                    "true_cate": true_cate,
                    "pred_cate": pred_cate,
                    "latency_ms": round(lat_per_sample, 2)
                })

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"Processed {(batch_idx + 1) * args.batch_size}/{len(test_dataset)} samples | Latency: {lat_per_sample:.2f} ms/sample")

    total_time = time.time() - t_start_all
    y_trues = np.array(y_trues)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)

    m = calc_metrics(y_trues, y_preds, y_scores)
    ece = compute_ece(y_scores, y_trues)
    avg_lat = float(np.mean(latencies))

    summary = {
        "model": "Qwen3-VL-8B + Laya Calibrated Decision Head",
        "device": "Huawei Ascend 910C",
        "num_test_samples": len(y_trues),
        "accuracy": round(m["accuracy"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "roc_auc": round(m["roc_auc"], 4),
        "pr_auc": round(m["pr_auc"], 4),
        "ece": round(ece, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "throughput_samples_per_sec": round(len(y_trues) / total_time, 2),
        "total_eval_time_sec": round(total_time, 2)
    }

    print("\n" + "=" * 70)
    print("📈 EVALUATION RESULTS SUMMARY:")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved benchmark details to: {args.output_file}")

    with open(args.summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved benchmark summary to: {args.summary_file}")


if __name__ == "__main__":
    main()

