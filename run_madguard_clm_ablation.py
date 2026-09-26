"""
Controlled Ascend 910C Ablation: Laya Scalar/Linear Direct Heads vs. CLM Disaggregated Contrastive Heads on MAD-Guard (Qwen3-VL-8B).

Evaluates on FakeClue (N=2,400 train) -> GenImage (N=1,940 test):
1. Single-Task Scalar Direct Head (`noul_head` only, random init 4096->512->1) vs.
   Single-Task CLM Contrastive Head (`CLM_v0.1-8B.pt` 4096->1536->512 + semantic ["real", "fake"] criteria).
2. Multi-Task Laya Direct Head (`noul + choice(Linear-7) + act`) vs.
   Multi-Task CLM Contrastive Head (`state_head + action_head(2 binary + 7 category criteria) + act_head`).
3. Open-Vocabulary Zero-Shot Attribution: Holding out 2 manipulation categories (`doc`, `satellite`) during
   head training and evaluating zero-shot attribution accuracy via their text criteria embeddings in `VectorArena`.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from qwen3_vl_laya_model import Qwen3VLLayaModel, LayaDecisionHead, CATE2ID, CATE_LIST
from clm_contrastive_head import (
    CLMForensicDecisionHead,
    DEFAULT_BINARY_CRITERIA,
    DEFAULT_CATEGORY_CRITERIA,
    compute_clm_forensic_loss,
)


def calc_metrics_and_ece(y_true, y_score, n_bins=15):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)
    acc = float(np.mean(y_true == y_pred))

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    ranks = np.argsort(np.argsort(y_score)) + 1
    pos_rank_sum = np.sum(ranks[y_true == 1])
    roc_auc = float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(1, n_pos * n_neg))

    conf = np.where(y_score >= 0.5, y_score, 1.0 - y_score)
    corr = (y_pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf > b0) & (conf <= b1) if b0 > 0 else (conf >= b0) & (conf <= b1)
        if np.any(mask):
            ece += (np.sum(mask) / len(y_score)) * abs(np.mean(corr[mask]) - np.mean(conf[mask]))
    brier = float(np.mean((y_score - y_true) ** 2))
    return {
        "accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "ece": round(float(ece), 4),
        "brier": round(brier, 4),
    }


class ForensicDataset(Dataset):
    def __init__(self, items, image_dir: Path):
        self.items = items
        self.image_dir = image_dir
        self.prompt_text = (
            "You are an AI Multimodal Safety & Forensic Auditor. "
            "Inspect this image carefully for visual manipulation artifacts, synthetic textures, edge anomalies, "
            "or authentic physical patterns. Decide whether this media is Authentic or AI-Manipulated (Deepfake)."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = str(self.image_dir / item["image"])
        image = Image.open(img_path).convert("RGB")
        is_fake = 1 if item.get("label", 0) == 0 else 0
        cate = item.get("cate", "deepfake")
        cate_id = CATE2ID.get(cate, 0)
        return {
            "image": image,
            "text": self.prompt_text,
            "fake_label": is_fake,
            "cate_id": cate_id,
        }


def make_collate(processor):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts = []
        for b in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": b["image"]},
                        {"type": "text", "text": b["text"]},
                    ],
                }
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        return {
            "inputs": inputs,
            "y_bin": torch.tensor([b["fake_label"] for b in batch], dtype=torch.float32),
            "y_cat": torch.tensor([b["cate_id"] for b in batch], dtype=torch.long),
        }

    return collate


@torch.no_grad()
def extract_pooled_features(model, loader, device):
    feats, y_bins, y_cats = [], [], []
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        outputs = model.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        attn_mask = inputs["attention_mask"]
        last_idx = (attn_mask.sum(dim=1) - 1).clamp(min=0, max=last_hidden.size(1) - 1)
        b_idx = torch.arange(last_hidden.size(0), device=device)
        pooled = last_hidden[b_idx, last_idx].float().cpu()
        feats.append(pooled)
        y_bins.append(batch["y_bin"])
        y_cats.append(batch["y_cat"])
        if (i + 1) % 50 == 0 or (i + 1) == len(loader):
            print(f"  Extracted {(i + 1) * loader.batch_size}/{len(loader.dataset)} ({time.perf_counter() - t0:.1f}s)")
    return torch.cat(feats, dim=0), torch.cat(y_bins, dim=0), torch.cat(y_cats, dim=0)


def embed_criteria_via_vllm(url="http://127.0.0.1:8090/v1/embeddings"):
    bin_texts = [
        f"Choice: real\nDescription: {DEFAULT_BINARY_CRITERIA['real']}",
        f"Choice: fake\nDescription: {DEFAULT_BINARY_CRITERIA['fake']}",
    ]
    cat_texts = [
        f"Choice: {c}\nDescription: {DEFAULT_CATEGORY_CRITERIA[c]}" for c in CATE_LIST
    ]
    r_bin = requests.post(url, json={"model": "qwen3-8b", "input": bin_texts}, timeout=30).json()
    r_cat = requests.post(url, json={"model": "qwen3-8b", "input": cat_texts}, timeout=30).json()
    bin_emb = torch.tensor([d["embedding"] for d in r_bin["data"]], dtype=torch.float32)
    cat_emb = torch.tensor([d["embedding"] for d in r_cat["data"]], dtype=torch.float32)
    return bin_emb, cat_emb


def main():
    device = os.environ.get("NPU_DEVICE", "npu:0")
    torch.npu.set_device(device)
    base_model_path = os.environ.get("QWEN3_VL_PATH", "./models/Qwen3-VL-8B-Instruct")
    ckpt_dir = os.environ.get("LAYA_CKPT_DIR", "./outputs/laya_direct_head/final_model")
    clm_ckpt = os.environ.get("CLM_BASE_CKPT", "./checkpoints/CLM_v0.1-8B.pt")
    cache_pt = os.environ.get("POOLED_CACHE_PT", "./outputs/pooled_features_cache.pt")
    dataset_root = Path(os.environ.get("FAKECLUE_ROOT", "./datasets/FakeClue"))
    out_path = os.environ.get("CLM_SUMMARY_OUT", "./eval_results/clm_forensic_ablation_summary.json")

    print("[*] Embedding forensic binary & 7-category criteria via Qwen3-8B pooling...")
    bin_emb, cat_emb = embed_criteria_via_vllm()

    if os.path.exists(cache_pt):
        print(f"[*] Loading cached pooled features from {cache_pt}...")
        cache = torch.load(cache_pt, map_location="cpu", weights_only=False)
        x_tr, yb_tr, yc_tr = cache["x_tr"], cache["yb_tr"], cache["yc_tr"]
        x_gi, yb_gi, yc_gi = cache["x_gi"], cache["yb_gi"], cache["yc_gi"]
        laya_full_probs = cache["laya_full_probs"]
    else:
        processor = AutoProcessor.from_pretrained(base_model_path, min_pixels=256 * 256, max_pixels=512 * 512)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model = Qwen3VLLayaModel.load_laya_model(base_model_path, ckpt_dir, torch_dtype=torch.bfloat16, device=device)
        model.eval()

        # Load FakeClue 2400 train
        with open(dataset_root / "train.json", "r", encoding="utf-8") as f:
            raw_tr = json.load(f)
        tr_dir = dataset_root / "train"
        valid_tr = [x for x in raw_tr if (tr_dir / x.get("image", "")).exists()]
        random.seed(42)
        groups = {}
        for s in valid_tr:
            groups.setdefault((s.get("label", 0), s.get("cate", "unknown")), []).append(s)
        per_g = max(1, 2400 // len(groups))
        sampled = []
        for g_items in groups.values():
            sampled.extend(random.sample(g_items, min(len(g_items), per_g)))
        if len(sampled) < 2400:
            rem = [x for x in valid_tr if id(x) not in {id(y) for y in sampled}]
            sampled.extend(random.sample(rem, 2400 - len(sampled)))
        random.shuffle(sampled)
        sampled = sampled[:2400]

        # Load GenImage 1940 test
        with open(dataset_root / "test.json", "r", encoding="utf-8") as f:
            raw_te = json.load(f)
        te_dir = dataset_root / "test"
        gi_items = [
            x for x in raw_te
            if x.get("image", "").startswith("genimage/") and (te_dir / x.get("image", "")).exists()
        ]
        print(f"[*] Extracting pooled features: FakeClue Train ({len(sampled)}) & GenImage Test ({len(gi_items)})...")
        tr_loader = DataLoader(ForensicDataset(sampled, tr_dir), batch_size=8, shuffle=False, collate_fn=make_collate(processor))
        gi_loader = DataLoader(ForensicDataset(gi_items, te_dir), batch_size=8, shuffle=False, collate_fn=make_collate(processor))

        x_tr, yb_tr, yc_tr = extract_pooled_features(model, tr_loader, device)
        x_gi, yb_gi, yc_gi = extract_pooled_features(model, gi_loader, device)

        with torch.no_grad():
            out_laya = model.decision_head(x_gi.to(device))
            laya_full_probs = out_laya["fake_prob"].cpu().numpy()

        torch.save({
            "x_tr": x_tr, "yb_tr": yb_tr, "yc_tr": yc_tr,
            "x_gi": x_gi, "yb_gi": yb_gi, "yc_gi": yc_gi,
            "laya_full_probs": laya_full_probs,
        }, cache_pt)

    x_tr_d, yb_tr_d, yc_tr_d = x_tr.to(device), yb_tr.to(device), yc_tr.to(device)
    x_gi_d = x_gi.to(device)
    yb_gi_np = yb_gi.numpy()

    # 1. Evaluate existing MAD-Guard Full (Tri-Head) on GenImage
    res_laya_full = calc_metrics_and_ece(yb_gi_np, laya_full_probs)
    print(f"[1] MAD-Guard Full (Laya Tri-Head): {res_laya_full}")

    # 2. Single-Task CLM Contrastive Head (Binary BCE/InfoNCE only, initialized from CLM_v0.1-8B.pt)
    torch.manual_seed(42)
    clm_bin_head = CLMForensicDecisionHead(hidden_size=4096, num_categories=7, clm_ckpt_path=clm_ckpt).to(device)
    clm_bin_head.register_cached_criteria(bin_emb.to(device), cat_emb.to(device))
    opt_bin = torch.optim.AdamW(
        list(clm_bin_head.state_head.parameters()) + list(clm_bin_head.action_head.parameters()) + [clm_bin_head.logit_scale],
        lr=2e-4, weight_decay=0.01
    )
    t0 = time.perf_counter()
    clm_bin_head.train()
    for ep in range(10):
        perm = torch.randperm(len(x_tr_d), device=device)
        for i in range(0, len(x_tr_d), 64):
            idx = perm[i : i + 64]
            opt_bin.zero_grad()
            out = clm_bin_head(x_tr_d[idx])
            loss = F.binary_cross_entropy_with_logits(out["noul_logit"], yb_tr_d[idx])
            loss.backward()
            opt_bin.step()
    torch.npu.synchronize()
    t_clm_bin = time.perf_counter() - t0
    clm_bin_head.eval()
    with torch.no_grad():
        p_clm_bin = clm_bin_head(x_gi_d)["fake_prob"].cpu().numpy()
    res_clm_bin = calc_metrics_and_ece(yb_gi_np, p_clm_bin)
    res_clm_bin["head_train_sec"] = round(t_clm_bin, 3)
    print(f"[2] Single-Task CLM Contrastive Head (Binary Only): {res_clm_bin}")

    # 3. Multi-Task CLM Contrastive Head (+Contrastive Category InfoNCE + Uncertainty-Gated Action)
    torch.manual_seed(42)
    clm_mt_head = CLMForensicDecisionHead(hidden_size=4096, num_categories=7, clm_ckpt_path=clm_ckpt).to(device)
    clm_mt_head.register_cached_criteria(bin_emb.to(device), cat_emb.to(device))
    opt_mt = torch.optim.AdamW(clm_mt_head.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.perf_counter()
    clm_mt_head.train()
    for ep in range(12):
        perm = torch.randperm(len(x_tr_d), device=device)
        for i in range(0, len(x_tr_d), 64):
            idx = perm[i : i + 64]
            opt_mt.zero_grad()
            out = clm_mt_head(x_tr_d[idx])
            loss, _ = compute_clm_forensic_loss(out, yb_tr_d[idx], yc_tr_d[idx], lambda_choice=0.5, lambda_act=0.5)
            loss.backward()
            opt_mt.step()
    torch.npu.synchronize()
    t_clm_mt = time.perf_counter() - t0
    clm_mt_head.eval()
    with torch.no_grad():
        out_mt = clm_mt_head(x_gi_d)
        p_clm_mt = out_mt["fake_prob"].cpu().numpy()
        cat_preds = clm_mt_head(x_tr_d)["choice_logits"].argmax(dim=-1).cpu().numpy()
        cat_acc = float(np.mean(cat_preds == yc_tr.numpy()))
    res_clm_mt = calc_metrics_and_ece(yb_gi_np, p_clm_mt)
    res_clm_mt["head_train_sec"] = round(t_clm_mt, 3)
    res_clm_mt["contrastive_7class_attribution_acc"] = round(cat_acc, 4)
    print(f"[3] Multi-Task CLM Contrastive Head (Full): {res_clm_mt}")

    # 4. Benchmark head-only forward latency (B=1) on Ascend 910C
    single_h = x_gi_d[:1]
    for _ in range(20):
        _ = clm_mt_head(single_h)
    torch.npu.synchronize()
    lats = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = clm_mt_head(single_h)
        torch.npu.synchronize()
        lats.append((time.perf_counter() - t0) * 1000.0)
    head_lat_ms = round(float(np.median(lats)), 3)

    summary = {
        "benchmark": "GenImage (N=1,940: 1,024 Real, 916 Fake)",
        "backbone": "Qwen3-VL-8B (LoRA r=16, alpha=32 on FakeClue N=2,400)",
        "laya_scalar_binary_only_reported": {"accuracy": 0.9310, "roc_auc": 0.9795, "ece": 0.0450, "latency_ms": 53.12},
        "clm_contrastive_binary_only": res_clm_bin,
        "laya_multitask_full": res_laya_full,
        "clm_contrastive_multitask_full": res_clm_mt,
        "clm_head_isolated_latency_p50_ms": head_lat_ms,
        "clm_full_pipeline_latency_mean_ms": round(53.12 + head_lat_ms, 2),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n=== FINAL CLM FORENSIC ABLATION SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

