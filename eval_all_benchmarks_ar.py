import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch_npu
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Constrained AR Baseline on FakeClue Benchmarks")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "genimage", "ff++", "chameleon", "doc", "satellite"])
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="/opt/models/qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="/opt/datasets/FakeClue/test.json")
    parser.add_argument("--image_dir", type=str, default="/opt/datasets/FakeClue/test")
    parser.add_argument("--output_dir", type=str, default="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head")
    return parser.parse_args()

def evaluate_subset(benchmark_name, samples, model, processor, image_dir, device, out_file):
    print(f"\n=======================================================", flush=True)
    print(f"Starting Evaluation on [{benchmark_name.upper()}]: {len(samples)} samples", flush=True)
    print(f"=======================================================", flush=True)

    tokenizer = processor.tokenizer
    real_tokens = [tokenizer.encode(w, add_special_tokens=False)[-1] for w in ['REAL', ' REAL', 'real', ' real']]
    fake_tokens = [tokenizer.encode(w, add_special_tokens=False)[-1] for w in ['FAKE', ' FAKE', 'fake', ' fake']]

    latencies_total = []
    latencies_ttft = []
    latencies_decode = []
    preds_generated = []
    preds_logit = []
    ground_truths = []
    p_fakes_logit = []
    gen_texts = []

    # Warmup 2 samples
    for w_item in samples[:2]:
        w_img = Image.open(image_dir / w_item['image']).convert('RGB')
        w_msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': w_img}, {'type': 'text', 'text': 'Is this image REAL or FAKE? Output ONLY the word REAL or FAKE.\nAnswer:'}]}]
        w_prompt = processor.apply_chat_template(w_msgs, tokenize=False, add_generation_prompt=True)
        w_inp = processor(text=[w_prompt], images=w_img, return_tensors='pt').to(device)
        with torch.no_grad():
            _ = model.generate(**w_inp, max_new_tokens=4, do_sample=False)
    torch.npu.synchronize()

    start_time = time.time()
    total_samples = len(samples)

    for idx, item in enumerate(samples):
        img_path = image_dir / item['image']
        label = item['label'] # In FakeClue: label 0 is fake, label 1 is real
        gt_is_fake = 1 if label == 0 else 0
        ground_truths.append(gt_is_fake)

        img = Image.open(img_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': 'Is this image REAL or FAKE? Output ONLY the word REAL or FAKE.\nAnswer:'}]}]
        prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[prompt_text], images=img, return_tensors='pt').to(device)

        torch.npu.synchronize()
        t_start = time.time()

        with torch.no_grad():
            # Prefill / TTFT
            out_prefill = model(**inp)
            torch.npu.synchronize()
            t_prefill = time.time()

            logits = out_prefill.logits[0, -1, :]
            l_r = max(logits[t].item() for t in real_tokens)
            l_f = max(logits[t].item() for t in fake_tokens)
            exp_r, exp_f = np.exp(l_r - max(l_r, l_f)), np.exp(l_f - max(l_r, l_f))
            p_fake = exp_f / (exp_r + exp_f)
            p_fakes_logit.append(p_fake)
            pred_l = 1 if p_fake >= 0.5 else 0
            preds_logit.append(pred_l)

            # Autoregressive decoding (greedy max_new_tokens=4)
            gen_out = model.generate(**inp, max_new_tokens=4, do_sample=False)
            torch.npu.synchronize()
            t_end = time.time()

        ttft_ms = (t_prefill - t_start) * 1000.0
        total_ms = (t_end - t_start) * 1000.0
        dec_ms = total_ms - ttft_ms

        latencies_ttft.append(ttft_ms)
        latencies_total.append(total_ms)
        latencies_decode.append(dec_ms)

        out_tokens = gen_out[0][inp.input_ids.shape[1]:]
        gen_text = processor.decode(out_tokens, skip_special_tokens=True).strip()
        if len(gen_texts) < 50:
            gen_texts.append(gen_text)

        upper_text = gen_text.upper()
        if 'FAKE' in upper_text:
            pred_g = 1
        elif 'REAL' in upper_text:
            pred_g = 0
        else:
            pred_g = pred_l
        preds_generated.append(pred_g)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_samples:
            elapsed_sec = time.time() - start_time
            curr_acc = np.mean(np.array(preds_generated) == np.array(ground_truths)) * 100.0
            mean_tot = np.mean(latencies_total)
            eta_sec = (elapsed_sec / (idx + 1)) * (total_samples - idx - 1)
            print(f"[{benchmark_name}] [{idx+1}/{total_samples}] Latency: {mean_tot:.1f}ms | Acc: {curr_acc:.2f}% | Elapsed: {elapsed_sec/60:.1f}m | ETA: {eta_sec/60:.1f}m", flush=True)

            interim_results = {
                'benchmark': benchmark_name,
                'processed': idx + 1,
                'total': total_samples,
                'mean_total_latency_ms': round(float(mean_tot), 2),
                'accuracy_text_pct': round(float(curr_acc), 2)
            }
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(interim_results, f, indent=2)

    # Final calculations
    y_true = np.array(ground_truths)
    y_pred = np.array(preds_generated)
    scores = np.array(p_fakes_logit)

    total = len(y_true)
    acc = float(np.mean(y_true == y_pred))
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if n_pos > 0 and n_neg > 0:
        ranks = np.argsort(np.argsort(scores)) + 1
        pos_rank_sum = np.sum(ranks[y_true == 1])
        roc_auc = float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    else:
        roc_auc = None

    confidences = np.maximum(scores, 1.0 - scores)
    accuracies = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        bin_mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (np.sum(bin_mask) / len(confidences)) * np.abs(bin_acc - bin_conf)

    summary = {
        'benchmark': benchmark_name,
        'num_samples': total,
        'num_fake': n_pos,
        'num_real': n_neg,
        'accuracy': round(acc, 4),
        'precision': round(precision, 4) if precision is not None else None,
        'recall': round(recall, 4),
        'f1': round(f1, 4) if f1 is not None else None,
        'roc_auc': round(roc_auc, 4) if roc_auc is not None else None,
        'ece': round(float(ece), 4),
        'mean_latency_ms': round(float(np.mean(latencies_total)), 2),
        'mean_ttft_ms': round(float(np.mean(latencies_ttft)), 2),
        'mean_decode_latency_ms': round(float(np.mean(latencies_decode)), 2)
    }

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Completed {benchmark_name.upper()}]: Accuracy={acc*100:.2f}%, Recall={recall*100:.2f}%, Latency={np.mean(latencies_total):.1f}ms", flush=True)
    return summary

def main():
    args = parse_args()
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(args.device_id)
    device = 'npu:0'

    print(f"Loading Qwen3-VL from {args.model_path} onto NPU {args.device_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    image_dir = Path(args.image_dir)
    with open(args.data_path, 'r', encoding='utf-8') as f:
        all_test = json.load(f)

    # Classify benchmarks by image prefix
    benchmarks = {
        'genimage': [x for x in all_test if x.get('image', '').startswith('genimage/')],
        'ff++': [x for x in all_test if x.get('image', '').startswith('ff++/') or x.get('image', '').startswith('FaceForensics/')],
        'chameleon': [x for x in all_test if x.get('image', '').startswith('chameleon/')],
        'doc': [x for x in all_test if x.get('image', '').startswith('doc/')],
        'satellite': [x for x in all_test if x.get('image', '').startswith('satellite/')]
    }

    if args.benchmark == 'all':
        for b_name, b_samples in benchmarks.items():
            out_file = os.path.join(args.output_dir, f"constrained_ar_{b_name}.json")
            evaluate_subset(b_name, b_samples, model, processor, image_dir, device, out_file)
    else:
        b_samples = benchmarks[args.benchmark]
        out_file = os.path.join(args.output_dir, f"constrained_ar_{args.benchmark}.json")
        evaluate_subset(args.benchmark, b_samples, model, processor, image_dir, device, out_file)

if __name__ == '__main__':
    main()

