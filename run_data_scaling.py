import argparse
import json
import os
import subprocess
import time
from pathlib import Path

def run_experiment(n_samples, device_id):
    out_dir = f"/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head/scaling_n{n_samples}"
    os.makedirs(out_dir, exist_ok=True)
    log_file = f"{out_dir}/train.log"

    cmd = [
        "python3", "/opt/ch/qwen3_vl_laya_deepfake_audit/code/train_qwen3_vl_laya.py",
        "--max_samples", str(n_samples),
        "--device_id", str(device_id),
        "--output_dir", out_dir,
        "--epochs", "3",
        "--batch_size", "2",
        "--grad_accum", "8"
    ]

    print(f"[*] Starting training for N={n_samples} on NPU {device_id}...")
    t0 = time.time()
    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p, out_dir, t0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=int, nargs="+", default=[300, 600, 1200])
    parser.add_argument("--start_device", type=int, default=6)
    args = parser.parse_args()

    processes = []
    for i, n in enumerate(args.scales):
        dev = args.start_device + i
        p, out_dir, t0 = run_experiment(n, dev)
        processes.append((n, dev, p, out_dir, t0))

    print(f"Launched {len(processes)} data scaling runs on NPUs.")

if __name__ == "__main__":
    main()

