# Reproducibility & Benchmark Protocol

This document provides the hardware specifications, dataset deduplication protocol, and evaluation instructions to reproduce the experimental results reported in **MAD-Guard**.

---

## 1. Hardware & Software Environment

- **Accelerator**: Huawei Ascend 910C NPU (64 GB HBM per card).
- **Host / OS**: openEuler 22.03 LTS-SP4, `npu-smi` 25.5.2.
- **Software Stack**: Python 3.12, CANN 9.1.0, PyTorch 2.10.0, `torch_npu` 2.10.0.post4, Transformers 5.5.4, PEFT 0.21.0, Accelerate 1.15.0, safetensors 0.8.0.
- **Precision**: `torch.bfloat16` (BF16 mixed precision) for both backbone and direct decision heads.
- **Memory Optimization**: Gradient checkpointing (`use_reentrant=False`), micro-batch decoupling ($B_{\text{micro}}=2, N_{\text{accum}}=8$), and dynamic visual resolution capping (`max_pixels = 512 * 512`) reduce peak HBM usage from 58.63 GiB to **31.30 GiB**.

---

## 2. Dataset Provenance & Leak-Free Deduplication

1. **Training & Validation Split (`FakeClue` Multimodal Pool)**:
   - Stratified balanced sampling (`seed=42`) across `(label, category)` pairs: **$N=2,400$** training images (1,201 fake / 1,199 real; 50:50 balance) and **$N=300$** validation images for temperature scaling ($\tau$) and threshold calibration ($\theta^*$).
   - Covers 7 fine-grained tampering categories: `deepfake`, `human`, `doc`, `scene`, `animal`, `satellite`, and `object`.
2. **Strict Out-of-Sample Deduplication**:
   - All 5,000 evaluation images across the 5 benchmarks (**GenImage** $N=1,940$, **FaceForensics++ c23** $N=1,168$, **Chameleon** $N=441$, **Doc** $N=576$, **Satellite** $N=875$) are filtered via 64-bit DCT perceptual hashing (pHash) to enforce a minimum Hamming distance of $d_H \ge 8$ against all training and validation samples, guaranteeing zero near-duplicate overlap.

---

## 3. Script-to-Experiment Mapping

| Experiment Track | Script Entrypoint | Reference Summary Artifact |
| :--- | :--- | :--- |
| **Controlled Direct-Head vs. AR-SFT (`GenImage`)** | `eval_qwen3_vl_laya.py` / `eval_generative_sft.py` | `eval_results/sota_benchmark_comparison.json` |
| **5-Benchmark Out-of-Sample Evaluation ($N=5,000$)** | `eval_sota_benchmarks.py` | `eval_results/sota_benchmark_comparison.json` |
| **Constrained Zero-Shot AR Baseline ($K=1..4$)** | `eval_all_benchmarks_ar.py` | `eval_results/constrained_ar_benchmark_results.json` |
| **In-Domain Held-Out Evaluation ($N=500$)** | `eval_qwen3_vl_laya.py` | `eval_results/eval_test_500_summary.json` |
| **Data Scaling Laws ($N \in \{300, 600, 2400\}$)** | `run_data_scaling.py` | `eval_results/scaling_n300_eval.json`, `eval_results/scaling_n600_eval.json` |
| **Disaggregated Contrastive Direct Head (`CLM-Head`) Ablation** | `run_madguard_clm_ablation.py`, `clm_contrastive_head.py` | `eval_results/clm_forensic_ablation_summary.json` |
