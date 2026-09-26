#!/bin/bash
set -e

OUTPUT_DIR="/opt/ch/qwen3_vl_laya_deepfake_audit/outputs/laya_direct_head"
mkdir -p ${OUTPUT_DIR}

echo "=========================================================="
echo "🚀 Launching 5-NPU Parallel Evaluation for Constrained AR"
echo "   NPU 1 -> GenImage (1,940 samples)"
echo "   NPU 2 -> FF++ (1,168 samples)"
echo "   NPU 3 -> Chameleon (441 samples)"
echo "   NPU 4 -> Doc (576 samples)"
echo "   NPU 5 -> Satellite (875 samples)"
echo "=========================================================="

nohup python3 /opt/ch/qwen3_vl_laya_deepfake_audit/code/eval_all_benchmarks_ar.py --benchmark genimage --device_id 1 > ${OUTPUT_DIR}/eval_ar_genimage.log 2>&1 &
PID_GENIMAGE=$!

nohup python3 /opt/ch/qwen3_vl_laya_deepfake_audit/code/eval_all_benchmarks_ar.py --benchmark "ff++" --device_id 2 > ${OUTPUT_DIR}/eval_ar_ff++.log 2>&1 &
PID_FF=$!

nohup python3 /opt/ch/qwen3_vl_laya_deepfake_audit/code/eval_all_benchmarks_ar.py --benchmark chameleon --device_id 3 > ${OUTPUT_DIR}/eval_ar_chameleon.log 2>&1 &
PID_CHAM=$!

nohup python3 /opt/ch/qwen3_vl_laya_deepfake_audit/code/eval_all_benchmarks_ar.py --benchmark doc --device_id 4 > ${OUTPUT_DIR}/eval_ar_doc.log 2>&1 &
PID_DOC=$!

nohup python3 /opt/ch/qwen3_vl_laya_deepfake_audit/code/eval_all_benchmarks_ar.py --benchmark satellite --device_id 5 > ${OUTPUT_DIR}/eval_ar_satellite.log 2>&1 &
PID_SAT=$!

echo "Processes launched:"
echo "  GenImage PID: ${PID_GENIMAGE} on NPU 1"
echo "  FF++     PID: ${PID_FF} on NPU 2"
echo "  Chameleon PID: ${PID_CHAM} on NPU 3"
echo "  Doc      PID: ${PID_DOC} on NPU 4"
echo "  Satellite PID: ${PID_SAT} on NPU 5"
echo "All 5 benchmarks running concurrently across 5 NPUs!"

