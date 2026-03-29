#!/usr/bin/env bash
set -euo pipefail

# Run 3 operator-comparison experiments on POPE-1000:
# 1) AIS soft
# 2) AIS semi-hard
# 3) first-token safeguard only
#
# Usage:
#   bash /home/kms/LLaVA_calibration/scripts/run_pope_opcmp_v2.sh
# Optional env overrides:
#   GPU=6
#   PYTHON_BIN=python
#   MODEL_PATH=liuhaotian/llava-v1.5-7b
#   QFILE=/tmp/pope_1000_q.jsonl
#   IMAGE_ROOT=/home/kms/data/pope/val2014
#   OUT_DIR=/home/kms/LLaVA_calibration/experiments/pope_opcmp_v2
#   HEADSET_JSON=/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json
#   BUDGET_TOTAL=0.02
#   PATCH_TOPK=1
#   HARMFUL_TOP_RATIO=0.2
#   TAU=999
#   GAMMA=1.0
#   SEMIHARD_PENALTY=0.2
#   FIRSTTOKEN_YES_BIAS=-0.2
#   FIRSTTOKEN_NO_BIAS=0.2

GPU="${GPU:-6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
QFILE="${QFILE:-/tmp/pope_1000_q.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/pope_opcmp_v2}"
HEADSET_JSON="${HEADSET_JSON:-/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json}"

BUDGET_TOTAL="${BUDGET_TOTAL:-0.02}"
PATCH_TOPK="${PATCH_TOPK:-1}"
HARMFUL_TOP_RATIO="${HARMFUL_TOP_RATIO:-0.2}"
TAU="${TAU:-999}"
GAMMA="${GAMMA:-1.0}"
SEMIHARD_PENALTY="${SEMIHARD_PENALTY:-0.2}"
FIRSTTOKEN_YES_BIAS="${FIRSTTOKEN_YES_BIAS:--0.2}"
FIRSTTOKEN_NO_BIAS="${FIRSTTOKEN_NO_BIAS:-0.2}"

REPO_ROOT="/home/kms/LLaVA_calibration"
mkdir -p "${OUT_DIR}"

echo "[run 1/3] soft operator"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" \
  --image-folder "${IMAGE_ROOT}" \
  --question-file "${QFILE}" \
  --answers-file "${OUT_DIR}/soft.jsonl" \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 8 \
  --enable-ais-gating \
  --ais-arm harmful_only \
  --ais-headset-json "${HEADSET_JSON}" \
  --ais-use-budget-routing \
  --ais-budget-total "${BUDGET_TOTAL}" \
  --ais-budget-patch-topk "${PATCH_TOPK}" \
  --ais-harmful-top-ratio "${HARMFUL_TOP_RATIO}" \
  --ais-tau "${TAU}" \
  --ais-gamma "${GAMMA}" \
  --ais-operator soft \
  --ais-debug-log \
  --ais-debug-dump "${OUT_DIR}/soft_debug.csv" \
  --dump-first-token-logits "${OUT_DIR}/soft_first_token.csv" \
  --first-token-logits-include-pre

echo "[run 2/3] semi-hard operator"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" \
  --image-folder "${IMAGE_ROOT}" \
  --question-file "${QFILE}" \
  --answers-file "${OUT_DIR}/semihard.jsonl" \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 8 \
  --enable-ais-gating \
  --ais-arm harmful_only \
  --ais-headset-json "${HEADSET_JSON}" \
  --ais-use-budget-routing \
  --ais-budget-total "${BUDGET_TOTAL}" \
  --ais-budget-patch-topk "${PATCH_TOPK}" \
  --ais-harmful-top-ratio "${HARMFUL_TOP_RATIO}" \
  --ais-tau "${TAU}" \
  --ais-gamma "${GAMMA}" \
  --ais-operator semi_hard \
  --ais-semihard-penalty "${SEMIHARD_PENALTY}" \
  --ais-debug-log \
  --ais-debug-dump "${OUT_DIR}/semihard_debug.csv" \
  --dump-first-token-logits "${OUT_DIR}/semihard_first_token.csv" \
  --first-token-logits-include-pre

echo "[run 3/3] first-token safeguard only"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" \
  --image-folder "${IMAGE_ROOT}" \
  --question-file "${QFILE}" \
  --answers-file "${OUT_DIR}/firsttoken_guard.jsonl" \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 8 \
  --first-token-safeguard \
  --first-token-yes-bias "${FIRSTTOKEN_YES_BIAS}" \
  --first-token-no-bias "${FIRSTTOKEN_NO_BIAS}" \
  --dump-first-token-logits "${OUT_DIR}/firsttoken_guard_first_token.csv"

echo "[done] outputs -> ${OUT_DIR}"

