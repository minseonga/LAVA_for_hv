#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/greedy_token_profile_${NUM_SAMPLES}}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/data/gqa/testdev_balanced_questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
RANK_TOP_K="${RANK_TOP_K:-50}"
CURVE_BINS="${CURVE_BINS:-20}"
SEED="${SEED:-42}"

mkdir -p "${OUT_DIR}"
echo "[run] out=${OUT_DIR} num_samples=${NUM_SAMPLES} cuda=${CUDA_DEVICE}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" /home/kms/LLaVA_calibration/analyze_greedy_token_profile.py \
  --questions_json "${QUESTIONS_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --rank_top_k "${RANK_TOP_K}" \
  --curve_bins "${CURVE_BINS}" \
  --eval_match_mode heuristic \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}"

echo "[done] ${OUT_DIR}"
