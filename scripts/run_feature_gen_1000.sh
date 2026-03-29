#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_1000questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/feature_gen_1000}"
SEED="${SEED:-42}"

mkdir -p "${OUT_ROOT}"

COMMON_ARGS=(
  --questions_json "${QUESTIONS_JSON}"
  --image_root "${IMAGE_ROOT}"
  --model_path "${MODEL_PATH}"
  --eval_match_mode heuristic
  --num_samples 1000
  --save_top2_margin
  --save_core_tokenwise_vpmi
  --attn_impl sdpa
  --seed "${SEED}"
)

echo "[run] greedy b1"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  "${COMMON_ARGS[@]}" \
  --out_dir "${OUT_ROOT}/greedy_b1" \
  --num_beams 1 \
  --num_return_sequences 1

echo "[run] beam6 b6"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  "${COMMON_ARGS[@]}" \
  --out_dir "${OUT_ROOT}/beam6_b6" \
  --num_beams 6 \
  --num_return_sequences 6

echo "[done] ${OUT_ROOT}"
