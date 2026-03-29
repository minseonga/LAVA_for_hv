#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_1000questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/adaptive_microbranch_1000}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SEED="${SEED:-42}"
EVAL_MODE="${EVAL_MODE:-heuristic}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"

# Gate
GATE_MODE="${GATE_MODE:-and}"
GATE_TAU_MARGIN="${GATE_TAU_MARGIN:-1.0}"
GATE_TAU_SUFFIX="${GATE_TAU_SUFFIX:--0.3}"

# Micro-branch budget
MICRO_WIDTH="${MICRO_WIDTH:-3}"
MICRO_BUDGET="${MICRO_BUDGET:-3}"

# Selector
TAU_RANK_GAP="${TAU_RANK_GAP:-0.2}"
TAU_SUFFIX_GAP="${TAU_SUFFIX_GAP:-0.0}"
TAU_PREFIX_MAX="${TAU_PREFIX_MAX:-2.0}"
MAX_SFULL_DROP="${MAX_SFULL_DROP:-0.2}"
K_PREFIX="${K_PREFIX:-2}"
K_SUFFIX="${K_SUFFIX:-2}"

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" /home/kms/miniconda3/envs/vocot/bin/python /home/kms/LLaVA_calibration/analyze_adaptive_microbranch.py \
  --questions_json "${QUESTIONS_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --eval_match_mode "${EVAL_MODE}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --gate_mode "${GATE_MODE}" \
  --gate_tau_margin "${GATE_TAU_MARGIN}" \
  --gate_tau_suffix "${GATE_TAU_SUFFIX}" \
  --micro_width "${MICRO_WIDTH}" \
  --micro_budget "${MICRO_BUDGET}" \
  --tau_rank_gap "${TAU_RANK_GAP}" \
  --tau_suffix_gap "${TAU_SUFFIX_GAP}" \
  --tau_prefix_max "${TAU_PREFIX_MAX}" \
  --max_sfull_drop "${MAX_SFULL_DROP}" \
  --k_prefix "${K_PREFIX}" \
  --k_suffix "${K_SUFFIX}"

echo "[done] ${OUT_DIR}"

