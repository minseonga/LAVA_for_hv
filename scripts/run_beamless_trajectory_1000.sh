#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_1000questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/beamless_trajectory_1000}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SEED="${SEED:-42}"
EVAL_MODE="${EVAL_MODE:-heuristic}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"

# trajectory features
RANK_TOP_K="${RANK_TOP_K:-50}"
K_PREFIX="${K_PREFIX:-2}"
K_SUFFIX="${K_SUFFIX:-2}"

# risk gate
TAU_COLLAPSE_GAP="${TAU_COLLAPSE_GAP:--0.8}"
TAU_RANK_MIN="${TAU_RANK_MIN:-0.05}"
TAU_ENERGY="${TAU_ENERGY:-7.0}"
GATE_MODE="${GATE_MODE:-or}"

# tail micro-regeneration
BRANCH_TOP_K="${BRANCH_TOP_K:-8}"
BRANCH_BUDGET="${BRANCH_BUDGET:-3}"
BRANCH_LAMBDA_RANK="${BRANCH_LAMBDA_RANK:-0.8}"
BRANCH_LAMBDA_VPMI="${BRANCH_LAMBDA_VPMI:-0.8}"
SELECT_W_SUFFIX="${SELECT_W_SUFFIX:-0.5}"
SELECT_W_RANK="${SELECT_W_RANK:-0.2}"
SWITCH_MARGIN="${SWITCH_MARGIN:-0.05}"

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" /home/kms/miniconda3/envs/vocot/bin/python /home/kms/LLaVA_calibration/analyze_beamless_trajectory.py \
  --questions_json "${QUESTIONS_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --eval_match_mode "${EVAL_MODE}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --rank_top_k "${RANK_TOP_K}" \
  --k_prefix "${K_PREFIX}" \
  --k_suffix "${K_SUFFIX}" \
  --tau_collapse_gap "${TAU_COLLAPSE_GAP}" \
  --tau_rank_min "${TAU_RANK_MIN}" \
  --tau_energy "${TAU_ENERGY}" \
  --gate_mode "${GATE_MODE}" \
  --branch_top_k "${BRANCH_TOP_K}" \
  --branch_budget "${BRANCH_BUDGET}" \
  --branch_lambda_rank "${BRANCH_LAMBDA_RANK}" \
  --branch_lambda_vpmi "${BRANCH_LAMBDA_VPMI}" \
  --select_w_suffix "${SELECT_W_SUFFIX}" \
  --select_w_rank "${SELECT_W_RANK}" \
  --switch_margin "${SWITCH_MARGIN}"

echo "[done] ${OUT_DIR}"

