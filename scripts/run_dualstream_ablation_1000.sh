#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_1000questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/dualstream_ablation_1000}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SEED="${SEED:-42}"
EVAL_MODE="${EVAL_MODE:-heuristic}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"

# A/B hyperparameters (override via env)
TOP_K_RANK="${TOP_K_RANK:-50}"
ALPHA_RANK="${ALPHA_RANK:-1.2}"
STATE_TOP_K="${STATE_TOP_K:-50}"
PREFIX_SPIKE_Z="${PREFIX_SPIKE_Z:-1.0}"
BETA_PREFIX="${BETA_PREFIX:-1.0}"
SUFFIX_VPMI_FLOOR="${SUFFIX_VPMI_FLOOR:--0.5}"
BETA_SUFFIX="${BETA_SUFFIX:-0.8}"

mkdir -p "${OUT_ROOT}"

COMMON_ARGS=(
  --questions_json "${QUESTIONS_JSON}"
  --image_root "${IMAGE_ROOT}"
  --model_path "${MODEL_PATH}"
  --eval_match_mode "${EVAL_MODE}"
  --num_samples "${NUM_SAMPLES}"
  --seed "${SEED}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --top_k_rank "${TOP_K_RANK}"
  --alpha_rank "${ALPHA_RANK}"
  --state_top_k "${STATE_TOP_K}"
  --prefix_spike_z "${PREFIX_SPIKE_Z}"
  --beta_prefix "${BETA_PREFIX}"
  --suffix_vpmi_floor "${SUFFIX_VPMI_FLOOR}"
  --beta_suffix "${BETA_SUFFIX}"
)

for METHOD in a_only b_only a_plus_b; do
  OUT_DIR="${OUT_ROOT}/${METHOD}"
  mkdir -p "${OUT_DIR}"
  echo "[run] method=${METHOD} out=${OUT_DIR}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" /home/kms/miniconda3/envs/vocot/bin/python /home/kms/LLaVA_calibration/analyze_dualstream_rank_state.py \
    "${COMMON_ARGS[@]}" \
    --method "${METHOD}" \
    --out_dir "${OUT_DIR}"
done

echo "[done] ${OUT_ROOT}"
