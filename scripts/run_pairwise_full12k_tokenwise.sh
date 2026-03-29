#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
SEED="${SEED:-42}"
SAVE_TOP2_MARGIN="${SAVE_TOP2_MARGIN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/proof_pairwise_full12k_tokenwise_${STAMP}}"
mkdir -p "${OUT_ROOT}"

COMMON_ARGS=(
  --questions_json "${QUESTIONS_JSON}"
  --image_root "${IMAGE_ROOT}"
  --model_path "${MODEL_PATH}"
  --eval_match_mode heuristic
  --num_samples 0
  --save_core_tokenwise_vpmi
  --attn_impl "${ATTN_IMPL}"
  --seed "${SEED}"
)

if [[ "${SAVE_TOP2_MARGIN}" == "1" ]]; then
  COMMON_ARGS+=(--save_top2_margin)
fi

echo "[run] greedy_b1 -> ${OUT_ROOT}/greedy_b1"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  "${COMMON_ARGS[@]}" \
  --out_dir "${OUT_ROOT}/greedy_b1" \
  --num_beams 1 \
  --num_return_sequences 1

echo "[run] beam6_b6 -> ${OUT_ROOT}/beam6_b6"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  "${COMMON_ARGS[@]}" \
  --out_dir "${OUT_ROOT}/beam6_b6" \
  --num_beams 6 \
  --num_return_sequences 6

echo "[done] pairwise roots:"
echo "  ${OUT_ROOT}/greedy_b1"
echo "  ${OUT_ROOT}/beam6_b6"

