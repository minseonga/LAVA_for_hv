#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   GPU=4 bash /home/kms/LLaVA_calibration/scripts/run_raw_baseline_regen.sh
#
# Optional env overrides:
#   QUESTIONS_JSON, IMAGE_ROOT, MODEL_PATH, ATTN_IMPL, CONV_MODE, EVAL_MODE, OUT_ROOT, TAG

GPU="${GPU:-0}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_1000questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
CONV_MODE="${CONV_MODE:-llava_v1}"
EVAL_MODE="${EVAL_MODE:-heuristic}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments}"
TAG="${TAG:-raw_baseline_regen}"

OUT_GREEDY="${OUT_ROOT}/artrap_fragility_1000_${TAG}_greedy_b1"
OUT_BEAM6="${OUT_ROOT}/artrap_fragility_1000_${TAG}_beam6_b6"
OUT_EVAL="${OUT_ROOT}/artrap_fragility_1000_${TAG}_baseline_eval"

echo "[1/3] Greedy raw baseline generation (beam=1)"
CUDA_VISIBLE_DEVICES="${GPU}" python /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  --questions_json "${QUESTIONS_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --out_dir "${OUT_GREEDY}" \
  --model_path "${MODEL_PATH}" \
  --conv_mode "${CONV_MODE}" \
  --attn_impl "${ATTN_IMPL}" \
  --num_beams 1 \
  --num_beam_groups 1 \
  --diversity_penalty 0.0 \
  --num_return_sequences 1 \
  --num_extra_samples 0 \
  --max_new_tokens 24 \
  --eval_match_mode "${EVAL_MODE}" \
  --vpmi_min_mode raw

echo "[2/3] Beam-aligned raw baseline generation (beam=6)"
CUDA_VISIBLE_DEVICES="${GPU}" python /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  --questions_json "${QUESTIONS_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --out_dir "${OUT_BEAM6}" \
  --model_path "${MODEL_PATH}" \
  --conv_mode "${CONV_MODE}" \
  --attn_impl "${ATTN_IMPL}" \
  --num_beams 6 \
  --num_beam_groups 1 \
  --diversity_penalty 0.0 \
  --num_return_sequences 6 \
  --num_extra_samples 0 \
  --max_new_tokens 24 \
  --eval_match_mode "${EVAL_MODE}" \
  --vpmi_min_mode raw

echo "[3/3] Raw baseline evaluation table (no switching)"
python /home/kms/LLaVA_calibration/eval_raw_baseline.py \
  --in_dir "${OUT_GREEDY}" "${OUT_BEAM6}" \
  --out_dir "${OUT_EVAL}" \
  --eval_mode "${EVAL_MODE}" \
  --force_recompute

echo
echo "[done] summary:"
cat "${OUT_EVAL}/raw_baseline_table.csv"
echo
echo "[saved] ${OUT_EVAL}/raw_baseline_table.csv"

