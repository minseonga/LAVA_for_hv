#!/usr/bin/env bash
set -euo pipefail

# Example:
#   GPU=4 bash /home/kms/LLaVA_calibration/scripts/run_pope_adaptive_ondemand_eval.sh
#   SKIP_GREEDY_GEN=1 SKIP_EXPAND_GEN=1 bash /home/kms/LLaVA_calibration/scripts/run_pope_adaptive_ondemand_eval.sh

GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
QJSON="${QJSON:-/home/kms/LLaVA_calibration/experiments/pope_1000/pope_1000.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
EVAL_MATCH_MODE="${EVAL_MATCH_MODE:-strict}"
VPMI_MIN_MODE="${VPMI_MIN_MODE:-raw}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
BETA_Q="${BETA_Q:-0.8}"
TAU_GAP="${TAU_GAP:-0.65}"

# Expanded pool run (beam6, canonical) regenerated with current analyze code.
EXPAND_DIR="${EXPAND_DIR:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_canonical_v1_sdpa_fullfeat}"

# Greedy run for stage-1 baseline.
GREEDY_DIR="${GREEDY_DIR:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1}"

# Adaptive offline eval output.
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/pope_adaptive_ondemand_eval}"

# generation controls
SKIP_GREEDY_GEN="${SKIP_GREEDY_GEN:-0}"
SKIP_EXPAND_GEN="${SKIP_EXPAND_GEN:-0}"
# backward-compatible alias (previous script used SKIP_GEN for greedy generation)
SKIP_GEN="${SKIP_GEN:-0}"
if [[ "${SKIP_GEN}" == "1" ]]; then
  SKIP_GREEDY_GEN=1
fi

if [[ "${SKIP_GREEDY_GEN}" != "1" ]]; then
  mkdir -p "${GREEDY_DIR}"
  echo "[1/3] generate greedy -> ${GREEDY_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
    --questions_json "${QJSON}" \
    --image_root "${IMAGE_ROOT}" \
    --out_dir "${GREEDY_DIR}" \
    --model_path "${MODEL_PATH}" \
    --num_beams 1 \
    --num_return_sequences 1 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --beta_q "${BETA_Q}" \
    --tau_gap "${TAU_GAP}" \
    --eval_match_mode "${EVAL_MATCH_MODE}" \
    --attn_impl "${ATTN_IMPL}" \
    --vpmi_min_mode "${VPMI_MIN_MODE}"
fi

if [[ "${SKIP_EXPAND_GEN}" != "1" ]]; then
  mkdir -p "${EXPAND_DIR}"
  echo "[2/3] generate beam6 expand -> ${EXPAND_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
    --questions_json "${QJSON}" \
    --image_root "${IMAGE_ROOT}" \
    --out_dir "${EXPAND_DIR}" \
    --model_path "${MODEL_PATH}" \
    --num_beams 6 \
    --num_beam_groups 1 \
    --diversity_penalty 0.0 \
    --num_return_sequences 6 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --beta_q "${BETA_Q}" \
    --tau_gap "${TAU_GAP}" \
    --eval_match_mode "${EVAL_MATCH_MODE}" \
    --attn_impl "${ATTN_IMPL}" \
    --vpmi_min_mode "${VPMI_MIN_MODE}"
fi

mkdir -p "${OUT_DIR}"
echo "[3/3] adaptive offline eval -> ${OUT_DIR}"
python /home/kms/LLaVA_calibration/eval_adaptive_ondemand_offline.py \
  --greedy_dir "${GREEDY_DIR}" \
  --expand_dir "${EXPAND_DIR}" \
  --out_dir "${OUT_DIR}" \
  --eval_mode "${EVAL_MATCH_MODE}" \
  --subset_json "${QJSON}" \
  --policy "agree_vminpm_wmin_dfull_le:-0.05" \
  --trigger "P3" \
  --gates "gate_or_vpmi_m3.5_or_sfull_m10|or|-3.5|-10;gate_and_vpmi_m5.5_and_sfull_m6|and|-5.5|-6" \
  --extra_candidates_cost 5

echo "[done] ${OUT_DIR}"
