#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Defaults (override via env)
# -----------------------------
REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
PYTHON_BIN="${PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
QUESTION_FILE="${QUESTION_FILE:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/01_subset/pope_strict_1000_q.jsonl}"
GT_CSV="${GT_CSV:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/01_subset/pope_strict_1000_gt.csv}"
IDS_CSV="${IDS_CSV:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/01_subset/pope_strict_1000_ids.csv}"

FRGG_FEATS_JSON="${FRGG_FEATS_JSON:-/home/kms/LLaVA_calibration/experiments/frgg_1000/frgg_feats.jsonl}"
RFHAR_FEATS_JSON="${RFHAR_FEATS_JSON:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/05_rfhar_feats/rfhar_feats_oracle.jsonl}"

HEADSET_JSON="${HEADSET_JSON:-/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json}"

OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/frrs_1000}"

# FRRS hyperparams
FRRS_LATE_START="${FRRS_LATE_START:-18}"
FRRS_LATE_END="${FRRS_LATE_END:-21}"
FRRS_TAU_C="${FRRS_TAU_C:-0.0}"
FRRS_TAU_E="${FRRS_TAU_E:-0.0}"
FRRS_K_C="${FRRS_K_C:-8.0}"
FRRS_K_E="${FRRS_K_E:-8.0}"
FRRS_TOPK_RATIO="${FRRS_TOPK_RATIO:-0.2}"
FRRS_HEAD_MODE="${FRRS_HEAD_MODE:-dynamic}"
FRRS_R_PERCENT="${FRRS_R_PERCENT:-0.2}"
FRRS_ONLINE_RECOMPUTE_FEATS="${FRRS_ONLINE_RECOMPUTE_FEATS:-0}"
FRRS_ONLINE_BLEND="${FRRS_ONLINE_BLEND:-1.0}"
FRRS_SUP_ALPHA="${FRRS_SUP_ALPHA:-1.0}"
FRRS_BI_ALPHA="${FRRS_BI_ALPHA:-1.0}"
FRRS_BI_BETA="${FRRS_BI_BETA:-1.0}"

# Decode params
CONV_MODE="${CONV_MODE:-llava_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[info] out_dir=${OUT_DIR}"

# -----------------------------
# 1) Build FRRS features
# -----------------------------
"${PYTHON_BIN}" scripts/build_frrs_feats_from_existing.py \
  --frgg_feats_json "${FRGG_FEATS_JSON}" \
  --rfhar_feats_json "${RFHAR_FEATS_JSON}" \
  --ids_csv "${IDS_CSV}" \
  --out_json "${OUT_DIR}/frrs_feats.jsonl"

# -----------------------------
# 2) Baseline run
# -----------------------------
"${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" \
  --image-folder "${IMAGE_FOLDER}" \
  --question-file "${QUESTION_FILE}" \
  --answers-file "${OUT_DIR}/baseline.jsonl" \
  --conv-mode "${CONV_MODE}" \
  --temperature "${TEMPERATURE}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --dump-first-token-logits "${OUT_DIR}/baseline_first_token.csv"

# -----------------------------
# 3) FRRS supportive
# -----------------------------
"${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" \
  --image-folder "${IMAGE_FOLDER}" \
  --question-file "${QUESTION_FILE}" \
  --answers-file "${OUT_DIR}/frrs_supportive.jsonl" \
  --conv-mode "${CONV_MODE}" \
  --temperature "${TEMPERATURE}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --enable-frrs \
  --frrs-feats-json "${OUT_DIR}/frrs_feats.jsonl" \
  --frrs-arm supportive \
  --frrs-late-start "${FRRS_LATE_START}" \
  --frrs-late-end "${FRRS_LATE_END}" \
  --frrs-alpha "${FRRS_SUP_ALPHA}" \
  --frrs-beta 0.0 \
  --frrs-tau-c "${FRRS_TAU_C}" \
  --frrs-tau-e "${FRRS_TAU_E}" \
  --frrs-k-c "${FRRS_K_C}" \
  --frrs-k-e "${FRRS_K_E}" \
  --frrs-topk-ratio "${FRRS_TOPK_RATIO}" \
  --frrs-head-mode "${FRRS_HEAD_MODE}" \
  --frrs-r-percent "${FRRS_R_PERCENT}" \
  $( [ "${FRRS_ONLINE_RECOMPUTE_FEATS}" = "1" ] && echo "--frrs-online-recompute-feats" ) \
  --frrs-online-blend "${FRRS_ONLINE_BLEND}" \
  --ais-headset-json "${HEADSET_JSON}" \
  --frrs-debug-log \
  --ais-debug-dump "${OUT_DIR}/frrs_supportive_debug.csv" \
  --dump-first-token-logits "${OUT_DIR}/frrs_supportive_first_token.csv" \
  --first-token-logits-include-pre

# -----------------------------
# 4) FRRS bipolar
# -----------------------------
"${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" \
  --image-folder "${IMAGE_FOLDER}" \
  --question-file "${QUESTION_FILE}" \
  --answers-file "${OUT_DIR}/frrs_bipolar.jsonl" \
  --conv-mode "${CONV_MODE}" \
  --temperature "${TEMPERATURE}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --enable-frrs \
  --frrs-feats-json "${OUT_DIR}/frrs_feats.jsonl" \
  --frrs-arm bipolar \
  --frrs-late-start "${FRRS_LATE_START}" \
  --frrs-late-end "${FRRS_LATE_END}" \
  --frrs-alpha "${FRRS_BI_ALPHA}" \
  --frrs-beta "${FRRS_BI_BETA}" \
  --frrs-tau-c "${FRRS_TAU_C}" \
  --frrs-tau-e "${FRRS_TAU_E}" \
  --frrs-k-c "${FRRS_K_C}" \
  --frrs-k-e "${FRRS_K_E}" \
  --frrs-topk-ratio "${FRRS_TOPK_RATIO}" \
  --frrs-head-mode "${FRRS_HEAD_MODE}" \
  --frrs-r-percent "${FRRS_R_PERCENT}" \
  $( [ "${FRRS_ONLINE_RECOMPUTE_FEATS}" = "1" ] && echo "--frrs-online-recompute-feats" ) \
  --frrs-online-blend "${FRRS_ONLINE_BLEND}" \
  --ais-headset-json "${HEADSET_JSON}" \
  --frrs-debug-log \
  --ais-debug-dump "${OUT_DIR}/frrs_bipolar_debug.csv" \
  --dump-first-token-logits "${OUT_DIR}/frrs_bipolar_first_token.csv" \
  --first-token-logits-include-pre

# -----------------------------
# 5) Metrics
# -----------------------------
"${PYTHON_BIN}" scripts/eval_pope_subset_yesno.py \
  --gt_csv "${GT_CSV}" \
  --pred_jsonl "${OUT_DIR}/baseline.jsonl" \
  --out_json "${OUT_DIR}/metrics_baseline.json"

"${PYTHON_BIN}" scripts/eval_pope_subset_yesno.py \
  --gt_csv "${GT_CSV}" \
  --pred_jsonl "${OUT_DIR}/frrs_supportive.jsonl" \
  --out_json "${OUT_DIR}/metrics_frrs_supportive.json"

"${PYTHON_BIN}" scripts/eval_pope_subset_yesno.py \
  --gt_csv "${GT_CSV}" \
  --pred_jsonl "${OUT_DIR}/frrs_bipolar.jsonl" \
  --out_json "${OUT_DIR}/metrics_frrs_bipolar.json"

# -----------------------------
# 6) Baseline vs FRRS compare
# -----------------------------
"${PYTHON_BIN}" scripts/compare_pope_yesno_runs.py \
  --gt_csv "${GT_CSV}" \
  --base_pred_jsonl "${OUT_DIR}/baseline.jsonl" \
  --new_pred_jsonl "${OUT_DIR}/frrs_supportive.jsonl" \
  --out_json "${OUT_DIR}/compare_baseline_vs_frrs_supportive.json" \
  --out_fail_csv "${OUT_DIR}/fail_cases_baseline_vs_frrs_supportive.csv"

"${PYTHON_BIN}" scripts/compare_pope_yesno_runs.py \
  --gt_csv "${GT_CSV}" \
  --base_pred_jsonl "${OUT_DIR}/baseline.jsonl" \
  --new_pred_jsonl "${OUT_DIR}/frrs_bipolar.jsonl" \
  --out_json "${OUT_DIR}/compare_baseline_vs_frrs_bipolar.json" \
  --out_fail_csv "${OUT_DIR}/fail_cases_baseline_vs_frrs_bipolar.csv"

# -----------------------------
# 7) First-token margin compare
# -----------------------------
"${PYTHON_BIN}" scripts/compare_first_token_between_runs.py \
  --base_csv "${OUT_DIR}/baseline_first_token.csv" \
  --new_csv "${OUT_DIR}/frrs_supportive_first_token.csv" \
  --base_pred_jsonl "${OUT_DIR}/baseline.jsonl" \
  --new_pred_jsonl "${OUT_DIR}/frrs_supportive.jsonl" \
  --out_json "${OUT_DIR}/compare_first_token_baseline_vs_frrs_supportive.json" \
  --out_rows_csv "${OUT_DIR}/compare_first_token_baseline_vs_frrs_supportive_rows.csv"

"${PYTHON_BIN}" scripts/compare_first_token_between_runs.py \
  --base_csv "${OUT_DIR}/baseline_first_token.csv" \
  --new_csv "${OUT_DIR}/frrs_bipolar_first_token.csv" \
  --base_pred_jsonl "${OUT_DIR}/baseline.jsonl" \
  --new_pred_jsonl "${OUT_DIR}/frrs_bipolar.jsonl" \
  --out_json "${OUT_DIR}/compare_first_token_baseline_vs_frrs_bipolar.json" \
  --out_rows_csv "${OUT_DIR}/compare_first_token_baseline_vs_frrs_bipolar_rows.csv"

echo "[done] FRRS full run completed: ${OUT_DIR}"
