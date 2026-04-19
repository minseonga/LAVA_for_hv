#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
PY_BIN="${PY_BIN:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
RUNTIME_BACKEND="${RUNTIME_BACKEND:-llava15_cleanroom}"
LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-/home/kms/LLaVA-NeXT}"
LLAVA_NEXT_TORCH_TYPE="${LLAVA_NEXT_TORCH_TYPE:-fp16}"
LLAVA_NEXT_ATTN_IMPLEMENTATION="${LLAVA_NEXT_ATTN_IMPLEMENTATION:-eager}"

QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
POLICY_BUNDLE_JSON="${POLICY_BUNDLE_JSON:-$CAL_ROOT/experiments/paper_main_meta_vga_full_strong/discovery/meta_calibration/selected_meta_bundle.json}"

INTERVENTION_PRED_JSONL="${INTERVENTION_PRED_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_vga.jsonl}"
# Keep defaults aligned with scripts/run_method_posthoc_b_c_meta_full.sh so
# online parity uses the same teacher-forced candidate text as cached eval.
INTERVENTION_PRED_KEY="${INTERVENTION_PRED_KEY:-output}"
BASELINE_PRED_JSONL="${BASELINE_PRED_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_baseline.jsonl}"
BASELINE_PRED_KEY="${BASELINE_PRED_KEY:-text}"
case "${BASELINE_PRED_JSONL,,}" in
  none|null|live|lazy)
    BASELINE_PRED_JSONL=""
    ;;
esac

OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/discriminative_meta_strong_online}"
LIMIT="${LIMIT:-0}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-false}"
LOG_EVERY="${LOG_EVERY:-25}"
FEATURE_ORDER="${FEATURE_ORDER:-cheap_first}"
CONTROLLER_MODE="${CONTROLLER_MODE:-meta_strong}"
CHEAP_C_TAU_OVERRIDE="${CHEAP_C_TAU_OVERRIDE:-}"
CHEAP_HIDDEN_FEATURES="${CHEAP_HIDDEN_FEATURES:-false}"
STAGE_A_PREFILTER_C_SCORE_MIN="${STAGE_A_PREFILTER_C_SCORE_MIN:-}"

BETA="${BETA:-1.0}"
LAMBDA_A="${LAMBDA_A:-0.5}"
LATE_START="${LATE_START:--1}"
LATE_END="${LATE_END:--1}"
GENERATE_BASELINE_ON_FALLBACK="${GENERATE_BASELINE_ON_FALLBACK:-false}"
BASELINE_MAX_NEW_TOKENS="${BASELINE_MAX_NEW_TOKENS:-8}"

if [[ ! -f "$INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] missing intervention predictions: $INTERVENTION_PRED_JSONL" >&2
  exit 1
fi
if [[ ! -f "$POLICY_BUNDLE_JSON" ]]; then
  echo "[error] missing policy bundle: $POLICY_BUNDLE_JSON" >&2
  exit 1
fi
if [[ ! -f "$HEADSET_JSON" ]]; then
  echo "[error] missing headset: $HEADSET_JSON" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT_DIR"

PYTHONUNBUFFERED=1 PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/run_discriminative_meta_strong_online.py \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --intervention_pred_jsonl "$INTERVENTION_PRED_JSONL" \
  --intervention_pred_key "$INTERVENTION_PRED_KEY" \
  --baseline_pred_jsonl "$BASELINE_PRED_JSONL" \
  --baseline_pred_key "$BASELINE_PRED_KEY" \
  --gt_csv "$GT_CSV" \
  --gt_id_col id \
  --gt_label_col answer \
  --policy_bundle_json "$POLICY_BUNDLE_JSON" \
  --headset_json "$HEADSET_JSON" \
  --out_dir "$OUT_DIR" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --runtime_backend "$RUNTIME_BACKEND" \
  --llava_next_root "$LLAVA_NEXT_ROOT" \
  --llava_next_torch_type "$LLAVA_NEXT_TORCH_TYPE" \
  --llava_next_attn_implementation "$LLAVA_NEXT_ATTN_IMPLEMENTATION" \
  --limit "$LIMIT" \
  --beta "$BETA" \
  --lambda_a "$LAMBDA_A" \
  --late_start "$LATE_START" \
  --late_end "$LATE_END" \
  --feature_order "$FEATURE_ORDER" \
  --controller_mode "$CONTROLLER_MODE" \
  --cheap_hidden_features "$CHEAP_HIDDEN_FEATURES" \
  ${CHEAP_C_TAU_OVERRIDE:+--cheap_c_tau_override "$CHEAP_C_TAU_OVERRIDE"} \
  ${STAGE_A_PREFILTER_C_SCORE_MIN:+--stage_a_prefilter_c_score_min "$STAGE_A_PREFILTER_C_SCORE_MIN"} \
  --generate_baseline_on_fallback "$GENERATE_BASELINE_ON_FALLBACK" \
  --baseline_max_new_tokens "$BASELINE_MAX_NEW_TOKENS" \
  --reuse_if_exists "$REUSE_IF_EXISTS" \
  --log_every "$LOG_EVERY"

echo "[done] $OUT_DIR"
