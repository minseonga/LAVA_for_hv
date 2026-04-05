#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
PY_BIN="${PY_BIN:-python}"
METHOD_NAME="${METHOD_NAME:-method}"

DISCOVERY_IMAGE_FOLDER="${DISCOVERY_IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
DISCOVERY_QUESTION_FILE="${DISCOVERY_QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
DISCOVERY_BASELINE_QUESTION_FILE="${DISCOVERY_BASELINE_QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q.jsonl}"
DISCOVERY_GT_CSV="${DISCOVERY_GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
DISCOVERY_HEADSET_JSON="${DISCOVERY_HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
DISCOVERY_INTERVENTION_PRED_JSONL="${DISCOVERY_INTERVENTION_PRED_JSONL:-}"
DISCOVERY_INTERVENTION_PRED_KEY="${DISCOVERY_INTERVENTION_PRED_KEY:-output}"
DISCOVERY_BASELINE_PRED_JSONL="${DISCOVERY_BASELINE_PRED_JSONL:-$CAL_ROOT/experiments/common_pope_discovery_v3_panel_v1/discriminative/baseline/pred_vanilla_discovery.jsonl}"
DISCOVERY_BASELINE_PRED_KEY="${DISCOVERY_BASELINE_PRED_KEY:-text}"
DISCOVERY_TAXONOMY_CSV="${DISCOVERY_TAXONOMY_CSV:-}"

TEST_IMAGE_FOLDER="${TEST_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
TEST_QUESTION_FILE="${TEST_QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
TEST_BASELINE_QUESTION_FILE="${TEST_BASELINE_QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl}"
TEST_GT_CSV="${TEST_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
TEST_HEADSET_JSON="${TEST_HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
TEST_INTERVENTION_PRED_JSONL="${TEST_INTERVENTION_PRED_JSONL:-}"
TEST_INTERVENTION_PRED_KEY="${TEST_INTERVENTION_PRED_KEY:-output}"
TEST_BASELINE_PRED_JSONL="${TEST_BASELINE_PRED_JSONL:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/baseline/pred_vanilla_9000.jsonl}"
TEST_BASELINE_PRED_KEY="${TEST_BASELINE_PRED_KEY:-text}"
TEST_TAXONOMY_CSV="${TEST_TAXONOMY_CSV:-}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/main_b_c_fixed_policy_${METHOD_NAME}}"
DISCOVERY_STAGEB_OUT_DIR="${DISCOVERY_STAGEB_OUT_DIR:-$OUT_ROOT/discovery_stageb}"
DISCOVERY_OUT_DIR="${DISCOVERY_OUT_DIR:-$OUT_ROOT/discovery}"
TEST_STAGEB_OUT_DIR="${TEST_STAGEB_OUT_DIR:-$OUT_ROOT/test_stageb}"
TEST_OUT_DIR="${TEST_OUT_DIR:-$OUT_ROOT/test}"
DISCOVERY_CHEAP_CSV="${DISCOVERY_CHEAP_CSV:-$DISCOVERY_OUT_DIR/cheap_online_features.csv}"
TEST_CHEAP_CSV="${TEST_CHEAP_CSV:-$TEST_OUT_DIR/cheap_online_features.csv}"

SUBSET_PERCENTS="${SUBSET_PERCENTS:-1,2,5}"
FEATURE_COLS="${FEATURE_COLS:-cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy}"
MAX_RESCUE_RATE="${MAX_RESCUE_RATE:-0.03}"
REUSE_CHEAP="${REUSE_CHEAP:-true}"
REUSE_SCORES="${REUSE_SCORES:-true}"
LOG_EVERY="${LOG_EVERY:-25}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"

mkdir -p "$DISCOVERY_STAGEB_OUT_DIR" "$DISCOVERY_OUT_DIR" "$TEST_STAGEB_OUT_DIR" "$TEST_OUT_DIR"

if [[ -z "$DISCOVERY_INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] DISCOVERY_INTERVENTION_PRED_JSONL is required" >&2
  exit 1
fi
if [[ -z "$TEST_INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] TEST_INTERVENTION_PRED_JSONL is required" >&2
  exit 1
fi

echo "[1/6] discovery Stage-B from existing ${METHOD_NAME} predictions"
CUDA_VISIBLE_DEVICES="$GPU" \
CAL_ROOT="$CAL_ROOT" \
PY_BIN="$PY_BIN" \
DEVICE="$DEVICE" \
MODEL_PATH="$MODEL_PATH" \
MODEL_BASE="$MODEL_BASE" \
CONV_MODE="$CONV_MODE" \
IMAGE_FOLDER="$DISCOVERY_IMAGE_FOLDER" \
QUESTION_FILE="$DISCOVERY_QUESTION_FILE" \
BASELINE_QUESTION_FILE="$DISCOVERY_BASELINE_QUESTION_FILE" \
GT_CSV="$DISCOVERY_GT_CSV" \
HEADSET_JSON="$DISCOVERY_HEADSET_JSON" \
OUT_DIR="$DISCOVERY_STAGEB_OUT_DIR" \
INTERVENTION_PRED_JSONL="$DISCOVERY_INTERVENTION_PRED_JSONL" \
INTERVENTION_PRED_KEY="$DISCOVERY_INTERVENTION_PRED_KEY" \
BASELINE_PRED_JSONL="$DISCOVERY_BASELINE_PRED_JSONL" \
BASELINE_PRED_KEY="$DISCOVERY_BASELINE_PRED_KEY" \
MAX_GEN_LEN="$MAX_GEN_LEN" \
REUSE_SCORES="$REUSE_SCORES" \
bash scripts/run_stage_b_signal_validation_from_preds.sh

echo "[2/6] discovery cheap feature extraction"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$CAL_ROOT" \
"$PY_BIN" scripts/extract_c_stage_cheap_online_features.py \
  --question_file "$DISCOVERY_QUESTION_FILE" \
  --image_folder "$DISCOVERY_IMAGE_FOLDER" \
  --intervention_pred_jsonl "$DISCOVERY_INTERVENTION_PRED_JSONL" \
  --out_csv "$DISCOVERY_CHEAP_CSV" \
  --out_summary_json "$DISCOVERY_OUT_DIR/cheap_online_features_summary.json" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --reuse_if_exists "$REUSE_CHEAP" \
  --log_every "$LOG_EVERY"

echo "[3/6] discovery calibration"
PYTHONPATH="$CAL_ROOT" \
"$PY_BIN" scripts/run_b_c_fixed_policy.py calibrate \
  --scores_csv "$DISCOVERY_STAGEB_OUT_DIR/sample_scores.csv" \
  --taxonomy_csv "$DISCOVERY_TAXONOMY_CSV" \
  --features_csv "$DISCOVERY_CHEAP_CSV" \
  --subset_percents "$SUBSET_PERCENTS" \
  --feature_cols "$FEATURE_COLS" \
  --max_rescue_rate "$MAX_RESCUE_RATE" \
  --out_dir "$DISCOVERY_OUT_DIR/calibration"

echo "[4/6] held-out Stage-B from existing ${METHOD_NAME} predictions"
CUDA_VISIBLE_DEVICES="$GPU" \
CAL_ROOT="$CAL_ROOT" \
PY_BIN="$PY_BIN" \
DEVICE="$DEVICE" \
MODEL_PATH="$MODEL_PATH" \
MODEL_BASE="$MODEL_BASE" \
CONV_MODE="$CONV_MODE" \
IMAGE_FOLDER="$TEST_IMAGE_FOLDER" \
QUESTION_FILE="$TEST_QUESTION_FILE" \
BASELINE_QUESTION_FILE="$TEST_BASELINE_QUESTION_FILE" \
GT_CSV="$TEST_GT_CSV" \
HEADSET_JSON="$TEST_HEADSET_JSON" \
OUT_DIR="$TEST_STAGEB_OUT_DIR" \
INTERVENTION_PRED_JSONL="$TEST_INTERVENTION_PRED_JSONL" \
INTERVENTION_PRED_KEY="$TEST_INTERVENTION_PRED_KEY" \
BASELINE_PRED_JSONL="$TEST_BASELINE_PRED_JSONL" \
BASELINE_PRED_KEY="$TEST_BASELINE_PRED_KEY" \
MAX_GEN_LEN="$MAX_GEN_LEN" \
REUSE_SCORES="$REUSE_SCORES" \
bash scripts/run_stage_b_signal_validation_from_preds.sh

echo "[5/6] held-out cheap feature extraction"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$CAL_ROOT" \
"$PY_BIN" scripts/extract_c_stage_cheap_online_features.py \
  --question_file "$TEST_QUESTION_FILE" \
  --image_folder "$TEST_IMAGE_FOLDER" \
  --intervention_pred_jsonl "$TEST_INTERVENTION_PRED_JSONL" \
  --out_csv "$TEST_CHEAP_CSV" \
  --out_summary_json "$TEST_OUT_DIR/cheap_online_features_summary.json" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --reuse_if_exists "$REUSE_CHEAP" \
  --log_every "$LOG_EVERY"

echo "[6/6] held-out fixed eval"
PYTHONPATH="$CAL_ROOT" \
"$PY_BIN" scripts/run_b_c_fixed_policy.py apply \
  --scores_csv "$TEST_STAGEB_OUT_DIR/sample_scores.csv" \
  --taxonomy_csv "$TEST_TAXONOMY_CSV" \
  --features_csv "$TEST_CHEAP_CSV" \
  --policy_json "$DISCOVERY_OUT_DIR/calibration/selected_policy.json" \
  --out_dir "$TEST_OUT_DIR/fixed_eval"

echo "[done] discovery calibration -> $DISCOVERY_OUT_DIR/calibration"
echo "[done] held-out fixed eval -> $TEST_OUT_DIR/fixed_eval"
