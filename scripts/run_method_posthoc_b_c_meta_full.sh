#!/usr/bin/env bash
set -Eeuo pipefail

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

TEST_IMAGE_FOLDER="${TEST_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
TEST_QUESTION_FILE="${TEST_QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
TEST_BASELINE_QUESTION_FILE="${TEST_BASELINE_QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl}"
TEST_GT_CSV="${TEST_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
TEST_HEADSET_JSON="${TEST_HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
TEST_INTERVENTION_PRED_JSONL="${TEST_INTERVENTION_PRED_JSONL:-}"
TEST_INTERVENTION_PRED_KEY="${TEST_INTERVENTION_PRED_KEY:-output}"
TEST_BASELINE_PRED_JSONL="${TEST_BASELINE_PRED_JSONL:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/baseline/pred_vanilla_9000.jsonl}"
TEST_BASELINE_PRED_KEY="${TEST_BASELINE_PRED_KEY:-text}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/main_b_c_meta_${METHOD_NAME}}"
DISCOVERY_STAGEB_OUT_DIR="${DISCOVERY_STAGEB_OUT_DIR:-$OUT_ROOT/discovery_stageb}"
DISCOVERY_OUT_DIR="${DISCOVERY_OUT_DIR:-$OUT_ROOT/discovery}"
TEST_STAGEB_OUT_DIR="${TEST_STAGEB_OUT_DIR:-$OUT_ROOT/test_stageb}"
TEST_OUT_DIR="${TEST_OUT_DIR:-$OUT_ROOT/test}"
DISCOVERY_CHEAP_CSV="${DISCOVERY_CHEAP_CSV:-$DISCOVERY_OUT_DIR/cheap_online_features.csv}"
TEST_CHEAP_CSV="${TEST_CHEAP_CSV:-$TEST_OUT_DIR/cheap_online_features.csv}"

REUSE_CHEAP="${REUSE_CHEAP:-true}"
REUSE_SCORES="${REUSE_SCORES:-true}"
LOG_EVERY="${LOG_EVERY:-25}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"

B_FEATURE_COLS="${B_FEATURE_COLS:-stage_a_score}"
C_FEATURE_COLS="${C_FEATURE_COLS:-cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K_C="${TOP_K_C:-3}"
WEIGHT_GRID="${WEIGHT_GRID:-0.25,0.5,0.75,1.0,1.5,2.0,3.0}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-final_acc}"
MIN_BASELINE_RATE="${MIN_BASELINE_RATE:-0.0}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"
DELTA_GRID="${DELTA_GRID:-0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0}"
META_MODES="${META_MODES:-delta_then_fusion,delta_then_stronger,agree_fusion_else_stronger}"

mkdir -p "$DISCOVERY_STAGEB_OUT_DIR" "$DISCOVERY_OUT_DIR" "$TEST_STAGEB_OUT_DIR" "$TEST_OUT_DIR"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
mkdir -p "$LOG_DIR"
CURRENT_STEP=""

trap 'status=$?; if [[ $status -ne 0 ]]; then echo "[error] step ${CURRENT_STEP:-unknown} failed. log: $LOG_DIR/${CURRENT_STEP:-unknown}.log" >&2; fi' ERR

run_step() {
  local step_name="$1"
  shift
  CURRENT_STEP="$step_name"
  local log_path="$LOG_DIR/${step_name}.log"
  echo "[log] $log_path"
  "$@" 2>&1 | tee "$log_path"
}

if [[ -z "$DISCOVERY_INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] DISCOVERY_INTERVENTION_PRED_JSONL is required" >&2
  exit 1
fi
if [[ -z "$TEST_INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] TEST_INTERVENTION_PRED_JSONL is required" >&2
  exit 1
fi

echo "[1/6] discovery Stage-B from existing ${METHOD_NAME} predictions"
run_step "01_discovery_stageb" env \
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
run_step "02_discovery_cheap" env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONUNBUFFERED=1 \
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

echo "[3/6] discovery meta calibration"
run_step "03_discovery_meta" env \
  PYTHONPATH="$CAL_ROOT" \
  "$PY_BIN" scripts/build_posthoc_b_c_meta_controller.py \
    --scores_csv "$DISCOVERY_STAGEB_OUT_DIR/sample_scores.csv" \
    --features_csv "$DISCOVERY_CHEAP_CSV" \
    --out_dir "$DISCOVERY_OUT_DIR/meta_calibration" \
    --b_feature_cols "$B_FEATURE_COLS" \
    --c_feature_cols "$C_FEATURE_COLS" \
    --min_feature_auroc "$MIN_FEATURE_AUROC" \
    --top_k_c "$TOP_K_C" \
    --weight_grid "$WEIGHT_GRID" \
    --tau_objective "$TAU_OBJECTIVE" \
    --min_baseline_rate "$MIN_BASELINE_RATE" \
    --max_baseline_rate "$MAX_BASELINE_RATE" \
    --min_selected_count "$MIN_SELECTED_COUNT" \
    --delta_grid "$DELTA_GRID" \
    --meta_modes "$META_MODES"

echo "[4/6] held-out Stage-B from existing ${METHOD_NAME} predictions"
run_step "04_test_stageb" env \
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
run_step "05_test_cheap" env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONUNBUFFERED=1 \
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

echo "[6/6] held-out meta fixed eval"
run_step "06_test_meta_apply" env \
  PYTHONPATH="$CAL_ROOT" \
  "$PY_BIN" scripts/apply_posthoc_b_c_meta_controller.py \
    --scores_csv "$TEST_STAGEB_OUT_DIR/sample_scores.csv" \
    --features_csv "$TEST_CHEAP_CSV" \
    --policy_bundle_json "$DISCOVERY_OUT_DIR/meta_calibration/selected_meta_bundle.json" \
    --out_dir "$TEST_OUT_DIR/meta_fixed_eval"

echo "[done] discovery meta calibration -> $DISCOVERY_OUT_DIR/meta_calibration"
echo "[done] held-out meta fixed eval -> $TEST_OUT_DIR/meta_fixed_eval"
