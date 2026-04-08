#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
SRC_ROOT="${SRC_ROOT:-$CAL_ROOT/experiments/coco_chair_main_table_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_vga_tree_v1}"

METHOD_NAME="${METHOD_NAME:-vga}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
MAX_MENTIONS="${MAX_MENTIONS:-12}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

SUPPORTED_WEIGHT="${SUPPORTED_WEIGHT:-1.0}"
HALL_WEIGHT="${HALL_WEIGHT:-1.0}"
LENGTH_WEIGHT="${LENGTH_WEIGHT:-0.25}"

TEACHER_MODE="${TEACHER_MODE:-strict_pareto}"
MIN_F1_GAIN="${MIN_F1_GAIN:-0.0}"
FEATURE_COLS="${FEATURE_COLS:-auto}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_N_FEATURES="${TOP_N_FEATURES:-8}"
MAX_DEPTH_VALUES="${MAX_DEPTH_VALUES:-1,2,3}"
MIN_LEAF_VALUES="${MIN_LEAF_VALUES:-3,5,8,10}"
SPLIT_QUANTILES="${SPLIT_QUANTILES:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}"
TAU_QUANTILES="${TAU_QUANTILES:-0.0,0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.0}"
CONSTRAINT_MODE="${CONSTRAINT_MODE:-both}"
CHAIR_EPS="${CHAIR_EPS:-0.0}"
SELECTION_OBJECTIVE="${SELECTION_OBJECTIVE:-f1}"
MIN_BASELINE_RATE="${MIN_BASELINE_RATE:-0.0}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"

if [[ "$METHOD_NAME" != "vga" && "$METHOD_NAME" != "pai" ]]; then
  echo "[error] METHOD_NAME must be one of: vga, pai" >&2
  exit 1
fi

SPLIT_DIR="$SRC_ROOT/splits"
VAL_DIR="$SRC_ROOT/val"
TEST_DIR="$SRC_ROOT/test"

VAL_Q="$SPLIT_DIR/val_caption_q.jsonl"
TEST_Q="$SPLIT_DIR/test_caption_q.jsonl"
VAL_BASELINE_JSONL="$VAL_DIR/pred_vanilla_caption.jsonl"
TEST_BASELINE_JSONL="$TEST_DIR/pred_vanilla_caption.jsonl"
VAL_BASELINE_CHAIR_JSON="$VAL_DIR/chair_baseline.json"
TEST_BASELINE_CHAIR_JSON="$TEST_DIR/chair_baseline.json"

VAL_INTERVENTION_JSONL="$VAL_DIR/pred_${METHOD_NAME}_caption.jsonl"
TEST_INTERVENTION_JSONL="$TEST_DIR/pred_${METHOD_NAME}_caption.jsonl"
VAL_INTERVENTION_CHAIR_JSON="$VAL_DIR/chair_${METHOD_NAME}.json"
TEST_INTERVENTION_CHAIR_JSON="$TEST_DIR/chair_${METHOD_NAME}.json"

DISCOVERY_DIR="$OUT_ROOT/discovery"
TEST_APPLY_DIR="$OUT_ROOT/test_apply"
mkdir -p "$DISCOVERY_DIR" "$TEST_APPLY_DIR"

for path in \
  "$VAL_Q" "$TEST_Q" \
  "$VAL_BASELINE_JSONL" "$TEST_BASELINE_JSONL" \
  "$VAL_BASELINE_CHAIR_JSON" "$TEST_BASELINE_CHAIR_JSON" \
  "$VAL_INTERVENTION_JSONL" "$TEST_INTERVENTION_JSONL" \
  "$VAL_INTERVENTION_CHAIR_JSON" "$TEST_INTERVENTION_CHAIR_JSON"
do
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required input: $path" >&2
    exit 1
  fi
done

build_split_assets() {
  local split_name="$1"
  local q_jsonl="$2"
  local split_dir="$3"
  local baseline_jsonl="$4"
  local intervention_jsonl="$5"
  local baseline_chair_json="$6"
  local intervention_chair_json="$7"

  local feat_csv="$split_dir/coverage_features.csv"
  local feat_summary="$split_dir/coverage_features.summary.json"
  local chair_csv="$split_dir/${METHOD_NAME}_chair_table.csv"
  local chair_summary="$split_dir/${METHOD_NAME}_chair_table.summary.json"
  local claim_csv="$split_dir/${METHOD_NAME}_claim_table.csv"
  local claim_summary="$split_dir/${METHOD_NAME}_claim_table.summary.json"

  echo "[split:$split_name] extract mention features"
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/extract_vga_generative_mention_features.py \
      --question_file "$q_jsonl" \
      --image_folder "$IMAGE_FOLDER" \
      --baseline_pred_jsonl "$baseline_jsonl" \
      --out_csv "$feat_csv" \
      --out_summary_json "$feat_summary" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device cuda \
      --max_mentions "$MAX_MENTIONS" \
      --reuse_if_exists "$REUSE_IF_EXISTS"
  )

  echo "[split:$split_name] build CHAIR table"
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
      --baseline_features_csv "$feat_csv" \
      --baseline_pred_jsonl "$baseline_jsonl" \
      --intervention_pred_jsonl "$intervention_jsonl" \
      --baseline_chair_json "$baseline_chair_json" \
      --intervention_chair_json "$intervention_chair_json" \
      --method_name "$METHOD_NAME" \
      --benchmark_name coco_chair_random500 \
      --split_name "$split_name" \
      --chair_metric CHAIRi \
      --out_csv "$chair_csv" \
      --out_summary_json "$chair_summary" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key output
  )

  echo "[split:$split_name] build claim-aware table"
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_claim_aware_table.py \
      --baseline_features_csv "$feat_csv" \
      --baseline_pred_jsonl "$baseline_jsonl" \
      --intervention_pred_jsonl "$intervention_jsonl" \
      --baseline_chair_json "$baseline_chair_json" \
      --intervention_chair_json "$intervention_chair_json" \
      --method_name "$METHOD_NAME" \
      --benchmark_name coco_chair_random500 \
      --split_name "$split_name" \
      --supported_weight "$SUPPORTED_WEIGHT" \
      --hall_weight "$HALL_WEIGHT" \
      --length_weight "$LENGTH_WEIGHT" \
      --out_csv "$claim_csv" \
      --out_summary_json "$claim_summary" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key output
  )
}

build_split_assets "val" "$VAL_Q" "$DISCOVERY_DIR" "$VAL_BASELINE_JSONL" "$VAL_INTERVENTION_JSONL" "$VAL_BASELINE_CHAIR_JSON" "$VAL_INTERVENTION_CHAIR_JSON"
build_split_assets "test" "$TEST_Q" "$TEST_APPLY_DIR" "$TEST_BASELINE_JSONL" "$TEST_INTERVENTION_JSONL" "$TEST_BASELINE_CHAIR_JSON" "$TEST_INTERVENTION_CHAIR_JSON"

echo "[tree] fit on val"
(
  cd "$CAL_ROOT"
  CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
  CLAIM_TABLE_CSV="$DISCOVERY_DIR/${METHOD_NAME}_claim_table.csv" \
  CHAIR_TABLE_CSV="$DISCOVERY_DIR/${METHOD_NAME}_chair_table.csv" \
  BASELINE_CHAIR_JSON="$VAL_BASELINE_CHAIR_JSON" \
  INTERVENTION_CHAIR_JSON="$VAL_INTERVENTION_CHAIR_JSON" \
  OUT_ROOT="$DISCOVERY_DIR/tree_controller" \
  TEACHER_MODE="$TEACHER_MODE" \
  MIN_F1_GAIN="$MIN_F1_GAIN" \
  FEATURE_COLS="$FEATURE_COLS" \
  MIN_FEATURE_AUROC="$MIN_FEATURE_AUROC" \
  TOP_N_FEATURES="$TOP_N_FEATURES" \
  MAX_DEPTH_VALUES="$MAX_DEPTH_VALUES" \
  MIN_LEAF_VALUES="$MIN_LEAF_VALUES" \
  SPLIT_QUANTILES="$SPLIT_QUANTILES" \
  TAU_QUANTILES="$TAU_QUANTILES" \
  CONSTRAINT_MODE="$CONSTRAINT_MODE" \
  CHAIR_EPS="$CHAIR_EPS" \
  SELECTION_OBJECTIVE="$SELECTION_OBJECTIVE" \
  MIN_BASELINE_RATE="$MIN_BASELINE_RATE" \
  MAX_BASELINE_RATE="$MAX_BASELINE_RATE" \
  bash scripts/run_method_generative_pareto_teacher_tree.sh
)

echo "[tree] apply frozen policy to test"
(
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/apply_generative_pareto_teacher_tree_controller.py \
    --claim_table_csv "$TEST_APPLY_DIR/${METHOD_NAME}_claim_table.csv" \
    --chair_table_csv "$TEST_APPLY_DIR/${METHOD_NAME}_chair_table.csv" \
    --baseline_chair_json "$TEST_BASELINE_CHAIR_JSON" \
    --intervention_chair_json "$TEST_INTERVENTION_CHAIR_JSON" \
    --selected_tree_json "$DISCOVERY_DIR/tree_controller/selected_tree.json" \
    --out_dir "$TEST_APPLY_DIR/tree_apply"
)

echo "[done] $TEST_APPLY_DIR/tree_apply/summary.json"
