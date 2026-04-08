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
DISCOVERY_MODE="${DISCOVERY_MODE:-coco_val}"

INTERVENTION_PRED_TEXT_KEY="${INTERVENTION_PRED_TEXT_KEY:-}"
if [[ -z "$INTERVENTION_PRED_TEXT_KEY" ]]; then
  if [[ "$METHOD_NAME" == "pai" ]]; then
    INTERVENTION_PRED_TEXT_KEY="text"
  else
    INTERVENTION_PRED_TEXT_KEY="output"
  fi
fi

DISCOVERY_FEATURES_CSV="${DISCOVERY_FEATURES_CSV:-}"
DISCOVERY_CLAIM_TABLE_CSV="${DISCOVERY_CLAIM_TABLE_CSV:-}"
DISCOVERY_CHAIR_TABLE_CSV="${DISCOVERY_CHAIR_TABLE_CSV:-}"
DISCOVERY_BASELINE_PRED_JSONL="${DISCOVERY_BASELINE_PRED_JSONL:-}"
DISCOVERY_INTERVENTION_PRED_JSONL="${DISCOVERY_INTERVENTION_PRED_JSONL:-}"
DISCOVERY_BASELINE_CHAIR_JSON="${DISCOVERY_BASELINE_CHAIR_JSON:-}"
DISCOVERY_INTERVENTION_CHAIR_JSON="${DISCOVERY_INTERVENTION_CHAIR_JSON:-}"
DISCOVERY_BENCHMARK_NAME="${DISCOVERY_BENCHMARK_NAME:-}"
DISCOVERY_SPLIT_NAME="${DISCOVERY_SPLIT_NAME:-}"
DISCOVERY_BASELINE_PRED_TEXT_KEY="${DISCOVERY_BASELINE_PRED_TEXT_KEY:-auto}"
DISCOVERY_INTERVENTION_PRED_TEXT_KEY="${DISCOVERY_INTERVENTION_PRED_TEXT_KEY:-$INTERVENTION_PRED_TEXT_KEY}"

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

REQUIRED_INPUTS=(
  "$TEST_Q"
  "$TEST_BASELINE_JSONL"
  "$TEST_BASELINE_CHAIR_JSON"
  "$TEST_INTERVENTION_JSONL"
  "$TEST_INTERVENTION_CHAIR_JSON"
)

if [[ "$DISCOVERY_MODE" != "legacy_probe_200" ]]; then
  REQUIRED_INPUTS+=(
    "$VAL_Q"
    "$VAL_BASELINE_JSONL"
    "$VAL_BASELINE_CHAIR_JSON"
    "$VAL_INTERVENTION_JSONL"
    "$VAL_INTERVENTION_CHAIR_JSON"
  )
fi

for path in "${REQUIRED_INPUTS[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required input: $path" >&2
    exit 1
  fi
done

build_split_assets() {
  local split_name="$1"
  local benchmark_name="$2"
  local q_jsonl="$3"
  local split_dir="$4"
  local baseline_jsonl="$5"
  local intervention_jsonl="$6"
  local baseline_chair_json="$7"
  local intervention_chair_json="$8"

  local feat_csv="$split_dir/coverage_features.csv"
  local feat_probe_csv="$split_dir/coverage_features.probe.csv"
  local feat_probe_summary="$split_dir/coverage_features.probe.summary.json"
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
      --out_csv "$feat_probe_csv" \
      --out_summary_json "$feat_probe_summary" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device cuda \
      --max_mentions "$MAX_MENTIONS" \
      --reuse_if_exists "$REUSE_IF_EXISTS"
  )

  echo "[split:$split_name] extract pairwise features"
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/extract_generative_pairwise_features.py \
      --question_file "$q_jsonl" \
      --baseline_pred_jsonl "$baseline_jsonl" \
      --intervention_pred_jsonl "$intervention_jsonl" \
      --base_features_csv "$feat_probe_csv" \
      --out_csv "$feat_csv" \
      --out_summary_json "$feat_summary" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key "$INTERVENTION_PRED_TEXT_KEY" \
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
      --benchmark_name "$benchmark_name" \
      --split_name "$split_name" \
      --chair_metric CHAIRi \
      --out_csv "$chair_csv" \
      --out_summary_json "$chair_summary" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key "$INTERVENTION_PRED_TEXT_KEY"
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
      --benchmark_name "$benchmark_name" \
      --split_name "$split_name" \
      --supported_weight "$SUPPORTED_WEIGHT" \
      --hall_weight "$HALL_WEIGHT" \
      --length_weight "$LENGTH_WEIGHT" \
      --out_csv "$claim_csv" \
      --out_summary_json "$claim_summary" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key "$INTERVENTION_PRED_TEXT_KEY"
  )
}

TREE_DISCOVERY_CLAIM_TABLE_CSV=""
TREE_DISCOVERY_CHAIR_TABLE_CSV=""
TREE_DISCOVERY_BASELINE_CHAIR_JSON=""
TREE_DISCOVERY_INTERVENTION_CHAIR_JSON=""

build_legacy_discovery_assets() {
  if [[ -z "$DISCOVERY_CLAIM_TABLE_CSV" && -z "$DISCOVERY_FEATURES_CSV" && "$METHOD_NAME" == "vga" ]]; then
    DISCOVERY_FEATURES_CSV="$CAL_ROOT/experiments/vga_generative_coverage_probe_v1/coverage_features.csv"
    DISCOVERY_CLAIM_TABLE_CSV="$CAL_ROOT/experiments/vga_generative_coverage_probe_v1/vga_claim_aware_table.csv"
    DISCOVERY_BASELINE_PRED_JSONL="$CAL_ROOT/experiments/common_pope_discovery_harm_miner_v2/generative/baseline/pred_vanilla_caption.jsonl"
    DISCOVERY_INTERVENTION_PRED_JSONL="$CAL_ROOT/experiments/common_pope_discovery_harm_miner_v2/generative/vga/pred_vga_caption.jsonl"
    DISCOVERY_BASELINE_CHAIR_JSON="$CAL_ROOT/experiments/common_pope_discovery_harm_miner_v2/generative/baseline/chair_baseline.json"
    DISCOVERY_INTERVENTION_CHAIR_JSON="$CAL_ROOT/experiments/common_pope_discovery_harm_miner_v2/generative/vga/chair_vga.json"
    DISCOVERY_BENCHMARK_NAME="${DISCOVERY_BENCHMARK_NAME:-pope_discovery_caption}"
    DISCOVERY_SPLIT_NAME="${DISCOVERY_SPLIT_NAME:-coverage_probe}"
  fi

  DISCOVERY_CHAIR_TABLE_CSV="${DISCOVERY_CHAIR_TABLE_CSV:-$DISCOVERY_DIR/${METHOD_NAME}_chair_table.csv}"
  DISCOVERY_BENCHMARK_NAME="${DISCOVERY_BENCHMARK_NAME:-legacy_discovery}"
  DISCOVERY_SPLIT_NAME="${DISCOVERY_SPLIT_NAME:-legacy_discovery}"

  if [[ ! -f "$DISCOVERY_CHAIR_TABLE_CSV" ]]; then
    for path in \
      "$DISCOVERY_FEATURES_CSV" \
      "$DISCOVERY_BASELINE_PRED_JSONL" \
      "$DISCOVERY_INTERVENTION_PRED_JSONL" \
      "$DISCOVERY_BASELINE_CHAIR_JSON" \
      "$DISCOVERY_INTERVENTION_CHAIR_JSON"
    do
      if [[ -z "$path" || ! -f "$path" ]]; then
        echo "[error] legacy discovery chair-table rebuild requires existing file: $path" >&2
        exit 1
      fi
    done
    echo "[discovery] rebuild legacy chair table"
    (
      cd "$CAL_ROOT"
      PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
        --baseline_features_csv "$DISCOVERY_FEATURES_CSV" \
        --baseline_pred_jsonl "$DISCOVERY_BASELINE_PRED_JSONL" \
        --intervention_pred_jsonl "$DISCOVERY_INTERVENTION_PRED_JSONL" \
        --baseline_chair_json "$DISCOVERY_BASELINE_CHAIR_JSON" \
        --intervention_chair_json "$DISCOVERY_INTERVENTION_CHAIR_JSON" \
        --method_name "$METHOD_NAME" \
        --benchmark_name "$DISCOVERY_BENCHMARK_NAME" \
        --split_name "$DISCOVERY_SPLIT_NAME" \
        --chair_metric CHAIRi \
        --out_csv "$DISCOVERY_CHAIR_TABLE_CSV" \
        --out_summary_json "$DISCOVERY_DIR/${METHOD_NAME}_chair_table.summary.json" \
        --baseline_pred_text_key "$DISCOVERY_BASELINE_PRED_TEXT_KEY" \
        --intervention_pred_text_key "$DISCOVERY_INTERVENTION_PRED_TEXT_KEY"
    )
  fi

  if [[ -z "$DISCOVERY_CLAIM_TABLE_CSV" || ! -f "$DISCOVERY_CLAIM_TABLE_CSV" ]]; then
    for path in \
      "$DISCOVERY_FEATURES_CSV" \
      "$DISCOVERY_BASELINE_PRED_JSONL" \
      "$DISCOVERY_INTERVENTION_PRED_JSONL" \
      "$DISCOVERY_BASELINE_CHAIR_JSON" \
      "$DISCOVERY_INTERVENTION_CHAIR_JSON"
    do
      if [[ -z "$path" || ! -f "$path" ]]; then
        echo "[error] legacy discovery claim-table rebuild requires existing file: $path" >&2
        exit 1
      fi
    done
    DISCOVERY_CLAIM_TABLE_CSV="$DISCOVERY_DIR/${METHOD_NAME}_claim_table.csv"
    echo "[discovery] rebuild legacy claim-aware table"
    (
      cd "$CAL_ROOT"
      PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_claim_aware_table.py \
        --baseline_features_csv "$DISCOVERY_FEATURES_CSV" \
        --baseline_pred_jsonl "$DISCOVERY_BASELINE_PRED_JSONL" \
        --intervention_pred_jsonl "$DISCOVERY_INTERVENTION_PRED_JSONL" \
        --baseline_chair_json "$DISCOVERY_BASELINE_CHAIR_JSON" \
        --intervention_chair_json "$DISCOVERY_INTERVENTION_CHAIR_JSON" \
        --method_name "$METHOD_NAME" \
        --benchmark_name "$DISCOVERY_BENCHMARK_NAME" \
        --split_name "$DISCOVERY_SPLIT_NAME" \
        --supported_weight "$SUPPORTED_WEIGHT" \
        --hall_weight "$HALL_WEIGHT" \
        --length_weight "$LENGTH_WEIGHT" \
        --out_csv "$DISCOVERY_CLAIM_TABLE_CSV" \
        --out_summary_json "$DISCOVERY_DIR/${METHOD_NAME}_claim_table.summary.json" \
        --baseline_pred_text_key "$DISCOVERY_BASELINE_PRED_TEXT_KEY" \
        --intervention_pred_text_key "$DISCOVERY_INTERVENTION_PRED_TEXT_KEY"
    )
  fi

  for path in \
    "$DISCOVERY_CLAIM_TABLE_CSV" \
    "$DISCOVERY_CHAIR_TABLE_CSV" \
    "$DISCOVERY_BASELINE_CHAIR_JSON" \
    "$DISCOVERY_INTERVENTION_CHAIR_JSON"
  do
    if [[ -z "$path" || ! -f "$path" ]]; then
      echo "[error] missing legacy discovery input: $path" >&2
      exit 1
    fi
  done

  TREE_DISCOVERY_CLAIM_TABLE_CSV="$DISCOVERY_CLAIM_TABLE_CSV"
  TREE_DISCOVERY_CHAIR_TABLE_CSV="$DISCOVERY_CHAIR_TABLE_CSV"
  TREE_DISCOVERY_BASELINE_CHAIR_JSON="$DISCOVERY_BASELINE_CHAIR_JSON"
  TREE_DISCOVERY_INTERVENTION_CHAIR_JSON="$DISCOVERY_INTERVENTION_CHAIR_JSON"
}

if [[ "$DISCOVERY_MODE" == "legacy_probe_200" ]]; then
  build_legacy_discovery_assets
else
  build_split_assets "val" "coco_chair_random500" "$VAL_Q" "$DISCOVERY_DIR" "$VAL_BASELINE_JSONL" "$VAL_INTERVENTION_JSONL" "$VAL_BASELINE_CHAIR_JSON" "$VAL_INTERVENTION_CHAIR_JSON"
  TREE_DISCOVERY_CLAIM_TABLE_CSV="$DISCOVERY_DIR/${METHOD_NAME}_claim_table.csv"
  TREE_DISCOVERY_CHAIR_TABLE_CSV="$DISCOVERY_DIR/${METHOD_NAME}_chair_table.csv"
  TREE_DISCOVERY_BASELINE_CHAIR_JSON="$VAL_BASELINE_CHAIR_JSON"
  TREE_DISCOVERY_INTERVENTION_CHAIR_JSON="$VAL_INTERVENTION_CHAIR_JSON"
fi

build_split_assets "test" "coco_chair_random500" "$TEST_Q" "$TEST_APPLY_DIR" "$TEST_BASELINE_JSONL" "$TEST_INTERVENTION_JSONL" "$TEST_BASELINE_CHAIR_JSON" "$TEST_INTERVENTION_CHAIR_JSON"

echo "[tree] fit on val"
(
  cd "$CAL_ROOT"
  CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
  CLAIM_TABLE_CSV="$TREE_DISCOVERY_CLAIM_TABLE_CSV" \
  CHAIR_TABLE_CSV="$TREE_DISCOVERY_CHAIR_TABLE_CSV" \
  BASELINE_CHAIR_JSON="$TREE_DISCOVERY_BASELINE_CHAIR_JSON" \
  INTERVENTION_CHAIR_JSON="$TREE_DISCOVERY_INTERVENTION_CHAIR_JSON" \
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
