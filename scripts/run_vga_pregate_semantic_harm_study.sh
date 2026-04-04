#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
AMBER_ROOT="${AMBER_ROOT:-/home/kms/data/AMBER}"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_harm_v1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K="${TOP_K:-3}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"
AMBER_DISCOVERY_RATIO="${AMBER_DISCOVERY_RATIO:-0.2}"
FEATURE_COLS="${FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"

DISCOVERY_IMAGE_FOLDER="${DISCOVERY_IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
POPE_IMAGE_FOLDER="${POPE_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
AMBER_IMAGE_FOLDER="${AMBER_IMAGE_FOLDER:-$AMBER_ROOT/image}"

POPE_DISC_Q_FILE="${POPE_DISC_Q_FILE:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/assets/discovery_q_with_object.jsonl}"
POPE_DISC_GT_CSV="${POPE_DISC_GT_CSV:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/assets/discovery_gt.csv}"
POPE_DISC_BASELINE_JSONL="${POPE_DISC_BASELINE_JSONL:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/baseline/pred_baseline.jsonl}"
POPE_DISC_INTERVENTION_JSONL="${POPE_DISC_INTERVENTION_JSONL:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/vga/pred_vga.jsonl}"

POPE_TEST_Q_FILE="${POPE_TEST_Q_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
POPE_TEST_GT_CSV="${POPE_TEST_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
POPE_TEST_BASELINE_JSONL="${POPE_TEST_BASELINE_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_baseline.jsonl}"
POPE_TEST_INTERVENTION_JSONL="${POPE_TEST_INTERVENTION_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_vga.jsonl}"

AMBER_SPLIT_DIR="${AMBER_SPLIT_DIR:-$OUT_ROOT/assets/amber_discriminative_split}"
AMBER_DISC_Q_FILE="${AMBER_DISC_Q_FILE:-$AMBER_SPLIT_DIR/discovery/assets/discovery_q_with_object.jsonl}"
AMBER_DISC_GT_CSV="${AMBER_DISC_GT_CSV:-$AMBER_SPLIT_DIR/discovery/assets/discovery_gt.csv}"
AMBER_TEST_Q_FILE="${AMBER_TEST_Q_FILE:-$AMBER_SPLIT_DIR/test/assets/test_q_with_object.jsonl}"
AMBER_TEST_GT_CSV="${AMBER_TEST_GT_CSV:-$AMBER_SPLIT_DIR/test/assets/test_gt.csv}"
AMBER_BASELINE_JSONL="${AMBER_BASELINE_JSONL:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope/discriminative/pred_baseline.jsonl}"
AMBER_INTERVENTION_JSONL="${AMBER_INTERVENTION_JSONL:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope/discriminative/pred_vga.jsonl}"

POPE_DISC_FALLBACK_ROOT="$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial"

POPE_DISC_FEATURE_DIR="$OUT_ROOT/discovery/pope/features"
POPE_DISC_TABLE_DIR="$OUT_ROOT/discovery/pope/table"
AMBER_DISC_FEATURE_DIR="$OUT_ROOT/discovery/amber_disc/features"
AMBER_DISC_TABLE_DIR="$OUT_ROOT/discovery/amber_disc/table"
DISC_ANALYSIS_DIR="$OUT_ROOT/discovery/analysis"
DISC_CONTROLLER_DIR="$OUT_ROOT/discovery/unified_controller"

POPE_TEST_FEATURE_DIR="$OUT_ROOT/test/pope/features"
POPE_TEST_TABLE_DIR="$OUT_ROOT/test/pope/table"
POPE_TEST_APPLY_DIR="$OUT_ROOT/test/pope/apply"
AMBER_TEST_FEATURE_DIR="$OUT_ROOT/test/amber_disc/features"
AMBER_TEST_TABLE_DIR="$OUT_ROOT/test/amber_disc/table"
AMBER_TEST_APPLY_DIR="$OUT_ROOT/test/amber_disc/apply"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

if [[ ! -f "$POPE_DISC_Q_FILE" || ! -f "$POPE_DISC_GT_CSV" || ! -f "$POPE_DISC_BASELINE_JSONL" || ! -f "$POPE_DISC_INTERVENTION_JSONL" ]]; then
  FALLBACK_Q="$POPE_DISC_FALLBACK_ROOT/assets/discovery_q_with_object.jsonl"
  FALLBACK_GT="$POPE_DISC_FALLBACK_ROOT/assets/discovery_gt.csv"
  FALLBACK_BASELINE="$POPE_DISC_FALLBACK_ROOT/baseline/pred_baseline.jsonl"
  FALLBACK_INTERVENTION="$POPE_DISC_FALLBACK_ROOT/vga/pred_vga.jsonl"
  if [[ -f "$FALLBACK_Q" && -f "$FALLBACK_GT" && -f "$FALLBACK_BASELINE" && -f "$FALLBACK_INTERVENTION" ]]; then
    echo "[config] fallback to POPE adversarial discovery assets"
    POPE_DISC_Q_FILE="$FALLBACK_Q"
    POPE_DISC_GT_CSV="$FALLBACK_GT"
    POPE_DISC_BASELINE_JSONL="$FALLBACK_BASELINE"
    POPE_DISC_INTERVENTION_JSONL="$FALLBACK_INTERVENTION"
  fi
fi

if [[ ! -f "$AMBER_DISC_Q_FILE" || ! -f "$AMBER_TEST_Q_FILE" ]]; then
  echo "[0/10] prepare AMBER discriminative split assets"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/prepare_amber_discriminative_assets.py \
    --amber_root "$AMBER_ROOT" \
    --out_dir "$AMBER_SPLIT_DIR" \
    --discovery_ratio "$AMBER_DISCOVERY_RATIO" \
    --seed 42
fi

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$POPE_DISC_FEATURE_DIR/features.csv" ]]; then
  echo "[1/10] extract POPE discovery baseline semantic features"
  mkdir -p "$POPE_DISC_FEATURE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_baseline_semantic_features.py \
    --question_file "$POPE_DISC_Q_FILE" \
    --image_folder "$DISCOVERY_IMAGE_FOLDER" \
    --baseline_pred_jsonl "$POPE_DISC_BASELINE_JSONL" \
    --out_csv "$POPE_DISC_FEATURE_DIR/features.csv" \
    --out_summary_json "$POPE_DISC_FEATURE_DIR/summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$MAX_SAMPLES"
else
  echo "[1/10] reuse POPE discovery baseline semantic features"
fi

echo "[2/10] build POPE discovery pregate table"
mkdir -p "$POPE_DISC_TABLE_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
  --probe_features_csv "$POPE_DISC_FEATURE_DIR/features.csv" \
  --baseline_pred_jsonl "$POPE_DISC_BASELINE_JSONL" \
  --intervention_pred_jsonl "$POPE_DISC_INTERVENTION_JSONL" \
  --question_file "$POPE_DISC_Q_FILE" \
  --gt_csv "$POPE_DISC_GT_CSV" \
  --benchmark_name pope \
  --split_name discovery \
  --out_csv "$POPE_DISC_TABLE_DIR/table.csv" \
  --out_summary_json "$POPE_DISC_TABLE_DIR/summary.json"

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$AMBER_DISC_FEATURE_DIR/features.csv" ]]; then
  echo "[3/10] extract AMBER discovery baseline semantic features"
  mkdir -p "$AMBER_DISC_FEATURE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_baseline_semantic_features.py \
    --question_file "$AMBER_DISC_Q_FILE" \
    --image_folder "$AMBER_IMAGE_FOLDER" \
    --baseline_pred_jsonl "$AMBER_BASELINE_JSONL" \
    --out_csv "$AMBER_DISC_FEATURE_DIR/features.csv" \
    --out_summary_json "$AMBER_DISC_FEATURE_DIR/summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$MAX_SAMPLES"
else
  echo "[3/10] reuse AMBER discovery baseline semantic features"
fi

echo "[4/10] build AMBER discovery pregate table"
mkdir -p "$AMBER_DISC_TABLE_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
  --probe_features_csv "$AMBER_DISC_FEATURE_DIR/features.csv" \
  --baseline_pred_jsonl "$AMBER_BASELINE_JSONL" \
  --intervention_pred_jsonl "$AMBER_INTERVENTION_JSONL" \
  --question_file "$AMBER_DISC_Q_FILE" \
  --gt_csv "$AMBER_DISC_GT_CSV" \
  --benchmark_name amber_discriminative \
  --split_name discovery \
  --out_csv "$AMBER_DISC_TABLE_DIR/table.csv" \
  --out_summary_json "$AMBER_DISC_TABLE_DIR/summary.json"

echo "[5/10] phase-1 discovery analysis"
mkdir -p "$DISC_ANALYSIS_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/analyze_vga_pregate_harm.py \
  --table_csvs "$POPE_DISC_TABLE_DIR/table.csv" "$AMBER_DISC_TABLE_DIR/table.csv" \
  --feature_cols "$FEATURE_COLS" \
  --out_dir "$DISC_ANALYSIS_DIR"

echo "[6/10] phase-2 unified discovery controller"
mkdir -p "$DISC_CONTROLLER_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_harm_controller.py \
  --discovery_table_csvs "$POPE_DISC_TABLE_DIR/table.csv" "$AMBER_DISC_TABLE_DIR/table.csv" \
  --out_dir "$DISC_CONTROLLER_DIR" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_k "$TOP_K" \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --feature_cols "$FEATURE_COLS"

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$POPE_TEST_FEATURE_DIR/features.csv" ]]; then
  echo "[7/10] extract POPE held-out baseline semantic features"
  mkdir -p "$POPE_TEST_FEATURE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_baseline_semantic_features.py \
    --question_file "$POPE_TEST_Q_FILE" \
    --image_folder "$POPE_IMAGE_FOLDER" \
    --baseline_pred_jsonl "$POPE_TEST_BASELINE_JSONL" \
    --out_csv "$POPE_TEST_FEATURE_DIR/features.csv" \
    --out_summary_json "$POPE_TEST_FEATURE_DIR/summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$MAX_SAMPLES"
else
  echo "[7/10] reuse POPE held-out baseline semantic features"
fi

echo "[8/10] build/apply POPE held-out gate"
mkdir -p "$POPE_TEST_TABLE_DIR" "$POPE_TEST_APPLY_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
  --probe_features_csv "$POPE_TEST_FEATURE_DIR/features.csv" \
  --baseline_pred_jsonl "$POPE_TEST_BASELINE_JSONL" \
  --intervention_pred_jsonl "$POPE_TEST_INTERVENTION_JSONL" \
  --question_file "$POPE_TEST_Q_FILE" \
  --gt_csv "$POPE_TEST_GT_CSV" \
  --benchmark_name pope \
  --split_name heldout \
  --out_csv "$POPE_TEST_TABLE_DIR/table.csv" \
  --out_summary_json "$POPE_TEST_TABLE_DIR/summary.json"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_harm_controller.py \
  --table_csv "$POPE_TEST_TABLE_DIR/table.csv" \
  --policy_json "$DISC_CONTROLLER_DIR/selected_policy.json" \
  --out_dir "$POPE_TEST_APPLY_DIR"

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$AMBER_TEST_FEATURE_DIR/features.csv" ]]; then
  echo "[9/10] extract AMBER held-out baseline semantic features"
  mkdir -p "$AMBER_TEST_FEATURE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_baseline_semantic_features.py \
    --question_file "$AMBER_TEST_Q_FILE" \
    --image_folder "$AMBER_IMAGE_FOLDER" \
    --baseline_pred_jsonl "$AMBER_BASELINE_JSONL" \
    --out_csv "$AMBER_TEST_FEATURE_DIR/features.csv" \
    --out_summary_json "$AMBER_TEST_FEATURE_DIR/summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$MAX_SAMPLES"
else
  echo "[9/10] reuse AMBER held-out baseline semantic features"
fi

echo "[10/10] build/apply AMBER held-out gate"
mkdir -p "$AMBER_TEST_TABLE_DIR" "$AMBER_TEST_APPLY_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
  --probe_features_csv "$AMBER_TEST_FEATURE_DIR/features.csv" \
  --baseline_pred_jsonl "$AMBER_BASELINE_JSONL" \
  --intervention_pred_jsonl "$AMBER_INTERVENTION_JSONL" \
  --question_file "$AMBER_TEST_Q_FILE" \
  --gt_csv "$AMBER_TEST_GT_CSV" \
  --benchmark_name amber_discriminative \
  --split_name heldout \
  --out_csv "$AMBER_TEST_TABLE_DIR/table.csv" \
  --out_summary_json "$AMBER_TEST_TABLE_DIR/summary.json"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_harm_controller.py \
  --table_csv "$AMBER_TEST_TABLE_DIR/table.csv" \
  --policy_json "$DISC_CONTROLLER_DIR/selected_policy.json" \
  --out_dir "$AMBER_TEST_APPLY_DIR"

echo "[done] $DISC_ANALYSIS_DIR/summary.json"
echo "[done] $DISC_CONTROLLER_DIR/summary.json"
echo "[done] $POPE_TEST_APPLY_DIR/summary.json"
echo "[done] $AMBER_TEST_APPLY_DIR/summary.json"
