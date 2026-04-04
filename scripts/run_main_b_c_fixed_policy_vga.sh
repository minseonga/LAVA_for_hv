#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

DISCOVERY_IMAGE_FOLDER="${DISCOVERY_IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
DISCOVERY_QUESTION_FILE="${DISCOVERY_QUESTION_FILE:-$ROOT_DIR/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
DISCOVERY_INTERVENTION_PRED_JSONL="${DISCOVERY_INTERVENTION_PRED_JSONL:-$ROOT_DIR/experiments/pope_discovery/tau_c_calibration_adversarial/vga/pred_vga.jsonl}"
DISCOVERY_SCORES_CSV="${DISCOVERY_SCORES_CSV:-$ROOT_DIR/experiments/pope_discovery/stage_b_signal_validation_vga/sample_scores.csv}"
DISCOVERY_TAXONOMY_CSV="${DISCOVERY_TAXONOMY_CSV:-$ROOT_DIR/experiments/pope_discovery/tau_c_calibration_adversarial/taxonomy/per_case_compare.csv}"

TEST_IMAGE_FOLDER="${TEST_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
TEST_QUESTION_FILE="${TEST_QUESTION_FILE:-$ROOT_DIR/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
TEST_INTERVENTION_PRED_JSONL="${TEST_INTERVENTION_PRED_JSONL:-$ROOT_DIR/experiments/pope_full_9000/stage_b_signal_validation_vga/pred_vga.jsonl}"
TEST_SCORES_CSV="${TEST_SCORES_CSV:-$ROOT_DIR/experiments/pope_full_9000/stage_b_signal_validation_vga/sample_scores.csv}"
TEST_TAXONOMY_CSV="${TEST_TAXONOMY_CSV:-$ROOT_DIR/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv}"

OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/experiments/main_b_c_fixed_policy_vga}"
DISCOVERY_OUT_DIR="${DISCOVERY_OUT_DIR:-$OUT_ROOT/discovery}"
TEST_OUT_DIR="${TEST_OUT_DIR:-$OUT_ROOT/test}"
DISCOVERY_CHEAP_CSV="${DISCOVERY_CHEAP_CSV:-$DISCOVERY_OUT_DIR/cheap_online_features.csv}"
TEST_CHEAP_CSV="${TEST_CHEAP_CSV:-$TEST_OUT_DIR/cheap_online_features.csv}"

SUBSET_PERCENTS="${SUBSET_PERCENTS:-1,2,5}"
FEATURE_COLS="${FEATURE_COLS:-cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy}"
MAX_RESCUE_RATE="${MAX_RESCUE_RATE:-0.03}"
REUSE_CHEAP="${REUSE_CHEAP:-true}"
LOG_EVERY="${LOG_EVERY:-25}"

mkdir -p "$DISCOVERY_OUT_DIR" "$TEST_OUT_DIR"

echo "[1/4] extract cheap online features on discovery"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT_DIR" \
python scripts/extract_c_stage_cheap_online_features.py \
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

echo "[2/4] calibrate fixed B+C policy on discovery"
PYTHONPATH="$ROOT_DIR" \
python scripts/run_b_c_fixed_policy.py calibrate \
  --scores_csv "$DISCOVERY_SCORES_CSV" \
  --taxonomy_csv "$DISCOVERY_TAXONOMY_CSV" \
  --features_csv "$DISCOVERY_CHEAP_CSV" \
  --subset_percents "$SUBSET_PERCENTS" \
  --feature_cols "$FEATURE_COLS" \
  --max_rescue_rate "$MAX_RESCUE_RATE" \
  --out_dir "$DISCOVERY_OUT_DIR/calibration"

echo "[3/4] extract cheap online features on held-out test"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT_DIR" \
python scripts/extract_c_stage_cheap_online_features.py \
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

echo "[4/4] apply fixed B+C policy on held-out test"
PYTHONPATH="$ROOT_DIR" \
python scripts/run_b_c_fixed_policy.py apply \
  --scores_csv "$TEST_SCORES_CSV" \
  --taxonomy_csv "$TEST_TAXONOMY_CSV" \
  --features_csv "$TEST_CHEAP_CSV" \
  --policy_json "$DISCOVERY_OUT_DIR/calibration/selected_policy.json" \
  --out_dir "$TEST_OUT_DIR/fixed_eval"

echo "[done] discovery calibration -> $DISCOVERY_OUT_DIR/calibration"
echo "[done] held-out fixed eval -> $TEST_OUT_DIR/fixed_eval"
