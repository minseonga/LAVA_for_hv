#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REFERENCE_ROOT="${REFERENCE_ROOT:-$ROOT_DIR/experiments/paper_main_b_c_v1_full}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/experiments/aligned_cheap_proxy_from_reference_vga}"

DISCOVERY_GT_CSV="${DISCOVERY_GT_CSV:-$ROOT_DIR/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
TEST_GT_CSV="${TEST_GT_CSV:-$ROOT_DIR/experiments/pope_full_9000/pope_9000_gt.csv}"

DISCOVERY_BASELINE_PRED_JSONL="${DISCOVERY_BASELINE_PRED_JSONL:-$REFERENCE_ROOT/discovery_stageb/pred_baseline.jsonl}"
DISCOVERY_VGA_PRED_JSONL="${DISCOVERY_VGA_PRED_JSONL:-$REFERENCE_ROOT/discovery_stageb/pred_vga.jsonl}"
TEST_BASELINE_PRED_JSONL="${TEST_BASELINE_PRED_JSONL:-$REFERENCE_ROOT/test_stageb/pred_baseline.jsonl}"
TEST_VGA_PRED_JSONL="${TEST_VGA_PRED_JSONL:-$REFERENCE_ROOT/test_stageb/pred_vga.jsonl}"

DISCOVERY_FEATURES_CSV="${DISCOVERY_FEATURES_CSV:-$REFERENCE_ROOT/discovery/cheap_online_features.csv}"
TEST_FEATURES_CSV="${TEST_FEATURES_CSV:-$REFERENCE_ROOT/test/cheap_online_features.csv}"

DISCOVERY_TAX_DIR="${DISCOVERY_TAX_DIR:-$OUT_ROOT/discovery/taxonomy_fresh}"
TEST_TAX_DIR="${TEST_TAX_DIR:-$OUT_ROOT/test/taxonomy_fresh}"

DISCOVERY_CAL_DIR="${DISCOVERY_CAL_DIR:-$OUT_ROOT/discovery/calibration_actual}"
TEST_EVAL_DIR="${TEST_EVAL_DIR:-$OUT_ROOT/test/fixed_eval_actual}"

FEATURE_COLS="${FEATURE_COLS:-cheap_lp_content_mean,cheap_lp_content_std,cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_entropy_content_mean,cheap_entropy_content_std,cheap_margin_content_mean,cheap_margin_content_std,cheap_margin_content_min,cheap_target_gap_content_mean,cheap_target_gap_content_std,cheap_target_gap_content_min,cheap_target_argmax_content_mean,cheap_conflict_lp_minus_entropy,cheap_conflict_gap_minus_entropy,cheap_content_fraction}"
PAIR_FEATURE_TOPN="${PAIR_FEATURE_TOPN:-6}"
MAX_RESCUE_RATE="${MAX_RESCUE_RATE:-0.03}"
TARGET_LABEL="${TARGET_LABEL:-actual_rescue}"

mkdir -p "$OUT_ROOT"

echo "[1/4] build fresh discovery taxonomy from canonical reference preds"
PYTHONPATH="$ROOT_DIR" \
python scripts/build_vga_failure_taxonomy.py \
  --gt_csv "$DISCOVERY_GT_CSV" \
  --baseline_pred_jsonl "$DISCOVERY_BASELINE_PRED_JSONL" \
  --vga_pred_jsonl "$DISCOVERY_VGA_PRED_JSONL" \
  --baseline_pred_text_key text \
  --vga_pred_text_key output \
  --out_dir "$DISCOVERY_TAX_DIR"

echo "[2/4] calibrate cheap proxy on discovery"
PYTHONPATH="$ROOT_DIR" \
python scripts/run_decode_time_proxy_policy.py calibrate \
  --features_csv "$DISCOVERY_FEATURES_CSV" \
  --reference_decisions_csv "$DISCOVERY_TAX_DIR/per_case_compare.csv" \
  --feature_cols "$FEATURE_COLS" \
  --target_label "$TARGET_LABEL" \
  --pair_feature_topn "$PAIR_FEATURE_TOPN" \
  --max_rescue_rate "$MAX_RESCUE_RATE" \
  --out_dir "$DISCOVERY_CAL_DIR"

echo "[3/4] build fresh held-out taxonomy from canonical reference preds"
PYTHONPATH="$ROOT_DIR" \
python scripts/build_vga_failure_taxonomy.py \
  --gt_csv "$TEST_GT_CSV" \
  --baseline_pred_jsonl "$TEST_BASELINE_PRED_JSONL" \
  --vga_pred_jsonl "$TEST_VGA_PRED_JSONL" \
  --baseline_pred_text_key text \
  --vga_pred_text_key output \
  --out_dir "$TEST_TAX_DIR"

echo "[4/4] apply cheap proxy on held-out"
PYTHONPATH="$ROOT_DIR" \
python scripts/run_decode_time_proxy_policy.py apply \
  --features_csv "$TEST_FEATURES_CSV" \
  --reference_decisions_csv "$TEST_TAX_DIR/per_case_compare.csv" \
  --policy_json "$DISCOVERY_CAL_DIR/selected_policy.json" \
  --target_label "$TARGET_LABEL" \
  --out_dir "$TEST_EVAL_DIR"

echo "[done] discovery calibration -> $DISCOVERY_CAL_DIR"
echo "[done] held-out fixed eval -> $TEST_EVAL_DIR"
