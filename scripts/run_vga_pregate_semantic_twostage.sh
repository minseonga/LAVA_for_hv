#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_harm_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_twostage_v1}"
DISCOVERY_ONLY="${DISCOVERY_ONLY:-true}"

POPE_DISC_TABLE="${POPE_DISC_TABLE:-$SOURCE_ROOT/discovery/pope/table/table.csv}"
AMBER_DISC_TABLE="${AMBER_DISC_TABLE:-$SOURCE_ROOT/discovery/amber_disc/table/table.csv}"
POPE_TEST_TABLE="${POPE_TEST_TABLE:-$SOURCE_ROOT/test/pope/table/table.csv}"
AMBER_TEST_TABLE="${AMBER_TEST_TABLE:-$SOURCE_ROOT/test/amber_disc/table/table.csv}"

P1_FEATURE_COLS="${P1_FEATURE_COLS:-base_target_gap_content_min,base_lp_content_mean,base_conflict_lp_minus_entropy,base_target_argmax_content_mean,base_entropy_content_mean}"
P2_FEATURE_COLS="${P2_FEATURE_COLS:-base_target_argmax_content_mean,base_lp_content_mean,base_target_gap_content_min,base_conflict_lp_minus_entropy,base_entropy_content_mean}"
MIN_FEATURE_AUROC_P1="${MIN_FEATURE_AUROC_P1:-0.55}"
MIN_FEATURE_AUROC_P2="${MIN_FEATURE_AUROC_P2:-0.52}"
TOP_K_P1="${TOP_K_P1:-5}"
TOP_K_P2="${TOP_K_P2:-3}"
TAU_QUANTILES_P1="${TAU_QUANTILES_P1:-0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.95,0.97,0.98,0.99}"
TAU_QUANTILES_P2="${TAU_QUANTILES_P2:-0.20,0.30,0.40,0.50,0.60,0.70,0.80}"
OBJECTIVE="${OBJECTIVE:-balanced_utility}"
MIN_SENSITIVE_RATE="${MIN_SENSITIVE_RATE:-0.0}"
MAX_SENSITIVE_RATE="${MAX_SENSITIVE_RATE:-1.0}"
MIN_SENSITIVE_COUNT="${MIN_SENSITIVE_COUNT:-0}"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

for path in "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE"; do
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required table: $path" >&2
    exit 1
  fi
done

echo "[1/3] build two-stage discovery controller"
DISC_CONTROLLER_DIR="$OUT_ROOT/discovery/unified_controller"
mkdir -p "$DISC_CONTROLLER_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_two_stage_controller.py \
  --discovery_table_csvs "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE" \
  --out_dir "$DISC_CONTROLLER_DIR" \
  --p1_feature_cols "$P1_FEATURE_COLS" \
  --p2_feature_cols "$P2_FEATURE_COLS" \
  --min_feature_auroc_p1 "$MIN_FEATURE_AUROC_P1" \
  --min_feature_auroc_p2 "$MIN_FEATURE_AUROC_P2" \
  --top_k_p1 "$TOP_K_P1" \
  --top_k_p2 "$TOP_K_P2" \
  --tau_quantiles_p1 "$TAU_QUANTILES_P1" \
  --tau_quantiles_p2 "$TAU_QUANTILES_P2" \
  --objective "$OBJECTIVE" \
  --min_sensitive_rate "$MIN_SENSITIVE_RATE" \
  --max_sensitive_rate "$MAX_SENSITIVE_RATE" \
  --min_sensitive_count "$MIN_SENSITIVE_COUNT"

if [[ "$DISCOVERY_ONLY" != "true" ]]; then
  if [[ ! -f "$POPE_TEST_TABLE" || ! -f "$AMBER_TEST_TABLE" ]]; then
    echo "[error] held-out tables missing under SOURCE_ROOT; run semantic harm study first" >&2
    exit 1
  fi

  echo "[2/3] apply on POPE held-out"
  POPE_APPLY_DIR="$OUT_ROOT/test/pope/apply"
  mkdir -p "$POPE_APPLY_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_two_stage_controller.py \
    --table_csv "$POPE_TEST_TABLE" \
    --policy_json "$DISC_CONTROLLER_DIR/selected_policy.json" \
    --out_dir "$POPE_APPLY_DIR"

  echo "[3/3] apply on AMBER-disc held-out"
  AMBER_APPLY_DIR="$OUT_ROOT/test/amber_disc/apply"
  mkdir -p "$AMBER_APPLY_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_two_stage_controller.py \
    --table_csv "$AMBER_TEST_TABLE" \
    --policy_json "$DISC_CONTROLLER_DIR/selected_policy.json" \
    --out_dir "$AMBER_APPLY_DIR"

  echo "[done] $POPE_APPLY_DIR/summary.json"
  echo "[done] $AMBER_APPLY_DIR/summary.json"
else
  echo "[done] $DISC_CONTROLLER_DIR/summary.json"
fi
