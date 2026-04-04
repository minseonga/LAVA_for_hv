#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_harm_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_twostage_apply_v2}"
DISCOVERY_ONLY="${DISCOVERY_ONLY:-true}"
INCLUDE_AMBER_DISCOVERY="${INCLUDE_AMBER_DISCOVERY:-true}"

POPE_DISC_TABLE="${POPE_DISC_TABLE:-$SOURCE_ROOT/discovery/pope/table/table.csv}"
AMBER_DISC_TABLE="${AMBER_DISC_TABLE:-$SOURCE_ROOT/discovery/amber_disc/table/table.csv}"
POPE_TEST_TABLE="${POPE_TEST_TABLE:-$SOURCE_ROOT/test/pope/table/table.csv}"
AMBER_TEST_TABLE="${AMBER_TEST_TABLE:-$SOURCE_ROOT/test/amber_disc/table/table.csv}"

P1_FEATURE_COLS="${P1_FEATURE_COLS:-base_target_gap_content_min,base_lp_content_mean,base_conflict_lp_minus_entropy,base_target_argmax_content_mean,base_entropy_content_mean}"
P2_FEATURE_COLS="${P2_FEATURE_COLS:-base_target_gap_content_min,base_lp_content_mean,base_conflict_lp_minus_entropy,base_target_argmax_content_mean,base_entropy_content_mean}"
MIN_FEATURE_AUROC_P1="${MIN_FEATURE_AUROC_P1:-0.55}"
MIN_FEATURE_AUROC_P2="${MIN_FEATURE_AUROC_P2:-0.52}"
TOP_K_P1="${TOP_K_P1:-5}"
TOP_K_P2="${TOP_K_P2:-5}"
TAU_QUANTILES_P1="${TAU_QUANTILES_P1:-0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.95,0.97,0.98,0.99}"
TAU_QUANTILES_P2="${TAU_QUANTILES_P2:-0.20,0.30,0.40,0.50,0.60,0.70,0.80}"
P1_OBJECTIVE="${P1_OBJECTIVE:-subset_purity}"
OBJECTIVE="${OBJECTIVE:-balanced_utility}"
P1_LAMBDA_HARM="${P1_LAMBDA_HARM:-1.0}"
MIN_P1_APPLY_RATE="${MIN_P1_APPLY_RATE:-0.05}"
MAX_P1_APPLY_RATE="${MAX_P1_APPLY_RATE:-0.90}"
MIN_P1_SELECTED_COUNT="${MIN_P1_SELECTED_COUNT:-50}"
MIN_P1_SELECTED_PER_SOURCE="${MIN_P1_SELECTED_PER_SOURCE:-0}"
MIN_METHOD_RATE="${MIN_METHOD_RATE:-0.0}"
MAX_METHOD_RATE="${MAX_METHOD_RATE:-1.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"
MIN_SELECTED_PER_SOURCE="${MIN_SELECTED_PER_SOURCE:-0}"
P1_PAIR_MODE="${P1_PAIR_MODE:-none}"
P2_PAIR_MODE="${P2_PAIR_MODE:-pairwise}"
FIT_EPOCHS="${FIT_EPOCHS:-300}"
FIT_LR="${FIT_LR:-0.05}"
FIT_L2="${FIT_L2:-0.001}"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

for path in "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE"; do
  if [[ "$path" == "$AMBER_DISC_TABLE" && "$INCLUDE_AMBER_DISCOVERY" != "true" ]]; then
    continue
  fi
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required table: $path" >&2
    exit 1
  fi
done

DISCOVERY_TABLES=("$POPE_DISC_TABLE")
if [[ "$INCLUDE_AMBER_DISCOVERY" == "true" ]]; then
  DISCOVERY_TABLES+=("$AMBER_DISC_TABLE")
fi

echo "[1/3] build semantic two-stage apply v2 controller"
DISC_CONTROLLER_DIR="$OUT_ROOT/discovery/unified_controller"
mkdir -p "$DISC_CONTROLLER_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_two_stage_apply_v2_controller.py \
  --discovery_table_csvs "${DISCOVERY_TABLES[@]}" \
  --out_dir "$DISC_CONTROLLER_DIR" \
  --p1_feature_cols "$P1_FEATURE_COLS" \
  --p2_feature_cols "$P2_FEATURE_COLS" \
  --min_feature_auroc_p1 "$MIN_FEATURE_AUROC_P1" \
  --min_feature_auroc_p2 "$MIN_FEATURE_AUROC_P2" \
  --top_k_p1 "$TOP_K_P1" \
  --top_k_p2 "$TOP_K_P2" \
  --tau_quantiles_p1 "$TAU_QUANTILES_P1" \
  --tau_quantiles_p2 "$TAU_QUANTILES_P2" \
  --p1_objective "$P1_OBJECTIVE" \
  --objective "$OBJECTIVE" \
  --p1_lambda_harm "$P1_LAMBDA_HARM" \
  --min_p1_apply_rate "$MIN_P1_APPLY_RATE" \
  --max_p1_apply_rate "$MAX_P1_APPLY_RATE" \
  --min_p1_selected_count "$MIN_P1_SELECTED_COUNT" \
  --min_p1_selected_per_source "$MIN_P1_SELECTED_PER_SOURCE" \
  --min_method_rate "$MIN_METHOD_RATE" \
  --max_method_rate "$MAX_METHOD_RATE" \
  --min_selected_count "$MIN_SELECTED_COUNT" \
  --min_selected_per_source "$MIN_SELECTED_PER_SOURCE" \
  --p1_pair_mode "$P1_PAIR_MODE" \
  --p2_pair_mode "$P2_PAIR_MODE" \
  --fit_epochs "$FIT_EPOCHS" \
  --fit_lr "$FIT_LR" \
  --fit_l2 "$FIT_L2"

if [[ "$DISCOVERY_ONLY" == "true" ]]; then
  echo "[done] $DISC_CONTROLLER_DIR/summary.json"
  exit 0
fi

if [[ ! -f "$POPE_TEST_TABLE" || ! -f "$AMBER_TEST_TABLE" ]]; then
  echo "[error] held-out tables missing under SOURCE_ROOT; run semantic harm study first" >&2
  exit 1
fi

echo "[2/3] apply on POPE held-out"
POPE_APPLY_DIR="$OUT_ROOT/test/pope/apply"
mkdir -p "$POPE_APPLY_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_two_stage_apply_controller.py \
  --table_csv "$POPE_TEST_TABLE" \
  --policy_json "$DISC_CONTROLLER_DIR/selected_policy.json" \
  --out_dir "$POPE_APPLY_DIR"

echo "[3/3] apply on AMBER-disc held-out"
AMBER_APPLY_DIR="$OUT_ROOT/test/amber_disc/apply"
mkdir -p "$AMBER_APPLY_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_two_stage_apply_controller.py \
  --table_csv "$AMBER_TEST_TABLE" \
  --policy_json "$DISC_CONTROLLER_DIR/selected_policy.json" \
  --out_dir "$AMBER_APPLY_DIR"

echo "[done] $POPE_APPLY_DIR/summary.json"
echo "[done] $AMBER_APPLY_DIR/summary.json"
