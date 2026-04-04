#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_harm_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_v3}"
DISCOVERY_ONLY="${DISCOVERY_ONLY:-true}"
INCLUDE_AMBER_DISCOVERY="${INCLUDE_AMBER_DISCOVERY:-true}"

MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
HELP_FEATURE_COLS="${HELP_FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"
HARM_FEATURE_COLS="${HARM_FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"
TOP_K_HELP="${TOP_K_HELP:-3}"
TOP_K_HARM="${TOP_K_HARM:-3}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.5,1.0,1.5,2.0}"
TAU_QUANTILES="${TAU_QUANTILES:-0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-balanced_utility}"
MIN_METHOD_RATE="${MIN_METHOD_RATE:-0.0}"
MAX_METHOD_RATE="${MAX_METHOD_RATE:-1.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"

POPE_DISC_TABLE="${POPE_DISC_TABLE:-$SOURCE_ROOT/discovery/pope/table/table.csv}"
AMBER_DISC_TABLE="${AMBER_DISC_TABLE:-$SOURCE_ROOT/discovery/amber_disc/table/table.csv}"
POPE_TEST_TABLE="${POPE_TEST_TABLE:-$SOURCE_ROOT/test/pope/table/table.csv}"
AMBER_TEST_TABLE="${AMBER_TEST_TABLE:-$SOURCE_ROOT/test/amber_disc/table/table.csv}"

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

echo "[1/3] build unified pre-gating v3 controller"
mkdir -p "$OUT_ROOT/discovery/unified_controller"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_v3_controller.py \
  --discovery_table_csvs "${DISCOVERY_TABLES[@]}" \
  --out_dir "$OUT_ROOT/discovery/unified_controller" \
  --help_feature_cols "$HELP_FEATURE_COLS" \
  --harm_feature_cols "$HARM_FEATURE_COLS" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_k_help "$TOP_K_HELP" \
  --top_k_harm "$TOP_K_HARM" \
  --lambda_values "$LAMBDA_VALUES" \
  --tau_quantiles "$TAU_QUANTILES" \
  --tau_objective "$TAU_OBJECTIVE" \
  --min_method_rate "$MIN_METHOD_RATE" \
  --max_method_rate "$MAX_METHOD_RATE" \
  --min_selected_count "$MIN_SELECTED_COUNT"

if [[ "$DISCOVERY_ONLY" == "true" ]]; then
  echo "[done] $OUT_ROOT/discovery/unified_controller/summary.json"
  exit 0
fi

if [[ ! -f "$POPE_TEST_TABLE" || ! -f "$AMBER_TEST_TABLE" ]]; then
  echo "[error] held-out tables missing under SOURCE_ROOT; run semantic harm study first" >&2
  exit 1
fi

echo "[2/3] apply on POPE held-out"
mkdir -p "$OUT_ROOT/test/pope/apply"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_v3_controller.py \
  --table_csv "$POPE_TEST_TABLE" \
  --policy_json "$OUT_ROOT/discovery/unified_controller/selected_policy.json" \
  --out_dir "$OUT_ROOT/test/pope/apply"

echo "[3/3] apply on AMBER-disc held-out"
mkdir -p "$OUT_ROOT/test/amber_disc/apply"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_v3_controller.py \
  --table_csv "$AMBER_TEST_TABLE" \
  --policy_json "$OUT_ROOT/discovery/unified_controller/selected_policy.json" \
  --out_dir "$OUT_ROOT/test/amber_disc/apply"

echo "[done] $OUT_ROOT/test/pope/apply/summary.json"
echo "[done] $OUT_ROOT/test/amber_disc/apply/summary.json"
