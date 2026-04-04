#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"

SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_harm_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_v3_budget_sweep_v1}"
DISCOVERY_ONLY="${DISCOVERY_ONLY:-true}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
HELP_FEATURE_COLS="${HELP_FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"
HARM_FEATURE_COLS="${HARM_FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"
TOP_K_HELP="${TOP_K_HELP:-3}"
TOP_K_HARM="${TOP_K_HARM:-3}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.5,1.0,1.5,2.0}"
TAU_QUANTILES="${TAU_QUANTILES:-0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-balanced_utility}"
MIN_METHOD_RATE="${MIN_METHOD_RATE:-0.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"

POPE_DISC_TABLE="${POPE_DISC_TABLE:-$SOURCE_ROOT/discovery/pope/table/table.csv}"
AMBER_DISC_TABLE="${AMBER_DISC_TABLE:-$SOURCE_ROOT/discovery/amber_disc/table/table.csv}"
POPE_TEST_TABLE="${POPE_TEST_TABLE:-$SOURCE_ROOT/test/pope/table/table.csv}"
AMBER_TEST_TABLE="${AMBER_TEST_TABLE:-$SOURCE_ROOT/test/amber_disc/table/table.csv}"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

for path in "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE"; do
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required table: $path" >&2
    exit 1
  fi
done

run_variant() {
  local name="$1"
  local max_method_rate="$2"

  local variant_root="$OUT_ROOT/$name"
  echo "[v3-budget:$name] max_method_rate=$max_method_rate"
  CAL_ROOT="$CAL_ROOT" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  OUT_ROOT="$variant_root" \
  DISCOVERY_ONLY="$DISCOVERY_ONLY" \
  PYTHON_BIN="$PYTHON_BIN" \
  MIN_FEATURE_AUROC="$MIN_FEATURE_AUROC" \
  HELP_FEATURE_COLS="$HELP_FEATURE_COLS" \
  HARM_FEATURE_COLS="$HARM_FEATURE_COLS" \
  TOP_K_HELP="$TOP_K_HELP" \
  TOP_K_HARM="$TOP_K_HARM" \
  LAMBDA_VALUES="$LAMBDA_VALUES" \
  TAU_QUANTILES="$TAU_QUANTILES" \
  TAU_OBJECTIVE="$TAU_OBJECTIVE" \
  MIN_METHOD_RATE="$MIN_METHOD_RATE" \
  MAX_METHOD_RATE="$max_method_rate" \
  MIN_SELECTED_COUNT="$MIN_SELECTED_COUNT" \
  POPE_DISC_TABLE="$POPE_DISC_TABLE" \
  AMBER_DISC_TABLE="$AMBER_DISC_TABLE" \
  POPE_TEST_TABLE="$POPE_TEST_TABLE" \
  AMBER_TEST_TABLE="$AMBER_TEST_TABLE" \
  bash scripts/run_vga_pregate_semantic_v3.sh
}

run_variant "m40" "0.40"
run_variant "m60" "0.60"
run_variant "m80" "0.80"
run_variant "m95" "0.95"

if [[ "$DISCOVERY_ONLY" == "true" ]]; then
  echo "[done] $OUT_ROOT/m40/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/m60/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/m80/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/m95/discovery/unified_controller/summary.json"
else
  echo "[done] $OUT_ROOT/m40/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/m60/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/m80/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/m95/test/pope/apply/summary.json"
fi
