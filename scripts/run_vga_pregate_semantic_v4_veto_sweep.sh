#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_harm_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_v4_veto_sweep_v1}"
DISCOVERY_ONLY="${DISCOVERY_ONLY:-true}"
INCLUDE_AMBER_DISCOVERY="${INCLUDE_AMBER_DISCOVERY:-true}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
HELP_FEATURE_COLS="${HELP_FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"
HARM_FEATURE_COLS="${HARM_FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"
TOP_K_HELP="${TOP_K_HELP:-3}"
TOP_K_HARM="${TOP_K_HARM:-3}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.5,1.0,1.5,2.0}"
TAU_QUANTILES="${TAU_QUANTILES:-0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-balanced_utility}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"
FIT_EPOCHS="${FIT_EPOCHS:-300}"
FIT_LR="${FIT_LR:-0.05}"
FIT_L2="${FIT_L2:-0.001}"

run_variant() {
  local name="$1"
  local min_method_rate="$2"
  local max_method_rate="$3"
  local variant_root="$OUT_ROOT/$name"
  echo "[v4-veto:$name] method_rate in [$min_method_rate, $max_method_rate]"
  CAL_ROOT="$CAL_ROOT" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  OUT_ROOT="$variant_root" \
  DISCOVERY_ONLY="$DISCOVERY_ONLY" \
  INCLUDE_AMBER_DISCOVERY="$INCLUDE_AMBER_DISCOVERY" \
  PYTHON_BIN="$PYTHON_BIN" \
  MIN_FEATURE_AUROC="$MIN_FEATURE_AUROC" \
  HELP_FEATURE_COLS="$HELP_FEATURE_COLS" \
  HARM_FEATURE_COLS="$HARM_FEATURE_COLS" \
  TOP_K_HELP="$TOP_K_HELP" \
  TOP_K_HARM="$TOP_K_HARM" \
  LAMBDA_VALUES="$LAMBDA_VALUES" \
  TAU_QUANTILES="$TAU_QUANTILES" \
  TAU_OBJECTIVE="$TAU_OBJECTIVE" \
  MIN_METHOD_RATE="$min_method_rate" \
  MAX_METHOD_RATE="$max_method_rate" \
  MIN_SELECTED_COUNT="$MIN_SELECTED_COUNT" \
  FIT_EPOCHS="$FIT_EPOCHS" \
  FIT_LR="$FIT_LR" \
  FIT_L2="$FIT_L2" \
  bash scripts/run_vga_pregate_semantic_v4.sh
}

run_variant "veto_b2_5" "0.95" "0.98"
run_variant "veto_b5_10" "0.90" "0.95"
run_variant "veto_b10_20" "0.80" "0.90"

if [[ "$DISCOVERY_ONLY" == "true" ]]; then
  echo "[done] $OUT_ROOT/veto_b2_5/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/veto_b5_10/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/veto_b10_20/discovery/unified_controller/summary.json"
else
  echo "[done] $OUT_ROOT/veto_b2_5/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/veto_b5_10/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/veto_b10_20/test/pope/apply/summary.json"
fi
