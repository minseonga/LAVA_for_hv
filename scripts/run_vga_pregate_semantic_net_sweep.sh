#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_harm_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_net_sweep_v1}"
DISCOVERY_ONLY="${DISCOVERY_ONLY:-true}"

MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
HARM_FEATURE_COLS="${HARM_FEATURE_COLS:-base_target_gap_content_min,base_lp_content_mean,base_conflict_lp_minus_entropy}"
HELP_FEATURE_COLS="${HELP_FEATURE_COLS:-base_target_argmax_content_mean}"
TOP_K_HARM="${TOP_K_HARM:-3}"
TOP_K_HELP="${TOP_K_HELP:-1}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.5,1.0,1.5,2.0}"
MIN_SELECTED_PER_SOURCE="${MIN_SELECTED_PER_SOURCE:-0}"

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
  local objective="$2"
  local min_rate="$3"
  local max_rate="$4"
  local min_selected="$5"

  local variant_root="$OUT_ROOT/$name"
  local controller_dir="$variant_root/discovery/unified_controller"
  local pope_apply_dir="$variant_root/test/pope/apply"
  local amber_apply_dir="$variant_root/test/amber_disc/apply"

  echo "[semantic-net:$name] build controller"
  mkdir -p "$controller_dir"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_net_controller.py \
    --discovery_table_csvs "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE" \
    --out_dir "$controller_dir" \
    --harm_feature_cols "$HARM_FEATURE_COLS" \
    --help_feature_cols "$HELP_FEATURE_COLS" \
    --min_feature_auroc "$MIN_FEATURE_AUROC" \
    --top_k_harm "$TOP_K_HARM" \
    --top_k_help "$TOP_K_HELP" \
    --lambda_values "$LAMBDA_VALUES" \
    --tau_objective "$objective" \
    --min_baseline_rate "$min_rate" \
    --max_baseline_rate "$max_rate" \
    --min_selected_count "$min_selected" \
    --min_selected_per_source "$MIN_SELECTED_PER_SOURCE"

  if [[ "$DISCOVERY_ONLY" != "true" ]]; then
    if [[ ! -f "$POPE_TEST_TABLE" || ! -f "$AMBER_TEST_TABLE" ]]; then
      echo "[error] held-out tables missing under SOURCE_ROOT; run semantic harm study first" >&2
      exit 1
    fi
    echo "[semantic-net:$name] apply on POPE held-out"
    mkdir -p "$pope_apply_dir"
    PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_net_controller.py \
      --table_csv "$POPE_TEST_TABLE" \
      --policy_json "$controller_dir/selected_policy.json" \
      --out_dir "$pope_apply_dir"

    echo "[semantic-net:$name] apply on AMBER-disc held-out"
    mkdir -p "$amber_apply_dir"
    PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_net_controller.py \
      --table_csv "$AMBER_TEST_TABLE" \
      --policy_json "$controller_dir/selected_policy.json" \
      --out_dir "$amber_apply_dir"
  fi
}

run_variant "balanced_u_b1_s30" "balanced_utility" "0.01" "0.05" "30"
run_variant "balanced_u_b2_s30" "balanced_utility" "0.02" "0.05" "30"
run_variant "balanced_u_b5_s50" "balanced_utility" "0.05" "0.10" "50"
run_variant "harm_f1_b2_s30" "harm_f1" "0.02" "0.05" "30"

if [[ "$DISCOVERY_ONLY" == "true" ]]; then
  echo "[done] $OUT_ROOT/balanced_u_b1_s30/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/balanced_u_b2_s30/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/balanced_u_b5_s50/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/harm_f1_b2_s30/discovery/unified_controller/summary.json"
else
  echo "[done] $OUT_ROOT/balanced_u_b1_s30/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/balanced_u_b2_s30/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/balanced_u_b5_s50/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/harm_f1_b2_s30/test/pope/apply/summary.json"
fi
