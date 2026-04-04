#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_harm_v2}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_harm_objective_sweep}"

MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K="${TOP_K:-3}"
FEATURE_COLS="${FEATURE_COLS:-frg,frg_shared_mean,frg_shared_topk,g_top5_mass,gmi,c_agg_cos,c_agg_ip,e_agg_combo,e_agg_js,late_head_vis_ratio_mean,late_head_vis_ratio_topkmean}"

POPE_DISC_TABLE="${POPE_DISC_TABLE:-$SOURCE_ROOT/discovery/pope/table/table.csv}"
AMBER_DISC_TABLE="${AMBER_DISC_TABLE:-$SOURCE_ROOT/discovery/amber_disc/table/table.csv}"
POPE_TEST_TABLE="${POPE_TEST_TABLE:-$SOURCE_ROOT/test/pope/table/table.csv}"
AMBER_TEST_TABLE="${AMBER_TEST_TABLE:-$SOURCE_ROOT/test/amber_disc/table/table.csv}"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

for path in "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE" "$POPE_TEST_TABLE" "$AMBER_TEST_TABLE"; do
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

  echo "[sweep:$name] build controller"
  mkdir -p "$controller_dir"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_harm_controller.py \
    --discovery_table_csvs "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE" \
    --out_dir "$controller_dir" \
    --min_feature_auroc "$MIN_FEATURE_AUROC" \
    --top_k "$TOP_K" \
    --feature_cols "$FEATURE_COLS" \
    --tau_objective "$objective" \
    --min_baseline_rate "$min_rate" \
    --max_baseline_rate "$max_rate" \
    --min_selected_count "$min_selected"

  echo "[sweep:$name] apply on POPE held-out"
  mkdir -p "$pope_apply_dir"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_harm_controller.py \
    --table_csv "$POPE_TEST_TABLE" \
    --policy_json "$controller_dir/selected_policy.json" \
    --out_dir "$pope_apply_dir"

  echo "[sweep:$name] apply on AMBER-disc held-out"
  mkdir -p "$amber_apply_dir"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_harm_controller.py \
    --table_csv "$AMBER_TEST_TABLE" \
    --policy_json "$controller_dir/selected_policy.json" \
    --out_dir "$amber_apply_dir"
}

run_variant "acc_default" "final_acc" "0.0" "1.0" "0"
run_variant "harm_f1_b3" "harm_f1" "0.005" "0.03" "10"
run_variant "harm_precision_b3" "harm_precision" "0.005" "0.03" "10"
run_variant "harm_recall_b3" "harm_recall" "0.005" "0.03" "10"

echo "[done] $OUT_ROOT/acc_default/test/pope/apply/summary.json"
echo "[done] $OUT_ROOT/harm_f1_b3/test/pope/apply/summary.json"
echo "[done] $OUT_ROOT/harm_precision_b3/test/pope/apply/summary.json"
echo "[done] $OUT_ROOT/harm_recall_b3/test/pope/apply/summary.json"
