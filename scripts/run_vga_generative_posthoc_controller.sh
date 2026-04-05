#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

SRC_ROOT="${SRC_ROOT:-$CAL_ROOT/experiments/vga_generative_coverage_probe_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_generative_posthoc_controller_v1}"

TABLE_CSV="${TABLE_CSV:-$SRC_ROOT/vga_claim_aware_table.csv}"
FEATURE_COLS="${FEATURE_COLS:-probe_tail_tokens_after_last_mention,probe_last_mention_pos_frac,probe_entropy_tail_minus_head_real,probe_n_content_tokens,probe_entropy_tail_mean_real,probe_lp_tail_mean_real,probe_gap_tail_mean_real,probe_object_diversity,probe_mention_diversity}"
TARGET_SPEC="${TARGET_SPEC:-delta_supported_recall:lt:0}"
TOP_K_VALUES="${TOP_K_VALUES:-1,2,3,4,5}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-final_claim_utility}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
MIN_BASELINE_RATE="${MIN_BASELINE_RATE:-0.0}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"

CONTROLLER_DIR="$OUT_ROOT/discovery/controller"
APPLY_DIR="$OUT_ROOT/discovery/apply"

mkdir -p "$CONTROLLER_DIR" "$APPLY_DIR"

if [[ ! -f "$TABLE_CSV" ]]; then
  echo "[error] missing claim-aware table: $TABLE_CSV" >&2
  exit 1
fi

cd "$CAL_ROOT"

echo "[1/2] build generative posthoc controller"
"$CAL_PYTHON_BIN" scripts/build_generative_posthoc_controller.py \
  --discovery_table_csvs "$TABLE_CSV" \
  --out_dir "$CONTROLLER_DIR" \
  --target_spec "$TARGET_SPEC" \
  --feature_cols "$FEATURE_COLS" \
  --top_k_values "$TOP_K_VALUES" \
  --tau_objective "$TAU_OBJECTIVE" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --min_baseline_rate "$MIN_BASELINE_RATE" \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --min_selected_count "$MIN_SELECTED_COUNT"

echo "[2/2] apply generative posthoc controller"
"$CAL_PYTHON_BIN" scripts/apply_generative_posthoc_controller.py \
  --table_csv "$TABLE_CSV" \
  --policy_json "$CONTROLLER_DIR/selected_policy.json" \
  --out_dir "$APPLY_DIR"

echo "[done] $APPLY_DIR/summary.json"
