#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

SRC_ROOT="${SRC_ROOT:-$CAL_ROOT/experiments/vga_generative_coverage_probe_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_generative_reference_b_stage_v1}"

TABLE_CSV="${TABLE_CSV:-$SRC_ROOT/vga_claim_aware_table.csv}"
TARGET_SPEC="${TARGET_SPEC:-delta_supported_recall:lt:0}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-final_claim_utility}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K_VALUES="${TOP_K_VALUES:-1,2,3,4,5,6}"
TAU_QUANTILES="${TAU_QUANTILES:-0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99}"
MIN_BASELINE_RATE="${MIN_BASELINE_RATE:-0.0}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"

REFERENCE_B_FEATURE_COLS="${REFERENCE_B_FEATURE_COLS:-probe_tail_tokens_after_last_mention,probe_last_mention_pos_frac,probe_entropy_tail_minus_head_real,probe_n_content_tokens,probe_n_mentions_total,probe_n_object_mentions,probe_n_noun_phrases,probe_n_attribute_phrases,probe_n_relation_phrases,probe_n_count_phrases,probe_object_diversity,probe_mention_diversity,probe_first_half_object_mentions,probe_second_half_object_mentions,probe_lp_head_mean_real,probe_lp_tail_mean_real,probe_lp_tail_minus_head_real,probe_gap_head_mean_real,probe_gap_tail_mean_real,probe_gap_tail_minus_head_real,probe_entropy_head_mean_real,probe_entropy_tail_mean_real,probe_mention_entropy_max_real,probe_mention_lp_min_real,probe_mention_target_gap_min_real,probe_mention_lp_tail_gap_real}"

CONTROLLER_DIR="$OUT_ROOT/discovery/controller"
APPLY_DIR="$OUT_ROOT/discovery/apply"

mkdir -p "$CONTROLLER_DIR" "$APPLY_DIR"

if [[ ! -f "$TABLE_CSV" ]]; then
  echo "[error] missing claim-aware table: $TABLE_CSV" >&2
  exit 1
fi

echo "[1/2] build generative reference B-stage controller"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/build_generative_posthoc_controller.py \
    --discovery_table_csvs "$TABLE_CSV" \
    --target_spec "$TARGET_SPEC" \
    --feature_cols "$REFERENCE_B_FEATURE_COLS" \
    --min_feature_auroc "$MIN_FEATURE_AUROC" \
    --top_k_values "$TOP_K_VALUES" \
    --tau_quantiles "$TAU_QUANTILES" \
    --tau_objective "$TAU_OBJECTIVE" \
    --min_baseline_rate "$MIN_BASELINE_RATE" \
    --max_baseline_rate "$MAX_BASELINE_RATE" \
    --min_selected_count "$MIN_SELECTED_COUNT" \
    --out_dir "$CONTROLLER_DIR"
)

echo "[2/2] apply generative reference B-stage controller"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/apply_generative_posthoc_controller.py \
    --table_csv "$TABLE_CSV" \
    --policy_json "$CONTROLLER_DIR/selected_policy.json" \
    --out_dir "$APPLY_DIR"
)

echo "[done] $CONTROLLER_DIR/summary.json"
echo "[done] $APPLY_DIR/summary.json"
