#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
SRC="${SRC:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
FEATURE_EXP="${FEATURE_EXP:-$CAL_ROOT/experiments/coco_chair_vga_linear_v54_visual_trace_pairwise}"
FEATURE_NAME="${FEATURE_NAME:-semantic_visual_pairwise_features.csv}"
OUT="${OUT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v55_pairwise_safe_oracle_target}"
TEST_ARTIFACT_ROWS="${TEST_ARTIFACT_ROWS:-$CAL_ROOT/experiments/coco_chair_vga_linear_v48b_trace_cascade/coco_chair_intervention_hall_span_trace_full500/test_full500/span_trace_rows.csv}"
VAL_ARTIFACT_ROWS="${VAL_ARTIFACT_ROWS:-}"

TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_no_more_hall_set}"
SAFE_COL="${SAFE_COL:-net_safe_strict_v2}"
MIN_SELECTED="${MIN_SELECTED:-5}"
MAX_SELECTED="${MAX_SELECTED:-120}"
MIN_DELTA_RECALL="${MIN_DELTA_RECALL:-0.0}"
MIN_DELTA_F1="${MIN_DELTA_F1:-0.0}"
MAX_DELTA_CHAIR_I="${MAX_DELTA_CHAIR_I:-0.005}"
MAX_DELTA_CHAIR_S="${MAX_DELTA_CHAIR_S:-0.02}"
MAX_COMBO_SIZE="${MAX_COMBO_SIZE:-2}"
MAX_RULE_SPECS="${MAX_RULE_SPECS:-500}"
MIN_FEATURE_VALID_COUNT="${MIN_FEATURE_VALID_COUNT:-0}"
MIN_FEATURE_VALID_FRAC="${MIN_FEATURE_VALID_FRAC:-0.8}"

FEATURE_ARGS=(
  --feature sem_benefit_delta_unique_content_units
  --feature sem_benefit_base_only_sem_unit_count
  --feature sem_benefit_delta_last_new_content_pos_frac
  --feature sem_benefit_delta_tail_new_content_rate
  --feature sem_benefit_delta_content_diversity
  --feature sem_cost_base_new_content_lp_min
  --feature sem_cost_base_new_content_gap_min
  --feature sem_cost_base_new_content_entropy_max
  --feature sem_cost_base_stop_eos_margin
  --feature sem_cost_base_stop_eos_logprob
  --feature sem_cost_base_tail_repetition_rate
  --feature sem_cost_base_generic_phrase_ratio
  --feature sem_cost_base_low_lp_x_base_only
  --feature sem_cost_base_low_gap_x_base_only
  --feature sem_cost_base_high_entropy_x_base_only
  --feature sem_cost_base_new_content_word_count
  --feature sem_cost_base_new_content_low_lp_count_le_m2
  --feature sem_cost_base_new_content_low_lp_count_le_m3
  --feature sem_cost_base_new_content_low_gap_count_le_000
  --feature sem_cost_base_new_content_low_gap_count_le_025
  --feature sem_cost_base_new_content_high_entropy_count_ge_300
  --feature sem_cost_base_new_content_high_entropy_count_ge_350
  --feature sem_visual_suppression_content_mean
  --feature sem_visual_suppression_content_min
  --feature sem_visual_suppression_tail_mean
  --feature sem_visual_suppression_last4_mean
  --feature sem_visual_suppression_tail_minus_head
  --feature sem_visual_suppression_topk_content_mean
  --feature sem_visual_suppression_topk_tail_minus_head
  --feature sem_visual_uplift_entropy_content_mean
  --feature sem_visual_uplift_entropy_content_max
  --feature sem_visual_uplift_entropy_tail_minus_head
  --feature sem_visual_suppression_x_base_only_content_mean
  --feature sem_visual_suppression_x_base_only_tail_mean
  --feature sem_visual_suppression_x_base_only_last4_mean
  --feature sem_visual_suppression_x_base_only_tail_minus_head
)

echo "[config] SRC=$SRC"
echo "[config] FEATURE_EXP=$FEATURE_EXP"
echo "[config] FEATURE_NAME=$FEATURE_NAME"
echo "[config] OUT=$OUT"
echo "[config] TARGET_COL=$TARGET_COL"

for split in val test; do
  feature_csv="$FEATURE_EXP/features/$split/$FEATURE_NAME"
  if [[ ! -f "$feature_csv" ]]; then
    echo "[error] missing feature csv: $feature_csv" >&2
    echo "[hint] run v54 first, or set FEATURE_EXP/FEATURE_NAME to an existing pairwise feature table." >&2
    exit 1
  fi
done

mkdir -p "$OUT"

echo "[1/2][val] build oracle-B target rows"
VAL_ARTIFACT_ARG=()
if [[ -n "$VAL_ARTIFACT_ROWS" && -f "$VAL_ARTIFACT_ROWS" ]]; then
  VAL_ARTIFACT_ARG=(--parser_artifact_object_rows_csv "$VAL_ARTIFACT_ROWS")
fi
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_sample_net_harm_target.py \
  --baseline_chair_json "$SRC/val/chair_baseline.json" \
  --intervention_chair_json "$SRC/val/chair_vga_full_pvg.json" \
  "${VAL_ARTIFACT_ARG[@]}" \
  --feature_rows_csv "$FEATURE_EXP/features/val/$FEATURE_NAME" \
  --feature_prefix sem_ \
  --out_dir "$OUT/val_targets"

echo "[1/2][test] build oracle-B target rows"
TEST_ARTIFACT_ARG=()
if [[ -n "$TEST_ARTIFACT_ROWS" && -f "$TEST_ARTIFACT_ROWS" ]]; then
  TEST_ARTIFACT_ARG=(--parser_artifact_object_rows_csv "$TEST_ARTIFACT_ROWS")
fi
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_sample_net_harm_target.py \
  --baseline_chair_json "$SRC/test/chair_baseline.json" \
  --intervention_chair_json "$SRC/test/chair_vga_full_pvg.json" \
  "${TEST_ARTIFACT_ARG[@]}" \
  --feature_rows_csv "$FEATURE_EXP/features/test/$FEATURE_NAME" \
  --feature_prefix sem_ \
  --out_dir "$OUT/test_targets"

echo "[2/2] fit pairwise benefit-cost rule on val, apply to test"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/search_generative_semantic_pairwise_rules.py \
  --net_harm_rows_csv "$OUT/val_targets/net_harm_rows.csv" \
  --feature_rows_csv "$FEATURE_EXP/features/val/$FEATURE_NAME" \
  --apply_net_harm_rows_csv "$OUT/test_targets/net_harm_rows.csv" \
  --apply_feature_rows_csv "$FEATURE_EXP/features/test/$FEATURE_NAME" \
  --target_col "$TARGET_COL" \
  --safe_col "$SAFE_COL" \
  --feature_prefix sem_ \
  "${FEATURE_ARGS[@]}" \
  --require_benefit_cost_combo \
  --benefit_prefix sem_benefit_ \
  --cost_prefix sem_cost_,sem_visual_ \
  --max_combo_size "$MAX_COMBO_SIZE" \
  --max_rule_specs "$MAX_RULE_SPECS" \
  --min_feature_valid_count "$MIN_FEATURE_VALID_COUNT" \
  --min_feature_valid_frac "$MIN_FEATURE_VALID_FRAC" \
  --min_selected "$MIN_SELECTED" \
  --max_selected "$MAX_SELECTED" \
  --min_delta_recall "$MIN_DELTA_RECALL" \
  --min_delta_f1 "$MIN_DELTA_F1" \
  --max_delta_chair_i "$MAX_DELTA_CHAIR_I" \
  --max_delta_chair_s "$MAX_DELTA_CHAIR_S" \
  --out_dir "$OUT/rule_val_to_test_v55"

echo "[done] summary: $OUT/rule_val_to_test_v55/summary.json"
