#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
SRC="${SRC:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
FEATURE_EXP="${FEATURE_EXP:-$CAL_ROOT/experiments/coco_chair_vga_linear_v54_visual_trace_pairwise}"
FEATURE_NAME="${FEATURE_NAME:-semantic_visual_pairwise_features.csv}"
OUT="${OUT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v56_semantic_unit_verify}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"

LIMIT="${LIMIT:-0}"
MAX_UNITS="${MAX_UNITS:-6}"
SCORE_MODE="${SCORE_MODE:-yesno}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_no_more_hall_set}"
SAFE_COL="${SAFE_COL:-net_safe_strict_v2}"
MIN_SELECTED="${MIN_SELECTED:-30}"
MAX_SELECTED="${MAX_SELECTED:-120}"
MIN_DELTA_RECALL="${MIN_DELTA_RECALL:-0.0}"
MIN_DELTA_F1="${MIN_DELTA_F1:-0.0}"
MAX_DELTA_CHAIR_I="${MAX_DELTA_CHAIR_I:-0.005}"
MAX_DELTA_CHAIR_S="${MAX_DELTA_CHAIR_S:-0.02}"
MAX_COMBO_SIZE="${MAX_COMBO_SIZE:-2}"
MAX_RULE_SPECS="${MAX_RULE_SPECS:-700}"
MIN_FEATURE_VALID_COUNT="${MIN_FEATURE_VALID_COUNT:-0}"
MIN_FEATURE_VALID_FRAC="${MIN_FEATURE_VALID_FRAC:-0.8}"
BENEFIT_DIRECTION="${BENEFIT_DIRECTION:-high}"
COST_DIRECTION="${COST_DIRECTION:-low}"

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
  --feature sem_benefit_verify_base_only_yes_prob_sum
  --feature sem_benefit_verify_base_only_yes_prob_mean
  --feature sem_benefit_verify_base_only_yes_prob_min
  --feature sem_benefit_verify_base_only_yes_prob_max
  --feature sem_benefit_verify_base_only_yes_prob_top1
  --feature sem_benefit_verify_base_only_yes_prob_top3
  --feature sem_benefit_verify_base_only_yes_prob_gt050_count
  --feature sem_benefit_verify_base_only_yes_prob_gt060_count
  --feature sem_benefit_verify_base_only_yes_prob_gt070_count
  --feature sem_benefit_verify_base_only_yes_precision_gt050
  --feature sem_benefit_verify_base_only_yes_precision_gt060
  --feature sem_benefit_verify_base_only_margin_sum
  --feature sem_benefit_verify_base_only_margin_mean
  --feature sem_benefit_verify_base_only_gap_margin_mean
  --feature sem_benefit_verify_base_only_support_minus_risk
  --feature sem_cost_verify_base_only_no_risk_sum
  --feature sem_cost_verify_base_only_no_risk_mean
  --feature sem_cost_verify_base_only_no_risk_max
  --feature sem_cost_verify_base_only_yes_prob_lt050_count
  --feature sem_cost_verify_base_only_yes_prob_lt040_count
  --feature sem_cost_verify_base_only_yes_prob_lt030_count
  --feature sem_cost_verify_base_only_yes_prob_lt050_rate
  --feature sem_cost_verify_base_only_yes_prob_lt040_rate
  --feature sem_cost_verify_base_only_yes_entropy_mean
)

echo "[config] SRC=$SRC"
echo "[config] FEATURE_EXP=$FEATURE_EXP"
echo "[config] FEATURE_NAME=$FEATURE_NAME"
echo "[config] OUT=$OUT"
echo "[config] LIMIT=$LIMIT MAX_UNITS=$MAX_UNITS SCORE_MODE=$SCORE_MODE"

for split in val test; do
  feature_csv="$FEATURE_EXP/features/$split/$FEATURE_NAME"
  if [[ ! -f "$feature_csv" ]]; then
    echo "[error] missing feature csv: $feature_csv" >&2
    exit 1
  fi
done

mkdir -p "$OUT/features"

for split in val test; do
  mkdir -p "$OUT/features/$split"
  echo "[1/3][$split] verify baseline-only semantic units"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_generative_semantic_unit_yesno_features.py \
    --question_file "$SRC/splits/${split}_caption_q.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$SRC/$split/pred_vanilla_caption.jsonl" \
    --intervention_pred_jsonl "$SRC/$split/pred_vga_full_pvg_caption.jsonl" \
    --base_feature_csv "$FEATURE_EXP/features/$split/$FEATURE_NAME" \
    --out_csv "$OUT/features/$split/semantic_unit_verify_features.csv" \
    --out_summary_json "$OUT/features/$split/semantic_unit_verify_features.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --max_units "$MAX_UNITS" \
    --score_mode "$SCORE_MODE" \
    --reuse_if_exists "$REUSE_IF_EXISTS"
done

echo "[2/3][val] build target rows"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_sample_net_harm_target.py \
  --baseline_chair_json "$SRC/val/chair_baseline.json" \
  --intervention_chair_json "$SRC/val/chair_vga_full_pvg.json" \
  --feature_rows_csv "$OUT/features/val/semantic_unit_verify_features.csv" \
  --feature_prefix sem_ \
  --out_dir "$OUT/val_targets"

echo "[2/3][test] build target rows"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_sample_net_harm_target.py \
  --baseline_chair_json "$SRC/test/chair_baseline.json" \
  --intervention_chair_json "$SRC/test/chair_vga_full_pvg.json" \
  --feature_rows_csv "$OUT/features/test/semantic_unit_verify_features.csv" \
  --feature_prefix sem_ \
  --out_dir "$OUT/test_targets"

echo "[3/3] fit semantic-unit verifier benefit-cost rule on val, apply to test"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/search_generative_semantic_pairwise_rules.py \
  --net_harm_rows_csv "$OUT/val_targets/net_harm_rows.csv" \
  --feature_rows_csv "$OUT/features/val/semantic_unit_verify_features.csv" \
  --apply_net_harm_rows_csv "$OUT/test_targets/net_harm_rows.csv" \
  --apply_feature_rows_csv "$OUT/features/test/semantic_unit_verify_features.csv" \
  --target_col "$TARGET_COL" \
  --safe_col "$SAFE_COL" \
  --feature_prefix sem_ \
  "${FEATURE_ARGS[@]}" \
  --require_benefit_cost_combo \
  --benefit_prefix sem_benefit_ \
  --cost_prefix sem_cost_,sem_visual_ \
  --benefit_direction "$BENEFIT_DIRECTION" \
  --cost_direction "$COST_DIRECTION" \
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
  --out_dir "$OUT/rule_val_to_test_v56"

echo "[done] summary: $OUT/rule_val_to_test_v56/summary.json"
