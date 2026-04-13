#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  GPU="$1"
  shift
fi

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MAX_MENTIONS="${MAX_MENTIONS:-12}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
LIMIT="${LIMIT:-50}"
VISUAL_LATE_START="${VISUAL_LATE_START:-2}"
VISUAL_LATE_END="${VISUAL_LATE_END:-15}"
VISUAL_TOPK_HEADS="${VISUAL_TOPK_HEADS:-4}"

SRC="${SRC:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
OUT="${OUT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v54_visual_trace_pairwise}"
TEST_ARTIFACT_ROWS="${TEST_ARTIFACT_ROWS:-$CAL_ROOT/experiments/coco_chair_vga_linear_v48b_trace_cascade/coco_chair_intervention_hall_span_trace_full500/test_full500/span_trace_rows.csv}"
VAL_ARTIFACT_ROWS="${VAL_ARTIFACT_ROWS:-}"

TARGET_COL="${TARGET_COL:-net_harm_strict_v2}"
SAFE_COL="${SAFE_COL:-net_safe_strict_v2}"
MIN_SELECTED="${MIN_SELECTED:-3}"
MAX_SELECTED="${MAX_SELECTED:-60}"
MIN_DELTA_RECALL="${MIN_DELTA_RECALL:-0.0}"
MIN_DELTA_F1="${MIN_DELTA_F1:-0.0}"
MAX_DELTA_CHAIR_I="${MAX_DELTA_CHAIR_I:-0.005}"
MAX_DELTA_CHAIR_S="${MAX_DELTA_CHAIR_S:-0.02}"
MAX_COMBO_SIZE="${MAX_COMBO_SIZE:-2}"
MAX_RULE_SPECS="${MAX_RULE_SPECS:-200}"
if [[ -z "${MIN_FEATURE_VALID_COUNT:-}" ]]; then
  if [[ "$LIMIT" -gt 0 ]]; then
    MIN_FEATURE_VALID_COUNT="$LIMIT"
  else
    MIN_FEATURE_VALID_COUNT="0"
  fi
fi
MIN_FEATURE_VALID_FRAC="${MIN_FEATURE_VALID_FRAC:-0.8}"

FEATURE_ARGS=(
  --feature sem_benefit_delta_unique_content_units
  --feature sem_benefit_base_only_sem_unit_count
  --feature sem_benefit_delta_last_new_content_pos_frac
  --feature sem_benefit_delta_tail_new_content_rate
  --feature sem_benefit_delta_content_diversity
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
echo "[config] OUT=$OUT"
echo "[config] GPU=$GPU"
echo "[config] PY_BIN=$PY_BIN"
echo "[config] LIMIT=$LIMIT visual_layers=$VISUAL_LATE_START:$VISUAL_LATE_END topk_heads=$VISUAL_TOPK_HEADS"
echo "[config] MIN_FEATURE_VALID_COUNT=$MIN_FEATURE_VALID_COUNT MIN_FEATURE_VALID_FRAC=$MIN_FEATURE_VALID_FRAC"

for path in "$PY_BIN" "$SRC/splits/val_caption_q.jsonl" "$SRC/splits/test_caption_q.jsonl"; do
  if [[ ! -e "$path" ]]; then
    echo "[error] missing required path: $path" >&2
    exit 1
  fi
done

for split in val test; do
  mkdir -p "$OUT/features/$split"

  echo "[1/4][$split] extract baseline visual-token trace"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_vga_generative_mention_features.py \
    --question_file "$SRC/splits/${split}_caption_q.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$SRC/$split/pred_vanilla_caption.jsonl" \
    --out_csv "$OUT/features/$split/baseline_visual_trace.probe.csv" \
    --out_summary_json "$OUT/features/$split/baseline_visual_trace.probe.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --pred_text_key auto \
    --max_mentions "$MAX_MENTIONS" \
    --visual_trace true \
    --visual_late_start "$VISUAL_LATE_START" \
    --visual_late_end "$VISUAL_LATE_END" \
    --visual_topk_heads "$VISUAL_TOPK_HEADS" \
    --reuse_if_exists "$REUSE_IF_EXISTS"

  echo "[2/4][$split] extract intervention visual-token trace"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_vga_generative_mention_features.py \
    --question_file "$SRC/splits/${split}_caption_q.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$SRC/$split/pred_vga_full_pvg_caption.jsonl" \
    --out_csv "$OUT/features/$split/intervention_visual_trace.probe.csv" \
    --out_summary_json "$OUT/features/$split/intervention_visual_trace.probe.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --pred_text_key auto \
    --max_mentions "$MAX_MENTIONS" \
    --visual_trace true \
    --visual_late_start "$VISUAL_LATE_START" \
    --visual_late_end "$VISUAL_LATE_END" \
    --visual_topk_heads "$VISUAL_TOPK_HEADS" \
    --reuse_if_exists "$REUSE_IF_EXISTS"

  echo "[3/4][$split] build semantic + visual pairwise features"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_generative_semantic_pairwise_features.py \
    --baseline_pred_jsonl "$SRC/$split/pred_vanilla_caption.jsonl" \
    --intervention_pred_jsonl "$SRC/$split/pred_vga_full_pvg_caption.jsonl" \
    --baseline_trace_csv "$OUT/features/$split/baseline_visual_trace.probe.csv" \
    --intervention_trace_csv "$OUT/features/$split/intervention_visual_trace.probe.csv" \
    --out_csv "$OUT/features/$split/semantic_visual_pairwise_features.csv" \
    --out_summary_json "$OUT/features/$split/semantic_visual_pairwise_features.summary.json"
done

echo "[4/4][val] build net-harm labels"
VAL_ARTIFACT_ARG=()
if [[ -n "$VAL_ARTIFACT_ROWS" && -f "$VAL_ARTIFACT_ROWS" ]]; then
  VAL_ARTIFACT_ARG=(--parser_artifact_object_rows_csv "$VAL_ARTIFACT_ROWS")
fi
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_sample_net_harm_target.py \
  --baseline_chair_json "$SRC/val/chair_baseline.json" \
  --intervention_chair_json "$SRC/val/chair_vga_full_pvg.json" \
  "${VAL_ARTIFACT_ARG[@]}" \
  --feature_rows_csv "$OUT/features/val/semantic_visual_pairwise_features.csv" \
  --feature_prefix sem_ \
  --out_dir "$OUT/val_net_harm"

echo "[4/4][test] build net-harm labels"
TEST_ARTIFACT_ARG=()
if [[ -n "$TEST_ARTIFACT_ROWS" && -f "$TEST_ARTIFACT_ROWS" ]]; then
  TEST_ARTIFACT_ARG=(--parser_artifact_object_rows_csv "$TEST_ARTIFACT_ROWS")
fi
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_sample_net_harm_target.py \
  --baseline_chair_json "$SRC/test/chair_baseline.json" \
  --intervention_chair_json "$SRC/test/chair_vga_full_pvg.json" \
  "${TEST_ARTIFACT_ARG[@]}" \
  --feature_rows_csv "$OUT/features/test/semantic_visual_pairwise_features.csv" \
  --feature_prefix sem_ \
  --out_dir "$OUT/test_net_harm"

echo "[4/4][val->test] fit benefit + visual-intervention gate on val, apply to test"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/search_generative_semantic_pairwise_rules.py \
  --net_harm_rows_csv "$OUT/val_net_harm/net_harm_rows.csv" \
  --feature_rows_csv "$OUT/features/val/semantic_visual_pairwise_features.csv" \
  --apply_net_harm_rows_csv "$OUT/test_net_harm/net_harm_rows.csv" \
  --apply_feature_rows_csv "$OUT/features/test/semantic_visual_pairwise_features.csv" \
  --target_col "$TARGET_COL" \
  --safe_col "$SAFE_COL" \
  --feature_prefix sem_ \
  "${FEATURE_ARGS[@]}" \
  --require_benefit_cost_combo \
  --benefit_prefix sem_benefit_ \
  --cost_prefix sem_visual_ \
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
  --out_dir "$OUT/rule_val_to_test_v54"

echo "[done] summary: $OUT/rule_val_to_test_v54/summary.json"
