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
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-false}"
LIMIT="${LIMIT:-0}"

SRC="${SRC:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
V46_ROOT="${V46_ROOT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v46_competition_distill}"
OUT="${OUT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v50_trace_novelty}"
METHOD_PRED_NAME="${METHOD_PRED_NAME:-pred_vga_full_pvg_caption.jsonl}"
TARGET_COL="${TARGET_COL:-v46_safe_target}"

FEATURE_COLS="${FEATURE_COLS:-probe_content_word_count,probe_unique_content_word_count,probe_content_word_diversity,probe_tail_content_repeat_rate,probe_tail_content_new_word_rate,probe_tail_content_generic_rate,probe_generic_narration_word_rate,probe_last_new_content_word_pos_frac,probe_content_bigram_repeat_rate,probe_content_trigram_repeat_rate,probe_n_content_tokens,probe_lp_content_mean_real,probe_lp_content_min_real,probe_lp_content_std_real,probe_target_gap_content_mean_real,probe_target_gap_content_min_real,probe_entropy_content_mean_real,probe_entropy_content_max_real,probe_eos_margin_content_mean_real,probe_last4_lp_mean_real,probe_last4_gap_mean_real,probe_last4_entropy_mean_real,probe_last4_eos_margin_mean_real,probe_stop_eos_logprob_real,probe_stop_eos_margin_real,probe_stop_eos_rank_real,probe_lp_tail_minus_head_real,probe_gap_tail_minus_head_real,probe_entropy_tail_minus_head_real}"

TOP_N_FEATURES="${TOP_N_FEATURES:-32}"
MAX_COMBO_SIZE="${MAX_COMBO_SIZE:-3}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-0.08}"
MAX_CHAIR_I_DELTA="${MAX_CHAIR_I_DELTA:-0.005}"
MAX_CHAIR_S_DELTA="${MAX_CHAIR_S_DELTA:-0.005}"
MIN_RECALL_GAIN="${MIN_RECALL_GAIN:-0.005}"

echo "[config] CAL_ROOT=$CAL_ROOT"
echo "[config] SRC=$SRC"
echo "[config] V46_ROOT=$V46_ROOT"
echo "[config] OUT=$OUT"
echo "[config] GPU=$GPU"
echo "[config] PY_BIN=$PY_BIN"
echo "[config] LIMIT=$LIMIT REUSE_IF_EXISTS=$REUSE_IF_EXISTS"
echo "[config] FEATURE_COLS=$FEATURE_COLS"

for path in "$PY_BIN" "$SRC/splits/val_caption_q.jsonl" "$SRC/splits/test_caption_q.jsonl"; do
  if [[ ! -e "$path" ]]; then
    echo "[error] missing required path: $path" >&2
    exit 1
  fi
done

for split in val test; do
  if [[ ! -f "$SRC/$split/$METHOD_PRED_NAME" ]]; then
    echo "[error] missing prediction file: $SRC/$split/$METHOD_PRED_NAME" >&2
    exit 1
  fi
  if [[ ! -f "$V46_ROOT/pairwise_${split}_apply/decision_rows.csv" ]]; then
    echo "[error] missing v46 route rows: $V46_ROOT/pairwise_${split}_apply/decision_rows.csv" >&2
    exit 1
  fi
done

for split in val test; do
  split_out="$OUT/features/$split"
  mkdir -p "$split_out" "$OUT/distill"

  echo "[1/3][$split] extract object-free intervention trace/novelty features"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_vga_generative_mention_features.py \
    --question_file "$SRC/splits/${split}_caption_q.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$SRC/$split/$METHOD_PRED_NAME" \
    --out_csv "$split_out/intervention_trace.probe.csv" \
    --out_summary_json "$split_out/intervention_trace.probe.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --pred_text_key auto \
    --max_mentions "$MAX_MENTIONS" \
    --reuse_if_exists "$REUSE_IF_EXISTS"

  echo "[2/3][$split] merge v46-safe target with trace features"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/merge_generative_distill_features.py \
    --route_rows_csv "$V46_ROOT/pairwise_${split}_apply/decision_rows.csv" \
    --feature_rows_csv "$split_out/intervention_trace.probe.csv" \
    --feature_prefix probe_ \
    --route_col proxy_route \
    --positive_route baseline \
    --target_col "$TARGET_COL" \
    --target_requires_teacher_positive \
    --out_csv "$OUT/distill/${split}_rows.csv" \
    --out_summary_json "$OUT/distill/${split}_rows.summary.json" \
    --drop_unmatched_feature_rows
done

echo "[3/3][val] fit object-free trace novelty policy"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_route_distill_proxy.py \
  --decision_rows_csv "$OUT/distill/val_rows.csv" \
  --route_col proxy_route \
  --positive_route baseline \
  --target_col "$TARGET_COL" \
  --use_existing_target \
  --feature_prefix probe_ \
  --feature_cols "$FEATURE_COLS" \
  --top_n_features "$TOP_N_FEATURES" \
  --max_combo_size "$MAX_COMBO_SIZE" \
  --min_baseline_rate 0.0 \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --min_recall_gain_vs_intervention "$MIN_RECALL_GAIN" \
  --max_chair_i_delta_vs_intervention "$MAX_CHAIR_I_DELTA" \
  --max_chair_s_delta_vs_intervention "$MAX_CHAIR_S_DELTA" \
  --selection_objective recall_minus_chairi \
  --out_dir "$OUT/distill/trace_novelty_val"

echo "[3/3][test] apply object-free trace novelty policy"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/apply_generative_route_distill_proxy_to_rows.py \
  --rows_csv "$OUT/distill/test_rows.csv" \
  --selected_policy_json "$OUT/distill/trace_novelty_val/selected_policy.json" \
  --out_dir "$OUT/distill/test_apply_trace_novelty" \
  --target_col "$TARGET_COL"

echo "[done] val summary: $OUT/distill/trace_novelty_val/summary.json"
echo "[done] test summary: $OUT/distill/test_apply_trace_novelty/summary.json"
