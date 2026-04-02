#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

QUESTION_FILE="${QUESTION_FILE:-$ROOT_DIR/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
INTERVENTION_PRED_JSONL="${INTERVENTION_PRED_JSONL:-$ROOT_DIR/experiments/pope_full_9000/stage_b_signal_validation_vga/pred_vga.jsonl}"
SCORES_CSV="${SCORES_CSV:-$ROOT_DIR/experiments/pope_full_9000/stage_b_signal_validation_vga/sample_scores.csv}"
TAXONOMY_CSV="${TAXONOMY_CSV:-$ROOT_DIR/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv}"

OUT_DIR="${OUT_DIR:-$ROOT_DIR/experiments/pope_full_9000/c_stage_cheap_proxy_eval_vga}"
CHEAP_CSV="${CHEAP_CSV:-$OUT_DIR/cheap_online_features.csv}"
CHEAP_SUMMARY_JSON="${CHEAP_SUMMARY_JSON:-$OUT_DIR/cheap_online_features_summary.json}"

SUBSET_PERCENTS="${SUBSET_PERCENTS:-0.5,1,2,5}"
PAIR_FEATURE_TOPN="${PAIR_FEATURE_TOPN:-6}"
TOPK_ROWS="${TOPK_ROWS:-8}"
REUSE_CHEAP="${REUSE_CHEAP:-true}"
LOG_EVERY="${LOG_EVERY:-25}"

FEATURE_COLS="${FEATURE_COLS:-cheap_lp_content_mean,cheap_lp_content_std,cheap_lp_content_min,cheap_entropy_content_mean,cheap_entropy_content_std,cheap_margin_content_mean,cheap_margin_content_std,cheap_margin_content_min,cheap_target_gap_content_mean,cheap_target_gap_content_std,cheap_target_gap_content_min,cheap_target_argmax_content_mean,cheap_conflict_lp_minus_entropy,cheap_conflict_gap_minus_entropy,cheap_content_fraction}"

mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT_DIR" \
python scripts/extract_c_stage_cheap_online_features.py \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --intervention_pred_jsonl "$INTERVENTION_PRED_JSONL" \
  --out_csv "$CHEAP_CSV" \
  --out_summary_json "$CHEAP_SUMMARY_JSON" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --reuse_if_exists "$REUSE_CHEAP" \
  --log_every "$LOG_EVERY"

PYTHONPATH="$ROOT_DIR" \
python scripts/evaluate_c_stage_candidates.py \
  --scores_csv "$SCORES_CSV" \
  --taxonomy_csv "$TAXONOMY_CSV" \
  --features_csv "$CHEAP_CSV" \
  --feature_cols "$FEATURE_COLS" \
  --subset_percents "$SUBSET_PERCENTS" \
  --pair_feature_topn "$PAIR_FEATURE_TOPN" \
  --topk_rows "$TOPK_ROWS" \
  --out_dir "$OUT_DIR"

echo "[done] $OUT_DIR"
