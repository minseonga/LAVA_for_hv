#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REFERENCE_ROOT="${REFERENCE_ROOT:-$ROOT_DIR/experiments/paper_main_b_c_v1_full}"
ALIGNED_ROOT="${ALIGNED_ROOT:-$ROOT_DIR/experiments/aligned_cheap_proxy_from_reference_vga}"
OUT_DIR="${OUT_DIR:-$ALIGNED_ROOT/test/sequential_replay}"

QUESTION_FILE="${QUESTION_FILE:-$ROOT_DIR/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
GT_CSV="${GT_CSV:-$ROOT_DIR/experiments/pope_full_9000/pope_9000_gt.csv}"

INTERVENTION_PRED_JSONL="${INTERVENTION_PRED_JSONL:-$REFERENCE_ROOT/test_stageb/pred_vga.jsonl}"
BASELINE_EVAL_PRED_JSONL="${BASELINE_EVAL_PRED_JSONL:-$REFERENCE_ROOT/test_stageb/pred_baseline.jsonl}"
POLICY_JSON="${POLICY_JSON:-$ALIGNED_ROOT/discovery/calibration_actual/selected_policy.json}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
LIMIT="${LIMIT:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-false}"
VGA_TOTAL_WALL_SEC="${VGA_TOTAL_WALL_SEC:--1}"

mkdir -p "$OUT_DIR"

PYTHONPATH="$ROOT_DIR" \
python scripts/run_aligned_cheap_proxy_sequential_replay.py \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --intervention_pred_jsonl "$INTERVENTION_PRED_JSONL" \
  --policy_json "$POLICY_JSON" \
  --gt_csv "$GT_CSV" \
  --out_dir "$OUT_DIR" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --baseline_eval_pred_jsonl "$BASELINE_EVAL_PRED_JSONL" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --log_every "$LOG_EVERY" \
  --reuse_if_exists "$REUSE_IF_EXISTS" \
  --vga_total_wall_sec "$VGA_TOTAL_WALL_SEC"

echo "[done] sequential replay -> $OUT_DIR"
