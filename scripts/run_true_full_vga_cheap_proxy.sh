#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$ROOT_DIR/experiments/paper_main_b_c_v1_full}"
ALIGNED_ROOT="${ALIGNED_ROOT:-$ROOT_DIR/experiments/aligned_cheap_proxy_from_reference_vga}"
OUT_DIR="${OUT_DIR:-$ALIGNED_ROOT/test/true_full_replay}"

QUESTION_FILE="${QUESTION_FILE:-$ROOT_DIR/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
GT_CSV="${GT_CSV:-$ROOT_DIR/experiments/pope_full_9000/pope_9000_gt.csv}"
POLICY_JSON="${POLICY_JSON:-$ALIGNED_ROOT/discovery/calibration_actual/selected_policy.json}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
HEADSET_JSON="${HEADSET_JSON:-$ROOT_DIR/experiments/pope_discovery/discovery_headset.json}"

MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
USE_ADD="${USE_ADD:-true}"
ATTN_COEF="${ATTN_COEF:-0.2}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
SAMPLING="${SAMPLING:-false}"
CD_ALPHA="${CD_ALPHA:-0.02}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"
ATTN_NORM="${ATTN_NORM:-false}"
SEED="${SEED:-42}"

LIMIT="${LIMIT:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-false}"
SMOKE="${SMOKE:-false}"
SMOKE_LIMIT="${SMOKE_LIMIT:-64}"

REFERENCE_VGA_PRED_JSONL="${REFERENCE_VGA_PRED_JSONL:-$REFERENCE_ROOT/test_stageb/pred_vga.jsonl}"
REFERENCE_BASELINE_PRED_JSONL="${REFERENCE_BASELINE_PRED_JSONL:-$REFERENCE_ROOT/test_stageb/pred_baseline.jsonl}"

if [[ "$SMOKE" == "true" && "$LIMIT" == "0" ]]; then
  LIMIT="$SMOKE_LIMIT"
  echo "[smoke] overriding LIMIT -> $LIMIT"
fi

mkdir -p "$OUT_DIR"

if [[ ! -d "$VGA_ROOT" ]]; then
  echo "[error] VGA_ROOT does not exist: $VGA_ROOT" >&2
  echo "[hint] This runner needs the external VGA_origin checkout." >&2
  exit 1
fi

if [[ ! -f "$VGA_ROOT/eval/object_hallucination_vqa_llava.py" ]]; then
  echo "[error] missing VGA eval entrypoint: $VGA_ROOT/eval/object_hallucination_vqa_llava.py" >&2
  exit 1
fi

PYTHONPATH="$ROOT_DIR" \
python scripts/run_true_full_vga_cheap_proxy.py \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --policy_json "$POLICY_JSON" \
  --gt_csv "$GT_CSV" \
  --out_dir "$OUT_DIR" \
  --vga_root "$VGA_ROOT" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --headset_json "$HEADSET_JSON" \
  --max_gen_len "$MAX_GEN_LEN" \
  --use_add "$USE_ADD" \
  --attn_coef "$ATTN_COEF" \
  --head_balancing "$HEAD_BALANCING" \
  --sampling "$SAMPLING" \
  --cd_alpha "$CD_ALPHA" \
  --start_layer "$START_LAYER" \
  --end_layer "$END_LAYER" \
  --attn_norm "$ATTN_NORM" \
  --seed "$SEED" \
  --limit "$LIMIT" \
  --log_every "$LOG_EVERY" \
  --reuse_if_exists "$REUSE_IF_EXISTS" \
  --reference_vga_pred_jsonl "$REFERENCE_VGA_PRED_JSONL" \
  --reference_baseline_pred_jsonl "$REFERENCE_BASELINE_PRED_JSONL"

echo "[done] true full replay -> $OUT_DIR"
