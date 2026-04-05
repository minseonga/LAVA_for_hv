#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
PYTHON_BIN="${PYTHON_BIN:-python}"

QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_vsc_adaptive_sweep_v1}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
NUM_BEAMS="${NUM_BEAMS:-1}"
SAMPLING="${SAMPLING:-false}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
ATTN_NORM="${ATTN_NORM:-false}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

STRONG_ATTN_COEF="${STRONG_ATTN_COEF:-0.2}"
STRONG_CD_ALPHA="${STRONG_CD_ALPHA:-0.02}"
STRONG_START_LAYER="${STRONG_START_LAYER:-2}"
STRONG_END_LAYER="${STRONG_END_LAYER:-15}"

WEAK_ATTN_COEF="${WEAK_ATTN_COEF:-0.1}"
WEAK_CD_ALPHA="${WEAK_CD_ALPHA:-0.01}"
WEAK_START_LAYER="${WEAK_START_LAYER:-4}"
WEAK_END_LAYER="${WEAK_END_LAYER:-12}"

QUANTILES="${QUANTILES:-0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95}"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/run_vga_vsc_adaptive_sweep.py \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --gt_csv "$GT_CSV" \
  --out_dir "$OUT_ROOT" \
  --vga_root "$VGA_ROOT" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --max_gen_len "$MAX_GEN_LEN" \
  --sampling "$SAMPLING" \
  --num_beams "$NUM_BEAMS" \
  --head_balancing "$HEAD_BALANCING" \
  --attn_norm "$ATTN_NORM" \
  --seed "$SEED" \
  --limit "$LIMIT" \
  --log_every "$LOG_EVERY" \
  --reuse_if_exists "$REUSE_IF_EXISTS" \
  --strong_attn_coef "$STRONG_ATTN_COEF" \
  --strong_cd_alpha "$STRONG_CD_ALPHA" \
  --strong_start_layer "$STRONG_START_LAYER" \
  --strong_end_layer "$STRONG_END_LAYER" \
  --weak_attn_coef "$WEAK_ATTN_COEF" \
  --weak_cd_alpha "$WEAK_CD_ALPHA" \
  --weak_start_layer "$WEAK_START_LAYER" \
  --weak_end_layer "$WEAK_END_LAYER" \
  --quantiles "$QUANTILES"

echo "[done] $OUT_ROOT/summary.json"
