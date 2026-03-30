#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
CONTROLLER_SUMMARY_JSON="${CONTROLLER_SUMMARY_JSON:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/controller/summary.json}"
OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/vga_discovery_headset_frg_only_online_9000}"

CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SAMPLING="${SAMPLING:-false}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
NUM_BEAMS="${NUM_BEAMS:-1}"

CD_ALPHA="${CD_ALPHA:-0.02}"
ATTN_COEF="${ATTN_COEF:-0.2}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
ATTN_NORM="${ATTN_NORM:-false}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

mkdir -p "$OUT_DIR"
cd "$CAL_ROOT"

python scripts/run_pnp_hard_veto_online.py \
  --backend vga \
  --vga_root "$VGA_ROOT" \
  --model_path "$MODEL_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --question_file "$QUESTION_FILE" \
  --out_dir "$OUT_DIR" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --controller_summary_json "$CONTROLLER_SUMMARY_JSON" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --sampling "$SAMPLING" \
  --max_gen_len "$MAX_GEN_LEN" \
  --num_beams "$NUM_BEAMS" \
  --cd_alpha "$CD_ALPHA" \
  --attn_coef "$ATTN_COEF" \
  --start_layer "$START_LAYER" \
  --end_layer "$END_LAYER" \
  --head_balancing "$HEAD_BALANCING" \
  --attn_norm "$ATTN_NORM" \
  --late_start "$LATE_START" \
  --late_end "$LATE_END" \
  --probe_feature_mode static_headset \
  --headset_json "$HEADSET_JSON" \
  --use_gmi false \
  --seed "$SEED" \
  --max_samples "$MAX_SAMPLES" \
  --gt_csv "$GT_CSV" \
  --gt_id_col id \
  --gt_label_col answer

echo "[done] $OUT_DIR"
echo "[saved] $OUT_DIR/summary.json"
