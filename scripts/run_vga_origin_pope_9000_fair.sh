#!/usr/bin/env bash
set -euo pipefail

# Fair VGA_origin rerun on POPE-9000:
# - same generation/guidance settings for both no-object and with-object
# - only question_file differs
# - includes yes/no evaluation

# -----------------------------
# User-configurable env vars
# -----------------------------
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VGA_ENV="${VGA_ENV:-vga}"
EVAL_ENV="${EVAL_ENV:-vocot}"

CUDA_DEVICE="${CUDA_DEVICE:-6}"

VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
CALIB_ROOT="${CALIB_ROOT:-/home/kms/LLaVA_calibration}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"

Q_NOOBJ="${Q_NOOBJ:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_q.jsonl}"
Q_WITHOBJ="${Q_WITHOBJ:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_gt.csv}"

OUT_DIR="${OUT_DIR:-/home/kms/VGA_origin/outputs/pope_9000_fair_rerun}"

# Fixed "fair" settings (applied identically to both runs)
CONV_MODE="${CONV_MODE:-llava_v1}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
USE_ADD="${USE_ADD:-true}"
ATTN_COEF="${ATTN_COEF:-0.2}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
SAMPLING="${SAMPLING:-false}"
CD_ALPHA="${CD_ALPHA:-0.02}"
SEED="${SEED:-42}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"

NOOBJ_OUT="$OUT_DIR/noobj"
WITHOBJ_OUT="$OUT_DIR/withobj"
mkdir -p "$NOOBJ_OUT" "$WITHOBJ_OUT"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[error] conda.sh not found: $CONDA_SH" >&2
  exit 1
fi

for f in "$Q_NOOBJ" "$Q_WITHOBJ" "$GT_CSV"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing file: $f" >&2
    exit 1
  fi
done

if [[ ! -d "$VGA_ROOT" ]]; then
  echo "[error] missing directory: $VGA_ROOT" >&2
  exit 1
fi

if [[ ! -d "$CALIB_ROOT" ]]; then
  echo "[error] missing directory: $CALIB_ROOT" >&2
  exit 1
fi

source "$CONDA_SH"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

run_vga() {
  local qfile="$1"
  local ofile="$2"

  conda activate "$VGA_ENV"
  cd "$VGA_ROOT"
  python eval/object_hallucination_vqa_llava.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$qfile" \
    --answers-file "$ofile" \
    --conv-mode "$CONV_MODE" \
    --max_gen_len "$MAX_GEN_LEN" \
    --use_add "$USE_ADD" \
    --attn_coef "$ATTN_COEF" \
    --head_balancing "$HEAD_BALANCING" \
    --sampling "$SAMPLING" \
    --cd_alpha "$CD_ALPHA" \
    --seed "$SEED" \
    --start_layer "$START_LAYER" \
    --end_layer "$END_LAYER"
}

run_eval() {
  local pred_jsonl="$1"
  local out_json="$2"

  conda activate "$EVAL_ENV"
  cd "$CALIB_ROOT"
  python scripts/eval_pope_subset_yesno.py \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$pred_jsonl" \
    --pred_text_key output \
    --out_json "$out_json"
}

echo "[run] VGA no-object"
run_vga "$Q_NOOBJ" "$NOOBJ_OUT/pred.jsonl"

echo "[run] VGA with-object"
run_vga "$Q_WITHOBJ" "$WITHOBJ_OUT/pred.jsonl"

echo "[run] Eval no-object"
run_eval "$NOOBJ_OUT/pred.jsonl" "$NOOBJ_OUT/metrics.json"

echo "[run] Eval with-object"
run_eval "$WITHOBJ_OUT/pred.jsonl" "$WITHOBJ_OUT/metrics.json"

echo "[done] $OUT_DIR"
echo "[metrics] noobj:   $NOOBJ_OUT/metrics.json"
echo "[metrics] withobj: $WITHOBJ_OUT/metrics.json"

