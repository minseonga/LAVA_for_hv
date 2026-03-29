#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VGA_ENV="${VGA_ENV:-vga}"
EVAL_ENV="${EVAL_ENV:-vocot}"

CUDA_DEVICE="${CUDA_DEVICE:-6}"

VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
CALIB_ROOT="${CALIB_ROOT:-/home/kms/LLaVA_calibration}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
QUESTION_FILE="${QUESTION_FILE:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/01_subset/pope_strict_1000_q.jsonl}"
GT_CSV="${GT_CSV:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/01_subset/pope_strict_1000_gt.csv}"

OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/vga_with_frrs_1000}"
mkdir -p "$OUT_DIR"
RUN_TAG="${RUN_TAG:-vga_frrs}"
OVERWRITE="${OVERWRITE:-0}"

# VGA knobs
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

# FRRS knobs
ENABLE_FRRS="${ENABLE_FRRS:-true}"
FRRS_LATE_START="${FRRS_LATE_START:-18}"
FRRS_LATE_END="${FRRS_LATE_END:-21}"
FRRS_ALPHA="${FRRS_ALPHA:-0.5}"
FRRS_BETA="${FRRS_BETA:-0.0}"
FRRS_TAU_C="${FRRS_TAU_C:-0.0}"
FRRS_TAU_E="${FRRS_TAU_E:-0.0}"
FRRS_K_C="${FRRS_K_C:-8.0}"
FRRS_K_E="${FRRS_K_E:-8.0}"
FRRS_TOPK_RATIO="${FRRS_TOPK_RATIO:-0.2}"
FRRS_EPS="${FRRS_EPS:-1e-6}"
FRRS_ARM="${FRRS_ARM:-controller}"
FRRS_HEAD_MODE="${FRRS_HEAD_MODE:-dynamic}"
FRRS_R_PERCENT="${FRRS_R_PERCENT:-0.2}"
FRRS_ONLINE_RECOMPUTE_FEATS="${FRRS_ONLINE_RECOMPUTE_FEATS:-true}"
FRRS_ONLINE_BLEND="${FRRS_ONLINE_BLEND:-1.0}"
FRRS_FEATS_JSON="${FRRS_FEATS_JSON:-}"
FRRS_HEADSET_JSON="${FRRS_HEADSET_JSON:-}"

PRED_JSONL="$OUT_DIR/pred_${RUN_TAG}.jsonl"
METRICS_JSON="$OUT_DIR/metrics_${RUN_TAG}.json"

if [[ "$OVERWRITE" != "1" ]]; then
  if [[ -f "$PRED_JSONL" || -f "$METRICS_JSON" ]]; then
    echo "[error] output file already exists for RUN_TAG=${RUN_TAG}" >&2
    echo "        PRED_JSONL=$PRED_JSONL" >&2
    echo "        METRICS_JSON=$METRICS_JSON" >&2
    echo "        Use a different RUN_TAG or set OVERWRITE=1" >&2
    exit 1
  fi
fi

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[error] conda.sh not found: $CONDA_SH" >&2
  exit 1
fi
source "$CONDA_SH"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

conda activate "$VGA_ENV"
cd "$VGA_ROOT"

CMD=(python eval/object_hallucination_vqa_llava.py
  --model-path "$MODEL_PATH"
  --image-folder "$IMAGE_FOLDER"
  --question-file "$QUESTION_FILE"
  --answers-file "$PRED_JSONL"
  --conv-mode "$CONV_MODE"
  --max_gen_len "$MAX_GEN_LEN"
  --use_add "$USE_ADD"
  --attn_coef "$ATTN_COEF"
  --head_balancing "$HEAD_BALANCING"
  --sampling "$SAMPLING"
  --cd_alpha "$CD_ALPHA"
  --seed "$SEED"
  --start_layer "$START_LAYER"
  --end_layer "$END_LAYER"
  --enable_frrs "$ENABLE_FRRS"
  --frrs_late_start "$FRRS_LATE_START"
  --frrs_late_end "$FRRS_LATE_END"
  --frrs_alpha "$FRRS_ALPHA"
  --frrs_beta "$FRRS_BETA"
  --frrs_tau_c "$FRRS_TAU_C"
  --frrs_tau_e "$FRRS_TAU_E"
  --frrs_k_c "$FRRS_K_C"
  --frrs_k_e "$FRRS_K_E"
  --frrs_topk_ratio "$FRRS_TOPK_RATIO"
  --frrs_eps "$FRRS_EPS"
  --frrs_arm "$FRRS_ARM"
  --frrs_head_mode "$FRRS_HEAD_MODE"
  --frrs_r_percent "$FRRS_R_PERCENT"
  --frrs_online_recompute_feats "$FRRS_ONLINE_RECOMPUTE_FEATS"
  --frrs_online_blend "$FRRS_ONLINE_BLEND"
)
if [[ -n "$FRRS_FEATS_JSON" ]]; then
  CMD+=(--frrs_feats_json "$FRRS_FEATS_JSON")
fi
if [[ -n "$FRRS_HEADSET_JSON" ]]; then
  CMD+=(--frrs_headset_json "$FRRS_HEADSET_JSON")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

conda activate "$EVAL_ENV"
cd "$CALIB_ROOT"
python scripts/eval_pope_subset_yesno.py \
  --gt_csv "$GT_CSV" \
  --pred_jsonl "$PRED_JSONL" \
  --pred_text_key output \
  --out_json "$METRICS_JSON"

echo "[done] $OUT_DIR"
echo "[saved] $PRED_JSONL"
echo "[saved] $METRICS_JSON"
