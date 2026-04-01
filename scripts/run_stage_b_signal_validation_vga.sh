#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
BASELINE_QUESTION_FILE="${BASELINE_QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_discovery/stage_b_signal_validation_vga}"

INTERVENTION_PRED_JSONL="${INTERVENTION_PRED_JSONL:-$OUT_DIR/pred_vga.jsonl}"
BASELINE_PRED_JSONL="${BASELINE_PRED_JSONL:-$OUT_DIR/pred_baseline.jsonl}"

VGA_CONDA_SH="${VGA_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VGA_ENV="${VGA_ENV:-vga}"
PY_BIN="${PY_BIN:-python}"

CONV_MODE="${CONV_MODE:-llava_v1}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
USE_ADD="${USE_ADD:-true}"
ATTN_COEF="${ATTN_COEF:-0.2}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
SAMPLING="${SAMPLING:-false}"
CD_ALPHA="${CD_ALPHA:-0.02}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"
SEED="${SEED:-42}"

BETA="${BETA:-1.0}"
LAMBDA_A="${LAMBDA_A:-0.5}"
LAMBDA_B="${LAMBDA_B:-0.5}"
BLUR_RADIUS="${BLUR_RADIUS:-12.0}"
REUSE_SCORES="${REUSE_SCORES:-true}"
MAKE_PLOTS="${MAKE_PLOTS:-true}"

mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

if [[ ! -f "$BASELINE_PRED_JSONL" ]]; then
  echo "[run] baseline generation"
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$BASELINE_QUESTION_FILE" \
    --answers-file "$BASELINE_PRED_JSONL" \
    --conv-mode "$CONV_MODE" \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens "$MAX_GEN_LEN"
fi

if [[ ! -f "$INTERVENTION_PRED_JSONL" ]]; then
  echo "[run] VGA intervention generation"
  if [[ ! -f "$VGA_CONDA_SH" ]]; then
    echo "[error] missing conda.sh: $VGA_CONDA_SH" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$VGA_CONDA_SH"
  conda activate "$VGA_ENV"
  cd "$VGA_ROOT"
  python eval/object_hallucination_vqa_llava.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$INTERVENTION_PRED_JSONL" \
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
fi

cd "$CAL_ROOT"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/run_frgavr_cleanroom.py \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --intervention_pred_jsonl "$INTERVENTION_PRED_JSONL" \
  --intervention_pred_key output \
  --baseline_pred_jsonl "$BASELINE_PRED_JSONL" \
  --baseline_pred_key text \
  --gt_csv "$GT_CSV" \
  --gt_id_col id \
  --gt_label_col answer \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --conv_mode "$CONV_MODE" \
  --headset_json "$HEADSET_JSON" \
  --beta "$BETA" \
  --lambda_a "$LAMBDA_A" \
  --lambda_b "$LAMBDA_B" \
  --blur_radius "$BLUR_RADIUS" \
  --search_thresholds false \
  --tau_a 0.0 \
  --tau_b 0.0 \
  --reuse_scores "$REUSE_SCORES" \
  --out_dir "$OUT_DIR"

PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/validate_stage_b_signal.py \
  --scores_csv "$OUT_DIR/sample_scores.csv" \
  --out_dir "$OUT_DIR/stage_b_validation" \
  --make_plots "$MAKE_PLOTS"

echo "[done] $OUT_DIR/stage_b_validation"
