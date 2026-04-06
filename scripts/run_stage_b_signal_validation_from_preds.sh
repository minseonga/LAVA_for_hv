#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
PY_BIN="${PY_BIN:-python}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
BASELINE_QUESTION_FILE="${BASELINE_QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/stage_b_signal_validation_generic}"

INTERVENTION_PRED_JSONL="${INTERVENTION_PRED_JSONL:-}"
INTERVENTION_PRED_KEY="${INTERVENTION_PRED_KEY:-output}"
BASELINE_PRED_JSONL="${BASELINE_PRED_JSONL:-$OUT_DIR/pred_baseline.jsonl}"
BASELINE_PRED_KEY="${BASELINE_PRED_KEY:-text}"

MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
BETA="${BETA:-1.0}"
LAMBDA_A="${LAMBDA_A:-0.5}"
LAMBDA_B="${LAMBDA_B:-0.5}"
BLUR_RADIUS="${BLUR_RADIUS:-12.0}"
REUSE_SCORES="${REUSE_SCORES:-true}"
MAKE_PLOTS="${MAKE_PLOTS:-true}"

mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

if [[ -z "$INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] INTERVENTION_PRED_JSONL is required" >&2
  exit 1
fi

if [[ ! -f "$INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] missing intervention prediction file: $INTERVENTION_PRED_JSONL" >&2
  exit 1
fi

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

cd "$CAL_ROOT"
PYTHONUNBUFFERED=1 PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/run_frgavr_cleanroom.py \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --intervention_pred_jsonl "$INTERVENTION_PRED_JSONL" \
  --intervention_pred_key "$INTERVENTION_PRED_KEY" \
  --baseline_pred_jsonl "$BASELINE_PRED_JSONL" \
  --baseline_pred_key "$BASELINE_PRED_KEY" \
  --gt_csv "$GT_CSV" \
  --gt_id_col id \
  --gt_label_col answer \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
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
