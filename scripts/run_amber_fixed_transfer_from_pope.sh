#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
AMBER_ROOT="${AMBER_ROOT:-/home/kms/data/AMBER}"
AMBER_IMAGE_FOLDER="${AMBER_IMAGE_FOLDER:-$AMBER_ROOT/image}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

POLICY_JSON="${POLICY_JSON:-$CAL_ROOT/experiments/aligned_cheap_proxy_from_reference_vga/discovery/calibration_actual/selected_policy.json}"
ASSET_ROOT="${ASSET_ROOT:-$CAL_ROOT/experiments/amber_fixed_transfer_assets}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope}"

GENERATIVE_MAX_NEW_TOKENS="${GENERATIVE_MAX_NEW_TOKENS:-128}"
DISCRIMINATIVE_MAX_NEW_TOKENS="${DISCRIMINATIVE_MAX_NEW_TOKENS:-8}"

USE_ADD="${USE_ADD:-true}"
ATTN_COEF="${ATTN_COEF:-0.2}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
SAMPLING="${SAMPLING:-false}"
CD_ALPHA="${CD_ALPHA:-0.02}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"

REUSE_ASSETS="${REUSE_ASSETS:-true}"
REUSE_PREDS="${REUSE_PREDS:-true}"
REUSE_FEATURES="${REUSE_FEATURES:-true}"
LOG_EVERY="${LOG_EVERY:-25}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$PYTHON_BIN}"
VGA_CONDA_SH="${VGA_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VGA_ENV="${VGA_ENV:-vga}"
SMOKE="${SMOKE:-false}"
SMOKE_LIMIT_G="${SMOKE_LIMIT_G:-32}"
SMOKE_LIMIT_D="${SMOKE_LIMIT_D:-64}"

VGA_MODEL_BASE_ARGS=()
if [[ -n "${MODEL_BASE}" ]]; then
  VGA_MODEL_BASE_ARGS+=(--model-base "$MODEL_BASE")
fi

if [[ ! -d "$AMBER_ROOT" ]]; then
  echo "[error] AMBER_ROOT does not exist: $AMBER_ROOT" >&2
  exit 1
fi

if [[ ! -d "$VGA_ROOT" ]]; then
  echo "[error] VGA_ROOT does not exist: $VGA_ROOT" >&2
  exit 1
fi

if [[ ! -f "$VGA_ROOT/eval/object_hallucination_vqa_llava.py" ]]; then
  echo "[error] missing VGA entrypoint: $VGA_ROOT/eval/object_hallucination_vqa_llava.py" >&2
  exit 1
fi

if [[ ! -f "$POLICY_JSON" ]]; then
  echo "[error] POLICY_JSON does not exist: $POLICY_JSON" >&2
  exit 1
fi

mkdir -p "$ASSET_ROOT" "$OUT_ROOT/generative" "$OUT_ROOT/discriminative" "$OUT_ROOT/final"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

if [[ ! -f "$VGA_CONDA_SH" ]]; then
  echo "[error] missing conda.sh for VGA env: $VGA_CONDA_SH" >&2
  exit 1
fi

GEN_Q="$ASSET_ROOT/generative/assets/amber_generative_q.jsonl"
GEN_Q_OBJ="$ASSET_ROOT/generative/assets/amber_generative_q_with_object.jsonl"
DISC_Q="$ASSET_ROOT/discriminative/assets/amber_discriminative_q.jsonl"
DISC_Q_OBJ="$ASSET_ROOT/discriminative/assets/amber_discriminative_q_with_object.jsonl"
ALL_Q="$ASSET_ROOT/all/assets/amber_all_q.jsonl"

GEN_BASE="$OUT_ROOT/generative/pred_baseline.jsonl"
DISC_BASE="$OUT_ROOT/discriminative/pred_baseline.jsonl"
GEN_VGA="$OUT_ROOT/generative/pred_vga.jsonl"
DISC_VGA="$OUT_ROOT/discriminative/pred_vga.jsonl"
GEN_FEAT="$OUT_ROOT/generative/cheap_online_features.csv"
DISC_FEAT="$OUT_ROOT/discriminative/cheap_online_features.csv"

if [[ "$REUSE_ASSETS" != "true" || ! -f "$ALL_Q" ]]; then
  echo "[1/8] prepare AMBER assets"
  PYTHONPATH="$ROOT_DIR" \
  "$PYTHON_BIN" scripts/prepare_amber_fixed_transfer_assets.py \
    --amber_root "$AMBER_ROOT" \
    --out_dir "$ASSET_ROOT"
else
  echo "[1/8] reuse AMBER assets -> $ASSET_ROOT"
fi

if [[ "$SMOKE" == "true" ]]; then
  echo "[smoke] build reduced AMBER subset assets"
  SMOKE_ASSET_ROOT="$OUT_ROOT/smoke_assets"
  SMOKE_GEN_Q="$SMOKE_ASSET_ROOT/generative/assets/amber_generative_q.jsonl"
  SMOKE_GEN_Q_OBJ="$SMOKE_ASSET_ROOT/generative/assets/amber_generative_q_with_object.jsonl"
  SMOKE_DISC_Q="$SMOKE_ASSET_ROOT/discriminative/assets/amber_discriminative_q.jsonl"
  SMOKE_DISC_Q_OBJ="$SMOKE_ASSET_ROOT/discriminative/assets/amber_discriminative_q_with_object.jsonl"
  SMOKE_ALL_Q="$SMOKE_ASSET_ROOT/all/assets/amber_all_q.jsonl"
  mkdir -p "$(dirname "$SMOKE_GEN_Q")" "$(dirname "$SMOKE_DISC_Q")" "$(dirname "$SMOKE_ALL_Q")" "$OUT_ROOT/smoke/generative" "$OUT_ROOT/smoke/discriminative" "$OUT_ROOT/smoke/final"
  head -n "$SMOKE_LIMIT_G" "$GEN_Q" > "$SMOKE_GEN_Q"
  head -n "$SMOKE_LIMIT_G" "$GEN_Q_OBJ" > "$SMOKE_GEN_Q_OBJ"
  head -n "$SMOKE_LIMIT_D" "$DISC_Q" > "$SMOKE_DISC_Q"
  head -n "$SMOKE_LIMIT_D" "$DISC_Q_OBJ" > "$SMOKE_DISC_Q_OBJ"
  cat "$SMOKE_GEN_Q" "$SMOKE_DISC_Q" > "$SMOKE_ALL_Q"
  GEN_Q="$SMOKE_GEN_Q"
  GEN_Q_OBJ="$SMOKE_GEN_Q_OBJ"
  DISC_Q="$SMOKE_DISC_Q"
  DISC_Q_OBJ="$SMOKE_DISC_Q_OBJ"
  ALL_Q="$SMOKE_ALL_Q"
  OUT_ROOT="$OUT_ROOT/smoke"
  GEN_BASE="$OUT_ROOT/generative/pred_baseline.jsonl"
  DISC_BASE="$OUT_ROOT/discriminative/pred_baseline.jsonl"
  GEN_VGA="$OUT_ROOT/generative/pred_vga.jsonl"
  DISC_VGA="$OUT_ROOT/discriminative/pred_vga.jsonl"
  GEN_FEAT="$OUT_ROOT/generative/cheap_online_features.csv"
  DISC_FEAT="$OUT_ROOT/discriminative/cheap_online_features.csv"
  RUN_OFFICIAL_EVAL=false
else
  RUN_OFFICIAL_EVAL=true
fi

if [[ "$REUSE_PREDS" != "true" || ! -f "$GEN_BASE" ]]; then
  echo "[2/8] baseline generation on AMBER generative"
  PYTHONPATH="$ROOT_DIR" \
  "$PYTHON_BIN" -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --image-folder "$AMBER_IMAGE_FOLDER" \
    --question-file "$GEN_Q" \
    --answers-file "$GEN_BASE" \
    --conv-mode "$CONV_MODE" \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens "$GENERATIVE_MAX_NEW_TOKENS"
else
  echo "[2/8] reuse baseline generative preds -> $GEN_BASE"
fi

if [[ "$REUSE_PREDS" != "true" || ! -f "$DISC_BASE" ]]; then
  echo "[3/8] baseline generation on AMBER discriminative"
  PYTHONPATH="$ROOT_DIR" \
  "$PYTHON_BIN" -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --image-folder "$AMBER_IMAGE_FOLDER" \
    --question-file "$DISC_Q" \
    --answers-file "$DISC_BASE" \
    --conv-mode "$CONV_MODE" \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens "$DISCRIMINATIVE_MAX_NEW_TOKENS"
else
  echo "[3/8] reuse baseline discriminative preds -> $DISC_BASE"
fi

if [[ "$REUSE_PREDS" != "true" || ! -f "$GEN_VGA" ]]; then
  echo "[4/8] VGA generation on AMBER generative"
  (
    # shellcheck source=/dev/null
    source "$VGA_CONDA_SH"
    conda activate "$VGA_ENV"
    cd "$VGA_ROOT"
    python eval/object_hallucination_vqa_llava.py \
      --model-path "$MODEL_PATH" \
      "${VGA_MODEL_BASE_ARGS[@]}" \
      --image-folder "$AMBER_IMAGE_FOLDER" \
      --question-file "$GEN_Q_OBJ" \
      --answers-file "$GEN_VGA" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$GENERATIVE_MAX_NEW_TOKENS" \
      --use_add "$USE_ADD" \
      --attn_coef "$ATTN_COEF" \
      --head_balancing "$HEAD_BALANCING" \
      --sampling "$SAMPLING" \
      --cd_alpha "$CD_ALPHA" \
      --seed "$SEED" \
      --start_layer "$START_LAYER" \
      --end_layer "$END_LAYER"
  )
else
  echo "[4/8] reuse VGA generative preds -> $GEN_VGA"
fi

if [[ "$REUSE_PREDS" != "true" || ! -f "$DISC_VGA" ]]; then
  echo "[5/8] VGA generation on AMBER discriminative"
  (
    # shellcheck source=/dev/null
    source "$VGA_CONDA_SH"
    conda activate "$VGA_ENV"
    cd "$VGA_ROOT"
    python eval/object_hallucination_vqa_llava.py \
      --model-path "$MODEL_PATH" \
      "${VGA_MODEL_BASE_ARGS[@]}" \
      --image-folder "$AMBER_IMAGE_FOLDER" \
      --question-file "$DISC_Q_OBJ" \
      --answers-file "$DISC_VGA" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$DISCRIMINATIVE_MAX_NEW_TOKENS" \
      --use_add "$USE_ADD" \
      --attn_coef "$ATTN_COEF" \
      --head_balancing "$HEAD_BALANCING" \
      --sampling "$SAMPLING" \
      --cd_alpha "$CD_ALPHA" \
      --seed "$SEED" \
      --start_layer "$START_LAYER" \
      --end_layer "$END_LAYER"
  )
else
  echo "[5/8] reuse VGA discriminative preds -> $DISC_VGA"
fi

if [[ "$REUSE_FEATURES" != "true" || ! -f "$GEN_FEAT" ]]; then
  echo "[6/8] cheap feature extraction on AMBER generative"
  PYTHONPATH="$ROOT_DIR" \
  "$PYTHON_BIN" scripts/extract_c_stage_cheap_online_features.py \
    --question_file "$GEN_Q_OBJ" \
    --image_folder "$AMBER_IMAGE_FOLDER" \
    --intervention_pred_jsonl "$GEN_VGA" \
    --out_csv "$GEN_FEAT" \
    --out_summary_json "$OUT_ROOT/generative/cheap_online_features_summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --reuse_if_exists "$REUSE_FEATURES" \
    --log_every "$LOG_EVERY"
else
  echo "[6/8] reuse generative features -> $GEN_FEAT"
fi

if [[ "$REUSE_FEATURES" != "true" || ! -f "$DISC_FEAT" ]]; then
  echo "[7/8] cheap feature extraction on AMBER discriminative"
  PYTHONPATH="$ROOT_DIR" \
  "$PYTHON_BIN" scripts/extract_c_stage_cheap_online_features.py \
    --question_file "$DISC_Q_OBJ" \
    --image_folder "$AMBER_IMAGE_FOLDER" \
    --intervention_pred_jsonl "$DISC_VGA" \
    --out_csv "$DISC_FEAT" \
    --out_summary_json "$OUT_ROOT/discriminative/cheap_online_features_summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --reuse_if_exists "$REUSE_FEATURES" \
    --log_every "$LOG_EVERY"
else
  echo "[7/8] reuse discriminative features -> $DISC_FEAT"
fi

echo "[8/8] apply POPE-calibrated fixed transfer on AMBER and run official eval"
PYTHONPATH="$ROOT_DIR" \
"$PYTHON_BIN" scripts/apply_amber_fixed_transfer.py \
  --question_file_all "$ALL_Q" \
  --features_csv_generative "$GEN_FEAT" \
  --features_csv_discriminative "$DISC_FEAT" \
  --baseline_pred_jsonl_generative "$GEN_BASE" \
  --baseline_pred_jsonl_discriminative "$DISC_BASE" \
  --intervention_pred_jsonl_generative "$GEN_VGA" \
  --intervention_pred_jsonl_discriminative "$DISC_VGA" \
  --policy_json "$POLICY_JSON" \
  --amber_root "$AMBER_ROOT" \
  --out_dir "$OUT_ROOT/final" \
  --baseline_pred_text_key text \
  --intervention_pred_text_key output \
  --python_bin "$EVAL_PYTHON_BIN" \
  --run_official_eval "$RUN_OFFICIAL_EVAL"

echo "[done] AMBER fixed transfer -> $OUT_ROOT/final"
