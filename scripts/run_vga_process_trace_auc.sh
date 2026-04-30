#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-$(pwd)}"
PY_BIN="${PY_BIN:-python}"
BACKBONE="${BACKBONE:-llava_next}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"

OUT_DIR="${OUT_DIR:-}"
QUESTION_FILE="${QUESTION_FILE:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"
INTERVENTION_PRED_JSONL="${INTERVENTION_PRED_JSONL:-}"
INTERVENTION_PRED_KEY="${INTERVENTION_PRED_KEY:-output}"
LABEL_ROWS_CSV="${LABEL_ROWS_CSV:-}"

MODEL_PATH="${MODEL_PATH:-}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_llama_3}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"

VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_ATTN_NORM="${VGA_ATTN_NORM:-false}"
VGA_TORCH_TYPE="${VGA_TORCH_TYPE:-fp16}"
VGA_ATTN_TYPE="${VGA_ATTN_TYPE:-sdpa}"
TRACE_COLLECT_ATTENTION_FEATURES="${TRACE_COLLECT_ATTENTION_FEATURES:-false}"
TRACE_COLLECT_LAYER_FEATURES="${TRACE_COLLECT_LAYER_FEATURES:-true}"
TRACE_LAYERS="${TRACE_LAYERS:-8,16,24,32}"

CANDIDATE_FILTER="${CANDIDATE_FILTER:-changed_answer}"
TOP_K="${TOP_K:-40}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing file: $path" >&2
    exit 2
  fi
}

if [[ -z "$OUT_DIR" || -z "$QUESTION_FILE" || -z "$IMAGE_FOLDER" || -z "$INTERVENTION_PRED_JSONL" || -z "$LABEL_ROWS_CSV" || -z "$MODEL_PATH" ]]; then
  cat >&2 <<'EOF'
[error] required env:
  OUT_DIR
  QUESTION_FILE
  IMAGE_FOLDER
  INTERVENTION_PRED_JSONL
  LABEL_ROWS_CSV
  MODEL_PATH
EOF
  exit 2
fi

require_file "$QUESTION_FILE"
require_file "$INTERVENTION_PRED_JSONL"
require_file "$LABEL_ROWS_CSV"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export DEVICE
export PYTHONPATH="$CAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

FEATURES_CSV="$OUT_DIR/process_trace_features.csv"
STEPS_CSV="$OUT_DIR/process_trace_steps.csv"
TRACE_SUMMARY_JSON="$OUT_DIR/process_trace_summary.json"
AUC_CSV="$OUT_DIR/process_trace_auc.csv"
AUC_JSON="$OUT_DIR/process_trace_auc.json"

echo "[trace] backbone=$BACKBONE"
echo "[trace] model=$MODEL_PATH"
echo "[trace] question=$QUESTION_FILE"
echo "[trace] pred=$INTERVENTION_PRED_JSONL key=$INTERVENTION_PRED_KEY"
echo "[trace] labels=$LABEL_ROWS_CSV"
echo "[trace] out=$OUT_DIR"
echo "[trace] attention_features=$TRACE_COLLECT_ATTENTION_FEATURES layer_features=$TRACE_COLLECT_LAYER_FEATURES layers=$TRACE_LAYERS"

if [[ "$BACKBONE" == "llava_next" ]]; then
  model_base_args=()
  if [[ -n "$MODEL_BASE" ]]; then
    model_base_args+=(--model-base "$MODEL_BASE")
  fi
  "$PY_BIN" "$CAL_ROOT/scripts/extract_vga_next_intervention_process_features.py" \
    --vga-root "$VGA_ROOT" \
    --model-path "$MODEL_PATH" \
    "${model_base_args[@]}" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --intervention-pred-jsonl "$INTERVENTION_PRED_JSONL" \
    --pred-text-key "$INTERVENTION_PRED_KEY" \
    --label-rows-csv "$LABEL_ROWS_CSV" \
    --limit "$LIMIT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --conv-mode "$CONV_MODE" \
    --use-add "$VGA_USE_ADD" \
    --cd-alpha "$VGA_CD_ALPHA" \
    --attn-coef "$VGA_ATTN_COEF" \
    --start-layer "$VGA_START_LAYER" \
    --end-layer "$VGA_END_LAYER" \
    --head-balancing "$VGA_HEAD_BALANCING" \
    --attn-norm "$VGA_ATTN_NORM" \
    --torch-type "$VGA_TORCH_TYPE" \
    --attn-type "$VGA_ATTN_TYPE" \
    --collect-attention-features "$TRACE_COLLECT_ATTENTION_FEATURES" \
    --collect-layer-features "$TRACE_COLLECT_LAYER_FEATURES" \
    --trace-layers "$TRACE_LAYERS" \
    --out-steps-csv "$STEPS_CSV" \
    --out-features-csv "$FEATURES_CSV" \
    --out-summary-json "$TRACE_SUMMARY_JSON"
else
  echo "[error] BACKBONE=$BACKBONE is not implemented in this wrapper yet. Use llava_next for the current NeXT trace test." >&2
  exit 2
fi

"$PY_BIN" "$CAL_ROOT/scripts/diagnose_changed_feature_auc.py" \
  --rows_csv "$FEATURES_CSV" \
  --candidate_filter "$CANDIDATE_FILTER" \
  --feature_prefixes proc_ \
  --top_k "$TOP_K" \
  --out_csv "$AUC_CSV" \
  --out_json "$AUC_JSON"

echo "[done] features: $FEATURES_CSV"
echo "[done] auc: $AUC_CSV"
