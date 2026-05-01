#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/llava_next_official/bin/python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

MODEL_PATH="${MODEL_PATH:-/home/kms/models/llama3-llava-next-8b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_llama_3}"
LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-/home/kms/LLaVA-NeXT}"
LLAVA_NEXT_TORCH_TYPE="${LLAVA_NEXT_TORCH_TYPE:-fp16}"
LLAVA_NEXT_ATTN_IMPLEMENTATION="${LLAVA_NEXT_ATTN_IMPLEMENTATION:-sdpa}"

QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/paper_pcp_cd_tokfix_newline/llava_next/discovery/changed_subset/changed_q_with_object.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
INTERVENTION_PRED_JSONL="${INTERVENTION_PRED_JSONL:-$CAL_ROOT/experiments/paper_pcp_cd/llava_next/discovery/raw/vga/pred_vga.jsonl}"
INTERVENTION_PRED_KEY="${INTERVENTION_PRED_KEY:-output}"
LABEL_ROWS_CSV="${LABEL_ROWS_CSV:-$CAL_ROOT/experiments/paper_pcp_cd_tokfix_newline/llava_next/discovery/features_hidden_changed/online_feature_rows.csv}"
OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/paper_pcp_cd_tokfix_newline/llava_next/discovery/vision_ablation_replay_changed}"

ABLATION_MODES="${ABLATION_MODES:-black,gray,blur}"
BLUR_RADIUS="${BLUR_RADIUS:-32}"
OUTPUT_HIDDEN_STATES="${OUTPUT_HIDDEN_STATES:-true}"
LIMIT="${LIMIT:-0}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-false}"
LOG_EVERY="${LOG_EVERY:-25}"
TOP_K="${TOP_K:-80}"

mkdir -p "$OUT_DIR"

echo "[vision-ablation] question_file=$QUESTION_FILE"
echo "[vision-ablation] label_rows=$LABEL_ROWS_CSV"
echo "[vision-ablation] modes=$ABLATION_MODES out=$OUT_DIR"

"$PY_BIN" "$CAL_ROOT/scripts/extract_llava_next_vision_ablation_replay_features.py" \
  --question_file "$QUESTION_FILE" \
  --image_folder "$IMAGE_FOLDER" \
  --intervention_pred_jsonl "$INTERVENTION_PRED_JSONL" \
  --intervention_pred_key "$INTERVENTION_PRED_KEY" \
  --label_rows_csv "$LABEL_ROWS_CSV" \
  --out_dir "$OUT_DIR" \
  --llava_next_root "$LLAVA_NEXT_ROOT" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --conv_mode "$CONV_MODE" \
  --device cuda \
  --llava_next_torch_type "$LLAVA_NEXT_TORCH_TYPE" \
  --llava_next_attn_implementation "$LLAVA_NEXT_ATTN_IMPLEMENTATION" \
  --ablation_modes "$ABLATION_MODES" \
  --blur_radius "$BLUR_RADIUS" \
  --output_hidden_states "$OUTPUT_HIDDEN_STATES" \
  --limit "$LIMIT" \
  --reuse_if_exists "$REUSE_IF_EXISTS" \
  --log_every "$LOG_EVERY"

"$PY_BIN" "$CAL_ROOT/scripts/diagnose_changed_feature_auc.py" \
  --rows_csv "$OUT_DIR/vision_ablation_replay_rows.csv" \
  --out_csv "$OUT_DIR/vision_ablation_replay_auc.csv" \
  --out_json "$OUT_DIR/vision_ablation_replay_auc.json" \
  --candidate_filter changed_answer \
  --feature_prefixes abl_ \
  --top_k "$TOP_K"

echo "[done] rows=$OUT_DIR/vision_ablation_replay_rows.csv"
echo "[done] auc=$OUT_DIR/vision_ablation_replay_auc.csv"
