#!/usr/bin/env bash
set -Eeuo pipefail

# Run LLaVA-NeXT POPE experiments on MSCOCO, A-OKVQA, and GQA:
#   1) baseline with the official LLaVA-NeXT runner
#   2) VGA with the LLaVA-NeXT VGA compatibility runner
#
# Expected prepared dataset files:
#   $CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl
#   $CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl
#   $CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv
#   $CAL_ROOT/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_q.jsonl
#   $CAL_ROOT/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_q_with_object.jsonl
#   $CAL_ROOT/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_gt.csv
#   $CAL_ROOT/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_q.jsonl
#   $CAL_ROOT/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_q_with_object.jsonl
#   $CAL_ROOT/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_gt.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/llava_next_official/bin/python}"
LLAVA_NEXT_PYTHON_BIN="${LLAVA_NEXT_PYTHON_BIN:-$CAL_PYTHON_BIN}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-$CAL_PYTHON_BIN}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

MODEL_PATH="${MODEL_PATH:-/home/kms/models/llama3-llava-next-8b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_llama_3}"
BACKBONE_TAG="${BACKBONE_TAG:-llava_next_llama3_8b}"

DATA_ROOT="${DATA_ROOT:-$CAL_ROOT/experiments/pope_hf_multidataset}"
DATASETS="${DATASETS:-mscoco,aokvqa,gqa}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/paper_raw/pope}"

LIMIT="${LIMIT:-0}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

RUN_BASELINE="${RUN_BASELINE:-true}"
RUN_VGA="${RUN_VGA:-true}"

LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-/home/kms/LLaVA-NeXT}"
LLAVA_NEXT_TORCH_TYPE="${LLAVA_NEXT_TORCH_TYPE:-fp16}"
LLAVA_NEXT_ATTN_IMPLEMENTATION="${LLAVA_NEXT_ATTN_IMPLEMENTATION:-sdpa}"
LLAVA_NEXT_DO_SAMPLE="${LLAVA_NEXT_DO_SAMPLE:-false}"
LLAVA_NEXT_TEMPERATURE="${LLAVA_NEXT_TEMPERATURE:-0.0}"
LLAVA_NEXT_NUM_BEAMS="${LLAVA_NEXT_NUM_BEAMS:-1}"

VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_START_LAYER="${VGA_START_LAYER:-0}"
VGA_END_LAYER="${VGA_END_LAYER:-16}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_ATTN_NORM="${VGA_ATTN_NORM:-false}"
VGA_TORCH_TYPE="${VGA_TORCH_TYPE:-fp16}"
VGA_ATTN_TYPE="${VGA_ATTN_TYPE:-sdpa}"
SEED="${SEED:-17}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing file: $path" >&2
    exit 2
  fi
}

truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

dataset_config() {
  local dataset="$1"
  case "$dataset" in
    mscoco|coco|MSCOCO|COCO)
      DS_NAME="mscoco"
      DS_OUT_NAME="${MSCOCO_OUT_NAME:-llava_next_llama3_8b}"
      IMAGE_FOLDER="${MSCOCO_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
      Q_NOOBJ="${MSCOCO_Q_NOOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl}"
      Q_WITHOBJ="${MSCOCO_Q_WITHOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
      GT_CSV="${MSCOCO_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
      ;;
    aokvqa|a-okvqa|AOKVQA|A-OKVQA)
      DS_NAME="aokvqa"
      DS_OUT_NAME="$DS_NAME/$BACKBONE_TAG"
      IMAGE_FOLDER="${AOKVQA_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
      Q_NOOBJ="${AOKVQA_Q_NOOBJ:-$DATA_ROOT/aokvqa/pope_aokvqa_9000_q.jsonl}"
      Q_WITHOBJ="${AOKVQA_Q_WITHOBJ:-$DATA_ROOT/aokvqa/pope_aokvqa_9000_q_with_object.jsonl}"
      GT_CSV="${AOKVQA_GT_CSV:-$DATA_ROOT/aokvqa/pope_aokvqa_9000_gt.csv}"
      ;;
    gqa|GQA)
      DS_NAME="gqa"
      DS_OUT_NAME="$DS_NAME/$BACKBONE_TAG"
      IMAGE_FOLDER="${GQA_IMAGE_FOLDER:-/home/kms/data/GQA}"
      Q_NOOBJ="${GQA_Q_NOOBJ:-$DATA_ROOT/gqa/pope_gqa_9000_q.jsonl}"
      Q_WITHOBJ="${GQA_Q_WITHOBJ:-$DATA_ROOT/gqa/pope_gqa_9000_q_with_object.jsonl}"
      GT_CSV="${GQA_GT_CSV:-$DATA_ROOT/gqa/pope_gqa_9000_gt.csv}"
      ;;
    *)
      echo "[error] unsupported dataset: $dataset" >&2
      exit 2
      ;;
  esac
}

ensure_object_questions() {
  if [[ -f "$Q_WITHOBJ" ]]; then
    return
  fi
  require_file "$Q_NOOBJ"
  echo "[prep] materialize object questions: $Q_WITHOBJ"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/add_object_field_to_pope_jsonl.py" \
    --input_jsonl "$Q_NOOBJ" \
    --output_jsonl "$Q_WITHOBJ"
}

eval_grouped() {
  local pred_jsonl="$1"
  local pred_key="$2"
  local out_json="$3"
  local out_csv="${out_json%.json}.csv"

  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$pred_jsonl" \
    --pred_text_key "$pred_key" \
    --group_col category \
    --out_json "$out_json" \
    --out_csv "$out_csv"
}

run_raw_method() {
  local method="$1"
  local method_out="$2"

  CAL_ROOT="$CAL_ROOT" \
  CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
  LLAVA_NEXT_PYTHON_BIN="$LLAVA_NEXT_PYTHON_BIN" \
  VGA_PYTHON_BIN="$VGA_PYTHON_BIN" \
  LLAVA_NEXT_ROOT="$LLAVA_NEXT_ROOT" \
  VGA_ROOT="$VGA_ROOT" \
  GPU="$GPU" \
  BACKBONE=llava_next \
  METHOD="$method" \
  TASK=pope \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_BASE="$MODEL_BASE" \
  CONV_MODE="$CONV_MODE" \
  IMAGE_FOLDER="$IMAGE_FOLDER" \
  Q_NOOBJ="$Q_NOOBJ" \
  Q_WITHOBJ="$Q_WITHOBJ" \
  GT_CSV="$GT_CSV" \
  OUT_ROOT="$method_out" \
  LIMIT="$LIMIT" \
  REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  LLAVA_NEXT_TORCH_TYPE="$LLAVA_NEXT_TORCH_TYPE" \
  LLAVA_NEXT_ATTN_IMPLEMENTATION="$LLAVA_NEXT_ATTN_IMPLEMENTATION" \
  LLAVA_NEXT_DO_SAMPLE="$LLAVA_NEXT_DO_SAMPLE" \
  LLAVA_NEXT_TEMPERATURE="$LLAVA_NEXT_TEMPERATURE" \
  LLAVA_NEXT_NUM_BEAMS="$LLAVA_NEXT_NUM_BEAMS" \
  VGA_USE_ADD="$VGA_USE_ADD" \
  VGA_ATTN_COEF="$VGA_ATTN_COEF" \
  VGA_CD_ALPHA="$VGA_CD_ALPHA" \
  VGA_HEAD_BALANCING="$VGA_HEAD_BALANCING" \
  VGA_START_LAYER="$VGA_START_LAYER" \
  VGA_END_LAYER="$VGA_END_LAYER" \
  VGA_SAMPLING="$VGA_SAMPLING" \
  VGA_ATTN_NORM="$VGA_ATTN_NORM" \
  VGA_TORCH_TYPE="$VGA_TORCH_TYPE" \
  VGA_ATTN_TYPE="$VGA_ATTN_TYPE" \
  SEED="$SEED" \
  bash "$CAL_ROOT/scripts/run_multibackbone_method_prediction.sh"
}

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
for raw_dataset in "${DATASET_LIST[@]}"; do
  dataset="$(echo "$raw_dataset" | xargs)"
  [[ -z "$dataset" ]] && continue

  dataset_config "$dataset"
  ensure_object_questions
  require_file "$Q_NOOBJ"
  require_file "$Q_WITHOBJ"
  require_file "$GT_CSV"

  ds_root="$OUT_ROOT/$DS_OUT_NAME"
  baseline_out="$ds_root/baseline_${LLAVA_NEXT_ATTN_IMPLEMENTATION}_tok${MAX_NEW_TOKENS}_full9000"
  vga_out="$ds_root/vga_${VGA_ATTN_TYPE}_tok${MAX_NEW_TOKENS}_layers${VGA_START_LAYER}_${VGA_END_LAYER}_full9000"

  echo "== dataset=$DS_NAME backbone=$BACKBONE_TAG =="
  echo "[settings] model_path=$MODEL_PATH"
  echo "[settings] image_folder=$IMAGE_FOLDER"
  echo "[settings] q_noobj=$Q_NOOBJ"
  echo "[settings] q_with_object=$Q_WITHOBJ"
  echo "[settings] gt_csv=$GT_CSV"
  echo "[settings] baseline_attn=$LLAVA_NEXT_ATTN_IMPLEMENTATION vga_attn=$VGA_ATTN_TYPE"
  echo "[settings] vga_layers=${VGA_START_LAYER}-${VGA_END_LAYER}"

  if truthy "$RUN_BASELINE"; then
    echo "[1/2] baseline -> $baseline_out"
    run_raw_method baseline "$baseline_out"
    eval_grouped "$baseline_out/pred_baseline.jsonl" output "$baseline_out/metrics_baseline_by_category.json"
  fi

  if truthy "$RUN_VGA"; then
    echo "[2/2] VGA -> $vga_out"
    run_raw_method vga "$vga_out"
    eval_grouped "$vga_out/pred_vga.jsonl" output "$vga_out/metrics_vga_by_category.json"
  fi

  echo "[done] dataset=$DS_NAME root=$ds_root"
done
