#!/usr/bin/env bash
set -Eeuo pipefail

# Run LLaVA-1.5 POPE transfer experiments on A-OKVQA and GQA:
#   1) baseline
#   2) VGA
#   3) ours, using the frozen MSCOCO-discovery policy bundle/threshold
#
# Expected prepared dataset files:
#   $CAL_ROOT/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_q.jsonl
#   $CAL_ROOT/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_q_with_object.jsonl
#   $CAL_ROOT/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_gt.csv
#   $CAL_ROOT/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_q.jsonl
#   $CAL_ROOT/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_q_with_object.jsonl
#   $CAL_ROOT/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_gt.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-$PY_BIN}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-$PY_BIN}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

DATA_ROOT="${DATA_ROOT:-$CAL_ROOT/experiments/pope_hf_multidataset}"
DATASETS="${DATASETS:-aokvqa,gqa}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/paper_raw/pope_transfer_llava15_mscoco_policy}"

POLICY_BUNDLE_JSON="${POLICY_BUNDLE_JSON:-$CAL_ROOT/experiments/paper_main_meta_vga_full_strong/discovery/meta_calibration/selected_meta_bundle.json}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

LIMIT="${LIMIT:-0}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
LOG_EVERY="${LOG_EVERY:-25}"

RUN_BASELINE="${RUN_BASELINE:-true}"
RUN_VGA="${RUN_VGA:-true}"
RUN_OURS="${RUN_OURS:-true}"

# Keep this as cheap_c_only for the clean MSCOCO threshold-transfer experiment.
# Override to meta_strong only when explicitly testing the full B/C/fusion policy.
CONTROLLER_MODE="${CONTROLLER_MODE:-cheap_c_only}"
FEATURE_ORDER="${FEATURE_ORDER:-cheap_first}"
CHEAP_HIDDEN_FEATURES="${CHEAP_HIDDEN_FEATURES:-false}"
CHEAP_C_TAU_OVERRIDE="${CHEAP_C_TAU_OVERRIDE:-}"

VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_ATTN_NORM="${VGA_ATTN_NORM:-false}"
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
    aokvqa|a-okvqa|AOKVQA|A-OKVQA)
      DS_NAME="aokvqa"
      IMAGE_FOLDER="${AOKVQA_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
      Q_NOOBJ="${AOKVQA_Q_NOOBJ:-$DATA_ROOT/aokvqa/pope_aokvqa_9000_q.jsonl}"
      Q_WITHOBJ="${AOKVQA_Q_WITHOBJ:-$DATA_ROOT/aokvqa/pope_aokvqa_9000_q_with_object.jsonl}"
      GT_CSV="${AOKVQA_GT_CSV:-$DATA_ROOT/aokvqa/pope_aokvqa_9000_gt.csv}"
      ;;
    gqa|GQA)
      DS_NAME="gqa"
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
  VGA_PYTHON_BIN="$VGA_PYTHON_BIN" \
  VGA_ROOT="$VGA_ROOT" \
  GPU="$GPU" \
  BACKBONE=llava15 \
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
  VGA_USE_ADD="$VGA_USE_ADD" \
  VGA_ATTN_COEF="$VGA_ATTN_COEF" \
  VGA_CD_ALPHA="$VGA_CD_ALPHA" \
  VGA_HEAD_BALANCING="$VGA_HEAD_BALANCING" \
  VGA_START_LAYER="$VGA_START_LAYER" \
  VGA_END_LAYER="$VGA_END_LAYER" \
  VGA_SAMPLING="$VGA_SAMPLING" \
  VGA_ATTN_NORM="$VGA_ATTN_NORM" \
  SEED="$SEED" \
  bash "$CAL_ROOT/scripts/run_multibackbone_method_prediction.sh"
}

run_ours() {
  local baseline_pred="$1"
  local vga_pred="$2"
  local ours_out="$3"

  CLEANROOM_IMAGE_PREPROCESS_MODE=process_images \
  CLEANROOM_TF_FORWARD_MODE=model \
  CAL_ROOT="$CAL_ROOT" \
  PY_BIN="$PY_BIN" \
  GPU="$GPU" \
  DEVICE=cuda \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_BASE="$MODEL_BASE" \
  CONV_MODE="$CONV_MODE" \
  RUNTIME_BACKEND=llava15_cleanroom \
  QUESTION_FILE="$Q_WITHOBJ" \
  IMAGE_FOLDER="$IMAGE_FOLDER" \
  GT_CSV="$GT_CSV" \
  HEADSET_JSON="$HEADSET_JSON" \
  POLICY_BUNDLE_JSON="$POLICY_BUNDLE_JSON" \
  INTERVENTION_PRED_JSONL="$vga_pred" \
  INTERVENTION_PRED_KEY=output \
  BASELINE_PRED_JSONL="$baseline_pred" \
  BASELINE_PRED_KEY=text \
  OUT_DIR="$ours_out" \
  LIMIT="$LIMIT" \
  REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
  LOG_EVERY="$LOG_EVERY" \
  FEATURE_ORDER="$FEATURE_ORDER" \
  CONTROLLER_MODE="$CONTROLLER_MODE" \
  CHEAP_HIDDEN_FEATURES="$CHEAP_HIDDEN_FEATURES" \
  CHEAP_C_TAU_OVERRIDE="$CHEAP_C_TAU_OVERRIDE" \
  GENERATE_BASELINE_ON_FALLBACK=false \
  BASELINE_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  bash "$CAL_ROOT/scripts/run_discriminative_meta_strong_online.sh"
}

require_file "$POLICY_BUNDLE_JSON"
require_file "$HEADSET_JSON"

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
for raw_dataset in "${DATASET_LIST[@]}"; do
  dataset="$(echo "$raw_dataset" | xargs)"
  [[ -z "$dataset" ]] && continue

  dataset_config "$dataset"
  ensure_object_questions
  require_file "$Q_NOOBJ"
  require_file "$Q_WITHOBJ"
  require_file "$GT_CSV"

  ds_root="$OUT_ROOT/$DS_NAME/llava15_7b"
  baseline_out="$ds_root/baseline_full9000"
  vga_out="$ds_root/vga_full9000"
  ours_out="$ds_root/ours_mscoco_policy_${CONTROLLER_MODE}_full9000"

  echo "== dataset=$DS_NAME =="
  echo "[settings] image_folder=$IMAGE_FOLDER"
  echo "[settings] q_noobj=$Q_NOOBJ"
  echo "[settings] q_with_object=$Q_WITHOBJ"
  echo "[settings] gt_csv=$GT_CSV"
  echo "[settings] policy_bundle=$POLICY_BUNDLE_JSON"
  echo "[settings] controller_mode=$CONTROLLER_MODE"

  if truthy "$RUN_BASELINE"; then
    echo "[1/3] baseline -> $baseline_out"
    run_raw_method baseline "$baseline_out"
    eval_grouped "$baseline_out/pred_baseline.jsonl" text "$baseline_out/metrics_baseline_by_category.json"
  fi

  if truthy "$RUN_VGA"; then
    echo "[2/3] VGA -> $vga_out"
    run_raw_method vga "$vga_out"
    eval_grouped "$vga_out/pred_vga.jsonl" output "$vga_out/metrics_vga_by_category.json"
  fi

  if truthy "$RUN_OURS"; then
    require_file "$baseline_out/pred_baseline.jsonl"
    require_file "$vga_out/pred_vga.jsonl"
    echo "[3/3] ours frozen MSCOCO policy -> $ours_out"
    run_ours "$baseline_out/pred_baseline.jsonl" "$vga_out/pred_vga.jsonl" "$ours_out"
    eval_grouped "$ours_out/pred_meta_strong_online.jsonl" text "$ours_out/metrics_ours_by_category.json"
  fi

  echo "[done] dataset=$DS_NAME root=$ds_root"
done
