#!/usr/bin/env bash
set -euo pipefail

# CHAIR generative panel:
#   raw method captions for VGA / PAI-attn / VAF across backbones
#   -> validation-calibrated object-token suppression on each raw method source
#   -> summary table.
#
# LLaVA-1.5 + VGA is already available in the v59 source by default, so this
# wrapper defaults to the remaining targets. Set TARGETS to override.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
LLAVA_NEXT_PYTHON_BIN="${LLAVA_NEXT_PYTHON_BIN:-$VGA_PYTHON_BIN}"
QWEN25_PYTHON_BIN="${QWEN25_PYTHON_BIN:-$VGA_PYTHON_BIN}"
PAI_PYTHON_BIN="${PAI_PYTHON_BIN:-/home/kms/miniconda3/envs/pai_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
PAI_ROOT="${PAI_ROOT:-/home/kms/PAI}"
LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-/home/kms/LLaVA-NeXT}"
CLEARSIGHT_ROOT="${CLEARSIGHT_ROOT:-$CAL_ROOT/ClearSight}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_multibackbone_method_ours_panel}"
SOURCE_SPLIT_ROOT="${SOURCE_SPLIT_ROOT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
RAW_ROOT="${RAW_ROOT:-$OUT_ROOT/raw_sources}"
OURS_ROOT="${OURS_ROOT:-$OUT_ROOT/ours}"
SUMMARY_DIR="${SUMMARY_DIR:-$OUT_ROOT/summary}"

TARGETS="${TARGETS:-llava15_pai_attn llava15_vaf llava_next_vga llava_next_pai_attn llava_next_vaf qwen25_vga qwen25_pai_attn qwen25_vaf}"
SPLITS="${SPLITS:-val test}"

RAW_GPU="${RAW_GPU:-${GPU:-0}}"
OURS_GPU="${OURS_GPU:-${GPU:-0}}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/images/mscoco/annotations}"
CHAIR_CACHE="${CHAIR_CACHE:-$SOURCE_SPLIT_ROOT/chair_cache.pkl}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"
LIMIT="${LIMIT:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEED="${SEED:-17}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

RUN_RAW="${RUN_RAW:-true}"
RUN_BASELINES="${RUN_BASELINES:-true}"
RUN_OURS="${RUN_OURS:-true}"
RUN_SUMMARY="${RUN_SUMMARY:-true}"
BASELINE_BACKBONES="${BASELINE_BACKBONES:-llava_next qwen25}"

LLAVA15_MODEL="${LLAVA15_MODEL:-liuhaotian/llava-v1.5-7b}"
LLAVA_NEXT_MODEL="${LLAVA_NEXT_MODEL:-/home/kms/models/llama3-llava-next-8b}"
QWEN25_MODEL="${QWEN25_MODEL:-/home/kms/models/Qwen2.5-VL-7B-Instruct}"

LLAVA_NEXT_TORCH_TYPE="${LLAVA_NEXT_TORCH_TYPE:-bf16}"
LLAVA_NEXT_ATTN_IMPLEMENTATION="${LLAVA_NEXT_ATTN_IMPLEMENTATION:-sdpa}"
QWEN25_TORCH_TYPE="${QWEN25_TORCH_TYPE:-bf16}"
QWEN25_ATTN_TYPE="${QWEN25_ATTN_TYPE:-eager}"
QWEN25_DEVICE_MAP="${QWEN25_DEVICE_MAP:-cuda}"

THRESHOLDS="${THRESHOLDS:-0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60}"
RISK_SCORE_MODE="${RISK_SCORE_MODE:-next_token_yesno}"
RISK_OBJECT_VOCAB="${RISK_OBJECT_VOCAB:-coco80}"
RISK_FILTER_TO_VOCAB="${RISK_FILTER_TO_VOCAB:-true}"
RISK_MAX_OBJECTS="${RISK_MAX_OBJECTS:-8}"
SUPPRESS_MODE="${SUPPRESS_MODE:-first_token}"
SUPPRESS_BIAS="${SUPPRESS_BIAS:--1.0}"
SELECT_OBJECTIVE="${SELECT_OBJECTIVE:-chairi_then_f1}"
SELECT_MAX_RECALL_DROP="${SELECT_MAX_RECALL_DROP:-0.005}"
SELECT_MIN_DELTA_F1="${SELECT_MIN_DELTA_F1:--1.0}"
SELECT_MAX_DELTA_CHAIR_I="${SELECT_MAX_DELTA_CHAIR_I:-0.0}"
SELECT_MAX_DELTA_CHAIR_S="${SELECT_MAX_DELTA_CHAIR_S:-0.0}"

LLAVA15_VGA_SOURCE="${LLAVA15_VGA_SOURCE:-$SOURCE_SPLIT_ROOT}"
LLAVA15_BASELINE_CHAIR="${LLAVA15_BASELINE_CHAIR:-$LLAVA15_VGA_SOURCE/test/chair_baseline.json}"
LLAVA15_VGA_RAW_CHAIR="${LLAVA15_VGA_RAW_CHAIR:-$LLAVA15_VGA_SOURCE/test/chair_origin_entropy_simg.json}"
LLAVA15_VGA_OURS_CSV="${LLAVA15_VGA_OURS_CSV:-$CAL_ROOT/experiments/rapic_generative_v84_valcalib_vga_token_suppression/test_apply_next_token_yesno_yp0.6/summary/chair_v82_object_token_suppression_max8_vocab_first_token_bias-1.0_yp0.6.csv}"
INCLUDE_EXISTING_LLAVA15_VGA_IN_SUMMARY="${INCLUDE_EXISTING_LLAVA15_VGA_IN_SUMMARY:-true}"

mkdir -p "$RAW_ROOT" "$OURS_ROOT" "$SUMMARY_DIR"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing file: $path" >&2
    exit 2
  fi
}

copy_limited_split() {
  local split="$1"
  local out_dir="$2"
  mkdir -p "$out_dir/splits"
  local src="$SOURCE_SPLIT_ROOT/splits/${split}_caption_q_limited${SOURCE_LIMIT}.jsonl"
  if [[ ! -f "$src" ]]; then
    src="$SOURCE_SPLIT_ROOT/splits/${split}_caption_q.jsonl"
  fi
  require_file "$src"
  local dst="$out_dir/splits/${split}_caption_q_limited${SOURCE_LIMIT}.jsonl"
  if reuse_file "$dst"; then
    echo "[reuse] $dst"
    return
  fi
  "$CAL_PYTHON_BIN" - "$src" "$dst" "$SOURCE_LIMIT" <<'PY'
import json
import os
import sys

src, dst, limit_s = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(limit_s)
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
n = 0
with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue
        json.loads(line)
        g.write(line)
        n += 1
        if limit > 0 and n >= limit:
            break
print(f"[saved] {dst} n={n}")
PY
}

configure_target() {
  local target="$1"
  BACKBONE=""
  METHOD=""
  PRED_BASENAME=""
  MODEL_PATH=""
  CONV_MODE=""
  EXTRA_ENV=()
  case "$target" in
    llava15_vga)
      BACKBONE=llava15
      METHOD=vga
      PRED_BASENAME=pred_vga_caption.jsonl
      MODEL_PATH="$LLAVA15_MODEL"
      CONV_MODE=llava_v1
      ;;
    llava15_pai_attn)
      BACKBONE=llava15
      METHOD=pai_attn
      PRED_BASENAME=pred_pai_attn_caption.jsonl
      MODEL_PATH="$LLAVA15_MODEL"
      CONV_MODE=llava_v1
      EXTRA_ENV=(PAI_USE_ATTN=1 PAI_USE_CFG=0 PAI_START_LAYER=2 PAI_END_LAYER=15)
      ;;
    llava15_vaf)
      BACKBONE=llava15
      METHOD=vaf
      PRED_BASENAME=pred_vaf_caption.jsonl
      MODEL_PATH="$LLAVA15_MODEL"
      CONV_MODE=llava_v1
      EXTRA_ENV=(VAF_START_LAYER=9 VAF_END_LAYER=14)
      ;;
    llava_next_vga)
      BACKBONE=llava_next
      METHOD=vga
      PRED_BASENAME=pred_vga_caption.jsonl
      MODEL_PATH="$LLAVA_NEXT_MODEL"
      CONV_MODE=llava_llama_3
      EXTRA_ENV=(LLAVA_NEXT_TORCH_TYPE="$LLAVA_NEXT_TORCH_TYPE" LLAVA_NEXT_ATTN_IMPLEMENTATION="$LLAVA_NEXT_ATTN_IMPLEMENTATION" VGA_TORCH_TYPE="$LLAVA_NEXT_TORCH_TYPE" VGA_ATTN_TYPE="$LLAVA_NEXT_ATTN_IMPLEMENTATION")
      ;;
    llava_next_pai_attn)
      BACKBONE=llava_next
      METHOD=pai_attn
      PRED_BASENAME=pred_pai_attn_caption.jsonl
      MODEL_PATH="$LLAVA_NEXT_MODEL"
      CONV_MODE=llava_llama_3
      EXTRA_ENV=(LLAVA_NEXT_TORCH_TYPE="$LLAVA_NEXT_TORCH_TYPE" LLAVA_NEXT_ATTN_IMPLEMENTATION="$LLAVA_NEXT_ATTN_IMPLEMENTATION" PAI_USE_ATTN=1 PAI_USE_CFG=0 PAI_START_LAYER=0 PAI_END_LAYER=16)
      ;;
    llava_next_vaf)
      BACKBONE=llava_next
      METHOD=vaf
      PRED_BASENAME=pred_vaf_caption.jsonl
      MODEL_PATH="$LLAVA_NEXT_MODEL"
      CONV_MODE=llava_llama_3
      EXTRA_ENV=(LLAVA_NEXT_TORCH_TYPE="$LLAVA_NEXT_TORCH_TYPE" LLAVA_NEXT_ATTN_IMPLEMENTATION="$LLAVA_NEXT_ATTN_IMPLEMENTATION" VAF_START_LAYER=9 VAF_END_LAYER=14)
      ;;
    qwen25_vga)
      BACKBONE=qwen25_vl
      METHOD=vga
      PRED_BASENAME=pred_vga_caption.jsonl
      MODEL_PATH="$QWEN25_MODEL"
      CONV_MODE=llava_v1
      EXTRA_ENV=(VGA_TORCH_TYPE="$QWEN25_TORCH_TYPE" VGA_ATTN_TYPE="$QWEN25_ATTN_TYPE" QWEN25_DEVICE_MAP="$QWEN25_DEVICE_MAP" QWEN25_MODEL_BACKEND=official)
      ;;
    qwen25_pai_attn)
      BACKBONE=qwen25_vl
      METHOD=pai_attn
      PRED_BASENAME=pred_pai_attn_caption.jsonl
      MODEL_PATH="$QWEN25_MODEL"
      CONV_MODE=llava_v1
      EXTRA_ENV=(VGA_TORCH_TYPE="$QWEN25_TORCH_TYPE" QWEN25_DEVICE_MAP="$QWEN25_DEVICE_MAP" PAI_USE_ATTN=1 PAI_USE_CFG=0 PAI_START_LAYER=4 PAI_END_LAYER=16)
      ;;
    qwen25_vaf)
      BACKBONE=qwen25_vl
      METHOD=vaf
      PRED_BASENAME=pred_vaf_caption.jsonl
      MODEL_PATH="$QWEN25_MODEL"
      CONV_MODE=llava_v1
      EXTRA_ENV=(VGA_TORCH_TYPE="$QWEN25_TORCH_TYPE" QWEN25_DEVICE_MAP="$QWEN25_DEVICE_MAP" VAF_START_LAYER=9 VAF_END_LAYER=14)
      ;;
    *)
      echo "[error] unknown target: $target" >&2
      exit 2
      ;;
  esac
}

configure_baseline() {
  local backbone_key="$1"
  BACKBONE=""
  METHOD=baseline
  PRED_BASENAME=pred_baseline_caption.jsonl
  MODEL_PATH=""
  CONV_MODE=""
  EXTRA_ENV=()
  case "$backbone_key" in
    llava15)
      BACKBONE=llava15
      MODEL_PATH="$LLAVA15_MODEL"
      CONV_MODE=llava_v1
      ;;
    llava_next)
      BACKBONE=llava_next
      MODEL_PATH="$LLAVA_NEXT_MODEL"
      CONV_MODE=llava_llama_3
      EXTRA_ENV=(LLAVA_NEXT_TORCH_TYPE="$LLAVA_NEXT_TORCH_TYPE" LLAVA_NEXT_ATTN_IMPLEMENTATION="$LLAVA_NEXT_ATTN_IMPLEMENTATION")
      ;;
    qwen25)
      BACKBONE=qwen25_vl
      MODEL_PATH="$QWEN25_MODEL"
      CONV_MODE=llava_v1
      EXTRA_ENV=(VGA_TORCH_TYPE="$QWEN25_TORCH_TYPE" VGA_ATTN_TYPE="$QWEN25_ATTN_TYPE" QWEN25_DEVICE_MAP="$QWEN25_DEVICE_MAP" QWEN25_MODEL_BACKEND=official)
      ;;
    *)
      echo "[error] unknown baseline backbone: $backbone_key" >&2
      exit 2
      ;;
  esac
}

run_raw_baseline_split() {
  local backbone_key="$1"
  local split="$2"
  configure_baseline "$backbone_key"

  local target="baseline_${backbone_key}"
  local source_dir="$RAW_ROOT/$target"
  copy_limited_split "$split" "$source_dir"
  mkdir -p "$source_dir/$split"
  local q="$source_dir/splits/${split}_caption_q_limited${SOURCE_LIMIT}.jsonl"
  local pred="$source_dir/$split/$PRED_BASENAME"
  local chair="$source_dir/$split/chair_${target}.json"
  local chair_input="$source_dir/$split/chair_input_${target}.jsonl"
  local log="$source_dir/$split/run_${target}.log"

  if reuse_file "$chair"; then
    echo "[reuse] $chair"
    return
  fi

  echo "== raw $target/$split -> $pred"
  env \
    GPU="$RAW_GPU" \
    CUDA_VISIBLE_DEVICES="$RAW_GPU" \
    PYTHONPATH="$CAL_ROOT:${PYTHONPATH:-}" \
    CAL_ROOT="$CAL_ROOT" \
    VGA_ROOT="$VGA_ROOT" \
    PAI_ROOT="$PAI_ROOT" \
    LLAVA_NEXT_ROOT="$LLAVA_NEXT_ROOT" \
    CLEARSIGHT_ROOT="$CLEARSIGHT_ROOT" \
    EAZY_ROOT="$EAZY_ROOT" \
    CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
    VGA_PYTHON_BIN="$VGA_PYTHON_BIN" \
    LLAVA_NEXT_PYTHON_BIN="$LLAVA_NEXT_PYTHON_BIN" \
    QWEN25_PYTHON_BIN="$QWEN25_PYTHON_BIN" \
    PAI_PYTHON_BIN="$PAI_PYTHON_BIN" \
    EAZY_PYTHON_BIN="$EAZY_PYTHON_BIN" \
    TASK=chair \
    BACKBONE="$BACKBONE" \
    METHOD=baseline \
    OUT_ROOT="$source_dir/$split" \
    MODEL_PATH="$MODEL_PATH" \
    IMAGE_FOLDER="$IMAGE_FOLDER" \
    QUESTION_FILE="$q" \
    PRED_JSONL="$pred" \
    METRICS_JSON="$chair" \
    CHAIR_INPUT_JSONL="$chair_input" \
    COCO_ANN_ROOT="$COCO_ANN_ROOT" \
    CHAIR_CACHE="$CHAIR_CACHE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    LIMIT=0 \
    CONV_MODE="$CONV_MODE" \
    SEED="$SEED" \
    REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
    "${EXTRA_ENV[@]}" \
    bash "$CAL_ROOT/scripts/run_multibackbone_method_prediction.sh" \
    2>&1 | tee "$log"
}

run_raw_target_split() {
  local target="$1"
  local split="$2"
  configure_target "$target"

  local source_dir="$RAW_ROOT/$target"
  copy_limited_split "$split" "$source_dir"
  mkdir -p "$source_dir/$split"
  local q="$source_dir/splits/${split}_caption_q_limited${SOURCE_LIMIT}.jsonl"
  local pred="$source_dir/$split/$PRED_BASENAME"
  local chair="$source_dir/$split/chair_${target}.json"
  local chair_input="$source_dir/$split/chair_input_${target}.jsonl"
  local log="$source_dir/$split/run_${target}.log"

  if reuse_file "$chair"; then
    echo "[reuse] $chair"
    return
  fi

  echo "== raw $target/$split -> $pred"
  env \
    GPU="$RAW_GPU" \
    CUDA_VISIBLE_DEVICES="$RAW_GPU" \
    PYTHONPATH="$CAL_ROOT:${PYTHONPATH:-}" \
    CAL_ROOT="$CAL_ROOT" \
    VGA_ROOT="$VGA_ROOT" \
    PAI_ROOT="$PAI_ROOT" \
    LLAVA_NEXT_ROOT="$LLAVA_NEXT_ROOT" \
    CLEARSIGHT_ROOT="$CLEARSIGHT_ROOT" \
    EAZY_ROOT="$EAZY_ROOT" \
    CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
    VGA_PYTHON_BIN="$VGA_PYTHON_BIN" \
    LLAVA_NEXT_PYTHON_BIN="$LLAVA_NEXT_PYTHON_BIN" \
    QWEN25_PYTHON_BIN="$QWEN25_PYTHON_BIN" \
    PAI_PYTHON_BIN="$PAI_PYTHON_BIN" \
    EAZY_PYTHON_BIN="$EAZY_PYTHON_BIN" \
    TASK=chair \
    BACKBONE="$BACKBONE" \
    METHOD="$METHOD" \
    OUT_ROOT="$source_dir/$split" \
    MODEL_PATH="$MODEL_PATH" \
    IMAGE_FOLDER="$IMAGE_FOLDER" \
    QUESTION_FILE="$q" \
    PRED_JSONL="$pred" \
    METRICS_JSON="$chair" \
    CHAIR_INPUT_JSONL="$chair_input" \
    COCO_ANN_ROOT="$COCO_ANN_ROOT" \
    CHAIR_CACHE="$CHAIR_CACHE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    LIMIT=0 \
    CONV_MODE="$CONV_MODE" \
    SEED="$SEED" \
    REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
    "${EXTRA_ENV[@]}" \
    bash "$CAL_ROOT/scripts/run_multibackbone_method_prediction.sh" \
    2>&1 | tee "$log"
}

run_ours_target() {
  local target="$1"
  configure_target "$target"
  local source_dir="$RAW_ROOT/$target"
  local out="$OURS_ROOT/$target"
  mkdir -p "$out"
  require_file "$source_dir/val/$PRED_BASENAME"
  require_file "$source_dir/test/$PRED_BASENAME"

  echo "== ours $target -> $out"
  env \
    GPU="$OURS_GPU" \
    CUDA_VISIBLE_DEVICES="$OURS_GPU" \
    PYTHONPATH="$CAL_ROOT:${PYTHONPATH:-}" \
    CAL_ROOT="$CAL_ROOT" \
    CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
    VGA_PYTHON_BIN="$VGA_PYTHON_BIN" \
    EAZY_PYTHON_BIN="$EAZY_PYTHON_BIN" \
    EAZY_ROOT="$EAZY_ROOT" \
    VGA_ROOT="$VGA_ROOT" \
    SOURCE_OUT="$source_dir" \
    VAL_SOURCE_OUT="$source_dir" \
    TEST_SOURCE_OUT="$source_dir" \
    OUT_ROOT="$out" \
    VAL_SPLIT=val \
    TEST_SPLIT=test \
    VAL_LIMIT="$LIMIT" \
    TEST_LIMIT="$LIMIT" \
    SOURCE_LIMIT="$SOURCE_LIMIT" \
    RUN_TEST=true \
    VAL_INTERVENTION_PRED_BASENAME="$PRED_BASENAME" \
    TEST_INTERVENTION_PRED_BASENAME="$PRED_BASENAME" \
    IMAGE_FOLDER="$IMAGE_FOLDER" \
    COCO_ANN_ROOT="$COCO_ANN_ROOT" \
    CHAIR_CACHE="$CHAIR_CACHE" \
    RISK_SCORE_MODE="$RISK_SCORE_MODE" \
    RISK_OBJECT_VOCAB="$RISK_OBJECT_VOCAB" \
    RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
    RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
    THRESHOLDS="$THRESHOLDS" \
    SUPPRESS_MODE="$SUPPRESS_MODE" \
    SUPPRESS_BIAS="$SUPPRESS_BIAS" \
    SELECT_OBJECTIVE="$SELECT_OBJECTIVE" \
    SELECT_MAX_RECALL_DROP="$SELECT_MAX_RECALL_DROP" \
    SELECT_MIN_DELTA_F1="$SELECT_MIN_DELTA_F1" \
    SELECT_MAX_DELTA_CHAIR_I="$SELECT_MAX_DELTA_CHAIR_I" \
    SELECT_MAX_DELTA_CHAIR_S="$SELECT_MAX_DELTA_CHAIR_S" \
    REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
    bash "$CAL_ROOT/scripts/run_coco_chair_v84_validate_threshold_then_apply.sh" \
    2>&1 | tee "$out/run.log"
}

echo "[settings] out=$OUT_ROOT"
echo "[settings] source_split_root=$SOURCE_SPLIT_ROOT"
echo "[settings] targets=$TARGETS"
echo "[settings] baseline_backbones=$BASELINE_BACKBONES run_baselines=$RUN_BASELINES"
echo "[settings] raw_gpu=$RAW_GPU ours_gpu=$OURS_GPU"

if [[ "$RUN_BASELINES" == "true" ]]; then
  for backbone_key in $BASELINE_BACKBONES; do
    for split in $SPLITS; do
      run_raw_baseline_split "$backbone_key" "$split"
    done
  done
fi

if [[ "$RUN_RAW" == "true" ]]; then
  for target in $TARGETS; do
    for split in $SPLITS; do
      run_raw_target_split "$target" "$split"
    done
  done
fi

if [[ "$RUN_OURS" == "true" ]]; then
  for target in $TARGETS; do
    run_ours_target "$target"
  done
fi

if [[ "$RUN_SUMMARY" == "true" ]]; then
  summary_args=()
  if [[ -f "$LLAVA15_BASELINE_CHAIR" ]]; then
    summary_args+=(--baseline_entry "llava15::$LLAVA15_BASELINE_CHAIR")
  fi
  if [[ -f "$RAW_ROOT/baseline_llava_next/test/chair_baseline_llava_next.json" ]]; then
    summary_args+=(--baseline_entry "llava_next::$RAW_ROOT/baseline_llava_next/test/chair_baseline_llava_next.json")
  fi
  if [[ -f "$RAW_ROOT/baseline_qwen25/test/chair_baseline_qwen25.json" ]]; then
    summary_args+=(--baseline_entry "qwen25::$RAW_ROOT/baseline_qwen25/test/chair_baseline_qwen25.json")
  fi
  if [[ "$INCLUDE_EXISTING_LLAVA15_VGA_IN_SUMMARY" == "true" && -f "$LLAVA15_VGA_RAW_CHAIR" ]]; then
    existing="llava15_vga::VGA / LLaVA-1.5::$LLAVA15_VGA_RAW_CHAIR"
    if [[ -f "$LLAVA15_VGA_OURS_CSV" ]]; then
      existing="${existing}::$LLAVA15_VGA_OURS_CSV"
    fi
    summary_args+=(--existing_entry "$existing")
  fi
  for target in $TARGETS; do
    summary_args+=(--target "$target")
  done
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/summarize_coco_chair_generative_panel.py" \
    --raw_root "$RAW_ROOT" \
    --ours_root "$OURS_ROOT" \
    "${summary_args[@]}" \
    --delta_metric CHAIRi \
    --out_csv "$SUMMARY_DIR/chair_multibackbone_method_ours_panel.csv" \
    --out_md "$SUMMARY_DIR/chair_multibackbone_method_ours_panel.md"
fi

echo "[done] $OUT_ROOT"
