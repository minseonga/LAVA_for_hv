#!/usr/bin/env bash
set -euo pipefail

# Unified raw prediction runner for paper-scale backbone expansion.
#
# This script standardizes baseline/VGA/PAI prediction calls across the
# backbones currently represented in this repository:
#   - llava15:    LLaVA-1.5 style runner
#   - llava_next: VGA_origin/llava_next runner
#   - qwen25_vl:  VGA_origin/Qwen2.5-VL runner
#   - qwen35_vl:  Qwen3.5-VL raw baseline runner
#
# It intentionally runs raw methods only. Post-hoc "ours" controllers should
# consume these JSONL files through a backbone-specific runtime adapter.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
PAI_ROOT="${PAI_ROOT:-/home/kms/PAI}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
LLAVA_NEXT_PYTHON_BIN="${LLAVA_NEXT_PYTHON_BIN:-$VGA_PYTHON_BIN}"
QWEN35_PYTHON_BIN="${QWEN35_PYTHON_BIN:-$CAL_PYTHON_BIN}"
PAI_PYTHON_BIN="${PAI_PYTHON_BIN:-/home/kms/miniconda3/envs/pai_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

TASK="${TASK:-pope}"                 # pope | chair
BACKBONE="${BACKBONE:-llava15}"      # llava15 | llava_next | qwen25_vl | qwen35_vl
METHOD="${METHOD:-vga}"              # baseline | vga | pai
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/multibackbone_raw/${TASK}/${BACKBONE}/${METHOD}}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"

Q_NOOBJ="${Q_NOOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl}"
Q_WITHOBJ="${Q_WITHOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"

CHAIR_Q_JSONL="${CHAIR_Q_JSONL:-}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/images/mscoco/annotations}"
CHAIR_CACHE="${CHAIR_CACHE:-$OUT_ROOT/chair_cache.pkl}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
SEED="${SEED:-17}"

VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_ATTN_NORM="${VGA_ATTN_NORM:-false}"
VGA_TORCH_TYPE="${VGA_TORCH_TYPE:-bf16}"
VGA_ATTN_TYPE="${VGA_ATTN_TYPE:-eager}"

PAI_MODEL="${PAI_MODEL:-}"
PAI_USE_ATTN="${PAI_USE_ATTN:-1}"
PAI_USE_CFG="${PAI_USE_CFG:-1}"
PAI_BEAM="${PAI_BEAM:-1}"
PAI_SAMPLE="${PAI_SAMPLE:-0}"
PAI_ALPHA="${PAI_ALPHA:-0.2}"
PAI_GAMMA="${PAI_GAMMA:-1.1}"
PAI_START_LAYER="${PAI_START_LAYER:-}"
PAI_END_LAYER="${PAI_END_LAYER:-}"

mkdir -p "$OUT_ROOT"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

remove_if_overwrite() {
  local path="$1"
  if [[ "$REUSE_IF_EXISTS" != "true" && -f "$path" ]]; then
    rm -f "$path"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing file: $path" >&2
    exit 2
  fi
}

default_conv_mode() {
  case "$BACKBONE" in
    llava15) echo "llava_v1" ;;
    llava_next) echo "llava_llama_3" ;;
    qwen25_vl) echo "llava_v1" ;;
    qwen35_vl) echo "qwen35_vl" ;;
    *) echo "[error] unsupported BACKBONE=$BACKBONE" >&2; exit 2 ;;
  esac
}

default_start_layer() {
  case "$BACKBONE" in
    llava15) echo "2" ;;
    llava_next) echo "0" ;;
    qwen25_vl) echo "4" ;;
    qwen35_vl) echo "4" ;;
    *) echo "[error] unsupported BACKBONE=$BACKBONE" >&2; exit 2 ;;
  esac
}

default_end_layer() {
  case "$BACKBONE" in
    llava15) echo "15" ;;
    llava_next) echo "15" ;;
    qwen25_vl) echo "15" ;;
    qwen35_vl) echo "15" ;;
    *) echo "[error] unsupported BACKBONE=$BACKBONE" >&2; exit 2 ;;
  esac
}

default_pai_model() {
  case "$BACKBONE" in
    llava15) echo "llava-1.5" ;;
    llava_next) echo "llava-next" ;;
    qwen25_vl) echo "qwen25-vl" ;;
    qwen35_vl) echo "qwen35-vl" ;;
    *) echo "[error] unsupported BACKBONE=$BACKBONE" >&2; exit 2 ;;
  esac
}

vga_runner() {
  case "$BACKBONE" in
    llava15) echo "$CAL_ROOT/scripts/run_vga_origin_llava_caption_compat.py" ;;
    llava_next) echo "$CAL_ROOT/scripts/run_vga_origin_llava_next_compat.py" ;;
    qwen25_vl) echo "$VGA_ROOT/eval/object_hallucination_vqa_qwen25-vl.py" ;;
    qwen35_vl)
      echo "[error] VGA for BACKBONE=qwen35_vl requires porting the VGA attention/generation hooks." >&2
      exit 2
      ;;
    *) echo "[error] unsupported BACKBONE=$BACKBONE" >&2; exit 2 ;;
  esac
}

if [[ -z "$MAX_NEW_TOKENS" ]]; then
  if [[ "$TASK" == "pope" ]]; then
    MAX_NEW_TOKENS=8
  else
    MAX_NEW_TOKENS=512
  fi
fi

CONV_MODE="${CONV_MODE:-$(default_conv_mode)}"
VGA_START_LAYER="${VGA_START_LAYER:-$(default_start_layer)}"
VGA_END_LAYER="${VGA_END_LAYER:-$(default_end_layer)}"
PAI_MODEL="${PAI_MODEL:-$(default_pai_model)}"
PAI_START_LAYER="${PAI_START_LAYER:-$VGA_START_LAYER}"
PAI_END_LAYER="${PAI_END_LAYER:-$VGA_END_LAYER}"

if [[ "$TASK" == "pope" ]]; then
  if [[ "$METHOD" == "baseline" ]]; then
    QUESTION_FILE="${QUESTION_FILE:-$Q_NOOBJ}"
  else
    QUESTION_FILE="${QUESTION_FILE:-$Q_WITHOBJ}"
  fi
  require_file "$QUESTION_FILE"
  require_file "$GT_CSV"
elif [[ "$TASK" == "chair" ]]; then
  QUESTION_FILE="${QUESTION_FILE:-$CHAIR_Q_JSONL}"
  if [[ -z "$QUESTION_FILE" ]]; then
    echo "[error] TASK=chair requires QUESTION_FILE or CHAIR_Q_JSONL" >&2
    exit 2
  fi
  require_file "$QUESTION_FILE"
else
  echo "[error] unsupported TASK=$TASK" >&2
  exit 2
fi

PRED_JSONL="${PRED_JSONL:-$OUT_ROOT/pred_${METHOD}.jsonl}"
METRICS_JSON="${METRICS_JSON:-$OUT_ROOT/metrics_${METHOD}.json}"
CHAIR_INPUT_JSONL="${CHAIR_INPUT_JSONL:-$OUT_ROOT/chair_input_${METHOD}.jsonl}"

echo "[settings] task=$TASK backbone=$BACKBONE method=$METHOD"
echo "[settings] model_path=$MODEL_PATH"
echo "[settings] question_file=$QUESTION_FILE"
echo "[settings] image_folder=$IMAGE_FOLDER"
echo "[settings] out=$PRED_JSONL"

run_vga_like() {
  local use_add="$1"
  local start_layer="$2"
  local end_layer="$3"
  local runner
  runner="$(vga_runner)"

  if [[ "$BACKBONE" == "llava15" ]]; then
    "$VGA_PYTHON_BIN" "$runner" \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --model-base "${MODEL_BASE:-}" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$QUESTION_FILE" \
      --answers-file "$PRED_JSONL" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$MAX_NEW_TOKENS" \
      --use_add "$use_add" \
      --attn_coef "$VGA_ATTN_COEF" \
      --cd_alpha "$VGA_CD_ALPHA" \
      --start_layer "$start_layer" \
      --end_layer "$end_layer" \
      --head_balancing "$VGA_HEAD_BALANCING" \
      --attn_norm "$VGA_ATTN_NORM" \
      --sampling "$VGA_SAMPLING" \
      --seed "$SEED"
  elif [[ "$BACKBONE" == "llava_next" ]]; then
    "$VGA_PYTHON_BIN" "$runner" \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --model-base "${MODEL_BASE:-}" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$QUESTION_FILE" \
      --answers-file "$PRED_JSONL" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$MAX_NEW_TOKENS" \
      --use_add "$use_add" \
      --attn_coef "$VGA_ATTN_COEF" \
      --cd_alpha "$VGA_CD_ALPHA" \
      --start_layer "$start_layer" \
      --end_layer "$end_layer" \
      --head_balancing "$VGA_HEAD_BALANCING" \
      --attn_norm "$VGA_ATTN_NORM" \
      --sampling "$VGA_SAMPLING" \
      --torch_type "$VGA_TORCH_TYPE" \
      --attn_type "$VGA_ATTN_TYPE" \
      --seed "$SEED"
  else
    "$VGA_PYTHON_BIN" "$runner" \
      --model-path "$MODEL_PATH" \
      --model-base "${MODEL_BASE:-}" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$QUESTION_FILE" \
      --answers-file "$PRED_JSONL" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$MAX_NEW_TOKENS" \
      --use_add "$use_add" \
      --attn_coef "$VGA_ATTN_COEF" \
      --cd_alpha "$VGA_CD_ALPHA" \
      --start_layer "$start_layer" \
      --end_layer "$end_layer" \
      --head_balancing "$VGA_HEAD_BALANCING" \
      --attn_norm "$VGA_ATTN_NORM" \
      --sampling "$VGA_SAMPLING" \
      --torch_type "$VGA_TORCH_TYPE" \
      --attn_type "$VGA_ATTN_TYPE" \
      --seed "$SEED"
  fi
}

run_pai() {
  local flags=()
  if [[ "$PAI_USE_ATTN" == "1" ]]; then
    flags+=(--use_attn)
  fi
  if [[ "$PAI_USE_CFG" == "1" ]]; then
    flags+=(--use_cfg)
  fi
  if [[ "$PAI_SAMPLE" == "1" ]]; then
    flags+=(--sample)
  fi
  "$PAI_PYTHON_BIN" "$CAL_ROOT/scripts/run_pai_question_subset.py" \
    --pai_root "$PAI_ROOT" \
    --question_file "$QUESTION_FILE" \
    --image_folder "$IMAGE_FOLDER" \
    --answers_file "$PRED_JSONL" \
    --model "$PAI_MODEL" \
    --model_path "$MODEL_PATH" \
    --gpu_id 0 \
    --beam "$PAI_BEAM" \
    --alpha "$PAI_ALPHA" \
    --gamma "$PAI_GAMMA" \
    --start_layer "$PAI_START_LAYER" \
    --end_layer "$PAI_END_LAYER" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${flags[@]}" \
    --seed "$SEED"
}

if ! reuse_file "$PRED_JSONL"; then
  remove_if_overwrite "$PRED_JSONL"
  case "$METHOD" in
    baseline)
      if [[ "$BACKBONE" == "qwen35_vl" ]]; then
        "$QWEN35_PYTHON_BIN" "$CAL_ROOT/scripts/run_qwen35_vl_question_subset.py" \
          --model-path "$MODEL_PATH" \
          --image-folder "$IMAGE_FOLDER" \
          --question-file "$QUESTION_FILE" \
          --answers-file "$PRED_JSONL" \
          --max-new-tokens "$MAX_NEW_TOKENS" \
          --seed "$SEED"
      elif [[ "$BACKBONE" == "llava_next" ]]; then
        "$LLAVA_NEXT_PYTHON_BIN" "$CAL_ROOT/scripts/run_llava_next_question_subset.py" \
          --vga-root "$VGA_ROOT" \
          --model-path "$MODEL_PATH" \
          --model-base "${MODEL_BASE:-}" \
          --image-folder "$IMAGE_FOLDER" \
          --question-file "$QUESTION_FILE" \
          --answers-file "$PRED_JSONL" \
          --conv-mode "$CONV_MODE" \
          --max-new-tokens "$MAX_NEW_TOKENS" \
          --torch-type "$VGA_TORCH_TYPE" \
          --attn-type "$VGA_ATTN_TYPE" \
          --use-cache true \
          --generation-mode manual_greedy \
          --do-sample false \
          --num-beams 1 \
          --seed "$SEED"
      elif [[ "$BACKBONE" == "llava15" ]]; then
        "$CAL_PYTHON_BIN" -m llava.eval.model_vqa_loader \
          --model-path "$MODEL_PATH" \
          --model-base "${MODEL_BASE:-}" \
          --image-folder "$IMAGE_FOLDER" \
          --question-file "$QUESTION_FILE" \
          --answers-file "$PRED_JSONL" \
          --conv-mode "$CONV_MODE" \
          --temperature 0 \
          --num_beams 1 \
          --max_new_tokens "$MAX_NEW_TOKENS"
      else
        run_vga_like false 99 0
      fi
      ;;
    vga)
      run_vga_like "$VGA_USE_ADD" "$VGA_START_LAYER" "$VGA_END_LAYER"
      ;;
    pai)
      if [[ "$BACKBONE" == "qwen35_vl" ]]; then
        echo "[error] PAI for BACKBONE=qwen35_vl requires a PAI ModelLoader/adapter port." >&2
        exit 2
      fi
      run_pai
      ;;
    *)
      echo "[error] unsupported METHOD=$METHOD" >&2
      exit 2
      ;;
  esac
else
  echo "[reuse] $PRED_JSONL"
fi

if [[ "$TASK" == "pope" ]]; then
  if [[ -z "${PRED_TEXT_KEY:-}" ]]; then
    PRED_TEXT_KEY="output"
    if [[ "$METHOD" == "baseline" && "$BACKBONE" == "llava15" ]]; then
      PRED_TEXT_KEY="text"
    fi
    if [[ "$METHOD" == "pai" ]]; then
      PRED_TEXT_KEY="text"
    fi
  fi
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$PRED_JSONL" \
    --pred_text_key "$PRED_TEXT_KEY" \
    --out_json "$METRICS_JSON"
  echo "[saved] $METRICS_JSON"
else
  if [[ -z "${CAPTION_KEY:-}" ]]; then
    CAPTION_KEY="output"
    if [[ "$METHOD" == "baseline" && "$BACKBONE" == "llava15" ]]; then
      CAPTION_KEY="text"
    fi
    if [[ "$METHOD" == "pai" ]]; then
      CAPTION_KEY="text"
    fi
  fi
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/prepare_chair_caption_jsonl.py" \
    --in_file "$PRED_JSONL" \
    --out_file "$CHAIR_INPUT_JSONL" \
    --image_id_key image_id \
    --image_key image \
    --drop_missing
  PYTHONPATH="$EAZY_ROOT:${PYTHONPATH:-}" "$EAZY_PYTHON_BIN" "$EAZY_ROOT/eval_script/chair.py" \
    --cap_file "$CHAIR_INPUT_JSONL" \
    --image_id_key image_id \
    --caption_key "$CAPTION_KEY" \
    --coco_path "$COCO_ANN_ROOT" \
    --cache "$CHAIR_CACHE" \
    --save_path "$METRICS_JSON"
  echo "[saved] $METRICS_JSON"
fi

echo "[done] $PRED_JSONL"
