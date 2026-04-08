#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy/bin/python}"
POPE_COCO_FALLBACK="${POPE_COCO_FALLBACK:-/home/kms/VISTA/pope_coco}"
RUNTIME_SHIM_ROOT="${RUNTIME_SHIM_ROOT:-/tmp/eazy_origin_runtime_shim}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-$EAZY_ROOT/nltk_data}"
DOWNLOAD_NLTK="${DOWNLOAD_NLTK:-1}"
RUN_PREP="${RUN_PREP:-1}"
LOCAL_TRANSFORMERS_SRC="${LOCAL_TRANSFORMERS_SRC:-$EAZY_ROOT/transformers-4.29.2/src}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
MODEL="${MODEL:-llava-1.5}"
POPE_TYPE="${POPE_TYPE:-adversarial}"
DATA_PATH="${DATA_PATH:-/home/kms/data/pope/val2014}"
TOPK_K="${TOPK_K:-3}"
BEAM="${BEAM:-1}"

if [[ "$RUN_PREP" == "1" ]]; then
  EAZY_ROOT="$EAZY_ROOT" \
  EAZY_PYTHON_BIN="$EAZY_PYTHON_BIN" \
  POPE_COCO_FALLBACK="$POPE_COCO_FALLBACK" \
  RUNTIME_SHIM_ROOT="$RUNTIME_SHIM_ROOT" \
  NLTK_DATA_DIR="$NLTK_DATA_DIR" \
  DOWNLOAD_NLTK="$DOWNLOAD_NLTK" \
  bash "$SCRIPT_DIR/prepare_eazy_origin_runtime.sh"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
if [[ -d "$LOCAL_TRANSFORMERS_SRC" ]]; then
  export PYTHONPATH="$RUNTIME_SHIM_ROOT:$LOCAL_TRANSFORMERS_SRC:$EAZY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$RUNTIME_SHIM_ROOT:$EAZY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
export NLTK_DATA="$NLTK_DATA_DIR${NLTK_DATA:+:$NLTK_DATA}"

cd "$EAZY_ROOT"
"$EAZY_PYTHON_BIN" eval_script/pope_eval_eazy_onepass.py \
  --model "$MODEL" \
  --pope-type "$POPE_TYPE" \
  --gpu-id 0 \
  --data_path "$DATA_PATH" \
  --k "$TOPK_K" \
  --beam "$BEAM" \
  "$@"
