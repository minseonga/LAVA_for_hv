#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
VCD_ROOT="${VCD_ROOT:-/home/kms/VCD}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VCD_PYTHON_BIN="${VCD_PYTHON_BIN:-python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

QUESTION_FILE="${QUESTION_FILE:-}"
GT_CSV="${GT_CSV:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
BASELINE_PRED_JSONL="${BASELINE_PRED_JSONL:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/baseline/pred_vanilla_9000.jsonl}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vcd_officiallike_subset_eval}"
LOG_DIR="$OUT_ROOT/logs"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

VCD_USE_CD="${VCD_USE_CD:-1}"
VCD_NOISE_STEP="${VCD_NOISE_STEP:-500}"
VCD_CD_ALPHA="${VCD_CD_ALPHA:-1.0}"
VCD_CD_BETA="${VCD_CD_BETA:-0.2}"

TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-}"

SEEDS="${SEEDS:-1994}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
OFFICIAL_EVAL_SCRIPT="${OFFICIAL_EVAL_SCRIPT:-$VCD_ROOT/experiments/eval/object_hallucination_vqa_llava.py}"

if [[ -z "$QUESTION_FILE" ]]; then
  echo "[error] QUESTION_FILE is required" >&2
  exit 1
fi
if [[ -z "$GT_CSV" ]]; then
  echo "[error] GT_CSV is required" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_DIR"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

run_step() {
  local name="$1"
  local log="$2"
  shift 2
  echo "[step] $name"
  echo "[log] $log"
  "$@" > >(tee "$log") 2>&1
}

IFS=',' read -r -a seed_array <<< "$SEEDS"
for seed in "${seed_array[@]}"; do
  seed="$(echo "$seed" | xargs)"
  pred_jsonl="$OUT_ROOT/pred_seed${seed}.jsonl"
  metrics_json="$OUT_ROOT/metrics_seed${seed}.json"
  compare_json="$OUT_ROOT/compare_seed${seed}.json"
  compare_csv="$OUT_ROOT/compare_seed${seed}.csv"

  loader_log="$LOG_DIR/01_loader_seed${seed}.log"
  metrics_log="$LOG_DIR/02_metrics_seed${seed}.log"
  compare_log="$LOG_DIR/03_compare_seed${seed}.log"

  if ! reuse_file "$pred_jsonl"; then
    loader_args=(
      "$OFFICIAL_EVAL_SCRIPT"
      --model-path "$MODEL_PATH"
      --image-folder "$IMAGE_FOLDER"
      --question-file "$QUESTION_FILE"
      --answers-file "$pred_jsonl"
      --conv-mode "$CONV_MODE"
      --temperature "$TEMPERATURE"
    )
    if [[ -n "$MODEL_BASE" ]]; then
      loader_args+=(--model-base "$MODEL_BASE")
    fi
    if [[ -n "$TOP_P" ]]; then
      loader_args+=(--top_p "$TOP_P")
    fi
    if [[ "$VCD_USE_CD" == "1" ]]; then
      loader_args+=(
        --use_cd
        --noise_step "$VCD_NOISE_STEP"
        --cd_alpha "$VCD_CD_ALPHA"
        --cd_beta "$VCD_CD_BETA"
      )
    fi

    run_step "officiallike_loader_seed${seed}" "$loader_log" \
      env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$VCD_ROOT" PYTHONHASHSEED="$seed" "$VCD_PYTHON_BIN" -c '
import random
import runpy
import sys

seed = int(sys.argv[1])
script = sys.argv[2]
argv = sys.argv[3:]

random.seed(seed)
try:
    import numpy as np
    np.random.seed(seed)
except Exception:
    pass

try:
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
except Exception:
    pass

try:
    from transformers import set_seed as hf_set_seed
    hf_set_seed(seed)
except Exception:
    pass

sys.argv = [script] + argv
runpy.run_path(script, run_name="__main__")
' "$seed" "${loader_args[@]}"
  else
    echo "[reuse] $pred_jsonl"
  fi

  if ! reuse_file "$metrics_json"; then
    run_step "metrics_seed${seed}" "$metrics_log" \
      env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/eval_pope_subset_yesno.py \
        --gt_csv "$GT_CSV" \
        --pred_jsonl "$pred_jsonl" \
        --pred_text_key text \
        --out_json "$metrics_json"
  else
    echo "[reuse] $metrics_json"
  fi

  if ! reuse_file "$compare_json"; then
    run_step "compare_seed${seed}" "$compare_log" \
      env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/compare_pope_yesno_runs.py \
        --gt_csv "$GT_CSV" \
        --base_pred_jsonl "$BASELINE_PRED_JSONL" \
        --new_pred_jsonl "$pred_jsonl" \
        --pred_text_key auto \
        --out_json "$compare_json" \
        --out_fail_csv "$compare_csv"
  else
    echo "[reuse] $compare_json"
  fi
done

echo "[done] official-like subset eval -> $OUT_ROOT"
