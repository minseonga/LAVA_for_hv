#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
POPE_COCO_FALLBACK="${POPE_COCO_FALLBACK:-/home/kms/VISTA/pope_coco}"
RUNTIME_SHIM_ROOT="${RUNTIME_SHIM_ROOT:-/tmp/eazy_origin_runtime_shim}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-$EAZY_ROOT/nltk_data}"
DOWNLOAD_NLTK="${DOWNLOAD_NLTK:-1}"
RUN_PREP="${RUN_PREP:-1}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
MODEL="${MODEL:-llava-1.5}"
DATA_PATH="${DATA_PATH:-/home/kms/data/pope/val2014}"
TOPK_K="${TOPK_K:-2}"
BEAM="${BEAM:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
BASELINE_JSONL="${BASELINE_JSONL:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/baseline/pred_vanilla_9000.jsonl}"
OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/eazy_officiallike_full9000_v1}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

ADV_JSONL="$OUT_DIR/raw_adversarial.jsonl"
POP_JSONL="$OUT_DIR/raw_popular.jsonl"
RND_JSONL="$OUT_DIR/raw_random.jsonl"
PRED_JSONL="$OUT_DIR/pred_eazy_9000.jsonl"
METRICS_JSON="$OUT_DIR/metrics_eazy_9000.json"
COMPARE_JSON="$OUT_DIR/compare_vs_baseline.json"
COMPARE_CSV="$OUT_DIR/compare_vs_baseline.csv"

mkdir -p "$OUT_DIR" "$LOG_DIR"

for f in "$GT_CSV" "$BASELINE_JSONL"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing file: $f" >&2
    exit 1
  fi
done

run_step() {
  local name="$1"
  local logfile="$2"
  shift 2
  echo "[step] $name"
  echo "[log] $logfile"
  "$@" 2>&1 | tee "$logfile"
}

reuse_file() {
  local path="$1"
  if [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]; then
    return 0
  fi
  return 1
}

if [[ "$RUN_PREP" == "1" ]]; then
  run_step "prepare_runtime" "$LOG_DIR/00_prepare_runtime.log" \
    env EAZY_ROOT="$EAZY_ROOT" \
        EAZY_PYTHON_BIN="$EAZY_PYTHON_BIN" \
        POPE_COCO_FALLBACK="$POPE_COCO_FALLBACK" \
        RUNTIME_SHIM_ROOT="$RUNTIME_SHIM_ROOT" \
        NLTK_DATA_DIR="$NLTK_DATA_DIR" \
        DOWNLOAD_NLTK="$DOWNLOAD_NLTK" \
        bash "$SCRIPT_DIR/prepare_eazy_origin_runtime.sh"
fi

run_one_category() {
  local cat="$1"
  local save_path="$2"
  if reuse_file "$save_path"; then
    echo "[reuse] $save_path"
    return
  fi
  run_step "official_${cat}" "$LOG_DIR/official_${cat}.log" \
    env CUDA_VISIBLE_DEVICES="$GPU" \
        PYTHONPATH="$CAL_ROOT" \
        "$EAZY_PYTHON_BIN" "$CAL_ROOT/scripts/run_eazy_origin_repo_pope_dump.py" \
          --eazy_root "$EAZY_ROOT" \
          --runtime_shim_root "$RUNTIME_SHIM_ROOT" \
          --nltk_data_dir "$NLTK_DATA_DIR" \
          --model "$MODEL" \
          --pope_type "$cat" \
          --gpu_id 0 \
          --data_path "$DATA_PATH" \
          --batch_size "$BATCH_SIZE" \
          --num_workers "$NUM_WORKERS" \
          --beam "$BEAM" \
          --k "$TOPK_K" \
          --save_jsonl "$save_path"
}

run_one_category adversarial "$ADV_JSONL"
run_one_category popular "$POP_JSONL"
run_one_category random "$RND_JSONL"

run_step "normalize" "$LOG_DIR/04_normalize.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/normalize_pope_category_preds.py" \
    --adversarial_jsonl "$ADV_JSONL" \
    --popular_jsonl "$POP_JSONL" \
    --random_jsonl "$RND_JSONL" \
    --out_jsonl "$PRED_JSONL" \
    --text_keys "ans,output,text,answer" \
    --category_size 3000

run_step "metrics" "$LOG_DIR/05_metrics.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$PRED_JSONL" \
    --pred_text_key output \
    --out_json "$METRICS_JSON"

run_step "compare_vs_baseline" "$LOG_DIR/06_compare_vs_baseline.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/compare_pope_yesno_runs.py" \
    --gt_csv "$GT_CSV" \
    --base_pred_jsonl "$BASELINE_JSONL" \
    --new_pred_jsonl "$PRED_JSONL" \
    --pred_text_key auto \
    --out_json "$COMPARE_JSON" \
    --out_fail_csv "$COMPARE_CSV"

echo "[done] $OUT_DIR"
echo "[summary] $METRICS_JSON"
