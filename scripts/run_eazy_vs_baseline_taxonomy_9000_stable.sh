#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy/bin/python}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
POPE_COCO_FALLBACK="${POPE_COCO_FALLBACK:-/home/kms/VISTA/pope_coco}"
RUNTIME_SHIM_ROOT="${RUNTIME_SHIM_ROOT:-/tmp/eazy_origin_runtime_shim}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-$EAZY_ROOT/nltk_data}"
DOWNLOAD_NLTK="${DOWNLOAD_NLTK:-1}"
RUN_PREP="${RUN_PREP:-1}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
Q_WITHOBJ="${Q_WITHOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
BASELINE_JSONL="${BASELINE_JSONL:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/baseline/pred_vanilla_9000.jsonl}"
FEATURES_CSV="${FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_e1_4_l16_24/features/features_unified_table.csv}"

OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/eazy_vs_baseline_9000_stable}"
TAX_OUT_DIR="${TAX_OUT_DIR:-$OUT_DIR/taxonomy}"
D1D2_OUT_DIR="${D1D2_OUT_DIR:-$OUT_DIR/d1d2_audit}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"

MODEL="${MODEL:-llava-1.5}"
BEAM="${BEAM:-1}"
TOPK_K="${TOPK_K:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
SEED="${SEED:-1994}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

PRED_JSONL="$OUT_DIR/pred_eazy_9000.jsonl"
METRICS_JSON="$OUT_DIR/metrics_eazy_9000.json"

mkdir -p "$OUT_DIR" "$TAX_OUT_DIR" "$D1D2_OUT_DIR" "$LOG_DIR"

for f in "$Q_WITHOBJ" "$GT_CSV" "$BASELINE_JSONL" "$FEATURES_CSV"; do
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

if [[ "$REUSE_IF_EXISTS" == "true" && -f "$PRED_JSONL" ]]; then
  echo "[reuse] $PRED_JSONL"
else
  run_step "predict_full_9000" "$LOG_DIR/01_predict_full_9000.log" \
    env CUDA_VISIBLE_DEVICES="$GPU" \
        PYTHONPATH="$CAL_ROOT" \
        "$EAZY_PYTHON_BIN" "$CAL_ROOT/scripts/run_eazy_question_subset.py" \
          --eazy_root "$EAZY_ROOT" \
          --runtime_shim_root "$RUNTIME_SHIM_ROOT" \
          --nltk_data_dir "$NLTK_DATA_DIR" \
          --question_file "$Q_WITHOBJ" \
          --image_folder "$IMAGE_FOLDER" \
          --answers_file "$PRED_JSONL" \
          --model "$MODEL" \
          --gpu_id 0 \
          --beam "$BEAM" \
          --k "$TOPK_K" \
          --max_new_tokens "$MAX_NEW_TOKENS" \
          --seed "$SEED"
fi

run_step "metrics" "$LOG_DIR/02_metrics.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$PRED_JSONL" \
    --pred_text_key output \
    --out_json "$METRICS_JSON"

run_step "taxonomy" "$LOG_DIR/03_taxonomy.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_vga_failure_taxonomy.py" \
    --gt_csv "$GT_CSV" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --vga_pred_jsonl "$PRED_JSONL" \
    --baseline_pred_text_key text \
    --vga_pred_text_key output \
    --out_dir "$TAX_OUT_DIR"

run_step "d1d2_audit" "$LOG_DIR/04_d1d2_audit.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/analyze_yes_dampening_d1d2.py" \
    --per_case_csv "$TAX_OUT_DIR/per_case_compare.csv" \
    --features_csv "$FEATURES_CSV" \
    --out_dir "$D1D2_OUT_DIR"

echo "[done] $OUT_DIR"
echo "[summary] $METRICS_JSON"
