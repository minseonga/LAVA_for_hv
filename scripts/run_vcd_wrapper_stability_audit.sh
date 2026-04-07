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

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
FULL_Q_WITHOBJ="${FULL_Q_WITHOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
FULL_GT_CSV="${FULL_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
BASELINE_PRED_JSONL="${BASELINE_PRED_JSONL:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/baseline/pred_vanilla_9000.jsonl}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vcd_wrapper_stability_audit}"
LOG_DIR="$OUT_ROOT/logs"
RUNS_DIR="$OUT_ROOT/runs"
SUBSET_DIR="$OUT_ROOT/subset"

SUBSET_N="${SUBSET_N:-500}"
SUBSET_SEED="${SUBSET_SEED:-42}"
SEEDS="${SEEDS:-1994,1995,1996}"

VCD_CONV_MODE="${VCD_CONV_MODE:-llava_v1}"
VCD_NOISE_STEP="${VCD_NOISE_STEP:-500}"
VCD_CD_ALPHA="${VCD_CD_ALPHA:-1.0}"
VCD_CD_BETA="${VCD_CD_BETA:-0.1}"
VCD_TEMPERATURE="${VCD_TEMPERATURE:-1.0}"
VCD_TOP_P="${VCD_TOP_P:-1.0}"
VCD_TOP_K="${VCD_TOP_K:-0}"
VCD_MAX_NEW_TOKENS="${VCD_MAX_NEW_TOKENS:-8}"

RUN_CD_SAMPLE="${RUN_CD_SAMPLE:-1}"
RUN_CD_GREEDY="${RUN_CD_GREEDY:-1}"
RUN_NOCD_SAMPLE="${RUN_NOCD_SAMPLE:-1}"
RUN_NOCD_GREEDY="${RUN_NOCD_GREEDY:-1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

SUBSET_GT_CSV="$SUBSET_DIR/pope_strict_${SUBSET_N}_gt.csv"
SUBSET_Q_JSONL="$SUBSET_DIR/pope_strict_${SUBSET_N}_q.jsonl"
BASELINE_SUBSET_METRICS_JSON="$OUT_ROOT/baseline_subset_metrics.json"
SUMMARY_JSON="$OUT_ROOT/summary.json"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$RUNS_DIR"

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

echo "[1/4] build subset"
if [[ ! -f "$SUBSET_GT_CSV" || ! -f "$SUBSET_Q_JSONL" ]]; then
  run_step "build_subset" "$LOG_DIR/01_build_subset.log" \
    env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_pope_strict_subset.py \
      --full_gt_csv "$FULL_GT_CSV" \
      --full_q_jsonl "$FULL_Q_WITHOBJ" \
      --out_dir "$SUBSET_DIR" \
      --n_total "$SUBSET_N" \
      --seed "$SUBSET_SEED" \
      --balance_category true \
      --balance_answer true
else
  echo "[reuse] $SUBSET_GT_CSV"
  echo "[reuse] $SUBSET_Q_JSONL"
fi

echo "[2/4] baseline subset metrics"
if ! reuse_file "$BASELINE_SUBSET_METRICS_JSON"; then
  run_step "baseline_subset_metrics" "$LOG_DIR/02_baseline_subset_metrics.log" \
    env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/eval_pope_subset_yesno.py \
      --gt_csv "$SUBSET_GT_CSV" \
      --pred_jsonl "$BASELINE_PRED_JSONL" \
      --pred_text_key text \
      --out_json "$BASELINE_SUBSET_METRICS_JSON"
else
  echo "[reuse] $BASELINE_SUBSET_METRICS_JSON"
fi

variant_specs=()
if [[ "$RUN_CD_SAMPLE" == "1" ]]; then
  variant_specs+=("cd_sample|1|true")
fi
if [[ "$RUN_CD_GREEDY" == "1" ]]; then
  variant_specs+=("cd_greedy|1|false")
fi
if [[ "$RUN_NOCD_SAMPLE" == "1" ]]; then
  variant_specs+=("nocd_sample|0|true")
fi
if [[ "$RUN_NOCD_GREEDY" == "1" ]]; then
  variant_specs+=("nocd_greedy|0|false")
fi

echo "[3/4] run variants"
IFS=',' read -r -a seed_array <<< "$SEEDS"
for spec in "${variant_specs[@]}"; do
  IFS='|' read -r variant use_cd do_sample <<< "$spec"
  variant_dir="$RUNS_DIR/$variant"
  mkdir -p "$variant_dir"
  for seed in "${seed_array[@]}"; do
    seed="$(echo "$seed" | xargs)"
    pred_jsonl="$variant_dir/pred_seed${seed}.jsonl"
    metrics_json="$variant_dir/metrics_seed${seed}.json"
    compare_json="$variant_dir/compare_seed${seed}.json"
    compare_csv="$variant_dir/compare_seed${seed}.csv"
    predict_log="$LOG_DIR/03_${variant}_seed${seed}_predict.log"
    metrics_log="$LOG_DIR/03_${variant}_seed${seed}_metrics.log"
    compare_log="$LOG_DIR/03_${variant}_seed${seed}_compare.log"

    vcd_flags=()
    if [[ "$use_cd" == "1" ]]; then
      vcd_flags+=(--use_cd)
    fi

    if ! reuse_file "$pred_jsonl"; then
      run_step "${variant}_seed${seed}_predict" "$predict_log" \
        env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$VCD_PYTHON_BIN" scripts/run_vcd_question_subset.py \
          --vcd_root "$VCD_ROOT" \
          --question_file "$SUBSET_Q_JSONL" \
          --image_folder "$IMAGE_FOLDER" \
          --answers_file "$pred_jsonl" \
          --model_path "$MODEL_PATH" \
          --conv_mode "$VCD_CONV_MODE" \
          --gpu_id 0 \
          --temperature "$VCD_TEMPERATURE" \
          --top_p "$VCD_TOP_P" \
          --top_k "$VCD_TOP_K" \
          --do_sample "$do_sample" \
          --noise_step "$VCD_NOISE_STEP" \
          --cd_alpha "$VCD_CD_ALPHA" \
          --cd_beta "$VCD_CD_BETA" \
          --max_new_tokens "$VCD_MAX_NEW_TOKENS" \
          "${vcd_flags[@]}" \
          --seed "$seed"
    else
      echo "[reuse] $pred_jsonl"
    fi

    if ! reuse_file "$metrics_json"; then
      run_step "${variant}_seed${seed}_metrics" "$metrics_log" \
        env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/eval_pope_subset_yesno.py \
          --gt_csv "$SUBSET_GT_CSV" \
          --pred_jsonl "$pred_jsonl" \
          --pred_text_key text \
          --out_json "$metrics_json"
    else
      echo "[reuse] $metrics_json"
    fi

    if ! reuse_file "$compare_json"; then
      run_step "${variant}_seed${seed}_compare" "$compare_log" \
        env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/compare_pope_yesno_runs.py \
          --gt_csv "$SUBSET_GT_CSV" \
          --base_pred_jsonl "$BASELINE_PRED_JSONL" \
          --new_pred_jsonl "$pred_jsonl" \
          --pred_text_key auto \
          --out_json "$compare_json" \
          --out_fail_csv "$compare_csv"
    else
      echo "[reuse] $compare_json"
    fi
  done
done

echo "[4/4] summarize"
run_step "summarize" "$LOG_DIR/04_summarize.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/summarize_vcd_wrapper_stability_audit.py \
    --audit_dir "$OUT_ROOT" \
    --out_json "$SUMMARY_JSON"

echo "[done] $SUMMARY_JSON"
