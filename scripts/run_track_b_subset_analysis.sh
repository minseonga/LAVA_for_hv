#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/track_b_subset_analysis}"

PER_CASE_CSV="${PER_CASE_CSV:-}"
PER_LAYER_TRACE_CSV="${PER_LAYER_TRACE_CSV:-}"
PER_HEAD_TRACE_CSV="${PER_HEAD_TRACE_CSV:-}"

LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
LAMBDA_D1="${LAMBDA_D1:-1.0}"
LAMBDA_BC="${LAMBDA_BC:-0.5}"
MAX_D1_RATE="${MAX_D1_RATE:-0.05}"
MAX_BC_RATE="${MAX_BC_RATE:-0.20}"
CANDIDATE_HEAD_METRIC="${CANDIDATE_HEAD_METRIC:-head_attn_vis_ratio}"
CANDIDATE_TOPK_GLOBAL="${CANDIDATE_TOPK_GLOBAL:-16}"
CANDIDATE_TOPK_PER_LAYER="${CANDIDATE_TOPK_PER_LAYER:-4}"

pick_first_existing() {
  for path in "$@"; do
    if [[ -n "$path" && -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

if [[ -z "$PER_CASE_CSV" ]]; then
  PER_CASE_CSV="$(pick_first_existing \
    "$ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/taxonomy/per_case_compare.csv" \
    "$ROOT/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv" \
  )" || true
fi

if [[ -z "$PER_LAYER_TRACE_CSV" ]]; then
  PER_LAYER_TRACE_CSV="$(pick_first_existing \
    "$ROOT/experiments/pope_visual_disconnect_1000_alllayers_objpatch_pcs_v2/per_layer_yes_trace.csv" \
    "$ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/traces_baseline/per_layer_yes_trace.csv" \
  )" || true
fi

if [[ -z "$PER_HEAD_TRACE_CSV" ]]; then
  PER_HEAD_TRACE_CSV="$(pick_first_existing \
    "$ROOT/experiments/pope_visual_disconnect_1000_headscan_l10_24/per_head_yes_trace.csv" \
    "$ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/traces_baseline/per_head_yes_trace.csv" \
  )" || true
fi

if [[ -z "$PER_CASE_CSV" || ! -f "$PER_CASE_CSV" ]]; then
  echo "[error] PER_CASE_CSV not found."
  echo "        export PER_CASE_CSV=/path/to/per_case_compare.csv"
  exit 2
fi

if [[ -z "$PER_LAYER_TRACE_CSV" || ! -f "$PER_LAYER_TRACE_CSV" ]]; then
  echo "[error] PER_LAYER_TRACE_CSV not found."
  echo "        export PER_LAYER_TRACE_CSV=/path/to/per_layer_yes_trace.csv"
  exit 2
fi

if [[ -z "$PER_HEAD_TRACE_CSV" || ! -f "$PER_HEAD_TRACE_CSV" ]]; then
  echo "[error] PER_HEAD_TRACE_CSV not found."
  echo "        export PER_HEAD_TRACE_CSV=/path/to/per_head_yes_trace.csv"
  exit 2
fi

mkdir -p "$OUT_DIR"
cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

"$PYTHON_BIN" "$ROOT/scripts/analyze_track_b_subset.py" \
  --per_case_csv "$PER_CASE_CSV" \
  --per_layer_trace_csv "$PER_LAYER_TRACE_CSV" \
  --per_head_trace_csv "$PER_HEAD_TRACE_CSV" \
  --late_start "$LATE_START" \
  --late_end "$LATE_END" \
  --lambda_d1 "$LAMBDA_D1" \
  --lambda_bc "$LAMBDA_BC" \
  --max_d1_rate "$MAX_D1_RATE" \
  --max_bc_rate "$MAX_BC_RATE" \
  --candidate_head_metric "$CANDIDATE_HEAD_METRIC" \
  --candidate_topk_global "$CANDIDATE_TOPK_GLOBAL" \
  --candidate_topk_per_layer "$CANDIDATE_TOPK_PER_LAYER" \
  --out_dir "$OUT_DIR"

echo "[done] $OUT_DIR"
