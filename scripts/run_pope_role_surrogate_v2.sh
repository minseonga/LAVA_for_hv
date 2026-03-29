#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"

ROLE_CSV="${ROLE_CSV:-$REPO_ROOT/experiments/pope_patch_role_fast/per_patch_role_effect.csv}"
PER_LAYER_TRACE_CSV="${PER_LAYER_TRACE_CSV:-$REPO_ROOT/experiments/pope_visual_disconnect_1000_alllayers_objpatch_pcs_v2/per_layer_yes_trace.csv}"
PER_HEAD_TRACE_CSV="${PER_HEAD_TRACE_CSV:-$REPO_ROOT/experiments/pope_visual_disconnect_1000_headscan_l10_24/per_head_yes_trace.csv}"
HEADSET_JSON="${HEADSET_JSON:-$REPO_ROOT/experiments/pope_headsets_v1/headset.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/experiments/pope_role_surrogate_v2}"

EARLY_START="${EARLY_START:-10}"
EARLY_END="${EARLY_END:-15}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
SEED="${SEED:-42}"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

python analyze_pope_role_surrogate_v2.py \
  --role_csv "${ROLE_CSV}" \
  --per_layer_trace_csv "${PER_LAYER_TRACE_CSV}" \
  --per_head_trace_csv "${PER_HEAD_TRACE_CSV}" \
  --headset_json "${HEADSET_JSON}" \
  --out_dir "${OUT_DIR}" \
  --early_start "${EARLY_START}" \
  --early_end "${EARLY_END}" \
  --late_start "${LATE_START}" \
  --late_end "${LATE_END}" \
  --train_ratio "${TRAIN_RATIO}" \
  --seed "${SEED}"

echo "[done] ${OUT_DIR}"

