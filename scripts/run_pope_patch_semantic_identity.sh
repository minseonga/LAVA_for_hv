#!/usr/bin/env bash
set -euo pipefail

# Experiment 4) Semantic identity of selected patches.
# Purpose: distinguish target evidence vs distractor/background cue.
# Note: COCO instances json is optional. Without it, overlap-based labels are skipped.

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"

TRACE_CSV="${TRACE_CSV:-/home/kms/LLaVA_calibration/experiments/pope_visual_disconnect_1000_alllayers_objpatch_pcs_v2/per_layer_yes_trace.csv}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/pope_patch_semantic_identity_l17}"
INSTANCES_JSON="${INSTANCES_JSON:-}"

LAYER="${LAYER:-17}"
TOPK="${TOPK:-5}"
IOU_THR_TARGET="${IOU_THR_TARGET:-0.1}"
IOU_THR_DISTRACTOR="${IOU_THR_DISTRACTOR:-0.1}"

cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

cmd=(python analyze_pope_patch_semantic_identity.py \
  --trace_csv "${TRACE_CSV}" \
  --out_dir "${OUT_DIR}" \
  --layer "${LAYER}" \
  --topk "${TOPK}" \
  --iou_thr_target "${IOU_THR_TARGET}" \
  --iou_thr_distractor "${IOU_THR_DISTRACTOR}")

if [[ -n "${INSTANCES_JSON}" ]]; then
  cmd+=(--instances_json "${INSTANCES_JSON}")
fi

"${cmd[@]}"

echo "[done] ${OUT_DIR}"
