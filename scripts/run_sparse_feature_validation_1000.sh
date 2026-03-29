#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
GREEDY_DIR="${GREEDY_DIR:-/home/kms/LLaVA_calibration/experiments/feature_gen_1000/greedy_b1}"
BEAM_DIR="${BEAM_DIR:-/home/kms/LLaVA_calibration/experiments/feature_gen_1000/beam6_b6}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/feature_sparse_validate_1000}"
EVAL_MODE="${EVAL_MODE:-heuristic}"
K_PREFIX="${K_PREFIX:-2}"
K_SUFFIX="${K_SUFFIX:-2}"
FLIP_DEADBAND="${FLIP_DEADBAND:-0.05}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" /home/kms/LLaVA_calibration/validate_sparse_features_1000.py \
  --greedy_dir "${GREEDY_DIR}" \
  --beam_dir "${BEAM_DIR}" \
  --out_dir "${OUT_DIR}" \
  --eval_mode "${EVAL_MODE}" \
  --k_prefix "${K_PREFIX}" \
  --k_suffix "${K_SUFFIX}" \
  --flip_deadband "${FLIP_DEADBAND}"

echo "[done] ${OUT_DIR}"
