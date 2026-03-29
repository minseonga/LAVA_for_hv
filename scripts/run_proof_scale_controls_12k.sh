#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
BEAM_DIR="${BEAM_DIR:-/home/kms/LLaVA_calibration/experiments/proof_pairwise_full12k_tokenwise_latest/beam6_b6}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/proof_scale_controls_12k_tokenwise}"
EVAL_MODE="${EVAL_MODE:-heuristic}"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" /home/kms/LLaVA_calibration/prove_scale_invariance_and_controls.py \
  --beam_dir "${BEAM_DIR}" \
  --out_dir "${OUT_DIR}" \
  --eval_mode "${EVAL_MODE}"

echo "[done] ${OUT_DIR}"

