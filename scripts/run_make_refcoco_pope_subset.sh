#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
PYTHON_CMD="${PYTHON_CMD:-python}"

POPE_Q_JSONL="${POPE_Q_JSONL:-/tmp/pope_1000_q.jsonl}"
POPE_GT_CSV="${POPE_GT_CSV:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
POPE_ROLE_CSV="${POPE_ROLE_CSV:-/home/kms/LLaVA_calibration/experiments/pope_patch_role_fast/per_patch_role_effect.csv}"

REFCOCO_ROOT="${REFCOCO_ROOT:-/home/kms/data/refcoco_data}"
REF_SETS="${REF_SETS:-refcoco,refcoco+,refcocog}"
SUBSET_SIZE="${SUBSET_SIZE:-0}"
FALLBACK_REFONLY_SIZE="${FALLBACK_REFONLY_SIZE:-1000}"
SEED="${SEED:-42}"

OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/refcoco_pope_subset_from_pope1000}"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

${PYTHON_CMD} scripts/make_refcoco_pope_subset.py \
  --pope_question_jsonl "${POPE_Q_JSONL}" \
  --pope_gt_csv "${POPE_GT_CSV}" \
  --pope_role_csv "${POPE_ROLE_CSV}" \
  --refcoco_root "${REFCOCO_ROOT}" \
  --ref_sets "${REF_SETS}" \
  --subset_size "${SUBSET_SIZE}" \
  --fallback_refonly_size "${FALLBACK_REFONLY_SIZE}" \
  --seed "${SEED}" \
  --out_dir "${OUT_DIR}"

echo "[done] ${OUT_DIR}"

