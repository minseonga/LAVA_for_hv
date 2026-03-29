#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for 4 patch experiments:
# 1) keep_random control
# 2) K-curve
# 3) add-back
# 4) semantic identity (optional overlap if INSTANCES_JSON is provided)

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
GPU="${GPU:-6}"
REFCOCO_ROOT="${REFCOCO_ROOT:-/home/kms/data/refcoco_data}"
INSTANCES_JSON="${INSTANCES_JSON:-}"

if [[ -z "${INSTANCES_JSON}" ]]; then
  for cand in \
    "${REFCOCO_ROOT}/refcoco/instances.json" \
    "${REFCOCO_ROOT}/refcoco+/instances.json" \
    "${REFCOCO_ROOT}/refcocog/instances.json"; do
    if [[ -f "${cand}" ]]; then
      INSTANCES_JSON="${cand}"
      break
    fi
  done
fi

cd "${REPO_ROOT}"

echo "[config] GPU=${GPU}"
if [[ -n "${INSTANCES_JSON}" ]]; then
  echo "[config] INSTANCES_JSON=${INSTANCES_JSON}"
else
  echo "[warn] INSTANCES_JSON not found under ${REFCOCO_ROOT}; semantic overlap labels may be unavailable."
fi

echo "[1/4] keep_random control"
GPU="${GPU}" bash scripts/run_pope_patch_keep_random_control.sh

echo "[2/4] K-curve"
GPU="${GPU}" bash scripts/run_pope_patch_kcurve.sh

echo "[3/4] add-back"
GPU="${GPU}" bash scripts/run_pope_patch_addback.sh

echo "[4/4] semantic identity"
GPU="${GPU}" INSTANCES_JSON="${INSTANCES_JSON}" bash scripts/run_pope_patch_semantic_identity.sh

echo "[done] all 4 experiments finished"
