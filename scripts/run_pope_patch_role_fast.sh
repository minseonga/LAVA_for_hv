#!/usr/bin/env bash
set -euo pipefail

# Fast role labeling via single-patch addback from keep-k base.
# Default is speed-oriented:
# - top_n_per_group=100
# - candidate_topn=32
# - candidate_mode=hybrid (top16 sim + random fill)

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
GPU="${GPU:-6}"

SAMPLES_CSV="${SAMPLES_CSV:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
TRACE_CSV="${TRACE_CSV:-/home/kms/LLaVA_calibration/experiments/pope_visual_disconnect_1000_alllayers_objpatch_pcs_v2/per_layer_yes_trace.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/pope_patch_role_fast}"

TARGET_LAYER="${TARGET_LAYER:-17}"
TARGET_GROUPS="${TARGET_GROUPS:-fp_hall,tp_yes}"
TOP_N_PER_GROUP="${TOP_N_PER_GROUP:-100}"
KEEP_K="${KEEP_K:-5}"
CANDIDATE_TOPN="${CANDIDATE_TOPN:-32}"
CANDIDATE_MODE="${CANDIDATE_MODE:-hybrid}"   # sim_top | random | hybrid
HYBRID_TOPM="${HYBRID_TOPM:-16}"
CANDIDATE_POOL="${CANDIDATE_POOL:-valid}"    # valid | objpool
OBJECT_PATCH_TOPK="${OBJECT_PATCH_TOPK:-64}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-8}"
ROLE_EPS="${ROLE_EPS:-0.05}"
SEED="${SEED:-42}"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "[run] pope patch role fast"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. python analyze_pope_patch_role_fast.py \
  --samples_csv "${SAMPLES_CSV}" \
  --per_layer_trace_csv "${TRACE_CSV}" \
  --image_root "${IMAGE_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --target_layer "${TARGET_LAYER}" \
  --target_groups "${TARGET_GROUPS}" \
  --top_n_per_group "${TOP_N_PER_GROUP}" \
  --sort_metric yes_sim_objpatch_max \
  --sort_desc true \
  --keep_k "${KEEP_K}" \
  --candidate_topn "${CANDIDATE_TOPN}" \
  --candidate_mode "${CANDIDATE_MODE}" \
  --hybrid_topm "${HYBRID_TOPM}" \
  --candidate_pool "${CANDIDATE_POOL}" \
  --object_patch_topk "${OBJECT_PATCH_TOPK}" \
  --exclude_padding_patches true \
  --mask_mode black \
  --batch_candidates "${BATCH_CANDIDATES}" \
  --role_eps "${ROLE_EPS}" \
  --seed "${SEED}"

echo "[done] ${OUT_DIR}"

