#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
PYTHON_CMD="${PYTHON_CMD:-python}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
QUESTION_FILE="${QUESTION_FILE:-/tmp/pope_1000_q.jsonl}"
GT_CSV="${GT_CSV:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
HEADSET_JSON="${HEADSET_JSON:-/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json}"
ROLE_CSV="${ROLE_CSV:-/home/kms/LLaVA_calibration/experiments/pope_patch_role_fast/per_patch_role_effect.csv}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/pope_oracle_headaware_ablation_1000}"

CONV_MODE="${CONV_MODE:-llava_v1}"
TEMP="${TEMP:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
SUP_TOPK="${SUP_TOPK:-5}"
ASS_TOPK="${ASS_TOPK:-5}"
LAM_POS_LIST="${LAM_POS_LIST:-0.25}"
LAM_NEG_LIST="${LAM_NEG_LIST:-0.25}"
ARMS="${ARMS:-harmful_only,faithful_only,bipolar}"
CONTROLS="${CONTROLS:-random_patch_oracle,random_head_oracle,patch_only_oracle,gt_region_guidance_oracle}"
GT_REGION_ROLE_CSV="${GT_REGION_ROLE_CSV:-}"
CONTROL_SEED="${CONTROL_SEED:-42}"
N_HEADS="${N_HEADS:-32}"
SKIP_MAIN_ORACLE="${SKIP_MAIN_ORACLE:-0}"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

${PYTHON_CMD} scripts/run_pope_oracle_headaware_ablation.py \
  --repo_root "${REPO_ROOT}" \
  --python_cmd "${PYTHON_CMD}" \
  --model_path "${MODEL_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --question_file "${QUESTION_FILE}" \
  --gt_csv "${GT_CSV}" \
  --out_dir "${OUT_DIR}" \
  --headset_json "${HEADSET_JSON}" \
  --role_csv "${ROLE_CSV}" \
  --conv_mode "${CONV_MODE}" \
  --temperature "${TEMP}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --ais_late_start "${LATE_START}" \
  --ais_late_end "${LATE_END}" \
  --ais_oracle_supportive_topk "${SUP_TOPK}" \
  --ais_oracle_assertive_topk "${ASS_TOPK}" \
  --lambda_pos_list "${LAM_POS_LIST}" \
  --lambda_neg_list "${LAM_NEG_LIST}" \
  --arms "${ARMS}" \
  --controls "${CONTROLS}" \
  --gt_region_role_csv "${GT_REGION_ROLE_CSV}" \
  --control_seed "${CONTROL_SEED}" \
  --n_heads "${N_HEADS}" \
  --debug_dump \
  $([ "${SKIP_MAIN_ORACLE}" = "1" ] && echo "--skip_main_oracle")

echo "[done] ${OUT_DIR}"
