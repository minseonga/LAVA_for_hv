#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
PYTHON_CMD="${PYTHON_CMD:-python}"
# GPU selection priority:
# 1) explicit GPU env var
# 2) inherited CUDA_VISIBLE_DEVICES
# 3) fallback 6
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"

# Step 1: build RefCOCO QA/GT/role inputs
REFCOCO_ROOT="${REFCOCO_ROOT:-/home/kms/data/refcoco_data}"
DATASET="${DATASET:-refcoco}"
SPLITS="${SPLITS:-val,testA,testB,test}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
SUBSET_SIZE="${SUBSET_SIZE:-1000}"
SEED="${SEED:-42}"
GRID_SIZE="${GRID_SIZE:-24}"
SUP_TOPK="${SUP_TOPK:-16}"
ASS_TOPK="${ASS_TOPK:-16}"
INPUT_OUT_DIR="${INPUT_OUT_DIR:-/home/kms/LLaVA_calibration/experiments/refcoco_gt_region_inputs_${DATASET}_${SUBSET_SIZE}}"

# Step 2: run oracle (GT-region guidance only)
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
HEADSET_JSON="${HEADSET_JSON:-/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json}"
ORACLE_OUT_DIR="${ORACLE_OUT_DIR:-/home/kms/LLaVA_calibration/experiments/refcoco_gt_region_oracle_only_${DATASET}_${SUBSET_SIZE}}"
ARMS="${ARMS:-bipolar}"
LAM_POS_LIST="${LAM_POS_LIST:-0.25}"
LAM_NEG_LIST="${LAM_NEG_LIST:-0.25}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
N_HEADS="${N_HEADS:-32}"

cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

mkdir -p "${INPUT_OUT_DIR}"
${PYTHON_CMD} scripts/make_refcoco_gt_region_oracle_inputs.py \
  --refcoco_root "${REFCOCO_ROOT}" \
  --dataset "${DATASET}" \
  --splits "${SPLITS}" \
  --image_folder "${IMAGE_FOLDER}" \
  --ensure_image_exists \
  --subset_size "${SUBSET_SIZE}" \
  --seed "${SEED}" \
  --grid_size "${GRID_SIZE}" \
  --supportive_topk "${SUP_TOPK}" \
  --assertive_topk "${ASS_TOPK}" \
  --out_dir "${INPUT_OUT_DIR}"

SUMMARY_JSON="${INPUT_OUT_DIR}/summary.json"
if [[ ! -f "${SUMMARY_JSON}" ]]; then
  echo "[error] missing input summary: ${SUMMARY_JSON}"
  exit 1
fi

QUESTIONS_JSONL=$(${PYTHON_CMD} - "${SUMMARY_JSON}" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("outputs",{}).get("questions_jsonl",""))
PY
)
GT_CSV=$(${PYTHON_CMD} - "${SUMMARY_JSON}" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("outputs",{}).get("gt_csv",""))
PY
)
ROLE_CSV=$(${PYTHON_CMD} - "${SUMMARY_JSON}" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("outputs",{}).get("gt_region_role_csv",""))
PY
)

if [[ -z "${QUESTIONS_JSONL}" || -z "${GT_CSV}" || -z "${ROLE_CSV}" ]]; then
  echo "[error] failed to resolve generated input files"
  exit 1
fi
if [[ ! -s "${QUESTIONS_JSONL}" || ! -s "${GT_CSV}" || ! -s "${ROLE_CSV}" ]]; then
  echo "[error] one or more generated input files are empty"
  echo "  QUESTIONS_JSONL=${QUESTIONS_JSONL}"
  echo "  GT_CSV=${GT_CSV}"
  echo "  ROLE_CSV=${ROLE_CSV}"
  exit 1
fi

echo "[info] QUESTIONS_JSONL=${QUESTIONS_JSONL}"
echo "[info] GT_CSV=${GT_CSV}"
echo "[info] ROLE_CSV=${ROLE_CSV}"
echo "[info] GPU=${GPU}"

CUDA_VISIBLE_DEVICES="${GPU}" \
CONDA_ENV="${CONDA_ENV}" \
MODEL_PATH="${MODEL_PATH}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
QUESTION_FILE="${QUESTIONS_JSONL}" \
GT_CSV="${GT_CSV}" \
HEADSET_JSON="${HEADSET_JSON}" \
ROLE_CSV="${ROLE_CSV}" \
GT_REGION_ROLE_CSV="${ROLE_CSV}" \
OUT_DIR="${ORACLE_OUT_DIR}" \
CONTROLS="gt_region_guidance_oracle" \
ARMS="${ARMS}" \
LAM_POS_LIST="${LAM_POS_LIST}" \
LAM_NEG_LIST="${LAM_NEG_LIST}" \
LATE_START="${LATE_START}" \
LATE_END="${LATE_END}" \
N_HEADS="${N_HEADS}" \
SKIP_MAIN_ORACLE=1 \
bash scripts/run_pope_oracle_headaware_ablation.sh

echo "[done] ${ORACLE_OUT_DIR}"
