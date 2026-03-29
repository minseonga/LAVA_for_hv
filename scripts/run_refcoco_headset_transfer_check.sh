#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
PYTHON_CMD="${PYTHON_CMD:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"

# Input bundle from make_refcoco_gt_region_oracle_inputs.py
INPUT_SUMMARY="${INPUT_SUMMARY:-/home/kms/LLaVA_calibration/experiments/refcoco_gt_region_inputs_refcoco_1000/summary.json}"

# Model / generation
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
CONV_MODE="${CONV_MODE:-llava_v1}"
TEMP="${TEMP:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

# Oracle settings
HEADSET_JSON="${HEADSET_JSON:-/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json}"
REFCOCO_HEADSET_JSON="${REFCOCO_HEADSET_JSON:-}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/refcoco_headset_transfer_check}"
ARMS="${ARMS:-bipolar}"
LAM_POS_LIST="${LAM_POS_LIST:-0.25}"
LAM_NEG_LIST="${LAM_NEG_LIST:-0.25}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
SUP_TOPK="${SUP_TOPK:-5}"
ASS_TOPK="${ASS_TOPK:-5}"
N_HEADS="${N_HEADS:-32}"
CONTROL_SEED="${CONTROL_SEED:-42}"

# Keep main + controls in one run
CONTROLS="${CONTROLS:-random_head_oracle,patch_only_oracle,gt_region_guidance_oracle}"
SKIP_MAIN_ORACLE="${SKIP_MAIN_ORACLE:-0}"

if [[ ! -f "${INPUT_SUMMARY}" ]]; then
  echo "[error] missing INPUT_SUMMARY: ${INPUT_SUMMARY}"
  exit 1
fi

cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

QUESTIONS_JSONL=$(${PYTHON_CMD} - "${INPUT_SUMMARY}" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("outputs",{}).get("questions_jsonl",""))
PY
)
GT_CSV=$(${PYTHON_CMD} - "${INPUT_SUMMARY}" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("outputs",{}).get("gt_csv",""))
PY
)
ROLE_CSV=$(${PYTHON_CMD} - "${INPUT_SUMMARY}" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("outputs",{}).get("gt_region_role_csv",""))
PY
)

if [[ -z "${QUESTIONS_JSONL}" || -z "${GT_CSV}" || -z "${ROLE_CSV}" ]]; then
  echo "[error] failed to resolve questions/gt/role from ${INPUT_SUMMARY}"
  exit 1
fi
if [[ ! -s "${QUESTIONS_JSONL}" || ! -s "${GT_CSV}" || ! -s "${ROLE_CSV}" ]]; then
  echo "[error] resolved input files are missing/empty"
  echo "  QUESTIONS_JSONL=${QUESTIONS_JSONL}"
  echo "  GT_CSV=${GT_CSV}"
  echo "  ROLE_CSV=${ROLE_CSV}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

run_one() {
  local label="$1"
  local headset_json="$2"
  local out_dir="${OUT_ROOT}/${label}"

  if [[ ! -f "${headset_json}" ]]; then
    echo "[warn] skip ${label}: missing headset ${headset_json}"
    return
  fi
  mkdir -p "${out_dir}"

  echo "[run] label=${label}"
  echo "      headset=${headset_json}"
  echo "      out_dir=${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  CONDA_ENV="${CONDA_ENV}" \
  MODEL_PATH="${MODEL_PATH}" \
  IMAGE_FOLDER="${IMAGE_FOLDER}" \
  QUESTION_FILE="${QUESTIONS_JSONL}" \
  GT_CSV="${GT_CSV}" \
  HEADSET_JSON="${headset_json}" \
  ROLE_CSV="${ROLE_CSV}" \
  GT_REGION_ROLE_CSV="${ROLE_CSV}" \
  OUT_DIR="${out_dir}" \
  ARMS="${ARMS}" \
  LAM_POS_LIST="${LAM_POS_LIST}" \
  LAM_NEG_LIST="${LAM_NEG_LIST}" \
  LATE_START="${LATE_START}" \
  LATE_END="${LATE_END}" \
  SUP_TOPK="${SUP_TOPK}" \
  ASS_TOPK="${ASS_TOPK}" \
  N_HEADS="${N_HEADS}" \
  CONTROL_SEED="${CONTROL_SEED}" \
  CONTROLS="${CONTROLS}" \
  SKIP_MAIN_ORACLE="${SKIP_MAIN_ORACLE}" \
  CONV_MODE="${CONV_MODE}" \
  TEMP="${TEMP}" \
  NUM_BEAMS="${NUM_BEAMS}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  bash scripts/run_pope_oracle_headaware_ablation.sh

  ${PYTHON_CMD} scripts/summarize_oracle_headset_transfer.py \
    --ablation_csv "${out_dir}/oracle_headaware_ablation.csv" \
    --out_csv "${out_dir}/transfer_summary.csv"
}

run_one "pope_headset" "${HEADSET_JSON}"
if [[ -n "${REFCOCO_HEADSET_JSON}" ]]; then
  run_one "refcoco_headset" "${REFCOCO_HEADSET_JSON}"
fi

echo "[done] ${OUT_ROOT}"
