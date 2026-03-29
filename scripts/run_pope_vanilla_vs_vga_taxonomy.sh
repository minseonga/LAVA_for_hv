#!/usr/bin/env bash
set -euo pipefail

# Example:
# CUDA_VISIBLE_DEVICES=6 \
#   bash /home/kms/LLaVA_calibration/scripts/run_pope_vanilla_vs_vga_taxonomy.sh

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
PYTHON_CMD="${PYTHON_CMD:-python}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
QUESTION_FILE="${QUESTION_FILE:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_q.jsonl}"
GT_CSV="${GT_CSV:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_gt.csv}"
VGA_PRED_JSONL="${VGA_PRED_JSONL:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/vga_baseline_9000/pred_vga_single_9000.jsonl}"

OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/vanilla_vs_vga_taxonomy}"
VANILLA_PRED_JSONL="${VANILLA_PRED_JSONL:-${OUT_DIR}/pred_vanilla_9000.jsonl}"
OBJECT_PRIOR_THR="${OBJECT_PRIOR_THR:-0.55}"

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "[1/3] Run Vanilla LLaVA-1.5 on POPE 9000"
PYTHONPATH="${REPO_ROOT}" ${PYTHON_CMD} -m llava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" \
  --image-folder "${IMAGE_FOLDER}" \
  --question-file "${QUESTION_FILE}" \
  --answers-file "${VANILLA_PRED_JSONL}" \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 8

echo "[2/3] Evaluate Vanilla metrics"
${PYTHON_CMD} scripts/eval_pope_subset_yesno.py \
  --gt_csv "${GT_CSV}" \
  --pred_jsonl "${VANILLA_PRED_JSONL}" \
  --pred_text_key text \
  --out_json "${OUT_DIR}/metrics_vanilla.json"

echo "[3/3] Build VGA failure taxonomy vs Vanilla"
${PYTHON_CMD} scripts/build_vga_failure_taxonomy.py \
  --gt_csv "${GT_CSV}" \
  --baseline_pred_jsonl "${VANILLA_PRED_JSONL}" \
  --vga_pred_jsonl "${VGA_PRED_JSONL}" \
  --baseline_pred_text_key text \
  --vga_pred_text_key output \
  --object_prior_thr "${OBJECT_PRIOR_THR}" \
  --out_dir "${OUT_DIR}/taxonomy"

echo "[done] ${OUT_DIR}"
