#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_1000questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
SEED="${SEED:-42}"

# Gate used to choose B6 subset from B1 outputs.
GATES="${GATES:-gate_and_vpmi_m5.5_and_sfull_m6|and|-5.5|-6}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/feature_staged_subset_${STAMP}}"
mkdir -p "${OUT_ROOT}"

GREEDY_DIR="${OUT_ROOT}/greedy_b1"
BEAM_DIR="${OUT_ROOT}/beam6_b6_gatepass"
TMP_DIR="${OUT_ROOT}/_tmp"
mkdir -p "${GREEDY_DIR}" "${BEAM_DIR}" "${TMP_DIR}"

GATE_PASS_IDS_JSON="${TMP_DIR}/gate_pass_ids.json"
GATE_DEBUG_CSV="${TMP_DIR}/gate_debug.csv"
GATE_PASS_QUESTIONS_JSON="${TMP_DIR}/questions_gate_pass.json"

COMMON_ARGS=(
  --image_root "${IMAGE_ROOT}"
  --model_path "${MODEL_PATH}"
  --eval_match_mode heuristic
  --num_samples 0
  --save_core_tokenwise_vpmi
  --attn_impl "${ATTN_IMPL}"
  --seed "${SEED}"
)

echo "[1/4] Run B1 on full input subset"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  --questions_json "${QUESTIONS_JSON}" \
  "${COMMON_ARGS[@]}" \
  --out_dir "${GREEDY_DIR}" \
  --num_beams 1 \
  --num_return_sequences 1

echo "[2/4] Build gate-pass IDs from B1 per_sample"
"${PYTHON_BIN}" /home/kms/LLaVA_calibration/make_gate_pass_ids_from_greedy.py \
  --greedy_dir "${GREEDY_DIR}" \
  --gates "${GATES}" \
  --out_json "${GATE_PASS_IDS_JSON}" \
  --out_debug_csv "${GATE_DEBUG_CSV}"

echo "[3/4] Filter questions to gate-pass subset"
"${PYTHON_BIN}" /home/kms/LLaVA_calibration/filter_questions_by_ids.py \
  --questions_json "${QUESTIONS_JSON}" \
  --ids_json "${GATE_PASS_IDS_JSON}" \
  --out_json "${GATE_PASS_QUESTIONS_JSON}"

echo "[4/4] Run B6 only on gate-pass subset"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  --questions_json "${GATE_PASS_QUESTIONS_JSON}" \
  "${COMMON_ARGS[@]}" \
  --out_dir "${BEAM_DIR}" \
  --num_beams 6 \
  --num_return_sequences 6

echo "[done]"
echo "  greedy_b1: ${GREEDY_DIR}"
echo "  beam6_b6_gatepass: ${BEAM_DIR}"
echo "  gate_pass_ids: ${GATE_PASS_IDS_JSON}"

