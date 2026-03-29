#!/usr/bin/env bash
set -euo pipefail

# Cost-aware FAD experiment matrix
# - Stage A (default): offline trigger/policy comparison on existing in_dir
# - Stage B (optional): compact-pool generation (beam4 + extra2 samples), then eval

GPU="${GPU:-0}"
TAG="${TAG:-costaware_v1}"

IN_DIR="${IN_DIR:-/home/kms/LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat}"
BASE_OUT="/home/kms/LLaVA_calibration/experiments/artrap_${TAG}"

RUN_GENERATION="${RUN_GENERATION:-0}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_1000questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"

echo "[info] GPU=${GPU}"
echo "[info] IN_DIR=${IN_DIR}"
echo "[info] BASE_OUT=${BASE_OUT}"
mkdir -p "${BASE_OUT}"

############################################
# Stage A: Offline low-cost sweep
############################################
OFFLINE_OUT="${BASE_OUT}/offline_sweep"
mkdir -p "${OFFLINE_OUT}"

python /home/kms/LLaVA_calibration/eval_selector_tradeoff.py \
  --in_dir "${IN_DIR}" \
  --out_dir "${OFFLINE_OUT}" \
  --eval_mode heuristic \
  --triggers "P3,P3C_cvlt:-0.5,P3Q_pctl:70,P3Q_pctl:80,P3Q_pctl:90" \
  --policies "max_vpmi;agree_vminpm_wmin_dfull_le:-0.08;agree_vminpm_wmin_dfull_le:-0.05;agree_vminpm_wmin;max_vpmi_core_min_prior_masked_tb_vpmi"

python /home/kms/LLaVA_calibration/eval_policy_diagnostics.py \
  --in_dir "${IN_DIR}" \
  --out_dir "${OFFLINE_OUT}/diagnostics" \
  --eval_mode heuristic \
  --configs "ref_max_vpmi|P3|max_vpmi;holdout_locked|P3C_cvlt:-0.5|agree_vminpm_wmin_dfull_le:-0.08;dynamic_p80|P3Q_pctl:80|agree_vminpm_wmin_dfull_le:-0.08;leaky_best|P3|agree_vminpm_wmin_dfull_le:-0.05" \
  --ref_name ref_max_vpmi \
  --bootstrap_n 5000 \
  --bootstrap_seed 123

echo "[done] Stage A offline outputs at ${OFFLINE_OUT}"

############################################
# Stage B: Optional generation rerun
############################################
if [[ "${RUN_GENERATION}" == "1" ]]; then
  GEN_OUT="${BASE_OUT}/gen_beam4_extra2"
  mkdir -p "${GEN_OUT}"
  CUDA_VISIBLE_DEVICES="${GPU}" python /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
    --questions_json "${QUESTIONS_JSON}" \
    --image_root "${IMAGE_ROOT}" \
    --out_dir "${GEN_OUT}" \
    --model_path "${MODEL_PATH}" \
    --attn_impl sdpa \
    --num_beams 4 \
    --num_return_sequences 4 \
    --num_extra_samples 2 \
    --extra_sample_temperature 1.0 \
    --extra_sample_top_p 0.9 \
    --extra_sample_top_k 0 \
    --max_new_tokens 24 \
    --beta_q 0.8 \
    --tau_gap 0.65 \
    --eval_match_mode heuristic \
    --vpmi_min_mode prior_masked

  python /home/kms/LLaVA_calibration/eval_selector_tradeoff.py \
    --in_dir "${GEN_OUT}" \
    --out_dir "${GEN_OUT}/selector_eval" \
    --eval_mode heuristic \
    --triggers "P3,P3C_cvlt:-0.5,P3Q_pctl:70,P3Q_pctl:80,P3Q_pctl:90" \
    --policies "max_vpmi;agree_vminpm_wmin_dfull_le:-0.08;agree_vminpm_wmin_dfull_le:-0.05;agree_vminpm_wmin;max_vpmi_core_min_prior_masked_tb_vpmi"

  python /home/kms/LLaVA_calibration/eval_policy_diagnostics.py \
    --in_dir "${GEN_OUT}" \
    --out_dir "${GEN_OUT}/policy_diagnostics" \
    --eval_mode heuristic \
    --configs "ref_max_vpmi|P3|max_vpmi;holdout_locked|P3C_cvlt:-0.5|agree_vminpm_wmin_dfull_le:-0.08;dynamic_p80|P3Q_pctl:80|agree_vminpm_wmin_dfull_le:-0.08;leaky_best|P3|agree_vminpm_wmin_dfull_le:-0.05" \
    --ref_name ref_max_vpmi \
    --bootstrap_n 5000 \
    --bootstrap_seed 123

  echo "[done] Stage B generation outputs at ${GEN_OUT}"
fi

echo "[all done] ${BASE_OUT}"
