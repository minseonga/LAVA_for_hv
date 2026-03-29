#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/ab_bundle_validate_200v2}"
BEAM_DIR="${BEAM_DIR:-/home/kms/LLaVA_calibration/experiments/feature_gen_1000/beam6_b6}"
BASELINE_PER_SAMPLE="${BASELINE_PER_SAMPLE:-/home/kms/LLaVA_calibration/experiments/feature_gen_1000/greedy_b1/per_sample.csv}"
RUN200_ROOT="${RUN200_ROOT:-/home/kms/LLaVA_calibration/experiments/dualstream_ablation_200_v2}"
RUN1000_ROOT="${RUN1000_ROOT:-}"

RUN_REPLAY="${RUN_REPLAY:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
QUESTIONS_JSON="${QUESTIONS_JSON:-/home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_1000questions.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/gqa/images}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"

mkdir -p "${OUT_DIR}"

CMD=(
  /home/kms/miniconda3/envs/vocot/bin/python /home/kms/LLaVA_calibration/validate_ab_hypothesis_bundle.py
  --beam_dir "${BEAM_DIR}"
  --out_dir "${OUT_DIR}"
  --baseline_per_sample "${BASELINE_PER_SAMPLE}"
  --run200_root "${RUN200_ROOT}"
)

if [[ -n "${RUN1000_ROOT}" ]]; then
  CMD+=( --run1000_root "${RUN1000_ROOT}" )
fi

if [[ "${RUN_REPLAY}" == "1" ]]; then
  CMD+=(
    --run_replay
    --questions_json "${QUESTIONS_JSON}"
    --image_root "${IMAGE_ROOT}"
    --model_path "${MODEL_PATH}"
    --replay_num_samples "${REPLAY_NUM_SAMPLES:-200}"
    --replay_max_steps "${REPLAY_MAX_STEPS:-16}"
    --top_k_rank "${TOP_K_RANK:-50}"
    --alpha_rank "${ALPHA_RANK:-1.2}"
    --state_top_k "${STATE_TOP_K:-30}"
    --prefix_spike_z "${PREFIX_SPIKE_Z:-1.8}"
    --beta_prefix "${BETA_PREFIX:-0.25}"
    --suffix_vpmi_floor "${SUFFIX_VPMI_FLOOR:-0.0}"
    --beta_suffix "${BETA_SUFFIX:-0.2}"
    --tau_low_margin "${TAU_LOW_MARGIN:-1.0}"
    --tau_rank_gap "${TAU_RANK_GAP:-0.15}"
    --b_core_max_steps "${B_CORE_MAX_STEPS:-4}"
    --tau_suffix_collapse "${TAU_SUFFIX_COLLAPSE:--0.3}"
    --attn_impl "${ATTN_IMPL:-auto}"
  )
fi

echo "[run] ${CMD[*]}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}"
echo "[done] ${OUT_DIR}"

