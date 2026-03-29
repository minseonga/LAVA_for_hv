#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"

python /home/kms/LLaVA_calibration/analyze_artrap_pairwise_fragility.py \
  --questions_json /home/kms/LLaVA_calibration/testdev_balanced_questions_seed42_200questions.json \
  --image_root /home/kms/data/gqa/images \
  --out_dir /home/kms/LLaVA_calibration/experiments/artrap_fragility_200 \
  --model_path liuhaotian/llava-v1.5-7b \
  --num_beams 6 \
  --num_return_sequences 6 \
  --max_new_tokens 24 \
  --answer_span_max_tokens 4 \
  --beta_q 0.80 \
  --tau_gap 0.65 \
  --eval_match_mode heuristic \
  --attn_impl sdpa \
  --num_samples 200 \
  --seed 42
