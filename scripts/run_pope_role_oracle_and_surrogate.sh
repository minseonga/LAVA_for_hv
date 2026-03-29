#!/usr/bin/env bash
set -euo pipefail

# GT-aware oracle + GT-free surrogate calibration runner

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/experiments/pope_role_oracle_and_surrogate}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
SAMPLES_CSV="${SAMPLES_CSV:-$REPO_ROOT/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
ROLE_CSV="${ROLE_CSV:-$REPO_ROOT/experiments/pope_patch_role_fast/per_patch_role_effect.csv}"

mkdir -p "$OUT_ROOT"

echo "[1/2] GT-aware oracle (rebalance)"
python "$REPO_ROOT/eval_pope_role_oracle_intervention.py" \
  --samples_csv "$SAMPLES_CSV" \
  --role_csv "$ROLE_CSV" \
  --image_root "$IMAGE_ROOT" \
  --out_dir "$OUT_ROOT/oracle_rebalance" \
  --model_path "$MODEL_PATH" \
  --conv_mode llava_v1 \
  --target_groups fp_hall,tp_yes \
  --top_n_per_group 0 \
  --oracle_arm rebalance \
  --tp_rebalance_policy assertive_mask \
  --protect_supportive true \
  --assertive_topk 5 \
  --supportive_topk 5 \
  --exclude_padding_patches true \
  --mask_mode black \
  --max_new_tokens 8 \
  --num_beams 1 \
  --temperature 0.0 \
  --top_p 1.0 \
  --seed 42

echo "[2/2] GT-free surrogate calibration (teacher -> rule)"
python "$REPO_ROOT/analyze_pope_role_surrogate.py" \
  --role_csv "$ROLE_CSV" \
  --out_dir "$OUT_ROOT/surrogate_rules" \
  --train_ratio 0.7 \
  --seed 42

echo "[done] $OUT_ROOT"
