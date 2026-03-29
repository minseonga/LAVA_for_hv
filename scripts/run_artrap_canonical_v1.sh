#!/usr/bin/env bash
set -euo pipefail

# Canonical non-DBS run + selector/trigger sweep for paper-grade fair comparison.
#
# Usage:
#   bash /home/kms/LLaVA_calibration/scripts/run_artrap_canonical_v1.sh
#   GPU=4 TAG=canonical_v1_sdpa SWEEP_MODE=focused bash /home/kms/LLaVA_calibration/scripts/run_artrap_canonical_v1.sh
#
# Env vars:
#   GPU          CUDA device id (default: 0)
#   TAG          output tag suffix (default: canonical_v1_sdpa)
#   SWEEP_MODE   focused | full (default: focused)

ROOT="/home/kms/LLaVA_calibration"
GPU="${GPU:-0}"
TAG="${TAG:-canonical_v1_sdpa}"
SWEEP_MODE="${SWEEP_MODE:-focused}"

OUT_DIR="${ROOT}/experiments/artrap_fragility_1000_${TAG}"
EVAL_OUT_DIR="${OUT_DIR}_selector_eval"

QUESTIONS_JSON="${ROOT}/testdev_balanced_questions_seed42_1000questions.json"
IMAGE_ROOT="/home/kms/data/gqa/images"
MODEL_PATH="liuhaotian/llava-v1.5-7b"

TRIGGERS="P3,P5,P3C_cvlt:0.0,P3C_cvlt:0.25,P3C_cvlt:0.5,P3C_cvlt:0.75,P3C_cvlt:1.0,P3CI_cvlt:0.5_ig:1.0,P3CI_cvlt:1.0_ig:1.0,P3CI_cvlt:1.0_ig:1.5,P3AF_alpha:1.0,P3AF_alpha:1.5,P3AF_alpha:2.0"
FOCUSED_POLICIES="max_vpmi,max_vpmi_core_min_raw,max_vpmi_core_min_prior_masked,max_vpmi_core_min_prior_masked_tb_vpmi,max_vpmi_word_min,max_vpmi_core_tail_min,max_vpmi_core_tail_min_tb_vpmi,max_vpmi_core_tail_stable,agree_tailmin_wordmin,agree_vminpm_wmin,agree_vminpm_wmin_dfull_le:-0.05,rankmix:1,1,1,0,max_vpmi_top2_visual,vpmi_pd_lambda:0.75,max_vpmi_sfull_topk:3,fixE_sq_clip:-6.0"

echo "[1/3] analyze -> ${OUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/analyze_artrap_pairwise_fragility.py" \
  --questions_json "${QUESTIONS_JSON}" \
  --image_root "${IMAGE_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --num_beams 6 \
  --num_beam_groups 1 \
  --diversity_penalty 0.0 \
  --num_return_sequences 6 \
  --max_new_tokens 24 \
  --beta_q 0.8 \
  --tau_gap 0.65 \
  --eval_match_mode heuristic \
  --attn_impl sdpa \
  --vpmi_min_mode raw

echo "[2/3] selector eval (${SWEEP_MODE}) -> ${EVAL_OUT_DIR}"
if [[ "${SWEEP_MODE}" == "full" ]]; then
  python "${ROOT}/eval_selector_tradeoff.py" \
    --in_dir "${OUT_DIR}" \
    --out_dir "${EVAL_OUT_DIR}" \
    --eval_mode heuristic \
    --triggers "${TRIGGERS}"
else
  python "${ROOT}/eval_selector_tradeoff.py" \
    --in_dir "${OUT_DIR}" \
    --out_dir "${EVAL_OUT_DIR}" \
    --eval_mode heuristic \
    --triggers "${TRIGGERS}" \
    --policies "${FOCUSED_POLICIES}"
fi

echo "[3/3] quick top-10 report"
EVAL_TABLE="${EVAL_OUT_DIR}/policy_table.csv"
python - "$EVAL_TABLE" <<'PY'
import csv, os, sys
target = sys.argv[1]
if not os.path.isfile(target):
    raise SystemExit(f"policy_table.csv not found: {target}")
rows = list(csv.DictReader(open(target, encoding="utf-8")))
rows = sorted(rows, key=lambda r: float(r["delta_acc"]), reverse=True)
print("policy_table:", target)
for r in rows[:10]:
    print(r["trigger"], r["policy"], "delta", r["delta_acc"], "gain", r["gain"], "harm", r["harm"], "final", r["final_acc"])
PY

echo "[done] canonical run ready"
