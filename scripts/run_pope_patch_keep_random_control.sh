#!/usr/bin/env bash
set -euo pipefail

# Experiment 1) keep_selected vs keep_random (+ optional random_selected)
# Purpose: distinguish "selected patch sufficiency" vs "context-removal side-effect"

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
GPU="${GPU:-6}"

SAMPLES_CSV="${SAMPLES_CSV:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
TRACE_CSV="${TRACE_CSV:-/home/kms/LLaVA_calibration/experiments/pope_visual_disconnect_1000_alllayers_objpatch_pcs_v2/per_layer_yes_trace.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/pope_patch_keep_random_control}"

LAYER="${LAYER:-17}"
MASK_TOPK="${MASK_TOPK:-5}"
OBJECT_PATCH_TOPK="${OBJECT_PATCH_TOPK:-64}"
SEED="${SEED:-42}"

# target-group list and strategy list are space-separated
TARGET_GROUPS="${TARGET_GROUPS:-fp_hall tp_yes}"
STRATEGIES="${STRATEGIES:-keep_selected keep_random random_selected}"
RANDOM_POOL="${RANDOM_POOL:-valid}"

mkdir -p "${OUT_ROOT}"
export OUT_ROOT
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

for g in ${TARGET_GROUPS}; do
  for s in ${STRATEGIES}; do
    out_dir="${OUT_ROOT}/${g}_${s}"
    mkdir -p "${out_dir}"
    echo "[run] group=${g} strategy=${s} -> ${out_dir}"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. python eval_pope_objpatch_mask_rerun.py \
      --samples_csv "${SAMPLES_CSV}" \
      --per_layer_trace_csv "${TRACE_CSV}" \
      --image_root "${IMAGE_ROOT}" \
      --out_dir "${out_dir}" \
      --model_path "${MODEL_PATH}" \
      --target_layer "${LAYER}" \
      --target_group "${g}" \
      --mask_topk_patches "${MASK_TOPK}" \
      --object_patch_topk "${OBJECT_PATCH_TOPK}" \
      --exclude_padding_patches true \
      --mask_mode black \
      --mask_strategy "${s}" \
      --random_pool "${RANDOM_POOL}" \
      --max_new_tokens 8 \
      --num_beams 1 \
      --temperature 0 \
      --top_p 1.0 \
      --seed "${SEED}"
  done
done

python - << 'PY'
import os, json, csv
out_root = os.environ.get("OUT_ROOT", "/home/kms/LLaVA_calibration/experiments/pope_patch_keep_random_control")
rows = []
for d in sorted(os.listdir(out_root)):
    p = os.path.join(out_root, d, "summary.json")
    if not os.path.isfile(p):
        continue
    js = json.load(open(p))
    c = js.get("counts", {})
    m = js.get("metrics", {})
    inp = js.get("inputs", {})
    rows.append({
        "run": d,
        "group": inp.get("target_group"),
        "strategy": inp.get("mask_strategy"),
        "n": c.get("n_valid"),
        "changed": c.get("n_changed_pred"),
        "repair": c.get("repair"),
        "harm": c.get("harm"),
        "yes_to_no": c.get("yes_to_no"),
        "no_to_yes": c.get("no_to_yes"),
        "delta_acc": m.get("delta_acc"),
        "mean_drop_margin_yes_minus_no": m.get("mean_drop_margin_yes_minus_no"),
    })
out_csv = os.path.join(out_root, "control_summary.csv")
if rows:
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
print("[saved]", out_csv)
PY

echo "[done] ${OUT_ROOT}"
