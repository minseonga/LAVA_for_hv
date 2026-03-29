#!/usr/bin/env bash
set -euo pipefail

# Experiment 2) K-curve for mask_selected / keep_selected
# Purpose: identify sparse-shortcut vs distributed-evidence behavior.

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
GPU="${GPU:-6}"

SAMPLES_CSV="${SAMPLES_CSV:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
TRACE_CSV="${TRACE_CSV:-/home/kms/LLaVA_calibration/experiments/pope_visual_disconnect_1000_alllayers_objpatch_pcs_v2/per_layer_yes_trace.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/pope_patch_kcurve}"

LAYER="${LAYER:-17}"
OBJECT_PATCH_TOPK="${OBJECT_PATCH_TOPK:-64}"
SEED="${SEED:-42}"

TARGET_GROUPS="${TARGET_GROUPS:-fp_hall tp_yes}"
ARMS="${ARMS:-mask_selected keep_selected}"
K_LIST="${K_LIST:-1 2 4 8 16 32}"

mkdir -p "${OUT_ROOT}"
export OUT_ROOT
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

for g in ${TARGET_GROUPS}; do
  for a in ${ARMS}; do
    for k in ${K_LIST}; do
      out_dir="${OUT_ROOT}/${g}_${a}_k${k}"
      mkdir -p "${out_dir}"
      echo "[run] group=${g} arm=${a} k=${k}"
      CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. python eval_pope_objpatch_mask_rerun.py \
        --samples_csv "${SAMPLES_CSV}" \
        --per_layer_trace_csv "${TRACE_CSV}" \
        --image_root "${IMAGE_ROOT}" \
        --out_dir "${out_dir}" \
        --model_path "${MODEL_PATH}" \
        --target_layer "${LAYER}" \
        --target_group "${g}" \
        --mask_topk_patches "${k}" \
        --object_patch_topk "${OBJECT_PATCH_TOPK}" \
        --exclude_padding_patches true \
        --mask_mode black \
        --mask_strategy "${a}" \
        --max_new_tokens 8 \
        --num_beams 1 \
        --temperature 0 \
        --top_p 1.0 \
        --seed "${SEED}"
    done
  done
done

python - << 'PY'
import os, json, csv, re
out_root = os.environ["OUT_ROOT"]
rows = []
pat = re.compile(r"(.+)_(mask_selected|keep_selected)_k(\d+)$")
for d in sorted(os.listdir(out_root)):
    m = pat.match(d)
    if not m:
        continue
    p = os.path.join(out_root, d, "summary.json")
    if not os.path.isfile(p):
        continue
    js = json.load(open(p))
    c = js.get("counts", {})
    mm = js.get("metrics", {})
    g, a, k = m.group(1), m.group(2), int(m.group(3))
    rows.append({
        "run": d,
        "group": g,
        "arm": a,
        "k": k,
        "n": c.get("n_valid"),
        "changed": c.get("n_changed_pred"),
        "repair": c.get("repair"),
        "harm": c.get("harm"),
        "delta_acc": mm.get("delta_acc"),
        "mean_drop_margin_yes_minus_no": mm.get("mean_drop_margin_yes_minus_no"),
    })
rows = sorted(rows, key=lambda r: (str(r["group"]), str(r["arm"]), int(r["k"])))
out_csv = os.path.join(out_root, "kcurve_summary.csv")
if rows:
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
print("[saved]", out_csv)
PY

echo "[done] ${OUT_ROOT}"
