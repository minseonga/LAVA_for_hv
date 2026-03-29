#!/usr/bin/env bash
set -euo pipefail

# Experiment 3) Removed-context add-back from keep_selected baseline.
# Purpose: test whether harmful-ranked removed patches cause rapid FP relapse.

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
GPU="${GPU:-6}"

SAMPLES_CSV="${SAMPLES_CSV:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
TRACE_CSV="${TRACE_CSV:-/home/kms/LLaVA_calibration/experiments/pope_visual_disconnect_1000_alllayers_objpatch_pcs_v2/per_layer_yes_trace.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/pope_patch_addback}"

LAYER="${LAYER:-17}"
KEEP_K="${KEEP_K:-5}"
OBJECT_PATCH_TOPK="${OBJECT_PATCH_TOPK:-64}"
SEED="${SEED:-42}"

TARGET_GROUPS="${TARGET_GROUPS:-fp_hall tp_yes}"
ADD_MODES="${ADD_MODES:-harmful random}"
ADD_K_LIST="${ADD_K_LIST:-1 2 4 8 16 32}"
RUN_BASE_KEEP="${RUN_BASE_KEEP:-1}"

mkdir -p "${OUT_ROOT}"
export OUT_ROOT
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

for g in ${TARGET_GROUPS}; do
  if [[ "${RUN_BASE_KEEP}" == "1" ]]; then
    base_dir="${OUT_ROOT}/${g}_keep_base_k${KEEP_K}"
    mkdir -p "${base_dir}"
    echo "[run] base keep_selected group=${g} keep_k=${KEEP_K}"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. python eval_pope_objpatch_mask_rerun.py \
      --samples_csv "${SAMPLES_CSV}" \
      --per_layer_trace_csv "${TRACE_CSV}" \
      --image_root "${IMAGE_ROOT}" \
      --out_dir "${base_dir}" \
      --model_path "${MODEL_PATH}" \
      --target_layer "${LAYER}" \
      --target_group "${g}" \
      --mask_topk_patches "${KEEP_K}" \
      --object_patch_topk "${OBJECT_PATCH_TOPK}" \
      --exclude_padding_patches true \
      --mask_mode black \
      --mask_strategy keep_selected \
      --max_new_tokens 8 \
      --num_beams 1 \
      --temperature 0 \
      --top_p 1.0 \
      --seed "${SEED}"
  fi

  for mode in ${ADD_MODES}; do
    for ak in ${ADD_K_LIST}; do
      out_dir="${OUT_ROOT}/${g}_keep${KEEP_K}_add${ak}_${mode}"
      mkdir -p "${out_dir}"
      echo "[run] group=${g} add_mode=${mode} add_k=${ak}"
      CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. python eval_pope_objpatch_mask_rerun.py \
        --samples_csv "${SAMPLES_CSV}" \
        --per_layer_trace_csv "${TRACE_CSV}" \
        --image_root "${IMAGE_ROOT}" \
        --out_dir "${out_dir}" \
        --model_path "${MODEL_PATH}" \
        --target_layer "${LAYER}" \
        --target_group "${g}" \
        --mask_topk_patches "${KEEP_K}" \
        --object_patch_topk "${OBJECT_PATCH_TOPK}" \
        --exclude_padding_patches true \
        --mask_mode black \
        --mask_strategy keep_selected_addback \
        --addback_mode "${mode}" \
        --addback_k "${ak}" \
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
pat = re.compile(r"(.+)_keep(\d+)_add(\d+)_(harmful|random)$")
for d in sorted(os.listdir(out_root)):
    p = os.path.join(out_root, d, "summary.json")
    if not os.path.isfile(p):
        continue
    js = json.load(open(p))
    c = js.get("counts", {})
    m = js.get("metrics", {})
    inp = js.get("inputs", {})
    mm = pat.match(d)
    row = {
        "run": d,
        "group": inp.get("target_group"),
        "mask_strategy": inp.get("mask_strategy"),
        "keep_k": inp.get("mask_topk_patches"),
        "addback_k": inp.get("addback_k"),
        "addback_mode": inp.get("addback_mode"),
        "n": c.get("n_valid"),
        "changed": c.get("n_changed_pred"),
        "repair": c.get("repair"),
        "harm": c.get("harm"),
        "delta_acc": m.get("delta_acc"),
        "mean_drop_margin_yes_minus_no": m.get("mean_drop_margin_yes_minus_no"),
    }
    if mm:
        row["group"] = mm.group(1)
        row["keep_k"] = int(mm.group(2))
        row["addback_k"] = int(mm.group(3))
        row["addback_mode"] = mm.group(4)
    rows.append(row)
rows = sorted(rows, key=lambda r: (str(r.get("group")), str(r.get("addback_mode")), float(r.get("addback_k") or -1)))
out_csv = os.path.join(out_root, "addback_summary.csv")
if rows:
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
print("[saved]", out_csv)
PY

echo "[done] ${OUT_ROOT}"
