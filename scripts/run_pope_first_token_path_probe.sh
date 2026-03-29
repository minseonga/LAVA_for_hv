#!/usr/bin/env bash
set -euo pipefail

# First-token path dominance diagnostic on POPE subset.
# Conditions:
#   1) baseline
#   2) drop_img   (weaken image->output-text path)
#   3) drop_text  (weaken text->text path)
#   4) drop_both
#
# Usage:
#   bash /home/kms/LLaVA_calibration/scripts/run_pope_first_token_path_probe.sh
# Optional env:
#   GPU=6
#   PYTHON_BIN=python
#   MODEL_PATH=liuhaotian/llava-v1.5-7b
#   IMAGE_ROOT=/home/kms/data/pope/val2014
#   QFILE=/tmp/pope_1000_q.jsonl
#   GT_CSV=/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv
#   OUT_DIR=/home/kms/LLaVA_calibration/experiments/pope_path_probe_firsttoken_1000
#   PROBE_PENALTY=8.0
#   MAX_NEW_TOKENS=8

GPU="${GPU:-6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
QFILE="${QFILE:-/tmp/pope_1000_q.jsonl}"
GT_CSV="${GT_CSV:-/home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
OUT_DIR="${OUT_DIR:-/home/kms/LLaVA_calibration/experiments/pope_path_probe_firsttoken_1000}"
PROBE_PENALTY="${PROBE_PENALTY:-8.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
export OUT_DIR

mkdir -p "${OUT_DIR}"

run_case () {
  local name="$1"
  local mode="$2"

  echo "[run] ${name} (mode=${mode})"
  local extra=()
  if [[ "${mode}" != "none" ]]; then
    extra+=(
      --path-probe-mode "${mode}"
      --path-probe-penalty "${PROBE_PENALTY}"
      --path-probe-first-step-only
      --first-token-logits-include-pre
      --ais-debug-log
      --ais-debug-dump "${OUT_DIR}/${name}_debug.csv"
    )
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --image-folder "${IMAGE_ROOT}" \
    --question-file "${QFILE}" \
    --answers-file "${OUT_DIR}/${name}.jsonl" \
    --conv-mode llava_v1 \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --dump-first-token-logits "${OUT_DIR}/${name}_first_token.csv" \
    "${extra[@]}"

  "${PYTHON_BIN}" /home/kms/LLaVA_calibration/scripts/eval_pope_subset_yesno.py \
    --gt_csv "${GT_CSV}" \
    --pred_jsonl "${OUT_DIR}/${name}.jsonl" \
    --id_col id \
    --label_col answer \
    --out_json "${OUT_DIR}/${name}_metrics.json"
}

run_case "baseline" "none"
run_case "drop_img" "drop_img"
run_case "drop_text" "drop_text"
run_case "drop_both" "drop_both"

echo "[summary] path dominance aggregation"
"${PYTHON_BIN}" - <<'PY'
import csv, json, os

out_dir = os.path.abspath(os.environ.get("OUT_DIR", "/home/kms/LLaVA_calibration/experiments/pope_path_probe_firsttoken_1000"))
runs = ["baseline", "drop_img", "drop_text", "drop_both"]

def load_ft(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            qid = str(r.get("question_id", "")).strip()
            if not qid:
                continue
            try:
                d[qid] = {
                    "margin_post": float(r.get("margin_post", "nan")),
                    "margin_pre": float(r.get("margin_pre", "nan")) if r.get("margin_pre", "") != "" else None,
                    "margin_pre_nogate": float(r.get("margin_pre_nogate", "nan")) if r.get("margin_pre_nogate", "") != "" else None,
                }
            except Exception:
                continue
    return d

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

base = load_ft(os.path.join(out_dir, "baseline_first_token.csv"))
summary = {
    "inputs": {"out_dir": out_dir},
    "per_run_metrics": {},
    "margin_shift_vs_baseline": {},
}

for rn in runs:
    mpath = os.path.join(out_dir, f"{rn}_metrics.json")
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            summary["per_run_metrics"][rn] = json.load(f).get("metrics", {})
    cur = load_ft(os.path.join(out_dir, f"{rn}_first_token.csv"))
    common = sorted(set(base.keys()) & set(cur.keys()))
    dpost = [cur[k]["margin_post"] - base[k]["margin_post"] for k in common]
    summary["margin_shift_vs_baseline"][rn] = {
        "n_common": len(common),
        "mean_delta_margin_post": mean(dpost),
        "mean_abs_delta_margin_post": mean([abs(x) for x in dpost]),
    }

spath = os.path.join(out_dir, "path_probe_summary.json")
with open(spath, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[saved]", spath)
print(json.dumps(summary["margin_shift_vs_baseline"], ensure_ascii=False, indent=2))
PY

echo "[done] ${OUT_DIR}"
