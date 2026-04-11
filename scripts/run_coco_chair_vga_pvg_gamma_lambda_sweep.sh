#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_vga_pvg_gamma_lambda_sweep_first_next_len512}"
SPLIT_SRC_ROOT="${SPLIT_SRC_ROOT:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512/splits}"
ANCHOR_ROOT="${ANCHOR_ROOT:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
GAMMA_GRID="${GAMMA_GRID:-0.15 0.20 0.25}"
LAMBDA_GRID="${LAMBDA_GRID:-0.01 0.02 0.04}"

REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_NO_PVG="${RUN_NO_PVG:-0}"
RUN_FULL_PVG="${RUN_FULL_PVG:-1}"

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

copy_splits() {
  local out_root="$1"
  local out_split="$out_root/splits"
  mkdir -p "$out_split"
  local name=""
  for name in summary.json val_caption_q.jsonl test_caption_q.jsonl val_images.csv test_images.csv; do
    if [[ ! -f "$SPLIT_SRC_ROOT/$name" ]]; then
      echo "[error] missing split source: $SPLIT_SRC_ROOT/$name" >&2
      exit 1
    fi
    cp "$SPLIT_SRC_ROOT/$name" "$out_split/$name"
  done
}

run_combo() {
  local gamma="$1"
  local lambda="$2"
  local gamma_tag=""
  local lambda_tag=""
  gamma_tag="$(sanitize_value "$gamma")"
  lambda_tag="$(sanitize_value "$lambda")"
  local out_root="$BASE_OUT_ROOT/gamma_${gamma_tag}_lambda_${lambda_tag}"

  echo "[combo] gamma=$gamma lambda=$lambda out=$out_root"
  copy_splits "$out_root"
  (
    cd "$CAL_ROOT"
    RUN_BASELINE="$RUN_BASELINE" \
    RUN_FULL_PVG="$RUN_FULL_PVG" \
    RUN_NO_PVG="$RUN_NO_PVG" \
    OUT_ROOT="$out_root" \
    VGA_ATTN_COEF="$gamma" \
    VGA_CD_ALPHA="$lambda" \
    REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
    bash scripts/run_coco_chair_vga_pvg_ablation.sh
  )
}

summarize_sweep() {
  mkdir -p "$BASE_OUT_ROOT/summary"
  "$CAL_PYTHON_BIN" - "$BASE_OUT_ROOT" "$ANCHOR_ROOT" "$GAMMA_GRID" "$LAMBDA_GRID" <<'PY'
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Set

base_out, anchor_root, gamma_grid, lambda_grid = sys.argv[1:5]


def sanitize(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def rate(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if abs(out) > 1.0:
        out /= 100.0
    return out


def obj(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[-1]).strip()
    return str(value).strip()


def obj_set(values: Iterable[Any]) -> Set[str]:
    return {obj(value) for value in values if obj(value)}


def obj_list(values: Iterable[Any]) -> list[str]:
    return [obj(value) for value in values if obj(value)]


def f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom <= 0.0 else float(2.0 * precision * recall / denom)


def recompute(sentences: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n_caps = 0
    n_words = 0
    n_hall_caps = 0
    n_hall = 0
    n_gen_inst = 0
    n_gen_unique = 0
    n_gt = 0
    n_supported = 0
    for row in sentences:
        caption = str(row.get("caption", "")).strip()
        generated = obj_list(row.get("mscoco_generated_words", []))
        gt = obj_set(row.get("mscoco_gt_words", []))
        hallucinated = obj_list(row.get("mscoco_hallucinated_words", []))
        if not hallucinated and generated:
            hallucinated = [x for x in generated if x not in gt]
        supported = {x for x in generated if x in gt}
        words = row.get("words")
        if isinstance(words, list):
            n_words += len(words)
        elif caption:
            n_words += len(caption.split())
        n_caps += 1
        n_hall_caps += int(bool(hallucinated))
        n_hall += len(hallucinated)
        n_gen_inst += len(generated)
        n_gen_unique += len(set(generated))
        n_gt += len(gt)
        n_supported += len(supported)
    precision = n_supported / float(n_gen_unique) if n_gen_unique else 0.0
    recall = n_supported / float(n_gt) if n_gt else 0.0
    return {
        "chair_s": n_hall_caps / float(n_caps) if n_caps else 0.0,
        "chair_i": n_hall / float(n_gen_inst) if n_gen_inst else 0.0,
        "recall": recall,
        "precision": precision,
        "f1": f1(precision, recall),
        "len_words": n_words / float(n_caps) if n_caps else 0.0,
        "avg_generated_object_mentions": n_gen_inst / float(n_caps) if n_caps else 0.0,
        "avg_generated_unique_objects": n_gen_unique / float(n_caps) if n_caps else 0.0,
        "avg_supported_unique_objects": n_supported / float(n_caps) if n_caps else 0.0,
        "avg_hallucinated_object_mentions": n_hall / float(n_caps) if n_caps else 0.0,
    }


def load_row(path: str, method: str, split: str, gamma: str, lambda_: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    payload = json.load(open(path, "r", encoding="utf-8"))
    overall = payload.get("overall_metrics", {})
    metrics = recompute(payload.get("sentences", []))
    return {
        "method": method,
        "split": split,
        "gamma": gamma,
        "lambda": lambda_,
        "chair_json": os.path.abspath(path),
        "CHAIRs": metrics["chair_s"],
        "CHAIRi": metrics["chair_i"],
        "Recall": metrics["recall"],
        "Precision": metrics["precision"],
        "F1": metrics["f1"],
        "Len_words": metrics["len_words"],
        "avg_generated_object_mentions": metrics["avg_generated_object_mentions"],
        "avg_generated_unique_objects": metrics["avg_generated_unique_objects"],
        "avg_supported_unique_objects": metrics["avg_supported_unique_objects"],
        "avg_hallucinated_object_mentions": metrics["avg_hallucinated_object_mentions"],
        "reported_CHAIRs": rate(overall.get("CHAIRs")),
        "reported_CHAIRi": rate(overall.get("CHAIRi")),
        "reported_Recall": rate(overall.get("Recall")),
    }


rows: list[Dict[str, Any]] = []
for split in ("val", "test"):
    rows.append(load_row(os.path.join(anchor_root, split, "chair_baseline.json"), "baseline_anchor", split, "", ""))
    rows.append(load_row(os.path.join(anchor_root, split, "chair_vga_no_pvg.json"), "vga_use_add_false_anchor", split, "0.20", "0.02"))

for gamma in gamma_grid.split():
    for lambda_ in lambda_grid.split():
        combo = os.path.join(base_out, f"gamma_{sanitize(gamma)}_lambda_{sanitize(lambda_)}")
        for split in ("val", "test"):
            rows.append(load_row(os.path.join(combo, split, "chair_vga_full_pvg.json"), "vga_full_pvg", split, gamma, lambda_))

rows = [row for row in rows if row is not None]
rows.sort(key=lambda r: (str(r["split"]), str(r["method"]), str(r["gamma"]), str(r["lambda"])))

out_csv = os.path.join(base_out, "summary", "gamma_lambda_sweep.csv")
out_json = os.path.join(base_out, "summary", "gamma_lambda_sweep.json")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
fieldnames = list(rows[0].keys()) if rows else []
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"rows": rows, "outputs": {"csv": os.path.abspath(out_csv), "json": os.path.abspath(out_json)}}, f, ensure_ascii=False, indent=2)
print("[saved]", out_csv)
print("[saved]", out_json)
PY
}

mkdir -p "$BASE_OUT_ROOT"
echo "[sweep] base_out=$BASE_OUT_ROOT"
echo "[sweep] split_src=$SPLIT_SRC_ROOT"
echo "[sweep] anchor_root=$ANCHOR_ROOT"
echo "[sweep] gamma_grid=$GAMMA_GRID"
echo "[sweep] lambda_grid=$LAMBDA_GRID"

read -r -a gamma_values <<< "$GAMMA_GRID"
read -r -a lambda_values <<< "$LAMBDA_GRID"

for gamma in "${gamma_values[@]}"; do
  for lambda in "${lambda_values[@]}"; do
    run_combo "$gamma" "$lambda"
  done
done

summarize_sweep
echo "[done] $BASE_OUT_ROOT"
