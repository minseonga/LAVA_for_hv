#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import load_prediction_text_map, read_csv_rows, safe_id, write_json


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def load_feature_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        if sid:
            out[sid] = row
    return out


def load_chair_map(path: str, metric: str) -> Dict[str, Dict[str, Any]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    rows = obj.get("sentences", [])
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image_id = safe_id(row.get("image_id"))
        metrics = row.get("metrics", {})
        score = maybe_float(metrics.get(metric))
        if not image_id or score is None:
            continue
        out[image_id] = {
            "score": float(score),
            "caption": str(row.get("caption", "")).strip(),
            "metrics": metrics,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build generative method tables from sample-level CHAIR deltas.")
    ap.add_argument("--baseline_features_csv", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--method_name", type=str, required=True)
    ap.add_argument("--benchmark_name", type=str, default="pope_discovery_caption")
    ap.add_argument("--split_name", type=str, default="discovery")
    ap.add_argument("--chair_metric", type=str, default="CHAIRi")
    ap.add_argument("--epsilon", type=float, default=1e-12)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, required=True)
    ap.add_argument("--baseline_pred_text_key", type=str, default="auto")
    ap.add_argument("--intervention_pred_text_key", type=str, default="auto")
    args = ap.parse_args()

    feature_map = load_feature_map(os.path.abspath(args.baseline_features_csv))
    baseline_map = load_prediction_text_map(os.path.abspath(args.baseline_pred_jsonl), args.baseline_pred_text_key)
    intervention_map = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), args.intervention_pred_text_key)
    chair_base = load_chair_map(os.path.abspath(args.baseline_chair_json), args.chair_metric)
    chair_int = load_chair_map(os.path.abspath(args.intervention_chair_json), args.chair_metric)

    rows: List[Dict[str, Any]] = []
    missing_chair = 0
    for sid, feat in feature_map.items():
        image_id = safe_id(feat.get("image_id"))
        if not image_id:
            continue
        base_score = chair_base.get(image_id, {}).get("score")
        int_score = chair_int.get(image_id, {}).get("score")
        if base_score is None or int_score is None:
            missing_chair += 1
            continue

        base_score = float(base_score)
        int_score = float(int_score)
        help_ = int(int_score < (base_score - float(args.epsilon)))
        harm = int(int_score > (base_score + float(args.epsilon)))
        neutral = int((help_ == 0) and (harm == 0))
        utility = float(base_score - int_score)

        label = "help" if help_ else "harm" if harm else "neutral"
        row: Dict[str, Any] = {
            "id": sid,
            "method": args.method_name,
            "benchmark": args.benchmark_name,
            "split": args.split_name,
            "image": safe_id(feat.get("image")),
            "image_id": image_id,
            "question": safe_id(feat.get("question")),
            "baseline_text": str(baseline_map.get(sid, "")).strip(),
            "intervention_text": str(intervention_map.get(sid, "")).strip(),
            "chair_metric": args.chair_metric,
            "baseline_chair": base_score,
            "intervention_chair": int_score,
            "utility": utility,
            "help": help_,
            "harm": harm,
            "neutral": neutral,
            "label_delta": label,
            "oracle_route": "baseline" if harm else "method",
        }
        for key, value in feat.items():
            if key in row:
                continue
            row[key] = value
        rows.append(row)

    rows.sort(key=lambda r: int(str(r["id"])))
    write_csv(args.out_csv, rows)

    n = len(rows)
    summary = {
        "inputs": {
            "baseline_features_csv": os.path.abspath(args.baseline_features_csv),
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
            "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
            "chair_metric": args.chair_metric,
            "method_name": args.method_name,
        },
        "counts": {
            "n_rows": int(n),
            "missing_chair": int(missing_chair),
            "baseline_mean_chair": float(sum(float(r["baseline_chair"]) for r in rows) / float(max(1, n))),
            "intervention_mean_chair": float(sum(float(r["intervention_chair"]) for r in rows) / float(max(1, n))),
            "harm_rate": float(sum(int(r["harm"]) for r in rows) / float(max(1, n))),
            "help_rate": float(sum(int(r["help"]) for r in rows) / float(max(1, n))),
            "neutral_rate": float(sum(int(r["neutral"]) for r in rows) / float(max(1, n))),
            "mean_utility": float(sum(float(r["utility"]) for r in rows) / float(max(1, n))),
        },
        "outputs": {
            "table_csv": os.path.abspath(args.out_csv),
        },
    }
    write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
