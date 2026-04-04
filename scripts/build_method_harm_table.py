#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import (
    load_prediction_text_map,
    parse_yes_no,
    read_csv_rows,
    safe_id,
    write_json,
)


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


def load_gt_map_from_csv(path: str, id_col: str = "id", label_col: str = "answer") -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get(id_col))
        label = safe_id(row.get(label_col)).lower()
        if not sid or label not in {"yes", "no"}:
            continue
        out[sid] = {
            "gt_label": label,
            "question": safe_id(row.get("question")),
            "image_id": safe_id(row.get("image_id")),
            "category": safe_id(row.get("category")),
        }
    return out


def load_feature_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        if sid:
            out[sid] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build method-specific 4-way intervention tables using shared baseline-side semantic features.")
    ap.add_argument("--baseline_features_csv", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--method_name", type=str, required=True)
    ap.add_argument("--benchmark_name", type=str, default="pope")
    ap.add_argument("--split_name", type=str, default="full")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, required=True)
    ap.add_argument("--baseline_pred_text_key", type=str, default="auto")
    ap.add_argument("--intervention_pred_text_key", type=str, default="auto")
    args = ap.parse_args()

    gt_map = load_gt_map_from_csv(os.path.abspath(args.gt_csv))
    feature_map = load_feature_map(os.path.abspath(args.baseline_features_csv))
    baseline_map = load_prediction_text_map(os.path.abspath(args.baseline_pred_jsonl), args.baseline_pred_text_key)
    intervention_map = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), args.intervention_pred_text_key)

    rows: List[Dict[str, Any]] = []
    missing_features = 0
    missing_predictions = 0

    for sid, gt in tqdm(gt_map.items(), total=len(gt_map), desc=f"table:{args.method_name}", unit="sample"):
        feat = feature_map.get(sid)
        baseline_text = str(baseline_map.get(sid, "")).strip()
        intervention_text = str(intervention_map.get(sid, "")).strip()
        if feat is None:
            missing_features += 1
            continue
        if not baseline_text or not intervention_text:
            missing_predictions += 1
            continue

        gt_label = safe_id(gt.get("gt_label")).lower()
        baseline_label = parse_yes_no(baseline_text)
        intervention_label = parse_yes_no(intervention_text)
        baseline_correct = int(baseline_label == gt_label)
        intervention_correct = int(intervention_label == gt_label)
        harm = int((baseline_correct == 1) and (intervention_correct == 0))
        help_ = int((baseline_correct == 0) and (intervention_correct == 1))
        both_correct = int((baseline_correct == 1) and (intervention_correct == 1))
        both_wrong = int((baseline_correct == 0) and (intervention_correct == 0))
        neutral = int(both_correct or both_wrong)
        utility = int(intervention_correct - baseline_correct)
        oracle_route = "baseline" if harm else "method"
        oracle_correct = max(baseline_correct, intervention_correct)

        if help_:
            fourway = "help"
        elif harm:
            fourway = "harm"
        elif both_correct:
            fourway = "both_correct"
        else:
            fourway = "both_wrong"

        row: Dict[str, Any] = {
            "id": sid,
            "method": args.method_name,
            "benchmark": args.benchmark_name,
            "split": args.split_name,
            "question": safe_id(feat.get("question")) or safe_id(gt.get("question")),
            "image": safe_id(feat.get("image")),
            "image_id": safe_id(feat.get("image_id")) or safe_id(gt.get("image_id")),
            "category": safe_id(feat.get("category")) or safe_id(gt.get("category")),
            "gt_label": gt_label,
            "baseline_text": baseline_text,
            "intervention_text": intervention_text,
            "baseline_label": baseline_label,
            "intervention_label": intervention_label,
            "baseline_correct": baseline_correct,
            "intervention_correct": intervention_correct,
            "harm": harm,
            "help": help_,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "neutral": neutral,
            "non_harm": int(1 - harm),
            "non_help": int(1 - help_),
            "utility": utility,
            "oracle_route": oracle_route,
            "oracle_correct": oracle_correct,
            "label_4way": fourway,
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
            "gt_csv": os.path.abspath(args.gt_csv),
            "method_name": args.method_name,
            "benchmark_name": args.benchmark_name,
            "split_name": args.split_name,
        },
        "counts": {
            "n_rows": int(n),
            "missing_features": int(missing_features),
            "missing_predictions": int(missing_predictions),
            "baseline_acc": float(sum(int(r["baseline_correct"]) for r in rows) / float(max(1, n))),
            "intervention_acc": float(sum(int(r["intervention_correct"]) for r in rows) / float(max(1, n))),
            "harm_rate": float(sum(int(r["harm"]) for r in rows) / float(max(1, n))),
            "help_rate": float(sum(int(r["help"]) for r in rows) / float(max(1, n))),
            "both_correct_rate": float(sum(int(r["both_correct"]) for r in rows) / float(max(1, n))),
            "both_wrong_rate": float(sum(int(r["both_wrong"]) for r in rows) / float(max(1, n))),
            "oracle_posthoc_acc": float(sum(int(r["oracle_correct"]) for r in rows) / float(max(1, n))),
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
