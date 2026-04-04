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

from frgavr_cleanroom.runtime import load_prediction_text_map, parse_yes_no, safe_id, write_json


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def load_amber_gt_map(amber_root: str) -> Dict[str, Dict[str, str]]:
    query_json = os.path.join(amber_root, "data", "query", "query_discriminative.json")
    annotations_json = os.path.join(amber_root, "data", "annotations.json")
    queries = read_json(query_json)
    annotations = read_json(annotations_json)
    ann_by_id = {safe_id(row.get("id")): row for row in annotations if safe_id(row.get("id"))}
    out: Dict[str, Dict[str, str]] = {}
    for item in queries:
        sid = safe_id(item.get("id"))
        ann = ann_by_id.get(sid)
        if not sid or ann is None:
            continue
        label = safe_id(ann.get("truth")).lower()
        if label not in {"yes", "no"}:
            continue
        image = safe_id(item.get("image"))
        image_id = os.path.splitext(os.path.basename(image))[0]
        out[sid] = {
            "gt_label": label,
            "question": safe_id(item.get("query")),
            "image_id": image_id,
            "category": safe_id(ann.get("type")),
        }
    return out


def load_question_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_jsonl(path):
        sid = safe_id(row.get("question_id", row.get("id")))
        if not sid:
            continue
        out[sid] = {
            "question": safe_id(row.get("question", row.get("text"))),
            "image": safe_id(row.get("image")),
            "image_id": safe_id(row.get("image_id")),
            "category": safe_id(row.get("amber_type", row.get("category"))),
        }
    return out


def load_probe_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        if sid:
            out[sid] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pre-intervention harm-gating tables from probe features and branch outputs.")
    ap.add_argument("--probe_features_csv", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, required=True)
    ap.add_argument("--benchmark_name", type=str, required=True)
    ap.add_argument("--split_name", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, default="")
    ap.add_argument("--amber_root", type=str, default="")
    ap.add_argument("--baseline_pred_text_key", type=str, default="text")
    ap.add_argument("--intervention_pred_text_key", type=str, default="output")
    args = ap.parse_args()

    if args.gt_csv:
        gt_map = load_gt_map_from_csv(os.path.abspath(args.gt_csv))
    elif args.amber_root:
        gt_map = load_amber_gt_map(os.path.abspath(args.amber_root))
    else:
        raise ValueError("Need either --gt_csv or --amber_root.")

    question_map = load_question_map(os.path.abspath(args.question_file))
    probe_map = load_probe_map(os.path.abspath(args.probe_features_csv))
    baseline_map = load_prediction_text_map(os.path.abspath(args.baseline_pred_jsonl), args.baseline_pred_text_key)
    intervention_map = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), args.intervention_pred_text_key)

    rows: List[Dict[str, Any]] = []
    missing_probe = 0
    missing_pred = 0
    for sid, gt in gt_map.items():
        q = question_map.get(sid, {})
        probe = probe_map.get(sid)
        baseline_text = str(baseline_map.get(sid, "")).strip()
        intervention_text = str(intervention_map.get(sid, "")).strip()
        if probe is None:
            missing_probe += 1
            continue
        if not baseline_text or not intervention_text:
            missing_pred += 1
            continue

        gt_label = safe_id(gt.get("gt_label")).lower()
        baseline_label = parse_yes_no(baseline_text)
        intervention_label = parse_yes_no(intervention_text)
        baseline_correct = int(baseline_label == gt_label)
        intervention_correct = int(intervention_label == gt_label)
        harm = int((baseline_correct == 1) and (intervention_correct == 0))
        help_ = int((baseline_correct == 0) and (intervention_correct == 1))
        utility = int(intervention_correct - baseline_correct)
        oracle_route = "baseline" if harm else "method"
        oracle_correct = max(baseline_correct, intervention_correct)

        row: Dict[str, Any] = {
            "id": sid,
            "benchmark": args.benchmark_name,
            "split": args.split_name,
            "question": safe_id(q.get("question")) or safe_id(gt.get("question")),
            "image": safe_id(q.get("image")),
            "image_id": safe_id(q.get("image_id")) or safe_id(gt.get("image_id")),
            "category": safe_id(q.get("category")) or safe_id(gt.get("category")),
            "gt_label": gt_label,
            "baseline_text": baseline_text,
            "intervention_text": intervention_text,
            "baseline_label": baseline_label,
            "intervention_label": intervention_label,
            "baseline_correct": baseline_correct,
            "intervention_correct": intervention_correct,
            "harm": harm,
            "help": help_,
            "utility": utility,
            "oracle_route": oracle_route,
            "oracle_correct": oracle_correct,
        }
        for key, value in probe.items():
            if key == "id":
                continue
            row[key] = value
        rows.append(row)

    rows.sort(key=lambda r: int(str(r["id"])))
    write_csv(args.out_csv, rows)

    n = len(rows)
    summary = {
        "inputs": {
            "probe_features_csv": os.path.abspath(args.probe_features_csv),
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "question_file": os.path.abspath(args.question_file),
            "gt_csv": os.path.abspath(args.gt_csv) if args.gt_csv else "",
            "amber_root": os.path.abspath(args.amber_root) if args.amber_root else "",
            "benchmark_name": args.benchmark_name,
            "split_name": args.split_name,
        },
        "counts": {
            "n_rows": int(n),
            "missing_probe": int(missing_probe),
            "missing_pred": int(missing_pred),
            "baseline_acc": float(sum(int(r["baseline_correct"]) for r in rows) / float(max(1, n))),
            "intervention_acc": float(sum(int(r["intervention_correct"]) for r in rows) / float(max(1, n))),
            "harm_rate": float(sum(int(r["harm"]) for r in rows) / float(max(1, n))),
            "help_rate": float(sum(int(r["help"]) for r in rows) / float(max(1, n))),
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
