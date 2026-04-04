#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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


def maybe_int(value: object) -> Optional[int]:
    s = str(value if value is not None else "").strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def load_feature_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        if sid:
            out[sid] = row
    return out


def load_gt_label_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        label = safe_id(row.get("answer")).lower()
        if not sid or label not in {"yes", "no"}:
            continue
        out[sid] = {
            "gt_label": label,
            "question": safe_id(row.get("question")),
            "category": safe_id(row.get("category")),
            "image_id": safe_id(row.get("image_id")),
        }
    return out


def load_amber_gt_map_from_root(amber_root: str) -> Dict[str, Dict[str, str]]:
    query_json = os.path.join(amber_root, "data", "query", "query_discriminative.json")
    annotations_json = os.path.join(amber_root, "data", "annotations.json")
    queries = read_json(query_json)
    annotations = read_json(annotations_json)
    ann_by_id = {
        safe_id(row.get("id")): row
        for row in annotations
        if safe_id(row.get("id"))
    }
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
            "category": safe_id(ann.get("type")),
            "image_id": image_id,
        }
    return out


def feature_columns_from_rows(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key.startswith("cheap_") or key in {"n_cont_tokens", "n_content_tokens", "score_error"}:
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    return out


def add_outcome_fields(row: Dict[str, Any], *, baseline_correct: int, intervention_correct: int) -> None:
    harm = int((baseline_correct == 1) and (intervention_correct == 0))
    help_ = int((baseline_correct == 0) and (intervention_correct == 1))
    utility = int(intervention_correct - baseline_correct)
    row["baseline_correct"] = int(baseline_correct)
    row["intervention_correct"] = int(intervention_correct)
    row["harm"] = harm
    row["help"] = help_
    row["utility"] = utility
    if baseline_correct == intervention_correct:
        row["outcome_case"] = "no_change_correct" if baseline_correct == 1 else "no_change_wrong"
    elif harm:
        row["outcome_case"] = "harm"
    else:
        row["outcome_case"] = "help"


def build_pope_rows(
    *,
    features_csv: str,
    decisions_csv: str,
    gt_csv: str,
    method_name: str,
) -> List[Dict[str, Any]]:
    feature_map = load_feature_map(features_csv)
    gt_map = load_gt_label_map(gt_csv) if gt_csv else {}
    rows: List[Dict[str, Any]] = []
    for dec in read_csv_rows(decisions_csv):
        sid = safe_id(dec.get("id"))
        feat = feature_map.get(sid)
        if not sid or feat is None:
            continue
        baseline_correct = maybe_int(dec.get("baseline_correct"))
        intervention_correct = maybe_int(dec.get("intervention_correct"))
        if baseline_correct is None or intervention_correct is None:
            continue
        gt = gt_map.get(sid, {})
        row: Dict[str, Any] = {
            "id": sid,
            "dataset": "pope",
            "benchmark": "pope",
            "method": method_name,
            "question": safe_id(feat.get("question")) or safe_id(gt.get("question")),
            "image": safe_id(feat.get("image")),
            "image_id": safe_id(gt.get("image_id")),
            "gt_label": safe_id(gt.get("gt_label")),
            "category": safe_id(gt.get("category")),
            "case_type_reference": safe_id(dec.get("case_type")),
        }
        add_outcome_fields(
            row,
            baseline_correct=baseline_correct,
            intervention_correct=intervention_correct,
        )
        for key, value in feat.items():
            if key in {"id", "question", "image"}:
                continue
            row[key] = value
        rows.append(row)
    return rows


def build_amber_discriminative_rows(
    *,
    features_csv: str,
    baseline_pred_jsonl: str,
    intervention_pred_jsonl: str,
    amber_gt_csv: str,
    amber_root: str,
    baseline_pred_text_key: str,
    intervention_pred_text_key: str,
    method_name: str,
) -> List[Dict[str, Any]]:
    feature_map = load_feature_map(features_csv)
    if amber_gt_csv:
        gt_map = load_gt_label_map(amber_gt_csv)
    elif amber_root:
        gt_map = load_amber_gt_map_from_root(amber_root)
    else:
        raise ValueError("AMBER labels require either --amber_gt_csv or --amber_root.")
    baseline_map = load_prediction_text_map(baseline_pred_jsonl, baseline_pred_text_key)
    intervention_map = load_prediction_text_map(intervention_pred_jsonl, intervention_pred_text_key)

    rows: List[Dict[str, Any]] = []
    for sid, gt in gt_map.items():
        feat = feature_map.get(sid)
        base_text = baseline_map.get(sid, "")
        inter_text = intervention_map.get(sid, "")
        if feat is None or not base_text or not inter_text:
            continue
        gt_label = safe_id(gt.get("gt_label")).lower()
        base_label = parse_yes_no(base_text)
        inter_label = parse_yes_no(inter_text)
        row: Dict[str, Any] = {
            "id": sid,
            "dataset": "amber",
            "benchmark": "amber_discriminative",
            "method": method_name,
            "question": safe_id(feat.get("question")) or safe_id(gt.get("question")),
            "image": safe_id(feat.get("image")),
            "image_id": safe_id(gt.get("image_id")),
            "gt_label": gt_label,
            "category": safe_id(gt.get("category")),
            "baseline_label": base_label,
            "intervention_label": inter_label,
        }
        add_outcome_fields(
            row,
            baseline_correct=int(base_label == gt_label),
            intervention_correct=int(inter_label == gt_label),
        )
        for key, value in feat.items():
            if key in {"id", "question", "image"}:
                continue
            row[key] = value
        rows.append(row)
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_benchmark: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_benchmark.setdefault(str(row.get("benchmark")), []).append(row)
    out: Dict[str, Any] = {}
    for benchmark, bench_rows in by_benchmark.items():
        n = len(bench_rows)
        harm = sum(int(row.get("harm", 0)) for row in bench_rows)
        help_ = sum(int(row.get("help", 0)) for row in bench_rows)
        base_acc = sum(int(row.get("baseline_correct", 0)) for row in bench_rows) / float(max(1, n))
        inter_acc = sum(int(row.get("intervention_correct", 0)) for row in bench_rows) / float(max(1, n))
        cats = Counter(str(row.get("category", "")) for row in bench_rows if str(row.get("category", "")))
        out[benchmark] = {
            "n": int(n),
            "baseline_acc": float(base_acc),
            "intervention_acc": float(inter_acc),
            "harm_count": int(harm),
            "harm_rate": float(harm / float(max(1, n))),
            "help_count": int(help_),
            "help_rate": float(help_ / float(max(1, n))),
            "categories_top10": dict(cats.most_common(10)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a VGA susceptibility panel over POPE and AMBER discriminative.")
    ap.add_argument("--pope_features_csv", type=str, required=True)
    ap.add_argument("--pope_decisions_csv", type=str, required=True)
    ap.add_argument("--pope_gt_csv", type=str, default="")
    ap.add_argument("--amber_features_csv", type=str, required=True)
    ap.add_argument("--amber_baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--amber_intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--amber_gt_csv", type=str, default="")
    ap.add_argument("--amber_root", type=str, default="")
    ap.add_argument("--baseline_pred_text_key", type=str, default="text")
    ap.add_argument("--intervention_pred_text_key", type=str, default="output")
    ap.add_argument("--method_name", type=str, default="vga")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, required=True)
    args = ap.parse_args()

    pope_rows = build_pope_rows(
        features_csv=os.path.abspath(args.pope_features_csv),
        decisions_csv=os.path.abspath(args.pope_decisions_csv),
        gt_csv=os.path.abspath(args.pope_gt_csv) if args.pope_gt_csv else "",
        method_name=args.method_name,
    )
    amber_rows = build_amber_discriminative_rows(
        features_csv=os.path.abspath(args.amber_features_csv),
        baseline_pred_jsonl=os.path.abspath(args.amber_baseline_pred_jsonl),
        intervention_pred_jsonl=os.path.abspath(args.amber_intervention_pred_jsonl),
        amber_gt_csv=os.path.abspath(args.amber_gt_csv) if args.amber_gt_csv else "",
        amber_root=os.path.abspath(args.amber_root) if args.amber_root else "",
        baseline_pred_text_key=args.baseline_pred_text_key,
        intervention_pred_text_key=args.intervention_pred_text_key,
        method_name=args.method_name,
    )

    rows = pope_rows + amber_rows
    rows.sort(key=lambda r: (str(r.get("benchmark")), int(str(r.get("id", "0")))))
    write_csv(args.out_csv, rows)

    summary = {
        "inputs": {
            "pope_features_csv": os.path.abspath(args.pope_features_csv),
            "pope_decisions_csv": os.path.abspath(args.pope_decisions_csv),
            "pope_gt_csv": os.path.abspath(args.pope_gt_csv) if args.pope_gt_csv else "",
            "amber_features_csv": os.path.abspath(args.amber_features_csv),
            "amber_baseline_pred_jsonl": os.path.abspath(args.amber_baseline_pred_jsonl),
            "amber_intervention_pred_jsonl": os.path.abspath(args.amber_intervention_pred_jsonl),
            "amber_gt_csv": os.path.abspath(args.amber_gt_csv) if args.amber_gt_csv else "",
            "amber_root": os.path.abspath(args.amber_root) if args.amber_root else "",
            "method_name": args.method_name,
        },
        "counts": summarize(rows),
        "n_rows_total": int(len(rows)),
        "feature_columns": feature_columns_from_rows(rows),
        "outputs": {
            "panel_csv": os.path.abspath(args.out_csv),
        },
    }
    write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
