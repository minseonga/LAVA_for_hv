#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable


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


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def maybe_int(value: object) -> Optional[int]:
    v = maybe_float(value)
    if v is None:
        return None
    return int(round(v))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(int(y) for y in labels)
    n_neg = len(labels) - n_pos
    if len(scores) != len(labels) or n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = sum(rank for rank, y in zip(ranks, labels) if int(y) == 1)
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def binary_average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    if not pairs:
        return None
    n_pos = sum(y for _, y in pairs)
    if n_pos == 0:
        return None
    pairs.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    ap = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            ap += float(tp) / float(rank)
    return float(ap / float(n_pos))


def feature_cols(rows: Sequence[Dict[str, str]], allowlist: Sequence[str]) -> List[str]:
    allow = set(str(x) for x in allowlist if str(x).strip())
    cols: List[str] = []
    if not rows:
        return cols
    for key in rows[0].keys():
        if allow and key not in allow:
            continue
        if maybe_float(rows[0].get(key)) is not None:
            cols.append(key)
    return cols


def evaluate_feature(rows: Sequence[Dict[str, str]], feature: str, target: str) -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = maybe_float(row.get(feature))
        y = maybe_int(row.get(target))
        if x is None or y not in {0, 1}:
            continue
        xs.append(float(x))
        ys.append(int(y))
    if len(xs) < 2:
        return None
    auc_high = binary_auroc(xs, ys)
    auc_low = binary_auroc([-x for x in xs], ys)
    if auc_high is None or auc_low is None:
        return None
    direction = "high" if auc_high >= auc_low else "low"
    oriented = xs if direction == "high" else [-x for x in xs]
    ap = binary_average_precision(oriented, ys)
    return {
        "feature": feature,
        "target": target,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "average_precision": None if ap is None else float(ap),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
        "positive_rate": float(sum(ys) / float(len(xs))),
    }


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def method_core_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    n = len(rows)
    harm = sum(int(maybe_int(r.get("harm")) or 0) for r in rows)
    help_ = sum(int(maybe_int(r.get("help")) or 0) for r in rows)
    both_correct = sum(int(maybe_int(r.get("both_correct")) or 0) for r in rows)
    both_wrong = sum(int(maybe_int(r.get("both_wrong")) or 0) for r in rows)
    baseline_correct = sum(int(maybe_int(r.get("baseline_correct")) or 0) for r in rows)
    intervention_correct = sum(int(maybe_int(r.get("intervention_correct")) or 0) for r in rows)
    return {
        "n_rows": int(n),
        "baseline_acc": safe_div(float(baseline_correct), float(max(1, n))),
        "intervention_acc": safe_div(float(intervention_correct), float(max(1, n))),
        "harm_rate": safe_div(float(harm), float(max(1, n))),
        "help_rate": safe_div(float(help_), float(max(1, n))),
        "both_correct_rate": safe_div(float(both_correct), float(max(1, n))),
        "both_wrong_rate": safe_div(float(both_wrong), float(max(1, n))),
        "n_harm": int(harm),
        "n_help": int(help_),
        "n_both_correct": int(both_correct),
        "n_both_wrong": int(both_wrong),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze shared baseline-side harm features across VGA/VISTA/EAZY tables.")
    ap.add_argument("--table_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--feature_cols",
        type=str,
        default="base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy",
    )
    args = ap.parse_args()

    allowlist = [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]
    source_rows: Dict[str, List[Dict[str, str]]] = {}
    method_summaries: Dict[str, Dict[str, Any]] = {}
    harm_metric_rows: List[Dict[str, Any]] = []
    consistency_map: Dict[str, List[Dict[str, Any]]] = {}

    for path in tqdm(args.table_csvs, desc="harm-tables", unit="table"):
        rows = read_csv_rows(path)
        if not rows:
            continue
        method = str(rows[0].get("method", os.path.basename(path)))
        source_rows[method] = rows
        method_summaries[method] = method_core_summary(rows)
        feats = feature_cols(rows, allowlist)
        for feat in tqdm(feats, desc=f"harm:{method}", unit="feature", leave=False):
            result = evaluate_feature(rows, feat, target="harm")
            if result is None:
                continue
            result["method"] = method
            harm_metric_rows.append(result)
            consistency_map.setdefault(feat, []).append(result)

    harm_metric_rows.sort(key=lambda r: (str(r["method"]), -float(r["auroc"]), str(r["feature"])))
    harm_metrics_csv = os.path.join(args.out_dir, "harm_feature_metrics.csv")
    write_csv(harm_metrics_csv, harm_metric_rows)

    consistency_rows: List[Dict[str, Any]] = []
    for feat, rows in consistency_map.items():
        if len(rows) != len(source_rows):
            continue
        directions = sorted(set(str(r["direction"]) for r in rows))
        aucs = [float(r["auroc"]) for r in rows]
        aps = [float(r["average_precision"]) for r in rows if r.get("average_precision") is not None]
        consistency_rows.append(
            {
                "feature": feat,
                "methods": ",".join(sorted(str(r["method"]) for r in rows)),
                "direction_set": ",".join(directions),
                "consistent_direction": int(len(directions) == 1),
                "mean_auroc": mean(aucs),
                "min_auroc": float(min(aucs)),
                "max_auroc": float(max(aucs)),
                "mean_average_precision": mean(aps),
                "min_average_precision": float(min(aps)) if aps else None,
            }
        )
    consistency_rows.sort(key=lambda r: (-int(r["consistent_direction"]), -float(r["mean_auroc"]), str(r["feature"])))
    consistency_csv = os.path.join(args.out_dir, "shared_harm_consistency.csv")
    write_csv(consistency_csv, consistency_rows)

    best_by_method: Dict[str, Dict[str, Any]] = {}
    for method, rows in source_rows.items():
        sub = [r for r in harm_metric_rows if str(r["method"]) == method]
        if sub:
            sub.sort(key=lambda r: (-float(r["auroc"]), -float(r.get("average_precision") or 0.0), str(r["feature"])))
            best_by_method[method] = {
                "feature": str(sub[0]["feature"]),
                "direction": str(sub[0]["direction"]),
                "auroc": float(sub[0]["auroc"]),
                "average_precision": None if sub[0].get("average_precision") is None else float(sub[0]["average_precision"]),
            }

    summary = {
        "inputs": {
            "table_csvs": [os.path.abspath(x) for x in args.table_csvs],
            "feature_cols": allowlist,
        },
        "methods": method_summaries,
        "best_harm_feature_by_method": best_by_method,
        "top_consistent_harm_features": consistency_rows[:10],
        "outputs": {
            "harm_feature_metrics_csv": os.path.abspath(harm_metrics_csv),
            "shared_harm_consistency_csv": os.path.abspath(consistency_csv),
        },
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[saved]", harm_metrics_csv)
    print("[saved]", consistency_csv)
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
