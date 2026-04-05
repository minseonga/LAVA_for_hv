#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 1.0
    mu = mean(seq)
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(max(math.sqrt(max(var, 0.0)), 1e-6))


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


def parse_target_spec(spec: str) -> Tuple[str, str, float]:
    parts = [x.strip() for x in str(spec).split(":")]
    if len(parts) == 1:
        return parts[0], "binary", 0.0
    if len(parts) != 3:
        raise ValueError(f"Invalid target spec: {spec}")
    return parts[0], parts[1], float(parts[2])


def target_value(row: Dict[str, str], spec: str) -> Optional[int]:
    col, op, threshold = parse_target_spec(spec)
    raw = maybe_float(row.get(col))
    if raw is None:
        return None
    if op == "binary":
        if int(round(raw)) in {0, 1}:
            return int(round(raw))
        return None
    if op == "lt":
        return int(float(raw) < float(threshold))
    if op == "le":
        return int(float(raw) <= float(threshold))
    if op == "gt":
        return int(float(raw) > float(threshold))
    if op == "ge":
        return int(float(raw) >= float(threshold))
    if op == "eq":
        return int(float(raw) == float(threshold))
    raise ValueError(f"Unsupported target op: {op}")


def zscore_oriented(values: Sequence[float], direction: str) -> List[float]:
    seq = [float(v) for v in values]
    oriented = seq if str(direction) == "high" else [-float(v) for v in seq]
    mu = mean(oriented)
    sd = std(oriented)
    return [(float(v) - mu) / sd for v in oriented]


def evaluate_feature(rows: Sequence[Dict[str, str]], feature: str, target_spec: str) -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = maybe_float(row.get(feature))
        y = target_value(row, target_spec)
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
        "target_spec": target_spec,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "average_precision": None if ap is None else float(ap),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
        "positive_rate": float(sum(ys) / float(max(1, len(ys)))),
    }


def build_composite(
    rows: Sequence[Dict[str, str]],
    feature_specs: Sequence[Tuple[str, str]],
    target_spec: str,
) -> Optional[Dict[str, Any]]:
    if not feature_specs:
        return None
    labels: List[int] = []
    matrix: List[List[float]] = []
    for row in rows:
        y = target_value(row, target_spec)
        if y not in {0, 1}:
            continue
        vals: List[float] = []
        ok = True
        for feature, _direction in feature_specs:
            x = maybe_float(row.get(feature))
            if x is None:
                ok = False
                break
            vals.append(float(x))
        if not ok:
            continue
        labels.append(int(y))
        matrix.append(vals)
    if not matrix:
        return None
    cols = list(zip(*matrix))
    zcols: List[List[float]] = []
    for (_feature, direction), col in zip(feature_specs, cols):
        zcols.append(zscore_oriented(col, direction))
    scores = [mean(zvals) for zvals in zip(*zcols)]
    auc = binary_auroc(scores, labels)
    ap = binary_average_precision(scores, labels)
    return {
        "auroc": None if auc is None else float(auc),
        "average_precision": None if ap is None else float(ap),
        "n": int(len(scores)),
        "n_pos": int(sum(labels)),
        "positive_rate": float(sum(labels) / float(max(1, len(labels)))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze composite coverage proxies for generative recall-drop targets.")
    ap.add_argument("--table_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--feature_cols",
        type=str,
        default="probe_n_content_tokens,probe_tail_tokens_after_last_mention,probe_entropy_tail_mean_real,probe_lp_tail_mean_real,probe_last_mention_pos_frac,probe_gap_tail_mean_real,probe_object_diversity,probe_mention_diversity,probe_entropy_tail_minus_head_real,probe_gap_tail_minus_head_real",
    )
    ap.add_argument(
        "--target_specs",
        type=str,
        default="delta_supported_recall:lt:0,claim_supported_dropped",
    )
    ap.add_argument("--top_k_values", type=str, default="1,2,3,4,5")
    args = ap.parse_args()

    rows = read_csv_rows(os.path.abspath(args.table_csv))
    feature_cols = [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]
    target_specs = [x.strip() for x in str(args.target_specs).split(",") if x.strip()]
    top_k_values = [max(1, int(x.strip())) for x in str(args.top_k_values).split(",") if x.strip()]

    feature_metric_rows: List[Dict[str, Any]] = []
    composite_rows: List[Dict[str, Any]] = []
    best_feature_by_target: Dict[str, Dict[str, Any]] = {}
    best_composite_by_target: Dict[str, Dict[str, Any]] = {}

    for target_spec in target_specs:
        per_feature: List[Dict[str, Any]] = []
        for feature in feature_cols:
            result = evaluate_feature(rows, feature, target_spec)
            if result is None:
                continue
            per_feature.append(result)
        per_feature.sort(key=lambda r: (-float(r["auroc"]), -float(r.get("average_precision") or 0.0), str(r["feature"])))
        feature_metric_rows.extend(per_feature)
        if per_feature:
            best_feature_by_target[target_spec] = {
                "feature": str(per_feature[0]["feature"]),
                "direction": str(per_feature[0]["direction"]),
                "auroc": float(per_feature[0]["auroc"]),
                "average_precision": None if per_feature[0].get("average_precision") is None else float(per_feature[0]["average_precision"]),
            }

        for k in top_k_values:
            chosen = per_feature[: max(1, int(k))]
            if not chosen:
                continue
            comp = build_composite(
                rows,
                [(str(r["feature"]), str(r["direction"])) for r in chosen],
                target_spec,
            )
            composite_rows.append(
                {
                    "target_spec": target_spec,
                    "k": int(k),
                    "features": ",".join(str(r["feature"]) for r in chosen),
                    "directions": ",".join(str(r["direction"]) for r in chosen),
                    "auroc": None if comp is None else comp["auroc"],
                    "average_precision": None if comp is None else comp["average_precision"],
                    "n": None if comp is None else comp["n"],
                    "n_pos": None if comp is None else comp["n_pos"],
                    "positive_rate": None if comp is None else comp["positive_rate"],
                }
            )

    for target_spec in target_specs:
        sub = [r for r in composite_rows if str(r["target_spec"]) == target_spec]
        if not sub:
            continue
        sub.sort(key=lambda r: (-float(r.get("auroc") or 0.0), -float(r.get("average_precision") or 0.0), int(r["k"])))
        best = sub[0]
        best_composite_by_target[target_spec] = {
            "k": int(best["k"]),
            "features": str(best["features"]),
            "directions": str(best["directions"]),
            "auroc": None if best.get("auroc") is None else float(best["auroc"]),
            "average_precision": None if best.get("average_precision") is None else float(best["average_precision"]),
        }

    feature_csv = os.path.join(args.out_dir, "feature_metrics.csv")
    composite_csv = os.path.join(args.out_dir, "composite_metrics.csv")
    write_csv(feature_csv, feature_metric_rows)
    write_csv(composite_csv, composite_rows)

    summary = {
        "inputs": {
            "table_csv": os.path.abspath(args.table_csv),
            "feature_cols": feature_cols,
            "target_specs": target_specs,
            "top_k_values": top_k_values,
        },
        "best_feature_by_target": best_feature_by_target,
        "best_composite_by_target": best_composite_by_target,
        "outputs": {
            "feature_metrics_csv": os.path.abspath(feature_csv),
            "composite_metrics_csv": os.path.abspath(composite_csv),
        },
    }
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_json(summary_json, summary)
    print("[saved]", feature_csv)
    print("[saved]", composite_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
