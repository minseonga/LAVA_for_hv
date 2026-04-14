#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("id") or row.get("image_id") or row.get("question_id") or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def flag(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def auc_high(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    good = 0.0
    total = 0
    for a in pos:
        for b in neg:
            total += 1
            if a > b:
                good += 1.0
            elif a == b:
                good += 0.5
    return good / float(total) if total else None


def average_precision(items: Sequence[Tuple[int, float]]) -> Optional[float]:
    ranked = sorted(items, key=lambda item: item[1], reverse=True)
    n_pos = sum(label for label, _ in ranked)
    if n_pos <= 0:
        return None
    hits = 0
    total = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label:
            hits += 1
            total += hits / float(rank)
    return total / float(n_pos)


def precision_at(items: Sequence[Tuple[int, float]], k: int) -> Optional[float]:
    top = sorted(items, key=lambda item: item[1], reverse=True)[: int(k)]
    if not top:
        return None
    return sum(label for label, _ in top) / float(len(top))


def numeric_feature_metrics(rows: Sequence[Dict[str, Any]], target_col: str) -> List[Dict[str, Any]]:
    if not rows:
        return []
    excluded = {"id", "image", target_col}
    metrics: List[Dict[str, Any]] = []
    for feature in rows[0]:
        if feature in excluded or feature.endswith("_names") or feature.endswith("_error"):
            continue
        pairs: List[Tuple[int, float]] = []
        for row in rows:
            value = safe_float(row.get(feature))
            if value is None:
                continue
            pairs.append((flag(row.get(target_col)), value))
        if len(pairs) < max(10, int(0.8 * len(rows))):
            continue
        if len({round(score, 12) for _, score in pairs}) < 3:
            continue
        pos = [score for label, score in pairs if label]
        neg = [score for label, score in pairs if not label]
        auc = auc_high(pos, neg)
        if auc is None:
            continue
        direction = "high" if auc >= 0.5 else "low"
        oriented = [(label, score if direction == "high" else -score) for label, score in pairs]
        metrics.append(
            {
                "feature": feature,
                "direction": direction,
                "auroc": max(float(auc), float(1.0 - auc)),
                "auroc_high": float(auc),
                "ap": average_precision(oriented),
                "p_at_10": precision_at(oriented, 10),
                "p_at_25": precision_at(oriented, 25),
                "p_at_50": precision_at(oriented, 50),
                "n": len(pairs),
                "n_pos": sum(label for label, _ in pairs),
                "pos_mean": sum(pos) / float(len(pos)) if pos else "",
                "neg_mean": sum(neg) / float(len(neg)) if neg else "",
            }
        )
    metrics.sort(key=lambda row: (float(row.get("auroc") or 0.0), float(row.get("ap") or 0.0)), reverse=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate COCO-aware object yes/no support features against oracle target.")
    ap.add_argument("--yesno_features_csv", required=True)
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_feature_metrics_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    args = ap.parse_args()

    yesno = {safe_id(row): row for row in read_csv_rows(args.yesno_features_csv)}
    oracle = {safe_id(row): row for row in read_csv_rows(args.oracle_rows_csv)}
    ids = sorted(
        set(yesno) & set(oracle),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        row: Dict[str, Any] = {"id": sid, args.target_col: flag(oracle[sid].get(args.target_col))}
        row.update(yesno[sid])
        row["id"] = sid
        row[args.target_col] = flag(oracle[sid].get(args.target_col))
        rows.append(row)

    metrics = numeric_feature_metrics(rows, args.target_col)
    n_target = sum(flag(row.get(args.target_col)) for row in rows)
    write_csv(args.out_csv, rows)
    write_csv(args.out_feature_metrics_csv, metrics)
    write_json(
        args.out_summary_json,
        {
            "inputs": {
                "yesno_features_csv": os.path.abspath(args.yesno_features_csv),
                "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
                "target_col": args.target_col,
            },
            "counts": {
                "n_rows": len(rows),
                "n_target": int(n_target),
                "target_rate": float(n_target / float(max(1, len(rows)))),
                "n_feature_metrics": len(metrics),
            },
            "top_feature_metrics": metrics[:30],
            "outputs": {
                "joined_csv": os.path.abspath(args.out_csv),
                "feature_metrics_csv": os.path.abspath(args.out_feature_metrics_csv),
                "summary_json": os.path.abspath(args.out_summary_json),
            },
        },
    )
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_feature_metrics_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))
    for metric in metrics[:15]:
        print(
            "[metric]",
            metric["feature"],
            "dir=",
            metric["direction"],
            "auc=",
            metric["auroc"],
            "ap=",
            metric["ap"],
        )


if __name__ == "__main__":
    main()
