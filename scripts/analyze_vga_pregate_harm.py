#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


def feature_cols(rows: Sequence[Dict[str, str]], allowlist: Optional[Sequence[str]] = None) -> List[str]:
    reserved = {
        "id",
        "benchmark",
        "split",
        "question",
        "image",
        "image_id",
        "category",
        "gt_label",
        "baseline_text",
        "intervention_text",
        "baseline_label",
        "intervention_label",
        "baseline_correct",
        "intervention_correct",
        "harm",
        "help",
        "utility",
        "oracle_route",
        "oracle_correct",
    }
    allow = set(str(x) for x in allowlist) if allowlist else None
    cols: List[str] = []
    if not rows:
        return cols
    for key in rows[0].keys():
        if key in reserved:
            continue
        if allow is not None and key not in allow:
            continue
        if maybe_float(rows[0].get(key)) is not None:
            cols.append(key)
    return cols


def analyze_feature(rows: Sequence[Dict[str, str]], feature: str, target: str = "harm") -> Optional[Dict[str, Any]]:
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
    auc = max(float(auc_high), float(auc_low))
    pos = [x for x, y in zip(xs, ys) if y == 1]
    neg = [x for x, y in zip(xs, ys) if y == 0]
    return {
        "feature": feature,
        "target": target,
        "direction": direction,
        "auroc": auc,
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
        "positive_rate": float(sum(ys) / float(len(xs))),
        "mean_pos": float(sum(pos) / float(max(1, len(pos)))),
        "mean_neg": float(sum(neg) / float(max(1, len(neg)))),
    }


def plot_top(rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benches = sorted({str(r["benchmark"]) for r in rows})
    fig, axes = plt.subplots(1, len(benches), figsize=(6 * len(benches), 4), squeeze=False)
    for ax, bench in zip(axes[0], benches):
        sub = [r for r in rows if str(r["benchmark"]) == bench]
        sub.sort(key=lambda r: float(r["auroc"]), reverse=True)
        top = sub[:10]
        vals = [float(r["auroc"]) for r in top]
        names = [str(r["feature"]) for r in top]
        ax.barh(range(len(top)), vals, color="#2563eb")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 1.0)
        ax.set_title(bench)
        ax.set_xlabel("harm AUROC")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze pre-intervention harm separability from probe-side features.")
    ap.add_argument("--table_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--feature_cols", type=str, default="")
    args = ap.parse_args()

    allowlist = [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]

    metric_rows: List[Dict[str, Any]] = []
    by_benchmark_best: Dict[str, Dict[str, Any]] = {}
    consistency_map: Dict[str, List[Dict[str, Any]]] = {}
    for path in tqdm(args.table_csvs, desc="analysis-benchmark", unit="table"):
        rows = read_csv_rows(path)
        if not rows:
            continue
        benchmark = str(rows[0].get("benchmark", os.path.basename(path)))
        feats = feature_cols(rows, allowlist=allowlist if allowlist else None)
        bench_rows: List[Dict[str, Any]] = []
        for feat in tqdm(feats, desc=f"harm-features:{benchmark}", unit="feature", leave=False):
            result = analyze_feature(rows, feat, target="harm")
            if result is None:
                continue
            result["benchmark"] = benchmark
            metric_rows.append(result)
            bench_rows.append(result)
            consistency_map.setdefault(feat, []).append(result)
        if bench_rows:
            bench_rows.sort(key=lambda r: float(r["auroc"]), reverse=True)
            by_benchmark_best[benchmark] = bench_rows[0]

    metric_rows.sort(key=lambda r: (str(r["benchmark"]), -float(r["auroc"]), str(r["feature"])))
    metrics_csv = os.path.join(args.out_dir, "feature_metrics.csv")
    write_csv(metrics_csv, metric_rows)

    consistency_rows: List[Dict[str, Any]] = []
    for feat, rows in consistency_map.items():
        dirs = [str(r["direction"]) for r in rows]
        aucs = [float(r["auroc"]) for r in rows]
        consistency_rows.append(
            {
                "feature": feat,
                "benchmarks": ",".join(sorted(str(r["benchmark"]) for r in rows)),
                "direction_set": ",".join(sorted(set(dirs))),
                "consistent_direction": int(len(set(dirs)) == 1),
                "mean_auroc": float(sum(aucs) / float(len(aucs))),
                "min_auroc": float(min(aucs)),
                "max_auroc": float(max(aucs)),
            }
        )
    consistency_rows.sort(key=lambda r: (-int(r["consistent_direction"]), -float(r["mean_auroc"]), str(r["feature"])))
    consistency_csv = os.path.join(args.out_dir, "harm_consistency.csv")
    write_csv(consistency_csv, consistency_rows)

    if metric_rows:
        plot_top(metric_rows, os.path.join(args.out_dir, "harm_top_features.png"))

    summary = {
        "inputs": {
            "table_csvs": [os.path.abspath(x) for x in args.table_csvs],
            "feature_cols": allowlist,
        },
        "best_by_benchmark": by_benchmark_best,
        "top_consistent_features": consistency_rows[:15],
        "outputs": {
            "feature_metrics_csv": os.path.abspath(metrics_csv),
            "harm_consistency_csv": os.path.abspath(consistency_csv),
        },
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[saved]", os.path.abspath(metrics_csv))
    print("[saved]", os.path.abspath(consistency_csv))
    print("[saved]", os.path.abspath(os.path.join(args.out_dir, "summary.json")))


if __name__ == "__main__":
    main()
