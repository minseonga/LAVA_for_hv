#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
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
    if not scores or len(scores) != len(labels):
        return None
    n_pos = sum(1 for y in labels if int(y) == 1)
    n_neg = sum(1 for y in labels if int(y) == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = sum(rank for rank, y in zip(ranks, labels) if int(y) == 1)
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def precision_at_k(scores: Sequence[float], labels: Sequence[int], k: int) -> Optional[float]:
    if k <= 0 or not scores or len(scores) != len(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top = order[:k]
    if not top:
        return None
    return float(sum(int(labels[i]) for i in top) / float(len(top)))


def quantile_edges(values: Sequence[float], n_bins: int) -> List[float]:
    seq = sorted(float(v) for v in values)
    if not seq:
        return []
    edges: List[float] = []
    last_idx = len(seq) - 1
    for i in range(1, n_bins):
        pos = int(round((i / float(n_bins)) * last_idx))
        pos = max(0, min(last_idx, pos))
        edges.append(seq[pos])
    return edges


def assign_bin(value: float, edges: Sequence[float]) -> int:
    idx = 0
    for edge in edges:
        if value > edge:
            idx += 1
        else:
            break
    return idx


def feature_columns(rows: Sequence[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    cols: List[str] = []
    for key in rows[0].keys():
        if key.startswith("cheap_"):
            cols.append(key)
    return cols


def analyze_feature(rows: Sequence[Dict[str, str]], feature: str, target: str) -> Optional[Dict[str, Any]]:
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
    scores = xs if direction == "high" else [-x for x in xs]
    auc = max(float(auc_high), float(auc_low))
    k = sum(ys)
    p_at_k = precision_at_k(scores, ys, k)
    pos_vals = [x for x, y in zip(xs, ys) if y == 1]
    neg_vals = [x for x, y in zip(xs, ys) if y == 0]
    return {
        "feature": feature,
        "target": target,
        "n": int(len(xs)),
        "n_pos": int(k),
        "positive_rate": float(k / float(len(xs))),
        "direction": direction,
        "auroc": auc,
        "precision_at_pos_k": p_at_k,
        "mean_pos": float(sum(pos_vals) / float(max(1, len(pos_vals)))),
        "mean_neg": float(sum(neg_vals) / float(max(1, len(neg_vals)))),
    }


def summarize_consistency(metric_rows: Sequence[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    for row in metric_rows:
        if row.get("target") != target:
            continue
        by_feature.setdefault(str(row["feature"]), []).append(dict(row))
    out: List[Dict[str, Any]] = []
    for feature, rows in by_feature.items():
        benches = [str(r["benchmark"]) for r in rows if str(r["benchmark"]) != "all"]
        dirs = [str(r["direction"]) for r in rows if str(r["benchmark"]) != "all"]
        aucs = [float(r["auroc"]) for r in rows if str(r["benchmark"]) != "all"]
        if not benches:
            continue
        out.append(
            {
                "feature": feature,
                "target": target,
                "benchmarks": ",".join(sorted(benches)),
                "direction_set": ",".join(sorted(set(dirs))),
                "consistent_direction": int(len(set(dirs)) == 1),
                "mean_auroc": float(sum(aucs) / float(len(aucs))),
                "min_auroc": float(min(aucs)),
                "max_auroc": float(max(aucs)),
            }
        )
    out.sort(key=lambda r: (-int(r["consistent_direction"]), -float(r["mean_auroc"]), str(r["feature"])))
    return out


def plot_top_features(metric_rows: Sequence[Dict[str, Any]], out_path: str, target: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench_rows = [row for row in metric_rows if row.get("target") == target and row.get("benchmark") != "all"]
    benchmarks = sorted({str(row["benchmark"]) for row in bench_rows})
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 5), squeeze=False)
    for ax, bench in zip(axes[0], benchmarks):
        rows = [row for row in bench_rows if str(row["benchmark"]) == bench]
        rows.sort(key=lambda r: float(r["auroc"]), reverse=True)
        top = rows[:10]
        names = [str(row["feature"]).replace("cheap_", "") for row in top]
        aucs = [float(row["auroc"]) for row in top]
        ax.barh(range(len(top)), aucs, color="#3b82f6")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("AUROC")
        ax.set_title(f"{bench}: {target}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_quantile_risk(rows: Sequence[Dict[str, str]], feature: str, out_path: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmarks = sorted({str(row["benchmark"]) for row in rows})
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 4), squeeze=False)
    for ax, bench in zip(axes[0], benchmarks):
        xs: List[float] = []
        harm: List[int] = []
        help_: List[int] = []
        for row in rows:
            if str(row.get("benchmark")) != bench:
                continue
            x = maybe_float(row.get(feature))
            h = maybe_int(row.get("harm"))
            hp = maybe_int(row.get("help"))
            if x is None or h is None or hp is None:
                continue
            xs.append(-float(x))
            harm.append(int(h))
            help_.append(int(hp))
        if not xs:
            ax.set_title(f"{bench}: no data")
            continue
        edges = quantile_edges(xs, 10)
        bucket_harm: List[List[int]] = [[] for _ in range(10)]
        bucket_help: List[List[int]] = [[] for _ in range(10)]
        for score, h, hp in zip(xs, harm, help_):
            idx = assign_bin(score, edges)
            bucket_harm[idx].append(h)
            bucket_help[idx].append(hp)
        harm_rate = [
            (sum(bucket) / float(len(bucket))) if bucket else 0.0
            for bucket in bucket_harm
        ]
        help_rate = [
            (sum(bucket) / float(len(bucket))) if bucket else 0.0
            for bucket in bucket_help
        ]
        xs_plot = list(range(1, 11))
        ax.plot(xs_plot, harm_rate, marker="o", color="#dc2626", label="harm rate")
        ax.plot(xs_plot, help_rate, marker="o", color="#16a34a", label="help rate")
        ax.set_title(bench)
        ax.set_xlabel("Risk decile\n(low lp min -> high risk)")
        ax.set_ylabel("Rate")
        ax.set_ylim(0.0, max(0.05, max(harm_rate + help_rate) * 1.15))
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a VGA susceptibility panel.")
    ap.add_argument("--panel_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--focus_feature", type=str, default="cheap_lp_content_min")
    args = ap.parse_args()

    rows = read_csv_rows(args.panel_csv)
    feats = feature_columns(rows)
    benchmarks = sorted({str(row.get("benchmark")) for row in rows})
    all_benches = benchmarks + ["all"]
    metric_rows: List[Dict[str, Any]] = []
    for benchmark in all_benches:
        bench_rows = rows if benchmark == "all" else [row for row in rows if str(row.get("benchmark")) == benchmark]
        for target in ("harm", "help"):
            for feature in feats:
                result = analyze_feature(bench_rows, feature, target)
                if result is None:
                    continue
                result["benchmark"] = benchmark
                metric_rows.append(result)

    metric_rows.sort(key=lambda r: (str(r["target"]), str(r["benchmark"]), -float(r["auroc"]), str(r["feature"])))
    metrics_csv = os.path.join(args.out_dir, "feature_metrics.csv")
    write_csv(metrics_csv, metric_rows)

    harm_consistency = summarize_consistency(metric_rows, "harm")
    help_consistency = summarize_consistency(metric_rows, "help")
    harm_consistency_csv = os.path.join(args.out_dir, "harm_consistency.csv")
    help_consistency_csv = os.path.join(args.out_dir, "help_consistency.csv")
    write_csv(harm_consistency_csv, harm_consistency)
    write_csv(help_consistency_csv, help_consistency)

    best_by_benchmark: Dict[str, Dict[str, Any]] = {}
    for benchmark in benchmarks:
        best_by_benchmark[benchmark] = {}
        for target in ("harm", "help"):
            sub = [row for row in metric_rows if row["benchmark"] == benchmark and row["target"] == target]
            if not sub:
                continue
            sub.sort(key=lambda r: float(r["auroc"]), reverse=True)
            best_by_benchmark[benchmark][target] = sub[0]

    if metric_rows:
        plot_top_features(metric_rows, os.path.join(args.out_dir, "harm_top_features.png"), "harm")
        plot_top_features(metric_rows, os.path.join(args.out_dir, "help_top_features.png"), "help")
    if args.focus_feature in feats:
        plot_quantile_risk(rows, args.focus_feature, os.path.join(args.out_dir, f"{args.focus_feature}_quantile_risk.png"))

    summary = {
        "inputs": {
            "panel_csv": os.path.abspath(args.panel_csv),
            "focus_feature": args.focus_feature,
        },
        "counts": {
            "n_rows": int(len(rows)),
            "benchmarks": benchmarks,
            "n_features": int(len(feats)),
        },
        "best_by_benchmark": best_by_benchmark,
        "top_consistent_harm_features": harm_consistency[:10],
        "top_consistent_help_features": help_consistency[:10],
        "outputs": {
            "feature_metrics_csv": os.path.abspath(metrics_csv),
            "harm_consistency_csv": os.path.abspath(harm_consistency_csv),
            "help_consistency_csv": os.path.abspath(help_consistency_csv),
        },
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[saved]", os.path.abspath(metrics_csv))
    print("[saved]", os.path.abspath(harm_consistency_csv))
    print("[saved]", os.path.abspath(help_consistency_csv))
    print("[saved]", os.path.abspath(os.path.join(args.out_dir, "summary.json")))


if __name__ == "__main__":
    main()
