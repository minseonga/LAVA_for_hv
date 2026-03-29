#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def read_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def parse_bool(x: Any) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def load_summary_best_metric(summary_json: str) -> Tuple[Optional[str], Optional[int]]:
    if not os.path.isfile(summary_json):
        return None, None
    try:
        with open(summary_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None, None
    best = obj.get("best_layer_eval") or {}
    metric_base = best.get("metric_base")
    layer = best.get("block_layer_idx")
    try:
        layer = int(layer) if layer is not None else None
    except Exception:
        layer = None
    return (None if metric_base is None else str(metric_base), layer)


def cohen_d(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx = float(sum(xs) / len(xs))
    my = float(sum(ys) / len(ys))
    vx = float(sum((x - mx) ** 2 for x in xs) / (len(xs) - 1))
    vy = float(sum((y - my) ** 2 for y in ys) / (len(ys) - 1))
    denom = math.sqrt(max(1e-12, ((len(xs) - 1) * vx + (len(ys) - 1) * vy) / (len(xs) + len(ys) - 2)))
    return float((mx - my) / denom)


def collect_metric_by_layer(
    rows_trace: List[Dict[str, Any]],
    metric_name: str,
    group_pos: str,
    group_neg: str,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], List[int]]:
    vals_pos_by_layer: Dict[int, List[float]] = defaultdict(list)
    vals_neg_by_layer: Dict[int, List[float]] = defaultdict(list)
    layers_set = set()

    for r in rows_trace:
        layer = safe_float(r.get("block_layer_idx"))
        if layer is None:
            continue
        li = int(layer)
        v = safe_float(r.get(metric_name))
        if v is None:
            continue
        layers_set.add(li)

        if parse_bool(r.get("is_fp_hallucination")) and group_pos == "fp_hall":
            vals_pos_by_layer[li].append(float(v))
        elif parse_bool(r.get("is_tp_yes")) and group_pos == "tp_yes":
            vals_pos_by_layer[li].append(float(v))

        if parse_bool(r.get("is_fp_hallucination")) and group_neg == "fp_hall":
            vals_neg_by_layer[li].append(float(v))
        elif parse_bool(r.get("is_tp_yes")) and group_neg == "tp_yes":
            vals_neg_by_layer[li].append(float(v))

    return vals_pos_by_layer, vals_neg_by_layer, sorted(layers_set)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize FP-vs-TP layer separation in POPE layer-trace outputs.")
    ap.add_argument("--exp_dir", type=str, required=True, help="Experiment directory with layer_eval/layer_curve/per_layer files.")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument(
        "--metrics",
        type=str,
        default="yes_sim_local_topk,yes_sim_objpatch_topk,yes_attn_vis_sum",
        help="Comma-separated metric_base names from layer_eval (e.g., yes_sim_local_topk).",
    )
    ap.add_argument("--group_pos", type=str, default="fp_hall")
    ap.add_argument("--group_neg", type=str, default="tp_yes")
    ap.add_argument("--best_metric", type=str, default="", help="Override best metric for distribution plot.")
    ap.add_argument("--best_layer", type=int, default=-999, help="Override best layer for distribution plot.")
    args = ap.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(exp_dir, "viz_layer_separation"))
    os.makedirs(out_dir, exist_ok=True)

    path_eval = os.path.join(exp_dir, "layer_eval_fp_vs_tp_yes.csv")
    path_curve = os.path.join(exp_dir, "layer_curve_yes_by_group.csv")
    path_trace = os.path.join(exp_dir, "per_layer_yes_trace.csv")
    path_summary = os.path.join(exp_dir, "summary.json")
    for p in [path_eval, path_curve, path_trace]:
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing required file: {p}")

    rows_eval = read_csv(path_eval)
    rows_curve = read_csv(path_curve)
    rows_trace = read_csv(path_trace)

    metrics_req = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if len(metrics_req) == 0:
        raise RuntimeError("No metrics selected.")

    # metric -> sorted (layer, auc_best, auc_hall, ks, direction)
    eval_by_metric: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows_eval:
        mb = str(r.get("metric_base") or "")
        if mb == "":
            continue
        if mb not in metrics_req:
            continue
        layer = safe_float(r.get("block_layer_idx"))
        auc_best = safe_float(r.get("auc_best_dir"))
        auc_hall = safe_float(r.get("auc_hall_high"))
        ks = safe_float(r.get("ks_hall_high"))
        if layer is None or auc_best is None:
            continue
        eval_by_metric[mb].append(
            {
                "layer": int(layer),
                "auc_best": float(auc_best),
                "auc_hall": (None if auc_hall is None else float(auc_hall)),
                "ks": (None if ks is None else float(ks)),
                "direction": str(r.get("direction") or ""),
            }
        )
    for m in list(eval_by_metric.keys()):
        eval_by_metric[m] = sorted(eval_by_metric[m], key=lambda x: int(x["layer"]))

    # metric -> group -> sorted (layer, mean)
    curve_by_metric_group: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows_curve:
        g = str(r.get("group") or "")
        layer = safe_float(r.get("block_layer_idx"))
        if layer is None:
            continue
        for m in metrics_req:
            key_mean = f"{m}__mean"
            v = safe_float(r.get(key_mean))
            if v is None:
                continue
            curve_by_metric_group[m][g].append((int(layer), float(v)))
    for m in list(curve_by_metric_group.keys()):
        for g in list(curve_by_metric_group[m].keys()):
            curve_by_metric_group[m][g] = sorted(curve_by_metric_group[m][g], key=lambda x: x[0])

    # 01: AUC-by-layer
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8), dpi=150)
    for m in metrics_req:
        vals = eval_by_metric.get(m, [])
        if len(vals) == 0:
            continue
        xs = [v["layer"] for v in vals]
        ys = [v["auc_best"] for v in vals]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.7, label=m)
    ax.axhline(0.5, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Block Layer")
    ax.set_ylabel("AUC (best direction)")
    ax.set_title("FP vs TP Separation by Layer")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_auc_by_layer.png"))
    plt.close(fig)

    # 02: Group mean curves by layer
    ncols = 2
    nrows = int(math.ceil(len(metrics_req) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), dpi=150)
    axes_list = np.array(axes).reshape(-1)
    for i, m in enumerate(metrics_req):
        ax = axes_list[i]
        for g, color in [(args.group_pos, "#d62728"), (args.group_neg, "#1f77b4")]:
            vals = curve_by_metric_group.get(m, {}).get(g, [])
            if len(vals) == 0:
                continue
            xs = [x for x, _ in vals]
            ys = [y for _, y in vals]
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.5, label=g, color=color)
        ax.set_title(m)
        ax.set_xlabel("Block Layer")
        ax.set_ylabel("Mean")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    for j in range(len(metrics_req), len(axes_list)):
        axes_list[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_group_mean_by_layer.png"))
    plt.close(fig)

    # 03: FP-TP gap by layer
    gap_rows: List[Dict[str, Any]] = []
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8), dpi=150)
    for m in metrics_req:
        pos = dict(curve_by_metric_group.get(m, {}).get(args.group_pos, []))
        neg = dict(curve_by_metric_group.get(m, {}).get(args.group_neg, []))
        layers = sorted(set(pos.keys()) & set(neg.keys()))
        if len(layers) == 0:
            continue
        gaps = [float(pos[l] - neg[l]) for l in layers]
        ax.plot(layers, gaps, marker="o", ms=3, lw=1.6, label=m)
        for l, g in zip(layers, gaps):
            gap_rows.append({"metric_base": m, "block_layer_idx": int(l), "mean_gap_pos_minus_neg": float(g)})
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Block Layer")
    ax.set_ylabel(f"Mean Gap ({args.group_pos} - {args.group_neg})")
    ax.set_title("Where The Split Comes From (Mean Gap)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_group_gap_by_layer.png"))
    plt.close(fig)
    write_csv(os.path.join(out_dir, "layer_gap_table.csv"), gap_rows)

    # 04: Distribution on best layer/metric
    best_metric_auto, best_layer_auto = load_summary_best_metric(path_summary)
    best_metric = str(args.best_metric).strip() or (best_metric_auto or metrics_req[0])
    best_layer = int(args.best_layer) if int(args.best_layer) != -999 else (best_layer_auto if best_layer_auto is not None else 0)

    vals_pos_by_layer, vals_neg_by_layer, layers_all = collect_metric_by_layer(
        rows_trace=rows_trace,
        metric_name=best_metric,
        group_pos=args.group_pos,
        group_neg=args.group_neg,
    )
    vals_pos: List[float] = vals_pos_by_layer.get(int(best_layer), [])
    vals_neg: List[float] = vals_neg_by_layer.get(int(best_layer), [])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax0, ax1 = axes
    if len(vals_pos) > 0 and len(vals_neg) > 0:
        bins = 40
        ax0.hist(vals_neg, bins=bins, alpha=0.55, label=args.group_neg, color="#1f77b4", density=True)
        ax0.hist(vals_pos, bins=bins, alpha=0.55, label=args.group_pos, color="#d62728", density=True)
        ax0.set_title(f"Distribution @ layer {best_layer} ({best_metric})")
        ax0.set_xlabel(best_metric)
        ax0.set_ylabel("Density")
        ax0.legend(loc="best", fontsize=8)
        ax0.grid(True, alpha=0.2)

        ax1.boxplot([vals_neg, vals_pos], tick_labels=[args.group_neg, args.group_pos], showfliers=False)
        ax1.set_title("Boxplot")
        ax1.set_ylabel(best_metric)
        ax1.grid(True, axis="y", alpha=0.2)
    else:
        ax0.text(0.5, 0.5, "Insufficient values", ha="center", va="center")
        ax1.text(0.5, 0.5, "Insufficient values", ha="center", va="center")
        ax0.axis("off")
        ax1.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_distribution_best_layer.png"))
    plt.close(fig)

    # 05: Distribution grid across all layers for selected metric
    if len(layers_all) > 0:
        n_layers = len(layers_all)
        ncols = 8
        nrows = int(math.ceil(n_layers / ncols))
        fig_w = max(16.0, 2.2 * ncols)
        fig_h = max(7.0, 1.9 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=150, sharex=True, sharey=True)
        axes_list = np.array(axes).reshape(-1)

        all_vals = []
        for l in layers_all:
            all_vals.extend(vals_pos_by_layer.get(int(l), []))
            all_vals.extend(vals_neg_by_layer.get(int(l), []))
        bins = 30
        hist_range = None
        if len(all_vals) > 5:
            lo = float(np.quantile(np.array(all_vals), 0.01))
            hi = float(np.quantile(np.array(all_vals), 0.99))
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                hist_range = (lo, hi)

        for i, layer in enumerate(layers_all):
            ax = axes_list[i]
            vp = vals_pos_by_layer.get(int(layer), [])
            vn = vals_neg_by_layer.get(int(layer), [])
            if len(vp) > 0 and len(vn) > 0:
                ax.hist(vn, bins=bins, range=hist_range, alpha=0.50, color="#1f77b4", density=True)
                ax.hist(vp, bins=bins, range=hist_range, alpha=0.50, color="#d62728", density=True)
                mu_n = float(sum(vn) / len(vn))
                mu_p = float(sum(vp) / len(vp))
                ax.axvline(mu_n, color="#1f77b4", lw=0.8, ls="--")
                ax.axvline(mu_p, color="#d62728", lw=0.8, ls="--")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.set_title(f"L{int(layer)}", fontsize=8)
            ax.grid(True, alpha=0.15)

        for j in range(n_layers, len(axes_list)):
            axes_list[j].axis("off")

        fig.suptitle(
            f"Distribution Grid by Layer ({best_metric}): {args.group_pos} vs {args.group_neg}",
            fontsize=11,
            y=0.995,
        )
        handles = [
            plt.Line2D([0], [0], color="#1f77b4", lw=6, alpha=0.5),
            plt.Line2D([0], [0], color="#d62728", lw=6, alpha=0.5),
        ]
        fig.legend(handles, [args.group_neg, args.group_pos], loc="upper right", fontsize=8)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(os.path.join(out_dir, "05_distribution_all_layers_grid.png"))
        plt.close(fig)

    # Optional numeric note for best layer
    summary_rows: List[Dict[str, Any]] = []
    if len(vals_pos) > 0 and len(vals_neg) > 0:
        summary_rows.append(
            {
                "metric_base": best_metric,
                "block_layer_idx": int(best_layer),
                "group_pos": args.group_pos,
                "group_neg": args.group_neg,
                "n_pos": int(len(vals_pos)),
                "n_neg": int(len(vals_neg)),
                "mean_pos": float(sum(vals_pos) / len(vals_pos)),
                "mean_neg": float(sum(vals_neg) / len(vals_neg)),
                "median_pos": float(np.median(vals_pos)),
                "median_neg": float(np.median(vals_neg)),
                "cohen_d_pos_minus_neg": cohen_d(vals_pos, vals_neg),
            }
        )
    write_csv(os.path.join(out_dir, "best_layer_distribution_stats.csv"), summary_rows)

    print("[saved]", os.path.join(out_dir, "01_auc_by_layer.png"))
    print("[saved]", os.path.join(out_dir, "02_group_mean_by_layer.png"))
    print("[saved]", os.path.join(out_dir, "03_group_gap_by_layer.png"))
    print("[saved]", os.path.join(out_dir, "04_distribution_best_layer.png"))
    if len(layers_all) > 0:
        print("[saved]", os.path.join(out_dir, "05_distribution_all_layers_grid.png"))
    print("[saved]", os.path.join(out_dir, "layer_gap_table.csv"))
    print("[saved]", os.path.join(out_dir, "best_layer_distribution_stats.csv"))


if __name__ == "__main__":
    main()
