#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import os
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


def parse_metrics(s: str) -> List[str]:
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    if len(out) == 0:
        raise RuntimeError("No metrics specified.")
    return out


def parse_int_list(s: str) -> List[int]:
    out = []
    for x in str(s).split(","):
        xx = x.strip()
        if xx == "":
            continue
        out.append(int(xx))
    if len(out) == 0:
        raise RuntimeError("No integer values parsed.")
    return out


def save_heatmap(
    arr: np.ndarray,
    x_ticks: List[int],
    y_ticks: List[int],
    title: str,
    cbar_label: str,
    out_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(max(8.0, 0.55 * len(x_ticks)), max(4.5, 0.45 * len(y_ticks))), dpi=150)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels([str(x) for x in x_ticks], rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=8)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_distribution_grid(
    rows_trace: List[Dict[str, Any]],
    metric: str,
    layer: int,
    group_pos: str,
    group_neg: str,
    out_path: str,
) -> None:
    heads = sorted(
        {
            int(r.get("head_idx"))
            for r in rows_trace
            if safe_float(r.get("block_layer_idx")) is not None
            and int(float(r.get("block_layer_idx"))) == int(layer)
            and safe_float(r.get("head_idx")) is not None
        }
    )
    if len(heads) == 0:
        return

    vals_by_head: Dict[int, Dict[str, List[float]]] = {}
    all_vals: List[float] = []
    for head in heads:
        vals_pos = [
            float(v)
            for v in [
                safe_float(r.get(metric))
                for r in rows_trace
                if int(float(r.get("block_layer_idx"))) == int(layer)
                and int(float(r.get("head_idx"))) == int(head)
                and str(r.get("is_fp_hallucination")).strip().lower() in {"1", "true", "t", "yes", "y"}
                and group_pos == "fp_hall"
            ]
            if v is not None
        ]
        if group_pos != "fp_hall":
            vals_pos = [
                float(v)
                for v in [
                    safe_float(r.get(metric))
                    for r in rows_trace
                    if int(float(r.get("block_layer_idx"))) == int(layer)
                    and int(float(r.get("head_idx"))) == int(head)
                    and str(r.get(f"is_{group_pos}")).strip().lower() in {"1", "true", "t", "yes", "y"}
                ]
                if v is not None
            ]
        vals_neg = [
            float(v)
            for v in [
                safe_float(r.get(metric))
                for r in rows_trace
                if int(float(r.get("block_layer_idx"))) == int(layer)
                and int(float(r.get("head_idx"))) == int(head)
                and str(r.get("is_tp_yes")).strip().lower() in {"1", "true", "t", "yes", "y"}
                and group_neg == "tp_yes"
            ]
            if v is not None
        ]
        if group_neg != "tp_yes":
            vals_neg = [
                float(v)
                for v in [
                    safe_float(r.get(metric))
                    for r in rows_trace
                    if int(float(r.get("block_layer_idx"))) == int(layer)
                    and int(float(r.get("head_idx"))) == int(head)
                    and str(r.get(f"is_{group_neg}")).strip().lower() in {"1", "true", "t", "yes", "y"}
                ]
                if v is not None
            ]
        vals_by_head[int(head)] = {"pos": vals_pos, "neg": vals_neg}
        all_vals.extend(vals_pos)
        all_vals.extend(vals_neg)

    if len(all_vals) == 0:
        return

    arr_all = np.array(all_vals, dtype=np.float32)
    lo = float(np.quantile(arr_all, 0.01))
    hi = float(np.quantile(arr_all, 0.99))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr_all))
        hi = float(np.max(arr_all))
    if hi <= lo:
        hi = lo + 1e-6

    ncols = 8
    nrows = int(math.ceil(len(heads) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 1.8 * nrows), dpi=150, sharex=True, sharey=True)
    axes_list = np.array(axes).reshape(-1)

    for i, head in enumerate(heads):
        ax = axes_list[i]
        vals_pos = vals_by_head[int(head)]["pos"]
        vals_neg = vals_by_head[int(head)]["neg"]
        if len(vals_neg) > 0:
            ax.hist(vals_neg, bins=25, range=(lo, hi), alpha=0.55, density=True, color="#1f77b4")
            ax.axvline(float(np.mean(vals_neg)), color="#1f77b4", lw=0.8, ls="--")
        if len(vals_pos) > 0:
            ax.hist(vals_pos, bins=25, range=(lo, hi), alpha=0.55, density=True, color="#d62728")
            ax.axvline(float(np.mean(vals_pos)), color="#d62728", lw=0.8, ls="--")
        ax.set_title(f"H{int(head)}", fontsize=8)
        ax.grid(True, alpha=0.15)

    for j in range(len(heads), len(axes_list)):
        axes_list[j].axis("off")

    fig.suptitle(f"Layer {int(layer)} Head Distributions: {metric} ({group_pos} vs {group_neg})", fontsize=11, y=0.995)
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=6, alpha=0.55),
        plt.Line2D([0], [0], color="#d62728", lw=6, alpha=0.55),
    ]
    fig.legend(handles, [group_neg, group_pos], loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize POPE head-level separation as layer-head heatmaps.")
    ap.add_argument("--exp_dir", type=str, required=True, help="Experiment dir with head_eval/head_curve outputs.")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument(
        "--metrics",
        type=str,
        default="head_attn_vis_sum,head_attn_vis_ratio,head_attn_vis_peak,head_attn_vis_entropy",
    )
    ap.add_argument("--group_pos", type=str, default="fp_hall")
    ap.add_argument("--group_neg", type=str, default="tp_yes")
    ap.add_argument("--distribution_layers", type=str, default="17,18,19,20")
    args = ap.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(exp_dir, "viz_head_separation"))
    os.makedirs(out_dir, exist_ok=True)

    path_eval = os.path.join(exp_dir, "head_eval_fp_vs_tp_yes.csv")
    path_curve = os.path.join(exp_dir, "head_curve_yes_by_group.csv")
    path_trace = os.path.join(exp_dir, "per_head_yes_trace.csv")
    if not os.path.isfile(path_eval):
        raise RuntimeError(f"Missing required file: {path_eval}")
    if not os.path.isfile(path_curve):
        raise RuntimeError(f"Missing required file: {path_curve}")
    if not os.path.isfile(path_trace):
        raise RuntimeError(f"Missing required file: {path_trace}")

    rows_eval = read_csv(path_eval)
    rows_curve = read_csv(path_curve)
    rows_trace = read_csv(path_trace)
    metrics = parse_metrics(args.metrics)
    distribution_layers = parse_int_list(args.distribution_layers)

    summary_rows: List[Dict[str, Any]] = []
    for metric in metrics:
        metric_eval = [r for r in rows_eval if str(r.get("metric_base") or "") == metric]
        metric_curve = [r for r in rows_curve if True]
        layers = sorted(
            {
                int(r.get("block_layer_idx"))
                for r in metric_eval
                if safe_float(r.get("block_layer_idx")) is not None
            }
        )
        heads = sorted(
            {
                int(r.get("head_idx"))
                for r in metric_eval
                if safe_float(r.get("head_idx")) is not None
            }
        )
        if len(layers) == 0 or len(heads) == 0:
            continue

        auc_arr = np.full((len(layers), len(heads)), np.nan, dtype=np.float32)
        ks_arr = np.full((len(layers), len(heads)), np.nan, dtype=np.float32)
        gap_arr = np.full((len(layers), len(heads)), np.nan, dtype=np.float32)

        idx_l = {v: i for i, v in enumerate(layers)}
        idx_h = {v: i for i, v in enumerate(heads)}

        for r in metric_eval:
            l = safe_float(r.get("block_layer_idx"))
            h = safe_float(r.get("head_idx"))
            auc = safe_float(r.get("auc_best_dir"))
            ks = safe_float(r.get("ks_hall_high"))
            if l is None or h is None:
                continue
            ii = idx_l[int(l)]
            jj = idx_h[int(h)]
            if auc is not None:
                auc_arr[ii, jj] = float(auc)
            if ks is not None:
                ks_arr[ii, jj] = float(ks)

        curve_lookup: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        for r in metric_curve:
            l = safe_float(r.get("block_layer_idx"))
            h = safe_float(r.get("head_idx"))
            g = str(r.get("group") or "")
            if l is None or h is None or g == "":
                continue
            curve_lookup[(int(l), int(h), g)] = r

        for l in layers:
            for h in heads:
                rp = curve_lookup.get((int(l), int(h), str(args.group_pos)))
                rn = curve_lookup.get((int(l), int(h), str(args.group_neg)))
                if rp is None or rn is None:
                    continue
                vp = safe_float(rp.get(f"{metric}__mean"))
                vn = safe_float(rn.get(f"{metric}__mean"))
                if vp is None or vn is None:
                    continue
                gap_arr[idx_l[int(l)], idx_h[int(h)]] = float(vp - vn)

        save_heatmap(
            arr=auc_arr,
            x_ticks=heads,
            y_ticks=layers,
            title=f"AUC Heatmap: {metric}",
            cbar_label="AUC (best direction)",
            out_path=os.path.join(out_dir, f"01_auc_heatmap_{metric}.png"),
            vmin=0.5,
            vmax=max(0.8, float(np.nanmax(auc_arr)) if np.isfinite(np.nanmax(auc_arr)) else 0.8),
            cmap="viridis",
        )
        save_heatmap(
            arr=ks_arr,
            x_ticks=heads,
            y_ticks=layers,
            title=f"KS Heatmap: {metric}",
            cbar_label="KS",
            out_path=os.path.join(out_dir, f"02_ks_heatmap_{metric}.png"),
            vmin=0.0,
            vmax=max(0.4, float(np.nanmax(ks_arr)) if np.isfinite(np.nanmax(ks_arr)) else 0.4),
            cmap="magma",
        )
        vmax_gap = float(np.nanmax(np.abs(gap_arr))) if np.isfinite(np.nanmax(np.abs(gap_arr))) else 0.1
        vmax_gap = max(0.05, vmax_gap)
        save_heatmap(
            arr=gap_arr,
            x_ticks=heads,
            y_ticks=layers,
            title=f"Mean Gap Heatmap ({args.group_pos} - {args.group_neg}): {metric}",
            cbar_label="Mean gap",
            out_path=os.path.join(out_dir, f"03_gap_heatmap_{metric}.png"),
            vmin=-vmax_gap,
            vmax=vmax_gap,
            cmap="coolwarm",
        )

        best = None
        for r in metric_eval:
            auc = safe_float(r.get("auc_best_dir"))
            if auc is None:
                continue
            if best is None or float(auc) > float(safe_float(best.get("auc_best_dir")) or -1.0):
                best = r
        if best is not None:
            summary_rows.append(
                {
                    "metric_base": metric,
                    "best_layer": int(best["block_layer_idx"]),
                    "best_head": int(best["head_idx"]),
                    "auc_best_dir": float(best["auc_best_dir"]),
                    "ks_hall_high": safe_float(best.get("ks_hall_high")),
                    "direction": str(best.get("direction") or ""),
                }
            )

        for layer in distribution_layers:
            save_distribution_grid(
                rows_trace=rows_trace,
                metric=metric,
                layer=int(layer),
                group_pos=str(args.group_pos),
                group_neg=str(args.group_neg),
                out_path=os.path.join(out_dir, f"04_distribution_grid_layer_{int(layer):02d}_{metric}.png"),
            )

    write_csv(os.path.join(out_dir, "head_metric_best_table.csv"), summary_rows)
    for metric in metrics:
        print("[saved]", os.path.join(out_dir, f"01_auc_heatmap_{metric}.png"))
        print("[saved]", os.path.join(out_dir, f"02_ks_heatmap_{metric}.png"))
        print("[saved]", os.path.join(out_dir, f"03_gap_heatmap_{metric}.png"))
        for layer in distribution_layers:
            print("[saved]", os.path.join(out_dir, f"04_distribution_grid_layer_{int(layer):02d}_{metric}.png"))
    print("[saved]", os.path.join(out_dir, "head_metric_best_table.csv"))


if __name__ == "__main__":
    main()
