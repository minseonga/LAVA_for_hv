#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
    }
)


CATEGORY_ORDER = ["adversarial", "popular", "random"]
PRETTY = {
    "adversarial": "Adversarial",
    "popular": "Popular",
    "random": "Random",
}
METHODS = ["Baseline", "Intervention", "Our Method"]
COLORS = {
    "Baseline": "#9AA0A6",
    "Intervention": "#D97706",
    "Our Method": "#1D4ED8",
}
TRANSITION_COLORS = {
    "Resolved Harm": "#2E8B57",
    "Lost Gain": "#C0392B",
}


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_category_acc(category_metrics_csv: str) -> Dict[str, Dict[str, float]]:
    rows = read_csv(category_metrics_csv)
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        cat = row["category"]
        if cat == "overall":
            continue
        out[cat] = {
            "Baseline": float(row["baseline_acc"]),
            "Intervention": float(row["intervention_acc"]),
            "Our Method": float(row["final_acc"]),
        }
    return out


def load_gain_harm_ratio(
    scores_csv: str,
    meta_route_rows_csv: str,
    gt_csv: str,
) -> Dict[str, Dict[str, float]]:
    gt = {row["id"]: row["category"] for row in read_csv(gt_csv)}
    scores = read_csv(scores_csv)
    meta = {row["id"]: row for row in read_csv(meta_route_rows_csv)}

    counts = {
        cat: {"raw_gain": 0, "raw_harm": 0, "meta_gain": 0, "meta_harm": 0}
        for cat in CATEGORY_ORDER
    }

    for row in scores:
        sid = row["id"]
        cat = gt[sid]
        base = int(float(row["baseline_correct"]))
        intr = int(float(row["intervention_correct"]))
        final = int(float(meta[sid]["final_correct"]))

        if base == 0 and intr == 1:
            counts[cat]["raw_gain"] += 1
        elif base == 1 and intr == 0:
            counts[cat]["raw_harm"] += 1

        if base == 0 and final == 1:
            counts[cat]["meta_gain"] += 1
        elif base == 1 and final == 0:
            counts[cat]["meta_harm"] += 1

    out: Dict[str, Dict[str, float]] = {}
    for cat, c in counts.items():
        raw_ratio = c["raw_gain"] / c["raw_harm"] if c["raw_harm"] else 0.0
        meta_ratio = c["meta_gain"] / c["meta_harm"] if c["meta_harm"] else 0.0
        out[cat] = {
            "raw_ratio": raw_ratio,
            "meta_ratio": meta_ratio,
            "raw_gain": float(c["raw_gain"]),
            "raw_harm": float(c["raw_harm"]),
            "meta_gain": float(c["meta_gain"]),
            "meta_harm": float(c["meta_harm"]),
        }
    return out


def build_transition_rates(ratio: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cat, vals in ratio.items():
        raw_harm = vals["raw_harm"]
        raw_gain = vals["raw_gain"]
        meta_harm = vals["meta_harm"]
        meta_gain = vals["meta_gain"]
        resolved_harm = raw_harm - meta_harm
        lost_gain = raw_gain - meta_gain
        out[cat] = {
            "resolved_harm_count": resolved_harm,
            "lost_gain_count": lost_gain,
            "resolved_harm_rate": resolved_harm / raw_harm if raw_harm else 0.0,
            "lost_gain_rate": lost_gain / raw_gain if raw_gain else 0.0,
            "raw_harm": raw_harm,
            "raw_gain": raw_gain,
        }
    return out


def plot_category_acc(ax: Any, acc: Dict[str, Dict[str, float]]) -> None:
    xs = np.arange(len(CATEGORY_ORDER))
    width = 0.23
    for idx, method in enumerate(METHODS):
        vals = [acc[cat][method] for cat in CATEGORY_ORDER]
        bars = ax.bar(xs + (idx - 1) * width, vals, width=width, color=COLORS[method], label=method)
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                v + 0.0025,
                f"{100*v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1F2937",
            )
    ax.set_xticks(xs, [PRETTY[c] for c in CATEGORY_ORDER])
    all_vals = [acc[c][m] for c in CATEGORY_ORDER for m in METHODS]
    ax.set_ylim(min(all_vals) - 0.012, max(all_vals) + 0.02)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")


def plot_gain_harm_ratio(ax: Any, ratio: Dict[str, Dict[str, float]]) -> None:
    ys = np.arange(len(CATEGORY_ORDER))[::-1]
    xmax = max(max(v["raw_ratio"], v["meta_ratio"]) for v in ratio.values()) + 0.22
    for y, cat in zip(ys, CATEGORY_ORDER):
        raw = ratio[cat]["raw_ratio"]
        meta = ratio[cat]["meta_ratio"]
        ax.plot([raw, meta], [y, y], color="#94A3B8", linewidth=2.2, zorder=1)
        ax.scatter([raw], [y], color=COLORS["Intervention"], s=72, zorder=2, label="VGA" if y == ys[0] else None)
        ax.scatter([meta], [y], color=COLORS["Our Method"], s=72, zorder=3, label="Post-hoc" if y == ys[0] else None)
        ax.text(
            raw,
            y + 0.12,
            f"VGA {int(ratio[cat]['raw_gain'])}/{int(ratio[cat]['raw_harm'])}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=COLORS["Intervention"],
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
        ax.text(
            meta,
            y + 0.14,
            f"Post-hoc {int(ratio[cat]['meta_gain'])}/{int(ratio[cat]['meta_harm'])}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=COLORS["Our Method"],
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
    ax.set_yticks(ys, [PRETTY[c] for c in CATEGORY_ORDER])
    ax.set_xlim(0.9, xmax)
    ax.set_xlabel("Gain/Harm Ratio vs Baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="lower right")


def plot_resolved_vs_lost(ax: Any, transition: Dict[str, Dict[str, float]]) -> None:
    xs = np.arange(len(CATEGORY_ORDER))
    width = 0.3
    resolved_vals = [transition[cat]["resolved_harm_rate"] for cat in CATEGORY_ORDER]
    lost_vals = [transition[cat]["lost_gain_rate"] for cat in CATEGORY_ORDER]
    bars_res = ax.bar(
        xs - width / 2,
        resolved_vals,
        width=width,
        color=TRANSITION_COLORS["Resolved Harm"],
        label="Resolved Harm",
    )
    bars_lost = ax.bar(
        xs + width / 2,
        lost_vals,
        width=width,
        color=TRANSITION_COLORS["Lost Gain"],
        label="Lost Gain",
    )

    for rect, cat, val in zip(bars_res, CATEGORY_ORDER, resolved_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.012,
            f"{100*val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1F2937",
        )
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.045,
            f"{int(transition[cat]['resolved_harm_count'])}/{int(transition[cat]['raw_harm'])}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=TRANSITION_COLORS["Resolved Harm"],
        )
    for rect, cat, val in zip(bars_lost, CATEGORY_ORDER, lost_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.012,
            f"{100*val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1F2937",
        )
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.045,
            f"{int(transition[cat]['lost_gain_count'])}/{int(transition[cat]['raw_gain'])}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=TRANSITION_COLORS["Lost Gain"],
        )

    ax.set_xticks(xs, [PRETTY[c] for c in CATEGORY_ORDER])
    ax.set_ylabel("Fraction of VGA Subset")
    ax.set_ylim(0, max(resolved_vals + lost_vals) + 0.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="upper right")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a VGA-only paper-style figure with category accuracy and gain/harm ratio.")
    ap.add_argument(
        "--vga_category_metrics_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_vga_full_strong/test/meta_fixed_eval/category_metrics.csv",
    )
    ap.add_argument(
        "--scores_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/test_stageb/sample_scores.csv",
    )
    ap.add_argument(
        "--meta_route_rows_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_vga_full_strong/test/meta_fixed_eval/meta_route_rows.csv",
    )
    ap.add_argument(
        "--gt_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/pope_full_9000/pope_9000_gt.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/discriminative_method_summary_viz",
    )
    args = ap.parse_args()

    acc = load_category_acc(os.path.abspath(args.vga_category_metrics_csv))
    ratio = load_gain_harm_ratio(
        scores_csv=os.path.abspath(args.scores_csv),
        meta_route_rows_csv=os.path.abspath(args.meta_route_rows_csv),
        gt_csv=os.path.abspath(args.gt_csv),
    )
    transition = build_transition_rates(ratio)

    out_dir = Path(os.path.abspath(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "figure_vga_category_acc_gainharm.png"
    out_pdf = out_dir / "figure_vga_category_acc_gainharm.pdf"
    out_json = out_dir / "figure_vga_category_acc_gainharm.summary.json"

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8), gridspec_kw={"width_ratios": [1.05, 1.02, 0.98]})
    plot_category_acc(axes[0], acc)
    plot_gain_harm_ratio(axes[1], ratio)
    plot_resolved_vs_lost(axes[2], transition)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in METHODS]
    fig.legend(handles, METHODS, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.22, 0.055))
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.98))
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "inputs": {
            "vga_category_metrics_csv": os.path.abspath(args.vga_category_metrics_csv),
            "scores_csv": os.path.abspath(args.scores_csv),
            "meta_route_rows_csv": os.path.abspath(args.meta_route_rows_csv),
            "gt_csv": os.path.abspath(args.gt_csv),
        },
        "category_accuracy": acc,
        "gain_harm_ratio": ratio,
        "transition_rates": transition,
        "outputs": {
            "figure_png": str(out_png),
            "figure_pdf": str(out_pdf),
        },
    }
    write_json(str(out_json), summary)
    print("[saved]", out_png)
    print("[saved]", out_pdf)
    print("[saved]", out_json)


if __name__ == "__main__":
    main()
