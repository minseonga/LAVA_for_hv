#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    }
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_boolish(x: str) -> int:
    s = str(x).strip().lower()
    return 1 if s in {"1", "true", "yes"} else 0


def classify_disc_case(row: dict) -> str:
    base = parse_boolish(row.get("baseline_correct", "0"))
    intr = parse_boolish(row.get("intervention_correct", "0"))
    if intr == 1 and base == 0:
        return "help"
    if intr == 0 and base == 1:
        return "harm"
    return "neutral"


def classify_gen_case(row: dict) -> str:
    if parse_boolish(row.get("help", "0")) == 1:
        return "help"
    if parse_boolish(row.get("harm", "0")) == 1:
        return "harm"
    return "neutral"


def orientation(values: List[float], direction: str) -> List[float]:
    sign = 1.0 if direction == "high" else -1.0
    return [sign * v for v in values]


def zscore(values: List[float]) -> List[float]:
    if not values:
        return values
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0 for _ in values]
    return [(v - mean) / sd for v in values]


def save_fig(fig: plt.Figure, path: Path) -> None:
    ensure_parent(path)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_figure2_always_on_unsafe(
    vga_test_scores_csv: Path,
    pai_test_scores_csv: Path,
    gen_claim_table_csv: Path,
    out_path: Path,
) -> None:
    disc_specs = [
        ("VGA\nDiscriminative", read_csv(vga_test_scores_csv), classify_disc_case),
        ("PAI\nDiscriminative", read_csv(pai_test_scores_csv), classify_disc_case),
        ("VGA\nGenerative", read_csv(gen_claim_table_csv), classify_gen_case),
    ]
    colors = {"help": "#2E8B57", "harm": "#C0392B", "neutral": "#9AA0A6"}
    labels = []
    help_rates = []
    harm_rates = []
    neutral_rates = []
    counts = []

    for name, rows, fn in disc_specs:
        n = len(rows)
        help_n = sum(1 for r in rows if fn(r) == "help")
        harm_n = sum(1 for r in rows if fn(r) == "harm")
        neutral_n = n - help_n - harm_n
        labels.append(name)
        help_rates.append(help_n / n if n else 0.0)
        harm_rates.append(harm_n / n if n else 0.0)
        neutral_rates.append(neutral_n / n if n else 0.0)
        counts.append(n)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = list(range(len(labels)))
    ax.bar(x, help_rates, color=colors["help"], label="Improvement")
    ax.bar(x, harm_rates, bottom=help_rates, color=colors["harm"], label="Regression")
    ax.bar(
        x,
        neutral_rates,
        bottom=[a + b for a, b in zip(help_rates, harm_rates)],
        color=colors["neutral"],
        label="Neutral",
    )

    for i, n in enumerate(counts):
        ax.text(i, 1.02, f"n={n}", ha="center", va="bottom", fontsize=9)
        ax.text(
            i,
            help_rates[i] / 2 if help_rates[i] > 0.04 else help_rates[i] + 0.03,
            f"{help_rates[i]*100:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if help_rates[i] > 0.08 else colors["help"],
            fontweight="bold",
        )
        ax.text(
            i,
            help_rates[i] + harm_rates[i] / 2 if harm_rates[i] > 0.05 else help_rates[i] + harm_rates[i] + 0.03,
            f"{harm_rates[i]*100:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if harm_rates[i] > 0.08 else colors["harm"],
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Sample Fraction")
    ax.set_title("Figure 2. Always-On Intervention Hides Harmful Subsets")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, out_path)


def plot_figure3_discriminative_signature(
    disc_scores_csv: Path,
    disc_cheap_csv: Path,
    out_path: Path,
) -> None:
    scores = {row["id"]: row for row in read_csv(disc_scores_csv)}
    cheap_rows = read_csv(disc_cheap_csv)
    merged = []
    for row in cheap_rows:
        sid = row["id"]
        if sid not in scores:
            continue
        both = dict(scores[sid])
        both.update(row)
        both["case_type"] = classify_disc_case(both)
        merged.append(both)

    feature_specs = [
        ("cheap_target_gap_content_min", "low", "Target Gap Min"),
        ("cheap_lp_content_min", "low", "Target LogProb Min"),
        ("cheap_conflict_gap_minus_entropy", "high", "Gap-Entropy Conflict"),
    ]
    colors = {"help": "#2E8B57", "harm": "#C0392B", "neutral": "#9AA0A6"}
    order = ["help", "neutral", "harm"]
    pretty = {"help": "Improvement", "neutral": "Neutral", "harm": "Regression"}

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharey=True)
    for ax, (feature, direction, title) in zip(axes, feature_specs):
        vals = []
        labels = []
        for row in merged:
            try:
                vals.append(float(row[feature]))
                labels.append(row["case_type"])
            except Exception:
                continue
        oriented = zscore(orientation(vals, direction))
        grouped = {k: [] for k in order}
        for v, lab in zip(oriented, labels):
            grouped[lab].append(v)
        bp = ax.boxplot(
            [grouped[k] for k in order],
            patch_artist=True,
            widths=0.55,
            showfliers=False,
        )
        for patch, k in zip(bp["boxes"], order):
            patch.set_facecolor(colors[k])
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("#222222")
            med.set_linewidth(1.4)
        ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
        ax.set_xticklabels([pretty[k] for k in order], rotation=15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Oriented z-score")
    fig.suptitle("Figure 3. Discriminative Harmful Signature", y=1.06, fontsize=16)
    fig.text(
        0.5,
        0.99,
        "Answer-critical token collapse features rise sharply on regression cases.",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    save_fig(fig, out_path)


def plot_figure4_meta_arbitration(
    vga_route_csv: Path,
    pai_route_csv: Path,
    out_path: Path,
) -> None:
    configs = [
        ("VGA", read_csv(vga_route_csv)),
        ("PAI", read_csv(pai_route_csv)),
    ]
    colors = {"b_only": "#D55E00", "c_only": "#0072B2", "fusion": "#6A3D9A"}
    markers = {"harm": "X", "help": "o", "neutral": "."}
    labels = {"b_only": "B-only", "c_only": "C-only", "fusion": "Fusion"}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharex=True, sharey=True)
    for ax, (title, rows) in zip(axes, configs):
        expert_rates = {}
        total = len(rows)
        for expert in ["b_only", "c_only", "fusion"]:
            expert_rates[expert] = sum(1 for r in rows if r["expert"] == expert) / total if total else 0.0
        for expert in ["b_only", "c_only", "fusion"]:
            for tag in ["harm", "help", "neutral"]:
                xs, ys = [], []
                for row in rows:
                    if row["expert"] != expert:
                        continue
                    case = "neutral"
                    if parse_boolish(row.get("harm", "0")) == 1:
                        case = "harm"
                    elif parse_boolish(row.get("help", "0")) == 1:
                        case = "help"
                    if case != tag:
                        continue
                    xs.append(float(row["b_score"]))
                    ys.append(float(row["c_score"]))
                if not xs:
                    continue
                ax.scatter(
                    xs,
                    ys,
                    s=26 if tag != "neutral" else 10,
                    c=colors[expert],
                    marker=markers[tag],
                    alpha=0.65 if tag != "neutral" else 0.25,
                    linewidths=0.8,
                )
        ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color="#666666", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.text(
            0.98,
            0.98,
            f"B {expert_rates['b_only']*100:.1f}%\nC {expert_rates['c_only']*100:.1f}%\nF {expert_rates['fusion']*100:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#DDDDDD", alpha=0.9),
        )
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("C risk score")
    for ax in axes:
        ax.set_xlabel("B risk score")
    expert_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[k], markeredgecolor=colors[k], label=labels[k], markersize=7)
        for k in ["b_only", "c_only", "fusion"]
    ]
    case_handles = [
        Line2D([0], [0], marker=markers[k], linestyle="", color="#555555", label={"harm": "Regression", "help": "Improvement", "neutral": "Neutral"}[k], markersize=7)
        for k in ["harm", "help", "neutral"]
    ]
    fig.legend(expert_handles, [h.get_label() for h in expert_handles], ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1.05), title="Chosen expert")
    fig.legend(case_handles, [h.get_label() for h in case_handles], ncol=3, frameon=False, loc="upper right", bbox_to_anchor=(0.89, 1.05), title="Outcome type")
    fig.suptitle("Figure 4. Sample-Wise Expert Arbitration in the Meta-Controller", y=1.12, fontsize=16)
    save_fig(fig, out_path)


def plot_figure5_generative_signature(
    claim_table_csv: Path,
    out_path: Path,
) -> None:
    rows = read_csv(claim_table_csv)
    colors = {"help": "#2E8B57", "harm": "#C0392B", "neutral": "#9AA0A6"}
    xs = {"help": [], "harm": [], "neutral": []}
    ys = {"help": [], "harm": [], "neutral": []}

    for row in rows:
        case = classify_gen_case(row)
        try:
            xs[case].append(float(row["delta_hall_rate"]))
            ys[case].append(float(row["delta_supported_recall"]))
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(7.1, 5.8))
    ax.axvspan(-1.2, 0.0, ymin=0.0, ymax=0.5, color="#F7E6E3", alpha=0.5)
    ax.axvspan(-1.2, 0.0, ymin=0.5, ymax=1.0, color="#E6F4EA", alpha=0.45)
    for case in ["help", "harm", "neutral"]:
        ax.scatter(
            xs[case],
            ys[case],
            s=22 if case != "neutral" else 14,
            alpha=0.72 if case != "neutral" else 0.23,
            c=colors[case],
            label={"help": "Helpful", "harm": "Harmful", "neutral": "Neutral"}[case],
            edgecolors="none",
        )

    ax.axhline(0.0, color="#666666", linewidth=0.9, alpha=0.6)
    ax.axvline(0.0, color="#666666", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Δ Hallucination Rate")
    ax.set_ylabel("Δ Supported Recall")
    ax.set_title("Figure 5. Generative Harmful Signature")
    ax.text(-0.86, 0.44, "Hallucination ↓\nRecall ↑", fontsize=10, color="#2E8B57", ha="left", va="center")
    ax.text(-0.86, -0.60, "Hallucination ↓\nRecall ↓", fontsize=10, color="#B03A2E", ha="left", va="center")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    save_fig(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="figures/paper_main")

    ap.add_argument(
        "--vga_test_scores_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_vga_full/test_stageb/sample_scores.csv",
    )
    ap.add_argument(
        "--pai_test_scores_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/test_stageb/sample_scores.csv",
    )
    ap.add_argument(
        "--vga_disc_scores_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/discovery_stageb/sample_scores.csv",
    )
    ap.add_argument(
        "--vga_disc_cheap_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/discovery/cheap_online_features.csv",
    )
    ap.add_argument(
        "--vga_meta_route_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_vga_full/test/meta_fixed_eval/meta_route_rows.csv",
    )
    ap.add_argument(
        "--pai_meta_route_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/test/meta_fixed_eval/meta_route_rows.csv",
    )
    ap.add_argument(
        "--gen_claim_table_csv",
        type=str,
        default="/Users/gangminseong/LAVA_for_hv/experiments/vga_generative_coverage_probe_v1/vga_claim_aware_table.csv",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_figure2_always_on_unsafe(
        Path(args.vga_test_scores_csv),
        Path(args.pai_test_scores_csv),
        Path(args.gen_claim_table_csv),
        out_dir / "figure2_always_on_unsafe.png",
    )
    plot_figure3_discriminative_signature(
        Path(args.vga_disc_scores_csv),
        Path(args.vga_disc_cheap_csv),
        out_dir / "figure3_discriminative_signature.png",
    )
    plot_figure4_meta_arbitration(
        Path(args.vga_meta_route_csv),
        Path(args.pai_meta_route_csv),
        out_dir / "figure4_meta_arbitration.png",
    )
    plot_figure5_generative_signature(
        Path(args.gen_claim_table_csv),
        out_dir / "figure5_generative_signature.png",
    )
    print("[saved]", out_dir)


if __name__ == "__main__":
    main()
