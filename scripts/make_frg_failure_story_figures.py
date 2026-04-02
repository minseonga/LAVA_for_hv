#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OFFLINE_SUMMARY = REPO_ROOT / "experiments/pope_full_9000/vga_discovery_headset_frg_only_offline_9000/summary.json"
DEFAULT_DISCOVERY_CONTROLLER_SUMMARY = REPO_ROOT / "experiments/pope_discovery/tau_c_calibration_adversarial/controller/summary.json"
DEFAULT_ONLINE_LIVE_SUMMARY = REPO_ROOT / "experiments/vga_discovery_headset_frg_only_online_9000_live_branch/summary.json"
DEFAULT_ONLINE_SAME_SUMMARY = REPO_ROOT / "experiments/vga_discovery_headset_frg_only_online_9000_same_branch/summary.json"
DEFAULT_PARITY_SUMMARY = REPO_ROOT / "experiments/vga_discovery_headset_frg_only_online_9000_same_branch/parity_report/summary.json"
DEFAULT_ORDERING_SUMMARY = REPO_ROOT / "experiments/vga_discovery_headset_frg_only_online_9000_same_branch/ordering_report/summary.json"
DEFAULT_DISCOVERY_OFFLINE_FEATURE_SUMMARY = REPO_ROOT / "experiments/pope_discovery/tau_c_runtime_fullseq_calibonly/controller_offline_feature_check/summary.json"
DEFAULT_DISCOVERY_RUNTIME_PROBE_SUMMARY = REPO_ROOT / "experiments/pope_discovery/tau_c_runtime_fullseq_calibonly/controller/summary.json"
DEFAULT_STAGEB_SUMMARY = REPO_ROOT / "experiments/stage_b_validation/summary.json"
DEFAULT_STAGEB_PAIRWISE = REPO_ROOT / "experiments/stage_b_validation/pairwise_metrics.csv"
DEFAULT_STAGEB_GROUP = REPO_ROOT / "experiments/stage_b_validation/group_stats.csv"
DEFAULT_STAGEB_BUDGET = REPO_ROOT / "experiments/stage_b_validation/budget_best.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/presentation_figures/frg_failure_story"


COLORS = {
    "baseline": "#7f8c8d",
    "vga": "#1f77b4",
    "offline": "#1b9e77",
    "live": "#d95f02",
    "same": "#e7298a",
    "runtime": "#d95f02",
    "ordering": "#7570b3",
    "stageb": "#66a61e",
    "danger": "#d73027",
    "warning": "#fc8d59",
    "good": "#1a9850",
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(value: float) -> float:
    return 100.0 * float(value)


def maybe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str) and value.strip() == "":
        return float("nan")
    return float(value)


def annotate_bars(ax: plt.Axes, bars: Iterable[Any], fmt: str, dy: float) -> None:
    for bar in bars:
        height = float(bar.get_height())
        if not math.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def add_note_box(ax: plt.Axes, title: str, lines: Sequence[str], xy: tuple[float, float]) -> None:
    text = title + "\n" + "\n".join(lines)
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f7f7f7",
            "edgecolor": "#333333",
            "linewidth": 1.0,
        },
    )


def add_footer(fig: plt.Figure, lines: Sequence[str]) -> None:
    if not lines:
        return
    text = "\n".join(lines)
    fig.text(
        0.5,
        0.018,
        text,
        ha="center",
        va="bottom",
        fontsize=10.8,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f7f7f7",
            "edgecolor": "#444444",
            "linewidth": 1.0,
        },
    )


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
        }
    )


def savefig(fig: plt.Figure, path: Path, rect: tuple[float, float, float, float] = (0.02, 0.04, 0.98, 0.94)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_01_offline_vs_online_gap(
    out_path: Path,
    offline_summary: Dict[str, Any],
    discovery_summary: Dict[str, Any],
    online_live_summary: Dict[str, Any],
    online_same_summary: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    acc_labels = ["Baseline", "VGA", "Offline\nFRG-only", "Online\nlive", "Online\nsame"]
    acc_values = [
        pct(offline_summary["metrics"]["baseline"]["acc"]),
        pct(offline_summary["metrics"]["vga"]["acc"]),
        pct(offline_summary["metrics"]["controller"]["acc"]),
        pct(online_live_summary["metrics"]["acc"]),
        pct(online_same_summary["metrics"]["acc"]),
    ]
    acc_colors = [
        COLORS["baseline"],
        COLORS["vga"],
        COLORS["offline"],
        COLORS["live"],
        COLORS["same"],
    ]
    bars = axes[0].bar(acc_labels, acc_values, color=acc_colors, width=0.72)
    annotate_bars(axes[0], bars, "{:.2f}", dy=0.25)
    axes[0].set_ylim(82.5, 88.5)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Offline Success, Online Collapse")
    axes[0].grid(axis="y")
    axes[0].grid(axis="x", visible=False)

    veto_labels = ["Offline\nFRG-only", "Online\nlive", "Online\nsame"]
    veto_values = [
        pct(offline_summary["counts"]["veto_rate"]),
        pct(online_live_summary["counts"]["veto_rate"]),
        pct(online_same_summary["counts"]["veto_rate"]),
    ]
    veto_colors = [COLORS["offline"], COLORS["live"], COLORS["same"]]
    bars = axes[1].bar(veto_labels, veto_values, color=veto_colors, width=0.72)
    annotate_bars(axes[1], bars, "{:.1f}", dy=0.8)
    axes[1].set_ylim(0.0, 85.0)
    axes[1].set_ylabel("Veto rate (%)")
    axes[1].set_title("Same Discovery Tau, Much More Baseline Fallback", pad=10)
    axes[1].grid(axis="y")
    axes[1].grid(axis="x", visible=False)

    add_footer(
        fig,
        [
            f"Reuse online discovery tau_c = {discovery_summary['thresholds']['tau_c']:.5f}",
            f"Offline delta vs VGA = {offline_summary['metrics']['controller_delta_vs_vga']['delta_acc'] * 100:+.2f} pts | "
            f"Online same delta vs VGA = {(online_same_summary['metrics']['acc'] - offline_summary['metrics']['vga']['acc']) * 100:+.2f} pts",
        ],
    )
    fig.suptitle("FRG-only Failure Story 1: the offline controller did not transfer to runtime", fontsize=17, fontweight="bold")
    savefig(fig, out_path, rect=(0.02, 0.12, 0.98, 0.93))


def figure_02_branch_source_not_main_cause(
    out_path: Path,
    online_live_summary: Dict[str, Any],
    online_same_summary: Dict[str, Any],
    parity_summary: Dict[str, Any],
) -> None:
    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    labels = ["live", "same"]
    acc = [pct(online_live_summary["metrics"]["acc"]), pct(online_same_summary["metrics"]["acc"])]
    veto = [pct(online_live_summary["counts"]["veto_rate"]), pct(online_same_summary["counts"]["veto_rate"])]
    x = list(range(len(labels)))
    width = 0.36
    bars1 = ax.bar([v - width / 2.0 for v in x], acc, width=width, color=COLORS["vga"], label="accuracy")
    bars2 = ax.bar([v + width / 2.0 for v in x], veto, width=width, color=COLORS["warning"], label="veto rate")
    annotate_bars(ax, bars1, "{:.2f}", dy=0.6)
    annotate_bars(ax, bars2, "{:.1f}", dy=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 85)
    ax.set_ylabel("Percent")
    ax.set_title("Changing branch source barely moved the controller")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax2.axis("off")
    metrics = [
        ("baseline branch match", f"{pct(parity_summary['baseline_branch']['match_rate']):.1f}%"),
        ("FRG Pearson", f"{parity_summary['frg_compare']['pearson']:.3f}"),
        ("mean shift (runtime - offline)", f"{parity_summary['frg_compare']['delta_runtime_minus_offline_mean']:+.4f}"),
        ("sign flips", f"{int(parity_summary['frg_compare']['sign_flip_count']):,}"),
        ("runtime-only >= tau", f"{int(parity_summary['tau_compare']['runtime_only_ge_tau']):,}"),
        ("offline-only >= tau", f"{int(parity_summary['tau_compare']['offline_only_ge_tau']):,}"),
    ]
    ax2.text(0.0, 1.02, "Same-branch parity still showed FRG mismatch", fontsize=15, fontweight="bold", ha="left", va="top")
    y = 0.88
    for name, value in metrics:
        ax2.text(
            0.02,
            y,
            f"{name}: {value}",
            ha="left",
            va="top",
            fontsize=11.5,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#f7f7f7",
                "edgecolor": "#555555",
                "linewidth": 0.9,
            },
        )
        y -= 0.135
    ax2.text(
        0.02,
        0.05,
        "Conclusion: branch mismatch was not the main failure mode.\nEven when the branch text matched, runtime FRG stayed misaligned.",
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.suptitle("FRG-only Failure Story 2: branch source was not the root cause", fontsize=17, fontweight="bold")
    savefig(fig, out_path)


def figure_03_ordering_mismatch(
    out_path: Path,
    ordering_summary: Dict[str, Any],
    offline_summary: Dict[str, Any],
    online_same_summary: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))

    corr_labels = ["Pearson", "Spearman", "Kendall"]
    corr_values = [
        ordering_summary["correlation"]["pearson"],
        ordering_summary["correlation"]["spearman"],
        ordering_summary["correlation"]["kendall"],
    ]
    bars = axes[0].bar(corr_labels, corr_values, color=[COLORS["vga"], COLORS["ordering"], COLORS["warning"]], width=0.68)
    annotate_bars(axes[0], bars, "{:.3f}", dy=0.018)
    axes[0].set_ylim(0.0, 0.82)
    axes[0].set_ylabel("Correlation")
    axes[0].set_title("Runtime FRG did not preserve offline ordering")
    axes[0].grid(axis="y")
    axes[0].grid(axis="x", visible=False)

    acc_labels = ["Online same\nactual", "Runtime best\nthreshold upper bound", "VGA-only", "Offline\nFRG-only"]
    acc_values = [
        pct(online_same_summary["metrics"]["acc"]),
        pct(ordering_summary["runtime_best_threshold_upper_bound"]["acc"]),
        pct(offline_summary["metrics"]["vga"]["acc"]),
        pct(offline_summary["metrics"]["controller"]["acc"]),
    ]
    bars = axes[1].bar(
        acc_labels,
        acc_values,
        color=[COLORS["danger"], COLORS["warning"], COLORS["vga"], COLORS["offline"]],
        width=0.72,
    )
    annotate_bars(axes[1], bars, "{:.2f}", dy=0.22)
    axes[1].set_ylim(84.0, 88.2)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Monotone retuning could not recover offline performance")
    axes[1].grid(axis="y")
    axes[1].grid(axis="x", visible=False)

    add_footer(
        fig,
        [
            f"Best runtime tau = {ordering_summary['runtime_best_threshold_upper_bound']['tau']:.6f} | "
            f"Best runtime veto rate = {pct(ordering_summary['runtime_best_threshold_upper_bound']['veto_rate']):.1f}%",
            "Even the best monotone threshold on current runtime ordering stays below offline FRG-only.",
        ],
    )
    fig.suptitle("FRG-only Failure Story 3: the bottleneck was ordering mismatch, not just mean shift", fontsize=17, fontweight="bold")
    savefig(fig, out_path, rect=(0.02, 0.12, 0.98, 0.93))


def figure_04_extractor_mismatch(
    out_path: Path,
    discovery_offline_feature_summary: Dict[str, Any],
    discovery_runtime_probe_summary: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))

    labels = ["Baseline", "VGA", "Offline feature\ncontroller", "Runtime fullseq\ncontroller"]
    acc_values = [
        pct(discovery_offline_feature_summary["metrics"]["baseline"]["acc"]),
        pct(discovery_offline_feature_summary["metrics"]["vga"]["acc"]),
        pct(discovery_offline_feature_summary["metrics"]["controller"]["acc"]),
        pct(discovery_runtime_probe_summary["metrics"]["controller"]["acc"]),
    ]
    bars = axes[0].bar(
        labels,
        acc_values,
        color=[COLORS["baseline"], COLORS["vga"], COLORS["offline"], COLORS["runtime"]],
        width=0.72,
    )
    annotate_bars(axes[0], bars, "{:.2f}", dy=0.20)
    axes[0].set_ylim(80.5, 87.0)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Same discovery slice, very different controller outcomes")
    axes[0].grid(axis="y")
    axes[0].grid(axis="x", visible=False)

    veto_labels = ["Offline feature\ncontroller", "Runtime fullseq\ncontroller"]
    veto_values = [
        pct(discovery_offline_feature_summary["counts"]["veto_rate"]),
        pct(discovery_runtime_probe_summary["counts"]["veto_rate"]),
    ]
    bars = axes[1].bar(veto_labels, veto_values, color=[COLORS["offline"], COLORS["runtime"]], width=0.6)
    annotate_bars(axes[1], bars, "{:.2f}", dy=0.8)
    axes[1].set_ylim(0.0, 35.0)
    axes[1].set_ylabel("Veto rate (%)")
    axes[1].set_title("Runtime probe almost never fired")
    axes[1].grid(axis="y")
    axes[1].grid(axis="x", visible=False)

    add_footer(
        fig,
        [
            f"Offline feature net gain vs VGA = {discovery_offline_feature_summary['metrics']['controller_delta_vs_vga']['net_gain']:+d}",
            f"Runtime fullseq net gain vs VGA = {discovery_runtime_probe_summary['metrics']['controller_delta_vs_vga']['net_gain']:+d}",
            "Same final formula, but a different raw producer path made the runtime probe almost inert.",
        ],
    )
    fig.suptitle("FRG-only Failure Story 4: the runtime extractor path did not reproduce the offline signal", fontsize=17, fontweight="bold")
    savefig(fig, out_path, rect=(0.02, 0.12, 0.98, 0.93))


def figure_05_stageb_survives(
    out_path: Path,
    stageb_summary: Dict[str, Any],
    stageb_pairwise_rows: List[Dict[str, str]],
    stageb_group_rows: List[Dict[str, str]],
    stageb_budget_rows: List[Dict[str, str]],
    discovery_controller_summary: Dict[str, Any],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))

    pair_map = {row["comparison"]: row for row in stageb_pairwise_rows}
    order = [
        ("regression_vs_non_regression", "regression vs\nnon-regression"),
        ("regression_vs_improvement", "regression vs\nimprovement"),
        ("regression_vs_both_correct", "regression vs\nboth correct"),
        ("regression_vs_both_wrong", "regression vs\nboth wrong"),
    ]
    auroc_values = [maybe_float(pair_map[key]["auroc"]) for key, _ in order]
    bars = axes[0].bar(
        [label for _, label in order],
        auroc_values,
        color=[COLORS["stageb"], COLORS["warning"], COLORS["vga"], COLORS["baseline"]],
        width=0.72,
    )
    annotate_bars(axes[0], bars, "{:.3f}", dy=0.018)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Stage-B path verifier retained real signal")
    axes[0].grid(axis="y")
    axes[0].grid(axis="x", visible=False)

    group_map = {row["case_type"]: row for row in stageb_group_rows}
    group_order = [
        ("both_correct", "both\ncorrect"),
        ("both_wrong", "both\nwrong"),
        ("vga_improvement", "VGA\nimprovement"),
        ("vga_regression", "VGA\nregression"),
    ]
    stageb_scores = [maybe_float(group_map[key]["stage_b_score_mean"]) for key, _ in group_order]
    bars = axes[1].bar(
        [label for _, label in group_order],
        stageb_scores,
        color=[COLORS["vga"], COLORS["baseline"], COLORS["good"], COLORS["danger"]],
        width=0.72,
    )
    annotate_bars(axes[1], bars, "{:.3f}", dy=0.02)
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_ylim(-0.82, 0.08)
    axes[1].set_ylabel("Mean Stage-B score")
    axes[1].set_title("Regression cases were the least grounded on average", pad=10)
    axes[1].grid(axis="y")
    axes[1].grid(axis="x", visible=False)

    budget_best = next((row for row in stageb_budget_rows if str(row.get("flag_budget")) == "0.1"), None)
    if budget_best is None and stageb_budget_rows:
        budget_best = stageb_budget_rows[0]
    if budget_best is not None:
        delta_vs_vga = maybe_float(budget_best["counterfactual_rescue_final_acc"]) - discovery_controller_summary["metrics"]["vga"]["acc"]
        add_footer(
            fig,
            [
                f"Low-budget rescue regime: flagged rate = {pct(budget_best['flagged_rate']):.2f}%, "
                f"regression precision = {pct(budget_best['regression_precision']):.1f}%, "
                f"regression recall = {pct(budget_best['regression_recall']):.1f}%",
                f"Counterfactual accuracy delta vs VGA = {delta_vs_vga * 100:+.2f} pts",
            ],
        )
    fig.suptitle("What survived: Stage-B path verification behaved like a high-precision risk trigger", fontsize=17, fontweight="bold")
    savefig(fig, out_path, rect=(0.02, 0.12, 0.98, 0.93))


def build_manifest(out_dir: Path, files: Sequence[Path]) -> None:
    manifest = {
        "out_dir": str(out_dir.resolve()),
        "figures": [{"file": path.name, "path": str(path.resolve())} for path in files],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build presentation-ready figures for the FRG failure story.")
    ap.add_argument("--offline_summary", type=str, default=str(DEFAULT_OFFLINE_SUMMARY))
    ap.add_argument("--discovery_controller_summary", type=str, default=str(DEFAULT_DISCOVERY_CONTROLLER_SUMMARY))
    ap.add_argument("--online_live_summary", type=str, default=str(DEFAULT_ONLINE_LIVE_SUMMARY))
    ap.add_argument("--online_same_summary", type=str, default=str(DEFAULT_ONLINE_SAME_SUMMARY))
    ap.add_argument("--parity_summary", type=str, default=str(DEFAULT_PARITY_SUMMARY))
    ap.add_argument("--ordering_summary", type=str, default=str(DEFAULT_ORDERING_SUMMARY))
    ap.add_argument("--discovery_offline_feature_summary", type=str, default=str(DEFAULT_DISCOVERY_OFFLINE_FEATURE_SUMMARY))
    ap.add_argument("--discovery_runtime_probe_summary", type=str, default=str(DEFAULT_DISCOVERY_RUNTIME_PROBE_SUMMARY))
    ap.add_argument("--stageb_summary", type=str, default=str(DEFAULT_STAGEB_SUMMARY))
    ap.add_argument("--stageb_pairwise_csv", type=str, default=str(DEFAULT_STAGEB_PAIRWISE))
    ap.add_argument("--stageb_group_csv", type=str, default=str(DEFAULT_STAGEB_GROUP))
    ap.add_argument("--stageb_budget_csv", type=str, default=str(DEFAULT_STAGEB_BUDGET))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    configure_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_summary = load_json(Path(args.offline_summary))
    discovery_controller_summary = load_json(Path(args.discovery_controller_summary))
    online_live_summary = load_json(Path(args.online_live_summary))
    online_same_summary = load_json(Path(args.online_same_summary))
    parity_summary = load_json(Path(args.parity_summary))
    ordering_summary = load_json(Path(args.ordering_summary))
    discovery_offline_feature_summary = load_json(Path(args.discovery_offline_feature_summary))
    discovery_runtime_probe_summary = load_json(Path(args.discovery_runtime_probe_summary))
    stageb_summary = load_json(Path(args.stageb_summary))
    stageb_pairwise_rows = load_csv(Path(args.stageb_pairwise_csv))
    stageb_group_rows = load_csv(Path(args.stageb_group_csv))
    stageb_budget_rows = load_csv(Path(args.stageb_budget_csv))

    files = [
        out_dir / "01_offline_vs_online_gap.png",
        out_dir / "02_branch_source_not_main_cause.png",
        out_dir / "03_ordering_mismatch_ceiling.png",
        out_dir / "04_extractor_mismatch_on_discovery.png",
        out_dir / "05_stageb_path_signal.png",
    ]

    figure_01_offline_vs_online_gap(files[0], offline_summary, discovery_controller_summary, online_live_summary, online_same_summary)
    figure_02_branch_source_not_main_cause(files[1], online_live_summary, online_same_summary, parity_summary)
    figure_03_ordering_mismatch(files[2], ordering_summary, offline_summary, online_same_summary)
    figure_04_extractor_mismatch(files[3], discovery_offline_feature_summary, discovery_runtime_probe_summary)
    figure_05_stageb_survives(files[4], stageb_summary, stageb_pairwise_rows, stageb_group_rows, stageb_budget_rows, discovery_controller_summary)
    build_manifest(out_dir, files)

    for path in files:
        print(f"[saved] {path}")
    print(f"[saved] {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
