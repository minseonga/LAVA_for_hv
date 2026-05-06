#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import build_pcp_c_d_controller as pcp
import build_transition_split_fixed_c_median_ensemble as fixed


DIRECTIONS = ("yes_to_no", "no_to_yes")
COLORS = {
    "harm": "#C62828",
    "help": "#2E7D32",
    "random": "#64748B",
    "enrichment": "#F87171",
    "overall": "#111827",
    "yes_to_no": "#2563EB",
    "no_to_yes": "#D97706",
}
VALUE_FONTSIZE = 10.0
LEGEND_FONTSIZE = 11.0
CAPTION_FONTSIZE = 13.2


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def policy_from_item(item: Dict[str, Any], direction: str) -> Dict[str, Any]:
    key = "yes_policy_json" if direction == "yes_to_no" else "no_policy_json"
    policy = item.get(key, {})
    if isinstance(policy, str):
        try:
            return json.loads(policy)
        except Exception:
            return {}
    return dict(policy or {})


def direction_label(direction: str) -> str:
    return "Y->N" if direction == "yes_to_no" else "N->Y"


def add_scores(
    rows: Sequence[Dict[str, Any]],
    *,
    item: Dict[str, Any],
    dataset: str,
    source: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for direction in DIRECTIONS:
        policy = policy_from_item(item, direction)
        features = list(policy.get("selected_c_features") or [])
        tau = policy.get("tau", "")
        disabled = int(bool(policy.get("disabled")) or str(policy.get("family", "")) == "noop")
        for row in rows:
            if not pcp.is_route_candidate(row, direction):
                continue
            score = fixed.median_z_score(row, features) if features else None
            if score is None:
                continue
            harm = int(row.get("harm", 0) or 0)
            help_ = int(row.get("help", 0) or 0)
            selected = int((not disabled) and float(score) >= float(tau))
            out.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "direction": direction,
                    "direction_label": direction_label(direction),
                    "score": float(score),
                    "tau": "" if disabled else float(tau),
                    "disabled": disabled,
                    "harm": harm,
                    "help": help_,
                    "outcome": "harm" if harm else "help" if help_ else "neutral",
                    "selected": selected,
                }
            )
    return out


def threshold_sweep(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = [r for r in rows if int(r.get("harm", 0)) or int(r.get("help", 0))]
    thresholds = sorted({float(r["score"]) for r in scored}, reverse=True)
    harm_total = sum(int(r["harm"]) for r in scored)
    help_total = sum(int(r["help"]) for r in scored)
    out: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []
    prev_tau: Optional[float] = None
    for tau in thresholds:
        selected = [r for r in scored if float(r["score"]) >= float(tau)]
        if prev_tau is not None and math.isclose(float(tau), float(prev_tau), rel_tol=0.0, abs_tol=1e-12):
            continue
        sel_h = sum(int(r["harm"]) for r in selected)
        sel_g = sum(int(r["help"]) for r in selected)
        budget = len(selected) / float(max(1, len(scored)))
        out.append(
            {
                "tau": float(tau),
                "fallback_changed_rate": budget,
                "harm_caught": sel_h / float(max(1, harm_total)),
                "help_lost": sel_g / float(max(1, help_total)),
                "random_expected": budget,
                "selected": len(selected),
                "selected_harm": sel_h,
                "selected_help": sel_g,
            }
        )
        prev_tau = tau
    return out


def harm_enrichment_bins(rows: Sequence[Dict[str, Any]], n_bins: int = 5) -> List[Dict[str, Any]]:
    scored = [
        r
        for r in rows
        if (int(r.get("harm", 0)) or int(r.get("help", 0))) and math.isfinite(float(r.get("score", float("nan"))))
    ]
    scored = sorted(scored, key=lambda r: float(r["score"]))
    if not scored:
        return []
    total_harm = sum(int(r["harm"]) for r in scored)
    overall = total_harm / float(max(1, len(scored)))
    out: List[Dict[str, Any]] = []
    for bin_idx in range(n_bins):
        lo = int(round(bin_idx * len(scored) / float(n_bins)))
        hi = int(round((bin_idx + 1) * len(scored) / float(n_bins)))
        chunk = scored[lo:hi]
        if not chunk:
            continue
        harm = sum(int(r["harm"]) for r in chunk)
        help_ = sum(int(r["help"]) for r in chunk)
        scores = [float(r["score"]) for r in chunk]
        out.append(
            {
                "bin": bin_idx + 1,
                "label": f"Q{bin_idx + 1}",
                "risk_label": "low risk" if bin_idx == 0 else "high risk" if bin_idx == n_bins - 1 else "",
                "n": len(chunk),
                "harm": harm,
                "help": help_,
                "harmful_fraction": harm / float(max(1, len(chunk))),
                "overall_harmful_fraction": overall,
                "score_min": min(scores),
                "score_max": max(scores),
            }
        )
    return out


def bin_edges(values: Sequence[float], n: int = 36) -> np.ndarray:
    vals = np.array(finite(values), dtype=float)
    if vals.size == 0:
        return np.linspace(-1.0, 1.0, n)
    lo = float(np.nanpercentile(vals, 1))
    hi = float(np.nanpercentile(vals, 99))
    if not math.isfinite(lo) or not math.isfinite(hi) or math.isclose(lo, hi):
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if math.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    pad = 0.06 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n)


def plot_panel_a(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> None:
    harm = [float(r["score"]) for r in rows if int(r["harm"]) == 1]
    help_ = [float(r["score"]) for r in rows if int(r["help"]) == 1]
    bins = bin_edges(harm + help_)
    ax.hist(help_, bins=bins, density=True, alpha=0.46, color=COLORS["help"], label="Helpful")
    ax.hist(harm, bins=bins, density=True, alpha=0.46, color=COLORS["harm"], label="Harmful")
    ax.set_xlabel("Fixed C3 replay score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0, handlelength=1.5)
    ax.grid(axis="y", alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_caption(ax, "(a) Helpful vs harmful")


def plot_panel_b(ax: plt.Axes, sweep: Sequence[Dict[str, Any]]) -> None:
    if not sweep:
        ax.axis("off")
        return
    x = np.array([100.0 * float(r["fallback_changed_rate"]) for r in sweep])
    harm = np.array([100.0 * float(r["harm_caught"]) for r in sweep])
    help_ = np.array([100.0 * float(r["help_lost"]) for r in sweep])
    random = np.array([100.0 * float(r["random_expected"]) for r in sweep])
    order = np.argsort(x)
    ax.plot(x[order], harm[order], color=COLORS["harm"], linewidth=2.8, label="Harm caught")
    ax.plot(x[order], help_[order], color=COLORS["help"], linewidth=2.8, label="Help lost")
    ax.plot(x[order], random[order], color=COLORS["random"], linestyle=":", linewidth=2.2, label="Random")
    ax.set_xlabel("Fallback budget among changed samples (%)")
    ax.set_ylabel("Selected fraction (%)")
    ax.set_xlim(0, min(100, max(40, float(np.nanmax(x)) + 4)))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="lower right")
    ax.grid(alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_caption(ax, "(b) Fallback budget curve")


def plot_panel_c(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> None:
    bins = harm_enrichment_bins(rows)
    if not bins:
        ax.axis("off")
        return
    x = np.arange(len(bins))
    vals = [100.0 * float(r["harmful_fraction"]) for r in bins]
    overall = 100.0 * float(bins[0]["overall_harmful_fraction"])
    labels = [str(r["label"]) for r in bins]
    ax.plot(x, vals, color=COLORS["harm"], marker="o", linewidth=2.8, markersize=7.2, label="Harm fraction")
    ax.axhline(overall, color=COLORS["overall"], linestyle="--", linewidth=2.0, label=f"Overall ({overall:.1f}%)")
    label_offsets = [(7, 6), (0, -13), (0, 6), (0, -14), (-7, 6)]
    label_align = [("left", "bottom"), ("center", "top"), ("center", "bottom"), ("center", "top"), ("right", "bottom")]
    for i, (xx, val) in enumerate(zip(x, vals)):
        dx, dy = label_offsets[min(i, len(label_offsets) - 1)]
        ha, va = label_align[min(i, len(label_align) - 1)]
        if abs(val - overall) < 4.5:
            dy = -14
            va = "top"
        ax.annotate(
            f"{val:.1f}%",
            xy=(xx, val),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=VALUE_FONTSIZE,
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Fixed C3 score quantile")
    ax.set_ylabel("Harmful fraction (%)")
    ax.set_ylim(0, min(100.0, max(max(vals) + 8.0, overall + 8.0)))
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")
    ax.grid(axis="y", alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_caption(ax, "(c) Harm enrichment by score bin")


def add_panel_caption(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.5,
        -0.22,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=CAPTION_FONTSIZE,
    )


def make_figure(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 12.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.4,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": LEGEND_FONTSIZE,
        }
    )
    plot_rows = [r for r in rows if str(r["source"]) == "apply" and str(r["outcome"]) in {"harm", "help"}]
    sweep = threshold_sweep(plot_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), gridspec_kw={"wspace": 0.42})
    plot_panel_a(axes[0], plot_rows)
    plot_panel_b(axes[1], sweep)
    plot_panel_c(axes[2], plot_rows)
    save_fig(fig, out_path)


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    apply = [r for r in rows if str(r["source"]) == "apply"]
    selected = [r for r in apply if int(r["selected"]) == 1]
    return {
        "apply_changed_with_score": len(apply),
        "apply_harm": sum(int(r["harm"]) for r in apply),
        "apply_help": sum(int(r["help"]) for r in apply),
        "selected": len(selected),
        "selected_harm": sum(int(r["harm"]) for r in selected),
        "selected_help": sum(int(r["help"]) for r in selected),
        "by_direction": {
            direction: {
                "n": sum(1 for r in apply if str(r["direction"]) == direction),
                "harm": sum(int(r["harm"]) for r in apply if str(r["direction"]) == direction),
                "help": sum(int(r["help"]) for r in apply if str(r["direction"]) == direction),
                "selected": sum(1 for r in apply if str(r["direction"]) == direction and int(r["selected"]) == 1),
                "tau": sorted({r["tau"] for r in apply if str(r["direction"]) == direction and str(r.get("tau", "")) != ""}),
            }
            for direction in DIRECTIONS
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create fixed C3 RAPIC figure artifacts.")
    ap.add_argument("--fixed_json", required=True, help="fixed_c_median_ensemble.json")
    ap.add_argument("--target", default="vga_llava15")
    ap.add_argument("--dataset", default="mscoco", choices=["mscoco", "aokvqa", "gqa"])
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--derive_decision_kl", type=pcp.parse_bool, default=True)
    args = ap.parse_args()

    fixed_json = Path(args.fixed_json).resolve()
    bundle = read_json(fixed_json)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else fixed_json.parent / "fixed_c3_figures"
    item = next(
        (
            dict(row)
            for row in bundle.get("per_dataset", [])
            if str(row.get("target", "")) == str(args.target) and str(row.get("dataset", "")) == str(args.dataset)
        ),
        None,
    )
    if item is None:
        raise RuntimeError(f"No fixed-C row for target={args.target!r}, dataset={args.dataset!r}")

    discovery_rows = pcp.load_rows(str(Path(str(item["discovery_rows_csv"])).resolve()), derive_decision_kl=bool(args.derive_decision_kl))
    apply_rows = pcp.load_rows(str(Path(str(item["apply_rows_csv"])).resolve()), derive_decision_kl=bool(args.derive_decision_kl))
    score_rows = add_scores(discovery_rows, item=item, dataset=str(args.dataset), source="discovery")
    score_rows.extend(add_scores(apply_rows, item=item, dataset=str(args.dataset), source="apply"))

    prefix = f"{args.target}_{args.dataset}_fixed_c3"
    score_csv = out_dir / f"{prefix}_scores.csv"
    sweep_csv = out_dir / f"{prefix}_budget_curve.csv"
    enrichment_csv = out_dir / f"{prefix}_harm_enrichment_bins.csv"
    fig_path = out_dir / f"{prefix}_figure.png"
    summary_path = out_dir / f"{prefix}_summary.json"
    write_csv(score_csv, score_rows)
    plot_rows = [r for r in score_rows if str(r["source"]) == "apply" and str(r["outcome"]) in {"harm", "help"}]
    write_csv(sweep_csv, threshold_sweep(plot_rows))
    write_csv(enrichment_csv, harm_enrichment_bins(plot_rows))
    make_figure(score_rows, fig_path)
    write_json(
        summary_path,
        {
            "inputs": {
                "fixed_json": str(fixed_json),
                "target": str(args.target),
                "dataset": str(args.dataset),
            },
            "summary": summarize(score_rows),
            "outputs": {
                "score_csv": str(score_csv),
                "budget_curve_csv": str(sweep_csv),
                "harm_enrichment_bins_csv": str(enrichment_csv),
                "png": str(fig_path),
                "pdf": str(fig_path.with_suffix(".pdf")),
            },
        },
    )
    print("[saved]", fig_path)
    print("[saved]", fig_path.with_suffix(".pdf"))
    print("[saved]", score_csv)
    print("[saved]", sweep_csv)
    print("[saved]", enrichment_csv)
    print("[saved]", summary_path)


if __name__ == "__main__":
    main()
