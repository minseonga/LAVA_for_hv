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
    "yes_to_no": "#2563EB",
    "no_to_yes": "#D97706",
}
DATASET_PRETTY = {"mscoco": "MSCOCO", "aokvqa": "AOKVQA", "gqa": "GQA"}


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
    ax.hist(help_, bins=bins, density=True, alpha=0.46, color=COLORS["help"], label=f"Helpful gains (n={len(help_)})")
    ax.hist(harm, bins=bins, density=True, alpha=0.46, color=COLORS["harm"], label=f"Harmful flips (n={len(harm)})")
    ax.set_xlabel("Fixed C3 replay score")
    ax.set_ylabel("Density")
    ax.set_title("(a) Helpful vs harmful")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel_b(ax: plt.Axes, sweep: Sequence[Dict[str, Any]]) -> None:
    if not sweep:
        ax.axis("off")
        return
    x = np.array([100.0 * float(r["fallback_changed_rate"]) for r in sweep])
    harm = np.array([100.0 * float(r["harm_caught"]) for r in sweep])
    help_ = np.array([100.0 * float(r["help_lost"]) for r in sweep])
    random = np.array([100.0 * float(r["random_expected"]) for r in sweep])
    order = np.argsort(x)
    ax.plot(x[order], harm[order], color=COLORS["harm"], linewidth=2.1, label="Harm caught")
    ax.plot(x[order], help_[order], color=COLORS["help"], linewidth=2.1, label="Help lost")
    ax.plot(x[order], random[order], color=COLORS["random"], linestyle=":", linewidth=1.6, label="Random")
    ax.set_xlabel("Fallback budget among changed samples (%)")
    ax.set_ylabel("Selected fraction (%)")
    ax.set_title("(b) Fallback budget curve")
    ax.set_xlim(0, min(100, max(40, float(np.nanmax(x)) + 4)))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel_c(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> None:
    all_scores = [float(r["score"]) for r in rows]
    bins = bin_edges(all_scores)
    for direction in DIRECTIONS:
        vals = [float(r["score"]) for r in rows if str(r["direction"]) == direction]
        if not vals:
            continue
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=COLORS[direction],
            label=f"{direction_label(direction)} (n={len(vals)})",
        )
        taus = sorted({float(r["tau"]) for r in rows if str(r["direction"]) == direction and str(r.get("tau", "")) != ""})
        for tau in taus[:1]:
            ax.axvline(tau, color=COLORS[direction], linestyle="--", linewidth=1.5)
            ax.text(tau, ax.get_ylim()[1] * 0.92, f"tau={tau:.2f}", color=COLORS[direction], rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("Fixed C3 replay score")
    ax.set_ylabel("Density")
    ax.set_title("(c) Transition split")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(rows: Sequence[Dict[str, Any]], out_path: Path, title: str) -> None:
    plot_rows = [r for r in rows if str(r["source"]) == "apply" and str(r["outcome"]) in {"harm", "help"}]
    sweep = threshold_sweep(plot_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9), gridspec_kw={"wspace": 0.34})
    plot_panel_a(axes[0], plot_rows)
    plot_panel_b(axes[1], sweep)
    plot_panel_c(axes[2], plot_rows)
    fig.suptitle(title, y=1.05, fontsize=13)
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
    fig_path = out_dir / f"{prefix}_figure.png"
    summary_path = out_dir / f"{prefix}_summary.json"
    write_csv(score_csv, score_rows)
    write_csv(sweep_csv, threshold_sweep([r for r in score_rows if str(r["source"]) == "apply" and str(r["outcome"]) in {"harm", "help"}]))
    title = f"Fixed C3 RAPIC diagnostics: {args.target} / {DATASET_PRETTY.get(str(args.dataset), str(args.dataset))}"
    make_figure(score_rows, fig_path, title)
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
                "png": str(fig_path),
                "pdf": str(fig_path.with_suffix(".pdf")),
            },
        },
    )
    print("[saved]", fig_path)
    print("[saved]", fig_path.with_suffix(".pdf"))
    print("[saved]", score_csv)
    print("[saved]", sweep_csv)
    print("[saved]", summary_path)


if __name__ == "__main__":
    main()
