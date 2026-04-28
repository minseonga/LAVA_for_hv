#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_pcp_c_d_controller as pcp
import build_posthoc_b_c_fusion_controller as base


DEFAULT_C_FEATURES = [
    "cheap_target_gap_content_min",
    "cheap_lp_content_min",
    "cheap_lp_content_std",
]
DEFAULT_D_FEATURES = [
    "cheap_decision_candidate_minus_alt",
    "cheap_decision_candidate_prob_binary",
    "cheap_decision_candidate_label_lp",
    "cheap_decision_candidate_kl_uniform",
]

GROUP_ORDER = ["harm", "help", "neutral"]
GROUP_COLORS = {
    "harm": "#c43c39",
    "help": "#2b7bba",
    "neutral": "#7f7f7f",
}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_features(value: str, default: Sequence[str]) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_float(value: object) -> Optional[float]:
    x = base.maybe_float(value)
    if x is None or not math.isfinite(float(x)):
        return None
    return float(x)


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.abspath(path), exist_ok=True)


def read_policy(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def outcome_group(row: Dict[str, Any]) -> str:
    harm = int(base.maybe_int(row.get("harm")) or 0)
    help_ = int(base.maybe_int(row.get("help")) or 0)
    if harm:
        return "harm"
    if help_:
        return "help"
    return "neutral"


def filtered_rows(rows: Sequence[Dict[str, Any]], candidate_filter: str, include_neutral: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not pcp.is_route_candidate(row, candidate_filter):
            continue
        group = outcome_group(row)
        if group == "neutral" and not include_neutral:
            continue
        rec = dict(row)
        rec["_group"] = group
        out.append(rec)
    return out


def numeric_values(rows: Iterable[Dict[str, Any]], feature: str, group: Optional[str] = None) -> List[float]:
    vals: List[float] = []
    for row in rows:
        if group is not None and str(row.get("_group")) != group:
            continue
        x = safe_float(row.get(feature))
        if x is not None:
            vals.append(x)
    return vals


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def summarize_feature(rows_by_run: Dict[str, List[Dict[str, Any]]], features: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for run_name, rows in rows_by_run.items():
        for feature in features:
            for group in GROUP_ORDER:
                vals = numeric_values(rows, feature, group=group)
                if not vals:
                    out.append(
                        {
                            "run": run_name,
                            "feature": feature,
                            "group": group,
                            "n": 0,
                            "mean": "",
                            "std": "",
                            "min": "",
                            "q10": "",
                            "q25": "",
                            "median": "",
                            "q75": "",
                            "q90": "",
                            "max": "",
                        }
                    )
                    continue
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
                out.append(
                    {
                        "run": run_name,
                        "feature": feature,
                        "group": group,
                        "n": len(vals),
                        "mean": mean,
                        "std": math.sqrt(var),
                        "min": min(vals),
                        "q10": quantile(vals, 0.10),
                        "q25": quantile(vals, 0.25),
                        "median": quantile(vals, 0.50),
                        "q75": quantile(vals, 0.75),
                        "q90": quantile(vals, 0.90),
                        "max": max(vals),
                    }
                )
    return out


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(rows[0].keys())
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def load_feature_specs(policy: Dict[str, Any], family: str, c_names: Sequence[str], d_names: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    c_specs = list(policy.get("selected_c_features") or [])
    d_specs = list(policy.get("selected_d_features") or [])
    if c_specs or d_specs:
        return c_specs, d_specs

    # Fallback for plotting raw feature distributions without a policy file.
    c_specs = [{"feature": name, "direction": "high", "mu": 0.0, "sd": 1.0} for name in c_names]
    d_specs = [{"feature": name, "direction": "high", "mu": 0.0, "sd": 1.0} for name in d_names]
    if family == "c_only":
        return c_specs, []
    if family == "d_only":
        return [], d_specs
    return c_specs, d_specs


def add_scores(rows: List[Dict[str, Any]], policy: Dict[str, Any], c_names: Sequence[str], d_names: Sequence[str]) -> List[str]:
    c_specs, d_specs = load_feature_specs(policy, "cd_fusion", c_names, d_names)
    score_names: List[str] = []
    if c_specs:
        score_names.append("pcp_c_score")
    if d_specs:
        score_names.append("pcp_d_score")
    if c_specs and d_specs:
        score_names.append("pcp_cd_score_alpha_0_5")
    for row in rows:
        c_score = pcp.mean_z_score(row, c_specs)
        d_score = pcp.mean_z_score(row, d_specs)
        if c_score is not None:
            row["pcp_c_score"] = c_score
        if d_score is not None:
            row["pcp_d_score"] = d_score
        if c_score is not None and d_score is not None:
            row["pcp_cd_score_alpha_0_5"] = 0.5 * float(c_score) + 0.5 * float(d_score)
    return score_names


def count_groups(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {g: 0 for g in GROUP_ORDER}
    for row in rows:
        g = str(row.get("_group"))
        if g in counts:
            counts[g] += 1
    return counts


def plot_feature(feature: str, rows_by_run: Dict[str, List[Dict[str, Any]]], out_dir: str, bins: int) -> str:
    run_names = list(rows_by_run.keys())
    fig, axes = plt.subplots(1, len(run_names), figsize=(5.8 * len(run_names), 4.2), squeeze=False)
    for ax, run_name in zip(axes[0], run_names):
        rows = rows_by_run[run_name]
        all_vals = numeric_values(rows, feature)
        if not all_vals:
            ax.set_title(f"{run_name}: {feature}\n(no values)")
            ax.axis("off")
            continue
        lo = quantile(all_vals, 0.01)
        hi = quantile(all_vals, 0.99)
        if lo is None or hi is None or lo == hi:
            lo = min(all_vals)
            hi = max(all_vals)
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        for group in GROUP_ORDER:
            vals = [x for x in numeric_values(rows, feature, group=group) if float(lo) <= x <= float(hi)]
            if not vals:
                continue
            ax.hist(
                vals,
                bins=bins,
                range=(float(lo), float(hi)),
                density=True,
                alpha=0.35,
                color=GROUP_COLORS[group],
                label=f"{group} (n={len(vals)})",
            )
        counts = count_groups(rows)
        ax.set_title(
            f"{run_name}\n"
            f"harm={counts['harm']} help={counts['help']} neutral={counts['neutral']}"
        )
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"PCP feature distribution: {feature}", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"dist_{feature}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_scatter(rows_by_run: Dict[str, List[Dict[str, Any]]], out_dir: str) -> Optional[str]:
    if not all(any("pcp_c_score" in r and "pcp_d_score" in r for r in rows) for rows in rows_by_run.values()):
        return None
    run_names = list(rows_by_run.keys())
    fig, axes = plt.subplots(1, len(run_names), figsize=(5.4 * len(run_names), 4.8), squeeze=False)
    for ax, run_name in zip(axes[0], run_names):
        rows = rows_by_run[run_name]
        for group in reversed(GROUP_ORDER):
            xs: List[float] = []
            ys: List[float] = []
            for row in rows:
                if str(row.get("_group")) != group:
                    continue
                x = safe_float(row.get("pcp_c_score"))
                y = safe_float(row.get("pcp_d_score"))
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            if xs:
                ax.scatter(xs, ys, s=15, alpha=0.55, color=GROUP_COLORS[group], label=f"{group} (n={len(xs)})")
        ax.axhline(0.0, color="#333333", lw=0.8, alpha=0.4)
        ax.axvline(0.0, color="#333333", lw=0.8, alpha=0.4)
        ax.set_title(run_name)
        ax.set_xlabel("C score")
        ax.set_ylabel("D score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("PCP C/D score scatter", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "scatter_c_score_vs_d_score.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot PCP feature distributions for LLaVA-1.5 and LLaVA-Next discovery rows.")
    ap.add_argument("--llava15_rows", type=str, required=True)
    ap.add_argument("--llava_next_rows", type=str, required=True)
    ap.add_argument("--llava15_policy_json", type=str, default="")
    ap.add_argument("--llava_next_policy_json", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--candidate_filter", type=str, default="changed_answer", choices=["all", "changed_answer", "yes_to_no"])
    ap.add_argument("--include_neutral", type=parse_bool, default=False)
    ap.add_argument("--c_feature_cols", type=str, default=",".join(DEFAULT_C_FEATURES))
    ap.add_argument("--d_feature_cols", type=str, default=",".join(DEFAULT_D_FEATURES))
    ap.add_argument("--extra_feature_cols", type=str, default="")
    ap.add_argument("--bins", type=int, default=36)
    ap.add_argument("--derive_decision_kl", type=parse_bool, default=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    c_features = parse_features(args.c_feature_cols, DEFAULT_C_FEATURES)
    d_features = parse_features(args.d_feature_cols, DEFAULT_D_FEATURES)
    extra_features = parse_features(args.extra_feature_cols, [])

    raw_by_run = {
        "llava15": pcp.load_rows(args.llava15_rows, derive_decision_kl=bool(args.derive_decision_kl)),
        "llava_next": pcp.load_rows(args.llava_next_rows, derive_decision_kl=bool(args.derive_decision_kl)),
    }
    rows_by_run = {
        name: filtered_rows(rows, str(args.candidate_filter), bool(args.include_neutral))
        for name, rows in raw_by_run.items()
    }
    policy_by_run = {
        "llava15": read_policy(args.llava15_policy_json) if args.llava15_policy_json else {},
        "llava_next": read_policy(args.llava_next_policy_json) if args.llava_next_policy_json else {},
    }

    score_features: List[str] = []
    for name, rows in rows_by_run.items():
        score_features.extend(add_scores(rows, policy_by_run.get(name, {}), c_features, d_features))
    score_features = sorted(set(score_features))

    features = []
    for feature in list(c_features) + list(d_features) + list(extra_features) + score_features:
        if feature not in features:
            features.append(feature)

    summary_rows = summarize_feature(rows_by_run, features)
    summary_csv = os.path.join(args.out_dir, "feature_distribution_summary.csv")
    write_csv(summary_csv, summary_rows)

    figure_paths = []
    for feature in features:
        figure_paths.append(plot_feature(feature, rows_by_run, args.out_dir, int(args.bins)))
    scatter_path = plot_score_scatter(rows_by_run, args.out_dir)
    if scatter_path:
        figure_paths.append(scatter_path)

    manifest = {
        "inputs": {
            "llava15_rows": os.path.abspath(args.llava15_rows),
            "llava_next_rows": os.path.abspath(args.llava_next_rows),
            "llava15_policy_json": os.path.abspath(args.llava15_policy_json) if args.llava15_policy_json else "",
            "llava_next_policy_json": os.path.abspath(args.llava_next_policy_json) if args.llava_next_policy_json else "",
            "candidate_filter": str(args.candidate_filter),
            "include_neutral": bool(args.include_neutral),
            "c_feature_cols": c_features,
            "d_feature_cols": d_features,
            "extra_feature_cols": extra_features,
        },
        "counts": {
            name: count_groups(rows)
            for name, rows in rows_by_run.items()
        },
        "outputs": {
            "summary_csv": os.path.abspath(summary_csv),
            "figures": [os.path.abspath(p) for p in figure_paths],
        },
    }
    manifest_path = os.path.join(args.out_dir, "summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[saved]", os.path.abspath(summary_csv))
    for path in figure_paths:
        print("[saved]", os.path.abspath(path))
    print("[saved]", os.path.abspath(manifest_path))


if __name__ == "__main__":
    main()
