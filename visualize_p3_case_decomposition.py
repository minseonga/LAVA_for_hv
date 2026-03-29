#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_title(g: str) -> str:
    mp = {
        "G1_both_correct": "both_correct",
        "G2_gain": "gain",
        "G3_harm": "harm",
        "G4_both_wrong": "both_wrong",
    }
    return mp.get(g, g)


def qtype_bucket(q: str) -> str:
    q = str(q or "")
    if q in {"yesno", "what/which", "who", "where", "other"}:
        return q
    return "other"


def box_values(rows: List[Dict[str, Any]], key: str) -> List[float]:
    xs: List[float] = []
    for r in rows:
        v = safe_float(r.get(key))
        if v is not None:
            xs.append(float(v))
    return xs


def median_or_none(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    if n % 2 == 1:
        return float(ys[n // 2])
    return float((ys[n // 2 - 1] + ys[n // 2]) / 2.0)


def write_stats_csv(path: str, cohort_rows: Dict[str, List[Dict[str, Any]]], features: List[str]) -> None:
    out: List[Dict[str, Any]] = []
    for name, rows in cohort_rows.items():
        for feat in features:
            xs = box_values(rows, feat)
            out.append(
                {
                    "cohort": name,
                    "feature": feat,
                    "n": len(xs),
                    "mean": (None if not xs else sum(xs) / len(xs)),
                    "median": median_or_none(xs),
                    "min": (None if not xs else min(xs)),
                    "max": (None if not xs else max(xs)),
                }
            )
    if not out:
        return
    keys = list(out[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case_rows_csv",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/artrap_fragility_1000/final_switch_table_vpmi/p3_case_decomposition/p3_case_rows.csv",
    )
    ap.add_argument(
        "--eval_csv",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/artrap_fragility_1000/final_switch_table_vpmi/vpmi_rules_p4p5/p3_max_vpmi_all_switched_eval.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/artrap_fragility_1000/final_switch_table_vpmi/p3_case_decomposition/viz_multiview",
    )
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    case_rows = read_csv(args.case_rows_csv)
    eval_rows = read_csv(args.eval_csv)
    eval_map = {str(r.get("id", "")): r for r in eval_rows}

    rows: List[Dict[str, Any]] = []
    for r in case_rows:
        qid = str(r.get("id", ""))
        e = eval_map.get(qid, {})
        rr = dict(r)
        rr["switched"] = as_bool(r.get("switched", False))
        rr["base_ok"] = as_bool(r.get("base_ok", False))
        rr["p3_final_ok"] = as_bool(r.get("p3_final_ok", False))
        rr["has_correct_nonchamp"] = as_bool(r.get("has_correct_nonchamp", False))

        # attach numeric fields from eval table for richer plotting
        rr["champ_vpmi"] = safe_float(e.get("champ_vpmi"))
        rr["safe_vpmi_eval"] = safe_float(e.get("safe_vpmi"))
        rr["delta_vpmi_safe_minus_full"] = safe_float(e.get("delta_vpmi_safe_minus_full"))
        rr["m_full"] = safe_float(e.get("m_full"))
        rr["m_prior"] = safe_float(e.get("m_prior"))
        rr["champ_p_q"] = safe_float(e.get("champ_p_q"))
        rr["safe_p_q"] = safe_float(e.get("safe_p_q"))
        rr["qtype"] = str(e.get("qtype", rr.get("qtype", "")))

        rows.append(rr)

    # cohorts
    cohorts: Dict[str, List[Dict[str, Any]]] = {
        "both_correct": [r for r in rows if r.get("group") == "G1_both_correct"],
        "gain": [r for r in rows if r.get("group") == "G2_gain"],
        "harm": [r for r in rows if r.get("group") == "G3_harm"],
        "both_wrong": [r for r in rows if r.get("group") == "G4_both_wrong"],
        "selector_fail": [
            r
            for r in rows
            if r.get("failure_subtype") == "has_correct_candidate_but_failed"
            and bool(r.get("switched"))
            and (not bool(r.get("p3_final_ok")))
        ],
        "trigger_miss": [
            r
            for r in rows
            if r.get("failure_subtype") == "has_correct_candidate_but_failed"
            and (not bool(r.get("switched")))
            and r.get("group") == "G4_both_wrong"
        ],
    }

    # 1) Group count + qtype composition
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
    ax = axes[0]
    grp_order = ["both_correct", "gain", "harm", "both_wrong"]
    counts = [len(cohorts[g]) for g in grp_order]
    colors = ["#2e7d32", "#0277bd", "#c62828", "#6d4c41"]
    ax.bar(grp_order, counts, color=colors)
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_title("P3 Outcome Group Counts")
    ax.set_ylabel("n")

    ax = axes[1]
    q_order = ["yesno", "what/which", "who", "where", "other"]
    bottoms = [0.0 for _ in grp_order]
    q_colors = {
        "yesno": "#1e88e5",
        "what/which": "#43a047",
        "who": "#fb8c00",
        "where": "#8e24aa",
        "other": "#546e7a",
    }
    for q in q_order:
        vals = []
        for g in grp_order:
            c = Counter(qtype_bucket(r.get("qtype", "")) for r in cohorts[g])
            denom = max(1, len(cohorts[g]))
            vals.append(c.get(q, 0) / denom)
        ax.bar(grp_order, vals, bottom=bottoms, color=q_colors[q], label=q)
        bottoms = [bottoms[i] + vals[i] for i in range(len(vals))]
    ax.set_ylim(0, 1.0)
    ax.set_title("Q-type Composition by Group (ratio)")
    ax.set_ylabel("ratio")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "01_group_counts_qtype_mix.svg"), bbox_inches="tight")
    plt.close(fig)

    # 2) 4-group feature boxplots
    feature_specs = [
        ("champ_vpmi", "champ_vpmi"),
        ("delta_vpmi_safe_minus_full", "safe_vpmi - champ_vpmi"),
        ("m_prior", "m_prior"),
        ("m_full", "m_full"),
        ("champ_p_q", "champ_p_q"),
        ("safe_p_q", "safe_p_q"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    axes = axes.flatten()
    for i, (feat, title) in enumerate(feature_specs):
        ax = axes[i]
        data = [box_values(cohorts[g], feat) for g in grp_order]
        ax.boxplot(data, tick_labels=grp_order, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Feature Distributions Across 4 Core Outcome Groups", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out_dir, "02_feature_boxplots_4groups.svg"), bbox_inches="tight")
    plt.close(fig)

    # 3) gain vs both_correct vs selector_fail vs trigger_miss
    comp_order = ["gain", "both_correct", "selector_fail", "trigger_miss"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # (a) delta_safe_minus_bestcorr_vpmi
    ax = axes[0, 0]
    data = [box_values(cohorts[g], "delta_safe_minus_bestcorr_vpmi") for g in comp_order]
    ax.boxplot(data, tick_labels=comp_order, showfliers=False)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("safe - bestCorrect (VPMI)")
    ax.tick_params(axis="x", rotation=20)

    # (b) delta_safe_minus_bestcorr_s_q
    ax = axes[0, 1]
    data = [box_values(cohorts[g], "delta_safe_minus_bestcorr_s_q") for g in comp_order]
    ax.boxplot(data, tick_labels=comp_order, showfliers=False)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("safe - bestCorrect (S_q)")
    ax.tick_params(axis="x", rotation=20)

    # (c) scatter champ_vpmi vs delta_vpmi_safe_minus_full
    ax = axes[1, 0]
    color_map = {
        "gain": "#0277bd",
        "both_correct": "#2e7d32",
        "selector_fail": "#c62828",
        "trigger_miss": "#6d4c41",
    }
    for g in comp_order:
        xs = box_values(cohorts[g], "champ_vpmi")
        ys = box_values(cohorts[g], "delta_vpmi_safe_minus_full")
        # aligned filtering
        pts = []
        for r in cohorts[g]:
            xv = safe_float(r.get("champ_vpmi"))
            yv = safe_float(r.get("delta_vpmi_safe_minus_full"))
            if xv is None or yv is None:
                continue
            pts.append((xv, yv))
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, alpha=0.45, label=f"{g} (n={len(pts)})", color=color_map[g])
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("champ_vpmi")
    ax.set_ylabel("safe_vpmi - champ_vpmi")
    ax.set_title("Trigger Plane by Cohort")
    ax.legend(fontsize=8)

    # (d) switched ratio in each cohort
    ax = axes[1, 1]
    ratios = []
    for g in comp_order:
        denom = max(1, len(cohorts[g]))
        ratios.append(sum(1 for r in cohorts[g] if bool(r.get("switched"))) / denom)
    ax.bar(comp_order, ratios, color=[color_map[g] for g in comp_order])
    for i, v in enumerate(ratios):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Switched Ratio by Cohort")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Gain/Both-correct vs Failure Subtypes: Multi-view", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(args.out_dir, "03_gain_vs_bothcorrect_multiview.svg"), bbox_inches="tight")
    plt.close(fig)

    # 4) switched-only outcomes (gain/both_correct/harm/both_wrong)
    switched = [r for r in rows if bool(r.get("switched"))]
    # derive switched outcomes explicitly
    sw_cohorts = {
        "gain": [r for r in switched if (not bool(r.get("base_ok"))) and bool(r.get("p3_final_ok"))],
        "both_correct": [r for r in switched if bool(r.get("base_ok")) and bool(r.get("p3_final_ok"))],
        "harm": [r for r in switched if bool(r.get("base_ok")) and (not bool(r.get("p3_final_ok")))],
        "both_wrong": [r for r in switched if (not bool(r.get("base_ok"))) and (not bool(r.get("p3_final_ok")))],
    }
    sw_order = ["gain", "both_correct", "harm", "both_wrong"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    for ax, feat, title in [
        (axes[0], "delta_vpmi_safe_minus_full", "safe_vpmi - champ_vpmi"),
        (axes[1], "m_full", "m_full"),
        (axes[2], "m_prior", "m_prior"),
    ]:
        data = [box_values(sw_cohorts[g], feat) for g in sw_order]
        ax.boxplot(data, tick_labels=sw_order, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Switched-only Outcome Comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(args.out_dir, "04_switched_only_outcome_boxplots.svg"), bbox_inches="tight")
    plt.close(fig)

    # save compact stats csv
    stat_features = [
        "champ_vpmi",
        "safe_vpmi_eval",
        "delta_vpmi_safe_minus_full",
        "m_full",
        "m_prior",
        "champ_p_q",
        "safe_p_q",
        "delta_safe_minus_bestcorr_vpmi",
        "delta_safe_minus_bestcorr_s_full",
        "delta_safe_minus_bestcorr_s_q",
        "delta_safe_minus_bestcorr_s_core",
    ]
    write_stats_csv(
        path=os.path.join(args.out_dir, "feature_stats_by_cohort.csv"),
        cohort_rows={
            "both_correct": cohorts["both_correct"],
            "gain": cohorts["gain"],
            "harm": cohorts["harm"],
            "both_wrong": cohorts["both_wrong"],
            "selector_fail": cohorts["selector_fail"],
            "trigger_miss": cohorts["trigger_miss"],
        },
        features=stat_features,
    )

    # qtype table csv
    qrows: List[Dict[str, Any]] = []
    for name, subset in {
        "both_correct": cohorts["both_correct"],
        "gain": cohorts["gain"],
        "harm": cohorts["harm"],
        "both_wrong": cohorts["both_wrong"],
        "selector_fail": cohorts["selector_fail"],
        "trigger_miss": cohorts["trigger_miss"],
    }.items():
        c = Counter(qtype_bucket(r.get("qtype", "")) for r in subset)
        denom = max(1, len(subset))
        for q in ["yesno", "what/which", "who", "where", "other"]:
            qrows.append(
                {
                    "cohort": name,
                    "qtype": q,
                    "n": c.get(q, 0),
                    "ratio": c.get(q, 0) / denom,
                }
            )
    with open(os.path.join(args.out_dir, "qtype_mix_by_cohort.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["cohort", "qtype", "n", "ratio"])
        wr.writeheader()
        wr.writerows(qrows)

    print("[saved]", os.path.join(args.out_dir, "01_group_counts_qtype_mix.svg"))
    print("[saved]", os.path.join(args.out_dir, "02_feature_boxplots_4groups.svg"))
    print("[saved]", os.path.join(args.out_dir, "03_gain_vs_bothcorrect_multiview.svg"))
    print("[saved]", os.path.join(args.out_dir, "04_switched_only_outcome_boxplots.svg"))
    print("[saved]", os.path.join(args.out_dir, "feature_stats_by_cohort.csv"))
    print("[saved]", os.path.join(args.out_dir, "qtype_mix_by_cohort.csv"))


if __name__ == "__main__":
    main()
