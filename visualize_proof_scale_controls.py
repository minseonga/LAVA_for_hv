#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_row(rows: List[Dict[str, Any]], **conds: str) -> Optional[Dict[str, Any]]:
    for r in rows:
        ok = True
        for k, v in conds.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize proof-scale-control outputs into one dashboard figure.")
    ap.add_argument("--proof_dir", type=str, required=True)
    ap.add_argument("--out_png", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    proof_dir = os.path.abspath(args.proof_dir)
    out_png = (
        os.path.abspath(args.out_png)
        if args.out_png
        else os.path.join(proof_dir, "proof_dashboard.png")
    )
    out_csv = (
        os.path.abspath(args.out_csv)
        if args.out_csv
        else os.path.join(proof_dir, "proof_dashboard_key_metrics.csv")
    )

    ensure_dir(os.path.dirname(out_png))
    ensure_dir(os.path.dirname(out_csv))

    p1 = read_csv(os.path.join(proof_dir, "proof1_scale_invariance_correlations.csv"))
    p2 = read_csv(os.path.join(proof_dir, "proof2_token_position_curves.csv"))
    p3 = read_csv(os.path.join(proof_dir, "proof3_negative_controls_and_features.csv"))

    # Panel A: correlations for scale invariance.
    corr_labels = [
        "abs vs q_len",
        "rank vs q_len",
        "abs vs rarity",
        "rank vs rarity",
    ]
    corr_vals = []
    corr_rows = [
        pick_row(p1, covariate="q_len", feature="vpmi_abs"),
        pick_row(p1, covariate="q_len", feature="vpmi_rank_pct"),
        pick_row(p1, covariate="ans_rarity", feature="vpmi_abs"),
        pick_row(p1, covariate="ans_rarity", feature="vpmi_rank_pct"),
    ]
    for r in corr_rows:
        corr_vals.append(0.0 if r is None else float(safe_float(r.get("pearson_r")) or 0.0))

    # Panel B: KS comparison among key features/controls.
    feature_order = [
        "d_rank_vpmi_pct",
        "d_vpmi_suffix_min_k",
        "c_vpmi_prefix_mean_k",
        "d_vpmi_mean",
        "z_vpmi_cand_global",
        "d_z_vpmi_global",
    ]
    ks_labels = []
    ks_vals = []
    for f in feature_order:
        r = pick_row(p3, feature=f)
        if r is None:
            continue
        ks = safe_float(r.get("ks"))
        if ks is None:
            continue
        ks_labels.append(f)
        ks_vals.append(float(ks))

    # Panel C/D: token-position curves.
    t = []
    cand_b = []
    cand_w = []
    delta_b = []
    delta_w = []
    for r in p2:
        tt = safe_float(r.get("token_pos_1based"))
        if tt is None:
            continue
        t.append(int(tt))
        cand_b.append(float(safe_float(r.get("cand_better_mean")) or float("nan")))
        cand_w.append(float(safe_float(r.get("cand_worse_mean")) or float("nan")))
        delta_b.append(float(safe_float(r.get("delta_better_mean")) or float("nan")))
        delta_w.append(float(safe_float(r.get("delta_worse_mean")) or float("nan")))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A: Correlations
    ax = axes[0, 0]
    x = np.arange(len(corr_labels))
    colors = ["#d62728", "#1f77b4", "#d62728", "#1f77b4"]
    ax.bar(x, corr_vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, corr_labels, rotation=15, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("B1 Scale Invariance: Absolute vs Rank")

    # B: KS
    ax = axes[0, 1]
    x = np.arange(len(ks_labels))
    bar_colors = []
    for k in ks_labels:
        if k == "d_rank_vpmi_pct":
            bar_colors.append("#1f77b4")
        elif k in {"d_vpmi_suffix_min_k", "c_vpmi_prefix_mean_k"}:
            bar_colors.append("#2ca02c")
        elif k in {"d_vpmi_mean", "z_vpmi_cand_global", "d_z_vpmi_global"}:
            bar_colors.append("#ff7f0e")
        else:
            bar_colors.append("#7f7f7f")
    ax.bar(x, ks_vals, color=bar_colors)
    ax.set_xticks(x, ks_labels, rotation=25, ha="right")
    ax.set_ylabel("KS distance")
    ax.set_title("Feature vs Negative Controls (J1/J2)")

    # C: candidate tokenwise curve
    ax = axes[1, 0]
    if len(t) > 0:
        ax.plot(t, cand_b, marker="o", color="#1f77b4", label="better: cand")
        ax.plot(t, cand_w, marker="o", color="#d62728", label="worse: cand")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Core token position t")
    ax.set_ylabel("Mean VPMI(t)")
    ax.set_title("B2 Tokenwise Candidate Curve")
    ax.legend(loc="best")

    # D: delta tokenwise curve
    ax = axes[1, 1]
    if len(t) > 0:
        ax.plot(t, delta_b, marker="o", color="#1f77b4", label="better: cand-champ")
        ax.plot(t, delta_w, marker="o", color="#d62728", label="worse: cand-champ")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Core token position t")
    ax.set_ylabel("Mean ΔVPMI(t)")
    ax.set_title("B2 Tokenwise Delta Curve")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Save key metrics table for paper text.
    key_rows: List[Dict[str, Any]] = []
    for lbl, val in zip(corr_labels, corr_vals):
        key_rows.append({"group": "proof1_corr", "name": lbl, "value": val})
    for lbl, val in zip(ks_labels, ks_vals):
        key_rows.append({"group": "proof3_ks", "name": lbl, "value": val})
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["group", "name", "value"])
        wr.writeheader()
        for r in key_rows:
            wr.writerow(r)

    print("[saved]", out_png)
    print("[saved]", out_csv)


if __name__ == "__main__":
    main()

