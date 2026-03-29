#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def map_group(g: str) -> str:
    m = {
        "both_correct": "VV",
        "vcs_wrong_vga_correct": "FV",
        "vcs_correct_vga_wrong": "VF",
        "both_wrong": "FF",
    }
    return m.get(str(g), str(g))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    sa = len(a)
    sb = len(b)
    sp = np.sqrt(((sa - 1) * va + (sb - 1) * vb) / max(1, (sa + sb - 2)))
    if sp == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)


def auc_rank(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)


def confusion_from_compare(df: pd.DataFrame) -> Dict[str, int]:
    y = df["answer"].astype(str).str.lower().to_numpy()
    p = df["base_pred"].astype(str).str.lower().to_numpy()
    q = df["new_pred"].astype(str).str.lower().to_numpy()

    def conf(pred: np.ndarray) -> Dict[str, int]:
        return {
            "TP": int(((pred == "yes") & (y == "yes")).sum()),
            "FP": int(((pred == "yes") & (y == "no")).sum()),
            "TN": int(((pred == "no") & (y == "no")).sum()),
            "FN": int(((pred == "no") & (y == "yes")).sum()),
        }

    c0 = conf(p)
    c1 = conf(q)
    out = {
        "base_TP": c0["TP"],
        "base_FP": c0["FP"],
        "base_TN": c0["TN"],
        "base_FN": c0["FN"],
        "new_TP": c1["TP"],
        "new_FP": c1["FP"],
        "new_TN": c1["TN"],
        "new_FN": c1["FN"],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build proof-oriented visualization pack for A/D/B1/B2/B3/P hypotheses")
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--family_root", type=str, required=True, help=".../screen_eval_structure_v2")
    ap.add_argument("--per_case_csv", type=str, required=True)
    ap.add_argument("--d1d2_dir", type=str, required=True)
    ap.add_argument("--frrs_compare_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # -------------------------------
    # Load inputs
    # -------------------------------
    feat = pd.read_csv(args.features_csv)
    feat["group4"] = feat["group"].map(map_group)

    comps = ["fp_vs_tp_yes", "fv_vs_vf", "incorrect_vs_correct"]
    comp_rows = []
    single_rows = []
    for c in comps:
        p_comp = os.path.join(args.family_root, c, "family_composite_metrics.csv")
        p_single = os.path.join(args.family_root, c, "family_single_feature_metrics.csv")
        dc = pd.read_csv(p_comp)
        ds = pd.read_csv(p_single)
        dc["comparison"] = c
        ds["comparison"] = c
        comp_rows.append(dc)
        single_rows.append(ds)
    comp_df = pd.concat(comp_rows, ignore_index=True)
    single_df = pd.concat(single_rows, ignore_index=True)

    per_case = pd.read_csv(args.per_case_csv)

    d1d2_count_path = os.path.join(args.d1d2_dir, "d1_d2_counts_within_yes2no.csv")
    d1d2_sep_path = os.path.join(args.d1d2_dir, "d1_d2_feature_separation.csv")
    d1d2_counts = pd.read_csv(d1d2_count_path) if os.path.exists(d1d2_count_path) else pd.DataFrame()
    d1d2_sep = pd.read_csv(d1d2_sep_path) if os.path.exists(d1d2_sep_path) else pd.DataFrame()

    frrs_cmp = pd.read_csv(args.frrs_compare_csv)

    # -------------------------------
    # Fig1: Family composite heatmap (AUC)
    # -------------------------------
    piv = comp_df.pivot_table(index="comparison", columns="set_name", values="auc_best_dir", aggfunc="mean")
    piv = piv.reindex(index=comps)
    piv = piv[[c for c in ["A", "B", "C", "D", "E"] if c in piv.columns]]

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlGnBu", vmin=0.5, vmax=max(0.8, float(np.nanmax(piv.values))))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title("Family Composite AUC (best direction)")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig1_family_composite_auc_heatmap.png"), dpi=180)
    plt.close(fig)

    # -------------------------------
    # Fig2: A vs C with E overlay (group4)
    # -------------------------------
    x = feat["obj_token_prob_lse"].to_numpy()
    y = feat["faithful_minus_global_attn"].to_numpy()
    s = feat["guidance_mismatch_score"].to_numpy()
    g = feat["group4"].astype(str).to_numpy()
    colors = {"VV": "#1f77b4", "FV": "#2ca02c", "VF": "#d62728", "FF": "#9467bd"}

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for kk in ["VV", "FV", "VF", "FF"]:
        m = g == kk
        if m.sum() == 0:
            continue
        size = 20 + 180 * (s[m] - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-8)
        ax.scatter(x[m], y[m], s=size, c=colors[kk], alpha=0.35, label=f"{kk} (n={m.sum()})", edgecolors="none")
        ax.scatter(np.nanmean(x[m]), np.nanmean(y[m]), c=colors[kk], s=120, marker="X", edgecolors="black", linewidths=0.6)
    ax.set_xlabel("A: obj_token_prob_lse (availability)")
    ax.set_ylabel("C: faithful_minus_global_attn (usability)")
    ax.set_title("A×C Matrix with E overlay (marker size ~ mismatch)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig2_AxC_scatter_E_overlay.png"), dpi=180)
    plt.close(fig)

    # -------------------------------
    # Fig3: D1/D2 yes->no transition
    # -------------------------------
    y2n = per_case[(per_case["pred_baseline"] == "yes") & (per_case["pred_vga"] == "no")]
    n_d1 = int((y2n["gt"] == "no").sum())
    n_d2 = int((y2n["gt"] == "yes").sum())

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.bar(["D1\n(beneficial)", "D2\n(harmful)"], [n_d1, n_d2], color=["#2ca02c", "#d62728"], alpha=0.85)
    total = max(1, n_d1 + n_d2)
    ax.text(0, n_d1 + total * 0.01, f"{n_d1} ({n_d1/total:.1%})", ha="center", fontsize=9)
    ax.text(1, n_d2 + total * 0.01, f"{n_d2} ({n_d2/total:.1%})", ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("VGA yes→no transitions: D1 vs D2")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig3_d1_d2_yes2no_counts.png"), dpi=180)
    plt.close(fig)

    # -------------------------------
    # Fig4: D1 vs D2 key feature delta bar (if available)
    # -------------------------------
    if not d1d2_sep.empty:
        pick = d1d2_sep.copy()
        pick = pick[pick["feature"].isin(["obj_token_prob_lse", "faithful_minus_global_attn", "guidance_mismatch_score", "Y_overassert_proxy", "faithful_on_nonG_mass", "harmful_inside_G"])]
        if len(pick) > 0:
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ax.barh(pick["feature"], pick["delta_D2_minus_D1"], color=["#ff7f0e" if v > 0 else "#1f77b4" for v in pick["delta_D2_minus_D1"]])
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel("D2 - D1 mean")
            ax.set_title("D1 vs D2 key feature deltas")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "fig4_d1d2_key_feature_deltas.png"), dpi=180)
            plt.close(fig)

    # -------------------------------
    # Fig5: FRRS supportive confusion delta
    # -------------------------------
    conf = confusion_from_compare(frrs_cmp)
    labels = ["TP", "FP", "TN", "FN"]
    base_vals = [conf["base_TP"], conf["base_FP"], conf["base_TN"], conf["base_FN"]]
    new_vals = [conf["new_TP"], conf["new_FP"], conf["new_TN"], conf["new_FN"]]

    xloc = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(xloc - w / 2, base_vals, width=w, label="Baseline", color="#7f7f7f", alpha=0.85)
    ax.bar(xloc + w / 2, new_vals, width=w, label="FRRS-supportive", color="#2ca02c", alpha=0.85)
    ax.set_xticks(xloc)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("P2 check: TP 유지 + FP 감소 여부")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig5_frrs_supportive_confusion_delta.png"), dpi=180)
    plt.close(fig)

    # -------------------------------
    # Fig6: Effect-size summary (A/C/E) for tasks
    # -------------------------------
    rows = []

    # Task1: fp_hall vs tp_yes
    t1 = feat[(feat["target_is_fp_hallucination"] == 1) | (feat["target_is_tp_yes"] == 1)].copy()
    y1 = (t1["target_is_fp_hallucination"] == 1).to_numpy()

    # Task2: FV vs VF (mapped groups)
    t2 = feat[feat["group4"].isin(["FV", "VF"])].copy()
    y2 = (t2["group4"] == "VF").to_numpy()

    feats = {
        "A_obj_token_prob_lse": "obj_token_prob_lse",
        "C_faithful_minus_global_attn": "faithful_minus_global_attn",
        "E_guidance_mismatch_score": "guidance_mismatch_score",
    }

    for nm, col in feats.items():
        s1 = t1[col].to_numpy(dtype=float)
        s2 = t2[col].to_numpy(dtype=float)

        d1 = cohens_d(s1[y1], s1[~y1])
        d2 = cohens_d(s2[y2], s2[~y2])
        a1 = auc_rank(y1.astype(int), s1)
        a2 = auc_rank(y2.astype(int), s2)

        rows.append({"task": "fp_hall_vs_tp_yes", "metric": nm, "cohens_d": d1, "auc": max(a1, 1 - a1)})
        rows.append({"task": "fv_vs_vf", "metric": nm, "cohens_d": d2, "auc": max(a2, 1 - a2)})

    edf = pd.DataFrame(rows)
    edf.to_csv(os.path.join(args.out_dir, "fig6_effect_sizes.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, task in enumerate(["fp_hall_vs_tp_yes", "fv_vs_vf"]):
        part = edf[edf["task"] == task]
        xpos = np.arange(len(part)) + i * (len(part) + 1)
        ax.bar(xpos, part["auc"], label=task)
        for x_, y_, m_ in zip(xpos, part["auc"], part["metric"]):
            ax.text(x_, y_ + 0.005, m_.split("_", 1)[0], ha="center", fontsize=8, rotation=0)
    ax.set_ylim(0.48, max(0.85, float(edf["auc"].max()) + 0.03))
    ax.set_ylabel("AUC (best direction)")
    ax.set_title("P1: A vs C vs E separation strength by task")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig6_A_C_E_taskwise_auc.png"), dpi=180)
    plt.close(fig)

    # Save summary
    summary = {
        "inputs": {
            "features_csv": os.path.abspath(args.features_csv),
            "family_root": os.path.abspath(args.family_root),
            "per_case_csv": os.path.abspath(args.per_case_csv),
            "d1d2_dir": os.path.abspath(args.d1d2_dir),
            "frrs_compare_csv": os.path.abspath(args.frrs_compare_csv),
        },
        "counts": {
            "features_n": int(len(feat)),
            "taxonomy_n": int(len(per_case)),
            "yes2no_n": int(len(y2n)),
            "d1_n": int(n_d1),
            "d2_n": int(n_d2),
        },
        "key_numbers": {
            "family_auc_table": piv.reset_index().to_dict(orient="records"),
            "frrs_confusion": conf,
        },
        "outputs": {
            "fig1": os.path.join(args.out_dir, "fig1_family_composite_auc_heatmap.png"),
            "fig2": os.path.join(args.out_dir, "fig2_AxC_scatter_E_overlay.png"),
            "fig3": os.path.join(args.out_dir, "fig3_d1_d2_yes2no_counts.png"),
            "fig4": os.path.join(args.out_dir, "fig4_d1d2_key_feature_deltas.png"),
            "fig5": os.path.join(args.out_dir, "fig5_frrs_supportive_confusion_delta.png"),
            "fig6": os.path.join(args.out_dir, "fig6_A_C_E_taskwise_auc.png"),
            "fig6_effect_sizes_csv": os.path.join(args.out_dir, "fig6_effect_sizes.csv"),
        },
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", os.path.join(args.out_dir, "summary.json"))
    for k, v in summary["outputs"].items():
        print("[saved]", v)


if __name__ == "__main__":
    main()
