#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def map_group(g: str) -> str:
    g = str(g)
    mp = {
        "both_correct": "VV",
        "vcs_wrong_vga_correct": "FV",
        "vcs_correct_vga_wrong": "VF",
        "vga_only_correct": "FV",
        "baseline_only_correct": "VF",
        "vga_improvement": "FV",
        "vga_regression": "VF",
        "both_wrong": "FF",
    }
    return mp.get(g, g)


def assign_quadrant(x: float, y: float, x_thr: float, y_thr: float) -> str:
    hx = x >= x_thr
    hy = y >= y_thr
    if hx and hy:
        return "highA_highY"
    if hx and not hy:
        return "highA_lowY"
    if (not hx) and hy:
        return "lowA_highY"
    return "lowA_lowY"


def compute_scores(df: pd.DataFrame, a_feature: str, c_feature: str) -> pd.DataFrame:
    out = df.copy()
    A_raw = out[a_feature].to_numpy(dtype=float)
    C_raw = out[c_feature].to_numpy(dtype=float)
    out["A_alone"] = sigmoid(zscore(A_raw))
    out["C_alone"] = sigmoid(zscore(C_raw))
    out["A_supportive"] = out["A_alone"] * out["C_alone"]
    out["C_plus_A_supportive"] = out["C_alone"] + out["A_supportive"]
    return out


def compute_group_means(df: pd.DataFrame, y_cols: List[str]) -> pd.DataFrame:
    cols = ["A_alone"] + y_cols + ["A_supportive", "C_plus_A_supportive"]
    cols = [c for c in cols if c in df.columns]
    means = df.groupby("matrix_group", as_index=False)[cols].mean().sort_values("matrix_group")
    return means


def compute_quadrant_table(df: pd.DataFrame, y_col: str, x_thr: float, y_thr: float) -> pd.DataFrame:
    qcol = f"quadrant_{y_col}"
    tmp = df.copy()
    tmp[qcol] = [assign_quadrant(x, y, x_thr, y_thr) for x, y in zip(tmp["A_alone"], tmp[y_col])]
    q = (
        tmp.groupby(["matrix_group", qcol], as_index=False)
        .size()
        .rename(columns={"size": "count", qcol: "quadrant"})
    )
    n = tmp.groupby("matrix_group", as_index=False).size().rename(columns={"size": "n_group"})
    q = q.merge(n, on="matrix_group", how="left")
    q["ratio"] = q["count"] / q["n_group"]
    q["y_col"] = y_col
    return q


def compute_marker_sizes(x: np.ndarray, smin: float = 20.0, smax: float = 140.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax - xmin < 1e-12:
        return np.full_like(x, (smin + smax) * 0.5)
    y = (x - xmin) / (xmax - xmin)
    return smin + (smax - smin) * y


def sample_balanced(df: pd.DataFrame, groups: List[str], seed: int) -> Tuple[pd.DataFrame, int]:
    counts = df[df["matrix_group"].isin(groups)]["matrix_group"].value_counts().to_dict()
    if not counts:
        raise RuntimeError("No groups available for balanced sampling")
    min_n = min(int(counts.get(g, 0)) for g in groups)
    if min_n <= 0:
        raise RuntimeError(f"Cannot balanced-sample; group counts: {counts}")
    parts = []
    for gi, g in enumerate(groups):
        d = df[df["matrix_group"] == g]
        parts.append(d.sample(n=min_n, random_state=seed + gi))
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out, min_n


def main() -> None:
    ap = argparse.ArgumentParser(description="Cue Availability × Cue Usability Matrix plotter")
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--subset_group_csv", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--group_col", type=str, default="group")
    ap.add_argument("--a_feature", type=str, default="G_top1_mass")
    ap.add_argument("--c_feature", type=str, default="faithful_minus_global_attn")
    ap.add_argument("--size_feature", type=str, default="")
    ap.add_argument("--plot_mode", type=str, default="pure", choices=["pure", "both"])
    ap.add_argument("--balanced_repeats", type=int, default=0)
    ap.add_argument("--balanced_groups", type=str, default="VV,FV,VF,FF")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(args.features_csv)
    grp = pd.read_csv(args.subset_group_csv)

    feat[args.id_col] = feat[args.id_col].astype(str)
    grp[args.id_col] = grp[args.id_col].astype(str)

    grp = grp[[args.id_col, args.group_col]].copy()
    grp["matrix_group"] = grp[args.group_col].map(map_group)

    df = feat.merge(grp[[args.id_col, "matrix_group"]], on=args.id_col, how="inner")
    if df.empty:
        raise RuntimeError("No merged rows for matrix plot")

    if args.a_feature not in df.columns:
        raise RuntimeError(f"A feature not found: {args.a_feature}")
    if args.c_feature not in df.columns:
        raise RuntimeError(f"C feature not found: {args.c_feature}")

    base_cols = [args.id_col, "matrix_group", args.a_feature, args.c_feature]
    df_out = compute_scores(df[base_cols].copy(), args.a_feature, args.c_feature)
    df_out = df_out[[args.id_col, "matrix_group", "A_alone", "C_alone", "A_supportive", "C_plus_A_supportive"]]

    for col in [
        "harmful_minus_faithful",
        "guidance_mismatch_score",
        "supportive_outside_G",
        "harmful_inside_G",
        "G_overfocus",
        args.size_feature,
    ]:
        if col and col in df.columns:
            df_out[col] = df[col].to_numpy(dtype=float)

    x_thr = float(np.median(df_out["A_alone"].to_numpy(dtype=float)))
    y1_thr = float(np.median(df_out["C_alone"].to_numpy(dtype=float)))
    y2_thr = float(np.median(df_out["C_plus_A_supportive"].to_numpy(dtype=float)))

    df_out["quadrant_C_alone"] = [assign_quadrant(x, y, x_thr, y1_thr) for x, y in zip(df_out["A_alone"], df_out["C_alone"])]
    df_out["quadrant_C_plus_A_supportive"] = [assign_quadrant(x, y, x_thr, y2_thr) for x, y in zip(df_out["A_alone"], df_out["C_plus_A_supportive"])]

    means = compute_group_means(df_out, ["C_alone"])
    q1 = compute_quadrant_table(df_out, "C_alone", x_thr, y1_thr)
    q2 = compute_quadrant_table(df_out, "C_plus_A_supportive", x_thr, y2_thr)

    df_out_csv = out_dir / "matrix_scores_per_id.csv"
    means_csv = out_dir / "matrix_group_means.csv"
    q1_csv = out_dir / "matrix_quadrant_counts_C_alone.csv"
    q2_csv = out_dir / "matrix_quadrant_counts_C_plus_A_supportive.csv"
    df_out.to_csv(df_out_csv, index=False)
    means.to_csv(means_csv, index=False)
    q1.to_csv(q1_csv, index=False)
    q2.to_csv(q2_csv, index=False)

    balanced_means_csv = out_dir / "matrix_group_means_balanced_repeats.csv"
    balanced_quad_csv = out_dir / "matrix_quadrant_balanced_repeats.csv"
    balanced_sample_csv = out_dir / "matrix_scores_balanced_sample.csv"
    balanced_summary: Dict[str, object] = {}
    if int(args.balanced_repeats) > 0:
        groups = [s.strip() for s in str(args.balanced_groups).split(",") if s.strip()]
        rep_means = []
        rep_quads = []
        first_bal = None
        min_n = 0
        for r in range(int(args.balanced_repeats)):
            dbr, min_n = sample_balanced(df_out, groups, seed=int(args.seed) + r * 1009)
            if first_bal is None:
                first_bal = dbr.copy()
            m = compute_group_means(dbr, ["C_alone"])
            m["repeat"] = r
            rep_means.append(m)
            q = compute_quadrant_table(dbr, "C_alone", x_thr, y1_thr)
            q["repeat"] = r
            rep_quads.append(q)
        rep_means_df = pd.concat(rep_means, axis=0, ignore_index=True)
        rep_quads_df = pd.concat(rep_quads, axis=0, ignore_index=True)

        means_bal = (
            rep_means_df.groupby("matrix_group", as_index=False)[["A_alone", "C_alone", "A_supportive", "C_plus_A_supportive"]]
            .agg(["mean", "std"])
        )
        means_bal.columns = [
            "matrix_group",
            "A_alone_mean",
            "A_alone_std",
            "C_alone_mean",
            "C_alone_std",
            "A_supportive_mean",
            "A_supportive_std",
            "C_plus_A_supportive_mean",
            "C_plus_A_supportive_std",
        ]
        means_bal.to_csv(balanced_means_csv, index=False)

        quads_bal = (
            rep_quads_df.groupby(["matrix_group", "quadrant"], as_index=False)["ratio"]
            .agg(ratio_mean="mean", ratio_std="std")
        )
        quads_bal.to_csv(balanced_quad_csv, index=False)

        if first_bal is not None:
            first_bal.to_csv(balanced_sample_csv, index=False)
        balanced_summary = {
            "repeats": int(args.balanced_repeats),
            "groups": groups,
            "min_group_size": int(min_n),
            "balanced_means_csv": str(balanced_means_csv.resolve()),
            "balanced_quadrants_csv": str(balanced_quad_csv.resolve()),
            "balanced_first_sample_csv": str(balanced_sample_csv.resolve()) if first_bal is not None else "",
        }

    colors: Dict[str, str] = {
        "VV": "#2ca02c",
        "FV": "#1f77b4",
        "VF": "#ff7f0e",
        "FF": "#d62728",
    }

    def scatter_plot(y_col: str, y_thr: float, title_suffix: str, out_png: Path) -> None:
        plt.figure(figsize=(6.8, 5.2))
        for g in ["VV", "FV", "VF", "FF"]:
            d = df_out[df_out["matrix_group"] == g]
            if d.empty:
                continue
            x = d["A_alone"].to_numpy(dtype=float)
            y = d[y_col].to_numpy(dtype=float)
            xj = x + rng.normal(0, 0.003, size=len(x))
            yj = y + rng.normal(0, 0.003, size=len(y))
            if str(args.size_feature).strip() and args.size_feature in d.columns:
                s = compute_marker_sizes(d[args.size_feature].to_numpy(dtype=float), smin=20, smax=140)
            else:
                s = 24
            plt.scatter(xj, yj, s=s, alpha=0.62, c=colors.get(g, "gray"), label=f"{g} (n={len(d)})", edgecolors="none")

        plt.axvline(x_thr, color="black", linestyle="--", linewidth=1)
        plt.axhline(y_thr, color="black", linestyle="--", linewidth=1)
        plt.xlabel(f"A_alone from `{args.a_feature}` (Cue Availability)")
        plt.ylabel(f"{y_col} from `{args.c_feature}` (Cue Usability)")
        plt.title(f"Cue Availability × Cue Usability Matrix\n{title_suffix}")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    png1 = out_dir / "matrix_scatter_A_vs_C_alone.png"
    png2 = out_dir / "matrix_scatter_A_vs_C_plus_A_supportive.png"
    scatter_plot("C_alone", y1_thr, "y = C_alone", png1)
    if args.plot_mode == "both":
        scatter_plot("C_plus_A_supportive", y2_thr, "y = C + A_supportive", png2)

    summary = {
        "inputs": {
            "features_csv": str(Path(args.features_csv).resolve()),
            "subset_group_csv": str(Path(args.subset_group_csv).resolve()),
            "a_feature": args.a_feature,
            "c_feature": args.c_feature,
            "size_feature": args.size_feature,
            "plot_mode": args.plot_mode,
            "balanced_repeats": int(args.balanced_repeats),
            "balanced_groups": str(args.balanced_groups),
        },
        "counts": {
            "n_rows": int(len(df_out)),
            "group_counts": {k: int(v) for k, v in df_out["matrix_group"].value_counts().to_dict().items()},
        },
        "thresholds": {
            "A_alone_median": x_thr,
            "C_alone_median": y1_thr,
            "C_plus_A_supportive_median": y2_thr,
        },
        "outputs": {
            "scores_per_id_csv": str(df_out_csv.resolve()),
            "group_means_csv": str(means_csv.resolve()),
            "quadrant_C_alone_csv": str(q1_csv.resolve()),
            "quadrant_C_plus_A_supportive_csv": str(q2_csv.resolve()),
            "scatter_A_vs_C_alone_png": str(png1.resolve()),
            "scatter_A_vs_C_plus_A_supportive_png": str(png2.resolve()) if args.plot_mode == "both" else "",
        },
        "balanced_summary": balanced_summary,
    }
    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", df_out_csv)
    print("[saved]", means_csv)
    print("[saved]", q1_csv)
    print("[saved]", q2_csv)
    print("[saved]", png1)
    if args.plot_mode == "both":
        print("[saved]", png2)
    if balanced_summary:
        print("[saved]", balanced_means_csv)
        print("[saved]", balanced_quad_csv)
        print("[saved]", balanced_sample_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
