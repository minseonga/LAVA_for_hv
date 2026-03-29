#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rankdata(a: np.ndarray) -> np.ndarray:
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    arr = a[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum() - 1
    counts = np.bincount(dense)
    starts = np.r_[0, counts.cumsum()[:-1]]
    avg = starts + (counts - 1) / 2.0
    ranks = avg[dense] + 1.0
    return ranks[inv]


def auc_from_scores(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(s)
    sum_r_pos = float(np.sum(r[y == 1]))
    auc = (sum_r_pos - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(auc)


def ks_stat(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.sort(np.unique(s))
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    cdf_pos = np.searchsorted(pos_sorted, vals, side="right") / float(len(pos_sorted))
    cdf_neg = np.searchsorted(neg_sorted, vals, side="right") / float(len(neg_sorted))
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_scores_by_task(
    feat_csv: Path,
    label_csv: Path,
    id_col: str,
    label_col: str,
    pos_label: str,
    neg_label: str,
    task_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat = pd.read_csv(feat_csv)
    feat["id"] = feat["id"].astype(str)

    lab = pd.read_csv(label_csv)
    lab[id_col] = lab[id_col].astype(str)
    lab = lab[[id_col, label_col]].rename(columns={id_col: "id", label_col: "label"})
    lab = lab[lab["label"].isin([pos_label, neg_label])].copy()

    df = feat.merge(lab, on="id", how="inner")
    if df.empty:
        raise RuntimeError(f"No merged rows for task={task_name}")

    all_rows: List[Dict] = []
    eval_rows: List[Dict] = []

    for li, g in df.groupby("block_layer_idx"):
        y = (g["label"].values == pos_label).astype(int)

        Cz = sigmoid(zscore(g["C"].to_numpy(dtype=float)))
        Az = sigmoid(zscore(g["A"].to_numpy(dtype=float)))
        Dz = sigmoid(zscore(g["D"].to_numpy(dtype=float)))

        A_supportive = Az * Cz
        A_misleading = Az * Dz
        C_plus_A_supportive = Cz + A_supportive
        D_plus_A_misleading = Dz + A_misleading

        score_map = {
            "C_alone": Cz,
            "D_alone": Dz,
            "A_alone": Az,
            "A_supportive": A_supportive,
            "A_misleading": A_misleading,
            "C_plus_A_supportive": C_plus_A_supportive,
            "D_plus_A_misleading": D_plus_A_misleading,
        }

        ids = g["id"].tolist()
        labels = g["label"].tolist()
        for i in range(len(g)):
            row = {
                "task": task_name,
                "id": ids[i],
                "label": labels[i],
                "is_pos": int(y[i]),
                "block_layer_idx": int(li),
            }
            for m, arr in score_map.items():
                row[m] = float(arr[i])
            all_rows.append(row)

        for m, s in score_map.items():
            auc_hi = auc_from_scores(y, s)
            auc_best = max(auc_hi, 1.0 - auc_hi) if math.isfinite(auc_hi) else float("nan")
            direction = "higher_in_pos" if auc_hi >= 0.5 else "lower_in_pos"
            ks = ks_stat(y, s)
            eval_rows.append(
                {
                    "task": task_name,
                    "block_layer_idx": int(li),
                    "metric": m,
                    "n": int(len(g)),
                    "n_pos": int(np.sum(y == 1)),
                    "n_neg": int(np.sum(y == 0)),
                    "auc_pos_high": float(auc_hi),
                    "auc_best_dir": float(auc_best),
                    "ks": float(ks),
                    "direction": direction,
                }
            )

    return pd.DataFrame(all_rows), pd.DataFrame(eval_rows)


def bootstrap_auc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        sb = s[idx]
        if np.unique(yb).size < 2:
            continue
        vals.append(auc_from_scores(yb, sb))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def bootstrap_auc_diff_ci(
    y: np.ndarray, s1: np.ndarray, s2: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        d = auc_from_scores(yb, s1[idx]) - auc_from_scores(yb, s2[idx])
        vals.append(d)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def plot_distributions_vf(vf_df: pd.DataFrame, layer: int, out_dir: Path) -> None:
    metrics = [
        "A_alone",
        "A_supportive",
        "A_misleading",
        "C_alone",
        "C_plus_A_supportive",
    ]
    d = vf_df[vf_df["block_layer_idx"] == int(layer)].copy()
    if d.empty:
        return

    for m in metrics:
        plt.figure(figsize=(5.2, 3.6))
        for label, color in [("vf", "tab:red"), ("fv", "tab:blue")]:
            x = d.loc[d["label"] == label, m].to_numpy(dtype=float)
            if len(x) == 0:
                continue
            plt.hist(x, bins=16, alpha=0.45, density=True, color=color, label=f"{label} (n={len(x)})")
        plt.title(f"{m} @ L{layer} (VF vs FV)")
        plt.xlabel("score")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        out = out_dir / f"dist_vf_fv_{m}_L{layer}.png"
        plt.savefig(out, dpi=180)
        plt.close()


def plot_quadrant(vf_df: pd.DataFrame, layer: int, out_dir: Path) -> pd.DataFrame:
    d = vf_df[vf_df["block_layer_idx"] == int(layer)].copy()
    if d.empty:
        return pd.DataFrame()

    xcol = "A_alone"
    ycol = "C_plus_A_supportive"

    xthr = float(np.median(d[xcol].to_numpy(dtype=float)))
    ythr = float(np.median(d[ycol].to_numpy(dtype=float)))

    def quad(r):
        xh = r[xcol] >= xthr
        yh = r[ycol] >= ythr
        if xh and yh:
            return "highA_highCsup"
        if xh and not yh:
            return "highA_lowCsup"
        if (not xh) and yh:
            return "lowA_highCsup"
        return "lowA_lowCsup"

    d["quadrant"] = d.apply(quad, axis=1)
    q = (
        d.groupby(["label", "quadrant"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    total = d.groupby("label", as_index=False).size().rename(columns={"size": "n_label"})
    q = q.merge(total, on="label", how="left")
    q["ratio"] = q["count"] / q["n_label"]
    q["x_threshold"] = xthr
    q["y_threshold"] = ythr

    q_csv = out_dir / f"quadrant_vf_fv_L{layer}.csv"
    q.to_csv(q_csv, index=False)

    order = ["highA_highCsup", "highA_lowCsup", "lowA_highCsup", "lowA_lowCsup"]
    pivot = q.pivot(index="quadrant", columns="label", values="ratio").reindex(order)
    pivot = pivot.fillna(0.0)

    plt.figure(figsize=(5.8, 3.8))
    x = np.arange(len(order))
    w = 0.38
    plt.bar(x - w / 2, pivot.get("vf", pd.Series(np.zeros(len(order)), index=order)).values, width=w, label="VF", color="tab:red")
    plt.bar(x + w / 2, pivot.get("fv", pd.Series(np.zeros(len(order)), index=order)).values, width=w, label="FV", color="tab:blue")
    plt.xticks(x, order, rotation=20)
    plt.ylabel("ratio in label")
    plt.title(f"Quadrant split @ L{layer}\n(A_alone vs C_plus_A_supportive)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"quadrant_vf_fv_L{layer}.png", dpi=180)
    plt.close()

    return q


def plot_layer_shift(eval_fp: pd.DataFrame, eval_vf: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["A_alone", "A_misleading", "C_plus_A_supportive", "D_plus_A_misleading"]
    plt.figure(figsize=(8.2, 4.6))
    for m, ls, task, df in [
        ("A_alone", "-", "fp_hall_vs_tp_yes", eval_fp),
        ("A_alone", "--", "vf_vs_fv", eval_vf),
        ("C_plus_A_supportive", "-", "fp_hall_vs_tp_yes", eval_fp),
        ("C_plus_A_supportive", "--", "vf_vs_fv", eval_vf),
    ]:
        d = df[df["metric"] == m].sort_values("block_layer_idx")
        if d.empty:
            continue
        plt.plot(d["block_layer_idx"], d["auc_best_dir"], ls, label=f"{task}:{m}")
    plt.xlabel("layer")
    plt.ylabel("AUC(best direction)")
    plt.title("Layer shift of separability")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "layer_shift_auc.png", dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep-dive analysis for conditional A scores.")
    ap.add_argument("--feature_csv", type=str, required=True)
    ap.add_argument("--label_fp_csv", type=str, required=True)
    ap.add_argument("--label_vf_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_scores, fp_eval = build_scores_by_task(
        feat_csv=Path(args.feature_csv),
        label_csv=Path(args.label_fp_csv),
        id_col="id",
        label_col="group",
        pos_label="fp_hall",
        neg_label="tp_yes",
        task_name="fp_hall_vs_tp_yes",
    )

    vf_scores, vf_eval = build_scores_by_task(
        feat_csv=Path(args.feature_csv),
        label_csv=Path(args.label_vf_csv),
        id_col="id",
        label_col="group",
        pos_label="vf",
        neg_label="fv",
        task_name="vf_vs_fv",
    )

    fp_scores.to_csv(out_dir / "scores_fp_hall_vs_tp_yes.csv", index=False)
    vf_scores.to_csv(out_dir / "scores_vf_vs_fv.csv", index=False)
    fp_eval.sort_values(["auc_best_dir", "ks"], ascending=[False, False]).to_csv(out_dir / "eval_fp_hall_vs_tp_yes.csv", index=False)
    vf_eval.sort_values(["auc_best_dir", "ks"], ascending=[False, False]).to_csv(out_dir / "eval_vf_vs_fv.csv", index=False)

    # Group means/distributions for VF/FV.
    metrics = ["A_alone", "A_supportive", "A_misleading", "C_alone", "C_plus_A_supportive"]
    means = (
        vf_scores.groupby(["block_layer_idx", "label"], as_index=False)[metrics]
        .mean()
        .sort_values(["block_layer_idx", "label"])
    )
    means.to_csv(out_dir / "vf_fv_group_means_by_layer.csv", index=False)

    # best layer for interaction metric
    best_comp = vf_eval[vf_eval["metric"] == "C_plus_A_supportive"].sort_values("auc_best_dir", ascending=False).head(1)
    layer_comp = int(best_comp.iloc[0]["block_layer_idx"]) if len(best_comp) else 28

    plot_distributions_vf(vf_scores, layer_comp, out_dir)
    quad_df = plot_quadrant(vf_scores, layer_comp, out_dir)
    plot_layer_shift(fp_eval, vf_eval, out_dir)

    # Bootstrap CI for key comparisons.
    boot_rows: List[Dict] = []

    def add_boot(task_scores: pd.DataFrame, pos_label: str, metric: str, layer: int, tag: str):
        d = task_scores[task_scores["block_layer_idx"] == int(layer)]
        y = (d["label"].values == pos_label).astype(int)
        s = d[metric].to_numpy(dtype=float)
        auc = auc_from_scores(y, s)
        m, lo, hi = bootstrap_auc_ci(y, s, n_boot=int(args.bootstrap), seed=int(args.seed))
        boot_rows.append(
            {
                "task": tag,
                "metric": metric,
                "layer": int(layer),
                "auc": float(auc),
                "auc_boot_mean": float(m),
                "auc_ci95_lo": float(lo),
                "auc_ci95_hi": float(hi),
                "n": int(len(d)),
                "n_pos": int(np.sum(y == 1)),
                "n_neg": int(np.sum(y == 0)),
            }
        )

    # fp/tp
    l_fp_A = int(fp_eval[fp_eval.metric == "A_alone"].sort_values("auc_best_dir", ascending=False).iloc[0]["block_layer_idx"])
    l_fp_Am = int(fp_eval[fp_eval.metric == "A_misleading"].sort_values("auc_best_dir", ascending=False).iloc[0]["block_layer_idx"])
    add_boot(fp_scores, "fp_hall", "A_alone", l_fp_A, "fp_hall_vs_tp_yes")
    add_boot(fp_scores, "fp_hall", "A_misleading", l_fp_Am, "fp_hall_vs_tp_yes")
    add_boot(fp_scores, "fp_hall", "C_plus_A_supportive", int(fp_eval[fp_eval.metric == "C_plus_A_supportive"].sort_values("auc_best_dir", ascending=False).iloc[0]["block_layer_idx"]), "fp_hall_vs_tp_yes")

    # vf/fv
    l_vf_A = int(vf_eval[vf_eval.metric == "A_alone"].sort_values("auc_best_dir", ascending=False).iloc[0]["block_layer_idx"])
    l_vf_Cp = int(vf_eval[vf_eval.metric == "C_plus_A_supportive"].sort_values("auc_best_dir", ascending=False).iloc[0]["block_layer_idx"])
    add_boot(vf_scores, "vf", "A_alone", l_vf_A, "vf_vs_fv")
    add_boot(vf_scores, "vf", "C_plus_A_supportive", l_vf_Cp, "vf_vs_fv")

    # diff CI: same layer (C+A_supportive vs A_alone) at l_vf_Cp
    d = vf_scores[vf_scores["block_layer_idx"] == int(l_vf_Cp)]
    y = (d["label"].values == "vf").astype(int)
    s1 = d["C_plus_A_supportive"].to_numpy(dtype=float)
    s2 = d["A_alone"].to_numpy(dtype=float)
    dm, dlo, dhi = bootstrap_auc_diff_ci(y, s1, s2, n_boot=int(args.bootstrap), seed=int(args.seed))
    boot_rows.append(
        {
            "task": "vf_vs_fv",
            "metric": "AUC_diff_(C_plus_A_supportive - A_alone)_same_layer",
            "layer": int(l_vf_Cp),
            "auc": float(auc_from_scores(y, s1) - auc_from_scores(y, s2)),
            "auc_boot_mean": float(dm),
            "auc_ci95_lo": float(dlo),
            "auc_ci95_hi": float(dhi),
            "n": int(len(d)),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
        }
    )

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(out_dir / "bootstrap_ci_key_metrics.csv", index=False)

    # crisp checks requested
    checks = {}
    d_layer = vf_scores[vf_scores["block_layer_idx"] == int(layer_comp)]
    for metric in ["A_alone", "C_plus_A_supportive"]:
        g = d_layer.groupby("label", as_index=False)[metric].mean()
        checks[f"L{layer_comp}_{metric}_means"] = {row["label"]: float(row[metric]) for _, row in g.iterrows()}

    summary = {
        "inputs": {
            "feature_csv": str(Path(args.feature_csv).resolve()),
            "label_fp_csv": str(Path(args.label_fp_csv).resolve()),
            "label_vf_csv": str(Path(args.label_vf_csv).resolve()),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "counts": {
            "fp_rows": int(len(fp_scores)),
            "vf_rows": int(len(vf_scores)),
            "fp_ids": int(fp_scores["id"].nunique()),
            "vf_ids": int(vf_scores["id"].nunique()),
        },
        "best_layers": {
            "fp_A_alone": int(l_fp_A),
            "fp_A_misleading": int(l_fp_Am),
            "vf_A_alone": int(l_vf_A),
            "vf_C_plus_A_supportive": int(l_vf_Cp),
            "vf_interaction_layer": int(layer_comp),
        },
        "checks": checks,
        "outputs": {
            "eval_fp_csv": str((out_dir / "eval_fp_hall_vs_tp_yes.csv").resolve()),
            "eval_vf_csv": str((out_dir / "eval_vf_vs_fv.csv").resolve()),
            "group_means_vf_csv": str((out_dir / "vf_fv_group_means_by_layer.csv").resolve()),
            "quadrant_csv": str((out_dir / f"quadrant_vf_fv_L{layer_comp}.csv").resolve()),
            "bootstrap_csv": str((out_dir / "bootstrap_ci_key_metrics.csv").resolve()),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_dir / "scores_fp_hall_vs_tp_yes.csv")
    print("[saved]", out_dir / "scores_vf_vs_fv.csv")
    print("[saved]", out_dir / "eval_fp_hall_vs_tp_yes.csv")
    print("[saved]", out_dir / "eval_vf_vs_fv.csv")
    print("[saved]", out_dir / "vf_fv_group_means_by_layer.csv")
    print("[saved]", out_dir / "bootstrap_ci_key_metrics.csv")
    print("[saved]", out_dir / "summary.json")


if __name__ == "__main__":
    main()
