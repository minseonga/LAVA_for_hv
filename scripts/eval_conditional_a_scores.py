#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def auc_from_scores(y: np.ndarray, s: np.ndarray) -> float:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(s.astype(float))
    sum_r_pos = float(np.sum(r[y == 1]))
    auc = (sum_r_pos - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(auc)


def ks_stat(y: np.ndarray, s: np.ndarray) -> float:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate conditional A scores by layer.")
    ap.add_argument("--feature_csv", type=str, required=True, help="layerwise_rf_proxy_features.csv")
    ap.add_argument("--label_csv", type=str, required=True, help="CSV with id + label column")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--label_col", type=str, default="group")
    ap.add_argument("--pos_label", type=str, required=True)
    ap.add_argument("--neg_label", type=str, required=True)
    ap.add_argument("--task_name", type=str, default="task")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(args.feature_csv)
    feat["id"] = feat["id"].astype(str)

    lab = pd.read_csv(args.label_csv)
    lab[args.id_col] = lab[args.id_col].astype(str)
    lab = lab[[args.id_col, args.label_col]].rename(columns={args.id_col: "id", args.label_col: "label"})
    lab = lab[lab["label"].isin([args.pos_label, args.neg_label])].copy()

    df = feat.merge(lab, on="id", how="inner")
    if df.empty:
        raise RuntimeError("No merged rows for selected labels")

    rows: List[Dict[str, Any]] = []
    metric_names = [
        "C_alone",
        "D_alone",
        "A_alone",
        "A_supportive",
        "A_misleading",
        "C_plus_A_supportive",
        "D_plus_A_misleading",
    ]

    for li, gdf in df.groupby("block_layer_idx"):
        y = (gdf["label"].values == args.pos_label).astype(int)
        C = gdf["C"].to_numpy(dtype=float)
        A = gdf["A"].to_numpy(dtype=float)
        D = gdf["D"].to_numpy(dtype=float)

        Cz = zscore(C)
        Az = zscore(A)
        Dz = zscore(D)

        Csig = sigmoid(Cz)
        Asig = sigmoid(Az)
        Dsig = sigmoid(Dz)

        scores = {
            "C_alone": Csig,
            "D_alone": Dsig,
            "A_alone": Asig,
            "A_supportive": Asig * Csig,
            "A_misleading": Asig * Dsig,
            "C_plus_A_supportive": Csig + (Asig * Csig),
            "D_plus_A_misleading": Dsig + (Asig * Dsig),
        }

        for m in metric_names:
            s = scores[m]
            auc_hi = auc_from_scores(y, s)
            auc_best = max(auc_hi, 1.0 - auc_hi) if math.isfinite(auc_hi) else float("nan")
            direction = "higher_in_pos" if auc_hi >= 0.5 else "lower_in_pos"
            ks = ks_stat(y, s)
            rows.append(
                {
                    "task": args.task_name,
                    "block_layer_idx": int(li),
                    "metric": m,
                    "n": int(len(gdf)),
                    "n_pos": int(np.sum(y == 1)),
                    "n_neg": int(np.sum(y == 0)),
                    "auc_pos_high": float(auc_hi),
                    "auc_best_dir": float(auc_best),
                    "ks": float(ks),
                    "direction": direction,
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["auc_best_dir", "ks"], ascending=[False, False])
    out_csv = out_dir / f"conditional_scores_{args.task_name}.csv"
    out_df.to_csv(out_csv, index=False)

    best_by_metric = (
        out_df.sort_values(["metric", "auc_best_dir", "ks"], ascending=[True, False, False])
        .groupby("metric", as_index=False)
        .first()
    )
    best_csv = out_dir / f"best_by_metric_{args.task_name}.csv"
    best_by_metric.to_csv(best_csv, index=False)

    summary = {
        "task": args.task_name,
        "inputs": {
            "feature_csv": str(Path(args.feature_csv).resolve()),
            "label_csv": str(Path(args.label_csv).resolve()),
            "id_col": args.id_col,
            "label_col": args.label_col,
            "pos_label": args.pos_label,
            "neg_label": args.neg_label,
        },
        "counts": {
            "n_rows_merged": int(len(df)),
            "n_ids": int(df["id"].nunique()),
            "n_layers": int(df["block_layer_idx"].nunique()),
            "n_pos_ids": int(df[df["label"] == args.pos_label]["id"].nunique()),
            "n_neg_ids": int(df[df["label"] == args.neg_label]["id"].nunique()),
        },
        "outputs": {
            "all_eval_csv": str(out_csv.resolve()),
            "best_by_metric_csv": str(best_csv.resolve()),
        },
    }
    summary_json = out_dir / f"summary_{args.task_name}.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_csv)
    print("[saved]", best_csv)
    print("[saved]", summary_json)
    print("[top]", out_df.head(10).to_dict(orient="records"))


if __name__ == "__main__":
    main()
