#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sigma


def summarize_group(g: pd.DataFrame, top_ratio: float, selector_col: str, value_col: str) -> pd.Series:
    score = g[selector_col].to_numpy(dtype=float)
    value = g[value_col].to_numpy(dtype=float)
    k = max(1, int(round(len(g) * top_ratio)))
    order = np.argsort(-score)
    keep = order[:k]
    mu_late = float(np.mean(value)) if len(value) else 0.0
    mu_selected = float(np.mean(value[keep])) if len(keep) else 0.0
    selector_mean = float(np.mean(score[keep])) if len(keep) else 0.0
    return pd.Series(
        {
            "dynamic_frg_mu_late": mu_late,
            "dynamic_frg_mu_selected": mu_selected,
            "dynamic_frg_faithful_gap": mu_selected - mu_late,
            "dynamic_frg_selector_selected_mean": selector_mean,
            "dynamic_frg_selected_k": int(k),
            "dynamic_frg_n_points": int(len(g)),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build dynamic faithful-membership FRG from per-head late-window probe traces."
    )
    ap.add_argument("--per_head_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--top_ratio", type=float, default=0.2)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--layer_col", type=str, default="block_layer_idx")
    ap.add_argument("--ratio_col", type=str, default="head_attn_vis_ratio")
    ap.add_argument("--peak_col", type=str, default="head_attn_vis_peak")
    ap.add_argument("--entropy_col", type=str, default="head_attn_vis_entropy")
    ap.add_argument("--w_ratio", type=float, default=1.0)
    ap.add_argument("--w_peak", type=float, default=1.0)
    ap.add_argument("--w_entropy", type=float, default=1.0)
    args = ap.parse_args()

    if not (0.0 < args.top_ratio <= 1.0):
        raise ValueError("--top_ratio must be in (0, 1].")

    usecols = [args.id_col, args.layer_col, args.ratio_col, args.peak_col, args.entropy_col]
    df = pd.read_csv(args.per_head_csv, usecols=usecols)
    df = df.dropna(subset=usecols).copy()
    df = df[(df[args.layer_col] >= args.late_start) & (df[args.layer_col] <= args.late_end)].copy()
    if len(df) == 0:
        raise RuntimeError("No rows remained after late-window filtering.")

    df["__z_ratio__"] = df.groupby(args.id_col)[args.ratio_col].transform(
        lambda s: zscore(s.to_numpy(dtype=float))
    )
    df["__z_peak__"] = df.groupby(args.id_col)[args.peak_col].transform(
        lambda s: zscore(s.to_numpy(dtype=float))
    )
    df["__z_entropy__"] = df.groupby(args.id_col)[args.entropy_col].transform(
        lambda s: zscore(s.to_numpy(dtype=float))
    )
    df["__selector__"] = (
        float(args.w_ratio) * df["__z_ratio__"].to_numpy(dtype=float)
        + float(args.w_peak) * df["__z_peak__"].to_numpy(dtype=float)
        - float(args.w_entropy) * df["__z_entropy__"].to_numpy(dtype=float)
    )
    out = (
        df.groupby(args.id_col, group_keys=False)[[args.ratio_col, "__selector__"]]
        .apply(lambda g: summarize_group(g, args.top_ratio, "__selector__", args.ratio_col))
        .reset_index()
    )
    out["dynamic_frg_late_start"] = args.late_start
    out["dynamic_frg_late_end"] = args.late_end
    out["dynamic_frg_top_ratio"] = args.top_ratio
    out["dynamic_frg_w_ratio"] = args.w_ratio
    out["dynamic_frg_w_peak"] = args.w_peak
    out["dynamic_frg_w_entropy"] = args.w_entropy

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = {
        "per_head_csv": os.path.abspath(args.per_head_csv),
        "out_csv": os.path.abspath(args.out_csv),
        "late_start": args.late_start,
        "late_end": args.late_end,
        "top_ratio": args.top_ratio,
        "weights": {
            "ratio": args.w_ratio,
            "peak": args.w_peak,
            "entropy": args.w_entropy,
        },
        "n_ids": int(out[args.id_col].nunique()),
        "n_rows_filtered": int(len(df)),
        "feature_col": "dynamic_frg_faithful_gap",
        "mu_late_mean": float(out["dynamic_frg_mu_late"].mean()),
        "mu_selected_mean": float(out["dynamic_frg_mu_selected"].mean()),
        "gap_mean": float(out["dynamic_frg_faithful_gap"].mean()),
        "selector_selected_mean": float(out["dynamic_frg_selector_selected_mean"].mean()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
