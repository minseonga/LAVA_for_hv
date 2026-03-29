#!/usr/bin/env python3
import argparse
import json
import os

import pandas as pd


def topk_mean(series: pd.Series, top_ratio: float) -> float:
    if len(series) == 0:
        return 0.0
    k = max(1, int(round(len(series) * top_ratio)))
    return float(series.nlargest(k).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Build shared headset-free FRG from per-head late-window visual usage.")
    ap.add_argument("--per_head_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--top_ratio", type=float, default=0.2)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--layer_col", type=str, default="block_layer_idx")
    ap.add_argument("--value_col", type=str, default="head_attn_vis_ratio")
    args = ap.parse_args()

    if not (0.0 < args.top_ratio <= 1.0):
        raise ValueError("--top_ratio must be in (0, 1].")

    df = pd.read_csv(args.per_head_csv, usecols=[args.id_col, args.layer_col, args.value_col])
    df = df.dropna(subset=[args.id_col, args.layer_col, args.value_col]).copy()
    df = df[(df[args.layer_col] >= args.late_start) & (df[args.layer_col] <= args.late_end)].copy()
    if len(df) == 0:
        raise RuntimeError("No rows remained after late-window filtering.")

    grouped = df.groupby(args.id_col)[args.value_col]
    out = grouped.agg(new_frg_mu_late="mean").reset_index()
    out["new_frg_mu_top"] = grouped.apply(lambda s: topk_mean(s, args.top_ratio)).values
    out["new_frg_shared_topk_gap"] = out["new_frg_mu_top"] - out["new_frg_mu_late"]
    out["new_frg_n_points"] = grouped.size().values.astype(int)
    out["new_frg_late_start"] = args.late_start
    out["new_frg_late_end"] = args.late_end
    out["new_frg_top_ratio"] = args.top_ratio

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    summary = {
        "per_head_csv": os.path.abspath(args.per_head_csv),
        "out_csv": os.path.abspath(args.out_csv),
        "late_start": args.late_start,
        "late_end": args.late_end,
        "top_ratio": args.top_ratio,
        "n_ids": int(out[args.id_col].nunique()),
        "n_rows_filtered": int(len(df)),
        "feature_col": "new_frg_shared_topk_gap",
        "mu_late_mean": float(out["new_frg_mu_late"].mean()),
        "mu_top_mean": float(out["new_frg_mu_top"].mean()),
        "frg_mean": float(out["new_frg_shared_topk_gap"].mean()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
