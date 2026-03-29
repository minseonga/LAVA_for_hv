#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def map_regime(case_type: str) -> str:
    s = str(case_type)
    mp = {
        "both_correct": "VV",
        "vga_improvement": "FV",
        "vga_regression": "VF",
        "both_wrong": "FF",
    }
    return mp.get(s, s)


def zscore_series(s: pd.Series, eps: float = 1e-8) -> pd.Series:
    x = s.astype(float)
    mu = float(x.mean())
    sd = float(x.std())
    if sd < eps:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return (x - mu) / sd


def draw_heatmap(df: pd.DataFrame, rows: List[str], cols: List[str], out_png: Path, title: str) -> None:
    mat = []
    for r in rows:
        rr = df[df["regime"] == r]
        if rr.empty:
            mat.append([np.nan] * len(cols))
        else:
            mat.append([float(rr.iloc[0][c]) for c in cols])
    arr = np.array(mat, dtype=float)

    plt.figure(figsize=(1.6 * len(cols) + 2, 0.7 * len(rows) + 2))
    im = plt.imshow(arr, aspect="auto", cmap="RdBu_r")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=25, ha="right")
    plt.yticks(range(len(rows)), rows)
    plt.title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Regime × Stage matrix (A, C_mid, C_late, E, Y).")
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--layerwise_csv", type=str, required=True)
    ap.add_argument("--regime_csv", type=str, required=True, help="per_case_compare.csv")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--a_col", type=str, default="obj_token_prob_lse")
    ap.add_argument("--e_col", type=str, default="guidance_mismatch_score")
    ap.add_argument("--c_col", type=str, default="C")
    ap.add_argument("--mid_start", type=int, default=17)
    ap.add_argument("--mid_end", type=int, default=18)
    ap.add_argument("--late_start", type=int, default=26)
    ap.add_argument("--late_end", type=int, default=30)
    ap.add_argument("--y_mode", type=str, default="overassert_baseline", choices=["overassert_baseline", "pred_yes_baseline"])
    ap.add_argument("--y_margin_csv", type=str, default="", help="Optional CSV with id,y_margin")
    ap.add_argument("--y_margin_col", type=str, default="y_margin")
    ap.add_argument("--include_pope_groups", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(args.features_csv)
    lay = pd.read_csv(args.layerwise_csv)
    reg = pd.read_csv(args.regime_csv)

    feat[args.id_col] = feat[args.id_col].astype(str)
    lay[args.id_col] = lay[args.id_col].astype(str)
    reg[args.id_col] = reg[args.id_col].astype(str)

    # Stage A/E from sample-level features
    need_feat = [args.id_col, args.a_col, args.e_col]
    pope_flag_cols = [c for c in ["target_is_fp_hallucination", "target_is_tp_yes"] if c in feat.columns]
    need_feat += pope_flag_cols
    sf = feat[need_feat].copy()

    # Stage C_mid / C_late from layerwise proxy
    if "block_layer_idx" not in lay.columns:
        raise RuntimeError("layerwise_csv missing block_layer_idx")
    if args.c_col not in lay.columns:
        raise RuntimeError(f"layerwise_csv missing C column: {args.c_col}")

    mid = lay[(lay["block_layer_idx"] >= args.mid_start) & (lay["block_layer_idx"] <= args.mid_end)]
    late = lay[(lay["block_layer_idx"] >= args.late_start) & (lay["block_layer_idx"] <= args.late_end)]

    c_mid = mid.groupby(args.id_col, as_index=False)[args.c_col].mean().rename(columns={args.c_col: "C_mid"})
    c_late = late.groupby(args.id_col, as_index=False)[args.c_col].mean().rename(columns={args.c_col: "C_late"})

    # Regime + Y
    needed_reg_cols = [args.id_col, "case_type", "pred_baseline", "gt"]
    for c in needed_reg_cols:
        if c not in reg.columns:
            raise RuntimeError(f"regime_csv missing column: {c}")
    rg = reg[needed_reg_cols].copy()
    rg["regime"] = rg["case_type"].map(map_regime)

    pb = rg["pred_baseline"].astype(str).str.lower().str.strip()
    gt = rg["gt"].astype(str).str.lower().str.strip()
    rg["Y_pred_yes_baseline"] = (pb == "yes").astype(float)
    rg["Y_overassert_baseline"] = ((pb == "yes") & (gt == "no")).astype(float)

    # optional real y_margin override
    y_col_final = "Y_proxy"
    if args.y_margin_csv.strip():
        ym = pd.read_csv(args.y_margin_csv)
        ym[args.id_col] = ym[args.id_col].astype(str)
        if args.y_margin_col not in ym.columns:
            raise RuntimeError(f"y_margin_csv missing: {args.y_margin_col}")
        rg = rg.merge(ym[[args.id_col, args.y_margin_col]], on=args.id_col, how="left")
        rg[y_col_final] = rg[args.y_margin_col].astype(float)
    else:
        if args.y_mode == "pred_yes_baseline":
            rg[y_col_final] = rg["Y_pred_yes_baseline"].astype(float)
        else:
            rg[y_col_final] = rg["Y_overassert_baseline"].astype(float)

    # Merge all
    df = sf.merge(c_mid, on=args.id_col, how="inner").merge(c_late, on=args.id_col, how="inner").merge(rg[[args.id_col, "regime", y_col_final]], on=args.id_col, how="inner")
    if df.empty:
        raise RuntimeError("No merged rows for stage matrix")

    # rename stage columns
    df = df.rename(columns={args.a_col: "A", args.e_col: "E", y_col_final: "Y"})

    # z-normalized stage columns for comparable heatmap
    for c in ["A", "C_mid", "C_late", "E", "Y"]:
        df[f"{c}_z"] = zscore_series(df[c])

    # regime summary
    stage_cols = ["A", "C_mid", "C_late", "E", "Y"]
    stage_z_cols = [f"{c}_z" for c in stage_cols]
    reg_sum = df.groupby("regime", as_index=False)[stage_cols + stage_z_cols].mean()
    reg_cnt = df.groupby("regime", as_index=False).size().rename(columns={"size": "n"})
    reg_sum = reg_sum.merge(reg_cnt, on="regime", how="left")

    # add optional fp_hall / tp_yes rows from feature flags
    extra_rows = []
    if args.include_pope_groups and pope_flag_cols:
        if "target_is_fp_hallucination" in df.columns:
            d = df[df["target_is_fp_hallucination"] == 1]
            if not d.empty:
                r = {"regime": "fp_hall", "n": int(len(d))}
                for c in stage_cols + stage_z_cols:
                    r[c] = float(d[c].mean())
                extra_rows.append(r)
        if "target_is_tp_yes" in df.columns:
            d = df[df["target_is_tp_yes"] == 1]
            if not d.empty:
                r = {"regime": "tp_yes", "n": int(len(d))}
                for c in stage_cols + stage_z_cols:
                    r[c] = float(d[c].mean())
                extra_rows.append(r)

    if extra_rows:
        reg_sum = pd.concat([reg_sum, pd.DataFrame(extra_rows)], axis=0, ignore_index=True)

    # Save tables
    per_id_csv = out_dir / "per_id_stage_features.csv"
    regime_csv = out_dir / "regime_stage_matrix.csv"
    df.to_csv(per_id_csv, index=False)
    reg_sum.to_csv(regime_csv, index=False)

    # Heatmaps
    main_order = [r for r in ["VV", "FV", "VF", "FF"] if r in set(reg_sum["regime"])]
    aux_order = [r for r in ["fp_hall", "tp_yes"] if r in set(reg_sum["regime"])]

    heat_main_raw = out_dir / "stage_regime_heatmap_raw.png"
    heat_main_z = out_dir / "stage_regime_heatmap_z.png"
    draw_heatmap(reg_sum, main_order, stage_cols, heat_main_raw, "Regime × Stage (raw means)")
    draw_heatmap(reg_sum, main_order, stage_z_cols, heat_main_z, "Regime × Stage (z-normalized means)")

    heat_aux_raw = ""
    heat_aux_z = ""
    if aux_order:
        heat_aux_raw_p = out_dir / "stage_regime_heatmap_aux_raw.png"
        heat_aux_z_p = out_dir / "stage_regime_heatmap_aux_z.png"
        draw_heatmap(reg_sum, aux_order, stage_cols, heat_aux_raw_p, "Aux groups × Stage (raw means)")
        draw_heatmap(reg_sum, aux_order, stage_z_cols, heat_aux_z_p, "Aux groups × Stage (z-normalized means)")
        heat_aux_raw = str(heat_aux_raw_p.resolve())
        heat_aux_z = str(heat_aux_z_p.resolve())

    summary = {
        "inputs": {
            "features_csv": str(Path(args.features_csv).resolve()),
            "layerwise_csv": str(Path(args.layerwise_csv).resolve()),
            "regime_csv": str(Path(args.regime_csv).resolve()),
            "a_col": args.a_col,
            "e_col": args.e_col,
            "c_col": args.c_col,
            "mid_range": [int(args.mid_start), int(args.mid_end)],
            "late_range": [int(args.late_start), int(args.late_end)],
            "y_mode": args.y_mode,
            "y_margin_csv": str(Path(args.y_margin_csv).resolve()) if args.y_margin_csv.strip() else "",
            "y_margin_col": args.y_margin_col,
            "include_pope_groups": bool(args.include_pope_groups),
        },
        "counts": {
            "n_rows": int(len(df)),
            "n_ids": int(df[args.id_col].nunique()),
            "regime_counts": {str(k): int(v) for k, v in df["regime"].value_counts().to_dict().items()},
        },
        "outputs": {
            "per_id_csv": str(per_id_csv.resolve()),
            "regime_matrix_csv": str(regime_csv.resolve()),
            "heatmap_main_raw": str(heat_main_raw.resolve()),
            "heatmap_main_z": str(heat_main_z.resolve()),
            "heatmap_aux_raw": heat_aux_raw,
            "heatmap_aux_z": heat_aux_z,
        },
    }
    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", per_id_csv)
    print("[saved]", regime_csv)
    print("[saved]", heat_main_raw)
    print("[saved]", heat_main_z)
    if aux_order:
        print("[saved]", heat_aux_raw)
        print("[saved]", heat_aux_z)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
