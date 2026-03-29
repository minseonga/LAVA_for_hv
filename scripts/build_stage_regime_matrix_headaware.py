#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


def build_faithful_heads(
    head_eval: pd.DataFrame,
    metric_base: str,
    auc_min: float,
    topk_per_layer: int,
) -> Dict[int, Set[int]]:
    df = head_eval.copy()
    if "metric_base" in df.columns:
        df = df[df["metric_base"] == metric_base]
    if "comparison" in df.columns:
        df = df[df["comparison"] == "fp_hall_vs_tp_yes"]

    # faithful-like: lower_in_hallucination
    df = df[df["direction"] == "lower_in_hallucination"].copy()
    if df.empty:
        raise RuntimeError("No faithful candidates in head_eval")

    out: Dict[int, Set[int]] = {}
    for li, dli in df.groupby("block_layer_idx"):
        d = dli.sort_values("auc_best_dir", ascending=False).copy()
        keep = d[d["auc_best_dir"] >= auc_min]
        if keep.empty:
            keep = d.head(max(1, topk_per_layer))
        else:
            keep = keep.head(max(1, topk_per_layer))
        out[int(li)] = set(int(x) for x in keep["head_idx"].tolist())
    return out


def faithful_layer_values(
    per_head: pd.DataFrame,
    faithful_heads: Dict[int, Set[int]],
    id_col: str,
    attn_col: str,
) -> pd.DataFrame:
    ph = per_head.copy()
    ph = ph[[id_col, "block_layer_idx", "head_idx", attn_col]]
    ph[id_col] = ph[id_col].astype(str)

    rows = []
    for li, dli in ph.groupby("block_layer_idx"):
        lii = int(li)
        hs = faithful_heads.get(lii, set())
        if not hs:
            continue
        d = dli[dli["head_idx"].isin(hs)]
        if d.empty:
            continue
        g = d.groupby(id_col, as_index=False)[attn_col].mean()
        g["block_layer_idx"] = lii
        g = g.rename(columns={attn_col: "C_layer"})
        rows.append(g)
    if not rows:
        raise RuntimeError("No faithful layer values built")
    return pd.concat(rows, axis=0, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Regime × Stage matrix using head-aware faithful routing C_mid/C_late.")
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--per_head_trace_csv", type=str, required=True)
    ap.add_argument("--head_eval_csv", type=str, required=True)
    ap.add_argument("--regime_csv", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--a_col", type=str, default="obj_token_prob_lse")
    ap.add_argument("--e_col", type=str, default="guidance_mismatch_score")
    ap.add_argument("--head_metric_base", type=str, default="head_attn_vis_ratio")
    ap.add_argument("--head_attn_col", type=str, default="head_attn_vis_ratio")
    ap.add_argument("--faithful_auc_min", type=float, default=0.6)
    ap.add_argument("--faithful_topk_per_layer", type=int, default=6)
    ap.add_argument("--mid_start", type=int, default=17)
    ap.add_argument("--mid_end", type=int, default=18)
    ap.add_argument("--late_start", type=int, default=26)
    ap.add_argument("--late_end", type=int, default=30)
    ap.add_argument("--y_mode", type=str, default="overassert_baseline", choices=["overassert_baseline", "pred_yes_baseline"])
    ap.add_argument("--include_pope_groups", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(args.features_csv)
    ph = pd.read_csv(args.per_head_trace_csv)
    he = pd.read_csv(args.head_eval_csv)
    reg = pd.read_csv(args.regime_csv)

    for d in [feat, ph, he, reg]:
        if args.id_col in d.columns:
            d[args.id_col] = d[args.id_col].astype(str)

    # A/E
    need_feat = [args.id_col, args.a_col, args.e_col]
    pope_flag_cols = [c for c in ["target_is_fp_hallucination", "target_is_tp_yes"] if c in feat.columns]
    need_feat += pope_flag_cols
    sf = feat[need_feat].copy()

    faithful_heads = build_faithful_heads(
        he,
        metric_base=args.head_metric_base,
        auc_min=float(args.faithful_auc_min),
        topk_per_layer=int(args.faithful_topk_per_layer),
    )

    c_layer = faithful_layer_values(ph, faithful_heads, id_col=args.id_col, attn_col=args.head_attn_col)

    mid = c_layer[(c_layer["block_layer_idx"] >= args.mid_start) & (c_layer["block_layer_idx"] <= args.mid_end)]
    late = c_layer[(c_layer["block_layer_idx"] >= args.late_start) & (c_layer["block_layer_idx"] <= args.late_end)]

    c_mid = mid.groupby(args.id_col, as_index=False)["C_layer"].mean().rename(columns={"C_layer": "C_mid"})
    c_late = late.groupby(args.id_col, as_index=False)["C_layer"].mean().rename(columns={"C_layer": "C_late"})

    # Regime + Y
    rg = reg[[args.id_col, "case_type", "pred_baseline", "gt"]].copy()
    rg["regime"] = rg["case_type"].map(map_regime)
    pb = rg["pred_baseline"].astype(str).str.lower().str.strip()
    gt = rg["gt"].astype(str).str.lower().str.strip()
    rg["Y_pred_yes_baseline"] = (pb == "yes").astype(float)
    rg["Y_overassert_baseline"] = ((pb == "yes") & (gt == "no")).astype(float)
    rg["Y"] = rg["Y_overassert_baseline"] if args.y_mode == "overassert_baseline" else rg["Y_pred_yes_baseline"]

    df = sf.merge(c_mid, on=args.id_col, how="inner").merge(c_late, on=args.id_col, how="inner").merge(rg[[args.id_col, "regime", "Y"]], on=args.id_col, how="inner")
    if df.empty:
        raise RuntimeError("No merged rows for stage matrix")

    df = df.rename(columns={args.a_col: "A", args.e_col: "E"})

    for c in ["A", "C_mid", "C_late", "E", "Y"]:
        df[f"{c}_z"] = zscore_series(df[c])

    stage_cols = ["A", "C_mid", "C_late", "E", "Y"]
    stage_z_cols = [f"{c}_z" for c in stage_cols]
    reg_sum = df.groupby("regime", as_index=False)[stage_cols + stage_z_cols].mean()
    reg_cnt = df.groupby("regime", as_index=False).size().rename(columns={"size": "n"})
    reg_sum = reg_sum.merge(reg_cnt, on="regime", how="left")

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

    per_id_csv = out_dir / "per_id_stage_features.csv"
    regime_csv = out_dir / "regime_stage_matrix.csv"
    heads_json = out_dir / "faithful_heads_by_layer.json"
    df.to_csv(per_id_csv, index=False)
    reg_sum.to_csv(regime_csv, index=False)
    with heads_json.open("w", encoding="utf-8") as f:
        json.dump({str(k): sorted(list(v)) for k, v in faithful_heads.items()}, f, ensure_ascii=False, indent=2)

    main_order = [r for r in ["VV", "FV", "VF", "FF"] if r in set(reg_sum["regime"])]
    aux_order = [r for r in ["fp_hall", "tp_yes"] if r in set(reg_sum["regime"])]

    heat_main_raw = out_dir / "stage_regime_heatmap_raw.png"
    heat_main_z = out_dir / "stage_regime_heatmap_z.png"
    draw_heatmap(reg_sum, main_order, stage_cols, heat_main_raw, "Regime × Stage (head-aware C, raw)")
    draw_heatmap(reg_sum, main_order, stage_z_cols, heat_main_z, "Regime × Stage (head-aware C, z)")

    heat_aux_raw = ""
    heat_aux_z = ""
    if aux_order:
        heat_aux_raw_p = out_dir / "stage_regime_heatmap_aux_raw.png"
        heat_aux_z_p = out_dir / "stage_regime_heatmap_aux_z.png"
        draw_heatmap(reg_sum, aux_order, stage_cols, heat_aux_raw_p, "Aux groups × Stage (head-aware C, raw)")
        draw_heatmap(reg_sum, aux_order, stage_z_cols, heat_aux_z_p, "Aux groups × Stage (head-aware C, z)")
        heat_aux_raw = str(heat_aux_raw_p.resolve())
        heat_aux_z = str(heat_aux_z_p.resolve())

    summary = {
        "inputs": {
            "features_csv": str(Path(args.features_csv).resolve()),
            "per_head_trace_csv": str(Path(args.per_head_trace_csv).resolve()),
            "head_eval_csv": str(Path(args.head_eval_csv).resolve()),
            "regime_csv": str(Path(args.regime_csv).resolve()),
            "a_col": args.a_col,
            "e_col": args.e_col,
            "head_metric_base": args.head_metric_base,
            "head_attn_col": args.head_attn_col,
            "faithful_auc_min": float(args.faithful_auc_min),
            "faithful_topk_per_layer": int(args.faithful_topk_per_layer),
            "mid_range": [int(args.mid_start), int(args.mid_end)],
            "late_range": [int(args.late_start), int(args.late_end)],
            "y_mode": args.y_mode,
            "include_pope_groups": bool(args.include_pope_groups),
        },
        "counts": {
            "n_rows": int(len(df)),
            "n_ids": int(df[args.id_col].nunique()),
            "regime_counts": {str(k): int(v) for k, v in df["regime"].value_counts().to_dict().items()},
            "n_faithful_layers": int(len(faithful_heads)),
        },
        "outputs": {
            "per_id_csv": str(per_id_csv.resolve()),
            "regime_matrix_csv": str(regime_csv.resolve()),
            "faithful_heads_json": str(heads_json.resolve()),
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
    print("[saved]", heads_json)
    print("[saved]", heat_main_raw)
    print("[saved]", heat_main_z)
    if aux_order:
        print("[saved]", heat_aux_raw)
        print("[saved]", heat_aux_z)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
