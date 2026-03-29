#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Tuple

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def parse_layers(s: str) -> List[int]:
    out: List[int] = []
    for t in str(s).split(","):
        tt = t.strip()
        if tt == "":
            continue
        out.append(int(tt))
    if len(out) == 0:
        raise RuntimeError("No valid layers parsed.")
    return out


def parse_json_list(x: str) -> List[Any]:
    try:
        v = json.loads(x)
        if isinstance(v, list):
            return v
        return []
    except Exception:
        return []


def matrix_zeros(h: int, w: int) -> List[List[float]]:
    return [[0.0 for _ in range(w)] for _ in range(h)]


def matrix_add_inplace(a: List[List[float]], b: List[List[float]]) -> None:
    for r in range(len(a)):
        ar = a[r]
        br = b[r]
        for c in range(len(ar)):
            ar[c] += br[c]


def matrix_div_scalar_inplace(a: List[List[float]], s: float) -> None:
    if s == 0.0:
        return
    for r in range(len(a)):
        ar = a[r]
        for c in range(len(ar)):
            ar[c] /= s


def matrix_minmax(a: List[List[float]]) -> Tuple[float, float]:
    mn = float("inf")
    mx = float("-inf")
    for row in a:
        for v in row:
            if v < mn:
                mn = v
            if v > mx:
                mx = v
    if mn == float("inf"):
        mn = 0.0
    if mx == float("-inf"):
        mx = 0.0
    return mn, mx


def matrix_to_csv(path: str, mat: List[List[float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        h = len(mat)
        w = len(mat[0]) if h > 0 else 0
        wr.writerow(["row"] + [f"c{c}" for c in range(w)])
        for r in range(h):
            wr.writerow([r] + [f"{mat[r][c]:.8f}" for c in range(w)])


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def value_to_rgb_sequential(v: float, vmin: float, vmax: float) -> Tuple[int, int, int]:
    # White -> red
    if vmax <= vmin:
        t = 0.0
    else:
        t = _clamp01((v - vmin) / (vmax - vmin))
    r = int(round(255 * (0.2 + 0.8 * t)))
    g = int(round(255 * (1.0 - 0.7 * t)))
    b = int(round(255 * (1.0 - 0.7 * t)))
    return r, g, b


def value_to_rgb_diverging(v: float, vmax_abs: float) -> Tuple[int, int, int]:
    # Blue(negative) -> white(0) -> red(positive)
    if vmax_abs <= 0.0:
        t = 0.0
    else:
        t = _clamp01(v / vmax_abs)
    if v >= 0:
        rr = int(round(255 * (1.0)))
        gg = int(round(255 * (1.0 - 0.85 * t)))
        bb = int(round(255 * (1.0 - 0.85 * t)))
        return rr, gg, bb
    tn = _clamp01((-v) / vmax_abs)
    rr = int(round(255 * (1.0 - 0.85 * tn)))
    gg = int(round(255 * (1.0 - 0.85 * tn)))
    bb = int(round(255 * (1.0)))
    return rr, gg, bb


def save_heatmap_png(
    path: str,
    mat: List[List[float]],
    mode: str,
    title: str,
    cell: int = 18,
) -> None:
    if not PIL_AVAILABLE:
        return
    h = len(mat)
    w = len(mat[0]) if h > 0 else 0
    if h <= 0 or w <= 0:
        raise RuntimeError("empty matrix")
    pad_top = 28
    img = Image.new("RGB", (w * cell, h * cell + pad_top), (255, 255, 255))
    dr = ImageDraw.Draw(img)

    if mode == "seq":
        vmin, vmax = matrix_minmax(mat)
        for r in range(h):
            for c in range(w):
                col = value_to_rgb_sequential(mat[r][c], vmin, vmax)
                x0 = c * cell
                y0 = pad_top + r * cell
                dr.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=col)
    elif mode == "diff":
        vmax_abs = 0.0
        for r in range(h):
            for c in range(w):
                vmax_abs = max(vmax_abs, abs(mat[r][c]))
        for r in range(h):
            for c in range(w):
                col = value_to_rgb_diverging(mat[r][c], vmax_abs)
                x0 = c * cell
                y0 = pad_top + r * cell
                dr.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=col)
    else:
        raise RuntimeError(f"unknown mode: {mode}")

    dr.text((6, 6), title, fill=(20, 20, 20))
    img.save(path)


def row_to_hitmap(
    row: Dict[str, str],
    grid_w: int,
    grid_h: int,
    idx_col: str,
    sim_col: str,
    weight_mode: str,
) -> List[List[float]]:
    hit = matrix_zeros(grid_h, grid_w)
    idx_list = [int(x) for x in parse_json_list(row.get(idx_col, ""))]
    sim_list_raw = parse_json_list(row.get(sim_col, ""))
    sim_list: List[float] = []
    for x in sim_list_raw:
        try:
            sim_list.append(float(x))
        except Exception:
            sim_list.append(0.0)

    for rank, pidx in enumerate(idx_list):
        rr = int(pidx) // int(grid_w)
        cc = int(pidx) % int(grid_w)
        if rr < 0 or cc < 0 or rr >= grid_h or cc >= grid_w:
            continue
        if weight_mode == "uniform":
            w = 1.0
        elif weight_mode == "rank":
            w = 1.0 / float(rank + 1)
        elif weight_mode == "sim":
            if rank < len(sim_list):
                w = float(sim_list[rank])
            else:
                w = 0.0
        else:
            raise RuntimeError(f"unknown weight_mode: {weight_mode}")
        hit[rr][cc] += float(w)
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate patch top-k into layer/group heatmaps.")
    ap.add_argument("--overlay_manifest_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--layers", type=str, default="10,17,24")
    ap.add_argument("--idx_col", type=str, default="top_obj_idx_global", choices=["top_obj_idx_global", "top_global_idx"])
    ap.add_argument("--sim_col", type=str, default="top_obj_sim", choices=["top_obj_sim", "top_global_sim"])
    ap.add_argument("--weight_mode", type=str, default="sim", choices=["uniform", "rank", "sim"])
    args = ap.parse_args()

    layers = set(parse_layers(args.layers))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not PIL_AVAILABLE:
        print("[warn] PIL is not available. Only CSV outputs will be written (no PNG heatmaps).")

    rows: List[Dict[str, str]] = []
    with open(os.path.abspath(args.overlay_manifest_csv), "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    if len(rows) == 0:
        raise RuntimeError("overlay manifest is empty")

    groups = ["fp_hall", "tp_yes"]
    by_layer_group: Dict[Tuple[int, str], List[Dict[str, str]]] = {}
    for r in rows:
        try:
            layer = int(float(r.get("layer", "-1")))
        except Exception:
            continue
        if layer not in layers:
            continue
        group = str(r.get("group", ""))
        if group not in groups:
            continue
        by_layer_group.setdefault((layer, group), []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    for layer in sorted(layers):
        # Infer grid from available rows
        base_rows = by_layer_group.get((layer, "fp_hall"), []) + by_layer_group.get((layer, "tp_yes"), [])
        if len(base_rows) == 0:
            continue
        try:
            gh = int(float(base_rows[0].get("grid_h", "24")))
            gw = int(float(base_rows[0].get("grid_w", "24")))
        except Exception:
            gh, gw = 24, 24

        mats: Dict[str, List[List[float]]] = {}
        n_per_group: Dict[str, int] = {}
        for g in groups:
            sub = by_layer_group.get((layer, g), [])
            n_per_group[g] = int(len(sub))
            acc = matrix_zeros(gh, gw)
            for r in sub:
                hmap = row_to_hitmap(
                    row=r,
                    grid_w=gw,
                    grid_h=gh,
                    idx_col=str(args.idx_col),
                    sim_col=str(args.sim_col),
                    weight_mode=str(args.weight_mode),
                )
                matrix_add_inplace(acc, hmap)
            matrix_div_scalar_inplace(acc, float(max(1, len(sub))))
            mats[g] = acc

            csv_path = os.path.join(out_dir, f"heatmap_{g}_layer_{layer:02d}.csv")
            png_path = os.path.join(out_dir, f"heatmap_{g}_layer_{layer:02d}.png")
            matrix_to_csv(csv_path, acc)
            save_heatmap_png(
                path=png_path,
                mat=acc,
                mode="seq",
                title=f"{g} layer={layer} ({args.idx_col}, {args.weight_mode})",
            )

            mn, mx = matrix_minmax(acc)
            summary_rows.append(
                {
                    "layer": int(layer),
                    "group": g,
                    "n_rows": int(len(sub)),
                    "matrix_min": float(mn),
                    "matrix_max": float(mx),
                    "csv_path": csv_path,
                    "png_path": png_path,
                }
            )

        # fp - tp diff
        if ("fp_hall" in mats) and ("tp_yes" in mats):
            diff = matrix_zeros(gh, gw)
            for rr in range(gh):
                for cc in range(gw):
                    diff[rr][cc] = float(mats["fp_hall"][rr][cc] - mats["tp_yes"][rr][cc])
            d_csv = os.path.join(out_dir, f"heatmap_diff_fp_minus_tp_layer_{layer:02d}.csv")
            d_png = os.path.join(out_dir, f"heatmap_diff_fp_minus_tp_layer_{layer:02d}.png")
            matrix_to_csv(d_csv, diff)
            save_heatmap_png(
                path=d_png,
                mat=diff,
                mode="diff",
                title=f"diff(fp-tp) layer={layer} ({args.idx_col}, {args.weight_mode})",
            )
            mn, mx = matrix_minmax(diff)
            summary_rows.append(
                {
                    "layer": int(layer),
                    "group": "diff_fp_minus_tp",
                    "n_rows": int(min(n_per_group.get("fp_hall", 0), n_per_group.get("tp_yes", 0))),
                    "matrix_min": float(mn),
                    "matrix_max": float(mx),
                    "csv_path": d_csv,
                    "png_path": d_png,
                }
            )

    summary_csv = os.path.join(out_dir, "heatmap_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        if len(summary_rows) > 0:
            wr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            wr.writeheader()
            for r in summary_rows:
                wr.writerow(r)
        else:
            wr = csv.writer(f)
            wr.writerow(["layer", "group", "n_rows", "matrix_min", "matrix_max", "csv_path", "png_path"])

    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "overlay_manifest_csv": os.path.abspath(args.overlay_manifest_csv),
                    "layers": sorted(list(layers)),
                    "idx_col": str(args.idx_col),
                    "sim_col": str(args.sim_col),
                    "weight_mode": str(args.weight_mode),
                },
                "counts": {
                    "n_rows_input": int(len(rows)),
                    "n_summary_rows": int(len(summary_rows)),
                },
                "outputs": {
                    "heatmap_summary_csv": summary_csv,
                    "summary_json": summary_json,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("[saved]", summary_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
