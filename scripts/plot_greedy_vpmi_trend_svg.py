#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple


def parse_bool(x: object) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_champion_rows(per_candidate_csv: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(per_candidate_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if not parse_bool(r.get("is_champion")):
                continue
            toks_json = str(r.get("core_vpmi_toks_json") or "").strip()
            if toks_json == "":
                continue
            try:
                toks = json.loads(toks_json)
            except Exception:
                continue
            if not isinstance(toks, list):
                continue
            vals = [safe_float(x) for x in toks]
            vals = [float(x) for x in vals if x is not None]
            if len(vals) == 0:
                continue
            label = str(r.get("label") or "").strip().lower()
            if label not in {"success", "failure"}:
                label = "success" if parse_bool(r.get("is_correct_eval")) else "failure"
            out.append(
                {
                    "id": str(r.get("id") or ""),
                    "label": label,
                    "vpmi_toks": vals,
                    "core_len": len(vals),
                }
            )
    return out


def assign_bin(pos_norm: float, n_bins: int) -> int:
    p = min(1.0, max(0.0, float(pos_norm)))
    idx = int(round(p * float(max(1, n_bins - 1))))
    return int(min(n_bins - 1, max(0, idx)))


def build_position_curves(
    rows: List[Dict[str, object]],
    n_bins: int,
) -> Dict[str, List[Optional[float]]]:
    by_label: Dict[str, List[List[float]]] = {
        "success": [[] for _ in range(n_bins)],
        "failure": [[] for _ in range(n_bins)],
    }
    for r in rows:
        label = str(r["label"])
        toks = [float(x) for x in r["vpmi_toks"]]  # type: ignore[index]
        m = len(toks)
        if m == 1:
            # Single-token answer has no trajectory; keep constant along normalized axis.
            for b in range(n_bins):
                by_label[label][b].append(float(toks[0]))
            continue
        # Resample each variable-length token trajectory onto fixed bins (linear interpolation).
        for b in range(n_bins):
            pos = float(b) / float(max(1, n_bins - 1))
            t = pos * float(m - 1)
            lo = int(math.floor(t))
            hi = int(math.ceil(t))
            if lo == hi:
                vv = float(toks[lo])
            else:
                w = t - float(lo)
                vv = float((1.0 - w) * float(toks[lo]) + w * float(toks[hi]))
            by_label[label][b].append(vv)
    return {
        "success": [mean(x) for x in by_label["success"]],
        "failure": [mean(x) for x in by_label["failure"]],
    }


def suffix_gap_values(rows: List[Dict[str, object]]) -> Dict[str, List[float]]:
    out = {"success": [], "failure": []}
    for r in rows:
        label = str(r["label"])
        toks = [float(x) for x in r["vpmi_toks"]]  # type: ignore[index]
        if len(toks) < 2:
            continue
        split = int(max(1, len(toks) // 2))
        prefix = toks[:split]
        suffix = toks[split:] if split < len(toks) else [toks[-1]]
        prefix_mean = float(sum(prefix) / len(prefix))
        suffix_min = float(min(suffix))
        gap = float(suffix_min - prefix_mean)
        out[label].append(gap)
    return out


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in keys})


def svg_line_chart(
    out_path: str,
    y_succ: List[Optional[float]],
    y_fail: List[Optional[float]],
    title: str,
    subtitle: str,
) -> None:
    w, h = 1080, 640
    ml, mr, mt, mb = 80, 40, 90, 70
    pw = w - ml - mr
    ph = h - mt - mb

    vals = [float(v) for v in (y_succ + y_fail) if v is not None and math.isfinite(float(v))]
    if not vals:
        vals = [-1.0, 1.0]
    y_min = float(min(vals))
    y_max = float(max(vals))
    if y_max <= y_min + 1e-9:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    n = int(max(1, len(y_succ)))

    def x_of(i: int) -> float:
        if n <= 1:
            return float(ml + pw / 2.0)
        return float(ml + (float(i) / float(n - 1)) * pw)

    def y_of(v: float) -> float:
        return float(mt + (1.0 - (float(v) - y_min) / (y_max - y_min)) * ph)

    def poly_points(ys: List[Optional[float]]) -> str:
        pts: List[str] = []
        for i, v in enumerate(ys):
            if v is None:
                continue
            pts.append(f"{x_of(i):.2f},{y_of(float(v)):.2f}")
        return " ".join(pts)

    succ_points = poly_points(y_succ)
    fail_points = poly_points(y_fail)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')
    lines.append(f'<text x="{ml}" y="64" font-size="15" font-family="Arial" fill="#444">{subtitle}</text>')

    # grid
    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = y_max - (y_max - y_min) * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="#e9e9e9" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{yv:.2f}</text>')

    # axes
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<text x="{ml + pw / 2:.2f}" y="{h - 22}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">normalized token position (prefix → suffix)</text>')
    lines.append(f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">mean token V-PMI</text>')

    # x ticks
    for k in range(6):
        i = int(round((n - 1) * (k / 5.0)))
        xx = x_of(i)
        lines.append(f'<line x1="{xx:.2f}" y1="{mt + ph}" x2="{xx:.2f}" y2="{mt + ph + 6}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{mt + ph + 24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#666">{k/5.0:.1f}</text>')

    if succ_points:
        lines.append(f'<polyline points="{succ_points}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    if fail_points:
        lines.append(f'<polyline points="{fail_points}" fill="none" stroke="#d62728" stroke-width="3"/>')

    # legend
    lx, ly = ml + 20, mt + 20
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 36}" y2="{ly}" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 44}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#1f77b4">correct</text>')
    lines.append(f'<line x1="{lx + 130}" y1="{ly}" x2="{lx + 166}" y2="{ly}" stroke="#d62728" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 174}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#d62728">incorrect</text>')

    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def histogram_density(values: List[float], edges: List[float]) -> List[float]:
    dens = [0.0 for _ in range(len(edges) - 1)]
    if len(values) == 0:
        return dens
    for v in values:
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            if (v >= lo and v < hi) or (i == len(edges) - 2 and v == hi):
                dens[i] += 1.0
                break
    n = float(len(values))
    return [d / n for d in dens]


def svg_hist_overlay(
    out_path: str,
    v_succ: List[float],
    v_fail: List[float],
    title: str,
    subtitle: str,
    n_bins: int = 36,
) -> None:
    w, h = 1080, 640
    ml, mr, mt, mb = 80, 40, 90, 70
    pw = w - ml - mr
    ph = h - mt - mb
    all_vals = [float(x) for x in (v_succ + v_fail) if math.isfinite(float(x))]
    if not all_vals:
        all_vals = [-1.0, 1.0]
    x_min = float(min(all_vals))
    x_max = float(max(all_vals))
    if x_max <= x_min + 1e-9:
        x_min -= 1.0
        x_max += 1.0
    pad = 0.05 * (x_max - x_min)
    x_min -= pad
    x_max += pad
    step = (x_max - x_min) / float(max(1, n_bins))
    edges = [x_min + i * step for i in range(n_bins + 1)]
    d_s = histogram_density(v_succ, edges)
    d_f = histogram_density(v_fail, edges)
    y_max = max([0.001] + d_s + d_f) * 1.08

    def x_of(x: float) -> float:
        return float(ml + (float(x) - x_min) / (x_max - x_min) * pw)

    def y_of(y: float) -> float:
        return float(mt + (1.0 - float(y) / y_max) * ph)

    def poly_from_density(d: List[float]) -> str:
        pts: List[str] = []
        for i, yy in enumerate(d):
            xc = 0.5 * (edges[i] + edges[i + 1])
            pts.append(f"{x_of(xc):.2f},{y_of(float(yy)):.2f}")
        return " ".join(pts)

    s_pts = poly_from_density(d_s)
    f_pts = poly_from_density(d_f)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')
    lines.append(f'<text x="{ml}" y="64" font-size="15" font-family="Arial" fill="#444">{subtitle}</text>')

    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = y_max - y_max * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="#e9e9e9" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{yv:.3f}</text>')

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<text x="{ml + pw / 2:.2f}" y="{h - 22}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">suffix_min - prefix_mean (more negative = stronger collapse)</text>')
    lines.append(f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">density</text>')

    for k in range(6):
        xv = x_min + (x_max - x_min) * (k / 5.0)
        xx = x_of(xv)
        lines.append(f'<line x1="{xx:.2f}" y1="{mt + ph}" x2="{xx:.2f}" y2="{mt + ph + 6}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{mt + ph + 24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#666">{xv:.2f}</text>')

    if s_pts:
        lines.append(f'<polyline points="{s_pts}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    if f_pts:
        lines.append(f'<polyline points="{f_pts}" fill="none" stroke="#d62728" stroke-width="3"/>')

    lx, ly = ml + 20, mt + 20
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 36}" y2="{ly}" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 44}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#1f77b4">correct</text>')
    lines.append(f'<line x1="{lx + 130}" y1="{ly}" x2="{lx + 166}" y2="{ly}" stroke="#d62728" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 174}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#d62728">incorrect</text>')

    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot greedy champion V-PMI trends (SVG, no matplotlib).")
    ap.add_argument("--per_candidate_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--hist_bins", type=int, default=36)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    rows = read_champion_rows(os.path.abspath(args.per_candidate_csv))

    curves = build_position_curves(rows, n_bins=int(max(5, args.bins)))
    gaps = suffix_gap_values(rows)

    curve_rows: List[Dict[str, object]] = []
    for i in range(int(max(5, args.bins))):
        curve_rows.append(
            {
                "bin_idx": int(i),
                "pos_norm": float(i / max(1, int(max(5, args.bins)) - 1)),
                "vpmi_mean_success": curves["success"][i],
                "vpmi_mean_failure": curves["failure"][i],
            }
        )
    write_csv(os.path.join(out_dir, "vpmi_curve_by_position.csv"), curve_rows)

    gap_rows: List[Dict[str, object]] = []
    for label in ("success", "failure"):
        vs = [float(x) for x in gaps[label]]
        if len(vs) == 0:
            gap_rows.append({"label": label, "n": 0, "mean": None, "median": None, "p25": None, "p75": None})
            continue
        xs = sorted(vs)
        n = len(xs)
        def q(p: float) -> float:
            pos = p * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return float(xs[lo])
            w = pos - lo
            return float((1.0 - w) * xs[lo] + w * xs[hi])
        gap_rows.append(
            {
                "label": label,
                "n": int(n),
                "mean": float(sum(xs) / n),
                "median": q(0.5),
                "p25": q(0.25),
                "p75": q(0.75),
            }
        )
    write_csv(os.path.join(out_dir, "suffix_collapse_gap_summary.csv"), gap_rows)

    title1 = "Greedy Champion Token V-PMI Trend"
    subtitle1 = f"source={os.path.abspath(args.per_candidate_csv)} | champion only | success vs failure"
    svg_line_chart(
        out_path=os.path.join(out_dir, "vpmi_token_position_curve.svg"),
        y_succ=curves["success"],
        y_fail=curves["failure"],
        title=title1,
        subtitle=subtitle1,
    )

    title2 = "Suffix Collapse Distribution"
    subtitle2 = "metric = suffix_min - prefix_mean (core V-PMI tokens), champion only, len>=2"
    svg_hist_overlay(
        out_path=os.path.join(out_dir, "suffix_collapse_gap_hist.svg"),
        v_succ=[float(x) for x in gaps["success"]],
        v_fail=[float(x) for x in gaps["failure"]],
        title=title2,
        subtitle=subtitle2,
        n_bins=int(max(10, args.hist_bins)),
    )

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "per_candidate_csv": os.path.abspath(args.per_candidate_csv),
                    "bins": int(max(5, args.bins)),
                    "hist_bins": int(max(10, args.hist_bins)),
                },
                "counts": {
                    "n_champion_rows": int(len(rows)),
                    "n_success": int(sum(1 for r in rows if str(r["label"]) == "success")),
                    "n_failure": int(sum(1 for r in rows if str(r["label"]) == "failure")),
                    "n_len_ge_2_for_gap": int(len(gaps["success"]) + len(gaps["failure"])),
                },
                "outputs": {
                    "vpmi_curve_csv": os.path.join(out_dir, "vpmi_curve_by_position.csv"),
                    "suffix_gap_summary_csv": os.path.join(out_dir, "suffix_collapse_gap_summary.csv"),
                    "vpmi_curve_svg": os.path.join(out_dir, "vpmi_token_position_curve.svg"),
                    "suffix_gap_hist_svg": os.path.join(out_dir, "suffix_collapse_gap_hist.svg"),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[saved]", os.path.join(out_dir, "vpmi_curve_by_position.csv"))
    print("[saved]", os.path.join(out_dir, "suffix_collapse_gap_summary.csv"))
    print("[saved]", os.path.join(out_dir, "vpmi_token_position_curve.svg"))
    print("[saved]", os.path.join(out_dir, "suffix_collapse_gap_hist.svg"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
