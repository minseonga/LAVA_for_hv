#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_bool(x: Any) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in keys})


def quantile(vals: List[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if safe_float(v) is not None)
    if len(xs) == 0:
        return None
    if len(xs) == 1:
        return float(xs[0])
    p = min(1.0, max(0.0, float(q))) * (len(xs) - 1)
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    if lo == hi:
        return float(xs[lo])
    w = p - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def interpolate_curve(vals: List[float], n_bins: int) -> List[float]:
    m = len(vals)
    if m <= 0:
        return [0.0 for _ in range(n_bins)]
    if m == 1:
        return [float(vals[0]) for _ in range(n_bins)]
    out: List[float] = []
    for b in range(n_bins):
        pos = float(b / max(1, n_bins - 1))
        t = pos * float(m - 1)
        lo = int(math.floor(t))
        hi = int(math.ceil(t))
        if lo == hi:
            out.append(float(vals[lo]))
        else:
            w = float(t - lo)
            out.append(float((1.0 - w) * vals[lo] + w * vals[hi]))
    return out


def svg_hist_stacked(
    out_path: str,
    rows: List[Dict[str, Any]],
    title: str,
    x_label: str,
) -> None:
    if len(rows) == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return
    w, h = 1200, 640
    ml, mr, mt, mb = 80, 40, 90, 80
    pw = w - ml - mr
    ph = h - mt - mb

    x_vals = [int(r["n_gen_tokens"]) for r in rows]
    x_min = int(min(x_vals))
    x_max = int(max(x_vals))
    n_bins = x_max - x_min + 1
    max_cnt = max(int(r["n_total"]) for r in rows)
    if max_cnt <= 0:
        max_cnt = 1

    def x_of(v: int) -> float:
        if n_bins <= 1:
            return float(ml + pw / 2.0)
        return float(ml + ((v - x_min) / max(1, n_bins - 1)) * pw)

    def y_of(c: float) -> float:
        return float(mt + (1.0 - c / max_cnt) * ph)

    bar_w = float(max(2.0, pw / max(1, n_bins) * 0.72))
    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')

    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = max_cnt - max_cnt * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="#ececec" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{int(round(yv))}</text>')

    # bars
    for r in rows:
        x = x_of(int(r["n_gen_tokens"]))
        n_s = int(r["n_success"])
        n_f = int(r["n_failure"])
        y_s = y_of(n_s)
        y_t = y_of(n_s + n_f)
        y0 = y_of(0.0)
        lines.append(
            f'<rect x="{x - bar_w/2:.2f}" y="{y_s:.2f}" width="{bar_w:.2f}" height="{max(0.0, y0 - y_s):.2f}" fill="#4C72B0" fill-opacity="0.85"/>'
        )
        lines.append(
            f'<rect x="{x - bar_w/2:.2f}" y="{y_t:.2f}" width="{bar_w:.2f}" height="{max(0.0, y_s - y_t):.2f}" fill="#D95F5F" fill-opacity="0.85"/>'
        )

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')

    # x ticks
    step = max(1, n_bins // 12)
    for v in range(x_min, x_max + 1, step):
        xx = x_of(v)
        lines.append(f'<line x1="{xx:.2f}" y1="{mt + ph}" x2="{xx:.2f}" y2="{mt + ph + 6}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{mt + ph + 24}" text-anchor="middle" font-size="11" font-family="Arial" fill="#666">{v}</text>')

    lines.append(f'<text x="{ml + pw/2:.2f}" y="{h-24}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">{x_label}</text>')
    lines.append(f'<text x="22" y="{mt + ph/2:.2f}" transform="rotate(-90 22 {mt + ph/2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">count</text>')

    lx, ly = ml + 20, mt + 18
    lines.append(f'<rect x="{lx}" y="{ly-10}" width="16" height="10" fill="#4C72B0" fill-opacity="0.85"/>')
    lines.append(f'<text x="{lx+24}" y="{ly}" font-size="12" font-family="Arial" fill="#4C72B0">correct</text>')
    lines.append(f'<rect x="{lx+110}" y="{ly-10}" width="16" height="10" fill="#D95F5F" fill-opacity="0.85"/>')
    lines.append(f'<text x="{lx+134}" y="{ly}" font-size="12" font-family="Arial" fill="#D95F5F">fail</text>')
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def svg_spaghetti(
    out_path: str,
    curves_success: List[List[float]],
    curves_failure: List[List[float]],
    title: str,
    y_label: str,
    sample_lines_per_group: int = 220,
    seed: int = 42,
) -> None:
    w, h = 1200, 700
    ml, mr, mt, mb = 90, 40, 90, 80
    pw = w - ml - mr
    ph = h - mt - mb

    # Subsample lines for readability.
    rnd = random.Random(int(seed))
    s_curves = list(curves_success)
    f_curves = list(curves_failure)
    rnd.shuffle(s_curves)
    rnd.shuffle(f_curves)
    s_curves = s_curves[: int(max(10, sample_lines_per_group))]
    f_curves = f_curves[: int(max(10, sample_lines_per_group))]

    all_vals: List[float] = []
    for c in s_curves + f_curves:
        all_vals.extend(c)
    if len(all_vals) == 0:
        all_vals = [-1.0, 1.0]
    y_min = float(min(all_vals))
    y_max = float(max(all_vals))
    if y_max <= y_min + 1e-12:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    n = len((s_curves + f_curves)[0]) if (s_curves + f_curves) else 20

    def x_of(i: int) -> float:
        if n <= 1:
            return float(ml + pw / 2.0)
        return float(ml + (i / (n - 1)) * pw)

    def y_of(v: float) -> float:
        return float(mt + (1.0 - (v - y_min) / (y_max - y_min)) * ph)

    def pts(c: List[float]) -> str:
        return " ".join(f"{x_of(i):.2f},{y_of(float(v)):.2f}" for i, v in enumerate(c))

    def mean_curve(curves: List[List[float]]) -> List[float]:
        if len(curves) == 0:
            return [0.0 for _ in range(n)]
        out = []
        for i in range(n):
            out.append(float(sum(c[i] for c in curves) / len(curves)))
        return out

    m_s = mean_curve(s_curves)
    m_f = mean_curve(f_curves)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')

    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = y_max - (y_max - y_min) * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="#ececec" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{yv:.2f}</text>')

    # spaghetti lines
    for c in s_curves:
        lines.append(f'<polyline points="{pts(c)}" fill="none" stroke="#4C72B0" stroke-width="1" stroke-opacity="0.09"/>')
    for c in f_curves:
        lines.append(f'<polyline points="{pts(c)}" fill="none" stroke="#D95F5F" stroke-width="1" stroke-opacity="0.09"/>')

    # means
    lines.append(f'<polyline points="{pts(m_s)}" fill="none" stroke="#1F4F99" stroke-width="3.2"/>')
    lines.append(f'<polyline points="{pts(m_f)}" fill="none" stroke="#B43636" stroke-width="3.2"/>')

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')

    for k in range(6):
        i = int(round((n - 1) * (k / 5.0)))
        xx = x_of(i)
        lines.append(f'<line x1="{xx:.2f}" y1="{mt+ph}" x2="{xx:.2f}" y2="{mt+ph+6}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{mt+ph+24}" text-anchor="middle" font-size="11" font-family="Arial" fill="#666">{k/5.0:.1f}</text>')

    lines.append(f'<text x="{ml + pw/2:.2f}" y="{h-24}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">normalized generation step (t)</text>')
    lines.append(f'<text x="24" y="{mt + ph/2:.2f}" transform="rotate(-90 24 {mt + ph/2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">{y_label}</text>')

    lx, ly = ml + 20, mt + 18
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="#1F4F99" stroke-width="3"/>')
    lines.append(f'<text x="{lx+30}" y="{ly+4}" font-size="12" font-family="Arial" fill="#1F4F99">correct mean</text>')
    lines.append(f'<line x1="{lx+150}" y1="{ly}" x2="{lx+174}" y2="{ly}" stroke="#B43636" stroke-width="3"/>')
    lines.append(f'<text x="{lx+180}" y="{ly+4}" font-size="12" font-family="Arial" fill="#B43636">fail mean</text>')
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Length histogram + per-sample V-PMI trajectory visualization.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--bins", type=int, default=40, help="Interpolation bins for normalized trajectory.")
    ap.add_argument("--spaghetti_lines", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, "analysis_length_vpmi"))
    os.makedirs(out_dir, exist_ok=True)

    per_sample_path = os.path.join(run_dir, "per_sample.csv")
    per_token_path = os.path.join(run_dir, "per_token.csv")

    samples: Dict[str, Dict[str, Any]] = {}
    len_rows: List[Dict[str, Any]] = []
    with open(per_sample_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if str(r.get("error") or "").strip() != "":
                continue
            sid = str(r.get("id") or "")
            n_tok = int(safe_float(r.get("n_gen_tokens")) or 0)
            ok = parse_bool(r.get("is_success"))
            samples[sid] = {"is_success": ok, "n_gen_tokens": n_tok}
            len_rows.append({"id": sid, "n_gen_tokens": n_tok, "is_success": ok})

    tok_map: Dict[str, List[Tuple[int, float]]] = {}
    with open(per_token_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid not in samples:
                continue
            step = int(safe_float(r.get("step_idx")) or 0)
            vpmi = safe_float(r.get("vpmi_logit"))
            if vpmi is None:
                continue
            tok_map.setdefault(sid, []).append((step, float(vpmi)))

    # build trajectories
    curves_s: List[List[float]] = []
    curves_f: List[List[float]] = []
    bins = int(max(8, args.bins))
    for sid, seq in tok_map.items():
        seq = sorted(seq, key=lambda x: int(x[0]))
        vals = [float(v) for _, v in seq]
        if len(vals) == 0:
            continue
        itp = interpolate_curve(vals, bins)
        if bool(samples[sid]["is_success"]):
            curves_s.append(itp)
        else:
            curves_f.append(itp)

    # length histogram rows
    by_len: Dict[int, Dict[str, int]] = {}
    for r in len_rows:
        L = int(r["n_gen_tokens"])
        if L not in by_len:
            by_len[L] = {"n_success": 0, "n_failure": 0}
        if bool(r["is_success"]):
            by_len[L]["n_success"] += 1
        else:
            by_len[L]["n_failure"] += 1
    hist_rows: List[Dict[str, Any]] = []
    for L in sorted(by_len.keys()):
        ns = int(by_len[L]["n_success"])
        nf = int(by_len[L]["n_failure"])
        hist_rows.append(
            {
                "n_gen_tokens": int(L),
                "n_success": ns,
                "n_failure": nf,
                "n_total": int(ns + nf),
                "success_rate": (None if (ns + nf) == 0 else float(ns / (ns + nf))),
            }
        )
    write_csv(os.path.join(out_dir, "length_histogram_counts.csv"), hist_rows)

    # mean trajectory diff summary
    def mean_at(curves: List[List[float]], i: int) -> Optional[float]:
        if len(curves) == 0:
            return None
        return float(sum(float(c[i]) for c in curves) / len(curves))

    curve_rows: List[Dict[str, Any]] = []
    max_abs_row: Optional[Dict[str, Any]] = None
    for i in range(bins):
        ms = mean_at(curves_s, i)
        mf = mean_at(curves_f, i)
        d = (None if ms is None or mf is None else float(ms - mf))
        row = {
            "bin_idx": int(i),
            "pos_norm": float(i / max(1, bins - 1)),
            "vpmi_mean_success": ms,
            "vpmi_mean_failure": mf,
            "vpmi_diff_success_minus_failure": d,
            "n_success": int(len(curves_s)),
            "n_failure": int(len(curves_f)),
        }
        curve_rows.append(row)
        if d is not None:
            if max_abs_row is None or abs(float(d)) > abs(float(max_abs_row["vpmi_diff_success_minus_failure"])):
                max_abs_row = dict(row)
    write_csv(os.path.join(out_dir, "vpmi_curve_correct_vs_fail.csv"), curve_rows)

    # SVGs
    svg_hist_stacked(
        out_path=os.path.join(out_dir, "length_histogram_stacked.svg"),
        rows=hist_rows,
        title="Generated Length Distribution by Label",
        x_label="generated token length",
    )
    svg_spaghetti(
        out_path=os.path.join(out_dir, "vpmi_spaghetti_correct_vs_fail.svg"),
        curves_success=curves_s,
        curves_failure=curves_f,
        title="Per-sample V-PMI Trajectories (Spaghetti + Mean)",
        y_label="V-PMI (logit_vq - logit_q)",
        sample_lines_per_group=int(max(20, args.spaghetti_lines)),
        seed=int(args.seed),
    )

    # simple scalar summaries for quick interpretation
    def avg_segment(curves: List[List[float]], lo: float, hi: float) -> Optional[float]:
        if len(curves) == 0:
            return None
        a = int(math.floor(lo * (bins - 1)))
        b = int(math.ceil(hi * (bins - 1)))
        idxs = list(range(max(0, a), min(bins, b + 1)))
        vals: List[float] = []
        for c in curves:
            vals.extend(float(c[i]) for i in idxs)
        if len(vals) == 0:
            return None
        return float(sum(vals) / len(vals))

    early_s = avg_segment(curves_s, 0.0, 0.2)
    early_f = avg_segment(curves_f, 0.0, 0.2)
    late_s = avg_segment(curves_s, 0.8, 1.0)
    late_f = avg_segment(curves_f, 0.8, 1.0)

    summary = {
        "inputs": {
            "run_dir": run_dir,
            "bins": int(bins),
            "spaghetti_lines_per_group": int(max(20, args.spaghetti_lines)),
        },
        "counts": {
            "n_samples": int(len(samples)),
            "n_success": int(sum(1 for s in samples.values() if bool(s["is_success"]))),
            "n_failure": int(sum(1 for s in samples.values() if not bool(s["is_success"]))),
            "n_trajectories_success": int(len(curves_s)),
            "n_trajectories_failure": int(len(curves_f)),
        },
        "difference_summary": {
            "early_mean_success": early_s,
            "early_mean_failure": early_f,
            "late_mean_success": late_s,
            "late_mean_failure": late_f,
            "early_diff_success_minus_failure": (None if early_s is None or early_f is None else float(early_s - early_f)),
            "late_diff_success_minus_failure": (None if late_s is None or late_f is None else float(late_s - late_f)),
            "max_abs_diff_bin": max_abs_row,
        },
        "outputs": {
            "length_histogram_counts_csv": os.path.join(out_dir, "length_histogram_counts.csv"),
            "vpmi_curve_csv": os.path.join(out_dir, "vpmi_curve_correct_vs_fail.csv"),
            "length_histogram_svg": os.path.join(out_dir, "length_histogram_stacked.svg"),
            "vpmi_spaghetti_svg": os.path.join(out_dir, "vpmi_spaghetti_correct_vs_fail.svg"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "length_histogram_counts.csv"))
    print("[saved]", os.path.join(out_dir, "vpmi_curve_correct_vs_fail.csv"))
    print("[saved]", os.path.join(out_dir, "length_histogram_stacked.svg"))
    print("[saved]", os.path.join(out_dir, "vpmi_spaghetti_correct_vs_fail.svg"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
