#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


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


def svg_bar(
    out_path: str,
    labels: List[str],
    values: List[float],
    title: str,
    y_label: str,
    bar_color: str = "#4C72B0",
    subtitle: Optional[str] = None,
    value_fmt: str = "{:.3f}",
) -> None:
    w, h = 1100, 620
    ml, mr, mt, mb = 90, 40, 95, 90
    pw = w - ml - mr
    ph = h - mt - mb
    n = len(values)
    y_min = 0.0
    y_max = max(values) if len(values) else 1.0
    if y_max <= 0:
        y_max = 1.0
    y_max *= 1.15

    def x_of(i: int) -> float:
        if n <= 1:
            return float(ml + pw / 2.0)
        return float(ml + (i / (n - 1)) * pw)

    def y_of(v: float) -> float:
        return float(mt + (1.0 - (v - y_min) / (y_max - y_min)) * ph)

    bar_w = max(20.0, min(120.0, 0.6 * (pw / max(1, n))))
    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')
    if subtitle:
        lines.append(f'<text x="{ml}" y="62" font-size="14" font-family="Arial" fill="#555">{subtitle}</text>')

    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = y_max - (y_max - y_min) * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+pw}" y2="{yy:.2f}" stroke="#e9e9e9" stroke-width="1"/>')
        lines.append(f'<text x="{ml-10}" y="{yy+5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{yv:.3f}</text>')

    for i, v in enumerate(values):
        xx = x_of(i)
        yv = y_of(v)
        y0 = y_of(0.0)
        lines.append(f'<rect x="{xx-bar_w/2:.2f}" y="{yv:.2f}" width="{bar_w:.2f}" height="{max(0.0,y0-yv):.2f}" fill="{bar_color}" fill-opacity="0.88"/>')
        lines.append(f'<text x="{xx:.2f}" y="{yv-8:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{value_fmt.format(v)}</text>')
        lines.append(f'<text x="{xx:.2f}" y="{y0+24:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#444">{labels[i]}</text>')

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<text x="22" y="{mt+ph/2:.2f}" transform="rotate(-90 22 {mt+ph/2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">{y_label}</text>')
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def svg_line(
    out_path: str,
    xs: List[float],
    ys: List[float],
    title: str,
    x_label: str,
    y_label: str,
    color: str = "#B43636",
    subtitle: Optional[str] = None,
) -> None:
    w, h = 1200, 640
    ml, mr, mt, mb = 90, 40, 95, 85
    pw = w - ml - mr
    ph = h - mt - mb
    if len(xs) == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def x_of(v: float) -> float:
        return float(ml + (v - x_min) / (x_max - x_min) * pw)

    def y_of(v: float) -> float:
        return float(mt + (1.0 - (v - y_min) / (y_max - y_min)) * ph)

    pts = " ".join(f"{x_of(x):.2f},{y_of(y):.2f}" for x, y in zip(xs, ys))
    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')
    if subtitle:
        lines.append(f'<text x="{ml}" y="62" font-size="14" font-family="Arial" fill="#555">{subtitle}</text>')

    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = y_max - (y_max - y_min) * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+pw}" y2="{yy:.2f}" stroke="#e9e9e9" stroke-width="1"/>')
        lines.append(f'<text x="{ml-10}" y="{yy+5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{yv:.3f}</text>')

    lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
    for x, y in zip(xs, ys):
        lines.append(f'<circle cx="{x_of(x):.2f}" cy="{y_of(y):.2f}" r="2.8" fill="{color}"/>')

    # ticks on x
    uniq = sorted(set(int(x) for x in xs))
    step = max(1, len(uniq) // 12)
    for i, xv in enumerate(uniq):
        if i % step != 0 and i != len(uniq) - 1:
            continue
        xx = x_of(float(xv))
        lines.append(f'<line x1="{xx:.2f}" y1="{mt+ph}" x2="{xx:.2f}" y2="{mt+ph+6}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{mt+ph+24:.2f}" text-anchor="middle" font-size="11" font-family="Arial" fill="#444">{xv}</text>')

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<text x="{ml+pw/2:.2f}" y="{h-24}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">{x_label}</text>')
    lines.append(f'<text x="24" y="{mt+ph/2:.2f}" transform="rotate(-90 24 {mt+ph/2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">{y_label}</text>')
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot length-pattern summary charts from existing CSV/JSON.")
    ap.add_argument("--len_quantile_summary_json", type=str, required=True)
    ap.add_argument("--length_hist_counts_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Parse q1/q2/q3 top-feature KS and failure rate
    with open(os.path.abspath(args.len_quantile_summary_json), "r", encoding="utf-8") as f:
        obj = json.load(f)
    buckets = [b for b in obj.get("buckets", []) if str(b.get("bucket", "")).startswith("q")]
    buckets.sort(key=lambda b: str(b.get("bucket", "")))
    labels = [str(b["bucket"]) for b in buckets]
    ks_vals = [float(b.get("top_feature_ks", 0.0) or 0.0) for b in buckets]
    fail_rates = [1.0 - float(b.get("accuracy", 0.0) or 0.0) for b in buckets]

    svg_bar(
        out_path=os.path.join(out_dir, "ks_top_feature_by_len_quantile.svg"),
        labels=labels,
        values=ks_vals,
        title="Top Feature KS by Length Quantile (q1/q2/q3)",
        y_label="KS (top feature per bucket)",
        bar_color="#4C72B0",
        subtitle="source: analysis_len_quantile/summary.json",
        value_fmt="{:.3f}",
    )
    svg_bar(
        out_path=os.path.join(out_dir, "fail_rate_by_len_quantile.svg"),
        labels=labels,
        values=fail_rates,
        title="Failure Rate by Length Quantile (q1/q2/q3)",
        y_label="failure rate",
        bar_color="#D95F5F",
        subtitle="failure_rate = 1 - accuracy",
        value_fmt="{:.3f}",
    )

    # 2) Parse length histogram counts and make fail-rate line by token length
    x_len: List[float] = []
    y_fail: List[float] = []
    rows_export: List[Dict[str, Any]] = []
    with open(os.path.abspath(args.length_hist_counts_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            L = int(float(r.get("n_gen_tokens") or 0))
            n_s = int(float(r.get("n_success") or 0))
            n_f = int(float(r.get("n_failure") or 0))
            n_t = int(float(r.get("n_total") or (n_s + n_f)))
            if n_t <= 0:
                continue
            fr = float(n_f / n_t)
            x_len.append(float(L))
            y_fail.append(fr)
            rows_export.append(
                {
                    "n_gen_tokens": int(L),
                    "n_success": int(n_s),
                    "n_failure": int(n_f),
                    "n_total": int(n_t),
                    "fail_rate": fr,
                }
            )
    x_len, y_fail = zip(*sorted(zip(x_len, y_fail), key=lambda z: z[0])) if len(x_len) > 0 else ([], [])
    x_len = list(x_len)
    y_fail = list(y_fail)

    svg_line(
        out_path=os.path.join(out_dir, "fail_rate_by_token_length.svg"),
        xs=x_len,
        ys=y_fail,
        title="Failure Rate by Generated Token Length",
        x_label="generated token length",
        y_label="failure rate",
        color="#B43636",
        subtitle="source: analysis_length_vpmi/length_histogram_counts.csv",
    )

    write_csv(os.path.join(out_dir, "plot_points_len_pattern.csv"), rows_export)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "len_quantile_summary_json": os.path.abspath(args.len_quantile_summary_json),
                    "length_hist_counts_csv": os.path.abspath(args.length_hist_counts_csv),
                },
                "buckets_q": [
                    {
                        "bucket": l,
                        "top_feature_ks": k,
                        "fail_rate": fr,
                    }
                    for l, k, fr in zip(labels, ks_vals, fail_rates)
                ],
                "outputs": {
                    "ks_top_feature_by_len_quantile_svg": os.path.join(out_dir, "ks_top_feature_by_len_quantile.svg"),
                    "fail_rate_by_len_quantile_svg": os.path.join(out_dir, "fail_rate_by_len_quantile.svg"),
                    "fail_rate_by_token_length_svg": os.path.join(out_dir, "fail_rate_by_token_length.svg"),
                    "plot_points_csv": os.path.join(out_dir, "plot_points_len_pattern.csv"),
                    "summary_json": os.path.join(out_dir, "summary.json"),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[saved]", os.path.join(out_dir, "ks_top_feature_by_len_quantile.svg"))
    print("[saved]", os.path.join(out_dir, "fail_rate_by_len_quantile.svg"))
    print("[saved]", os.path.join(out_dir, "fail_rate_by_token_length.svg"))
    print("[saved]", os.path.join(out_dir, "plot_points_len_pattern.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
