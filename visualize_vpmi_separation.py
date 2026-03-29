#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return None
    qq = min(1.0, max(0.0, float(q)))
    if len(xs) == 1:
        return float(xs[0])
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def svg_header(w: int, h: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'


def esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def hist_density(vals: Sequence[float], lo: float, hi: float, n_bins: int) -> List[float]:
    if n_bins <= 0:
        return []
    if hi <= lo:
        hi = lo + 1e-9
    bins = [0.0] * n_bins
    for v in vals:
        if not math.isfinite(float(v)):
            continue
        x = float(v)
        if x < lo or x > hi:
            continue
        t = (x - lo) / (hi - lo)
        k = int(min(n_bins - 1, max(0, math.floor(t * n_bins))))
        bins[k] += 1.0
    s = sum(bins)
    if s <= 0:
        return bins
    return [b / s for b in bins]


def read_dataset(in_dir: str, safe_mode_for_margin: str = "max_vpmi") -> Dict[str, Any]:
    per_sample_path = os.path.join(in_dir, "per_sample.csv")
    per_cand_path = os.path.join(in_dir, "per_candidate.csv")

    per_sample = list(csv.DictReader(open(per_sample_path, encoding="utf-8")))
    per_cand = list(csv.DictReader(open(per_cand_path, encoding="utf-8")))

    meta: Dict[str, Dict[str, Any]] = {}
    for r in per_sample:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", ""))
        if sid == "":
            continue
        meta[sid] = {
            "is_success": as_bool(r.get("is_success", "false")),
        }

    cand_by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_cand:
        sid = str(r.get("id", ""))
        if sid not in meta:
            continue
        idx = safe_float(r.get("cand_idx"))
        if idx is None:
            continue
        s_full = safe_float(r.get("s_full"))
        s_q = safe_float(r.get("s_ans_q"))
        s_core = safe_float(r.get("s_core_img"))
        vpmi = None if s_core is None or s_q is None else float(s_core - s_q)
        cand_by_sid[sid].append(
            {
                "idx": int(idx),
                "is_champion": as_bool(r.get("is_champion", "false")),
                "is_safe_existing": as_bool(r.get("is_safe", "false")),
                "s_full": s_full,
                "s_q": s_q,
                "s_core": s_core,
                "vpmi": vpmi,
            }
        )

    rows: List[Dict[str, Any]] = []
    for sid, m in meta.items():
        cands = cand_by_sid.get(sid, [])
        if not cands:
            continue
        champ = next((c for c in cands if c.get("is_champion", False)), None)
        if champ is None:
            pool = [c for c in cands if c.get("s_full") is not None]
            if not pool:
                continue
            champ = max(pool, key=lambda x: float(x["s_full"]))

        pool = [c for c in cands if int(c["idx"]) != int(champ["idx"])]
        safe = None
        if safe_mode_for_margin == "existing_safe":
            safe = next((c for c in cands if c.get("is_safe_existing", False)), None)
        elif safe_mode_for_margin == "max_visual_pmi":
            p = [c for c in pool if c.get("s_full") is not None and c.get("s_q") is not None]
            if p:
                safe = max(p, key=lambda x: float(x["s_full"] - x["s_q"]))
        else:  # max_vpmi
            p = [c for c in pool if c.get("vpmi") is not None]
            if p:
                safe = max(p, key=lambda x: float(x["vpmi"]))

        champ_v = champ.get("vpmi")
        safe_v = (None if safe is None else safe.get("vpmi"))
        p3_margin = None
        if champ_v is not None and safe_v is not None:
            p3_margin = float(safe_v - champ_v)
        p3_cond = bool(champ_v is not None and safe_v is not None and safe_v > champ_v and champ_v < 0.0)
        p5_cond = bool(p3_cond and safe_v is not None and safe_v > 0.0)

        rows.append(
            {
                "sid": sid,
                "wrong": (0 if bool(m["is_success"]) else 1),
                "champ_vpmi": champ_v,
                "p3_margin": p3_margin,
                "p3_cond": p3_cond,
                "p5_cond": p5_cond,
            }
        )

    return {
        "rows": rows,
        "n": len(rows),
        "n_wrong": int(sum(r["wrong"] for r in rows)),
        "n_correct": int(sum(1 for r in rows if r["wrong"] == 0)),
    }


def draw_hist_panel(
    parts: List[str],
    x0: float,
    y0: float,
    w: float,
    h: float,
    vals_correct: List[float],
    vals_wrong: List[float],
    title: str,
    n_bins: int = 36,
) -> None:
    all_vals = [v for v in vals_correct + vals_wrong if v is not None and math.isfinite(float(v))]
    if len(all_vals) == 0:
        return
    q01 = quantile(all_vals, 0.01)
    q99 = quantile(all_vals, 0.99)
    if q01 is None or q99 is None:
        return
    lo = float(q01)
    hi = float(q99)
    if hi <= lo:
        lo = float(min(all_vals))
        hi = float(max(all_vals))
        if hi <= lo:
            hi = lo + 1e-6

    hc = hist_density(vals_correct, lo, hi, n_bins=n_bins)
    hw = hist_density(vals_wrong, lo, hi, n_bins=n_bins)
    ymax = max(max(hc) if hc else 0.0, max(hw) if hw else 0.0, 1e-6)

    left = x0 + 46
    right = x0 + w - 14
    top = y0 + 24
    bottom = y0 + h - 34
    pw = max(1.0, (right - left) / n_bins)

    parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#ffffff" stroke="#d0d7de"/>')
    parts.append(f'<text x="{(x0 + w / 2):.2f}" y="{(y0 + 16):.2f}" text-anchor="middle" font-size="12" fill="#111827">{esc(title)}</text>')

    # axes
    parts.append(f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}" stroke="#6b7280" stroke-width="1"/>')
    parts.append(f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}" stroke="#6b7280" stroke-width="1"/>')

    # y ticks
    for t in [0.0, 0.5, 1.0]:
        y = bottom - (bottom - top) * t
        label = f"{ymax * t:.2f}"
        parts.append(f'<line x1="{left - 3:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}" stroke="#6b7280" stroke-width="1"/>')
        parts.append(f'<text x="{left - 6:.2f}" y="{(y + 4):.2f}" text-anchor="end" font-size="9" fill="#4b5563">{esc(label)}</text>')

    # x ticks
    for t in [0.0, 0.5, 1.0]:
        xv = lo + (hi - lo) * t
        x = left + (right - left) * t
        parts.append(f'<line x1="{x:.2f}" y1="{bottom:.2f}" x2="{x:.2f}" y2="{bottom + 3:.2f}" stroke="#6b7280" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom + 14:.2f}" text-anchor="middle" font-size="9" fill="#4b5563">{xv:.2f}</text>')

    # bars (left half: correct, right half: wrong)
    for i in range(n_bins):
        x = left + i * pw
        hc_i = hc[i] if i < len(hc) else 0.0
        hw_i = hw[i] if i < len(hw) else 0.0
        h_c = (bottom - top) * (hc_i / ymax)
        h_w = (bottom - top) * (hw_i / ymax)
        # correct
        parts.append(
            f'<rect x="{x + 0.08 * pw:.2f}" y="{(bottom - h_c):.2f}" width="{0.38 * pw:.2f}" height="{h_c:.2f}" fill="#2b6cb0" fill-opacity="0.55"/>'
        )
        # wrong
        parts.append(
            f'<rect x="{x + 0.54 * pw:.2f}" y="{(bottom - h_w):.2f}" width="{0.38 * pw:.2f}" height="{h_w:.2f}" fill="#c53030" fill-opacity="0.55"/>'
        )


def draw_activation_panel(
    parts: List[str],
    x0: float,
    y0: float,
    w: float,
    h: float,
    dataset_name: str,
    rate_c_p3: float,
    rate_w_p3: float,
    rate_c_p5: float,
    rate_w_p5: float,
) -> None:
    parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#ffffff" stroke="#d0d7de"/>')
    parts.append(f'<text x="{(x0 + w / 2):.2f}" y="{(y0 + 16):.2f}" text-anchor="middle" font-size="12" fill="#111827">{esc(dataset_name)}: P3/P5 activation rate</text>')

    left = x0 + 42
    right = x0 + w - 12
    top = y0 + 24
    bottom = y0 + h - 26
    parts.append(f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}" stroke="#6b7280" stroke-width="1"/>')
    parts.append(f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}" stroke="#6b7280" stroke-width="1"/>')

    for t in [0.0, 0.25, 0.5]:
        y = bottom - (bottom - top) * (t / 0.5)
        parts.append(f'<line x1="{left - 3:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}" stroke="#6b7280" stroke-width="1"/>')
        parts.append(f'<text x="{left - 6:.2f}" y="{(y + 4):.2f}" text-anchor="end" font-size="9" fill="#4b5563">{t:.2f}</text>')

    labels = ["P3", "P5"]
    vals_c = [rate_c_p3, rate_c_p5]
    vals_w = [rate_w_p3, rate_w_p5]
    g_w = (right - left) / len(labels)
    for i, lab in enumerate(labels):
        gx = left + i * g_w
        bw = g_w * 0.26
        hc = (bottom - top) * min(1.0, max(0.0, vals_c[i] / 0.5))
        hw = (bottom - top) * min(1.0, max(0.0, vals_w[i] / 0.5))
        parts.append(f'<rect x="{gx + 0.20 * g_w:.2f}" y="{(bottom - hc):.2f}" width="{bw:.2f}" height="{hc:.2f}" fill="#2b6cb0" fill-opacity="0.75"/>')
        parts.append(f'<rect x="{gx + 0.54 * g_w:.2f}" y="{(bottom - hw):.2f}" width="{bw:.2f}" height="{hw:.2f}" fill="#c53030" fill-opacity="0.75"/>')
        parts.append(f'<text x="{gx + 0.50 * g_w:.2f}" y="{bottom + 13:.2f}" text-anchor="middle" font-size="9" fill="#374151">{lab}</text>')
        parts.append(f'<text x="{gx + 0.33 * g_w:.2f}" y="{(bottom - hc - 3):.2f}" text-anchor="middle" font-size="8" fill="#1e3a8a">{vals_c[i]:.3f}</text>')
        parts.append(f'<text x="{gx + 0.67 * g_w:.2f}" y="{(bottom - hw - 3):.2f}" text-anchor="middle" font-size="8" fill="#7f1d1d">{vals_w[i]:.3f}</text>')


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize V-PMI separation between correct/wrong for GQA and POPE")
    ap.add_argument("--gqa_dir", type=str, required=True)
    ap.add_argument("--pope_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--safe_mode_for_margin", type=str, default="max_vpmi", choices=["max_vpmi", "max_visual_pmi", "existing_safe"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    datasets = [
        ("GQA-1000", read_dataset(args.gqa_dir, safe_mode_for_margin=args.safe_mode_for_margin)),
        ("POPE-1000", read_dataset(args.pope_dir, safe_mode_for_margin=args.safe_mode_for_margin)),
    ]

    # Save numeric summary too.
    summary_rows: List[Dict[str, Any]] = []
    for name, data in datasets:
        rows = data["rows"]
        c = [r for r in rows if r["wrong"] == 0]
        w = [r for r in rows if r["wrong"] == 1]
        rate_c_p3 = (sum(1 for r in c if r["p3_cond"]) / len(c)) if c else 0.0
        rate_w_p3 = (sum(1 for r in w if r["p3_cond"]) / len(w)) if w else 0.0
        rate_c_p5 = (sum(1 for r in c if r["p5_cond"]) / len(c)) if c else 0.0
        rate_w_p5 = (sum(1 for r in w if r["p5_cond"]) / len(w)) if w else 0.0
        summary_rows.append(
            {
                "dataset": name,
                "n": data["n"],
                "n_correct": data["n_correct"],
                "n_wrong": data["n_wrong"],
                "p3_rate_correct": rate_c_p3,
                "p3_rate_wrong": rate_w_p3,
                "p5_rate_correct": rate_c_p5,
                "p5_rate_wrong": rate_w_p5,
                "p3_wrong_over_correct": (None if rate_c_p3 == 0 else rate_w_p3 / rate_c_p3),
                "p5_wrong_over_correct": (None if rate_c_p5 == 0 else rate_w_p5 / rate_c_p5),
            }
        )
    summary_csv = os.path.join(args.out_dir, "vpmi_separation_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader()
        wr.writerows(summary_rows)

    # Figure 1: feature histograms
    w, h = 1240, 740
    parts: List[str] = [svg_header(w, h)]
    parts.append('<rect x="0" y="0" width="1240" height="740" fill="#f8fafc"/>')
    parts.append('<text x="620" y="26" text-anchor="middle" font-size="16" fill="#111827">V-PMI Feature Distributions: Correct vs Wrong</text>')
    parts.append('<rect x="980" y="42" width="12" height="12" fill="#2b6cb0" fill-opacity="0.55"/>')
    parts.append('<text x="998" y="52" font-size="11" fill="#1f2937">correct</text>')
    parts.append('<rect x="1074" y="42" width="12" height="12" fill="#c53030" fill-opacity="0.55"/>')
    parts.append('<text x="1092" y="52" font-size="11" fill="#1f2937">wrong</text>')

    panel_w = 590
    panel_h = 320
    x_left = 20
    x_right = 630
    y_top = 70
    y_bottom = 400

    for row_i, (name, data) in enumerate(datasets):
        yy = y_top if row_i == 0 else y_bottom
        rows = data["rows"]
        vc = [float(r["champ_vpmi"]) for r in rows if r["wrong"] == 0 and r["champ_vpmi"] is not None]
        vw = [float(r["champ_vpmi"]) for r in rows if r["wrong"] == 1 and r["champ_vpmi"] is not None]
        mc = [float(r["p3_margin"]) for r in rows if r["wrong"] == 0 and r["p3_margin"] is not None]
        mw = [float(r["p3_margin"]) for r in rows if r["wrong"] == 1 and r["p3_margin"] is not None]

        draw_hist_panel(parts, x_left, yy, panel_w, panel_h, vc, vw, f"{name} | champ_vpmi")
        draw_hist_panel(parts, x_right, yy, panel_w, panel_h, mc, mw, f"{name} | p3_margin (safe_vpmi - champ_vpmi)")

    parts.append("</svg>")
    hist_svg = os.path.join(args.out_dir, "vpmi_correct_wrong_hist.svg")
    with open(hist_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    # Figure 2: activation rates
    w2, h2 = 1240, 360
    parts2: List[str] = [svg_header(w2, h2)]
    parts2.append('<rect x="0" y="0" width="1240" height="360" fill="#f8fafc"/>')
    parts2.append('<text x="620" y="24" text-anchor="middle" font-size="16" fill="#111827">P3 / P5 Condition Activation Rate (safe mode for margin: max_vpmi)</text>')
    parts2.append('<rect x="1010" y="34" width="12" height="12" fill="#2b6cb0" fill-opacity="0.75"/>')
    parts2.append('<text x="1028" y="44" font-size="11" fill="#1f2937">correct</text>')
    parts2.append('<rect x="1100" y="34" width="12" height="12" fill="#c53030" fill-opacity="0.75"/>')
    parts2.append('<text x="1118" y="44" font-size="11" fill="#1f2937">wrong</text>')

    for i, (name, data) in enumerate(datasets):
        rows = data["rows"]
        c = [r for r in rows if r["wrong"] == 0]
        wrows = [r for r in rows if r["wrong"] == 1]
        rate_c_p3 = (sum(1 for r in c if r["p3_cond"]) / len(c)) if c else 0.0
        rate_w_p3 = (sum(1 for r in wrows if r["p3_cond"]) / len(wrows)) if wrows else 0.0
        rate_c_p5 = (sum(1 for r in c if r["p5_cond"]) / len(c)) if c else 0.0
        rate_w_p5 = (sum(1 for r in wrows if r["p5_cond"]) / len(wrows)) if wrows else 0.0
        draw_activation_panel(parts2, 20 + i * 610, 56, 590, 280, name, rate_c_p3, rate_w_p3, rate_c_p5, rate_w_p5)

    parts2.append("</svg>")
    act_svg = os.path.join(args.out_dir, "vpmi_activation_rates.svg")
    with open(act_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts2))

    print("[saved]", hist_svg)
    print("[saved]", act_svg)
    print("[saved]", summary_csv)


if __name__ == "__main__":
    main()
