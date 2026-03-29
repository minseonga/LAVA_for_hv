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


def ks_stat(pos: List[float], neg: List[float]) -> Optional[float]:
    if len(pos) == 0 or len(neg) == 0:
        return None
    a = sorted(float(x) for x in pos)
    b = sorted(float(x) for x in neg)
    i = j = 0
    n = len(a)
    m = len(b)
    d = 0.0
    while i < n or j < m:
        va = a[i] if i < n else float("inf")
        vb = b[j] if j < m else float("inf")
        v = va if va <= vb else vb
        while i < n and a[i] <= v:
            i += 1
        while j < m and b[j] <= v:
            j += 1
        d = max(d, abs(float(i) / n - float(j) / m))
    return float(d)


def auc_pos_gt_neg(pos: List[float], neg: List[float]) -> Optional[float]:
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = 0
    eq = 0
    for a in pos:
        for b in neg:
            if a > b:
                gt += 1
            elif a == b:
                eq += 1
    tot = len(pos) * len(neg)
    return float((gt + 0.5 * eq) / float(tot))


def threshold_scan(
    values: List[float],
    labels: List[int],
    direction: str,
) -> Dict[str, Any]:
    # direction: "ge" -> predict success if v>=tau, "le" -> success if v<=tau
    if len(values) == 0:
        return {"best_tau": None, "best_you": None, "precision": None, "coverage": None}
    uniq = sorted(set(float(v) for v in values))
    best = None
    for tau in uniq:
        tp = fp = tn = fn = 0
        for v, y in zip(values, labels):
            pred = (v >= tau) if direction == "ge" else (v <= tau)
            if pred and y == 1:
                tp += 1
            elif pred and y == 0:
                fp += 1
            elif (not pred) and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = float(tp / max(1, tp + fn))
        fpr = float(fp / max(1, fp + tn))
        you = float(tpr - fpr)
        prec = float(tp / max(1, tp + fp))
        cov = float((tp + fp) / max(1, len(values)))
        if best is None or you > best["best_you"]:
            best = {
                "best_tau": float(tau),
                "best_you": float(you),
                "tpr": tpr,
                "fpr": fpr,
                "precision": prec,
                "coverage": cov,
            }
    return best if best is not None else {"best_tau": None, "best_you": None, "precision": None, "coverage": None}


def svg_hist_overlay(
    out_path: str,
    succ: List[float],
    fail: List[float],
    title: str,
    xlabel: str,
    bins: int = 40,
) -> None:
    w, h = 1080, 640
    ml, mr, mt, mb = 80, 40, 90, 72
    pw = w - ml - mr
    ph = h - mt - mb

    vals = [float(x) for x in (succ + fail) if safe_float(x) is not None]
    if len(vals) == 0:
        vals = [-1.0, 1.0]
    x_min = float(min(vals))
    x_max = float(max(vals))
    if x_max <= x_min + 1e-12:
        x_min -= 1.0
        x_max += 1.0
    pad = 0.05 * (x_max - x_min)
    x_min -= pad
    x_max += pad
    step = float((x_max - x_min) / max(1, bins))
    edges = [x_min + i * step for i in range(bins + 1)]

    def density(xs: List[float]) -> List[float]:
        d = [0.0 for _ in range(bins)]
        if len(xs) == 0:
            return d
        for v in xs:
            for i in range(bins):
                lo = edges[i]
                hi = edges[i + 1]
                if (v >= lo and v < hi) or (i == bins - 1 and v == hi):
                    d[i] += 1.0
                    break
        n = float(len(xs))
        return [float(x / n) for x in d]

    ds = density([float(x) for x in succ if safe_float(x) is not None])
    df = density([float(x) for x in fail if safe_float(x) is not None])
    y_max = max([0.001] + ds + df) * 1.08

    def x_of(x: float) -> float:
        return float(ml + (x - x_min) / (x_max - x_min) * pw)

    def y_of(y: float) -> float:
        return float(mt + (1.0 - y / y_max) * ph)

    def poly(d: List[float]) -> str:
        pts: List[str] = []
        for i, yy in enumerate(d):
            xc = 0.5 * (edges[i] + edges[i + 1])
            pts.append(f"{x_of(xc):.2f},{y_of(float(yy)):.2f}")
        return " ".join(pts)

    ps = poly(ds)
    pf = poly(df)
    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')

    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = y_max - y_max * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="#e9e9e9" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{yv:.3f}</text>')
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<text x="{ml + pw / 2:.2f}" y="{h - 22}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">{xlabel}</text>')
    lines.append(f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">density</text>')
    if ps:
        lines.append(f'<polyline points="{ps}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    if pf:
        lines.append(f'<polyline points="{pf}" fill="none" stroke="#d62728" stroke-width="3"/>')

    lx, ly = ml + 20, mt + 20
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 36}" y2="{ly}" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 44}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#1f77b4">correct</text>')
    lines.append(f'<line x1="{lx + 130}" y1="{ly}" x2="{lx + 166}" y2="{ly}" stroke="#d62728" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 174}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#d62728">fail</text>')
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze outputs from analyze_greedy_token_profile.py")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--hist_bins", type=int, default=40)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, "analysis_viz"))
    os.makedirs(out_dir, exist_ok=True)

    per_sample_path = os.path.join(run_dir, "per_sample.csv")
    curve_vpmi_path = os.path.join(run_dir, "curve_vpmi_logit_correct_vs_fail.csv")
    curve_rank_path = os.path.join(run_dir, "curve_rankpct_correct_vs_fail.csv")

    per_sample: List[Dict[str, Any]] = []
    with open(per_sample_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if str(r.get("error") or "").strip() != "":
                continue
            per_sample.append(r)

    y = [1 if parse_bool(r.get("is_success")) else 0 for r in per_sample]
    features = [
        "vpmi_logit_mean",
        "vpmi_collapse_gap_k40",
        "vpmi_prefix_mean_k40",
        "vpmi_suffix_min_k40",
        "rank_pct_mean",
        "rank_pct_min",
        "n_gen_tokens",
    ]
    ranking_rows: List[Dict[str, Any]] = []
    for fn in features:
        vals = [safe_float(r.get(fn)) for r in per_sample]
        idx = [i for i, v in enumerate(vals) if v is not None]
        if len(idx) == 0:
            continue
        xs = [float(vals[i]) for i in idx]
        ys = [int(y[i]) for i in idx]
        pos = [x for x, yy in zip(xs, ys) if yy == 1]
        neg = [x for x, yy in zip(xs, ys) if yy == 0]
        ks = ks_stat(pos, neg)
        auc = auc_pos_gt_neg(pos, neg)
        ge = threshold_scan(xs, ys, "ge")
        le = threshold_scan(xs, ys, "le")
        best = ge if (ge.get("best_you") or -1e9) >= (le.get("best_you") or -1e9) else le
        direction = "ge" if best is ge else "le"
        ranking_rows.append(
            {
                "feature": fn,
                "ks": ks,
                "auc_pos_gt_neg": auc,
                "mean_success": (None if len(pos) == 0 else float(sum(pos) / len(pos))),
                "mean_failure": (None if len(neg) == 0 else float(sum(neg) / len(neg))),
                "median_success": quantile(pos, 0.5),
                "median_failure": quantile(neg, 0.5),
                "best_direction": direction,
                "best_tau": best.get("best_tau"),
                "best_youden": best.get("best_you"),
                "best_tpr": best.get("tpr"),
                "best_fpr": best.get("fpr"),
                "best_precision": best.get("precision"),
                "best_coverage": best.get("coverage"),
                "n": int(len(idx)),
            }
        )
    ranking_rows.sort(key=lambda r: float(r.get("ks") or -1.0), reverse=True)
    write_csv(os.path.join(out_dir, "feature_split_summary.csv"), ranking_rows)

    # Curve diagnostics.
    vpmi_rows: List[Dict[str, Any]] = []
    with open(curve_vpmi_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            vpmi_rows.append(r)
    rank_rows: List[Dict[str, Any]] = []
    with open(curve_rank_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rank_rows.append(r)

    max_abs_vpmi = None
    if len(vpmi_rows) > 0:
        max_abs_vpmi = max(
            vpmi_rows,
            key=lambda r: abs(float(r.get("vpmi_logit_diff_success_minus_failure") or 0.0)),
        )
    max_abs_rank = None
    if len(rank_rows) > 0:
        max_abs_rank = max(
            rank_rows,
            key=lambda r: abs(float(r.get("rank_pct_diff_success_minus_failure") or 0.0)),
        )

    # Histograms for top 3 KS features.
    top3 = [r["feature"] for r in ranking_rows[:3]]
    for fn in top3:
        succ = [float(r.get(fn)) for r in per_sample if parse_bool(r.get("is_success")) and safe_float(r.get(fn)) is not None]
        fail = [float(r.get(fn)) for r in per_sample if (not parse_bool(r.get("is_success"))) and safe_float(r.get(fn)) is not None]
        svg_hist_overlay(
            out_path=os.path.join(out_dir, f"hist_{fn}.svg"),
            succ=succ,
            fail=fail,
            title=f"Feature Distribution: {fn}",
            xlabel=fn,
            bins=int(max(10, args.hist_bins)),
        )

    summary = {
        "inputs": {
            "run_dir": run_dir,
            "n_valid_samples": int(len(per_sample)),
        },
        "top_features_by_ks": ranking_rows[:10],
        "curve_key_points": {
            "vpmi_max_abs_diff_bin": max_abs_vpmi,
            "rank_max_abs_diff_bin": max_abs_rank,
        },
        "outputs": {
            "feature_split_summary_csv": os.path.join(out_dir, "feature_split_summary.csv"),
            "hist_svgs": [os.path.join(out_dir, f"hist_{fn}.svg") for fn in top3],
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "feature_split_summary.csv"))
    for fn in top3:
        print("[saved]", os.path.join(out_dir, f"hist_{fn}.svg"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
