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


def ks_stat(pos: List[float], neg: List[float]) -> Optional[float]:
    if len(pos) == 0 or len(neg) == 0:
        return None
    a = sorted(float(x) for x in pos)
    b = sorted(float(x) for x in neg)
    i = j = 0
    n, m = len(a), len(b)
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
    # Mann-Whitney U / rank-sum based AUC (O((n+m)log(n+m))).
    n_pos = len(pos)
    n_neg = len(neg)
    arr: List[Tuple[float, int]] = [(float(v), 1) for v in pos] + [(float(v), 0) for v in neg]
    arr.sort(key=lambda x: x[0])

    rank_sum_pos = 0.0
    i = 0
    n = len(arr)
    while i < n:
        j = i + 1
        while j < n and arr[j][0] == arr[i][0]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # 1-indexed average rank on ties
        for k in range(i, j):
            if arr[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j

    u = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / float(n_pos * n_neg))


def pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = float(sum(xs) / len(xs))
    my = float(sum(ys) / len(ys))
    vx = float(sum((x - mx) * (x - mx) for x in xs))
    vy = float(sum((y - my) * (y - my) for y in ys))
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)))
    return float(cov / math.sqrt(vx * vy))


def slope_over_normalized_steps(vals: List[float]) -> Optional[float]:
    n = len(vals)
    if n < 2:
        return None
    xs = [float(i / (n - 1)) for i in range(n)]
    mx = float(sum(xs) / n)
    my = float(sum(vals) / n)
    vx = float(sum((x - mx) * (x - mx) for x in xs))
    if vx <= 0.0:
        return None
    cov = float(sum((x - mx) * (y - my) for x, y in zip(xs, vals)))
    return float(cov / vx)


def load_rows(run_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    per_sample_path = os.path.join(run_dir, "per_sample.csv")
    per_token_path = os.path.join(run_dir, "per_token.csv")

    samples: List[Dict[str, Any]] = []
    with open(per_sample_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if str(r.get("error") or "").strip() != "":
                continue
            samples.append(r)

    tok_map: Dict[str, List[Dict[str, Any]]] = {}
    with open(per_token_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            tok_map.setdefault(sid, []).append(r)

    for sid in tok_map:
        tok_map[sid].sort(key=lambda r: int(float(r.get("step_idx") or 0.0)))
    return samples, tok_map


def build_feature_table(samples: List[Dict[str, Any]], tok_map: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in samples:
        sid = str(s.get("id") or "")
        toks = tok_map.get(sid, [])
        vpmi = [safe_float(t.get("vpmi_logit")) for t in toks]
        vpmi = [float(v) for v in vpmi if v is not None]
        rank = [safe_float(t.get("vpmi_rank_pct_topk")) for t in toks]
        rank = [float(v) for v in rank if v is not None]

        # New morphology features (length-invariant by design)
        slope = slope_over_normalized_steps(vpmi)
        inversion_ratio = None
        if len(vpmi) > 0:
            inversion_ratio = float(sum(1 for v in vpmi if v < 0.0) / len(vpmi))
        rank_drop = None
        if len(rank) > 0:
            rank_drop = float(max(rank) - min(rank))

        rows.append(
            {
                "id": sid,
                "is_success": 1 if parse_bool(s.get("is_success")) else 0,
                "n_gen_tokens": safe_float(s.get("n_gen_tokens")),
                "vpmi_logit_mean": safe_float(s.get("vpmi_logit_mean")),
                "vpmi_prefix_mean_k40": safe_float(s.get("vpmi_prefix_mean_k40")),
                "vpmi_suffix_min_k40": safe_float(s.get("vpmi_suffix_min_k40")),
                "vpmi_collapse_gap_k40": safe_float(s.get("vpmi_collapse_gap_k40")),
                "rank_pct_mean": safe_float(s.get("rank_pct_mean")),
                "rank_pct_min": safe_float(s.get("rank_pct_min")),
                "morph_vpmi_slope": slope,
                "morph_vpmi_inversion_ratio": inversion_ratio,
                "morph_rank_drop": rank_drop,
            }
        )
    return rows


def bootstrap_ci(
    xs: List[float],
    ys: List[int],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    pos = [x for x, y in zip(xs, ys) if y == 1]
    neg = [x for x, y in zip(xs, ys) if y == 0]
    ks0 = ks_stat(pos, neg)
    auc0 = auc_pos_gt_neg(pos, neg)
    if len(pos) == 0 or len(neg) == 0:
        return {
            "ks": ks0,
            "ks_ci_lo": None,
            "ks_ci_hi": None,
            "auc": auc0,
            "auc_ci_lo": None,
            "auc_ci_hi": None,
        }

    rng = random.Random(int(seed))
    ks_bs: List[float] = []
    auc_bs: List[float] = []
    n_pos = len(pos)
    n_neg = len(neg)
    for _ in range(int(n_boot)):
        b_pos = [pos[rng.randrange(n_pos)] for _ in range(n_pos)]
        b_neg = [neg[rng.randrange(n_neg)] for _ in range(n_neg)]
        k = ks_stat(b_pos, b_neg)
        a = auc_pos_gt_neg(b_pos, b_neg)
        if k is not None:
            ks_bs.append(float(k))
        if a is not None:
            auc_bs.append(float(a))

    return {
        "ks": ks0,
        "ks_ci_lo": quantile(ks_bs, 0.025),
        "ks_ci_hi": quantile(ks_bs, 0.975),
        "auc": auc0,
        "auc_ci_lo": quantile(auc_bs, 0.025),
        "auc_ci_hi": quantile(auc_bs, 0.975),
        "n_boot_effective_ks": int(len(ks_bs)),
        "n_boot_effective_auc": int(len(auc_bs)),
    }


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
    ap = argparse.ArgumentParser(description="Bootstrap CI for KS/AUC + morphology features.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--n_boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hist_bins", type=int, default=40)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, "analysis_bootstrap_ci"))
    os.makedirs(out_dir, exist_ok=True)

    samples, tok_map = load_rows(run_dir)
    table = build_feature_table(samples, tok_map)

    features = [
        "vpmi_logit_mean",
        "vpmi_prefix_mean_k40",
        "vpmi_collapse_gap_k40",
        "rank_pct_mean",
        "rank_pct_min",
        "morph_vpmi_slope",
        "morph_vpmi_inversion_ratio",
        "morph_rank_drop",
    ]

    # Prepare rows for metrics.
    metrics_rows: List[Dict[str, Any]] = []
    for fn in features:
        vals = [safe_float(r.get(fn)) for r in table]
        lens = [safe_float(r.get("n_gen_tokens")) for r in table]
        ys = [int(r.get("is_success", 0)) for r in table]
        idx = [i for i, v in enumerate(vals) if v is not None]
        if len(idx) == 0:
            continue
        xs = [float(vals[i]) for i in idx]
        yl = [int(ys[i]) for i in idx]
        lz = [float(lens[i]) for i in idx if lens[i] is not None]
        xz = [float(vals[i]) for i in idx if lens[i] is not None]
        pos = [x for x, y in zip(xs, yl) if y == 1]
        neg = [x for x, y in zip(xs, yl) if y == 0]

        ci = bootstrap_ci(xs, yl, n_boot=int(args.n_boot), seed=int(args.seed) + (abs(hash(fn)) % 10000))
        metrics_rows.append(
            {
                "feature": fn,
                "n": int(len(xs)),
                "n_success": int(sum(1 for y in yl if y == 1)),
                "n_failure": int(sum(1 for y in yl if y == 0)),
                "mean_success": (None if len(pos) == 0 else float(sum(pos) / len(pos))),
                "mean_failure": (None if len(neg) == 0 else float(sum(neg) / len(neg))),
                "median_success": quantile(pos, 0.5),
                "median_failure": quantile(neg, 0.5),
                "ks": ci.get("ks"),
                "ks_ci_lo": ci.get("ks_ci_lo"),
                "ks_ci_hi": ci.get("ks_ci_hi"),
                "auc": ci.get("auc"),
                "auc_ci_lo": ci.get("auc_ci_lo"),
                "auc_ci_hi": ci.get("auc_ci_hi"),
                "auc_ci_excludes_0_5": (
                    None
                    if ci.get("auc_ci_lo") is None or ci.get("auc_ci_hi") is None
                    else bool(float(ci["auc_ci_lo"]) > 0.5 or float(ci["auc_ci_hi"]) < 0.5)
                ),
                "pearson_with_len": pearson_corr(xz, lz),
            }
        )

    metrics_rows.sort(key=lambda r: float(r.get("ks") or -1.0), reverse=True)
    write_csv(os.path.join(out_dir, "feature_bootstrap_ci.csv"), metrics_rows)

    # Build histograms for the three requested morphology features.
    morph_feats = ["morph_vpmi_slope", "morph_vpmi_inversion_ratio", "morph_rank_drop"]
    for fn in morph_feats:
        succ = [float(r[fn]) for r in table if r.get(fn) is not None and int(r["is_success"]) == 1]
        fail = [float(r[fn]) for r in table if r.get(fn) is not None and int(r["is_success"]) == 0]
        svg_hist_overlay(
            out_path=os.path.join(out_dir, f"hist_{fn}.svg"),
            succ=succ,
            fail=fail,
            title=f"Morphology Feature Distribution: {fn}",
            xlabel=fn,
            bins=int(max(10, args.hist_bins)),
        )

    summary = {
        "inputs": {
            "run_dir": run_dir,
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
        },
        "top_by_ks": metrics_rows[:10],
        "morphology_features": [r for r in metrics_rows if str(r.get("feature", "")).startswith("morph_")],
        "outputs": {
            "feature_bootstrap_ci_csv": os.path.join(out_dir, "feature_bootstrap_ci.csv"),
            "hist_svgs": [os.path.join(out_dir, f"hist_{fn}.svg") for fn in morph_feats],
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "feature_bootstrap_ci.csv"))
    for fn in morph_feats:
        print("[saved]", os.path.join(out_dir, f"hist_{fn}.svg"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
