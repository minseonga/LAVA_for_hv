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


def threshold_scan(values: List[float], labels: List[int], direction: str) -> Dict[str, Any]:
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


def svg_line_chart(
    out_path: str,
    y_success: List[Optional[float]],
    y_failure: List[Optional[float]],
    title: str,
    subtitle: str,
    y_label: str,
) -> None:
    w, h = 1080, 640
    ml, mr, mt, mb = 86, 40, 92, 72
    pw = w - ml - mr
    ph = h - mt - mb

    vals = [float(v) for v in (y_success + y_failure) if v is not None and safe_float(v) is not None]
    if len(vals) == 0:
        vals = [-1.0, 1.0]
    y_min = float(min(vals))
    y_max = float(max(vals))
    if y_max <= y_min + 1e-12:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    n = int(max(1, len(y_success)))

    def x_of(i: int) -> float:
        if n <= 1:
            return float(ml + pw / 2.0)
        return float(ml + (float(i) / float(n - 1)) * pw)

    def y_of(v: float) -> float:
        return float(mt + (1.0 - (float(v) - y_min) / (y_max - y_min)) * ph)

    def poly(ys: List[Optional[float]]) -> str:
        pts: List[str] = []
        for i, v in enumerate(ys):
            if v is None:
                continue
            pts.append(f"{x_of(i):.2f},{y_of(float(v)):.2f}")
        return " ".join(pts)

    p_s = poly(y_success)
    p_f = poly(y_failure)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')
    lines.append(f'<text x="{ml}" y="63" font-size="15" font-family="Arial" fill="#444">{subtitle}</text>')

    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = y_max - (y_max - y_min) * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="#e9e9e9" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{yv:.3f}</text>')
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<text x="{ml + pw / 2:.2f}" y="{h - 22}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">normalized step (prefix → suffix)</text>')
    lines.append(f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">{y_label}</text>')
    if p_s:
        lines.append(f'<polyline points="{p_s}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    if p_f:
        lines.append(f'<polyline points="{p_f}" fill="none" stroke="#d62728" stroke-width="3"/>')
    lx, ly = ml + 20, mt + 20
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 36}" y2="{ly}" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 44}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#1f77b4">correct</text>')
    lines.append(f'<line x1="{lx + 130}" y1="{ly}" x2="{lx + 166}" y2="{ly}" stroke="#d62728" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 174}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#d62728">fail</text>')
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def bucket_name(n_tok: int) -> str:
    if n_tok <= 0:
        return "len_0"
    if n_tok == 1:
        return "len_1"
    if n_tok <= 3:
        return "len_2_3"
    return "len_4_plus"


def main() -> None:
    ap = argparse.ArgumentParser(description="Length-bucket analysis for greedy token profile outputs.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--curve_bins", type=int, default=20)
    ap.add_argument("--bucket_mode", type=str, default="fixed", choices=["fixed", "quantile"])
    ap.add_argument("--quantile_buckets", type=int, default=3)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, "analysis_len_bucket"))
    os.makedirs(out_dir, exist_ok=True)
    bins = int(max(5, args.curve_bins))

    per_sample_path = os.path.join(run_dir, "per_sample.csv")
    per_token_path = os.path.join(run_dir, "per_token.csv")

    per_sample: Dict[str, Dict[str, Any]] = {}
    with open(per_sample_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if str(r.get("error") or "").strip() != "":
                continue
            sid = str(r.get("id") or "")
            per_sample[sid] = r

    seq_vpmi: Dict[str, List[Tuple[int, float]]] = {}
    seq_rank: Dict[str, List[Tuple[int, float]]] = {}
    with open(per_token_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid not in per_sample:
                continue
            step = safe_float(r.get("step_idx"))
            vv = safe_float(r.get("vpmi_logit"))
            rr = safe_float(r.get("vpmi_rank_pct_topk"))
            if step is None:
                continue
            if vv is not None:
                seq_vpmi.setdefault(sid, []).append((int(step), float(vv)))
            if rr is not None:
                seq_rank.setdefault(sid, []).append((int(step), float(rr)))

    # Sort trajectories by step.
    traj_vpmi: Dict[str, List[float]] = {}
    traj_rank: Dict[str, List[float]] = {}
    for sid, rows in seq_vpmi.items():
        rows_sorted = sorted(rows, key=lambda x: int(x[0]))
        traj_vpmi[sid] = [float(v) for _, v in rows_sorted]
    for sid, rows in seq_rank.items():
        rows_sorted = sorted(rows, key=lambda x: int(x[0]))
        traj_rank[sid] = [float(v) for _, v in rows_sorted]

    # Assign buckets.
    ids_by_bucket: Dict[str, List[str]] = {"all": list(per_sample.keys())}
    if str(args.bucket_mode) == "fixed":
        for sid, r in per_sample.items():
            n_tok = int(safe_float(r.get("n_gen_tokens")) or 0)
            b = bucket_name(n_tok)
            ids_by_bucket.setdefault(b, []).append(sid)
    else:
        qn = int(max(2, args.quantile_buckets))
        pairs = []
        for sid, r in per_sample.items():
            n_tok = int(safe_float(r.get("n_gen_tokens")) or 0)
            pairs.append((sid, n_tok))
        pairs.sort(key=lambda x: (int(x[1]), str(x[0])))
        n = len(pairs)
        if n > 0:
            for qi in range(qn):
                lo_idx = int(math.floor(qi * n / qn))
                hi_idx = int(math.floor((qi + 1) * n / qn))
                chunk = pairs[lo_idx:hi_idx]
                if len(chunk) == 0:
                    continue
                nmin = int(min(x[1] for x in chunk))
                nmax = int(max(x[1] for x in chunk))
                bname = f"q{qi+1}_len_{nmin}_{nmax}"
                ids_by_bucket[bname] = [str(x[0]) for x in chunk]

    features = [
        "vpmi_prefix_mean_k40",
        "vpmi_collapse_gap_k40",
        "vpmi_logit_mean",
        "vpmi_suffix_min_k40",
        "rank_pct_mean",
        "rank_pct_min",
    ]

    split_rows: List[Dict[str, Any]] = []
    curve_v_rows: List[Dict[str, Any]] = []
    curve_r_rows: List[Dict[str, Any]] = []
    bucket_summaries: List[Dict[str, Any]] = []

    for bname, bid_list in ids_by_bucket.items():
        if len(bid_list) == 0:
            continue
        # Feature split summary by bucket.
        for fn in features:
            vals: List[float] = []
            ys: List[int] = []
            for sid in bid_list:
                r = per_sample[sid]
                v = safe_float(r.get(fn))
                if v is None:
                    continue
                vals.append(float(v))
                ys.append(1 if parse_bool(r.get("is_success")) else 0)
            if len(vals) == 0:
                continue
            pos = [x for x, y in zip(vals, ys) if y == 1]
            neg = [x for x, y in zip(vals, ys) if y == 0]
            ks = ks_stat(pos, neg)
            auc = auc_pos_gt_neg(pos, neg)
            ge = threshold_scan(vals, ys, "ge")
            le = threshold_scan(vals, ys, "le")
            best = ge if (ge.get("best_you") or -1e9) >= (le.get("best_you") or -1e9) else le
            direction = "ge" if best is ge else "le"
            split_rows.append(
                {
                    "bucket": bname,
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
                    "best_precision": best.get("precision"),
                    "best_coverage": best.get("coverage"),
                    "n": int(len(vals)),
                    "n_success": int(sum(1 for y in ys if y == 1)),
                    "n_failure": int(sum(1 for y in ys if y == 0)),
                }
            )

        # Trajectory curves by bucket.
        s_v: List[List[float]] = []
        f_v: List[List[float]] = []
        s_r: List[List[float]] = []
        f_r: List[List[float]] = []
        for sid in bid_list:
            y = 1 if parse_bool(per_sample[sid].get("is_success")) else 0
            v_seq = traj_vpmi.get(sid, [])
            r_seq = traj_rank.get(sid, [])
            if len(v_seq) > 0:
                (s_v if y == 1 else f_v).append(interpolate_curve(v_seq, bins))
            if len(r_seq) > 0:
                (s_r if y == 1 else f_r).append(interpolate_curve(r_seq, bins))

        v_s_means: List[Optional[float]] = []
        v_f_means: List[Optional[float]] = []
        r_s_means: List[Optional[float]] = []
        r_f_means: List[Optional[float]] = []
        for bi in range(bins):
            sv = [float(x[bi]) for x in s_v]
            fv = [float(x[bi]) for x in f_v]
            sr = [float(x[bi]) for x in s_r]
            fr = [float(x[bi]) for x in f_r]
            msv = None if len(sv) == 0 else float(sum(sv) / len(sv))
            mfv = None if len(fv) == 0 else float(sum(fv) / len(fv))
            msr = None if len(sr) == 0 else float(sum(sr) / len(sr))
            mfr = None if len(fr) == 0 else float(sum(fr) / len(fr))
            v_s_means.append(msv)
            v_f_means.append(mfv)
            r_s_means.append(msr)
            r_f_means.append(mfr)

            curve_v_rows.append(
                {
                    "bucket": bname,
                    "bin_idx": int(bi),
                    "pos_norm": float(bi / max(1, bins - 1)),
                    "vpmi_logit_mean_success": msv,
                    "vpmi_logit_mean_failure": mfv,
                    "vpmi_logit_diff_success_minus_failure": (None if msv is None or mfv is None else float(msv - mfv)),
                    "n_success": int(len(s_v)),
                    "n_failure": int(len(f_v)),
                }
            )
            curve_r_rows.append(
                {
                    "bucket": bname,
                    "bin_idx": int(bi),
                    "pos_norm": float(bi / max(1, bins - 1)),
                    "rank_pct_mean_success": msr,
                    "rank_pct_mean_failure": mfr,
                    "rank_pct_diff_success_minus_failure": (None if msr is None or mfr is None else float(msr - mfr)),
                    "n_success": int(len(s_r)),
                    "n_failure": int(len(f_r)),
                }
            )

        # Per-bucket SVGs.
        svg_line_chart(
            out_path=os.path.join(out_dir, f"curve_vpmi_{bname}.svg"),
            y_success=v_s_means,
            y_failure=v_f_means,
            title=f"V-PMI Trajectory by Length Bucket: {bname}",
            subtitle=f"n_success={len(s_v)} n_failure={len(f_v)}",
            y_label="selected-token V-PMI (logit diff)",
        )
        svg_line_chart(
            out_path=os.path.join(out_dir, f"curve_rankpct_{bname}.svg"),
            y_success=r_s_means,
            y_failure=r_f_means,
            title=f"V-PMI Rank Trajectory by Length Bucket: {bname}",
            subtitle=f"n_success={len(s_r)} n_failure={len(f_r)}",
            y_label="selected-token V-PMI rank percentile",
        )

        # Bucket summary.
        acc = float(sum(1 for sid in bid_list if parse_bool(per_sample[sid].get("is_success"))) / max(1, len(bid_list)))
        best_split = [r for r in split_rows if r["bucket"] == bname]
        best_split.sort(key=lambda r: float(r.get("ks") or -1.0), reverse=True)
        bucket_summaries.append(
            {
                "bucket": bname,
                "n_samples": int(len(bid_list)),
                "accuracy": acc,
                "n_success": int(sum(1 for sid in bid_list if parse_bool(per_sample[sid].get("is_success")))),
                "n_failure": int(sum(1 for sid in bid_list if not parse_bool(per_sample[sid].get("is_success")))),
                "top_feature_by_ks": (None if len(best_split) == 0 else best_split[0]["feature"]),
                "top_feature_ks": (None if len(best_split) == 0 else best_split[0]["ks"]),
            }
        )

    split_rows.sort(
        key=lambda r: (str(r.get("bucket", "")), -(float(r.get("ks") or -1.0))),
    )
    bucket_summaries.sort(key=lambda r: str(r["bucket"]))
    write_csv(os.path.join(out_dir, "bucket_feature_split_summary.csv"), split_rows)
    write_csv(os.path.join(out_dir, "bucket_curve_vpmi.csv"), curve_v_rows)
    write_csv(os.path.join(out_dir, "bucket_curve_rankpct.csv"), curve_r_rows)
    write_csv(os.path.join(out_dir, "bucket_summary.csv"), bucket_summaries)

    summary = {
        "inputs": {
            "run_dir": run_dir,
            "curve_bins": int(bins),
        },
        "buckets": bucket_summaries,
        "outputs": {
            "bucket_feature_split_summary_csv": os.path.join(out_dir, "bucket_feature_split_summary.csv"),
            "bucket_curve_vpmi_csv": os.path.join(out_dir, "bucket_curve_vpmi.csv"),
            "bucket_curve_rankpct_csv": os.path.join(out_dir, "bucket_curve_rankpct.csv"),
            "bucket_summary_csv": os.path.join(out_dir, "bucket_summary.csv"),
            "svg_files": sorted(
                [
                    os.path.join(out_dir, x)
                    for x in os.listdir(out_dir)
                    if x.endswith(".svg")
                ]
            ),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "bucket_feature_split_summary.csv"))
    print("[saved]", os.path.join(out_dir, "bucket_curve_vpmi.csv"))
    print("[saved]", os.path.join(out_dir, "bucket_curve_rankpct.csv"))
    print("[saved]", os.path.join(out_dir, "bucket_summary.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
