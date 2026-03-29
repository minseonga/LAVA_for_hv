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


def quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return float(ys[0])
    p = min(1.0, max(0.0, float(q))) * (len(ys) - 1)
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    if lo == hi:
        return float(ys[lo])
    w = p - lo
    return float((1.0 - w) * ys[lo] + w * ys[hi])


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


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
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


def load_champion_rows(per_candidate_csv: str) -> List[Tuple[str, List[float], int]]:
    out: List[Tuple[str, List[float], int]] = []
    with open(per_candidate_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if not parse_bool(r.get("is_champion")):
                continue
            tj = str(r.get("core_vpmi_toks_json") or "").strip()
            if tj == "":
                continue
            try:
                toks_raw = json.loads(tj)
            except Exception:
                continue
            if not isinstance(toks_raw, list):
                continue
            toks = [safe_float(x) for x in toks_raw]
            toks = [float(x) for x in toks if x is not None]
            if len(toks) == 0:
                continue
            label = str(r.get("label") or "").strip().lower()
            y = 1 if label == "success" else 0
            out.append((str(r.get("id") or ""), toks, int(y)))
    return out


def extract_features(v: List[float]) -> Dict[str, float]:
    m = int(len(v))
    k = int(max(1, math.ceil(0.4 * m)))
    pref = v[:k]
    suff = v[m - k :]
    deltas = [float(v[i] - v[i - 1]) for i in range(1, m)]
    minv = float(min(v))
    minpos = float(v.index(minv) / max(1, m - 1))

    return {
        "m": float(m),
        "vpmi_mean": float(sum(v) / m),
        "vpmi_min": float(minv),
        "vpmi_max": float(max(v)),
        "prefix_mean_k": float(sum(pref) / len(pref)),
        "suffix_mean_k": float(sum(suff) / len(suff)),
        "suffix_min_k": float(min(suff)),
        "ps_gap_k": float(sum(suff) / len(suff) - sum(pref) / len(pref)),
        "suffix_min_minus_prefix_mean": float(min(suff) - sum(pref) / len(pref)),
        "slope_last_first": float(v[-1] - v[0]),
        "min_pos_norm": float(minpos),
        "max_drop_step": float(min(deltas) if len(deltas) > 0 else 0.0),
        "drop_count": float(sum(1 for d in deltas if d < 0.0)),
        "flip_count": float(sum(1 for i in range(1, len(deltas)) if deltas[i - 1] * deltas[i] < 0.0)),
        "max_drop_pos_norm": float((deltas.index(min(deltas)) + 1) / max(1, m - 1) if len(deltas) > 0 else 0.0),
    }


def interpolate_curve(vals: List[float], n_bins: int) -> List[float]:
    m = len(vals)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy champion V-PMI split/change-point analysis.")
    ap.add_argument("--per_candidate_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--curve_bins", type=int, default=20)
    ap.add_argument("--delta_bins", type=int, default=12)
    args = ap.parse_args()

    src = os.path.abspath(args.per_candidate_csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = load_champion_rows(src)
    feats: List[Dict[str, object]] = []
    for sid, toks, y in rows:
        d = extract_features(toks)
        d["id"] = sid
        d["y"] = int(y)
        feats.append(d)

    f_names = [k for k in feats[0].keys() if k not in {"id", "y"}]
    ranking: List[Dict[str, object]] = []
    for fn in f_names:
        pos = [float(r[fn]) for r in feats if int(r["y"]) == 1]
        neg = [float(r[fn]) for r in feats if int(r["y"]) == 0]
        ranking.append(
            {
                "feature": fn,
                "ks": ks_stat(pos, neg),
                "auc_pos_gt_neg": auc_pos_gt_neg(pos, neg),
                "mean_success": (None if len(pos) == 0 else float(sum(pos) / len(pos))),
                "mean_failure": (None if len(neg) == 0 else float(sum(neg) / len(neg))),
                "median_success": quantile(pos, 0.5),
                "median_failure": quantile(neg, 0.5),
                "n_success": int(len(pos)),
                "n_failure": int(len(neg)),
            }
        )
    ranking.sort(key=lambda x: (float(x["ks"]) if x["ks"] is not None else -1.0), reverse=True)
    write_csv(os.path.join(out_dir, "feature_split_ranking.csv"), ranking)

    b_curve = int(max(5, args.curve_bins))
    succ_bins: List[List[float]] = [[] for _ in range(b_curve)]
    fail_bins: List[List[float]] = [[] for _ in range(b_curve)]
    for _, toks, y in rows:
        interp = interpolate_curve(toks, b_curve)
        target = succ_bins if y == 1 else fail_bins
        for i, vv in enumerate(interp):
            target[i].append(float(vv))

    curve_rows: List[Dict[str, object]] = []
    for i in range(b_curve):
        ms = (None if len(succ_bins[i]) == 0 else float(sum(succ_bins[i]) / len(succ_bins[i])))
        mf = (None if len(fail_bins[i]) == 0 else float(sum(fail_bins[i]) / len(fail_bins[i])))
        diff = (None if ms is None or mf is None else float(ms - mf))
        curve_rows.append(
            {
                "bin_idx": int(i),
                "pos_norm": float(i / max(1, b_curve - 1)),
                "mean_success": ms,
                "mean_failure": mf,
                "diff_success_minus_failure": diff,
                "n_success": int(len(succ_bins[i])),
                "n_failure": int(len(fail_bins[i])),
            }
        )
    write_csv(os.path.join(out_dir, "vpmi_position_curve.csv"), curve_rows)

    b_delta = int(max(3, args.delta_bins))
    succ_d: List[List[float]] = [[] for _ in range(b_delta)]
    fail_d: List[List[float]] = [[] for _ in range(b_delta)]
    for _, toks, y in rows:
        if len(toks) < 2:
            continue
        dlt = [float(toks[i] - toks[i - 1]) for i in range(1, len(toks))]
        m = len(dlt)
        for j, dv in enumerate(dlt):
            pos = float(j / max(1, m - 1))
            bi = int(round(pos * (b_delta - 1)))
            bi = int(min(b_delta - 1, max(0, bi)))
            (succ_d if y == 1 else fail_d)[bi].append(float(dv))

    delta_rows: List[Dict[str, object]] = []
    for i in range(b_delta):
        if len(succ_d[i]) == 0 or len(fail_d[i]) == 0:
            continue
        ms = float(sum(succ_d[i]) / len(succ_d[i]))
        mf = float(sum(fail_d[i]) / len(fail_d[i]))
        delta_rows.append(
            {
                "delta_bin_idx": int(i),
                "pos_norm": float(i / max(1, b_delta - 1)),
                "mean_step_delta_success": ms,
                "mean_step_delta_failure": mf,
                "failure_minus_success": float(mf - ms),
                "n_success_steps": int(len(succ_d[i])),
                "n_failure_steps": int(len(fail_d[i])),
            }
        )
    write_csv(os.path.join(out_dir, "vpmi_step_delta_curve.csv"), delta_rows)

    max_abs_pos = None
    if len(curve_rows) > 0:
        max_abs_pos = max(
            curve_rows,
            key=lambda r: abs(float(r["diff_success_minus_failure"])) if r["diff_success_minus_failure"] is not None else -1.0,
        )
    worst_delta = None
    if len(delta_rows) > 0:
        worst_delta = min(delta_rows, key=lambda r: float(r["failure_minus_success"]))

    summary = {
        "inputs": {
            "per_candidate_csv": src,
            "curve_bins": b_curve,
            "delta_bins": b_delta,
        },
        "counts": {
            "n_rows": int(len(rows)),
            "n_success": int(sum(1 for _, _, y in rows if y == 1)),
            "n_failure": int(sum(1 for _, _, y in rows if y == 0)),
            "n_len_ge_2": int(sum(1 for _, t, _ in rows if len(t) >= 2)),
        },
        "top_features_by_ks": ranking[:10],
        "key_change_points": {
            "max_abs_diff_position_bin": max_abs_pos,
            "worst_failure_step_delta_bin": worst_delta,
        },
        "outputs": {
            "feature_split_ranking_csv": os.path.join(out_dir, "feature_split_ranking.csv"),
            "vpmi_position_curve_csv": os.path.join(out_dir, "vpmi_position_curve.csv"),
            "vpmi_step_delta_curve_csv": os.path.join(out_dir, "vpmi_step_delta_curve.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "feature_split_ranking.csv"))
    print("[saved]", os.path.join(out_dir, "vpmi_position_curve.csv"))
    print("[saved]", os.path.join(out_dir, "vpmi_step_delta_curve.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
