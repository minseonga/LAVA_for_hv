#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


TRUE_SET = {"1", "true", "t", "yes", "y"}


def as_bool(x: Any) -> bool:
    return str("" if x is None else x).strip().lower() in TRUE_SET


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_json_float_list(x: Any) -> List[float]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: List[float] = []
    for v in obj:
        vv = safe_float(v)
        if vv is not None:
            out.append(float(vv))
    return out


def mean(vals: Sequence[float]) -> Optional[float]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if len(xs) == 0:
        return None
    return float(sum(xs) / len(xs))


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if len(xs) == 0:
        return None
    qq = max(0.0, min(1.0, float(q)))
    if len(xs) == 1:
        return float(xs[0])
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def rank_desc_pct_map(items: Sequence[Dict[str, Any]], key: str) -> Dict[int, Optional[float]]:
    vals: List[Tuple[int, float]] = []
    for c in items:
        idx = int(c["idx"])
        v = safe_float(c.get(key))
        if v is None:
            continue
        vals.append((idx, float(v)))
    vals = sorted(vals, key=lambda x: float(x[1]), reverse=True)
    out: Dict[int, Optional[float]] = {int(c["idx"]): None for c in items}
    n = len(vals)
    if n == 0:
        return out
    if n == 1:
        out[int(vals[0][0])] = 1.0
        return out
    for r, (idx, _) in enumerate(vals):
        out[int(idx)] = float((n - 1 - r) / (n - 1))
    return out


def candidate_ts_features(c: Dict[str, Any], k_prefix: int, k_suffix: int) -> Dict[str, Optional[float]]:
    toks = [float(x) for x in (c.get("core_vpmi_toks") or []) if math.isfinite(float(x))]
    out: Dict[str, Optional[float]] = {
        "c_vpmi_prefix_mean_k": None,
        "c_vpmi_suffix_mean_k": None,
        "c_vpmi_suffix_min_k": None,
    }
    if len(toks) == 0:
        return out
    kp = int(max(1, min(int(k_prefix), len(toks))))
    ks = int(max(1, min(int(k_suffix), len(toks))))
    pref = toks[:kp]
    suff = toks[-ks:]
    out["c_vpmi_prefix_mean_k"] = mean(pref)
    out["c_vpmi_suffix_mean_k"] = mean(suff)
    out["c_vpmi_suffix_min_k"] = float(min(suff))
    return out


def load_beam_selector_rows(beam_dir: str, k_prefix: int, k_suffix: int) -> List[Dict[str, Any]]:
    per_sample = list(csv.DictReader(open(os.path.join(beam_dir, "per_sample.csv"), encoding="utf-8")))
    per_cand = list(csv.DictReader(open(os.path.join(beam_dir, "per_candidate.csv"), encoding="utf-8")))

    cands_by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_cand:
        sid = str(r.get("id", ""))
        idx_f = safe_float(r.get("cand_idx"))
        if sid == "" or idx_f is None:
            continue
        s_full = safe_float(r.get("s_full"))
        s_q = safe_float(r.get("s_ans_q"))
        s_core = safe_float(r.get("s_core_img"))
        vpmi = None if s_core is None or s_q is None else float(s_core - s_q)
        cands_by_sid[sid].append(
            {
                "id": sid,
                "idx": int(idx_f),
                "is_champion": as_bool(r.get("is_champion", "")),
                "is_correct_eval": as_bool(r.get("is_correct_eval", "")) if str(r.get("is_correct_eval", "")).strip() != "" else None,
                "is_correct_heuristic": as_bool(r.get("is_correct_heuristic", "")) if str(r.get("is_correct_heuristic", "")).strip() != "" else None,
                "is_correct_strict": as_bool(r.get("is_correct_strict", "")) if str(r.get("is_correct_strict", "")).strip() != "" else None,
                "s_full": s_full,
                "s_ans_q": s_q,
                "s_core_img": s_core,
                "vpmi": vpmi,
                "core_vpmi_toks": parse_json_float_list(r.get("core_vpmi_toks_json")),
            }
        )

    out_rows: List[Dict[str, Any]] = []
    for r in per_sample:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", ""))
        cands = cands_by_sid.get(sid, [])
        if len(cands) == 0:
            continue
        champ = next((c for c in cands if bool(c.get("is_champion", False))), None)
        if champ is None:
            finite = [c for c in cands if c.get("s_full") is not None]
            if len(finite) == 0:
                continue
            champ = max(finite, key=lambda x: float(x["s_full"]))
        champ_ok = as_bool(r.get("is_success", ""))
        pool = [c for c in cands if int(c["idx"]) != int(champ["idx"])]

        all_items = [champ] + pool
        rank_vpmi = rank_desc_pct_map(all_items, "vpmi")
        ch_ts = candidate_ts_features(champ, k_prefix=int(k_prefix), k_suffix=int(k_suffix))

        for c in pool:
            cand_ok = c.get("is_correct_eval")
            if cand_ok is None:
                cand_ok = bool(c.get("is_correct_heuristic", False))

            better = bool((not champ_ok) and cand_ok)
            worse = bool(champ_ok and (not cand_ok))
            if not better and not worse:
                continue

            c_ts = candidate_ts_features(c, k_prefix=int(k_prefix), k_suffix=int(k_suffix))
            d_rank = None
            if rank_vpmi.get(int(c["idx"])) is not None and rank_vpmi.get(int(champ["idx"])) is not None:
                d_rank = float(float(rank_vpmi[int(c["idx"])]) - float(rank_vpmi[int(champ["idx"])]))
            d_smin = None
            if c_ts["c_vpmi_suffix_min_k"] is not None and ch_ts["c_vpmi_suffix_min_k"] is not None:
                d_smin = float(float(c_ts["c_vpmi_suffix_min_k"]) - float(ch_ts["c_vpmi_suffix_min_k"]))

            out_rows.append(
                {
                    "sid": sid,
                    "cand_idx": int(c["idx"]),
                    "label": "better" if better else "worse",
                    "y": 1 if better else 0,
                    "d_rank_vpmi_pct": d_rank,
                    "d_vpmi_suffix_min_k": d_smin,
                    "c_vpmi_prefix_mean_k": c_ts["c_vpmi_prefix_mean_k"],
                }
            )
    return out_rows


def load_feature_meta(ranking_csv: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(ranking_csv):
        return {}
    rows = list(csv.DictReader(open(ranking_csv, encoding="utf-8")))
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        feat = str(r.get("feature", ""))
        if feat == "":
            continue
        out[feat] = dict(r)
    return out


def ecdf(vals: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.array(sorted(float(v) for v in vals if math.isfinite(float(v))), dtype=np.float64)
    if xs.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / float(xs.size)
    return xs, ys


def save_distribution_plot(
    path: str,
    feat: str,
    better_vals: Sequence[float],
    worse_vals: Sequence[float],
    meta: Optional[Dict[str, Any]],
) -> None:
    bvals = [float(x) for x in better_vals if math.isfinite(float(x))]
    wvals = [float(x) for x in worse_vals if math.isfinite(float(x))]
    pooled = bvals + wvals
    if len(pooled) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.set_title(f"{feat}: no valid values")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    x_min = quantile(pooled, 0.01)
    x_max = quantile(pooled, 0.99)
    if x_min is None or x_max is None or float(x_max) <= float(x_min):
        x_min = float(min(pooled))
        x_max = float(max(pooled) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax0, ax1 = axes

    bins = np.linspace(float(x_min), float(x_max), 50)
    ax0.hist(wvals, bins=bins, alpha=0.45, density=True, color="#d62728", label=f"worse (n={len(wvals)})")
    ax0.hist(bvals, bins=bins, alpha=0.45, density=True, color="#1f77b4", label=f"better (n={len(bvals)})")
    tau = safe_float(None if meta is None else meta.get("tau"))
    direction = None if meta is None else str(meta.get("direction", ""))
    if tau is not None and math.isfinite(float(tau)):
        ax0.axvline(float(tau), color="black", linestyle="--", linewidth=1.2, label=f"tau={tau:.4f} ({direction})")
    ax0.set_title(f"{feat} - Histogram")
    ax0.set_xlabel(feat)
    ax0.set_ylabel("Density")
    ax0.legend(loc="best", fontsize=8)

    xb, yb = ecdf(bvals)
    xw, yw = ecdf(wvals)
    if xw.size > 0:
        ax1.step(xw, yw, where="post", color="#d62728", label="worse")
    if xb.size > 0:
        ax1.step(xb, yb, where="post", color="#1f77b4", label="better")
    if tau is not None and math.isfinite(float(tau)):
        ax1.axvline(float(tau), color="black", linestyle="--", linewidth=1.2)
    title = f"{feat} - ECDF"
    if meta is not None:
        ks = safe_float(meta.get("ks"))
        prec = safe_float(meta.get("precision"))
        nd = safe_float(meta.get("near_density_eps2pct_iqr90"))
        parts: List[str] = []
        if ks is not None:
            parts.append(f"KS={ks:.3f}")
        if prec is not None:
            parts.append(f"Prec={prec:.3f}")
        if nd is not None:
            parts.append(f"Near={nd:.3f}")
        if len(parts) > 0:
            title += " | " + ", ".join(parts)
    ax1.set_title(title)
    ax1.set_xlabel(feat)
    ax1.set_ylabel("ECDF")
    ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
            wr.writerow({k: r.get(k, None) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize meaningful selector feature distributions (better vs worse).")
    ap.add_argument("--beam_dir", type=str, required=True)
    ap.add_argument("--ranking_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--k_prefix", type=int, default=2)
    ap.add_argument("--k_suffix", type=int, default=2)
    ap.add_argument(
        "--features",
        type=str,
        default="d_rank_vpmi_pct,d_vpmi_suffix_min_k,c_vpmi_prefix_mean_k",
        help="Comma-separated features to visualize.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_beam_selector_rows(
        beam_dir=os.path.abspath(args.beam_dir),
        k_prefix=int(args.k_prefix),
        k_suffix=int(args.k_suffix),
    )
    meta = load_feature_meta(os.path.abspath(args.ranking_csv))
    feats = [x.strip() for x in str(args.features).split(",") if x.strip() != ""]

    out_summary: List[Dict[str, Any]] = []
    for feat in feats:
        bvals = [safe_float(r.get(feat)) for r in rows if int(r["y"]) == 1]
        wvals = [safe_float(r.get(feat)) for r in rows if int(r["y"]) == 0]
        bvals = [float(x) for x in bvals if x is not None]
        wvals = [float(x) for x in wvals if x is not None]
        m = meta.get(feat, {})
        png_path = os.path.join(args.out_dir, f"dist_{feat}.png")
        save_distribution_plot(
            path=png_path,
            feat=feat,
            better_vals=bvals,
            worse_vals=wvals,
            meta=m if len(m) > 0 else None,
        )
        out_summary.append(
            {
                "feature": feat,
                "n_better": int(len(bvals)),
                "n_worse": int(len(wvals)),
                "better_mean": mean(bvals),
                "worse_mean": mean(wvals),
                "better_q25": quantile(bvals, 0.25),
                "better_q50": quantile(bvals, 0.50),
                "better_q75": quantile(bvals, 0.75),
                "worse_q25": quantile(wvals, 0.25),
                "worse_q50": quantile(wvals, 0.50),
                "worse_q75": quantile(wvals, 0.75),
                "tau": safe_float(m.get("tau")),
                "direction": m.get("direction"),
                "ks": safe_float(m.get("ks")),
                "precision": safe_float(m.get("precision")),
                "near_density_eps2pct_iqr90": safe_float(m.get("near_density_eps2pct_iqr90")),
                "plot_png": os.path.abspath(png_path),
            }
        )

    summary_csv = os.path.join(args.out_dir, "meaningful_feature_distribution_summary.csv")
    write_csv(summary_csv, out_summary)
    values_csv = os.path.join(args.out_dir, "selector_feature_values_long.csv")
    write_csv(values_csv, rows)

    out_json = {
        "inputs": {
            "beam_dir": os.path.abspath(args.beam_dir),
            "ranking_csv": os.path.abspath(args.ranking_csv),
            "k_prefix": int(args.k_prefix),
            "k_suffix": int(args.k_suffix),
            "features": feats,
        },
        "counts": {
            "n_selector_rows": int(len(rows)),
            "n_better": int(sum(1 for r in rows if int(r["y"]) == 1)),
            "n_worse": int(sum(1 for r in rows if int(r["y"]) == 0)),
        },
        "outputs": {
            "summary_csv": os.path.abspath(summary_csv),
            "values_csv": os.path.abspath(values_csv),
            "plots": [r["plot_png"] for r in out_summary],
        },
    }
    summary_json = os.path.join(args.out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print("[saved]", summary_csv)
    print("[saved]", values_csv)
    for p in out_json["outputs"]["plots"]:
        print("[saved]", p)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
