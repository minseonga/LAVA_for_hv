#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if len(xs) == 0:
        return {"n": 0, "mean": None, "q25": None, "median": None, "q75": None}
    return {
        "n": int(len(xs)),
        "mean": float(sum(xs) / len(xs)),
        "q25": quantile(xs, 0.25),
        "median": quantile(xs, 0.5),
        "q75": quantile(xs, 0.75),
    }


def ks_distance(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    xa = sorted(float(x) for x in a if math.isfinite(float(x)))
    xb = sorted(float(x) for x in b if math.isfinite(float(x)))
    if len(xa) == 0 or len(xb) == 0:
        return None
    vals = sorted(set(xa + xb))
    ia = 0
    ib = 0
    da = 0.0
    db = 0.0
    na = float(len(xa))
    nb = float(len(xb))
    best = 0.0
    for v in vals:
        while ia < len(xa) and xa[ia] <= v:
            ia += 1
        while ib < len(xb) and xb[ib] <= v:
            ib += 1
        da = ia / na
        db = ib / nb
        best = max(best, abs(da - db))
    return float(best)


def threshold_density(vals: Sequence[float], tau: float, eps: float) -> Optional[float]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if len(xs) == 0:
        return None
    near = sum(1 for x in xs if abs(float(x) - float(tau)) <= float(eps))
    return float(near / len(xs))


def hist_counts(vals: Sequence[float], x_min: float, x_max: float, bins: int) -> List[int]:
    out = [0 for _ in range(int(bins))]
    if bins <= 0 or x_max <= x_min:
        return out
    w = (float(x_max) - float(x_min)) / float(bins)
    for v in vals:
        x = float(v)
        if x < x_min or x > x_max:
            continue
        idx = int((x - x_min) / w)
        if idx == bins:
            idx = bins - 1
        if 0 <= idx < bins:
            out[idx] += 1
    return out


def save_grouped_hist_svg(
    path: str,
    title: str,
    series: Dict[str, Sequence[float]],
    x_min: Optional[float],
    x_max: Optional[float],
    bins: int = 50,
    vlines: Optional[List[Tuple[float, str]]] = None,
) -> None:
    labels = [k for k in series.keys()]
    palette = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"
    ]
    # Build finite pooled values.
    pooled: List[float] = []
    cleaned: Dict[str, List[float]] = {}
    for k in labels:
        xs = [float(v) for v in series[k] if math.isfinite(float(v))]
        cleaned[k] = xs
        pooled.extend(xs)
    if len(pooled) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='320'></svg>")
        return

    if x_min is None:
        x_min = quantile(pooled, 0.01)
    if x_max is None:
        x_max = quantile(pooled, 0.99)
    if x_min is None or x_max is None:
        x_min = min(pooled)
        x_max = max(pooled)
    if float(x_max) <= float(x_min):
        x_max = float(x_min) + 1e-6

    counts: Dict[str, List[int]] = {k: hist_counts(cleaned[k], float(x_min), float(x_max), int(bins)) for k in labels}
    ymax = 1
    for k in labels:
        ymax = max(ymax, max(counts[k]) if len(counts[k]) > 0 else 0)

    W = 1200
    H = 520
    ml = 80
    mr = 20
    mt = 50
    mb = 70
    pw = W - ml - mr
    ph = H - mt - mb

    def sx(x: float) -> float:
        return float(ml + (float(x) - float(x_min)) * pw / (float(x_max) - float(x_min)))

    def sy(y: float) -> float:
        return float(mt + ph - float(y) * ph / float(ymax))

    bin_w_px = float(pw / bins)
    group_n = max(1, len(labels))
    bar_w = float(bin_w_px / group_n)

    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append(f"<text x='{W/2}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>")
    # Axes
    lines.append(f"<line x1='{ml}' y1='{mt+ph}' x2='{ml+pw}' y2='{mt+ph}' stroke='black' stroke-width='1'/>")
    lines.append(f"<line x1='{ml}' y1='{mt}' x2='{ml}' y2='{mt+ph}' stroke='black' stroke-width='1'/>")

    # Y ticks
    for t in [0, int(ymax * 0.25), int(ymax * 0.5), int(ymax * 0.75), int(ymax)]:
        y = sy(t)
        lines.append(f"<line x1='{ml-4}' y1='{y:.2f}' x2='{ml}' y2='{y:.2f}' stroke='black'/>")
        lines.append(f"<text x='{ml-8}' y='{y+4:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{t}</text>")
    # X ticks
    for tt in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xv = float(x_min) + tt * (float(x_max) - float(x_min))
        x = sx(xv)
        lines.append(f"<line x1='{x:.2f}' y1='{mt+ph}' x2='{x:.2f}' y2='{mt+ph+4}' stroke='black'/>")
        lines.append(f"<text x='{x:.2f}' y='{mt+ph+20}' text-anchor='middle' font-size='11' font-family='Arial'>{xv:.2f}</text>")

    # Bars
    for li, k in enumerate(labels):
        col = palette[li % len(palette)]
        ys = counts[k]
        for bi in range(bins):
            c = ys[bi]
            if c <= 0:
                continue
            x = ml + bi * bin_w_px + li * bar_w
            y = sy(c)
            h = mt + ph - y
            lines.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(1.0, bar_w-0.5):.2f}' height='{h:.2f}' "
                f"fill='{col}' fill-opacity='0.55' stroke='{col}' stroke-width='0.5'/>"
            )

    # Threshold lines
    if vlines is not None:
        for v, lbl in vlines:
            if v < float(x_min) or v > float(x_max):
                continue
            x = sx(float(v))
            lines.append(f"<line x1='{x:.2f}' y1='{mt}' x2='{x:.2f}' y2='{mt+ph}' stroke='#111' stroke-width='1.5' stroke-dasharray='6,4'/>")
            lines.append(f"<text x='{x+4:.2f}' y='{mt+14}' font-size='11' font-family='Arial' fill='#111'>{lbl}</text>")

    # Legend
    lx = ml + 8
    ly = mt + 8
    for li, k in enumerate(labels):
        col = palette[li % len(palette)]
        yy = ly + li * 18
        lines.append(f"<rect x='{lx}' y='{yy}' width='12' height='12' fill='{col}' fill-opacity='0.55' stroke='{col}'/>")
        lines.append(f"<text x='{lx+18}' y='{yy+10}' font-size='12' font-family='Arial'>{k} (n={len(cleaned[k])})</text>")

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze feature density/sparsity and export histogram SVGs (offline).")
    ap.add_argument("--greedy_dir", type=str, required=True)
    ap.add_argument("--expand_dir", type=str, required=True)
    ap.add_argument("--adaptive_per_sample_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--gate_name", type=str, default="gate_and_vpmi_m5.5_and_sfull_m6")
    ap.add_argument("--policy_base", type=str, default="agree_vminpm_wmin")
    ap.add_argument("--tau_vpmi", type=float, default=-5.5)
    ap.add_argument("--tau_sfull", type=float, default=-6.0)
    ap.add_argument("--tau_dfull", type=float, default=-0.05)
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["auto", "strict", "heuristic"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Import local evaluator utilities.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_selector_tradeoff import load_samples, select_candidate

    # Load per-sample outcomes for adaptive row.
    ad_rows = [r for r in read_csv(args.adaptive_per_sample_csv) if str(r.get("gate", "")) == str(args.gate_name)]
    ad_by_id = {str(r.get("id", "")): r for r in ad_rows}

    # Load greedy/expand sample states.
    g_samples = load_samples(args.greedy_dir, eval_mode=str(args.eval_mode))
    e_samples = load_samples(args.expand_dir, eval_mode=str(args.eval_mode))
    g_by_id = {str(s.sid): s for s in g_samples}
    e_by_id = {str(s.sid): s for s in e_samples}
    ids_gate = sorted(set(g_by_id.keys()) & set(ad_by_id.keys()))
    ids_selector = sorted(set(e_by_id.keys()) & set(ad_by_id.keys()))

    gate_vpmi_all: List[float] = []
    gate_sfull_all: List[float] = []
    gate_vpmi_gain: List[float] = []
    gate_vpmi_harm: List[float] = []
    gate_vpmi_same: List[float] = []
    gate_sfull_gain: List[float] = []
    gate_sfull_harm: List[float] = []
    gate_sfull_same: List[float] = []

    # Expand-only selector/trigger features.
    dfull_all: List[float] = []
    dfull_switch_gain: List[float] = []
    dfull_switch_harm: List[float] = []
    dfull_switch_same: List[float] = []
    dvpmi_all: List[float] = []
    dvpmi_switch_gain: List[float] = []
    dvpmi_switch_harm: List[float] = []
    dvpmi_switch_same: List[float] = []
    champ_vpmi_all: List[float] = []
    champ_vpmi_switch_gain: List[float] = []
    champ_vpmi_switch_harm: List[float] = []
    champ_vpmi_switch_same: List[float] = []

    n_expand = 0
    n_gain = 0
    n_harm = 0
    n_same = 0
    n_base_ok = 0
    n_base_fail = 0
    gate_flag_and = 0
    gate_flag_and_gain = 0
    gate_flag_and_harm = 0
    gate_flag_and_same = 0
    gate_flag_and_base_fail = 0
    n_sel = 0
    n_switch = 0
    n_switch_gain = 0
    n_switch_harm = 0
    n_switch_same = 0

    for sid in ids_gate:
        gs = g_by_id[sid]
        row = ad_by_id[sid]

        # Gate features from greedy champion.
        if gs.champ.vpmi is not None:
            gate_vpmi_all.append(float(gs.champ.vpmi))
        if gs.champ.s_full is not None:
            gate_sfull_all.append(float(gs.champ.s_full))

        base_ok = as_bool(row.get("base_ok", ""))
        pred_ok = as_bool(row.get("pred_ok", ""))
        if base_ok:
            n_base_ok += 1
        else:
            n_base_fail += 1

        expand_gain = bool((not base_ok) and pred_ok)
        expand_harm = bool(base_ok and (not pred_ok))
        if expand_gain:
            n_gain += 1
            if gs.champ.vpmi is not None:
                gate_vpmi_gain.append(float(gs.champ.vpmi))
            if gs.champ.s_full is not None:
                gate_sfull_gain.append(float(gs.champ.s_full))
        elif expand_harm:
            n_harm += 1
            if gs.champ.vpmi is not None:
                gate_vpmi_harm.append(float(gs.champ.vpmi))
            if gs.champ.s_full is not None:
                gate_sfull_harm.append(float(gs.champ.s_full))
        else:
            n_same += 1
            if gs.champ.vpmi is not None:
                gate_vpmi_same.append(float(gs.champ.vpmi))
            if gs.champ.s_full is not None:
                gate_sfull_same.append(float(gs.champ.s_full))

        cond_v = bool(gs.champ.vpmi is not None and float(gs.champ.vpmi) < float(args.tau_vpmi))
        cond_s = bool(gs.champ.s_full is not None and float(gs.champ.s_full) < float(args.tau_sfull))
        flag_and = bool(cond_v and cond_s)
        if flag_and:
            gate_flag_and += 1
            if expand_gain:
                gate_flag_and_gain += 1
            elif expand_harm:
                gate_flag_and_harm += 1
            else:
                gate_flag_and_same += 1
            if not base_ok:
                gate_flag_and_base_fail += 1

    for sid in ids_selector:
        es = e_by_id[sid]
        row = ad_by_id[sid]
        expanded = as_bool(row.get("expanded", ""))
        if not expanded:
            continue
        n_expand += 1
        safe = select_candidate(str(args.policy_base), es)
        if safe is None:
            continue
        if safe.s_full is None or es.champ.s_full is None:
            continue
        if safe.vpmi is None or es.champ.vpmi is None:
            continue
        n_sel += 1

        d_full = float(safe.s_full) - float(es.champ.s_full)
        d_vpmi = float(safe.vpmi) - float(es.champ.vpmi)
        c_vpmi = float(es.champ.vpmi)
        dfull_all.append(d_full)
        dvpmi_all.append(d_vpmi)
        champ_vpmi_all.append(c_vpmi)

        pass_dfull = bool(d_full <= float(args.tau_dfull))
        pass_p3 = bool(d_vpmi > 0.0 and c_vpmi < 0.0)
        do_switch = bool(pass_dfull and pass_p3)
        if not do_switch:
            continue

        n_switch += 1
        safe_ok = bool(es.safe_ok_by_idx.get(int(safe.idx), False))
        champ_ok = bool(es.base_ok)
        if safe_ok and (not champ_ok):
            n_switch_gain += 1
            dfull_switch_gain.append(d_full)
            dvpmi_switch_gain.append(d_vpmi)
            champ_vpmi_switch_gain.append(c_vpmi)
        elif (not safe_ok) and champ_ok:
            n_switch_harm += 1
            dfull_switch_harm.append(d_full)
            dvpmi_switch_harm.append(d_vpmi)
            champ_vpmi_switch_harm.append(c_vpmi)
        else:
            n_switch_same += 1
            dfull_switch_same.append(d_full)
            dvpmi_switch_same.append(d_vpmi)
            champ_vpmi_switch_same.append(c_vpmi)

    # Stats summary.
    gate_flag_v = [v for v in gate_vpmi_all if v < float(args.tau_vpmi)]
    gate_flag_s = [v for v in gate_sfull_all if v < float(args.tau_sfull)]
    gate_dens = {
        "vpmi_near_tau": {f"eps_{e}": threshold_density(gate_vpmi_all, float(args.tau_vpmi), e) for e in [0.05, 0.1, 0.25, 0.5, 1.0]},
        "sfull_near_tau": {f"eps_{e}": threshold_density(gate_sfull_all, float(args.tau_sfull), e) for e in [0.05, 0.1, 0.25, 0.5, 1.0]},
    }

    selector_dens = {
        "dfull_near_tau": {f"eps_{e}": threshold_density(dfull_all, float(args.tau_dfull), e) for e in [0.01, 0.02, 0.05, 0.1, 0.2]},
        "dvpmi_near_zero": {f"eps_{e}": threshold_density(dvpmi_all, 0.0, e) for e in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]},
        "champ_vpmi_near_zero": {f"eps_{e}": threshold_density(champ_vpmi_all, 0.0, e) for e in [0.1, 0.25, 0.5, 1.0]},
    }

    summary = {
        "inputs": {
            "greedy_dir": os.path.abspath(args.greedy_dir),
            "expand_dir": os.path.abspath(args.expand_dir),
            "adaptive_per_sample_csv": os.path.abspath(args.adaptive_per_sample_csv),
            "gate_name": str(args.gate_name),
            "policy_base": str(args.policy_base),
            "tau_vpmi": float(args.tau_vpmi),
            "tau_sfull": float(args.tau_sfull),
            "tau_dfull": float(args.tau_dfull),
            "eval_mode": str(args.eval_mode),
        },
        "counts": {
            "n_ids_gate": int(len(ids_gate)),
            "n_ids_selector_pool": int(len(ids_selector)),
            "n_base_ok": int(n_base_ok),
            "n_base_fail": int(n_base_fail),
            "n_expand_gain": int(n_gain),
            "n_expand_harm": int(n_harm),
            "n_expand_same": int(n_same),
            "n_expanded_rows": int(n_expand),
            "n_selected_candidates": int(n_sel),
            "n_switch": int(n_switch),
            "n_switch_gain": int(n_switch_gain),
            "n_switch_harm": int(n_switch_harm),
            "n_switch_same": int(n_switch_same),
        },
        "gate_feature_stats": {
            "vpmi_all": stats(gate_vpmi_all),
            "vpmi_gain": stats(gate_vpmi_gain),
            "vpmi_harm": stats(gate_vpmi_harm),
            "vpmi_same": stats(gate_vpmi_same),
            "sfull_all": stats(gate_sfull_all),
            "sfull_gain": stats(gate_sfull_gain),
            "sfull_harm": stats(gate_sfull_harm),
            "sfull_same": stats(gate_sfull_same),
            "ks_gain_vs_harm_vpmi": ks_distance(gate_vpmi_gain, gate_vpmi_harm),
            "ks_gain_vs_harm_sfull": ks_distance(gate_sfull_gain, gate_sfull_harm),
            "flag_rate_vpmi_tau": (None if len(gate_vpmi_all) == 0 else float(len(gate_flag_v) / len(gate_vpmi_all))),
            "flag_rate_sfull_tau": (None if len(gate_sfull_all) == 0 else float(len(gate_flag_s) / len(gate_sfull_all))),
            "flag_rate_and_tau": (None if len(ids_gate) == 0 else float(gate_flag_and / len(ids_gate))),
            "flag_and_gain": int(gate_flag_and_gain),
            "flag_and_harm": int(gate_flag_and_harm),
            "flag_and_same": int(gate_flag_and_same),
            "flag_and_precision_gain": (
                None
                if (gate_flag_and_gain + gate_flag_and_harm) == 0
                else float(gate_flag_and_gain / (gate_flag_and_gain + gate_flag_and_harm))
            ),
            "flag_and_base_fail_rate": (
                None
                if gate_flag_and == 0
                else float(gate_flag_and_base_fail / gate_flag_and)
            ),
            "density": gate_dens,
        },
        "selector_trigger_feature_stats": {
            "dfull_all": stats(dfull_all),
            "dfull_switch_gain": stats(dfull_switch_gain),
            "dfull_switch_harm": stats(dfull_switch_harm),
            "dvpmi_all": stats(dvpmi_all),
            "dvpmi_switch_gain": stats(dvpmi_switch_gain),
            "dvpmi_switch_harm": stats(dvpmi_switch_harm),
            "champ_vpmi_all": stats(champ_vpmi_all),
            "champ_vpmi_switch_gain": stats(champ_vpmi_switch_gain),
            "champ_vpmi_switch_harm": stats(champ_vpmi_switch_harm),
            "ks_switch_gain_vs_harm_dfull": ks_distance(dfull_switch_gain, dfull_switch_harm),
            "ks_switch_gain_vs_harm_dvpmi": ks_distance(dvpmi_switch_gain, dvpmi_switch_harm),
            "ks_switch_gain_vs_harm_champ_vpmi": ks_distance(champ_vpmi_switch_gain, champ_vpmi_switch_harm),
            "density": selector_dens,
        },
        "outputs": {
            "gate_vpmi_hist_svg": os.path.join(os.path.abspath(args.out_dir), "hist_gate_vpmi_gain_harm_same.svg"),
            "gate_sfull_hist_svg": os.path.join(os.path.abspath(args.out_dir), "hist_gate_sfull_gain_harm_same.svg"),
            "selector_dfull_hist_svg": os.path.join(os.path.abspath(args.out_dir), "hist_selector_dfull_switch_outcome.svg"),
            "trigger_dvpmi_hist_svg": os.path.join(os.path.abspath(args.out_dir), "hist_trigger_dvpmi_switch_outcome.svg"),
            "trigger_champ_vpmi_hist_svg": os.path.join(os.path.abspath(args.out_dir), "hist_trigger_champ_vpmi_switch_outcome.svg"),
            "summary_json": os.path.join(os.path.abspath(args.out_dir), "feature_density_summary.json"),
        },
    }

    # Histograms
    save_grouped_hist_svg(
        path=summary["outputs"]["gate_vpmi_hist_svg"],
        title="Gate Feature: greedy_champ_vpmi (expand outcome groups)",
        series={
            "expand_gain": gate_vpmi_gain,
            "expand_harm": gate_vpmi_harm,
            "expand_same": gate_vpmi_same,
        },
        x_min=quantile(gate_vpmi_all, 0.01),
        x_max=quantile(gate_vpmi_all, 0.99),
        bins=60,
        vlines=[(float(args.tau_vpmi), f"tau_vpmi={args.tau_vpmi}")],
    )
    save_grouped_hist_svg(
        path=summary["outputs"]["gate_sfull_hist_svg"],
        title="Gate Feature: greedy_champ_s_full (expand outcome groups)",
        series={
            "expand_gain": gate_sfull_gain,
            "expand_harm": gate_sfull_harm,
            "expand_same": gate_sfull_same,
        },
        x_min=quantile(gate_sfull_all, 0.01),
        x_max=quantile(gate_sfull_all, 0.99),
        bins=60,
        vlines=[(float(args.tau_sfull), f"tau_sfull={args.tau_sfull}")],
    )
    save_grouped_hist_svg(
        path=summary["outputs"]["selector_dfull_hist_svg"],
        title="Selector Feature: d_full = safe.s_full - champ.s_full (switch outcomes)",
        series={
            "switch_gain": dfull_switch_gain,
            "switch_harm": dfull_switch_harm,
            "switch_same": dfull_switch_same,
        },
        x_min=quantile(dfull_all, 0.01),
        x_max=quantile(dfull_all, 0.99),
        bins=60,
        vlines=[(float(args.tau_dfull), f"tau_dfull={args.tau_dfull}")],
    )
    save_grouped_hist_svg(
        path=summary["outputs"]["trigger_dvpmi_hist_svg"],
        title="Trigger Feature: d_vpmi = safe.vpmi - champ.vpmi (switch outcomes)",
        series={
            "switch_gain": dvpmi_switch_gain,
            "switch_harm": dvpmi_switch_harm,
            "switch_same": dvpmi_switch_same,
        },
        x_min=quantile(dvpmi_all, 0.01),
        x_max=quantile(dvpmi_all, 0.99),
        bins=60,
        vlines=[(0.0, "P3 margin=0")],
    )
    save_grouped_hist_svg(
        path=summary["outputs"]["trigger_champ_vpmi_hist_svg"],
        title="Trigger Feature: champ.vpmi (switch outcomes)",
        series={
            "switch_gain": champ_vpmi_switch_gain,
            "switch_harm": champ_vpmi_switch_harm,
            "switch_same": champ_vpmi_switch_same,
        },
        x_min=quantile(champ_vpmi_all, 0.01),
        x_max=quantile(champ_vpmi_all, 0.99),
        bins=60,
        vlines=[(0.0, "P3 champ.vpmi<0")],
    )

    with open(summary["outputs"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", summary["outputs"]["summary_json"])
    print("[saved]", summary["outputs"]["gate_vpmi_hist_svg"])
    print("[saved]", summary["outputs"]["gate_sfull_hist_svg"])
    print("[saved]", summary["outputs"]["selector_dfull_hist_svg"])
    print("[saved]", summary["outputs"]["trigger_dvpmi_hist_svg"])
    print("[saved]", summary["outputs"]["trigger_champ_vpmi_hist_svg"])


if __name__ == "__main__":
    main()
