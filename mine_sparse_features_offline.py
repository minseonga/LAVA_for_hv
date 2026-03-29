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


def ks_distance(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    xa = sorted(float(x) for x in a if math.isfinite(float(x)))
    xb = sorted(float(x) for x in b if math.isfinite(float(x)))
    if len(xa) == 0 or len(xb) == 0:
        return None
    vals = sorted(set(xa + xb))
    ia = 0
    ib = 0
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


def best_threshold(
    pos_vals: Sequence[float],
    neg_vals: Sequence[float],
) -> Optional[Dict[str, Any]]:
    pos = [float(x) for x in pos_vals if math.isfinite(float(x))]
    neg = [float(x) for x in neg_vals if math.isfinite(float(x))]
    if len(pos) == 0 or len(neg) == 0:
        return None

    pooled = pos + neg
    cands: List[float] = []
    for q in [i / 100.0 for i in range(1, 100)]:
        v = quantile(pooled, q)
        if v is not None:
            cands.append(float(v))
    if len(cands) == 0:
        return None
    cands = sorted(set(cands))

    npos = float(len(pos))
    nneg = float(len(neg))
    best: Optional[Dict[str, Any]] = None

    def eval_one(direction: str, tau: float) -> Dict[str, Any]:
        if direction == "le":
            tp = sum(1 for x in pos if x <= tau)
            fp = sum(1 for x in neg if x <= tau)
        else:
            tp = sum(1 for x in pos if x >= tau)
            fp = sum(1 for x in neg if x >= tau)
        tpr = float(tp / npos)
        fpr = float(fp / nneg)
        sel = int(tp + fp)
        precision = (None if sel == 0 else float(tp / sel))
        youden = float(tpr - fpr)
        return {
            "direction": direction,
            "tau": float(tau),
            "tp": int(tp),
            "fp": int(fp),
            "tpr": tpr,
            "fpr": fpr,
            "youden": youden,
            "precision": precision,
            "selected": int(sel),
            "selected_rate": float(sel / (len(pos) + len(neg))),
        }

    for tau in cands:
        for direction in ("le", "ge"):
            cur = eval_one(direction, tau)
            if best is None:
                best = cur
                continue
            # prioritize better separation, then precision, then lower selected rate.
            key_cur = (
                float(cur["youden"]),
                float(-1e9 if cur["precision"] is None else cur["precision"]),
                -float(cur["selected_rate"]),
            )
            key_best = (
                float(best["youden"]),
                float(-1e9 if best["precision"] is None else best["precision"]),
                -float(best["selected_rate"]),
            )
            if key_cur > key_best:
                best = cur

    if best is None:
        return None

    q05 = quantile(pooled, 0.05)
    q95 = quantile(pooled, 0.95)
    if q05 is None or q95 is None or float(q95) <= float(q05):
        eps = 0.0
    else:
        eps = float(0.02 * (float(q95) - float(q05)))
    if eps <= 0.0:
        near = None
    else:
        near_cnt = sum(1 for x in pooled if abs(float(x) - float(best["tau"])) <= eps)
        near = float(near_cnt / len(pooled))
    best["near_density_eps2pct_iqr90"] = near
    best["ks"] = ks_distance(pos, neg)
    best["n_pos"] = int(len(pos))
    best["n_neg"] = int(len(neg))
    return best


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


def save_two_group_hist_svg(
    path: str,
    title: str,
    pos_label: str,
    neg_label: str,
    pos_vals: Sequence[float],
    neg_vals: Sequence[float],
    tau: Optional[float],
) -> None:
    p = [float(x) for x in pos_vals if math.isfinite(float(x))]
    n = [float(x) for x in neg_vals if math.isfinite(float(x))]
    pooled = p + n
    if len(pooled) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='320'></svg>")
        return
    x_min = quantile(pooled, 0.01)
    x_max = quantile(pooled, 0.99)
    if x_min is None or x_max is None or x_max <= x_min:
        x_min = min(pooled)
        x_max = max(pooled) + 1e-6
    bins = 60
    cp = hist_counts(p, x_min, x_max, bins)
    cn = hist_counts(n, x_min, x_max, bins)
    ymax = max(1, max(cp) if cp else 0, max(cn) if cn else 0)

    W = 1200
    H = 500
    ml = 80
    mr = 20
    mt = 50
    mb = 70
    pw = W - ml - mr
    ph = H - mt - mb
    bw = pw / bins

    def sx(x: float) -> float:
        return float(ml + (x - x_min) * pw / (x_max - x_min))

    def sy(y: float) -> float:
        return float(mt + ph - y * ph / ymax)

    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>")
    lines.append("<rect x='0' y='0' width='100%' height='100%' fill='white'/>")
    lines.append(f"<text x='{W/2}' y='28' text-anchor='middle' font-size='18' font-family='Arial'>{title}</text>")
    lines.append(f"<line x1='{ml}' y1='{mt+ph}' x2='{ml+pw}' y2='{mt+ph}' stroke='black'/>")
    lines.append(f"<line x1='{ml}' y1='{mt}' x2='{ml}' y2='{mt+ph}' stroke='black'/>")
    for t in [0, int(ymax * 0.25), int(ymax * 0.5), int(ymax * 0.75), int(ymax)]:
        y = sy(float(t))
        lines.append(f"<line x1='{ml-4}' y1='{y:.2f}' x2='{ml}' y2='{y:.2f}' stroke='black'/>")
        lines.append(f"<text x='{ml-8}' y='{y+4:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{t}</text>")
    for tt in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xv = float(x_min) + tt * (float(x_max) - float(x_min))
        x = sx(xv)
        lines.append(f"<line x1='{x:.2f}' y1='{mt+ph}' x2='{x:.2f}' y2='{mt+ph+4}' stroke='black'/>")
        lines.append(f"<text x='{x:.2f}' y='{mt+ph+20}' text-anchor='middle' font-size='11' font-family='Arial'>{xv:.3f}</text>")
    for i in range(bins):
        if cp[i] > 0:
            x = ml + i * bw
            y = sy(float(cp[i]))
            h = mt + ph - y
            lines.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(1.0,bw-0.6):.2f}' height='{h:.2f}' fill='#1f77b4' fill-opacity='0.45' stroke='#1f77b4'/>")
        if cn[i] > 0:
            x = ml + i * bw
            y = sy(float(cn[i]))
            h = mt + ph - y
            lines.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(1.0,bw-0.6):.2f}' height='{h:.2f}' fill='#d62728' fill-opacity='0.45' stroke='#d62728'/>")
    if tau is not None and x_min <= float(tau) <= x_max:
        x = sx(float(tau))
        lines.append(f"<line x1='{x:.2f}' y1='{mt}' x2='{x:.2f}' y2='{mt+ph}' stroke='#111' stroke-width='1.5' stroke-dasharray='6,4'/>")
        lines.append(f"<text x='{x+4:.2f}' y='{mt+14}' font-size='11' font-family='Arial'>tau={tau:.4f}</text>")

    lines.append(f"<rect x='{ml+10}' y='{mt+10}' width='12' height='12' fill='#1f77b4' fill-opacity='0.45' stroke='#1f77b4'/>")
    lines.append(f"<text x='{ml+28}' y='{mt+20}' font-size='12' font-family='Arial'>{pos_label} (n={len(p)})</text>")
    lines.append(f"<rect x='{ml+260}' y='{mt+10}' width='12' height='12' fill='#d62728' fill-opacity='0.45' stroke='#d62728'/>")
    lines.append(f"<text x='{ml+278}' y='{mt+20}' font-size='12' font-family='Arial'>{neg_label} (n={len(n)})</text>")
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine sparse-separation features for gate and beam-internal selector (offline).")
    ap.add_argument("--greedy_dir", type=str, required=True)
    ap.add_argument("--beam_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--topk_hist", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_selector_tradeoff import load_samples

    g_samples = load_samples(args.greedy_dir, eval_mode=str(args.eval_mode))
    b_samples = load_samples(args.beam_dir, eval_mode=str(args.eval_mode))
    g_by_id = {str(s.sid): s for s in g_samples}
    b_by_id = {str(s.sid): s for s in b_samples}
    ids_all = sorted(set(g_by_id.keys()) & set(b_by_id.keys()))

    # Stage-1: gate mining (gain vs harm only).
    # gain: greedy wrong -> beam champ correct
    # harm: greedy correct -> beam champ wrong
    gate_pairs: Dict[str, Tuple[List[float], List[float]]] = {
        "g_champ_vpmi": ([], []),
        "g_champ_s_full": ([], []),
        "g_champ_s_q": ([], []),
        "g_champ_s_core": ([], []),
        "g_champ_visual_pmi": ([], []),
    }
    n_gain = 0
    n_harm = 0
    for sid in ids_all:
        gs = g_by_id[sid]
        bs = b_by_id[sid]
        gain = bool((not gs.base_ok) and bs.base_ok)
        harm = bool(gs.base_ok and (not bs.base_ok))
        if not gain and not harm:
            continue
        if gain:
            n_gain += 1
        else:
            n_harm += 1

        feats = {
            "g_champ_vpmi": gs.champ.vpmi,
            "g_champ_s_full": gs.champ.s_full,
            "g_champ_s_q": gs.champ.s_q,
            "g_champ_s_core": gs.champ.s_core,
            "g_champ_visual_pmi": gs.champ.visual_pmi,
        }
        for k, v in feats.items():
            if v is None:
                continue
            if gain:
                gate_pairs[k][0].append(float(v))
            else:
                gate_pairs[k][1].append(float(v))

    gate_rows: List[Dict[str, Any]] = []
    for feat, (pos_vals, neg_vals) in gate_pairs.items():
        best = best_threshold(pos_vals, neg_vals)
        if best is None:
            continue
        row = {
            "stage": "gate",
            "feature": feat,
            "pos_label": "gain",
            "neg_label": "harm",
            **best,
        }
        # sparse-friendly score: separation with boundary sparsity penalty.
        near = row.get("near_density_eps2pct_iqr90")
        ks = row.get("ks")
        if ks is None or near is None:
            row["sparse_score"] = None
        else:
            row["sparse_score"] = float(ks * (1.0 - float(near)))
        gate_rows.append(row)
    gate_rows.sort(
        key=lambda r: (
            -(safe_float(r.get("sparse_score")) or -1e9),
            -(safe_float(r.get("ks")) or -1e9),
            -(safe_float(r.get("youden")) or -1e9),
        )
    )

    # Stage-2: beam internal mining (better-vs-worse candidates relative to beam champ).
    # better: champ wrong, candidate correct
    # worse:  champ correct, candidate wrong
    sel_pairs: Dict[str, Tuple[List[float], List[float]]] = {
        "c_vpmi": ([], []),
        "c_vpmi_core_min_prior_masked": ([], []),
        "c_vpmi_word_min": ([], []),
        "c_s_full": ([], []),
        "c_s_q": ([], []),
        "c_s_core": ([], []),
        "d_vpmi": ([], []),
        "d_vpmi_core_min_prior_masked": ([], []),
        "d_vpmi_word_min": ([], []),
        "d_s_full": ([], []),
        "d_s_q": ([], []),
        "d_s_core": ([], []),
        "d_visual_pmi": ([], []),
    }
    n_better = 0
    n_worse = 0
    for sid in ids_all:
        bs = b_by_id[sid]
        champ = bs.champ
        champ_ok = bool(bs.base_ok)
        for c in bs.pool:
            cand_ok = bool(bs.safe_ok_by_idx.get(int(c.idx), False))
            better = bool((not champ_ok) and cand_ok)
            worse = bool(champ_ok and (not cand_ok))
            if not better and not worse:
                continue
            if better:
                n_better += 1
            else:
                n_worse += 1
            feats_abs = {
                "c_vpmi": c.vpmi,
                "c_vpmi_core_min_prior_masked": c.vpmi_core_min_prior_masked,
                "c_vpmi_word_min": c.vpmi_word_min,
                "c_s_full": c.s_full,
                "c_s_q": c.s_q,
                "c_s_core": c.s_core,
            }
            feats_delta = {
                "d_vpmi": (None if c.vpmi is None or champ.vpmi is None else float(c.vpmi - champ.vpmi)),
                "d_vpmi_core_min_prior_masked": (
                    None
                    if c.vpmi_core_min_prior_masked is None or champ.vpmi_core_min_prior_masked is None
                    else float(c.vpmi_core_min_prior_masked - champ.vpmi_core_min_prior_masked)
                ),
                "d_vpmi_word_min": (
                    None
                    if c.vpmi_word_min is None or champ.vpmi_word_min is None
                    else float(c.vpmi_word_min - champ.vpmi_word_min)
                ),
                "d_s_full": (None if c.s_full is None or champ.s_full is None else float(c.s_full - champ.s_full)),
                "d_s_q": (None if c.s_q is None or champ.s_q is None else float(c.s_q - champ.s_q)),
                "d_s_core": (None if c.s_core is None or champ.s_core is None else float(c.s_core - champ.s_core)),
                "d_visual_pmi": (
                    None if c.visual_pmi is None or champ.visual_pmi is None else float(c.visual_pmi - champ.visual_pmi)
                ),
            }
            for k, v in {**feats_abs, **feats_delta}.items():
                if v is None:
                    continue
                if better:
                    sel_pairs[k][0].append(float(v))
                else:
                    sel_pairs[k][1].append(float(v))

    sel_rows: List[Dict[str, Any]] = []
    for feat, (pos_vals, neg_vals) in sel_pairs.items():
        best = best_threshold(pos_vals, neg_vals)
        if best is None:
            continue
        row = {
            "stage": "selector",
            "feature": feat,
            "pos_label": "better",
            "neg_label": "worse",
            **best,
        }
        near = row.get("near_density_eps2pct_iqr90")
        ks = row.get("ks")
        if ks is None or near is None:
            row["sparse_score"] = None
        else:
            row["sparse_score"] = float(ks * (1.0 - float(near)))
        sel_rows.append(row)
    sel_rows.sort(
        key=lambda r: (
            -(safe_float(r.get("sparse_score")) or -1e9),
            -(safe_float(r.get("ks")) or -1e9),
            -(safe_float(r.get("youden")) or -1e9),
        )
    )

    gate_csv = os.path.join(args.out_dir, "stage1_gate_sparse_ranking.csv")
    sel_csv = os.path.join(args.out_dir, "stage2_selector_sparse_ranking.csv")
    write_csv(gate_csv, gate_rows)
    write_csv(sel_csv, sel_rows)

    # Save top feature histograms.
    topk = int(max(1, args.topk_hist))
    hist_paths: List[str] = []
    for i, r in enumerate(gate_rows[:topk]):
        feat = str(r["feature"])
        pvals, nvals = gate_pairs.get(feat, ([], []))
        hp = os.path.join(args.out_dir, f"hist_stage1_top{i+1}_{feat}.svg")
        save_two_group_hist_svg(
            path=hp,
            title=f"Stage1 Gate: {feat} (gain vs harm)",
            pos_label="gain",
            neg_label="harm",
            pos_vals=pvals,
            neg_vals=nvals,
            tau=safe_float(r.get("tau")),
        )
        hist_paths.append(hp)
    for i, r in enumerate(sel_rows[:topk]):
        feat = str(r["feature"])
        pvals, nvals = sel_pairs.get(feat, ([], []))
        hp = os.path.join(args.out_dir, f"hist_stage2_top{i+1}_{feat}.svg")
        save_two_group_hist_svg(
            path=hp,
            title=f"Stage2 Selector: {feat} (better vs worse)",
            pos_label="better",
            neg_label="worse",
            pos_vals=pvals,
            neg_vals=nvals,
            tau=safe_float(r.get("tau")),
        )
        hist_paths.append(hp)

    summary = {
        "inputs": {
            "greedy_dir": os.path.abspath(args.greedy_dir),
            "beam_dir": os.path.abspath(args.beam_dir),
            "eval_mode": str(args.eval_mode),
        },
        "counts": {
            "n_ids_intersection": int(len(ids_all)),
            "stage1_gain_n": int(n_gain),
            "stage1_harm_n": int(n_harm),
            "stage2_better_n": int(n_better),
            "stage2_worse_n": int(n_worse),
        },
        "outputs": {
            "stage1_csv": os.path.abspath(gate_csv),
            "stage2_csv": os.path.abspath(sel_csv),
            "hist_svgs": [os.path.abspath(x) for x in hist_paths],
        },
    }
    summary_json = os.path.join(args.out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[saved]", gate_csv)
    print("[saved]", sel_csv)
    print("[saved]", summary_json)
    for p in hist_paths:
        print("[saved]", p)


if __name__ == "__main__":
    main()
