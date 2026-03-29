#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from eval_selector_tradeoff import is_success_heur, is_success_strict


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


def norm_text(x: Any) -> str:
    s = str("" if x is None else x).lower().strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    s2 = "".join(out)
    return " ".join(s2.split())


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


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(vals: Sequence[float]) -> List[float]:
    pairs = sorted((float(v), i) for i, v in enumerate(vals))
    out = [0.0] * len(vals)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            out[pairs[k][1]] = float(avg_rank)
        i = j
    return out


def spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx = rankdata(xs)
    ry = rankdata(ys)
    return pearson_corr(rx, ry)


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
        best = max(best, abs((ia / na) - (ib / nb)))
    return float(best)


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


def best_threshold(pos_vals: Sequence[float], neg_vals: Sequence[float]) -> Optional[Dict[str, Any]]:
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
    cands = sorted(set(cands))
    if len(cands) == 0:
        return None

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
        prec = None if sel == 0 else float(tp / sel)
        return {
            "direction": direction,
            "tau": float(tau),
            "tp": int(tp),
            "fp": int(fp),
            "tpr": tpr,
            "fpr": fpr,
            "youden": float(tpr - fpr),
            "precision": prec,
            "selected_rate": float(sel / (len(pos) + len(neg))),
        }

    for tau in cands:
        for direction in ("le", "ge"):
            cur = eval_one(direction, tau)
            if best is None:
                best = cur
                continue
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
    best["ks"] = ks_distance(pos, neg)
    return best


def load_pair(beam_dir: str, eval_mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    ps = list(csv.DictReader(open(os.path.join(beam_dir, "per_sample.csv"), encoding="utf-8")))
    pc = list(csv.DictReader(open(os.path.join(beam_dir, "per_candidate.csv"), encoding="utf-8")))

    by_id: Dict[str, Dict[str, Any]] = {}
    answer_counts: Counter = Counter()
    for r in ps:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", ""))
        if sid == "":
            continue
        q = str(r.get("question", ""))
        a = str(r.get("answer", ""))
        answer_counts[norm_text(a)] += 1
        mode = str(eval_mode).lower().strip()
        if mode == "auto":
            mm = str(r.get("eval_match_mode", "")).strip().lower()
            mode = mm if mm in {"strict", "heuristic"} else "heuristic"
        if mode not in {"strict", "heuristic"}:
            mode = "heuristic"
        if mode == "strict" and str(r.get("is_success_strict", "")).strip() != "":
            champ_ok = as_bool(r.get("is_success_strict", ""))
        elif mode == "heuristic" and str(r.get("is_success_heuristic", "")).strip() != "":
            champ_ok = as_bool(r.get("is_success_heuristic", ""))
        else:
            champ_ok = as_bool(r.get("is_success", ""))
        by_id[sid] = {
            "sid": sid,
            "question": q,
            "answer": a,
            "mode": mode,
            "champ_ok": bool(champ_ok),
            "q_len": len([t for t in q.split() if t.strip() != ""]),
            "ans_rarity": None,  # fill later
            "cands": [],
        }

    for k, v in list(by_id.items()):
        a_norm = norm_text(v["answer"])
        freq = int(answer_counts.get(a_norm, 1))
        v["ans_rarity"] = float(-math.log(float(freq) / max(1.0, float(len(by_id)))))

    for r in pc:
        sid = str(r.get("id", ""))
        s = by_id.get(sid)
        if s is None:
            continue
        idx_f = safe_float(r.get("cand_idx"))
        if idx_f is None:
            continue
        s_q = safe_float(r.get("s_ans_q"))
        s_core = safe_float(r.get("s_core_img"))
        vpmi = None if s_q is None or s_core is None else float(s_core - s_q)
        c = {
            "idx": int(idx_f),
            "text": str(r.get("text", "")),
            "short_answer": str(r.get("short_answer", "")),
            "is_champion": as_bool(r.get("is_champion", "")),
            "s_ans_q": s_q,
            "s_core_img": s_core,
            "vpmi": vpmi,
            "core_vpmi_toks": parse_json_float_list(r.get("core_vpmi_toks_json")),
        }
        s["cands"].append(c)

    # Build rows.
    scale_rows: List[Dict[str, Any]] = []
    selector_rows: List[Dict[str, Any]] = []
    tokenwise_available = False

    # global z-score stats for negative control J1.
    all_vpmi = [
        float(c["vpmi"])
        for s in by_id.values()
        for c in s["cands"]
        if c.get("vpmi") is not None and math.isfinite(float(c["vpmi"]))
    ]
    mu = float(sum(all_vpmi) / len(all_vpmi)) if len(all_vpmi) > 0 else 0.0
    sigma = float(np.std(np.array(all_vpmi, dtype=np.float64))) if len(all_vpmi) > 1 else 1.0
    if sigma <= 1e-12:
        sigma = 1.0

    for s in by_id.values():
        cands = s["cands"]
        if len(cands) == 0:
            continue
        champ = next((c for c in cands if bool(c.get("is_champion", False))), None)
        if champ is None:
            finite = [c for c in cands if c.get("vpmi") is not None]
            if len(finite) == 0:
                continue
            champ = max(finite, key=lambda x: float(x["vpmi"]))
        pool = [c for c in cands if int(c["idx"]) != int(champ["idx"])]
        ranks = rank_desc_pct_map(cands, "vpmi")
        champ_rank = ranks.get(int(champ["idx"]))

        # scale invariance rows (candidate level)
        for c in cands:
            vr = ranks.get(int(c["idx"]))
            if c.get("vpmi") is None or vr is None:
                continue
            scale_rows.append(
                {
                    "sid": s["sid"],
                    "q_len": float(s["q_len"]),
                    "ans_rarity": float(s["ans_rarity"]),
                    "vpmi_abs": float(c["vpmi"]),
                    "vpmi_rank_pct": float(vr),
                    "is_champion": bool(c.get("is_champion", False)),
                }
            )

        # selector better/worse rows
        for c in pool:
            if c.get("vpmi") is None or champ.get("vpmi") is None:
                continue
            cand_ok = (
                is_success_strict(s["answer"], c["text"], c["short_answer"])
                if s["mode"] == "strict"
                else is_success_heur(s["question"], s["answer"], c["text"], c["short_answer"])
            )
            better = bool((not s["champ_ok"]) and cand_ok)
            worse = bool(s["champ_ok"] and (not cand_ok))
            if not better and not worse:
                continue
            if len(c.get("core_vpmi_toks") or []) > 0 and len(champ.get("core_vpmi_toks") or []) > 0:
                tokenwise_available = True
                k = int(max(1, min(2, len(c["core_vpmi_toks"]), len(champ["core_vpmi_toks"]))))
                c_suffix_min = float(min(c["core_vpmi_toks"][-k:]))
                ch_suffix_min = float(min(champ["core_vpmi_toks"][-k:]))
                d_suffix_min = float(c_suffix_min - ch_suffix_min)
            else:
                d_suffix_min = None
            c_prefix_mean = (
                float(sum(c["core_vpmi_toks"][: max(1, min(2, len(c["core_vpmi_toks"])))]) / max(1, min(2, len(c["core_vpmi_toks"]))))
                if len(c.get("core_vpmi_toks") or []) > 0
                else None
            )
            d_rank = None
            if ranks.get(int(c["idx"])) is not None and champ_rank is not None:
                d_rank = float(float(ranks[int(c["idx"])]) - float(champ_rank))
            d_vpmi = float(float(c["vpmi"]) - float(champ["vpmi"]))
            z_c = float((float(c["vpmi"]) - mu) / sigma)
            z_ch = float((float(champ["vpmi"]) - mu) / sigma)
            selector_rows.append(
                {
                    "sid": s["sid"],
                    "y": 1 if better else 0,
                    "label": "better" if better else "worse",
                    "d_rank_vpmi_pct": d_rank,
                    "d_vpmi_mean": d_vpmi,  # J2
                    "z_vpmi_cand_global": z_c,  # J1
                    "z_vpmi_champ_global": z_ch,  # J1
                    "d_z_vpmi_global": float(z_c - z_ch),  # almost linear to d_vpmi_mean
                    "d_vpmi_suffix_min_k": d_suffix_min,
                    "c_vpmi_prefix_mean_k": c_prefix_mean,
                    "cand_core_vpmi_toks": [float(x) for x in c.get("core_vpmi_toks", [])],
                    "champ_core_vpmi_toks": [float(x) for x in champ.get("core_vpmi_toks", [])],
                }
            )

    meta = {"mu_vpmi_global": mu, "sigma_vpmi_global": sigma, "tokenwise_available": tokenwise_available}
    return scale_rows, selector_rows, meta


def save_scatter(path: str, x: Sequence[float], y: Sequence[float], xlab: str, ylab: str, title: str, max_points: int = 30000) -> None:
    xx = [float(a) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    yy = [float(b) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(xx) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        ax.set_title(title + " (no data)")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return
    if len(xx) > int(max_points):
        rng = random.Random(42)
        idxs = list(range(len(xx)))
        rng.shuffle(idxs)
        idxs = idxs[: int(max_points)]
        xx = [xx[i] for i in idxs]
        yy = [yy[i] for i in idxs]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
    ax.scatter(xx, yy, s=5, alpha=0.12, color="#1f77b4", edgecolors="none")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_tail_curve(path: str, selector_rows: List[Dict[str, Any]]) -> None:
    # Requires d_vpmi_suffix_min_k and c_vpmi_prefix_mean_k, not full token curves.
    # We visualize proxy decomposition: prefix mean vs suffix min by better/worse.
    b_pref = [float(r["c_vpmi_prefix_mean_k"]) for r in selector_rows if int(r["y"]) == 1 and r.get("c_vpmi_prefix_mean_k") is not None]
    w_pref = [float(r["c_vpmi_prefix_mean_k"]) for r in selector_rows if int(r["y"]) == 0 and r.get("c_vpmi_prefix_mean_k") is not None]
    b_suf = [float(r["d_vpmi_suffix_min_k"]) for r in selector_rows if int(r["y"]) == 1 and r.get("d_vpmi_suffix_min_k") is not None]
    w_suf = [float(r["d_vpmi_suffix_min_k"]) for r in selector_rows if int(r["y"]) == 0 and r.get("d_vpmi_suffix_min_k") is not None]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    labels = ["Prefix Mean(cand)", "Suffix Min Delta(cand-champ)"]
    b_means = [float(np.mean(b_pref)) if len(b_pref) > 0 else np.nan, float(np.mean(b_suf)) if len(b_suf) > 0 else np.nan]
    w_means = [float(np.mean(w_pref)) if len(w_pref) > 0 else np.nan, float(np.mean(w_suf)) if len(w_suf) > 0 else np.nan]
    x = np.arange(len(labels))
    ax.plot(x, b_means, marker="o", color="#1f77b4", label="better")
    ax.plot(x, w_means, marker="o", color="#d62728", label="worse")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean feature value")
    ax.set_title("Prefix vs Suffix proxy gap (better vs worse)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _mean_curve(seqs: Sequence[Sequence[float]], max_t: int) -> Tuple[List[float], List[int]]:
    sums = [0.0 for _ in range(int(max_t))]
    cnts = [0 for _ in range(int(max_t))]
    for s in seqs:
        t = int(min(int(max_t), len(s)))
        for i in range(t):
            v = safe_float(s[i])
            if v is None:
                continue
            sums[i] += float(v)
            cnts[i] += 1
    means: List[float] = []
    for i in range(int(max_t)):
        if cnts[i] <= 0:
            means.append(float("nan"))
        else:
            means.append(float(sums[i] / cnts[i]))
    return means, cnts


def save_token_position_curves(path_png: str, path_csv: str, selector_rows: List[Dict[str, Any]], max_t: int = 8) -> None:
    better_c = [
        [float(x) for x in r.get("cand_core_vpmi_toks", [])]
        for r in selector_rows
        if int(r["y"]) == 1 and len(r.get("cand_core_vpmi_toks", [])) > 0
    ]
    worse_c = [
        [float(x) for x in r.get("cand_core_vpmi_toks", [])]
        for r in selector_rows
        if int(r["y"]) == 0 and len(r.get("cand_core_vpmi_toks", [])) > 0
    ]

    better_d: List[List[float]] = []
    worse_d: List[List[float]] = []
    for r in selector_rows:
        c = [float(x) for x in r.get("cand_core_vpmi_toks", [])]
        h = [float(x) for x in r.get("champ_core_vpmi_toks", [])]
        m = int(min(len(c), len(h)))
        if m <= 0:
            continue
        d = [float(c[i] - h[i]) for i in range(m)]
        if int(r["y"]) == 1:
            better_d.append(d)
        else:
            worse_d.append(d)

    b_m, b_n = _mean_curve(better_c, int(max_t))
    w_m, w_n = _mean_curve(worse_c, int(max_t))
    bd_m, bd_n = _mean_curve(better_d, int(max_t))
    wd_m, wd_n = _mean_curve(worse_d, int(max_t))

    rows: List[Dict[str, Any]] = []
    for i in range(int(max_t)):
        rows.append(
            {
                "token_pos_1based": int(i + 1),
                "cand_better_mean": b_m[i],
                "cand_better_n": int(b_n[i]),
                "cand_worse_mean": w_m[i],
                "cand_worse_n": int(w_n[i]),
                "delta_better_mean": bd_m[i],
                "delta_better_n": int(bd_n[i]),
                "delta_worse_mean": wd_m[i],
                "delta_worse_n": int(wd_n[i]),
            }
        )
    write_csv(path_csv, rows)

    x = np.arange(1, int(max_t) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax0, ax1 = axes

    ax0.plot(x, b_m, marker="o", color="#1f77b4", label="better: cand vpmi(t)")
    ax0.plot(x, w_m, marker="o", color="#d62728", label="worse: cand vpmi(t)")
    ax0.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax0.set_xlabel("Core token position t")
    ax0.set_ylabel("Mean VPMI(t)")
    ax0.set_title("Candidate tokenwise VPMI by label")
    ax0.legend(loc="best", fontsize=8)

    ax1.plot(x, bd_m, marker="o", color="#1f77b4", label="better: cand-champ Δ(t)")
    ax1.plot(x, wd_m, marker="o", color="#d62728", label="worse: cand-champ Δ(t)")
    ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Core token position t")
    ax1.set_ylabel("Mean ΔVPMI(t)")
    ax1.set_title("Tokenwise delta (candidate - champion)")
    ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Proof package for scale invariance + negative controls.")
    ap.add_argument("--beam_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["auto", "strict", "heuristic"])
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    scale_rows, selector_rows, meta = load_pair(
        beam_dir=os.path.abspath(args.beam_dir),
        eval_mode=str(args.eval_mode),
    )

    # Proof 1: scale invariance correlations.
    corr_rows: List[Dict[str, Any]] = []
    for cov in ["q_len", "ans_rarity"]:
        for feat in ["vpmi_abs", "vpmi_rank_pct"]:
            xs = [float(r[cov]) for r in scale_rows if r.get(cov) is not None and r.get(feat) is not None]
            ys = [float(r[feat]) for r in scale_rows if r.get(cov) is not None and r.get(feat) is not None]
            corr_rows.append(
                {
                    "covariate": cov,
                    "feature": feat,
                    "n": int(len(xs)),
                    "pearson_r": pearson_corr(xs, ys),
                    "spearman_rho": spearman_corr(xs, ys),
                }
            )
            save_scatter(
                path=os.path.join(out_dir, f"scatter_{feat}_vs_{cov}.png"),
                x=xs,
                y=ys,
                xlab=cov,
                ylab=feat,
                title=f"{feat} vs {cov}",
            )
    write_csv(os.path.join(out_dir, "proof1_scale_invariance_correlations.csv"), corr_rows)

    # Negative controls + candidate features on selector rows (better vs worse).
    feat_defs: List[Tuple[str, str]] = [
        ("d_rank_vpmi_pct", "proposed_nonparametric"),
        ("d_vpmi_suffix_min_k", "proposed_structural"),
        ("c_vpmi_prefix_mean_k", "proposed_structural"),
        ("d_vpmi_mean", "J2_delta_mean"),
        ("z_vpmi_cand_global", "J1_global_zscore_abs"),
        ("d_z_vpmi_global", "J1_global_zscore_delta"),
    ]
    sep_rows: List[Dict[str, Any]] = []
    for feat, tag in feat_defs:
        pos = [float(r[feat]) for r in selector_rows if int(r["y"]) == 1 and r.get(feat) is not None and math.isfinite(float(r[feat]))]
        neg = [float(r[feat]) for r in selector_rows if int(r["y"]) == 0 and r.get(feat) is not None and math.isfinite(float(r[feat]))]
        best = best_threshold(pos, neg)
        row: Dict[str, Any] = {
            "feature": feat,
            "group": tag,
            "n_better": int(len(pos)),
            "n_worse": int(len(neg)),
            "ks": ks_distance(pos, neg),
        }
        if best is not None:
            row.update(best)
        sep_rows.append(row)
    sep_rows.sort(key=lambda r: -(safe_float(r.get("ks")) or -1e9))
    write_csv(os.path.join(out_dir, "proof3_negative_controls_and_features.csv"), sep_rows)

    # Proof 2 visualization.
    tail_proxy_path = None
    tail_token_png = None
    tail_token_csv = None
    if meta.get("tokenwise_available", False):
        tail_proxy_path = os.path.join(out_dir, "proof2_prefix_suffix_proxy.png")
        save_tail_curve(tail_proxy_path, selector_rows)
        tail_token_png = os.path.join(out_dir, "proof2_token_position_curves.png")
        tail_token_csv = os.path.join(out_dir, "proof2_token_position_curves.csv")
        save_token_position_curves(
            path_png=tail_token_png,
            path_csv=tail_token_csv,
            selector_rows=selector_rows,
            max_t=8,
        )

    # Save raw rows for audit.
    write_csv(os.path.join(out_dir, "selector_rows_long.csv"), selector_rows)

    summary = {
        "inputs": {
            "beam_dir": os.path.abspath(args.beam_dir),
            "eval_mode": str(args.eval_mode),
        },
        "counts": {
            "n_scale_rows": int(len(scale_rows)),
            "n_selector_rows": int(len(selector_rows)),
            "n_better": int(sum(1 for r in selector_rows if int(r["y"]) == 1)),
            "n_worse": int(sum(1 for r in selector_rows if int(r["y"]) == 0)),
            "tokenwise_available": bool(meta.get("tokenwise_available", False)),
        },
        "meta": meta,
        "outputs": {
            "proof1_correlations_csv": os.path.join(out_dir, "proof1_scale_invariance_correlations.csv"),
            "proof3_features_csv": os.path.join(out_dir, "proof3_negative_controls_and_features.csv"),
            "selector_rows_long_csv": os.path.join(out_dir, "selector_rows_long.csv"),
            "proof2_proxy_png": tail_proxy_path,
            "proof2_token_curve_png": tail_token_png,
            "proof2_token_curve_csv": tail_token_csv,
            "scatter_pngs": [
                os.path.join(out_dir, "scatter_vpmi_abs_vs_q_len.png"),
                os.path.join(out_dir, "scatter_vpmi_rank_pct_vs_q_len.png"),
                os.path.join(out_dir, "scatter_vpmi_abs_vs_ans_rarity.png"),
                os.path.join(out_dir, "scatter_vpmi_rank_pct_vs_ans_rarity.png"),
            ],
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "proof1_scale_invariance_correlations.csv"))
    print("[saved]", os.path.join(out_dir, "proof3_negative_controls_and_features.csv"))
    print("[saved]", os.path.join(out_dir, "selector_rows_long.csv"))
    if tail_proxy_path is not None:
        print("[saved]", tail_proxy_path)
    if tail_token_png is not None:
        print("[saved]", tail_token_png)
    if tail_token_csv is not None:
        print("[saved]", tail_token_csv)
    for p in summary["outputs"]["scatter_pngs"]:
        print("[saved]", p)
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
