#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
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


def norm(x: Any) -> str:
    s = str("" if x is None else x).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_clause(text: str) -> str:
    s = str(text or "").strip()
    s = re.split(r"[\n\.!?;]", s)[0].strip()
    return s


def contains_whole(needle: str, hay: str) -> bool:
    if needle == "":
        return False
    return re.search(rf"(^|\s){re.escape(needle)}(\s|$)", hay) is not None


def singularize_word(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def map_gender_tokens(s: str) -> str:
    out: List[str] = []
    for t in s.split():
        if t in {"man", "male", "boy", "gentleman", "guy"}:
            out.append("male")
        elif t in {"woman", "female", "girl", "lady"}:
            out.append("female")
        else:
            out.append(t)
    return " ".join(out)


def first_polarity(s: str) -> Optional[str]:
    m = re.match(r"^(yes|no)\b", s)
    return m.group(1) if m else None


def is_success_heur(question: str, answer: str, pred_text: str, pred_short: str = "") -> bool:
    q = norm(question)
    gt = norm(answer)
    pt = norm(pred_text)
    ps = norm(pred_short)
    if gt == "":
        return False

    if gt == ps or gt == pt:
        return True

    if gt in {"yes", "no"}:
        pol = first_polarity(pt) or first_polarity(ps)
        if pol is not None:
            return pol == gt
        return contains_whole(gt, pt)

    gt_gender = map_gender_tokens(gt)
    pt_gender = map_gender_tokens(pt)
    ps_gender = map_gender_tokens(ps)
    if gt_gender in {"male", "female"}:
        if contains_whole(gt_gender, pt_gender) or contains_whole(gt_gender, ps_gender):
            return True

    if len(gt.split()) >= 2:
        if contains_whole(gt, pt) or contains_whole(gt, ps):
            return True

    if len(gt.split()) == 1:
        g = singularize_word(gt)
        toks_t = [singularize_word(t) for t in pt.split()]
        toks_s = [singularize_word(t) for t in ps.split()]
        if g in toks_t or g in toks_s:
            return True

    if contains_whole(gt, pt):
        return True
    if ("what color" in q or "which color" in q) and contains_whole(gt, pt):
        return True
    if ("which side" in q or "left or right" in q) and contains_whole(gt, pt):
        return True
    return False


def is_success_strict(answer: str, pred_text: str, pred_short: str = "") -> bool:
    pred = first_clause(pred_short if str(pred_short).strip() else pred_text)
    return norm(pred) == norm(answer)


def percentile_rank(vals: Sequence[float], v: float) -> Optional[float]:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    if len(xs) == 0:
        return None
    return float(sum(1 for x in xs if x <= float(v)) / len(xs))


def quantile_linear(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(x) for x in vals if math.isfinite(float(x)))
    if len(xs) == 0:
        return None
    qq = min(1.0, max(0.0, float(q)))
    if len(xs) == 1:
        return float(xs[0])
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = float(pos - lo)
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


@dataclass
class Candidate:
    idx: int
    text: str
    short: str
    core_len: Optional[int]
    s_full: Optional[float]
    s_q: Optional[float]
    s_q_min: Optional[float]
    p_q: Optional[float]
    s_core: Optional[float]
    s_core_min: Optional[float]
    vpmi: Optional[float]
    vpmi_core_min: Optional[float]
    vpmi_core_min_raw: Optional[float]
    vpmi_core_min_prior_masked: Optional[float]
    vpmi_word_min: Optional[float]
    vpmi_core_tail_min: Optional[float]
    vpmi_core_min_pos_norm: Optional[float]
    vpmi_core_min_mean_gap: Optional[float]
    vpmi_core_sign_flip_count: Optional[float]
    vpmi_core_mean: Optional[float]
    margin_core_img_min: Optional[float]
    margin_core_img_mean: Optional[float]
    visual_pmi: Optional[float]
    is_champion: bool
    is_safe_existing: bool


@dataclass
class Sample:
    sid: str
    question: str
    answer: str
    eval_mode: str
    base_ok: bool
    champ: Candidate
    pool: List[Candidate]
    safe_ok_by_idx: Dict[int, bool]
    champ_illusion_gap: Optional[float]


def score_candidate(s: Sample, c: Candidate) -> bool:
    if s.eval_mode == "strict":
        return bool(is_success_strict(s.answer, c.text, c.short))
    return bool(is_success_heur(s.question, s.answer, c.text, c.short))


def rank_map(cands: Sequence[Candidate], feature: str, champ: Candidate) -> Dict[int, float]:
    vals: Dict[int, Optional[float]] = {}
    for c in cands:
        if feature == "vpmi":
            vals[c.idx] = c.vpmi
        elif feature == "visual_pmi":
            vals[c.idx] = c.visual_pmi
        elif feature == "minus_s_q":
            vals[c.idx] = (None if c.s_q is None else float(-c.s_q))
        elif feature == "minus_m_full":
            if c.s_full is None or champ.s_full is None:
                vals[c.idx] = None
            else:
                vals[c.idx] = float(c.s_full - champ.s_full)
        else:
            vals[c.idx] = None
    finite = [float(v) for v in vals.values() if v is not None]
    out: Dict[int, float] = {}
    if not finite:
        for c in cands:
            out[c.idx] = 0.0
        return out
    for c in cands:
        v = vals[c.idx]
        if v is None:
            out[c.idx] = 0.0
        else:
            rr = percentile_rank(finite, float(v))
            out[c.idx] = (0.0 if rr is None else float(rr))
    return out


def rank_desc_map(cands: Sequence[Candidate], key_fn) -> Dict[int, int]:
    vals: List[Tuple[int, float]] = []
    for c in cands:
        v = key_fn(c)
        if v is None or not math.isfinite(float(v)):
            continue
        vals.append((int(c.idx), float(v)))
    vals_sorted = sorted(vals, key=lambda x: float(x[1]), reverse=True)
    rank: Dict[int, int] = {}
    for i, (idx, _) in enumerate(vals_sorted):
        rank[int(idx)] = int(i + 1)
    return rank


def smoothed_vpmi(c: Candidate, sq_clip: float) -> Optional[float]:
    if c.s_core is None or c.s_q is None:
        return None
    sq_eff = max(float(c.s_q), float(sq_clip))
    return float(float(c.s_core) - sq_eff)


def prior_drop_discounted_vpmi(c: Candidate, champ: Candidate, lam: float) -> Optional[float]:
    if c.vpmi is None:
        return None
    # Prior-drop: penalize candidates that are much lower-prior than champion.
    # drop > 0 means candidate is linguistically less plausible than champion.
    if c.s_q is None or champ.s_q is None:
        drop = 0.0
    else:
        drop = max(0.0, float(champ.s_q) - float(c.s_q))
    return float(float(c.vpmi) - float(lam) * float(drop))


def select_candidate(policy: str, s: Sample) -> Optional[Candidate]:
    pool = s.pool
    if len(pool) == 0:
        return None

    if policy == "existing_safe":
        for c in pool:
            if bool(c.is_safe_existing):
                return c
        return None

    if policy == "max_vpmi":
        p = [c for c in pool if c.vpmi is not None]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.vpmi)))

    if policy == "max_vpmi_core_min":
        p = [c for c in pool if c.vpmi_core_min is not None]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.vpmi_core_min)))

    if policy == "max_vpmi_core_min_raw":
        p = [c for c in pool if c.vpmi_core_min_raw is not None]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.vpmi_core_min_raw)))

    if policy == "max_vpmi_core_min_prior_masked":
        p = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.vpmi_core_min_prior_masked)))

    # Prior-masked core-min with deterministic tie-break by VPMI.
    if policy == "max_vpmi_core_min_prior_masked_tb_vpmi":
        p = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        if len(p) == 0:
            return None
        return max(
            p,
            key=lambda x: (
                float(x.vpmi_core_min_prior_masked),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    if policy == "max_vpmi_word_min":
        p = [c for c in pool if c.vpmi_word_min is not None]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.vpmi_word_min)))

    if policy == "max_vpmi_core_tail_min":
        p = [c for c in pool if c.vpmi_core_tail_min is not None]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.vpmi_core_tail_min)))

    if policy == "max_vpmi_core_tail_min_tb_vpmi":
        p = [c for c in pool if c.vpmi_core_tail_min is not None]
        if len(p) == 0:
            return None
        return max(
            p,
            key=lambda x: (
                float(x.vpmi_core_tail_min),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Stable tail selector: prefer stronger tail VPMI and penalize unstable token dynamics.
    if policy == "max_vpmi_core_tail_stable":
        p = [c for c in pool if c.vpmi_core_tail_min is not None]
        if len(p) == 0:
            return None
        return max(
            p,
            key=lambda x: (
                float(x.vpmi_core_tail_min),
                -float(x.vpmi_core_min_mean_gap if x.vpmi_core_min_mean_gap is not None else 1e18),
                -float(x.vpmi_core_sign_flip_count if x.vpmi_core_sign_flip_count is not None else 1e18),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Agreement selector using two minimum-style views to suppress single-view outliers.
    if policy == "agree_tailmin_wordmin":
        p1 = [c for c in pool if c.vpmi_core_tail_min is not None]
        p2 = [c for c in pool if c.vpmi_word_min is not None]
        if len(p1) == 0 or len(p2) == 0:
            return None
        top1 = max(
            p1,
            key=lambda x: (
                float(x.vpmi_core_tail_min),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )
        top2 = max(
            p2,
            key=lambda x: (
                float(x.vpmi_word_min),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )
        if int(top1.idx) == int(top2.idx):
            return top1
        return None

    # Agreement selector: candidate must be top-1 by both vminpm and word-min.
    if policy == "agree_vminpm_wmin":
        p1 = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        p2 = [c for c in pool if c.vpmi_word_min is not None]
        if len(p1) == 0 or len(p2) == 0:
            return None
        top1 = max(
            p1,
            key=lambda x: (
                float(x.vpmi_core_min_prior_masked),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )
        top2 = max(
            p2,
            key=lambda x: (
                float(x.vpmi_word_min),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )
        if int(top1.idx) == int(top2.idx):
            return top1
        return None

    # Agreement + full-score relative gate.
    # format: agree_vminpm_wmin_dfull_le:-0.05
    if policy.startswith("agree_vminpm_wmin_dfull_le:"):
        try:
            thr = float(policy.split(":", 1)[1])
        except Exception:
            return None
        cand = select_candidate("agree_vminpm_wmin", s)
        if cand is None or cand.s_full is None or s.champ.s_full is None:
            return None
        if float(cand.s_full) - float(s.champ.s_full) <= float(thr):
            return cand
        return None

    # Agreement + NTP-min floor (image-conditional core min logprob).
    # format: agree_vminpm_wmin_ntpmin_ge:-4.0
    if policy.startswith("agree_vminpm_wmin_ntpmin_ge:"):
        try:
            thr = float(policy.split(":", 1)[1])
        except Exception:
            return None
        cand = select_candidate("agree_vminpm_wmin", s)
        if cand is None or cand.s_core_min is None:
            return None
        return cand if float(cand.s_core_min) >= float(thr) else None

    # Agreement + NTP spread stability gate:
    # ntp_drop = S_core_img_mean - S_core_img_min (smaller is more stable).
    # format: agree_vminpm_wmin_ntpdrop_le:1.5
    if policy.startswith("agree_vminpm_wmin_ntpdrop_le:"):
        try:
            thr = float(policy.split(":", 1)[1])
        except Exception:
            return None
        cand = select_candidate("agree_vminpm_wmin", s)
        if cand is None or cand.s_core is None or cand.s_core_min is None:
            return None
        ntp_drop = float(cand.s_core - cand.s_core_min)
        return cand if float(ntp_drop) <= float(thr) else None

    # Agreement + VPMI instability gate:
    # vpmi_core_min_mean_gap = mean(vpmi_toks) - min(vpmi_toks), lower is stabler.
    # format: agree_vminpm_wmin_vg_le:1.0
    if policy.startswith("agree_vminpm_wmin_vg_le:"):
        try:
            thr = float(policy.split(":", 1)[1])
        except Exception:
            return None
        cand = select_candidate("agree_vminpm_wmin", s)
        if cand is None or cand.vpmi_core_min_mean_gap is None:
            return None
        return cand if float(cand.vpmi_core_min_mean_gap) <= float(thr) else None

    # Threshold-free soft divergence weighting (product form, exploratory):
    # score = vminpm * (1 - rank(div_proxy)),
    # with short-core guard: if core_len<=2 -> div_proxy=0.
    if policy == "softdiv_vminpm_prod":
        p = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        if len(p) == 0:
            return None

        def div_proxy(c: Candidate) -> float:
            if c.core_len is not None and int(c.core_len) <= 2:
                return 0.0
            if c.vpmi_core_min_mean_gap is None or not math.isfinite(float(c.vpmi_core_min_mean_gap)):
                return 0.0
            return float(max(0.0, float(c.vpmi_core_min_mean_gap)))

        div_vals = [div_proxy(c) for c in p]

        def score(c: Candidate) -> float:
            base = float(c.vpmi_core_min_prior_masked)
            rr = percentile_rank(div_vals, div_proxy(c))
            div_rank = float(0.0 if rr is None else rr)
            w = float(max(0.0, 1.0 - div_rank))
            return float(base * w)

        return max(
            p,
            key=lambda x: (
                score(x),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Threshold-free soft divergence weighting (subtractive, safer than product):
    # score = vminpm - rank(div_proxy)
    if policy == "softdiv_vminpm_sub":
        p = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        if len(p) == 0:
            return None

        def div_proxy(c: Candidate) -> float:
            if c.core_len is not None and int(c.core_len) <= 2:
                return 0.0
            if c.vpmi_core_min_mean_gap is None or not math.isfinite(float(c.vpmi_core_min_mean_gap)):
                return 0.0
            return float(max(0.0, float(c.vpmi_core_min_mean_gap)))

        div_vals = [div_proxy(c) for c in p]

        def score(c: Candidate) -> float:
            base = float(c.vpmi_core_min_prior_masked)
            rr = percentile_rank(div_vals, div_proxy(c))
            div_rank = float(0.0 if rr is None else rr)
            return float(base - div_rank)

        return max(
            p,
            key=lambda x: (
                score(x),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Divergence + margin-proxy cross validation:
    # if div_high and margin_low_proxy then downweight by 0.7
    # - div_high: div_proxy > median(div_proxy)
    # - margin_low_proxy: ntp_drop = (s_core - s_core_min) > median(ntp_drop)
    if policy == "softdiv_xv_proxy":
        p = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        if len(p) == 0:
            return None

        def div_proxy(c: Candidate) -> float:
            if c.core_len is not None and int(c.core_len) <= 2:
                return 0.0
            if c.vpmi_core_min_mean_gap is None or not math.isfinite(float(c.vpmi_core_min_mean_gap)):
                return 0.0
            return float(max(0.0, float(c.vpmi_core_min_mean_gap)))

        def ntp_drop(c: Candidate) -> float:
            if c.s_core is None or c.s_core_min is None:
                return 0.0
            return float(max(0.0, float(c.s_core - c.s_core_min)))

        md_div = quantile_linear([div_proxy(c) for c in p], 0.5)
        md_drop = quantile_linear([ntp_drop(c) for c in p], 0.5)
        if md_div is None:
            md_div = 0.0
        if md_drop is None:
            md_drop = 0.0

        def score(c: Candidate) -> float:
            base = float(c.vpmi_core_min_prior_masked)
            div_high = bool(div_proxy(c) > float(md_div))
            margin_low = bool(ntp_drop(c) > float(md_drop))
            if div_high and margin_low:
                return float(base * 0.7)
            return base

        return max(
            p,
            key=lambda x: (
                score(x),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Formal divergence + true top1-top2 margin cross validation:
    # if div_high and margin_low then downweight by 0.7
    # - div_high: div_proxy > median(div_proxy)
    # - margin_low: margin_core_img_min < median(margin_core_img_min)
    if policy == "softdiv_xv_true_margin":
        p = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        if len(p) == 0:
            return None

        def div_proxy(c: Candidate) -> float:
            if c.core_len is not None and int(c.core_len) <= 2:
                return 0.0
            if c.vpmi_core_min_mean_gap is None or not math.isfinite(float(c.vpmi_core_min_mean_gap)):
                return 0.0
            return float(max(0.0, float(c.vpmi_core_min_mean_gap)))

        md_div_v = quantile_linear([div_proxy(c) for c in p], 0.5)
        md_div = float(0.0 if md_div_v is None else md_div_v)
        margins = [
            float(c.margin_core_img_min)
            for c in p
            if c.margin_core_img_min is not None and math.isfinite(float(c.margin_core_img_min))
        ]
        md_margin_v = quantile_linear(margins, 0.5)
        md_margin = float(0.0 if md_margin_v is None else md_margin_v)

        def score(c: Candidate) -> float:
            base = float(c.vpmi_core_min_prior_masked)
            div_high = bool(div_proxy(c) > float(md_div))
            margin_low = bool(
                c.margin_core_img_min is not None
                and math.isfinite(float(c.margin_core_img_min))
                and float(c.margin_core_img_min) < float(md_margin)
            )
            if div_high and margin_low:
                return float(base * 0.7)
            return base

        return max(
            p,
            key=lambda x: (
                score(x),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Threshold-free soft weighting (sigmoid form):
    # weight(c) = 1 / (1 + exp(z_div(c)))
    # score(c) = vpmi_core_min_prior_masked(c) * weight(c)
    # where z_div is robustly normalized with median/IQR in the sample pool.
    if policy == "softdiv_sigmoid_vminpm":
        p = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        if len(p) == 0:
            return None

        def div_proxy(c: Candidate) -> float:
            if c.core_len is not None and int(c.core_len) <= 2:
                return 0.0
            if c.vpmi_core_min_mean_gap is None or not math.isfinite(float(c.vpmi_core_min_mean_gap)):
                return 0.0
            return float(max(0.0, float(c.vpmi_core_min_mean_gap)))

        divs = [float(div_proxy(c)) for c in p]
        md = quantile_linear(divs, 0.5)
        q1 = quantile_linear(divs, 0.25)
        q3 = quantile_linear(divs, 0.75)
        center = float(0.0 if md is None else md)
        scale = float(1.0 if q1 is None or q3 is None else max(1e-6, float(q3 - q1)))

        def score(c: Candidate) -> float:
            base = float(c.vpmi_core_min_prior_masked)
            z = float((div_proxy(c) - center) / scale)
            z = float(max(-60.0, min(60.0, z)))
            w = float(1.0 / (1.0 + math.exp(z)))
            return float(base * w)

        return max(
            p,
            key=lambda x: (
                score(x),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Sigmoid soft weighting + divergence/margin proxy cross validation:
    # if div_high and margin_low_proxy then extra downweight by 0.7
    if policy == "softdiv_sigmoid_xv_proxy":
        p = [c for c in pool if c.vpmi_core_min_prior_masked is not None]
        if len(p) == 0:
            return None

        def div_proxy(c: Candidate) -> float:
            if c.core_len is not None and int(c.core_len) <= 2:
                return 0.0
            if c.vpmi_core_min_mean_gap is None or not math.isfinite(float(c.vpmi_core_min_mean_gap)):
                return 0.0
            return float(max(0.0, float(c.vpmi_core_min_mean_gap)))

        def ntp_drop(c: Candidate) -> float:
            if c.s_core is None or c.s_core_min is None:
                return 0.0
            return float(max(0.0, float(c.s_core - c.s_core_min)))

        divs = [float(div_proxy(c)) for c in p]
        md = quantile_linear(divs, 0.5)
        q1 = quantile_linear(divs, 0.25)
        q3 = quantile_linear(divs, 0.75)
        center = float(0.0 if md is None else md)
        scale = float(1.0 if q1 is None or q3 is None else max(1e-6, float(q3 - q1)))
        md_div = float(0.0 if md is None else md)
        md_drop_v = quantile_linear([ntp_drop(c) for c in p], 0.5)
        md_drop = float(0.0 if md_drop_v is None else md_drop_v)

        def score(c: Candidate) -> float:
            base = float(c.vpmi_core_min_prior_masked)
            z = float((div_proxy(c) - center) / scale)
            z = float(max(-60.0, min(60.0, z)))
            w = float(1.0 / (1.0 + math.exp(z)))
            div_high = bool(div_proxy(c) > float(md_div))
            margin_low = bool(ntp_drop(c) > float(md_drop))
            if div_high and margin_low:
                w = float(w * 0.7)
            return float(base * w)

        return max(
            p,
            key=lambda x: (
                score(x),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Two-parameter selector family:
    # score = VPMI - lambda * max(0, S_q(champ) - S_q(cand))
    # format: vpmi_pd_lambda:0.75
    if policy.startswith("vpmi_pd_lambda:"):
        try:
            lam = float(policy.split(":", 1)[1])
        except Exception:
            return None
        p = [c for c in pool if prior_drop_discounted_vpmi(c, s.champ, lam) is not None]
        if len(p) == 0:
            return None
        return max(
            p,
            key=lambda x: (
                float(prior_drop_discounted_vpmi(x, s.champ, lam)),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    # Parameter-free 3-feature selector:
    # rank candidates by (VPMI, S_q, S_full), then choose smallest rank-sum.
    if policy == "feat3_rank_noparam":
        p = [
            c
            for c in pool
            if c.vpmi is not None
            and c.s_q is not None
            and c.s_full is not None
        ]
        if len(p) == 0:
            return None
        r_v = rank_desc_map(p, lambda x: x.vpmi)
        r_q = rank_desc_map(p, lambda x: x.s_q)      # higher S_q is better
        r_f = rank_desc_map(p, lambda x: x.s_full)   # higher S_full is better
        worst = int(len(p) + 1)

        def score(c: Candidate) -> Tuple[float, float, float]:
            sv = float(r_v.get(int(c.idx), worst))
            sq = float(r_q.get(int(c.idx), worst))
            sf = float(r_f.get(int(c.idx), worst))
            return (
                sv + sq + sf,
                -float(c.vpmi if c.vpmi is not None else -1e18),
                -float(c.s_full if c.s_full is not None else -1e18),
            )

        return min(p, key=score)

    # Safety-net selector:
    # choose VPMI-best only among candidates ranked within top-k by S_full.
    # format: max_vpmi_sfull_topk:3
    if policy.startswith("max_vpmi_sfull_topk:"):
        try:
            k = int(float(policy.split(":", 1)[1]))
        except Exception:
            return None
        if k <= 0:
            return None
        r_f = rank_desc_map(pool, lambda x: x.s_full)
        p = [
            c
            for c in pool
            if c.vpmi is not None
            and int(r_f.get(int(c.idx), int(10**9))) <= int(k)
        ]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.vpmi)))

    # Fix E: prior-smoothed VPMI with S_q clip lower bound.
    # format: fixE_sq_clip:-5.0
    if policy.startswith("fixE_sq_clip:"):
        try:
            sq_clip = float(policy.split(":", 1)[1])
        except Exception:
            return None
        p = [c for c in pool if smoothed_vpmi(c, sq_clip) is not None]
        if len(p) == 0:
            return None
        return max(
            p,
            key=lambda x: (
                float(smoothed_vpmi(x, sq_clip)),
                float(x.vpmi if x.vpmi is not None else -1e18),
            ),
        )

    if policy == "max_visual_pmi":
        p = [c for c in pool if c.visual_pmi is not None]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.visual_pmi)))

    if policy == "max_s_full":
        p = [c for c in pool if c.s_full is not None]
        return (None if len(p) == 0 else max(p, key=lambda x: float(x.s_full)))

    if policy == "agreement_vpmi_visual":
        c1 = select_candidate("max_vpmi", s)
        c2 = select_candidate("max_visual_pmi", s)
        if c1 is None or c2 is None:
            return None
        if int(c1.idx) == int(c2.idx):
            return c1
        return None

    if policy == "max_vpmi_top2_visual":
        p = [c for c in pool if c.vpmi is not None]
        if len(p) == 0:
            return None
        p2 = sorted(p, key=lambda x: float(x.vpmi), reverse=True)[:2]
        p2v = [c for c in p2 if c.visual_pmi is not None]
        if len(p2v) == 0:
            return p2[0]
        return max(p2v, key=lambda x: float(x.visual_pmi))

    # Fix A: full-margin cutoff then max-vpmi select.
    if policy.startswith("fixA_gamma:"):
        try:
            gamma = float(policy.split(":", 1)[1])
        except Exception:
            return None
        if s.champ.s_full is None or not math.isfinite(float(s.champ.s_full)):
            return None
        p = [
            c
            for c in pool
            if c.s_full is not None
            and math.isfinite(float(c.s_full))
            and (float(s.champ.s_full) - float(c.s_full)) < float(gamma)
            and c.vpmi is not None
        ]
        if len(p) == 0:
            return None
        return max(p, key=lambda x: float(x.vpmi))

    # Fix B: prior lower-bound (p_q floor) then max-vpmi select.
    if policy.startswith("fixB_pq_ge:"):
        try:
            pq_floor = float(policy.split(":", 1)[1])
        except Exception:
            return None
        p = [c for c in pool if c.vpmi is not None and c.p_q is not None and float(c.p_q) >= float(pq_floor)]
        if len(p) == 0:
            return None
        return max(p, key=lambda x: float(x.vpmi))

    # Fix C: hybrid rank-sum (vpmi rank + s_full rank), smaller is better.
    if policy == "fixC_ranksum_vpmi_sfull":
        p = [c for c in pool if c.vpmi is not None and c.s_full is not None]
        if len(p) == 0:
            return None
        r_v = rank_desc_map(p, lambda x: x.vpmi)
        r_f = rank_desc_map(p, lambda x: x.s_full)
        worst = int(len(p) + 1)

        def score(c: Candidate) -> Tuple[float, float]:
            rv = float(r_v.get(int(c.idx), worst))
            rf = float(r_f.get(int(c.idx), worst))
            return (rv + rf, -float(c.vpmi if c.vpmi is not None else -1e18))

        return min(p, key=score)

    # Combination: Fix A + Fix B.
    # format: fixAB_gamma:<g>_pq:<p>
    if policy.startswith("fixAB_gamma:"):
        m = re.match(r"^fixAB_gamma:([0-9eE\+\-\.]+)_pq:([0-9eE\+\-\.]+)$", str(policy))
        if m is None:
            return None
        try:
            gamma = float(m.group(1))
            pq_floor = float(m.group(2))
        except Exception:
            return None
        if s.champ.s_full is None or not math.isfinite(float(s.champ.s_full)):
            return None
        p = [
            c
            for c in pool
            if c.s_full is not None
            and c.p_q is not None
            and c.vpmi is not None
            and (float(s.champ.s_full) - float(c.s_full)) < float(gamma)
            and float(c.p_q) >= float(pq_floor)
        ]
        if len(p) == 0:
            return None
        return max(p, key=lambda x: float(x.vpmi))

    if policy.startswith("rankmix:"):
        # format rankmix:wv,wvis,wsq,wgap
        body = policy.split(":", 1)[1]
        toks = [t.strip() for t in body.split(",")]
        if len(toks) != 4:
            return None
        try:
            wv, wvis, wsq, wgap = [float(t) for t in toks]
        except Exception:
            return None
        if abs(wv) + abs(wvis) + abs(wsq) + abs(wgap) <= 0:
            return None

        r_vpmi = rank_map(pool, "vpmi", s.champ)
        r_visual = rank_map(pool, "visual_pmi", s.champ)
        r_s_q = rank_map(pool, "minus_s_q", s.champ)
        r_gap = rank_map(pool, "minus_m_full", s.champ)

        def score(c: Candidate) -> float:
            return (
                float(wv) * float(r_vpmi.get(c.idx, 0.0))
                + float(wvis) * float(r_visual.get(c.idx, 0.0))
                + float(wsq) * float(r_s_q.get(c.idx, 0.0))
                + float(wgap) * float(r_gap.get(c.idx, 0.0))
            )

        return max(pool, key=lambda c: (score(c), float(c.vpmi if c.vpmi is not None else -1e18)))

    return None


def switch_cond(trigger: str, s: Sample, safe: Candidate) -> bool:
    champ = s.champ
    if champ.vpmi is None or safe.vpmi is None:
        return False
    p3 = bool(float(safe.vpmi) > float(champ.vpmi) and float(champ.vpmi) < 0.0)
    if trigger == "P3":
        return p3
    # P3 with explicit VPMI-improvement margin:
    # safe.vpmi >= champ.vpmi + dv, and champ.vpmi < 0.0
    # format: P3V_dv:0.05
    m = re.match(r"^P3V_dv:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        dv = float(m.group(1))
        return bool(
            float(safe.vpmi) >= float(champ.vpmi) + float(dv)
            and float(champ.vpmi) < 0.0
        )
    if trigger == "P5":
        return bool(p3 and float(safe.vpmi) > 0.0)

    # Parameter-free majority trigger:
    # switch when safe beats champ in at least 2 of 3 features:
    # (VPMI, S_q, S_full).
    def majority_2of3(require_vpmi_win: bool) -> bool:
        wins = 0
        avail = 0

        # feature 1: VPMI
        if champ.vpmi is not None and safe.vpmi is not None:
            avail += 1
            if float(safe.vpmi) > float(champ.vpmi):
                wins += 1
            elif require_vpmi_win:
                return False
        elif require_vpmi_win:
            return False

        # feature 2: S_q (higher is better; less negative)
        if champ.s_q is not None and safe.s_q is not None:
            avail += 1
            if float(safe.s_q) > float(champ.s_q):
                wins += 1

        # feature 3: S_full (higher is better)
        if champ.s_full is not None and safe.s_full is not None:
            avail += 1
            if float(safe.s_full) > float(champ.s_full):
                wins += 1

        return bool(avail >= 2 and wins >= 2)

    if trigger == "M3":
        return majority_2of3(require_vpmi_win=False)

    if trigger == "P3M3":
        return majority_2of3(require_vpmi_win=True)

    def margin_ok(tau: float) -> bool:
        if champ.s_full is None or safe.s_full is None:
            return False
        return bool(float(champ.s_full) - float(safe.s_full) <= float(tau))

    m = re.match(r"^P3M_tau:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        tau = float(m.group(1))
        return bool(p3 and margin_ok(tau))

    m = re.match(r"^P5M_tau:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        tau = float(m.group(1))
        return bool(p3 and float(safe.vpmi) > 0.0 and margin_ok(tau))

    m = re.match(r"^P3D_tau:([0-9eE\+\-\.]+)_alpha:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        base_tau = float(m.group(1))
        alpha = float(m.group(2))
        ig = 0.0 if s.champ_illusion_gap is None else max(0.0, float(s.champ_illusion_gap))
        tau_eff = float(base_tau + alpha * ig)
        return bool(p3 and margin_ok(tau_eff))

    # Relaxed P3: replace champ_vpmi<0.0 with champ_vpmi<cvlt.
    # format: P3C_cvlt:1.0
    m = re.match(r"^P3C_cvlt:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        cvlt = float(m.group(1))
        p3c = bool(float(safe.vpmi) > float(champ.vpmi) and float(champ.vpmi) < float(cvlt))
        return p3c

    # Core-visual win trigger:
    # switch if safe beats champ on S_core_img, while champ is still "fragile" by VPMI.
    # format: P3S_core_cvlt:0.0
    m = re.match(r"^P3S_core_cvlt:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        cvlt = float(m.group(1))
        if champ.s_core is None or safe.s_core is None:
            return False
        return bool(float(safe.s_core) > float(champ.s_core) and float(champ.vpmi) < float(cvlt))

    # Dynamic P3 threshold from in-sample VPMI percentile.
    # format: P3Q_pctl:80
    m = re.match(r"^P3Q_pctl:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        pctl = float(m.group(1))
        q = float(pctl / 100.0)
        vals = [float(champ.vpmi)] + [
            float(c.vpmi) for c in s.pool
            if c.vpmi is not None and math.isfinite(float(c.vpmi))
        ]
        tau = quantile_linear(vals, q)
        if tau is None:
            return False
        return bool(float(safe.vpmi) > float(champ.vpmi) and float(champ.vpmi) < float(tau))

    # Dynamic threshold by arena flatness:
    # safe.vpmi > champ.vpmi AND champ.vpmi < alpha * arena_flatness
    # where arena_flatness = champ_s_full - worst_candidate_s_full
    # format: P3AF_alpha:1.0
    m = re.match(r"^P3AF_alpha:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        alpha = float(m.group(1))
        if champ.s_full is None or not math.isfinite(float(champ.s_full)):
            return False
        sfull_vals = [float(champ.s_full)] + [
            float(c.s_full) for c in s.pool
            if c.s_full is not None and math.isfinite(float(c.s_full))
        ]
        if len(sfull_vals) == 0:
            return False
        flatness = float(float(champ.s_full) - float(min(sfull_vals)))
        return bool(float(safe.vpmi) > float(champ.vpmi) and float(champ.vpmi) < float(alpha) * flatness)

    # Relaxed P3 + illusion bypass:
    # switch if (champ_vpmi<cvlt OR champ_illusion_gap>=ig_thr), while safe must still beat champ on VPMI.
    # format: P3CI_cvlt:1.0_ig:1.0
    m = re.match(r"^P3CI_cvlt:([0-9eE\+\-\.]+)_ig:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        cvlt = float(m.group(1))
        ig_thr = float(m.group(2))
        ig = 0.0 if s.champ_illusion_gap is None else float(s.champ_illusion_gap)
        cond = bool(float(champ.vpmi) < float(cvlt) or float(ig) >= float(ig_thr))
        return bool(float(safe.vpmi) > float(champ.vpmi) and cond)

    # Relaxed P5 variants (requires safe_vpmi > 0.0).
    m = re.match(r"^P5C_cvlt:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        cvlt = float(m.group(1))
        p5c = bool(
            float(safe.vpmi) > float(champ.vpmi)
            and float(safe.vpmi) > 0.0
            and float(champ.vpmi) < float(cvlt)
        )
        return p5c

    m = re.match(r"^P5CI_cvlt:([0-9eE\+\-\.]+)_ig:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        cvlt = float(m.group(1))
        ig_thr = float(m.group(2))
        ig = 0.0 if s.champ_illusion_gap is None else float(s.champ_illusion_gap)
        cond = bool(float(champ.vpmi) < float(cvlt) or float(ig) >= float(ig_thr))
        return bool(
            float(safe.vpmi) > float(champ.vpmi)
            and float(safe.vpmi) > 0.0
            and cond
        )

    m = re.match(r"^P5D_tau:([0-9eE\+\-\.]+)_alpha:([0-9eE\+\-\.]+)$", str(trigger))
    if m is not None:
        base_tau = float(m.group(1))
        alpha = float(m.group(2))
        ig = 0.0 if s.champ_illusion_gap is None else max(0.0, float(s.champ_illusion_gap))
        tau_eff = float(base_tau + alpha * ig)
        return bool(p3 and float(safe.vpmi) > 0.0 and margin_ok(tau_eff))

    raise ValueError(f"Unknown trigger: {trigger}")


def evaluate(samples: Sequence[Sample], trigger: str, policy: str) -> Dict[str, Any]:
    n = int(len(samples))
    if n == 0:
        return {
            "trigger": trigger,
            "policy": policy,
            "n": 0,
            "base_acc": None,
            "final_acc": None,
            "delta_acc": None,
            "gain": 0,
            "harm": 0,
            "same": 0,
            "switch_rate": None,
            "precision_gain": None,
        }

    base_correct = 0
    final_correct = 0
    gain = 0
    harm = 0
    same = 0
    n_switch = 0

    for s in samples:
        pred = bool(s.base_ok)
        if s.base_ok:
            base_correct += 1

        safe = select_candidate(policy, s)
        if safe is not None and switch_cond(trigger, s, safe):
            n_switch += 1
            pred = bool(s.safe_ok_by_idx.get(int(safe.idx), False))

        if pred:
            final_correct += 1
        if pred and (not s.base_ok):
            gain += 1
        elif (not pred) and s.base_ok:
            harm += 1
        else:
            same += 1

    base_acc = float(base_correct / n)
    final_acc = float(final_correct / n)
    return {
        "trigger": trigger,
        "policy": policy,
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": float(final_acc - base_acc),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "switch_rate": float(n_switch / n),
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
    }


def pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # maximize gain, minimize harm
    out: List[Dict[str, Any]] = []
    for r in rows:
        dominated = False
        g = int(r["gain"])
        h = int(r["harm"])
        for q in rows:
            if q is r:
                continue
            gq = int(q["gain"])
            hq = int(q["harm"])
            if (gq >= g and hq <= h) and (gq > g or hq < h):
                dominated = True
                break
        if not dominated:
            out.append(dict(r))
    out.sort(key=lambda x: (int(x["harm"]), -int(x["gain"])))
    return out


def load_samples(in_dir: str, eval_mode: str) -> List[Sample]:
    per_sample = list(csv.DictReader(open(os.path.join(in_dir, "per_sample.csv"), encoding="utf-8")))
    per_cand = list(csv.DictReader(open(os.path.join(in_dir, "per_candidate.csv"), encoding="utf-8")))

    cand_by_sid: Dict[str, List[Candidate]] = {}
    for r in per_cand:
        sid = str(r.get("id", ""))
        idx_f = safe_float(r.get("cand_idx"))
        if idx_f is None:
            continue
        s_full = safe_float(r.get("s_full"))
        s_q = safe_float(r.get("s_ans_q"))
        s_q_min = safe_float(r.get("s_ans_q_min"))
        p_q = safe_float(r.get("p_q"))
        s_core = safe_float(r.get("s_core_img"))
        s_core_min = safe_float(r.get("s_core_img_min"))
        vpmi = (None if s_core is None or s_q is None else float(s_core - s_q))
        vpmi_core_min = safe_float(r.get("vpmi_core_min"))
        vpmi_core_min_raw = safe_float(r.get("vpmi_core_min_raw"))
        vpmi_core_min_prior_masked = safe_float(r.get("vpmi_core_min_prior_masked"))
        vpmi_word_min = safe_float(r.get("vpmi_word_min"))
        vpmi_core_tail_min = safe_float(r.get("vpmi_core_tail_min"))
        vpmi_core_min_pos_norm = safe_float(r.get("vpmi_core_min_pos_norm"))
        vpmi_core_min_mean_gap = safe_float(r.get("vpmi_core_min_mean_gap"))
        vpmi_core_sign_flip_count = safe_float(r.get("vpmi_core_sign_flip_count"))
        vpmi_core_mean = safe_float(r.get("vpmi_core_mean"))
        margin_core_img_min = safe_float(r.get("margin_core_img_min"))
        margin_core_img_mean = safe_float(r.get("margin_core_img_mean"))
        visual_pmi = (None if s_full is None or s_q is None else float(s_full - s_q))
        core_len_f = safe_float(r.get("core_len"))
        core_len = (None if core_len_f is None else int(core_len_f))
        c = Candidate(
            idx=int(idx_f),
            text=str(r.get("text", "")),
            short=str(r.get("short_answer", "")),
            core_len=core_len,
            s_full=s_full,
            s_q=s_q,
            s_q_min=s_q_min,
            p_q=p_q,
            s_core=s_core,
            s_core_min=s_core_min,
            vpmi=vpmi,
            vpmi_core_min=vpmi_core_min,
            vpmi_core_min_raw=vpmi_core_min_raw,
            vpmi_core_min_prior_masked=vpmi_core_min_prior_masked,
            vpmi_word_min=vpmi_word_min,
            vpmi_core_tail_min=vpmi_core_tail_min,
            vpmi_core_min_pos_norm=vpmi_core_min_pos_norm,
            vpmi_core_min_mean_gap=vpmi_core_min_mean_gap,
            vpmi_core_sign_flip_count=vpmi_core_sign_flip_count,
            vpmi_core_mean=vpmi_core_mean,
            margin_core_img_min=margin_core_img_min,
            margin_core_img_mean=margin_core_img_mean,
            visual_pmi=visual_pmi,
            is_champion=as_bool(r.get("is_champion", "")),
            is_safe_existing=as_bool(r.get("is_safe", "")),
        )
        cand_by_sid.setdefault(sid, []).append(c)

    out: List[Sample] = []
    for r in per_sample:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", ""))
        q = str(r.get("question", ""))
        a = str(r.get("answer", ""))
        cands = cand_by_sid.get(sid, [])
        if len(cands) == 0:
            continue

        champ = next((c for c in cands if bool(c.is_champion)), None)
        if champ is None:
            pool0 = [c for c in cands if c.s_full is not None]
            if len(pool0) == 0:
                continue
            champ = max(pool0, key=lambda x: float(x.s_full))

        mode = str(eval_mode)
        if mode == "auto":
            mm = str(r.get("eval_match_mode", "")).strip().lower()
            mode = (mm if mm in {"strict", "heuristic"} else "heuristic")
        if mode not in {"strict", "heuristic"}:
            mode = "heuristic"

        if mode == "strict" and str(r.get("is_success_strict", "")).strip() != "":
            base_ok = as_bool(r.get("is_success_strict", ""))
        elif mode == "heuristic" and str(r.get("is_success_heuristic", "")).strip() != "":
            base_ok = as_bool(r.get("is_success_heuristic", ""))
        elif str(r.get("is_success", "")).strip() != "":
            base_ok = as_bool(r.get("is_success", ""))
        else:
            base_ok = score_candidate(
                Sample(
                    sid=sid,
                    question=q,
                    answer=a,
                    eval_mode=mode,
                    base_ok=False,
                    champ=champ,
                    pool=[],
                    safe_ok_by_idx={},
                    champ_illusion_gap=safe_float(r.get("illusion_gap_format_minus_core")),
                ),
                champ,
            )

        pool = [c for c in cands if int(c.idx) != int(champ.idx)]
        safe_ok_by_idx: Dict[int, bool] = {}
        s_tmp = Sample(
            sid=sid,
            question=q,
            answer=a,
            eval_mode=mode,
            base_ok=bool(base_ok),
            champ=champ,
            pool=pool,
            safe_ok_by_idx={},
            champ_illusion_gap=safe_float(r.get("illusion_gap_format_minus_core")),
        )
        for c in pool:
            safe_ok_by_idx[int(c.idx)] = bool(score_candidate(s_tmp, c))

        out.append(
            Sample(
                sid=sid,
                question=q,
                answer=a,
                eval_mode=mode,
                base_ok=bool(base_ok),
                champ=champ,
                pool=pool,
                safe_ok_by_idx=safe_ok_by_idx,
                champ_illusion_gap=safe_float(r.get("illusion_gap_format_minus_core")),
            )
        )

    return out


def build_policies() -> List[str]:
    out = [
        "existing_safe",
        "max_vpmi",
        "max_vpmi_core_min",
        "max_vpmi_core_min_raw",
        "max_vpmi_core_min_prior_masked",
        "max_vpmi_core_min_prior_masked_tb_vpmi",
        "max_vpmi_word_min",
        "max_vpmi_core_tail_min",
        "max_vpmi_core_tail_min_tb_vpmi",
        "max_vpmi_core_tail_stable",
        "agree_tailmin_wordmin",
        "agree_vminpm_wmin",
        "agree_vminpm_wmin_dfull_le:-0.05",
        "softdiv_vminpm_prod",
        "softdiv_vminpm_sub",
        "softdiv_xv_proxy",
        "softdiv_xv_true_margin",
        "softdiv_sigmoid_vminpm",
        "softdiv_sigmoid_xv_proxy",
        "agree_vminpm_wmin_ntpmin_ge:-4.0",
        "agree_vminpm_wmin_ntpdrop_le:1.5",
        "agree_vminpm_wmin_vg_le:1.0",
        "vpmi_pd_lambda:0.25",
        "vpmi_pd_lambda:0.50",
        "vpmi_pd_lambda:0.75",
        "vpmi_pd_lambda:1.00",
        "vpmi_pd_lambda:1.25",
        "feat3_rank_noparam",
        "max_vpmi_sfull_topk:2",
        "max_vpmi_sfull_topk:3",
        "max_vpmi_sfull_topk:4",
        "max_visual_pmi",
        "max_s_full",
        "agreement_vpmi_visual",
        "max_vpmi_top2_visual",
        "fixC_ranksum_vpmi_sfull",
    ]
    for sq in [-8.0, -7.0, -6.0, -5.0, -4.0]:
        out.append(f"fixE_sq_clip:{sq}")
    for g in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        out.append(f"fixA_gamma:{g}")
    for p in [0.0, 0.1, 0.2, 0.3]:
        out.append(f"fixB_pq_ge:{p}")
    for g in [0.5, 1.0, 1.5]:
        for p in [0.1, 0.2]:
            out.append(f"fixAB_gamma:{g}_pq:{p}")

    # rankmix: w_vpmi, w_visual, w_minus_s_q, w_minus_m_full (each in {0,1})
    for wv in [0, 1]:
        for wvis in [0, 1]:
            for wsq in [0, 1]:
                for wgap in [0, 1]:
                    if wv + wvis + wsq + wgap == 0:
                        continue
                    out.append(f"rankmix:{wv},{wvis},{wsq},{wgap}")
    # dedupe while preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def parse_policies_arg(raw: str) -> List[str]:
    s = str(raw or "").strip()
    if s == "":
        return []

    # Prefer explicit separators when provided.
    if (";" in s) or ("\n" in s):
        return [p.strip() for p in re.split(r"[;\n]+", s) if p.strip() != ""]

    # Backward-compatible comma parsing with rankmix protection.
    parts = [p.strip() for p in s.split(",")]
    out: List[str] = []
    i = 0
    while i < len(parts):
        tok = str(parts[i]).strip()
        if tok == "":
            i += 1
            continue
        if tok.startswith("rankmix:"):
            if i + 3 >= len(parts):
                raise ValueError(
                    "Invalid --policies rankmix token; expected rankmix:wv,wvis,wsq,wgap"
                )
            out.append(",".join(parts[i:i + 4]).strip())
            i += 4
            continue
        out.append(tok)
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Selector trade-off study on top of P3/P5 triggers (offline)")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--triggers", type=str, default="P3,P5")
    ap.add_argument(
        "--policies",
        type=str,
        default="",
        help=(
            "Optional policy list. "
            "Use ';' (recommended) or newline separators. "
            "Comma separators are supported for backward compatibility, including rankmix."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_samples(args.in_dir, eval_mode=str(args.eval_mode))
    if str(args.policies).strip() == "":
        policies = build_policies()
    else:
        policies = parse_policies_arg(str(args.policies))
        if len(policies) == 0:
            raise ValueError("--policies provided but empty after parsing")

    triggers = [t.strip() for t in str(args.triggers).split(",") if str(t).strip() != ""]
    if len(triggers) == 0:
        triggers = ["P3", "P5"]

    rows: List[Dict[str, Any]] = []
    for trig in triggers:
        for pol in policies:
            rows.append(evaluate(samples, trigger=trig, policy=pol))

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            str(x["trigger"]),
            -float(x["delta_acc"] if x["delta_acc"] is not None else -1e9),
            float(x["harm"]),
        ),
    )

    by_trig: Dict[str, List[Dict[str, Any]]] = {str(t): [] for t in triggers}
    for r in rows:
        by_trig[str(r["trigger"])].append(r)

    pareto_rows: List[Dict[str, Any]] = []
    for trig in triggers:
        for rr in pareto_front(by_trig[trig]):
            x = dict(rr)
            x["front"] = f"{trig}_gain_harm_pareto"
            pareto_rows.append(x)

    # vs reference policy (max_vpmi) for quick diagnosis
    ref_rows: List[Dict[str, Any]] = []
    for trig in triggers:
        ref = next((r for r in by_trig[trig] if str(r["policy"]) == "max_vpmi"), None)
        if ref is None:
            continue
        for r in by_trig[trig]:
            ref_rows.append(
                {
                    "trigger": trig,
                    "policy": r["policy"],
                    "delta_acc": r["delta_acc"],
                    "gain": r["gain"],
                    "harm": r["harm"],
                    "switch_rate": r["switch_rate"],
                    "d_gain_vs_ref": int(r["gain"]) - int(ref["gain"]),
                    "d_harm_vs_ref": int(r["harm"]) - int(ref["harm"]),
                    "d_delta_acc_vs_ref": float(r["delta_acc"]) - float(ref["delta_acc"]),
                }
            )

    summary = {
        "inputs": {
            "in_dir": os.path.abspath(args.in_dir),
            "eval_mode": str(args.eval_mode),
            "triggers": triggers,
            "n_policies": int(len(policies)),
            "policies": policies,
        },
        "counts": {
            "n_samples": int(len(samples)),
            "n_rows": int(len(rows)),
        },
        "best_by_delta": {
            str(t): (max(by_trig[str(t)], key=lambda x: float(x["delta_acc"])) if by_trig[str(t)] else None)
            for t in triggers
        },
        "outputs": {
            "policy_table_csv": os.path.join(os.path.abspath(args.out_dir), "policy_table.csv"),
            "pareto_front_csv": os.path.join(os.path.abspath(args.out_dir), "pareto_front.csv"),
            "vs_ref_csv": os.path.join(os.path.abspath(args.out_dir), "vs_ref_max_vpmi.csv"),
            "summary_json": os.path.join(os.path.abspath(args.out_dir), "summary.json"),
        },
    }

    write_csv(os.path.join(args.out_dir, "policy_table.csv"), rows_sorted)
    write_csv(os.path.join(args.out_dir, "pareto_front.csv"), pareto_rows)
    write_csv(os.path.join(args.out_dir, "vs_ref_max_vpmi.csv"), ref_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(args.out_dir, "policy_table.csv"))
    print("[saved]", os.path.join(args.out_dir, "pareto_front.csv"))
    print("[saved]", os.path.join(args.out_dir, "vs_ref_max_vpmi.csv"))
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
