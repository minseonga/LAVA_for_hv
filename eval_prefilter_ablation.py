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
    if not xs:
        return None
    return float(sum(1 for x in xs if x <= float(v)) / len(xs))


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
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
    w = pos - lo
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
    s_full: Optional[float]
    s_q: Optional[float]
    s_core: Optional[float]
    vpmi: Optional[float]
    visual_pmi: Optional[float]
    core_len: Optional[int]
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


@dataclass
class Variant:
    name: str
    selector: str
    dedup: bool
    stable_gate: str  # none|le2|eq1
    vpmi_pct_min: Optional[float]


def canonical_answer(c: Candidate) -> str:
    s = str(c.short).strip() if str(c.short).strip() != "" else first_clause(c.text)
    n = norm(s)
    if n in {"yes", "no"}:
        return n
    n = map_gender_tokens(n)
    toks = n.split()
    if len(toks) == 1:
        toks = [singularize_word(toks[0])]
    if len(toks) > 4:
        toks = toks[:4]
    return " ".join(toks).strip()


def dedup_pool(pool: Sequence[Candidate]) -> List[Candidate]:
    keep: Dict[str, Candidate] = {}
    for c in pool:
        key = canonical_answer(c)
        if key == "":
            key = f"idx:{int(c.idx)}"
        old = keep.get(key)
        if old is None:
            keep[key] = c
            continue
        # keep stronger likelihood candidate
        s_new = c.s_full if c.s_full is not None else -1e18
        s_old = old.s_full if old.s_full is not None else -1e18
        if float(s_new) > float(s_old):
            keep[key] = c
    return list(keep.values())


def pick_selector(selector: str, pool: Sequence[Candidate]) -> Optional[Candidate]:
    if len(pool) == 0:
        return None
    if selector == "existing_safe":
        for c in pool:
            if bool(c.is_safe_existing):
                return c
        return None
    if selector == "max_vpmi":
        p = [c for c in pool if c.vpmi is not None]
        return None if len(p) == 0 else max(p, key=lambda x: float(x.vpmi))
    if selector == "max_visual_pmi":
        p = [c for c in pool if c.visual_pmi is not None]
        return None if len(p) == 0 else max(p, key=lambda x: float(x.visual_pmi))
    if selector == "top2_vpmi_then_visual":
        p = [c for c in pool if c.vpmi is not None]
        if len(p) == 0:
            return None
        p2 = sorted(p, key=lambda x: float(x.vpmi), reverse=True)[:2]
        pv = [c for c in p2 if c.visual_pmi is not None]
        return p2[0] if len(pv) == 0 else max(pv, key=lambda x: float(x.visual_pmi))
    if selector == "agreement_vpmi_visual":
        c1 = pick_selector("max_vpmi", pool)
        c2 = pick_selector("max_visual_pmi", pool)
        if c1 is None or c2 is None:
            return None
        if int(c1.idx) == int(c2.idx):
            return c1
        return None
    return None


def allow_stable(champ: Candidate, safe: Candidate, mode: str) -> bool:
    if mode == "none":
        return True
    if champ.core_len is None or safe.core_len is None:
        return False
    if mode == "le2":
        return int(champ.core_len) <= 2 and int(safe.core_len) <= 2
    if mode == "eq1":
        return int(champ.core_len) == 1 and int(safe.core_len) == 1
    return True


def _p0_cond(champ: Candidate, safe: Candidate) -> bool:
    if champ.s_full is None or safe.s_full is None or champ.s_q is None or safe.s_q is None:
        return False
    m_full = float(champ.s_full - safe.s_full)
    m_prior = float(champ.s_q - safe.s_q)
    return bool(m_full < m_prior)


def switch_cond(trigger: str, champ: Candidate, safe: Candidate, pool: Sequence[Candidate]) -> bool:
    if champ.vpmi is None or safe.vpmi is None:
        return False
    sv = float(safe.vpmi)
    cv = float(champ.vpmi)
    p3 = bool(sv > cv and cv < 0.0)

    if trigger == "P3":
        return p3
    if trigger == "P5":
        return bool(p3 and sv > 0.0)
    if trigger == "P3_relax":
        # Missed-switch recovery: drop champ<0 requirement.
        return bool(sv > cv)
    if trigger == "P3_relax_median":
        vals = [float(c.vpmi) for c in pool if c.vpmi is not None]
        med = quantile(vals, 0.5)
        if med is None:
            return False
        return bool(sv > cv and cv < float(med))
    if trigger == "P3_or_P0":
        return bool(p3 or _p0_cond(champ, safe))
    raise ValueError(f"unknown trigger: {trigger}")


def evaluate(samples: Sequence[Sample], trigger: str, var: Variant) -> Dict[str, Any]:
    n = int(len(samples))
    base_correct = 0
    final_correct = 0
    gain = 0
    harm = 0
    same = 0
    n_gate = 0
    n_switch = 0

    for s in samples:
        pred = bool(s.base_ok)
        if s.base_ok:
            base_correct += 1

        pool = list(s.pool)
        if var.dedup:
            pool = dedup_pool(pool)

        safe = pick_selector(var.selector, pool)
        if safe is not None and allow_stable(s.champ, safe, var.stable_gate):
            # percentile gate on safe vpmi within current pool
            pct_ok = True
            if var.vpmi_pct_min is not None:
                vals = [float(c.vpmi) for c in pool if c.vpmi is not None]
                if safe.vpmi is None or len(vals) == 0:
                    pct_ok = False
                else:
                    rr = percentile_rank(vals, float(safe.vpmi))
                    pct_ok = (rr is not None and float(rr) >= float(var.vpmi_pct_min))

            if pct_ok:
                n_gate += 1
                if switch_cond(trigger, s.champ, safe, pool):
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

    base_acc = (None if n == 0 else float(base_correct / n))
    final_acc = (None if n == 0 else float(final_correct / n))
    return {
        "trigger": trigger,
        "variant": var.name,
        "selector": var.selector,
        "dedup": bool(var.dedup),
        "stable_gate": var.stable_gate,
        "vpmi_pct_min": var.vpmi_pct_min,
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": (None if base_acc is None or final_acc is None else float(final_acc - base_acc)),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "n_gate": int(n_gate),
        "n_switch": int(n_switch),
        "switch_rate": (None if n == 0 else float(n_switch / n)),
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
    }


def pareto_front(rows: List[Dict[str, Any]], trig: str) -> List[Dict[str, Any]]:
    cand = [r for r in rows if str(r.get("trigger")) == trig]
    out: List[Dict[str, Any]] = []
    for r in cand:
        g = int(r["gain"])
        h = int(r["harm"])
        dominated = False
        for q in cand:
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
        s_core = safe_float(r.get("s_core_img"))
        vpmi = (None if s_core is None or s_q is None else float(s_core - s_q))
        visual_pmi = (None if s_full is None or s_q is None else float(s_full - s_q))
        cl = None
        try:
            cl = int(float(r.get("core_len", "")))
        except Exception:
            cl = None
        c = Candidate(
            idx=int(idx_f),
            text=str(r.get("text", "")),
            short=str(r.get("short_answer", "")),
            s_full=s_full,
            s_q=s_q,
            s_core=s_core,
            vpmi=vpmi,
            visual_pmi=visual_pmi,
            core_len=cl,
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
            base_ok = bool(
                is_success_strict(a, champ.text, champ.short)
                if mode == "strict"
                else is_success_heur(q, a, champ.text, champ.short)
            )

        pool = [c for c in cands if int(c.idx) != int(champ.idx)]
        safe_ok_by_idx: Dict[int, bool] = {}
        for c in pool:
            ok = (
                is_success_strict(a, c.text, c.short)
                if mode == "strict"
                else is_success_heur(q, a, c.text, c.short)
            )
            safe_ok_by_idx[int(c.idx)] = bool(ok)

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
            )
        )
    return out


def build_variants() -> List[Variant]:
    vars_out: List[Variant] = []
    selectors = ["max_vpmi", "max_visual_pmi", "top2_vpmi_then_visual", "agreement_vpmi_visual", "existing_safe"]
    dedups = [False, True]
    stable_modes = ["none", "le2", "eq1"]
    pct_gates: List[Optional[float]] = [None, 0.7]

    for sel in selectors:
        for dd in dedups:
            for sm in stable_modes:
                for pg in pct_gates:
                    # keep space moderate
                    if sel == "existing_safe" and dd:
                        continue
                    name = f"{sel}__dedup{int(dd)}__stable_{sm}__pct_{('none' if pg is None else pg)}"
                    vars_out.append(
                        Variant(
                            name=name,
                            selector=sel,
                            dedup=dd,
                            stable_gate=sm,
                            vpmi_pct_min=pg,
                        )
                    )
    return vars_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline prefilter ablation on top of P3/P5")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict", "heuristic"])
    ap.add_argument(
        "--triggers",
        type=str,
        default="P3,P5,P3_relax,P3_relax_median,P3_or_P0",
        help="Comma-separated trigger rules to evaluate",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_samples(args.in_dir, eval_mode=str(args.eval_mode))
    variants = build_variants()
    triggers = [x.strip() for x in str(args.triggers).split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for trig in triggers:
        for var in variants:
            rows.append(evaluate(samples, trigger=trig, var=var))

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            str(x["trigger"]),
            -(safe_float(x.get("delta_acc")) if safe_float(x.get("delta_acc")) is not None else -1e9),
            int(x.get("harm", 10**9)),
        ),
    )

    pareto_rows: List[Dict[str, Any]] = []
    for trig in triggers:
        pareto_rows.extend(pareto_front(rows, trig))

    summary = {
        "inputs": {
            "in_dir": os.path.abspath(args.in_dir),
            "eval_mode": str(args.eval_mode),
            "n_variants": int(len(variants)),
            "triggers": triggers,
        },
        "counts": {
            "n_samples": int(len(samples)),
        },
        "best_by_delta": {
            **{
                str(tr): max(
                    [r for r in rows if str(r["trigger"]) == str(tr)],
                    key=lambda x: float(x["delta_acc"]),
                    default=None,
                )
                for tr in triggers
            }
        },
        "outputs": {
            "table_csv": os.path.join(os.path.abspath(args.out_dir), "prefilter_table.csv"),
            "pareto_csv": os.path.join(os.path.abspath(args.out_dir), "prefilter_pareto.csv"),
            "summary_json": os.path.join(os.path.abspath(args.out_dir), "summary.json"),
        },
    }

    write_csv(os.path.join(args.out_dir, "prefilter_table.csv"), rows_sorted)
    write_csv(os.path.join(args.out_dir, "prefilter_pareto.csv"), pareto_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(args.out_dir, "prefilter_table.csv"))
    print("[saved]", os.path.join(args.out_dir, "prefilter_pareto.csv"))
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
