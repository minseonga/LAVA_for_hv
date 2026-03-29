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


def norm(x: Any) -> str:
    s = str("" if x is None else x).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
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


def is_yesno_question(q: str) -> bool:
    qq = norm(q)
    prefixes = (
        "is ", "are ", "do ", "does ", "did ", "can ", "could ",
        "will ", "would ", "has ", "have ", "had ", "was ", "were ",
    )
    return any(qq.startswith(p) for p in prefixes)


def is_success_heur(question: str, answer: str, pred_text: str, pred_short: str = "") -> bool:
    q = norm(question)
    gt = norm(answer)
    pt = norm(pred_text)
    ps = norm(pred_short)
    if gt == "":
        return False
    if gt == ps or gt == pt:
        return True

    if is_yesno_question(question) and gt in {"yes", "no"}:
        pol = first_polarity(pt) or first_polarity(ps)
        if pol is not None:
            return pol == gt
        return contains_whole(gt, pt)

    gtg = map_gender_tokens(gt)
    ptg = map_gender_tokens(pt)
    psg = map_gender_tokens(ps)
    if gtg in {"male", "female"}:
        if contains_whole(gtg, ptg) or contains_whole(gtg, psg):
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


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


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
class Sample:
    sid: str
    question: str
    answer: str
    base_ok: bool
    safe_ok: Optional[bool]
    safe_exists: bool
    pq: Optional[float]
    d: Optional[float]
    champ_core: Optional[float]
    safe_core: Optional[float]
    s_q_full: Optional[float]
    s_q_safe: Optional[float]
    s_full_full: Optional[float]
    s_full_safe: Optional[float]


@dataclass
class Cand:
    sid: str
    idx: int
    text: str
    short: str
    s_full: Optional[float]
    s_q: Optional[float]
    p_q: Optional[float]
    s_core: Optional[float]
    is_champion: bool
    is_safe_existing: bool


def evaluate(samples: Sequence[Sample], name: str, pq_t: float, d_t: float, core_gain_min: Optional[float]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = len(samples)
    gain = harm = same = 0
    n_gate = 0
    n_switch = 0
    final_correct = 0
    switched_rows: List[Dict[str, Any]] = []

    for s in samples:
        pred = bool(s.base_ok)
        cond = bool(
            s.safe_exists
            and s.safe_ok is not None
            and s.pq is not None
            and s.d is not None
            and float(s.pq) >= float(pq_t)
            and float(s.d) <= float(d_t)
        )
        if cond and core_gain_min is not None:
            if s.champ_core is None or s.safe_core is None:
                cond = False
            else:
                cond = bool(float(s.safe_core) - float(s.champ_core) >= float(core_gain_min))

        if cond:
            n_gate += 1
            pred = bool(s.safe_ok)
            n_switch += 1

        if pred:
            final_correct += 1

        outcome = "same"
        if pred and not s.base_ok:
            gain += 1
            outcome = "gain"
        elif (not pred) and s.base_ok:
            harm += 1
            outcome = "harm"
        else:
            same += 1

        if cond:
            switched_rows.append({
                "id": s.sid,
                "question": s.question,
                "answer": s.answer,
                "pq": s.pq,
                "delta_safe": s.d,
                "core_gain": (None if s.safe_core is None or s.champ_core is None else float(s.safe_core - s.champ_core)),
                "base_ok": bool(s.base_ok),
                "safe_ok": bool(s.safe_ok) if s.safe_ok is not None else None,
                "outcome": outcome,
                "rule": name,
            })

    acc = float(final_correct / n) if n > 0 else None
    base_acc = float(sum(1 for s in samples if s.base_ok) / n) if n > 0 else None
    row = {
        "rule": name,
        "pq_threshold": float(pq_t),
        "delta_threshold": float(d_t),
        "core_gain_min": core_gain_min,
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": acc,
        "delta_acc": (None if acc is None or base_acc is None else float(acc - base_acc)),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "n_gate": int(n_gate),
        "n_switch": int(n_switch),
        "switch_rate": (None if n == 0 else float(n_switch / n)),
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
    }
    return row, switched_rows


def evaluate_param_free(samples: Sequence[Sample], name: str, core_gain_min: Optional[float]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Parameter-free rule proposed by user:
      switch if M_full < M_prior
      M_full  = S_full(c_full) - S_full(c_safe)
      M_prior = S_q(c_full) - S_q(c_safe)
    """
    n = len(samples)
    gain = harm = same = 0
    n_gate = 0
    n_switch = 0
    final_correct = 0
    switched_rows: List[Dict[str, Any]] = []

    for s in samples:
        pred = bool(s.base_ok)
        m_full = None
        if s.s_full_full is not None and s.s_full_safe is not None:
            m_full = float(s.s_full_full - s.s_full_safe)
        elif s.d is not None:
            m_full = float(s.d)

        m_prior = None
        if s.s_q_full is not None and s.s_q_safe is not None:
            m_prior = float(s.s_q_full - s.s_q_safe)

        cond = bool(
            s.safe_exists
            and s.safe_ok is not None
            and m_full is not None
            and m_prior is not None
            and float(m_full) < float(m_prior)
        )
        if cond and core_gain_min is not None:
            if s.champ_core is None or s.safe_core is None:
                cond = False
            else:
                cond = bool(float(s.safe_core) - float(s.champ_core) >= float(core_gain_min))

        if cond:
            n_gate += 1
            pred = bool(s.safe_ok)
            n_switch += 1

        if pred:
            final_correct += 1

        outcome = "same"
        if pred and not s.base_ok:
            gain += 1
            outcome = "gain"
        elif (not pred) and s.base_ok:
            harm += 1
            outcome = "harm"
        else:
            same += 1

        if cond:
            switched_rows.append({
                "id": s.sid,
                "question": s.question,
                "answer": s.answer,
                "M_full": m_full,
                "M_prior": m_prior,
                "core_gain": (None if s.safe_core is None or s.champ_core is None else float(s.safe_core - s.champ_core)),
                "base_ok": bool(s.base_ok),
                "safe_ok": bool(s.safe_ok) if s.safe_ok is not None else None,
                "outcome": outcome,
                "rule": name,
            })

    acc = float(final_correct / n) if n > 0 else None
    base_acc = float(sum(1 for s in samples if s.base_ok) / n) if n > 0 else None
    row = {
        "rule": name,
        "pq_threshold": None,
        "delta_threshold": None,
        "core_gain_min": core_gain_min,
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": acc,
        "delta_acc": (None if acc is None or base_acc is None else float(acc - base_acc)),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "n_gate": int(n_gate),
        "n_switch": int(n_switch),
        "switch_rate": (None if n == 0 else float(n_switch / n)),
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
    }
    return row, switched_rows


def pick_champion(cands: Sequence[Cand], champ_idx_hint: Optional[int]) -> Optional[Cand]:
    if len(cands) == 0:
        return None
    if champ_idx_hint is not None:
        for c in cands:
            if int(c.idx) == int(champ_idx_hint):
                return c
    for c in cands:
        if bool(c.is_champion):
            return c
    pool = [c for c in cands if c.s_full is not None]
    if len(pool) == 0:
        return None
    return max(pool, key=lambda x: float(x.s_full))


def pick_safe_candidate(cands: Sequence[Cand], champ: Cand, selector: str, safe_idx_hint: Optional[int]) -> Optional[Cand]:
    if str(selector) == "existing_safe":
        if safe_idx_hint is not None:
            for c in cands:
                if int(c.idx) == int(safe_idx_hint):
                    return c
        for c in cands:
            if bool(c.is_safe_existing):
                return c
        return None

    # selector == max_visual_pmi
    pool = [
        c for c in cands
        if int(c.idx) != int(champ.idx)
        and c.s_full is not None
        and c.s_q is not None
    ]
    if len(pool) == 0:
        return None
    return max(pool, key=lambda x: float(float(x.s_full) - float(x.s_q)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build final zero-harm switch comparison table")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument(
        "--safe_selector",
        type=str,
        default="max_visual_pmi",
        choices=["max_visual_pmi", "existing_safe"],
        help="How to pick c_safe from arena candidates.",
    )
    ap.add_argument("--pq_sweep", type=str, default="0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--d_sweep", type=str, default="0.15,0.2,0.25,0.3,0.4,0.5,0.65")
    ap.add_argument("--core_gain_sweep", type=str, default="none,0.0,0.05,0.1")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir) if str(args.out_dir).strip() else os.path.join(in_dir, "final_switch_table")
    os.makedirs(out_dir, exist_ok=True)

    per_sample = read_csv(os.path.join(in_dir, "per_sample.csv"))
    per_cand = read_csv(os.path.join(in_dir, "per_candidate.csv"))

    cand_by_sid: Dict[str, List[Cand]] = {}
    for r in per_cand:
        sid = str(r.get("id", ""))
        try:
            idx = int(r.get("cand_idx", ""))
        except Exception:
            continue
        cand = Cand(
            sid=sid,
            idx=int(idx),
            text=str(r.get("text", "")),
            short=str(r.get("short_answer", "")),
            s_full=safe_float(r.get("s_full")),
            s_q=safe_float(r.get("s_ans_q")),
            p_q=safe_float(r.get("p_q")),
            s_core=safe_float(r.get("s_core_img")),
            is_champion=(str(r.get("is_champion", "")).strip().lower() == "true"),
            is_safe_existing=(str(r.get("is_safe", "")).strip().lower() == "true"),
        )
        cand_by_sid.setdefault(sid, []).append(cand)

    samples: List[Sample] = []
    for r in per_sample:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", ""))
        q = str(r.get("question", ""))
        a = str(r.get("answer", ""))
        champ_idx = None
        safe_idx = None
        try:
            champ_idx = int(r.get("champ_idx", ""))
        except Exception:
            pass
        try:
            safe_idx = int(r.get("safe_idx", ""))
        except Exception:
            pass

        cands = cand_by_sid.get(sid, [])
        champ = pick_champion(cands, champ_idx_hint=champ_idx)
        if champ is None:
            continue
        safe_c = pick_safe_candidate(
            cands=cands,
            champ=champ,
            selector=str(args.safe_selector),
            safe_idx_hint=safe_idx,
        )

        base_ok = is_success_heur(q, a, str(champ.text), str(champ.short))
        safe_exists = bool(safe_c is not None)
        safe_ok = (None if safe_c is None else is_success_heur(q, a, str(safe_c.text), str(safe_c.short)))

        d_val = None
        if safe_c is not None and champ.s_full is not None and safe_c.s_full is not None:
            d_val = float(float(champ.s_full) - float(safe_c.s_full))

        samples.append(Sample(
            sid=sid,
            question=q,
            answer=a,
            base_ok=bool(base_ok),
            safe_ok=safe_ok,
            safe_exists=bool(safe_exists),
            pq=(champ.p_q if champ.p_q is not None else safe_float(r.get("champ_p_q"))),
            d=d_val,
            champ_core=champ.s_core,
            safe_core=(None if safe_c is None else safe_c.s_core),
            s_q_full=champ.s_q,
            s_q_safe=(None if safe_c is None else safe_c.s_q),
            s_full_full=champ.s_full,
            s_full_safe=(None if safe_c is None else safe_c.s_full),
        ))

    pq_list = [float(x.strip()) for x in str(args.pq_sweep).split(",") if x.strip()]
    d_list = [float(x.strip()) for x in str(args.d_sweep).split(",") if x.strip()]
    cg_list: List[Optional[float]] = []
    for t in str(args.core_gain_sweep).split(","):
        tt = t.strip().lower()
        if tt in {"none", "na", ""}:
            cg_list.append(None)
        else:
            cg_list.append(float(tt))

    table_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []

    # Baseline row
    base_acc = float(sum(1 for s in samples if s.base_ok) / len(samples)) if samples else None
    table_rows.append({
        "rule": "baseline_no_switch",
        "pq_threshold": None,
        "delta_threshold": None,
        "core_gain_min": None,
        "n": int(len(samples)),
        "base_acc": base_acc,
        "final_acc": base_acc,
        "delta_acc": 0.0 if base_acc is not None else None,
        "gain": 0,
        "harm": 0,
        "same": int(len(samples)),
        "n_gate": 0,
        "n_switch": 0,
        "switch_rate": 0.0 if samples else None,
        "precision_gain": None,
    })

    all_switched: List[Dict[str, Any]] = []
    for pq_t in pq_list:
        for d_t in d_list:
            for cg in cg_list:
                name = f"switch_pq{pq_t}_d{d_t}_core{('none' if cg is None else cg)}"
                row, sw = evaluate(samples, name=name, pq_t=pq_t, d_t=d_t, core_gain_min=cg)
                sweep_rows.append(row)
                all_switched.extend(sw)

    # canonical rows
    def pick(pq_t: float, d_t: float, cg: Optional[float]) -> Optional[Dict[str, Any]]:
        for r in sweep_rows:
            if float(r["pq_threshold"]) == float(pq_t) and float(r["delta_threshold"]) == float(d_t):
                if (r["core_gain_min"] is None and cg is None) or (r["core_gain_min"] is not None and cg is not None and float(r["core_gain_min"]) == float(cg)):
                    return dict(r)
        return None

    for spec_name, pq_t, d_t, cg in [
        ("hotspot_rule_A", 0.9, 0.25, None),
        ("zero_harm_rule_B", 0.9, 0.25, 0.0),
    ]:
        rr = pick(pq_t, d_t, cg)
        if rr is not None:
            rr["rule"] = spec_name
            table_rows.append(rr)

    # Parameter-free inequality rules (no pq/d thresholds).
    pf_a, pf_sw_a = evaluate_param_free(samples, name="param_free_rule_P0", core_gain_min=None)
    table_rows.append(pf_a)
    pf_b, pf_sw_b = evaluate_param_free(samples, name="param_free_rule_P0_core_nonneg", core_gain_min=0.0)
    table_rows.append(pf_b)

    best_overall = None
    best_zero_harm = None
    for r in sweep_rows:
        if best_overall is None or (safe_float(r.get("delta_acc")) or -1e9) > (safe_float(best_overall.get("delta_acc")) or -1e9):
            best_overall = r
        if int(r.get("harm", 0)) == 0:
            if best_zero_harm is None or (safe_float(r.get("delta_acc")) or -1e9) > (safe_float(best_zero_harm.get("delta_acc")) or -1e9):
                best_zero_harm = r

    if best_overall is not None:
        b = dict(best_overall)
        b["rule"] = "best_overall_from_sweep"
        table_rows.append(b)
    if best_zero_harm is not None:
        b = dict(best_zero_harm)
        b["rule"] = "best_zero_harm_from_sweep"
        table_rows.append(b)

    # Save switched-case views for canonical rules
    canonical_names = {"hotspot_rule_A", "zero_harm_rule_B"}
    for cname in canonical_names:
        rule_rows = [r for r in table_rows if r.get("rule") == cname]
        if not rule_rows:
            continue
        rr = rule_rows[0]
        pq_t = float(rr["pq_threshold"])
        d_t = float(rr["delta_threshold"])
        cg = rr["core_gain_min"]
        row, sw = evaluate(samples, name=cname, pq_t=pq_t, d_t=d_t, core_gain_min=(None if cg is None else float(cg)))
        write_csv(os.path.join(out_dir, f"switched_cases_{cname}.csv"), sw)

    write_csv(os.path.join(out_dir, "switched_cases_param_free_rule_P0.csv"), pf_sw_a)
    write_csv(os.path.join(out_dir, "switched_cases_param_free_rule_P0_core_nonneg.csv"), pf_sw_b)

    write_csv(os.path.join(out_dir, "final_table.csv"), table_rows)
    write_csv(os.path.join(out_dir, "sweep_table.csv"), sweep_rows)
    write_csv(os.path.join(out_dir, "all_switched_cases.csv"), all_switched)

    summary = {
        "inputs": {
            "in_dir": in_dir,
            "safe_selector": str(args.safe_selector),
            "pq_sweep": pq_list,
            "d_sweep": d_list,
            "core_gain_sweep": cg_list,
        },
        "baseline": {
            "n": int(len(samples)),
            "acc": base_acc,
        },
        "best_overall_from_sweep": best_overall,
        "best_zero_harm_from_sweep": best_zero_harm,
        "outputs": {
            "final_table_csv": os.path.join(out_dir, "final_table.csv"),
            "sweep_table_csv": os.path.join(out_dir, "sweep_table.csv"),
            "all_switched_cases_csv": os.path.join(out_dir, "all_switched_cases.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "final_table.csv"))
    print("[saved]", os.path.join(out_dir, "sweep_table.csv"))
    print("[saved]", os.path.join(out_dir, "all_switched_cases.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
