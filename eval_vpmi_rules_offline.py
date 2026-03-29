#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import defaultdict
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


def is_yesno_question(q: str) -> bool:
    qq = norm(q)
    prefixes = (
        "is ",
        "are ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "will ",
        "would ",
        "has ",
        "have ",
        "had ",
        "was ",
        "were ",
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


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
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
    is_champion: bool
    is_safe_existing: bool


@dataclass
class Sample:
    sid: str
    question: str
    answer: str
    base_ok: bool
    champ: Candidate
    safe_map: Dict[str, Optional[Candidate]]
    safe_ok_map: Dict[str, Optional[bool]]


def pick_champion(cands: Sequence[Candidate], champ_idx_hint: Optional[int]) -> Optional[Candidate]:
    if not cands:
        return None
    if champ_idx_hint is not None:
        for c in cands:
            if int(c.idx) == int(champ_idx_hint):
                return c
    for c in cands:
        if bool(c.is_champion):
            return c
    pool = [c for c in cands if c.s_full is not None]
    if not pool:
        return None
    return max(pool, key=lambda x: float(x.s_full))


def pick_safe(cands: Sequence[Candidate], champ: Candidate, mode: str, safe_idx_hint: Optional[int]) -> Optional[Candidate]:
    if mode == "existing_safe":
        if safe_idx_hint is not None:
            for c in cands:
                if int(c.idx) == int(safe_idx_hint):
                    return c
        for c in cands:
            if bool(c.is_safe_existing):
                return c
        return None

    pool = [c for c in cands if int(c.idx) != int(champ.idx)]
    if not pool:
        return None

    if mode == "max_visual_pmi":
        p = [c for c in pool if c.s_full is not None and c.s_q is not None]
        if not p:
            return None
        return max(p, key=lambda x: float(float(x.s_full) - float(x.s_q)))

    if mode == "max_vpmi":
        p = [c for c in pool if c.vpmi is not None]
        if not p:
            return None
        return max(p, key=lambda x: float(x.vpmi))

    raise ValueError(f"Unknown safe mode: {mode}")


def rule_condition(rule: str, champ: Candidate, safe: Candidate) -> bool:
    m_full = None
    if champ.s_full is not None and safe.s_full is not None:
        m_full = float(champ.s_full - safe.s_full)
    m_prior = None
    if champ.s_q is not None and safe.s_q is not None:
        m_prior = float(champ.s_q - safe.s_q)

    cv = champ.vpmi
    sv = safe.vpmi

    p0 = bool(m_full is not None and m_prior is not None and m_full < m_prior)
    p2 = bool(p0 and cv is not None and float(cv) < 0.0)
    p3 = bool(cv is not None and sv is not None and float(sv) > float(cv) and float(cv) < 0.0)
    p4 = bool(p2 and sv is not None and float(sv) > 0.0)
    p5 = bool(p3 and sv is not None and float(sv) > 0.0)

    if rule == "P0":
        return p0
    if rule == "P2":
        return p2
    if rule == "P3":
        return p3
    if rule == "P4":
        return p4
    if rule == "P5":
        return p5
    raise ValueError(f"Unknown rule: {rule}")


def evaluate(
    samples: Sequence[Sample],
    rule: str,
    safe_mode: str,
    collect_switched: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = int(len(samples))
    if n == 0:
        row = {
            "rule": rule,
            "safe_mode": safe_mode,
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
        return row, []

    gain = 0
    harm = 0
    same = 0
    switched = 0
    base_correct = 0
    final_correct = 0
    switched_rows: List[Dict[str, Any]] = []

    for s in samples:
        base_ok = bool(s.base_ok)
        pred = bool(base_ok)
        if base_ok:
            base_correct += 1

        safe = s.safe_map.get(safe_mode)
        safe_ok = s.safe_ok_map.get(safe_mode)
        did_switch = False
        cond = False
        if safe is not None and safe_ok is not None:
            cond = rule_condition(rule, champ=s.champ, safe=safe)
            if cond:
                did_switch = True
                switched += 1
                pred = bool(safe_ok)

        if pred:
            final_correct += 1
        if pred and (not base_ok):
            gain += 1
            outcome = "gain"
        elif (not pred) and base_ok:
            harm += 1
            outcome = "harm"
        else:
            same += 1
            outcome = "same"

        if collect_switched and did_switch:
            switched_rows.append(
                {
                    "id": s.sid,
                    "rule": rule,
                    "safe_mode": safe_mode,
                    "outcome": outcome,
                    "base_ok": bool(base_ok),
                    "pred_ok_after_switch": bool(pred),
                    "question": s.question,
                    "answer": s.answer,
                    "champ_text": s.champ.text,
                    "safe_text": (None if safe is None else safe.text),
                    "champ_s_full": s.champ.s_full,
                    "safe_s_full": (None if safe is None else safe.s_full),
                    "champ_s_q": s.champ.s_q,
                    "safe_s_q": (None if safe is None else safe.s_q),
                    "champ_s_core": s.champ.s_core,
                    "safe_s_core": (None if safe is None else safe.s_core),
                    "champ_vpmi": s.champ.vpmi,
                    "safe_vpmi": (None if safe is None else safe.vpmi),
                    "switched": True,
                    "condition": bool(cond),
                }
            )

    base_acc = float(base_correct / n)
    final_acc = float(final_correct / n)
    row = {
        "rule": rule,
        "safe_mode": safe_mode,
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": float(final_acc - base_acc),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "switch_rate": float(switched / n),
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
    }
    return row, switched_rows


def bootstrap_ci_delta(
    samples: Sequence[Sample],
    rule: str,
    safe_mode: str,
    n_boot: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float]]:
    if len(samples) == 0 or int(n_boot) <= 0:
        return None, None
    rr = random.Random(int(seed))
    n = int(len(samples))
    deltas: List[float] = []
    for _ in range(int(n_boot)):
        ss = [samples[rr.randrange(n)] for _ in range(n)]
        row, _ = evaluate(ss, rule=rule, safe_mode=safe_mode, collect_switched=False)
        d = safe_float(row.get("delta_acc"))
        if d is not None:
            deltas.append(float(d))
    if not deltas:
        return None, None
    return quantile(deltas, 0.025), quantile(deltas, 0.975)


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n_splits": 0,
            "delta_acc_mean": None,
            "delta_acc_median": None,
            "delta_acc_q25": None,
            "delta_acc_q75": None,
            "delta_acc_min": None,
            "delta_acc_max": None,
            "n_pos_delta": 0,
            "n_neg_delta": 0,
            "mean_gain": None,
            "mean_harm": None,
            "mean_switch_rate": None,
            "mean_precision_gain": None,
        }

    ds = [float(r["delta_acc"]) for r in rows if r.get("delta_acc") is not None]
    return {
        "n_splits": int(len(rows)),
        "delta_acc_mean": (None if not ds else float(sum(ds) / len(ds))),
        "delta_acc_median": quantile(ds, 0.5),
        "delta_acc_q25": quantile(ds, 0.25),
        "delta_acc_q75": quantile(ds, 0.75),
        "delta_acc_min": (None if not ds else float(min(ds))),
        "delta_acc_max": (None if not ds else float(max(ds))),
        "n_pos_delta": int(sum(1 for d in ds if d > 0)),
        "n_neg_delta": int(sum(1 for d in ds if d < 0)),
        "mean_gain": float(sum(float(r["gain"]) for r in rows) / len(rows)),
        "mean_harm": float(sum(float(r["harm"]) for r in rows) / len(rows)),
        "mean_switch_rate": float(sum(float(r["switch_rate"]) for r in rows) / len(rows)),
        "mean_precision_gain": float(
            sum(float(r["precision_gain"]) for r in rows if r.get("precision_gain") is not None)
            / max(1, sum(1 for r in rows if r.get("precision_gain") is not None))
        ),
    }


def resolve_eval_mode(per_sample: List[Dict[str, Any]], arg_mode: str) -> str:
    if str(arg_mode) in {"strict", "heuristic"}:
        return str(arg_mode)
    for r in per_sample:
        m = str(r.get("eval_match_mode", "")).strip().lower()
        if m in {"strict", "heuristic"}:
            return m
    return "heuristic"


def score_candidate(eval_mode: str, question: str, answer: str, cand: Candidate) -> bool:
    if eval_mode == "strict":
        return bool(is_success_strict(answer, cand.text, cand.short))
    return bool(is_success_heur(question, answer, cand.text, cand.short))


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline evaluation for P0/P2/P3/P4/P5 switching rules")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--rules", type=str, default="P0,P2,P3,P4,P5")
    ap.add_argument("--safe_modes", type=str, default="max_visual_pmi,max_vpmi,existing_safe")
    ap.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--bootstrap_n", type=int, default=1000)
    ap.add_argument("--bootstrap_seed", type=int, default=123)
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--split_ratio", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    per_sample = list(csv.DictReader(open(os.path.join(args.in_dir, "per_sample.csv"), encoding="utf-8")))
    per_cand = list(csv.DictReader(open(os.path.join(args.in_dir, "per_candidate.csv"), encoding="utf-8")))

    eval_mode = resolve_eval_mode(per_sample, str(args.eval_mode))
    rules = [x.strip() for x in str(args.rules).split(",") if x.strip()]
    safe_modes = [x.strip() for x in str(args.safe_modes).split(",") if x.strip()]

    cand_by_sid: Dict[str, List[Candidate]] = defaultdict(list)
    for r in per_cand:
        sid = str(r.get("id", ""))
        idx_f = safe_float(r.get("cand_idx"))
        if idx_f is None:
            continue
        s_full = safe_float(r.get("s_full"))
        s_q = safe_float(r.get("s_ans_q"))
        s_core = safe_float(r.get("s_core_img"))
        vpmi = (None if s_core is None or s_q is None else float(s_core - s_q))
        cand_by_sid[sid].append(
            Candidate(
                idx=int(idx_f),
                text=str(r.get("text", "")),
                short=str(r.get("short_answer", "")),
                s_full=s_full,
                s_q=s_q,
                s_core=s_core,
                vpmi=vpmi,
                is_champion=as_bool(r.get("is_champion", "")),
                is_safe_existing=as_bool(r.get("is_safe", "")),
            )
        )

    samples: List[Sample] = []
    for r in per_sample:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", ""))
        q = str(r.get("question", ""))
        a = str(r.get("answer", ""))

        champ_idx_hint = None
        safe_idx_hint = None
        try:
            champ_idx_hint = int(r.get("champ_idx", ""))
        except Exception:
            pass
        try:
            safe_idx_hint = int(r.get("safe_idx", ""))
        except Exception:
            pass

        cands = cand_by_sid.get(sid, [])
        champ = pick_champion(cands, champ_idx_hint=champ_idx_hint)
        if champ is None:
            continue

        if eval_mode == "strict" and str(r.get("is_success_strict", "")).strip() != "":
            base_ok = as_bool(r.get("is_success_strict", ""))
        elif eval_mode == "heuristic" and str(r.get("is_success_heuristic", "")).strip() != "":
            base_ok = as_bool(r.get("is_success_heuristic", ""))
        elif str(r.get("is_success", "")).strip() != "":
            base_ok = as_bool(r.get("is_success", ""))
        else:
            base_ok = score_candidate(eval_mode, q, a, champ)

        safe_map: Dict[str, Optional[Candidate]] = {}
        safe_ok_map: Dict[str, Optional[bool]] = {}
        for mode in safe_modes:
            safe = pick_safe(cands, champ=champ, mode=mode, safe_idx_hint=safe_idx_hint)
            safe_map[mode] = safe
            safe_ok_map[mode] = (None if safe is None else score_candidate(eval_mode, q, a, safe))

        samples.append(
            Sample(
                sid=sid,
                question=q,
                answer=a,
                base_ok=bool(base_ok),
                champ=champ,
                safe_map=safe_map,
                safe_ok_map=safe_ok_map,
            )
        )

    full_rows: List[Dict[str, Any]] = []
    switched_rows: List[Dict[str, Any]] = []

    for mode in safe_modes:
        for rule in rules:
            row, sw = evaluate(samples, rule=rule, safe_mode=mode, collect_switched=True)
            ci_lo, ci_hi = bootstrap_ci_delta(
                samples=samples,
                rule=rule,
                safe_mode=mode,
                n_boot=int(args.bootstrap_n),
                seed=int(args.bootstrap_seed),
            )
            row["ci95_lo"] = ci_lo
            row["ci95_hi"] = ci_hi
            full_rows.append(row)
            switched_rows.extend(sw)

    full_sorted = sorted(
        full_rows,
        key=lambda x: (
            safe_float(x.get("delta_acc")) if safe_float(x.get("delta_acc")) is not None else -1e9,
            safe_float(x.get("precision_gain")) if safe_float(x.get("precision_gain")) is not None else -1e9,
        ),
        reverse=True,
    )

    # 70/30 multiseed split by id
    ids = sorted({s.sid for s in samples})
    seed_rows: List[Dict[str, Any]] = []
    for seed in range(int(args.n_seeds)):
        rr = random.Random(seed)
        order = ids[:]
        rr.shuffle(order)
        cut = int(len(order) * float(args.split_ratio))
        train_ids = set(order[:cut])
        test_ids = set(order[cut:])
        for split_name, split_ids in [("train", train_ids), ("test", test_ids)]:
            ss = [s for s in samples if s.sid in split_ids]
            for mode in safe_modes:
                for rule in rules:
                    met, _ = evaluate(ss, rule=rule, safe_mode=mode, collect_switched=False)
                    seed_rows.append(
                        {
                            "seed": int(seed),
                            "split": split_name,
                            "safe_mode": mode,
                            "rule": rule,
                            **met,
                        }
                    )

    agg_rows: List[Dict[str, Any]] = []
    for split_name in ["train", "test"]:
        for mode in safe_modes:
            for rule in rules:
                sub = [
                    r
                    for r in seed_rows
                    if str(r["split"]) == split_name and str(r["safe_mode"]) == mode and str(r["rule"]) == rule
                ]
                agg_rows.append(
                    {
                        "split": split_name,
                        "safe_mode": mode,
                        "rule": rule,
                        **summarize_rows(sub),
                    }
                )

    test_rank = sorted(
        [r for r in agg_rows if str(r.get("split")) == "test"],
        key=lambda x: (
            safe_float(x.get("delta_acc_mean")) if safe_float(x.get("delta_acc_mean")) is not None else -1e9
        ),
        reverse=True,
    )

    summary = {
        "inputs": {
            "in_dir": os.path.abspath(args.in_dir),
            "eval_mode": str(eval_mode),
            "rules": rules,
            "safe_modes": safe_modes,
            "bootstrap_n": int(args.bootstrap_n),
            "bootstrap_seed": int(args.bootstrap_seed),
            "split": "id_random_70_30",
            "n_seeds": int(args.n_seeds),
            "split_ratio": float(args.split_ratio),
        },
        "counts": {
            "n_samples": int(len(samples)),
            "n_ids": int(len(ids)),
        },
        "full_best": (full_sorted[0] if full_sorted else None),
        "test_ranking_by_mean_delta": test_rank,
        "outputs": {
            "full_rule_table_csv": os.path.join(os.path.abspath(args.out_dir), "full_rule_table.csv"),
            "switched_cases_csv": os.path.join(os.path.abspath(args.out_dir), "switched_cases.csv"),
            "multiseed_per_seed_csv": os.path.join(os.path.abspath(args.out_dir), "multiseed_per_seed.csv"),
            "multiseed_aggregated_csv": os.path.join(os.path.abspath(args.out_dir), "multiseed_aggregated.csv"),
            "summary_json": os.path.join(os.path.abspath(args.out_dir), "summary.json"),
        },
    }

    write_csv(os.path.join(args.out_dir, "full_rule_table.csv"), full_sorted)
    write_csv(os.path.join(args.out_dir, "switched_cases.csv"), switched_rows)
    write_csv(os.path.join(args.out_dir, "multiseed_per_seed.csv"), seed_rows)
    write_csv(os.path.join(args.out_dir, "multiseed_aggregated.csv"), agg_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(args.out_dir, "full_rule_table.csv"))
    print("[saved]", os.path.join(args.out_dir, "switched_cases.csv"))
    print("[saved]", os.path.join(args.out_dir, "multiseed_per_seed.csv"))
    print("[saved]", os.path.join(args.out_dir, "multiseed_aggregated.csv"))
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
