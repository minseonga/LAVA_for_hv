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

    if is_yesno_question(question) and gt in {"yes", "no"}:
        pol = first_polarity(pt) or first_polarity(ps)
        if pol is not None:
            return pol == gt
        return contains_whole(gt, pt)

    gtg = map_gender_tokens(gt)
    ptg = map_gender_tokens(pt)
    psg = map_gender_tokens(ps)
    if gtg in {"male", "female"} and (contains_whole(gtg, ptg) or contains_whole(gtg, psg)):
        return True

    if len(gt.split()) >= 2 and (contains_whole(gt, pt) or contains_whole(gt, ps)):
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


def percentile_rank(vals: Sequence[float], v: float) -> Optional[float]:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    if len(xs) == 0:
        return None
    return float(sum(1 for x in xs if x <= float(v)) / len(xs))


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


def eval_split(samples: List[Dict[str, Any]], tau: float, rule: str) -> Dict[str, Any]:
    n = len(samples)
    base_correct = int(sum(1 for s in samples if bool(s["base_ok"])))
    gain = harm = same = switched = 0
    final_correct = 0
    for s in samples:
        pred = bool(s["base_ok"])
        safe = s.get("safe")
        if safe is not None:
            champ = s["champ"]
            m_full = None
            if champ.get("s_full") is not None and safe.get("s_full") is not None:
                m_full = float(champ["s_full"] - safe["s_full"])
            m_prior = None
            if champ.get("s_q") is not None and safe.get("s_q") is not None:
                m_prior = float(champ["s_q"] - safe["s_q"])

            p2 = bool(
                m_full is not None
                and m_prior is not None
                and m_full < m_prior
                and champ.get("vpmi") is not None
                and float(champ["vpmi"]) < 0.0
            )
            p3 = bool(
                champ.get("vpmi") is not None
                and safe.get("vpmi") is not None
                and float(safe["vpmi"]) > float(champ["vpmi"])
                and float(champ["vpmi"]) < 0.0
            )
            safe_pct = safe.get("vpmi_pct")
            pct_gate = bool(safe_pct is not None and float(safe_pct) >= float(tau))
            cond = (p2 and pct_gate) if rule == "P6" else (p3 and pct_gate)
            if cond:
                switched += 1
                pred = bool(s["safe_ok"])

        if pred:
            final_correct += 1
        if pred and (not bool(s["base_ok"])):
            gain += 1
        elif (not pred) and bool(s["base_ok"]):
            harm += 1
        else:
            same += 1

    base_acc = float(base_correct / n) if n > 0 else None
    final_acc = float(final_correct / n) if n > 0 else None
    return {
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": (None if base_acc is None or final_acc is None else float(final_acc - base_acc)),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "switch_rate": (None if n == 0 else float(switched / n)),
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    ds = [float(r["delta_acc"]) for r in rows if r.get("delta_acc") is not None]
    ds_sorted = sorted(ds)

    def q(p: float) -> Optional[float]:
        if not ds_sorted:
            return None
        pos = p * (len(ds_sorted) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ds_sorted[lo])
        w = pos - lo
        return float((1.0 - w) * ds_sorted[lo] + w * ds_sorted[hi])

    return {
        "n_splits": int(len(rows)),
        "delta_acc_mean": (None if not ds else float(sum(ds) / len(ds))),
        "delta_acc_median": q(0.5),
        "delta_acc_q25": q(0.25),
        "delta_acc_q75": q(0.75),
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate P6/P7 (VPMI percentile gate) offline")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--safe_selector", type=str, default="max_visual_pmi", choices=["max_visual_pmi", "max_vpmi"])
    ap.add_argument("--taus", type=str, default="0.6,0.7,0.8,0.9,0.95,1.0")
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--split_ratio", type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    per_sample = list(csv.DictReader(open(os.path.join(args.in_dir, "per_sample.csv"), encoding="utf-8")))
    per_cand = list(csv.DictReader(open(os.path.join(args.in_dir, "per_candidate.csv"), encoding="utf-8")))

    cand_by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_cand:
        sid = str(r.get("id", ""))
        idx = safe_float(r.get("cand_idx"))
        if idx is None:
            continue
        s_core = safe_float(r.get("s_core_img"))
        s_q = safe_float(r.get("s_ans_q"))
        vpmi = None if s_core is None or s_q is None else float(s_core - s_q)
        cand_by_sid[sid].append(
            {
                "idx": int(idx),
                "text": str(r.get("text", "")),
                "short": str(r.get("short_answer", "")),
                "s_full": safe_float(r.get("s_full")),
                "s_q": s_q,
                "s_core": s_core,
                "vpmi": vpmi,
                "is_champ": str(r.get("is_champion", "")).strip().lower() == "true",
            }
        )

    samples: List[Dict[str, Any]] = []
    for r in per_sample:
        sid = str(r.get("id", ""))
        cands = cand_by_sid.get(sid, [])
        if not cands:
            continue
        champ = next((c for c in cands if bool(c.get("is_champ", False))), None)
        if champ is None:
            pool = [c for c in cands if c.get("s_full") is not None]
            if not pool:
                continue
            champ = max(pool, key=lambda x: float(x["s_full"]))

        pool = [c for c in cands if int(c["idx"]) != int(champ["idx"])]
        safe = None
        if args.safe_selector == "max_visual_pmi":
            p = [c for c in pool if c.get("s_full") is not None and c.get("s_q") is not None]
            if p:
                safe = max(p, key=lambda x: float(x["s_full"] - x["s_q"]))
        else:
            p = [c for c in pool if c.get("vpmi") is not None]
            if p:
                safe = max(p, key=lambda x: float(x["vpmi"]))

        vpmi_vals = [float(c["vpmi"]) for c in cands if c.get("vpmi") is not None]
        if safe is not None and safe.get("vpmi") is not None:
            safe["vpmi_pct"] = percentile_rank(vpmi_vals, float(safe["vpmi"]))

        base_ok = bool(is_success_heur(str(r.get("question", "")), str(r.get("answer", "")), str(champ["text"]), str(champ["short"])))
        safe_ok = None
        if safe is not None:
            safe_ok = bool(is_success_heur(str(r.get("question", "")), str(r.get("answer", "")), str(safe["text"]), str(safe["short"])))

        samples.append(
            {
                "sid": sid,
                "question": str(r.get("question", "")),
                "answer": str(r.get("answer", "")),
                "base_ok": base_ok,
                "safe_ok": safe_ok,
                "champ": champ,
                "safe": safe,
            }
        )

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]
    full_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []

    for tau in taus:
        for rule in ["P6", "P7"]:
            full = eval_split(samples, tau=tau, rule=rule)
            full_rows.append({"rule": rule, "tau": tau, "split": "full", **full})

    ids = sorted({s["sid"] for s in samples})
    for seed in range(int(args.n_seeds)):
        rr = random.Random(seed)
        order = ids[:]
        rr.shuffle(order)
        cut = int(len(order) * float(args.split_ratio))
        train_ids = set(order[:cut])
        test_ids = set(order[cut:])
        for split_name, split_ids in [("train", train_ids), ("test", test_ids)]:
            ss = [s for s in samples if s["sid"] in split_ids]
            for tau in taus:
                for rule in ["P6", "P7"]:
                    met = eval_split(ss, tau=tau, rule=rule)
                    seed_rows.append({"seed": seed, "split": split_name, "rule": rule, "tau": tau, **met})

    agg_rows: List[Dict[str, Any]] = []
    for split_name in ["train", "test"]:
        for tau in taus:
            for rule in ["P6", "P7"]:
                sub = [r for r in seed_rows if r["split"] == split_name and r["rule"] == rule and float(r["tau"]) == float(tau)]
                agg_rows.append({"split": split_name, "rule": rule, "tau": tau, **summarize_rows(sub)})

    full_sorted = sorted(full_rows, key=lambda x: float(x.get("delta_acc", -1e9)), reverse=True)
    test_sorted = sorted([r for r in agg_rows if r["split"] == "test"], key=lambda x: float(x.get("delta_acc_mean", -1e9)), reverse=True)

    write_csv(os.path.join(args.out_dir, "full_table.csv"), full_sorted)
    write_csv(os.path.join(args.out_dir, "multiseed_per_seed.csv"), seed_rows)
    write_csv(os.path.join(args.out_dir, "multiseed_aggregated.csv"), agg_rows)

    summary = {
        "inputs": {
            "in_dir": os.path.abspath(args.in_dir),
            "safe_selector": str(args.safe_selector),
            "taus": taus,
            "n_seeds": int(args.n_seeds),
            "split_ratio": float(args.split_ratio),
        },
        "counts": {
            "n_samples": int(len(samples)),
            "n_ids": int(len(ids)),
        },
        "full_best": (full_sorted[0] if full_sorted else None),
        "test_best_by_mean_delta": (test_sorted[0] if test_sorted else None),
        "test_top5": test_sorted[:5],
        "outputs": {
            "full_table_csv": os.path.join(args.out_dir, "full_table.csv"),
            "multiseed_per_seed_csv": os.path.join(args.out_dir, "multiseed_per_seed.csv"),
            "multiseed_aggregated_csv": os.path.join(args.out_dir, "multiseed_aggregated.csv"),
            "summary_json": os.path.join(args.out_dir, "summary.json"),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(args.out_dir, "full_table.csv"))
    print("[saved]", os.path.join(args.out_dir, "multiseed_per_seed.csv"))
    print("[saved]", os.path.join(args.out_dir, "multiseed_aggregated.csv"))
    print("[saved]", os.path.join(args.out_dir, "summary.json"))
    if full_sorted:
        print("[full_best]", full_sorted[0])
    if test_sorted:
        print("[test_best]", test_sorted[0])


if __name__ == "__main__":
    main()

