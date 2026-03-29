#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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


def build_samples(in_dir: str, safe_selector: str) -> List[Dict[str, Any]]:
    per_sample = list(csv.DictReader(open(os.path.join(in_dir, "per_sample.csv"), encoding="utf-8")))
    per_cand = list(csv.DictReader(open(os.path.join(in_dir, "per_candidate.csv"), encoding="utf-8")))

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
                "p_q": safe_float(r.get("p_q")),
                "vpmi": vpmi,
                "is_champ": str(r.get("is_champion", "")).strip().lower() == "true",
            }
        )

    samples: List[Dict[str, Any]] = []
    for r in per_sample:
        sid = str(r.get("id", ""))
        cands = cand_by_sid.get(sid, [])
        if len(cands) == 0:
            continue

        champ = next((c for c in cands if bool(c.get("is_champ", False))), None)
        if champ is None:
            pool = [c for c in cands if c.get("s_full") is not None]
            if len(pool) == 0:
                continue
            champ = max(pool, key=lambda x: float(x["s_full"]))

        pool = [c for c in cands if int(c["idx"]) != int(champ["idx"])]
        safe = None
        if safe_selector == "max_visual_pmi":
            p = [c for c in pool if c.get("s_full") is not None and c.get("s_q") is not None]
            if len(p) > 0:
                safe = max(p, key=lambda x: float(x["s_full"] - x["s_q"]))
        elif safe_selector == "max_s_full":
            p = [c for c in pool if c.get("s_full") is not None]
            if len(p) > 0:
                safe = max(p, key=lambda x: float(x["s_full"]))
        else:
            p = [c for c in pool if c.get("vpmi") is not None]
            if len(p) > 0:
                safe = max(p, key=lambda x: float(x["vpmi"]))

        base_ok = bool(
            is_success_heur(
                str(r.get("question", "")),
                str(r.get("answer", "")),
                str(champ.get("text", "")),
                str(champ.get("short", "")),
            )
        )
        safe_ok = None
        if safe is not None:
            safe_ok = bool(
                is_success_heur(
                    str(r.get("question", "")),
                    str(r.get("answer", "")),
                    str(safe.get("text", "")),
                    str(safe.get("short", "")),
                )
            )

        m_full = None
        m_prior = None
        if safe is not None and champ.get("s_full") is not None and safe.get("s_full") is not None:
            m_full = float(champ["s_full"] - safe["s_full"])
        if safe is not None and champ.get("s_q") is not None and safe.get("s_q") is not None:
            m_prior = float(champ["s_q"] - safe["s_q"])

        d_core = None
        if safe is not None and champ.get("s_core") is not None and safe.get("s_core") is not None:
            d_core = float(safe["s_core"] - champ["s_core"])
        d_full = None
        if safe is not None and champ.get("s_full") is not None and safe.get("s_full") is not None:
            d_full = float(safe["s_full"] - champ["s_full"])
        d_q = None
        if safe is not None and champ.get("s_q") is not None and safe.get("s_q") is not None:
            d_q = float(safe["s_q"] - champ["s_q"])

        champ_vpmi = champ.get("vpmi")
        safe_vpmi = (None if safe is None else safe.get("vpmi"))
        prior_minus_full = (None if m_full is None or m_prior is None else float(m_prior - m_full))

        cond_p3 = bool(
            safe is not None
            and safe_ok is not None
            and champ_vpmi is not None
            and safe_vpmi is not None
            and float(champ_vpmi) < 0.0
            and float(safe_vpmi) > float(champ_vpmi)
        )

        samples.append(
            {
                "id": sid,
                "question": str(r.get("question", "")),
                "answer": str(r.get("answer", "")),
                "base_ok": bool(base_ok),
                "safe_ok": safe_ok,
                "cond_p3": bool(cond_p3),
                "champ_vpmi": champ_vpmi,
                "safe_vpmi": safe_vpmi,
                "champ_s_core": champ.get("s_core"),
                "safe_s_core": (None if safe is None else safe.get("s_core")),
                "champ_s_full": champ.get("s_full"),
                "safe_s_full": (None if safe is None else safe.get("s_full")),
                "champ_s_q": champ.get("s_q"),
                "safe_s_q": (None if safe is None else safe.get("s_q")),
                "champ_p_q": champ.get("p_q"),
                "safe_p_q": (None if safe is None else safe.get("p_q")),
                "m_full": m_full,
                "m_prior": m_prior,
                "prior_minus_full": prior_minus_full,
                "d_core": d_core,
                "d_full": d_full,
                "d_q": d_q,
            }
        )
    return samples


Atom = Tuple[str, str, float]


def atom_to_str(a: Atom) -> str:
    return f"{a[0]}{a[1]}{a[2]:.6g}"


def eval_rule(
    n_total: int,
    base_correct_n: int,
    base_wrong_idx: Set[int],
    safe_true_idx: Set[int],
    p3_idx: Set[int],
    atom_masks: Dict[Atom, Set[int]],
    atoms: Sequence[Atom],
) -> Dict[str, Any]:
    switched = set(p3_idx)
    for a in atoms:
        switched = switched & atom_masks[a]

    gain = len(switched & base_wrong_idx & safe_true_idx)
    harm = len(switched - safe_true_idx - base_wrong_idx)
    final_correct = int(base_correct_n + gain - harm)
    final_acc = float(final_correct / n_total) if n_total > 0 else None
    base_acc = float(base_correct_n / n_total) if n_total > 0 else None
    return {
        "n": int(n_total),
        "n_atoms": int(len(atoms)),
        "atoms": " & ".join(atom_to_str(a) for a in atoms) if atoms else "P3_base",
        "gain": int(gain),
        "harm": int(harm),
        "same": int(n_total - gain - harm),
        "switch_rate": (None if n_total == 0 else float(len(switched) / n_total)),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": (None if base_acc is None or final_acc is None else float(final_acc - base_acc)),
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
    }


def pareto_frontier(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        gi = int(r["gain"])
        hi = int(r["harm"])
        dominated = False
        for j, s in enumerate(rows):
            if i == j:
                continue
            gj = int(s["gain"])
            hj = int(s["harm"])
            if (gj >= gi and hj <= hi) and (gj > gi or hj < hi):
                dominated = True
                break
        if not dominated:
            out.append(dict(r))
    out.sort(key=lambda x: (int(x["harm"]), -int(x["gain"]), -(safe_float(x.get("delta_acc")) or -1e9)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine 2/3-feature P3 refinement rules (offline)")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--safe_selector",
        type=str,
        default="max_vpmi",
        choices=["max_vpmi", "max_visual_pmi", "max_s_full"],
    )
    ap.add_argument(
        "--features",
        type=str,
        default="safe_vpmi,safe_s_q,champ_p_q,safe_p_q,champ_s_core,m_full,m_prior,d_core",
    )
    ap.add_argument("--quantiles", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--max_atoms", type=int, default=3)
    ap.add_argument("--max_rules", type=int, default=500000, help="hard cap for evaluated rules")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    samples = build_samples(in_dir=in_dir, safe_selector=str(args.safe_selector))
    n = len(samples)
    idx_all = set(range(n))
    base_correct_idx = {i for i, s in enumerate(samples) if bool(s["base_ok"])}
    base_wrong_idx = idx_all - base_correct_idx
    safe_true_idx = {i for i, s in enumerate(samples) if s.get("safe_ok") is True}
    p3_idx = {i for i, s in enumerate(samples) if bool(s.get("cond_p3", False))}
    base_correct_n = len(base_correct_idx)

    feats = [x.strip() for x in str(args.features).split(",") if x.strip()]
    qs = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]

    # Build candidate atoms from P3-active region only.
    atoms_by_feat: Dict[str, List[Atom]] = {}
    for f in feats:
        vals = [safe_float(samples[i].get(f)) for i in p3_idx]
        vals = [float(v) for v in vals if v is not None]
        uniq: List[float] = []
        for q in qs:
            qq = quantile(vals, q)
            if qq is None:
                continue
            if all(abs(float(qq) - float(u)) > 1e-12 for u in uniq):
                uniq.append(float(qq))
        atoms: List[Atom] = []
        for t in uniq:
            atoms.append((f, ">=", float(t)))
            atoms.append((f, "<=", float(t)))
        atoms_by_feat[f] = atoms

    # Precompute atom masks
    atom_masks: Dict[Atom, Set[int]] = {}
    for f, atoms in atoms_by_feat.items():
        for a in atoms:
            _, op, thr = a
            hit: Set[int] = set()
            for i, s in enumerate(samples):
                v = safe_float(s.get(f))
                if v is None:
                    continue
                if op == ">=" and float(v) >= float(thr):
                    hit.add(i)
                elif op == "<=" and float(v) <= float(thr):
                    hit.add(i)
            atom_masks[a] = hit

    rows: List[Dict[str, Any]] = []
    # Base P3 row
    base_row = eval_rule(
        n_total=n,
        base_correct_n=base_correct_n,
        base_wrong_idx=base_wrong_idx,
        safe_true_idx=safe_true_idx,
        p3_idx=p3_idx,
        atom_masks=atom_masks,
        atoms=[],
    )
    base_row["rule_type"] = "P3_base"
    rows.append(base_row)

    evaluated = 0
    for k in range(1, int(max(1, args.max_atoms)) + 1):
        feat_combos = itertools.combinations(feats, k)
        for feat_combo in feat_combos:
            atom_lists = [atoms_by_feat.get(f, []) for f in feat_combo]
            if any(len(al) == 0 for al in atom_lists):
                continue
            for atoms in itertools.product(*atom_lists):
                evaluated += 1
                rr = eval_rule(
                    n_total=n,
                    base_correct_n=base_correct_n,
                    base_wrong_idx=base_wrong_idx,
                    safe_true_idx=safe_true_idx,
                    p3_idx=p3_idx,
                    atom_masks=atom_masks,
                    atoms=list(atoms),
                )
                rr["rule_type"] = f"P3_plus_{k}atoms"
                rows.append(rr)
                if evaluated >= int(args.max_rules):
                    break
            if evaluated >= int(args.max_rules):
                break
        if evaluated >= int(args.max_rules):
            break

    # Sort full table by delta acc desc, then harm asc
    rows_sorted = sorted(
        rows,
        key=lambda x: (
            -(safe_float(x.get("delta_acc")) or -1e9),
            int(x.get("harm", 10**9)),
            -int(x.get("gain", -10**9)),
        ),
    )
    frontier = pareto_frontier(rows)

    # Small top lists for quick inspection.
    top_by_delta = rows_sorted[:200]
    top_by_precision = sorted(
        rows,
        key=lambda x: (
            -(safe_float(x.get("precision_gain")) or -1e9),
            int(x.get("harm", 10**9)),
            -(safe_float(x.get("delta_acc")) or -1e9),
        ),
    )[:200]

    write_csv(os.path.join(out_dir, "all_rules.csv"), rows_sorted)
    write_csv(os.path.join(out_dir, "pareto_frontier.csv"), frontier)
    write_csv(os.path.join(out_dir, "top_by_delta.csv"), top_by_delta)
    write_csv(os.path.join(out_dir, "top_by_precision.csv"), top_by_precision)

    summary = {
        "inputs": {
            "in_dir": in_dir,
            "safe_selector": str(args.safe_selector),
            "features": feats,
            "quantiles": qs,
            "max_atoms": int(args.max_atoms),
            "max_rules": int(args.max_rules),
        },
        "counts": {
            "n_samples": int(n),
            "n_p3_active": int(len(p3_idx)),
            "n_rules_evaluated": int(len(rows)),
        },
        "p3_base": base_row,
        "best_delta_rule": (rows_sorted[0] if rows_sorted else None),
        "best_precision_rule": (
            sorted(
                rows,
                key=lambda x: (
                    -(safe_float(x.get("precision_gain")) or -1e9),
                    int(x.get("harm", 10**9)),
                    -(safe_float(x.get("delta_acc")) or -1e9),
                ),
            )[0]
            if rows
            else None
        ),
        "pareto_size": int(len(frontier)),
        "outputs": {
            "all_rules_csv": os.path.join(out_dir, "all_rules.csv"),
            "pareto_frontier_csv": os.path.join(out_dir, "pareto_frontier.csv"),
            "top_by_delta_csv": os.path.join(out_dir, "top_by_delta.csv"),
            "top_by_precision_csv": os.path.join(out_dir, "top_by_precision.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "all_rules.csv"))
    print("[saved]", os.path.join(out_dir, "pareto_frontier.csv"))
    print("[saved]", os.path.join(out_dir, "top_by_delta.csv"))
    print("[saved]", os.path.join(out_dir, "top_by_precision.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))
    print("[best_delta]", summary["best_delta_rule"])
    print("[best_precision]", summary["best_precision_rule"])
    print("[pareto_size]", len(frontier))


if __name__ == "__main__":
    main()

