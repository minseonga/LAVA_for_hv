#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Minimal offline evaluator for the current adaptive FAD methodology.

Scope (intentionally minimal):
1) Stage-1 baseline: greedy champion prediction
2) Expand gate on greedy champion:
   - vpmi_only: champ_vpmi < tau_vpmi
   - and      : (champ_vpmi < tau_vpmi) AND (champ_s_full < tau_sfull)
   - or       : (champ_vpmi < tau_vpmi) OR  (champ_s_full < tau_sfull)
3) Stage-2 decision on expanded samples:
   - baseline prediction = expand-run champion
   - optional switch via:
     selector: agree_vminpm_wmin + dfull gate
     trigger : P3

This script intentionally omits unrelated policy families and trigger variants.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def as_bool(x: Any) -> bool:
    return str("" if x is None else x).strip().lower() in {"1", "true", "t", "yes", "y"}


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


def load_subset_ids(path: Optional[str]) -> Optional[Set[str]]:
    if path is None or str(path).strip() == "":
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Set[str] = set()
    if isinstance(obj, dict):
        for k in obj.keys():
            out.add(str(k))
        for v in obj.values():
            if isinstance(v, dict) and "id" in v:
                out.add(str(v.get("id")))
    elif isinstance(obj, list):
        for i, row in enumerate(obj):
            if isinstance(row, dict):
                out.add(str(row.get("id", i)))
            else:
                out.add(str(row))
    if len(out) == 0:
        raise RuntimeError(f"No IDs parsed from subset json: {path}")
    return out


@dataclass
class Candidate:
    idx: int
    s_full: Optional[float]
    s_core: Optional[float]
    s_q: Optional[float]
    vpmi: Optional[float]
    vpmi_core_min_prior_masked: Optional[float]
    vpmi_word_min: Optional[float]
    is_champion: bool
    ok_eval: Optional[bool]


@dataclass
class Sample:
    sid: str
    base_ok: bool
    champ: Candidate
    pool: List[Candidate]
    ok_by_idx: Dict[int, bool]


def resolve_mode(row: Dict[str, Any], eval_mode: str) -> str:
    mm = str(row.get("eval_match_mode", "")).strip().lower()
    if str(eval_mode) in {"strict", "heuristic"}:
        return str(eval_mode)
    if mm in {"strict", "heuristic"}:
        return mm
    return "heuristic"


def load_samples_minimal(in_dir: str, eval_mode: str) -> Dict[str, Sample]:
    per_sample = list(csv.DictReader(open(os.path.join(in_dir, "per_sample.csv"), encoding="utf-8")))
    per_cand = list(csv.DictReader(open(os.path.join(in_dir, "per_candidate.csv"), encoding="utf-8")))

    cand_by_sid: Dict[str, List[Candidate]] = {}
    for r in per_cand:
        sid = str(r.get("id", ""))
        idxf = safe_float(r.get("cand_idx"))
        if sid == "" or idxf is None:
            continue
        s_full = safe_float(r.get("s_full"))
        s_core = safe_float(r.get("s_core_img"))
        s_q = safe_float(r.get("s_ans_q"))
        vpmi = None if s_core is None or s_q is None else float(s_core - s_q)
        ok_eval = None
        if str(r.get("is_correct_eval", "")).strip() != "":
            ok_eval = as_bool(r.get("is_correct_eval"))
        c = Candidate(
            idx=int(idxf),
            s_full=s_full,
            s_core=s_core,
            s_q=s_q,
            vpmi=vpmi,
            vpmi_core_min_prior_masked=safe_float(r.get("vpmi_core_min_prior_masked")),
            vpmi_word_min=safe_float(r.get("vpmi_word_min")),
            is_champion=as_bool(r.get("is_champion", "")),
            ok_eval=ok_eval,
        )
        cand_by_sid.setdefault(sid, []).append(c)

    out: Dict[str, Sample] = {}
    for r in per_sample:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", ""))
        if sid == "" or sid not in cand_by_sid:
            continue
        cands = cand_by_sid[sid]

        champ = next((c for c in cands if c.is_champion), None)
        if champ is None:
            cc = [c for c in cands if c.s_full is not None]
            if len(cc) == 0:
                continue
            champ = max(cc, key=lambda x: float(x.s_full))

        mode = resolve_mode(r, eval_mode)
        if mode == "strict" and str(r.get("is_success_strict", "")).strip() != "":
            base_ok = as_bool(r.get("is_success_strict"))
        elif mode == "heuristic" and str(r.get("is_success_heuristic", "")).strip() != "":
            base_ok = as_bool(r.get("is_success_heuristic"))
        elif str(r.get("is_success", "")).strip() != "":
            base_ok = as_bool(r.get("is_success"))
        else:
            # Minimal fallback: use champion correctness from per_candidate if available.
            base_ok = bool(champ.ok_eval) if champ.ok_eval is not None else False

        pool = [c for c in cands if int(c.idx) != int(champ.idx)]
        ok_by_idx: Dict[int, bool] = {}
        for c in pool:
            if c.ok_eval is not None:
                ok_by_idx[int(c.idx)] = bool(c.ok_eval)

        out[sid] = Sample(
            sid=sid,
            base_ok=bool(base_ok),
            champ=champ,
            pool=pool,
            ok_by_idx=ok_by_idx,
        )
    return out


def select_safe_agree_vminpm_wmin_dfull(sample: Sample, dfull_le: float = -0.05) -> Optional[Candidate]:
    # top-1 by vpmi_core_min_prior_masked
    p1 = [c for c in sample.pool if c.vpmi_core_min_prior_masked is not None]
    # top-1 by vpmi_word_min
    p2 = [c for c in sample.pool if c.vpmi_word_min is not None]
    if len(p1) == 0 or len(p2) == 0:
        return None

    top1 = max(
        p1,
        key=lambda c: (
            float(c.vpmi_core_min_prior_masked),  # primary
            float(c.vpmi if c.vpmi is not None else -1e18),  # tie-break
        ),
    )
    top2 = max(
        p2,
        key=lambda c: (
            float(c.vpmi_word_min),
            float(c.vpmi if c.vpmi is not None else -1e18),
        ),
    )
    if int(top1.idx) != int(top2.idx):
        return None

    # dfull gate
    if top1.s_full is None or sample.champ.s_full is None:
        return None
    if float(top1.s_full) - float(sample.champ.s_full) <= float(dfull_le):
        return top1
    return None


def trigger_p3(champ: Candidate, safe: Candidate) -> bool:
    if champ.vpmi is None or safe.vpmi is None:
        return False
    return bool(float(safe.vpmi) > float(champ.vpmi) and float(champ.vpmi) < 0.0)


@dataclass
class Gate:
    name: str
    mode: str  # vpmi_only | and | or | never | always
    tau_vpmi: Optional[float]
    tau_sfull: Optional[float]


def parse_gates(raw: str) -> List[Gate]:
    txt = str(raw or "").strip()
    if txt == "":
        txt = "gate_or_vpmi_m3.5_or_sfull_m10|or|-3.5|-10;gate_and_vpmi_m5.5_and_sfull_m6|and|-5.5|-6"
    rows = [r.strip() for r in txt.split(";") if r.strip() != ""]
    out: List[Gate] = []
    for rr in rows:
        p = [x.strip() for x in rr.split("|")]
        if len(p) != 4:
            raise ValueError(f"Invalid gate spec: {rr}")
        name, mode, tv, tf = p
        ml = mode.lower()
        if ml not in {"vpmi_only", "and", "or", "never", "always"}:
            raise ValueError(f"Unknown gate mode: {mode}")
        tau_v = None if tv.lower() in {"", "none", "na"} else safe_float(tv)
        tau_f = None if tf.lower() in {"", "none", "na"} else safe_float(tf)
        out.append(Gate(name=name, mode=ml, tau_vpmi=tau_v, tau_sfull=tau_f))
    return out


def should_expand(gate: Gate, champ_vpmi: Optional[float], champ_sfull: Optional[float]) -> bool:
    if gate.mode == "never":
        return False
    if gate.mode == "always":
        return True

    cv = False
    cf = False
    if gate.tau_vpmi is not None and champ_vpmi is not None:
        cv = bool(float(champ_vpmi) < float(gate.tau_vpmi))
    if gate.tau_sfull is not None and champ_sfull is not None:
        cf = bool(float(champ_sfull) < float(gate.tau_sfull))

    if gate.mode == "vpmi_only":
        return cv
    if gate.mode == "and":
        return bool(cv and cf)
    if gate.mode == "or":
        return bool(cv or cf)
    return False


def eval_gate(
    gate: Gate,
    ids: Sequence[str],
    greedy: Dict[str, Sample],
    expand: Dict[str, Sample],
    dfull_le: float,
    extra_candidates_cost: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = int(len(ids))
    base_correct = 0
    final_correct = 0
    gain = 0
    harm = 0
    n_expand = 0
    n_switch = 0
    detail_rows: List[Dict[str, Any]] = []

    for sid in ids:
        gs = greedy[sid]
        es = expand[sid]

        base_ok = bool(gs.base_ok)
        pred = bool(base_ok)
        if base_ok:
            base_correct += 1

        expanded = should_expand(
            gate=gate,
            champ_vpmi=gs.champ.vpmi,
            champ_sfull=gs.champ.s_full,
        )
        switched = False
        safe_idx = None

        if expanded:
            n_expand += 1
            # On expanded samples, default prediction becomes expand-run champion.
            pred = bool(es.base_ok)
            safe = select_safe_agree_vminpm_wmin_dfull(es, dfull_le=dfull_le)
            if safe is not None and trigger_p3(es.champ, safe):
                switched = True
                n_switch += 1
                safe_idx = int(safe.idx)
                pred = bool(es.ok_by_idx.get(int(safe.idx), False))

        if pred:
            final_correct += 1
        if pred and (not base_ok):
            gain += 1
        elif (not pred) and base_ok:
            harm += 1

        detail_rows.append(
            {
                "id": sid,
                "gate": gate.name,
                "expanded": bool(expanded),
                "switched": bool(switched),
                "safe_idx": safe_idx,
                "base_ok": bool(base_ok),
                "pred_ok": bool(pred),
                "champ_vpmi_greedy": gs.champ.vpmi,
                "champ_s_full_greedy": gs.champ.s_full,
            }
        )

    same = int(n - gain - harm)
    base_acc = float(base_correct / n) if n > 0 else None
    final_acc = float(final_correct / n) if n > 0 else None
    expand_rate = float(n_expand / n) if n > 0 else None
    switch_rate = float(n_switch / n) if n > 0 else None
    avg_cost = None if expand_rate is None else float(1.0 + float(extra_candidates_cost) * expand_rate)
    speedup = None if avg_cost is None or avg_cost <= 0.0 else float(6.0 / avg_cost)

    row = {
        "gate": gate.name,
        "gate_mode": gate.mode,
        "tau_vpmi": gate.tau_vpmi,
        "tau_sfull": gate.tau_sfull,
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": (None if base_acc is None or final_acc is None else float(final_acc - base_acc)),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "expand_rate": expand_rate,
        "switch_rate": switch_rate,
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
        "avg_cost_rel": avg_cost,
        "speedup_vs_fixed6": speedup,
    }
    return row, detail_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal offline evaluator for adaptive FAD methodology.")
    ap.add_argument("--greedy_dir", type=str, required=True)
    ap.add_argument("--expand_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--subset_json", type=str, default="")
    ap.add_argument("--gates", type=str, default="")
    ap.add_argument("--dfull_le", type=float, default=-0.05, help="Selector dfull gate threshold")
    ap.add_argument("--extra_candidates_cost", type=int, default=5, help="Relative extra cost on expanded samples")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    greedy = load_samples_minimal(args.greedy_dir, eval_mode=str(args.eval_mode))
    expand = load_samples_minimal(args.expand_dir, eval_mode=str(args.eval_mode))
    ids = sorted(set(greedy.keys()) & set(expand.keys()))

    subset_ids = load_subset_ids(args.subset_json if str(args.subset_json).strip() != "" else None)
    if subset_ids is not None:
        ids = [sid for sid in ids if sid in subset_ids]
    if len(ids) == 0:
        raise RuntimeError("No common IDs to evaluate after filtering.")

    gates = parse_gates(args.gates)
    rows: List[Dict[str, Any]] = []
    detail: List[Dict[str, Any]] = []

    # Always include greedy-only row.
    all_gates = [Gate(name="greedy_only", mode="never", tau_vpmi=None, tau_sfull=None)] + gates
    for g in all_gates:
        r, d = eval_gate(
            gate=g,
            ids=ids,
            greedy=greedy,
            expand=expand,
            dfull_le=float(args.dfull_le),
            extra_candidates_cost=int(args.extra_candidates_cost),
        )
        rows.append(r)
        detail.extend(d)

    rows_sorted = sorted(rows, key=lambda x: (safe_float(x.get("delta_acc")) or -999.0), reverse=True)
    write_csv(os.path.join(args.out_dir, "adaptive_gate_table_minimal.csv"), rows_sorted)
    write_csv(os.path.join(args.out_dir, "adaptive_gate_per_sample_minimal.csv"), detail)

    summary = {
        "inputs": {
            "greedy_dir": os.path.abspath(args.greedy_dir),
            "expand_dir": os.path.abspath(args.expand_dir),
            "subset_json": (None if str(args.subset_json).strip() == "" else os.path.abspath(args.subset_json)),
            "eval_mode": str(args.eval_mode),
            "selector": f"agree_vminpm_wmin_dfull_le:{float(args.dfull_le)}",
            "trigger": "P3",
            "gates": [g.__dict__ for g in gates],
            "extra_candidates_cost": int(args.extra_candidates_cost),
        },
        "counts": {
            "n_greedy": int(len(greedy)),
            "n_expand": int(len(expand)),
            "n_eval_ids": int(len(ids)),
        },
        "outputs": {
            "table_csv": os.path.join(os.path.abspath(args.out_dir), "adaptive_gate_table_minimal.csv"),
            "per_sample_csv": os.path.join(os.path.abspath(args.out_dir), "adaptive_gate_per_sample_minimal.csv"),
            "summary_json": os.path.join(os.path.abspath(args.out_dir), "summary.json"),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(args.out_dir, "adaptive_gate_table_minimal.csv"))
    print("[saved]", os.path.join(args.out_dir, "adaptive_gate_per_sample_minimal.csv"))
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()

