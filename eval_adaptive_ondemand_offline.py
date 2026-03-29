#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from eval_selector_tradeoff import load_samples, select_candidate, switch_cond


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


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
                if "id" in row:
                    out.add(str(row.get("id")))
                else:
                    out.add(str(i))
            else:
                out.add(str(row))
    if len(out) == 0:
        raise RuntimeError(f"No IDs parsed from subset json: {path}")
    return out


@dataclass
class Gate:
    name: str
    mode: str
    tau_vpmi: Optional[float]
    tau_sfull: Optional[float]


def parse_gates(raw: str) -> List[Gate]:
    txt = str(raw or "").strip()
    if txt == "":
        txt = (
            "gate_or_vpmi_m3.5_or_sfull_m10|or|-3.5|-10;"
            "gate_and_vpmi_m5.5_and_sfull_m6|and|-5.5|-6"
        )
    rows = [r.strip() for r in txt.split(";") if r.strip() != ""]
    out: List[Gate] = []
    for rr in rows:
        parts = [p.strip() for p in rr.split("|")]
        if len(parts) != 4:
            raise ValueError(
                "Each gate must be 'name|mode|tau_vpmi|tau_sfull'. "
                f"Invalid gate: {rr}"
            )
        name, mode, tv, tf = parts
        mode_l = mode.lower()
        if mode_l not in {"vpmi_only", "and", "or", "never", "always"}:
            raise ValueError(f"Unknown gate mode: {mode}")
        tau_v = None if tv.lower() in {"none", "na", ""} else safe_float(tv)
        tau_f = None if tf.lower() in {"none", "na", ""} else safe_float(tf)
        out.append(Gate(name=name, mode=mode_l, tau_vpmi=tau_v, tau_sfull=tau_f))
    return out


def should_expand(gate: Gate, champ_vpmi: Optional[float], champ_sfull: Optional[float]) -> bool:
    if gate.mode == "never":
        return False
    if gate.mode == "always":
        return True

    c_v = False
    c_f = False
    if gate.tau_vpmi is not None and champ_vpmi is not None:
        c_v = bool(float(champ_vpmi) < float(gate.tau_vpmi))
    if gate.tau_sfull is not None and champ_sfull is not None:
        c_f = bool(float(champ_sfull) < float(gate.tau_sfull))

    if gate.mode == "vpmi_only":
        return c_v
    if gate.mode == "and":
        return bool(c_v and c_f)
    if gate.mode == "or":
        return bool(c_v or c_f)
    return False


def evaluate_gate(
    gate: Gate,
    ids: Sequence[str],
    greedy_by_id: Dict[str, Any],
    expand_by_id: Dict[str, Any],
    policy: str,
    trigger: str,
    extra_candidates_cost: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = int(len(ids))
    base_correct = 0
    final_correct = 0
    gain = 0
    harm = 0
    same = 0
    n_expand = 0
    n_switch = 0
    per_row: List[Dict[str, Any]] = []

    for sid in ids:
        gs = greedy_by_id[sid]
        bs = expand_by_id[sid]

        base_ok = bool(gs.base_ok)
        pred = bool(base_ok)
        if base_ok:
            base_correct += 1

        champ_v = gs.champ.vpmi
        champ_f = gs.champ.s_full
        expanded = should_expand(gate=gate, champ_vpmi=champ_v, champ_sfull=champ_f)
        switched = False
        safe_idx = None

        if expanded:
            n_expand += 1
            safe = select_candidate(policy, bs)
            pred = bool(bs.base_ok)
            if safe is not None and switch_cond(trigger, bs, safe):
                switched = True
                n_switch += 1
                safe_idx = int(safe.idx)
                pred = bool(bs.safe_ok_by_idx.get(int(safe.idx), False))

        if pred:
            final_correct += 1
        if pred and (not base_ok):
            gain += 1
        elif (not pred) and base_ok:
            harm += 1
        else:
            same += 1

        per_row.append(
            {
                "id": sid,
                "gate": gate.name,
                "base_ok": bool(base_ok),
                "pred_ok": bool(pred),
                "expanded": bool(expanded),
                "switched": bool(switched),
                "safe_idx": safe_idx,
                "champ_vpmi_greedy": champ_v,
                "champ_s_full_greedy": champ_f,
            }
        )

    base_acc = float(base_correct / n) if n > 0 else None
    final_acc = float(final_correct / n) if n > 0 else None
    expand_rate = float(n_expand / n) if n > 0 else None
    switch_rate = float(n_switch / n) if n > 0 else None
    avg_cost_rel = (None if expand_rate is None else float(1.0 + float(extra_candidates_cost) * expand_rate))
    speedup_vs_fixed6 = (
        None
        if avg_cost_rel is None or avg_cost_rel <= 0.0
        else float(6.0 / avg_cost_rel)
    )

    row = {
        "gate": gate.name,
        "gate_mode": gate.mode,
        "tau_vpmi": gate.tau_vpmi,
        "tau_sfull": gate.tau_sfull,
        "n": n,
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": (None if base_acc is None or final_acc is None else float(final_acc - base_acc)),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "expand_rate": expand_rate,
        "switch_rate": switch_rate,
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
        "avg_cost_rel": avg_cost_rel,
        "speedup_vs_fixed6": speedup_vs_fixed6,
    }
    return row, per_row


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline adaptive on-demand evaluation (greedy -> conditional expand)")
    ap.add_argument("--greedy_dir", type=str, required=True, help="in_dir generated with greedy (beam=1, return=1)")
    ap.add_argument("--expand_dir", type=str, required=True, help="in_dir generated with expanded pool (e.g., beam=6)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--subset_json", type=str, default="", help="Optional subset id json (list/dict with id)")
    ap.add_argument("--policy", type=str, default="agree_vminpm_wmin_dfull_le:-0.05")
    ap.add_argument("--trigger", type=str, default="P3")
    ap.add_argument(
        "--gates",
        type=str,
        default="",
        help=(
            "Semicolon-separated list. Each gate: name|mode|tau_vpmi|tau_sfull. "
            "mode in {vpmi_only,and,or,never,always}. "
            "Example: g1|or|-3.5|-10;g2|and|-5.5|-6"
        ),
    )
    ap.add_argument(
        "--extra_candidates_cost",
        type=int,
        default=5,
        help="Relative extra cost when expanded (greedy+5 => 5)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    greedy_samples = load_samples(args.greedy_dir, eval_mode=str(args.eval_mode))
    expand_samples = load_samples(args.expand_dir, eval_mode=str(args.eval_mode))
    greedy_by_id = {str(s.sid): s for s in greedy_samples}
    expand_by_id = {str(s.sid): s for s in expand_samples}

    ids = sorted(set(greedy_by_id.keys()) & set(expand_by_id.keys()))
    subset_ids = load_subset_ids(args.subset_json if str(args.subset_json).strip() != "" else None)
    if subset_ids is not None:
        ids = [sid for sid in ids if sid in subset_ids]
    if len(ids) == 0:
        raise RuntimeError("No common IDs to evaluate after subset filtering.")

    gates = parse_gates(args.gates)

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []

    # Always include greedy-only row as reference.
    baseline_gate = Gate(name="greedy_only", mode="never", tau_vpmi=None, tau_sfull=None)
    for gg in [baseline_gate] + gates:
        r, d = evaluate_gate(
            gate=gg,
            ids=ids,
            greedy_by_id=greedy_by_id,
            expand_by_id=expand_by_id,
            policy=str(args.policy),
            trigger=str(args.trigger),
            extra_candidates_cost=int(args.extra_candidates_cost),
        )
        rows.append(r)
        details.extend(d)

    rows_sorted = sorted(rows, key=lambda x: (safe_float(x.get("delta_acc")) or -999.0), reverse=True)
    write_csv(os.path.join(args.out_dir, "adaptive_gate_table.csv"), rows_sorted)
    write_csv(os.path.join(args.out_dir, "adaptive_gate_per_sample.csv"), details)

    summary = {
        "inputs": {
            "greedy_dir": os.path.abspath(args.greedy_dir),
            "expand_dir": os.path.abspath(args.expand_dir),
            "subset_json": (None if str(args.subset_json).strip() == "" else os.path.abspath(args.subset_json)),
            "eval_mode": str(args.eval_mode),
            "policy": str(args.policy),
            "trigger": str(args.trigger),
            "gates": [g.__dict__ for g in gates],
            "extra_candidates_cost": int(args.extra_candidates_cost),
        },
        "counts": {
            "n_greedy_samples": int(len(greedy_samples)),
            "n_expand_samples": int(len(expand_samples)),
            "n_common_eval_ids": int(len(ids)),
        },
        "outputs": {
            "adaptive_gate_table_csv": os.path.join(os.path.abspath(args.out_dir), "adaptive_gate_table.csv"),
            "adaptive_gate_per_sample_csv": os.path.join(os.path.abspath(args.out_dir), "adaptive_gate_per_sample.csv"),
            "summary_json": os.path.join(os.path.abspath(args.out_dir), "summary.json"),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(args.out_dir, "adaptive_gate_table.csv"))
    print("[saved]", os.path.join(args.out_dir, "adaptive_gate_per_sample.csv"))
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
