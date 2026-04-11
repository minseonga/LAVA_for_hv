#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build_generative_b_c_meta_controller as base
import build_generative_pareto_teacher_controller as pareto


def parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


def load_policy(policy_dir: str) -> Dict[str, Any]:
    path = os.path.join(os.path.abspath(policy_dir), "selected_policy.json")
    return json.load(open(path, "r", encoding="utf-8"))


def load_decision_rows(policy_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(os.path.abspath(policy_dir), "decision_rows.csv")
    return base.read_csv_rows(path)


def to_row_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out[str(row.get("id", "")).strip()] = dict(row)
    return out


def gate_ok(row: Dict[str, Any], mode: str) -> bool:
    mode_norm = str(mode or "none").strip().lower()
    if mode_norm in {"", "none"}:
        return True
    n_base_hall = int(base.maybe_int(row.get("n_base_hall")) or 0)
    n_int_hall = int(base.maybe_int(row.get("n_int_hall")) or 0)
    base_chair_i = float(base.maybe_float(row.get("base_chair_i")) or 0.0)
    int_chair_i = float(base.maybe_float(row.get("int_chair_i")) or 0.0)
    base_chair_s = float(base.maybe_float(row.get("base_chair_s")) or 0.0)
    int_chair_s = float(base.maybe_float(row.get("int_chair_s")) or 0.0)
    if mode_norm == "base_hall_le_int_hall":
        return n_base_hall <= n_int_hall
    if mode_norm == "base_hall_lt_int_hall":
        return n_base_hall < n_int_hall
    if mode_norm == "base_chairi_le_int_chairi":
        return base_chair_i <= int_chair_i
    if mode_norm == "base_chairs_le_int_chairs":
        return base_chair_s <= int_chair_s
    if mode_norm == "chair_both_nonworse":
        return base_chair_i <= int_chair_i and base_chair_s <= int_chair_s
    if mode_norm == "strict_pareto":
        base_f1 = float(base.maybe_float(row.get("base_f1")) or 0.0)
        int_f1 = float(base.maybe_float(row.get("int_f1")) or 0.0)
        return (
            base_f1 >= int_f1
            and base_chair_i <= int_chair_i
            and base_chair_s <= int_chair_s
        )
    raise ValueError(f"Unsupported gate mode: {mode}")


def baseline_flag(score: float, tau: float, offset: float) -> bool:
    return float(score) >= float(tau) + float(offset)


def teacher_label(row: Dict[str, Any]) -> Optional[int]:
    if "teacher_fallback" not in row:
        return None
    value = row.get("teacher_fallback")
    sval = str(value if value is not None else "").strip()
    if sval == "" or sval.lower() in {"nan", "none", "null"}:
        return None
    maybe = base.maybe_int(value)
    if maybe is None:
        return None
    return int(maybe)


def teacher_precision_recall(rows: Sequence[Dict[str, Any]], routes: Sequence[str]) -> Tuple[float, float]:
    tp = 0
    n_sel = 0
    n_pos = 0
    for row, route in zip(rows, routes):
        y_maybe = teacher_label(row)
        if y_maybe is None:
            continue
        y = int(y_maybe)
        if y == 1:
            n_pos += 1
        if route == "baseline":
            n_sel += 1
            if y == 1:
                tp += 1
    if n_pos == 0 and not any(teacher_label(row) is not None for row in rows):
        return float("nan"), float("nan")
    precision = base.safe_div(float(tp), float(max(1, n_sel)))
    recall = base.safe_div(float(tp), float(max(1, n_pos)))
    return float(precision), float(recall)


def route_summary(rows: Sequence[Dict[str, Any]], routes: Sequence[str]) -> Dict[str, Any]:
    summary = base.aggregate_routes(rows, routes)
    summary.pop("decision_rows", None)
    precision, recall = teacher_precision_recall(rows, routes)
    labels_available = not (precision != precision or recall != recall)
    summary["teacher_labels_available"] = bool(labels_available)
    summary["teacher_precision"] = None if not labels_available else float(precision)
    summary["teacher_recall"] = None if not labels_available else float(recall)
    return summary


def aggregate_without_rows(rows: Sequence[Dict[str, Any]], routes: Sequence[str]) -> Dict[str, Any]:
    summary = base.aggregate_routes(rows, routes)
    summary.pop("decision_rows", None)
    return summary


def build_joined_rows(
    expert_a_rows: Sequence[Dict[str, Any]],
    expert_b_rows: Sequence[Dict[str, Any]],
    aux_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    a_map = to_row_map(expert_a_rows)
    b_map = to_row_map(expert_b_rows)
    aux_map = to_row_map(aux_rows or [])
    ids = sorted(set(a_map.keys()) & set(b_map.keys()), key=lambda x: int(x or "0"))
    out: List[Dict[str, Any]] = []
    for sid in ids:
        a = a_map[sid]
        b = b_map[sid]
        row = dict(a)
        for key, value in b.items():
            if key not in row:
                row[key] = value
        aux = aux_map.get(sid)
        if aux is not None:
            for key, value in aux.items():
                if key not in row:
                    row[key] = value
        def pick_score(src: Dict[str, Any]) -> float:
            for key in ["core_score", "distill_proxy_score", "pareto_teacher_score"]:
                val = base.maybe_float(src.get(key))
                if val is not None:
                    return float(val)
            return 0.0
        row["_expert_a_score"] = float(pick_score(a))
        row["_expert_b_score"] = float(pick_score(b))
        row["_expert_a_route_raw"] = str(a.get("route", "method"))
        row["_expert_b_route_raw"] = str(b.get("route", "method"))
        out.append(row)
    return out


def combine_routes(
    rows: Sequence[Dict[str, Any]],
    *,
    union_mode: str,
    expert_a_gate: str,
    expert_b_gate: str,
    expert_a_tau: float,
    expert_b_tau: float,
    expert_a_offset: float,
    expert_b_offset: float,
    expert_b_aux_feature: str,
    expert_b_aux_direction: str,
    expert_b_aux_tau: Optional[float],
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, float]]:
    final_routes: List[str] = []
    decision_rows: List[Dict[str, Any]] = []
    a_raw = 0
    b_raw = 0
    a_gated = 0
    b_gated = 0
    overlap = 0
    b_aux_pass_count = 0
    for row in rows:
        a_score = float(row["_expert_a_score"])
        b_score = float(row["_expert_b_score"])
        a_flag_raw = baseline_flag(a_score, expert_a_tau, expert_a_offset)
        b_flag_raw = baseline_flag(b_score, expert_b_tau, expert_b_offset)
        a_gate_ok = gate_ok(row, expert_a_gate)
        b_gate_ok = gate_ok(row, expert_b_gate)
        a_flag = bool(a_flag_raw and a_gate_ok)
        b_aux_ok = True
        if str(expert_b_aux_feature or "").strip():
            aux_val = base.maybe_float(row.get(expert_b_aux_feature))
            if aux_val is None or expert_b_aux_tau is None:
                b_aux_ok = False
            elif str(expert_b_aux_direction).strip().lower() == "low":
                b_aux_ok = float(aux_val) <= float(expert_b_aux_tau)
            else:
                b_aux_ok = float(aux_val) >= float(expert_b_aux_tau)
        b_flag = bool(b_flag_raw and b_gate_ok and b_aux_ok)
        a_raw += int(a_flag_raw)
        b_raw += int(b_flag_raw)
        a_gated += int(a_flag)
        b_gated += int(b_flag)
        b_aux_pass_count += int(b_aux_ok)
        overlap += int(a_flag and b_flag)

        route_source = "method"
        route = "method"
        mode_norm = str(union_mode).strip().lower()
        if mode_norm == "priority_a":
            if a_flag:
                route = "baseline"
                route_source = "expert_a"
            elif b_flag:
                route = "baseline"
                route_source = "expert_b"
        elif mode_norm == "priority_b":
            if b_flag:
                route = "baseline"
                route_source = "expert_b"
            elif a_flag:
                route = "baseline"
                route_source = "expert_a"
        elif mode_norm == "or":
            if a_flag or b_flag:
                route = "baseline"
                route_source = "both" if (a_flag and b_flag) else ("expert_a" if a_flag else "expert_b")
        else:
            raise ValueError(f"Unsupported union_mode: {union_mode}")
        final_routes.append(route)

        out = dict(row)
        out["expert_a_score"] = float(a_score)
        out["expert_b_score"] = float(b_score)
        out["expert_a_tau"] = float(expert_a_tau)
        out["expert_b_tau"] = float(expert_b_tau)
        out["expert_a_tau_offset"] = float(expert_a_offset)
        out["expert_b_tau_offset"] = float(expert_b_offset)
        out["expert_a_flag_raw"] = int(a_flag_raw)
        out["expert_b_flag_raw"] = int(b_flag_raw)
        out["expert_a_gate_ok"] = int(a_gate_ok)
        out["expert_b_gate_ok"] = int(b_gate_ok)
        out["expert_b_aux_feature"] = str(expert_b_aux_feature or "")
        out["expert_b_aux_direction"] = str(expert_b_aux_direction or "")
        out["expert_b_aux_tau"] = "" if expert_b_aux_tau is None else float(expert_b_aux_tau)
        out["expert_b_aux_value"] = row.get(expert_b_aux_feature, "") if str(expert_b_aux_feature or "").strip() else ""
        out["expert_b_aux_ok"] = int(b_aux_ok)
        out["expert_a_flag"] = int(a_flag)
        out["expert_b_flag"] = int(b_flag)
        out["route"] = route
        out["route_source"] = route_source
        decision_rows.append(out)

    counts = {
        "expert_a_baseline_rate_raw": base.safe_div(float(a_raw), float(max(1, len(rows)))),
        "expert_b_baseline_rate_raw": base.safe_div(float(b_raw), float(max(1, len(rows)))),
        "expert_a_baseline_rate_gated": base.safe_div(float(a_gated), float(max(1, len(rows)))),
        "expert_b_baseline_rate_gated": base.safe_div(float(b_gated), float(max(1, len(rows)))),
        "expert_b_aux_pass_rate": base.safe_div(float(b_aux_pass_count), float(max(1, len(rows)))),
        "union_baseline_rate": base.safe_div(float(sum(1 for r in final_routes if r == "baseline")), float(max(1, len(rows)))),
        "overlap_baseline_rate": base.safe_div(float(overlap), float(max(1, len(rows)))),
    }
    return final_routes, decision_rows, counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine two generative pairwise experts with priority/OR routing and optional safety gates.")
    ap.add_argument("--expert_a_dir", type=str, required=True)
    ap.add_argument("--expert_b_dir", type=str, required=True)
    ap.add_argument("--expert_b_aux_source_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--expert_a_name", type=str, default="expert_a")
    ap.add_argument("--expert_b_name", type=str, default="expert_b")
    ap.add_argument("--union_mode", type=str, default="priority_a", choices=["priority_a", "priority_b", "or"])
    ap.add_argument("--expert_a_gate", type=str, default="none")
    ap.add_argument("--expert_b_gate", type=str, default="none")
    ap.add_argument("--expert_a_tau_offsets", type=str, default="0")
    ap.add_argument("--expert_b_tau_offsets", type=str, default="0")
    ap.add_argument("--expert_b_aux_feature", type=str, default="")
    ap.add_argument("--expert_b_aux_direction", type=str, default="high", choices=["high", "low"])
    ap.add_argument("--expert_b_aux_quantiles", type=str, default="")
    ap.add_argument("--constraint_mode", type=str, default="both", choices=["none", "chairi", "chairs", "both"])
    ap.add_argument("--chair_eps", type=float, default=0.0)
    ap.add_argument("--selection_objective", type=str, default="f1", choices=["f1", "f1_minus_chairi", "neg_chairi", "claim_utility", "recall", "recall_minus_chairi"])
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    args = ap.parse_args()

    expert_a_dir = os.path.abspath(args.expert_a_dir)
    expert_b_dir = os.path.abspath(args.expert_b_dir)
    aux_source_dir = os.path.abspath(args.expert_b_aux_source_dir) if str(args.expert_b_aux_source_dir).strip() else ""
    expert_a_policy = load_policy(expert_a_dir)
    expert_b_policy = load_policy(expert_b_dir)
    expert_a_rows = load_decision_rows(expert_a_dir)
    expert_b_rows = load_decision_rows(expert_b_dir)
    aux_rows = load_decision_rows(aux_source_dir) if aux_source_dir else None
    rows = build_joined_rows(expert_a_rows, expert_b_rows, aux_rows=aux_rows)
    if not rows:
        raise RuntimeError("No overlapping rows across the two experts.")

    baseline_summary = aggregate_without_rows(rows, ["baseline"] * len(rows))
    intervention_summary = aggregate_without_rows(rows, ["method"] * len(rows))
    labeled_teacher_rows = [teacher_label(row) for row in rows]
    teacher_labels_available = any(label is not None for label in labeled_teacher_rows)
    if teacher_labels_available:
        teacher_routes = [
            "baseline" if int(label or 0) == 1 else "method"
            for label in labeled_teacher_rows
        ]
        teacher_summary = route_summary(rows, teacher_routes)
        teacher_summary["teacher_rate"] = base.safe_div(
            float(sum(int(label or 0) for label in labeled_teacher_rows if label is not None)),
            float(max(1, sum(1 for label in labeled_teacher_rows if label is not None))),
        )
    else:
        teacher_summary = {
            "n_eval": int(len(rows)),
            "teacher_labels_available": False,
            "teacher_precision": None,
            "teacher_recall": None,
            "teacher_rate": None,
        }

    a_offsets = parse_float_list(args.expert_a_tau_offsets)
    b_offsets = parse_float_list(args.expert_b_tau_offsets)
    if not a_offsets:
        a_offsets = [0.0]
    if not b_offsets:
        b_offsets = [0.0]

    expert_a_tau = float(base.maybe_float(expert_a_policy.get("tau")) or 0.0)
    expert_b_tau = float(base.maybe_float(expert_b_policy.get("tau")) or 0.0)
    aux_feature = str(args.expert_b_aux_feature or "").strip()
    aux_direction = str(args.expert_b_aux_direction)
    aux_quantiles = parse_float_list(args.expert_b_aux_quantiles)
    aux_taus: List[Optional[float]] = [None]
    if not aux_feature:
        policy_aux_feature = str(expert_b_policy.get("aux_feature") or "").strip()
        if policy_aux_feature:
            aux_feature = policy_aux_feature
            aux_direction = str(expert_b_policy.get("aux_direction") or aux_direction or "high")
            policy_aux_tau = base.maybe_float(expert_b_policy.get("aux_tau"))
            if policy_aux_tau is not None:
                aux_taus = [float(policy_aux_tau)]
    if aux_feature:
        aux_values = [
            float(v) for v in
            [base.maybe_float(row.get(aux_feature)) for row in rows]
            if v is not None
        ]
        if not aux_values:
            raise RuntimeError(f"Aux feature not found or empty: {aux_feature}")
        if aux_taus != [None]:
            pass
        elif aux_quantiles:
            aux_taus = [float(x) for x in base.quantiles_to_thresholds(aux_values, aux_quantiles)]
        else:
            aux_taus = [float(x) for x in base.quantiles_to_thresholds(aux_values, [0.7, 0.8, 0.9, 0.95, 0.98])]

    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_decision_rows: List[Dict[str, Any]] = []
    best_counts: Dict[str, float] = {}
    for a_offset in a_offsets:
        for b_offset in b_offsets:
            for aux_tau in aux_taus:
                routes, decision_rows, counts = combine_routes(
                    rows,
                    union_mode=str(args.union_mode),
                    expert_a_gate=str(args.expert_a_gate),
                    expert_b_gate=str(args.expert_b_gate),
                    expert_a_tau=expert_a_tau,
                    expert_b_tau=expert_b_tau,
                    expert_a_offset=float(a_offset),
                    expert_b_offset=float(b_offset),
                    expert_b_aux_feature=aux_feature,
                    expert_b_aux_direction=str(aux_direction),
                    expert_b_aux_tau=aux_tau,
                )
                summary = route_summary(rows, routes)
                if float(summary["baseline_rate"]) < float(args.min_baseline_rate):
                    continue
                if float(summary["baseline_rate"]) > float(args.max_baseline_rate):
                    continue
                if not pareto.feasible_under_constraints(
                    summary,
                    intervention_summary,
                    str(args.constraint_mode),
                    float(args.chair_eps),
                ):
                    continue
                row = {
                    "expert_a_tau_offset": float(a_offset),
                    "expert_b_tau_offset": float(b_offset),
                    "expert_b_aux_feature": aux_feature,
                    "expert_b_aux_direction": str(aux_direction),
                    "expert_b_aux_tau": aux_tau,
                    "union_mode": str(args.union_mode),
                    "constraint_mode": str(args.constraint_mode),
                    "selection_objective": str(args.selection_objective),
                    **counts,
                    **summary,
                }
                sweep_rows.append(row)
                if best is None or pareto.selection_key(row, str(args.selection_objective)) > pareto.selection_key(best, str(args.selection_objective)):
                    best = dict(row)
                    best_decision_rows = decision_rows
                    best_counts = dict(counts)

    if best is None:
        raise RuntimeError("No feasible combined router satisfied the requested constraints.")

    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    base.write_csv(os.path.join(args.out_dir, "offset_sweep.csv"), sweep_rows)
    base.write_csv(os.path.join(args.out_dir, "decision_rows.csv"), best_decision_rows)
    base.write_json(
        os.path.join(args.out_dir, "selected_policy.json"),
        {
            "policy_type": "generative_teacher_union_v2",
            "union_mode": str(args.union_mode),
            "expert_a_name": str(args.expert_a_name),
            "expert_b_name": str(args.expert_b_name),
            "expert_a_gate": str(args.expert_a_gate),
            "expert_b_gate": str(args.expert_b_gate),
            "expert_a_dir": expert_a_dir,
            "expert_b_dir": expert_b_dir,
            "expert_b_aux_source_dir": aux_source_dir,
            "expert_a_tau": float(expert_a_tau),
            "expert_b_tau": float(expert_b_tau),
            "expert_a_tau_offset": float(best["expert_a_tau_offset"]),
            "expert_b_tau_offset": float(best["expert_b_tau_offset"]),
            "expert_b_aux_feature": aux_feature,
            "expert_b_aux_direction": str(aux_direction),
            "expert_b_aux_tau": best.get("expert_b_aux_tau"),
            "expert_a_policy": expert_a_policy,
            "expert_b_policy": expert_b_policy,
        },
    )
    base.write_json(
        os.path.join(args.out_dir, "summary.json"),
        {
            "inputs": {
                "expert_a_dir": expert_a_dir,
                "expert_b_dir": expert_b_dir,
                "expert_a_name": str(args.expert_a_name),
                "expert_b_name": str(args.expert_b_name),
                "expert_b_aux_source_dir": aux_source_dir,
                "union_mode": str(args.union_mode),
                "expert_a_gate": str(args.expert_a_gate),
                "expert_b_gate": str(args.expert_b_gate),
                "constraint_mode": str(args.constraint_mode),
                "chair_eps": float(args.chair_eps),
                "selection_objective": str(args.selection_objective),
                "expert_a_tau_offsets": a_offsets,
                "expert_b_tau_offsets": b_offsets,
                "expert_b_aux_feature": aux_feature,
                "expert_b_aux_direction": str(aux_direction),
                "expert_b_aux_quantiles": aux_quantiles,
            },
            "counts": {
                "n_rows": int(len(rows)),
                "teacher_positive_rate": teacher_summary.get("teacher_rate"),
                **best_counts,
            },
            "baseline": baseline_summary,
            "intervention": intervention_summary,
            "teacher_oracle": teacher_summary,
            "union_policy": best,
            "outputs": {
                "offset_sweep_csv": os.path.abspath(os.path.join(args.out_dir, "offset_sweep.csv")),
                "decision_rows_csv": os.path.abspath(os.path.join(args.out_dir, "decision_rows.csv")),
            },
        },
    )


if __name__ == "__main__":
    main()
