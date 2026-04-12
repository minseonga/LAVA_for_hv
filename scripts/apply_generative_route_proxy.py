#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import build_generative_b_c_meta_controller as base


def build_scores(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Dict[str, Any]],
    feature_stats: Dict[str, Dict[str, float]],
) -> List[Optional[float]]:
    scores: List[Optional[float]] = []
    for row in rows:
        zvals: List[float] = []
        ok = True
        for spec in feature_specs:
            feature = str(spec["feature"])
            direction = str(spec["direction"])
            x = base.maybe_float(row.get(feature))
            if x is None:
                ok = False
                break
            stats = dict(feature_stats.get(feature, {}))
            mu = float(stats.get("mean", 0.0))
            sd = float(stats.get("std", 1.0))
            if sd == 0.0:
                sd = 1.0
            oriented = float(x) if direction == "high" else -float(x)
            zvals.append(float((oriented - mu) / sd))
        scores.append(float(base.mean(zvals)) if ok and zvals else None)
    return scores


def aux_pass(value: Optional[float], *, direction: str, tau: Optional[float]) -> bool:
    if tau is None or value is None:
        return False
    if str(direction).strip().lower() == "low":
        return float(value) <= float(tau)
    return float(value) >= float(tau)


def threshold_pass(value: Optional[float], *, direction: str, tau: Optional[float]) -> bool:
    return aux_pass(value, direction=direction, tau=tau)


def oriented_threshold_pass(value: Optional[float], *, direction: str, tau: Optional[float]) -> bool:
    if tau is None or value is None:
        return False
    oriented = float(value) if str(direction).strip().lower() == "high" else -float(value)
    return oriented >= float(tau)


def build_routes(
    rows: Sequence[Dict[str, Any]],
    policy: Dict[str, Any],
) -> tuple[List[str], List[Optional[float]], List[Optional[float]]]:
    policy_type = str(policy.get("policy_type", "")).strip()
    if policy_type == "generative_trace_cascade_proxy_v1":
        anchor_feature = str(policy.get("anchor_feature") or "").strip()
        anchor_direction = str(policy.get("anchor_direction") or "high").strip()
        anchor_tau = base.maybe_float(policy.get("anchor_tau"))
        gate_feature = str(policy.get("gate_feature") or "").strip()
        gate_direction = str(policy.get("gate_direction") or "high").strip()
        gate_tau = base.maybe_float(policy.get("gate_tau"))
        anchor_values = [base.maybe_float(row.get(anchor_feature)) for row in rows]
        gate_values = [base.maybe_float(row.get(gate_feature)) for row in rows]
        routes: List[str] = []
        scores: List[Optional[float]] = []
        for anchor_value, gate_value in zip(anchor_values, gate_values):
            anchor_ok = oriented_threshold_pass(anchor_value, direction=anchor_direction, tau=anchor_tau)
            gate_ok = oriented_threshold_pass(gate_value, direction=gate_direction, tau=gate_tau)
            routes.append("baseline" if anchor_ok and gate_ok else "method")
            scores.append(gate_value if anchor_ok else None)
        return routes, scores, anchor_values

    feature_specs = list(policy.get("feature_specs", []))
    feature_stats = {
        str(k): {"mean": float(v.get("mean", 0.0)), "std": float(v.get("std", 1.0))}
        for k, v in dict(policy.get("feature_stats", {})).items()
    }
    tau = float(base.maybe_float(policy.get("tau")) or 0.0)
    scores = build_scores(rows, feature_specs, feature_stats)
    aux_feature = ""
    aux_direction = ""
    aux_tau: Optional[float] = None
    aux_values: List[Optional[float]] = [None] * len(rows)
    if policy_type == "generative_route_proxy_calibrated_v1":
        aux_feature = str(policy.get("aux_feature") or "").strip()
        aux_direction = str(policy.get("aux_direction") or "high")
        aux_tau = base.maybe_float(policy.get("aux_tau"))
        if aux_feature:
            aux_values = [base.maybe_float(row.get(aux_feature)) for row in rows]

    routes: List[str] = []
    for score, aux_value in zip(scores, aux_values):
        core_flag = score is not None and float(score) >= float(tau)
        if policy_type == "generative_route_proxy_calibrated_v1":
            flag = bool(core_flag and aux_pass(aux_value, direction=aux_direction, tau=aux_tau))
        elif policy_type == "generative_route_distill_proxy_v1":
            flag = bool(core_flag)
        else:
            raise ValueError(f"Unsupported policy_type: {policy_type}")
        routes.append("baseline" if flag else "method")
    return routes, scores, aux_values


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a frozen generative route proxy to held-out rows.")
    ap.add_argument("--claim_table_csv", type=str, required=True)
    ap.add_argument("--chair_table_csv", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--selected_policy_json", type=str, required=True)
    ap.add_argument("--subpolicy_key", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    claim_rows = base.read_csv_rows(os.path.abspath(args.claim_table_csv))
    chair_rows = base.read_csv_rows(os.path.abspath(args.chair_table_csv))
    rows = base.build_master_rows(
        claim_rows,
        chair_rows,
        os.path.abspath(args.baseline_chair_json),
        os.path.abspath(args.intervention_chair_json),
    )
    with open(os.path.abspath(args.selected_policy_json), "r", encoding="utf-8") as f:
        policy = json.load(f)
    subpolicy_key = str(args.subpolicy_key or "").strip()
    if subpolicy_key:
        nested = policy.get(subpolicy_key)
        if not isinstance(nested, dict):
            raise ValueError(f"subpolicy_key not found or not a JSON object: {subpolicy_key}")
        policy = dict(nested)

    routes, scores, aux_values = build_routes(rows, policy)
    summary = base.aggregate_routes(rows, routes)
    summary.pop("decision_rows", None)

    baseline_summary = base.aggregate_routes(rows, ["baseline"] * len(rows))
    baseline_summary.pop("decision_rows", None)
    intervention_summary = base.aggregate_routes(rows, ["method"] * len(rows))
    intervention_summary.pop("decision_rows", None)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    decision_rows_csv = os.path.join(out_dir, "decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    selected_policy_json = os.path.join(out_dir, "selected_policy.json")

    decision_rows: List[Dict[str, Any]] = []
    policy_type = str(policy.get("policy_type", "")).strip()
    aux_feature = str(policy.get("aux_feature") or "").strip()
    aux_direction = str(policy.get("aux_direction") or "")
    aux_tau = base.maybe_float(policy.get("aux_tau"))
    for row, score, aux_value, route in zip(rows, scores, aux_values, routes):
        out = dict(row)
        out["route"] = str(route)
        if policy_type == "generative_route_proxy_calibrated_v1":
            out["core_score"] = score
            out["aux_feature"] = aux_feature
            out["aux_direction"] = aux_direction
            out["aux_tau"] = "" if aux_tau is None else float(aux_tau)
            out["aux_value"] = aux_value
            out["aux_ok"] = int(aux_pass(aux_value, direction=aux_direction, tau=aux_tau))
        else:
            out["distill_proxy_score"] = score
        decision_rows.append(out)

    base.write_csv(decision_rows_csv, decision_rows)
    base.write_json(selected_policy_json, policy)
    base.write_json(
        summary_json,
        {
            "inputs": {
                "claim_table_csv": os.path.abspath(args.claim_table_csv),
                "chair_table_csv": os.path.abspath(args.chair_table_csv),
                "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
                "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
                "selected_policy_json": os.path.abspath(args.selected_policy_json),
                "subpolicy_key": subpolicy_key,
            },
            "policy": {
                "policy_type": policy_type,
                "feature_specs": list(policy.get("feature_specs", [])),
                "tau": float(base.maybe_float(policy.get("tau")) or 0.0),
                "aux_feature": aux_feature,
                "aux_direction": aux_direction,
                "aux_tau": aux_tau,
            },
            "baseline": baseline_summary,
            "intervention": intervention_summary,
            "evaluation": summary,
            "outputs": {
                "decision_rows_csv": decision_rows_csv,
                "selected_policy_json": selected_policy_json,
            },
        },
    )
    print("[saved]", decision_rows_csv)
    print("[saved]", selected_policy_json)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
