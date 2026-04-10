#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def target_stats(rows: Sequence[Dict[str, Any]], routes: Sequence[str], target_col: str) -> Dict[str, float]:
    n_selected = sum(1 for route in routes if route == "baseline")
    tp = sum(
        int(base.maybe_int(row.get(target_col)) or 0)
        for row, route in zip(rows, routes)
        if route == "baseline"
    )
    n_pos = sum(int(base.maybe_int(row.get(target_col)) or 0) for row in rows)
    return {
        "target_precision": base.safe_div(float(tp), float(max(1, n_selected))),
        "target_recall": base.safe_div(float(tp), float(max(1, n_pos))),
        "target_rate": base.safe_div(float(n_pos), float(max(1, len(rows)))),
    }


def teacher_stats(rows: Sequence[Dict[str, Any]], routes: Sequence[str]) -> Dict[str, float]:
    n_selected = sum(1 for route in routes if route == "baseline")
    tp = sum(
        int(base.maybe_int(row.get("teacher_fallback")) or 0)
        for row, route in zip(rows, routes)
        if route == "baseline"
    )
    n_pos = sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows)
    return {
        "teacher_precision": base.safe_div(float(tp), float(max(1, n_selected))),
        "teacher_recall": base.safe_div(float(tp), float(max(1, n_pos))),
        "teacher_rate": base.safe_div(float(n_pos), float(max(1, len(rows)))),
    }


def aux_pass(value: Optional[float], *, direction: str, tau: float) -> bool:
    if value is None:
        return False
    if str(direction) == "high":
        return float(value) >= float(tau)
    return float(value) <= float(tau)


def compare_key(summary: Dict[str, Any], objective: str) -> Tuple[float, float, float, float, float]:
    primary = float(summary["mean_f1"])
    if str(objective) == "teacher_precision":
        primary = float(summary.get("teacher_precision") or 0.0)
    elif str(objective) == "target_precision":
        primary = float(summary.get("target_precision") or 0.0)
    elif str(objective) == "f1_minus_chairi":
        primary = float(summary["mean_f1_minus_chairi"])
    return (
        primary,
        -float(summary["mean_chair_i"]),
        -float(summary["mean_chair_s"]),
        float(summary.get("teacher_precision") or 0.0),
        -float(summary["baseline_rate"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate a distill proxy with a 2D threshold on core score and auxiliary feature.")
    ap.add_argument("--decision_rows_csv", type=str, required=True)
    ap.add_argument("--selected_policy_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--aux_feature", type=str, required=True)
    ap.add_argument("--aux_direction", type=str, default="high", choices=["high", "low"])
    ap.add_argument("--aux_quantiles", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95,0.98")
    ap.add_argument("--tau_offsets", type=str, default="-1.0,-0.5,0,0.5,1.0,1.5,2.0")
    ap.add_argument("--target_col", type=str, default="distill_target")
    ap.add_argument("--min_baseline_rate", type=float, default=0.02)
    ap.add_argument("--max_baseline_rate", type=float, default=0.08)
    ap.add_argument("--min_f1_gain_vs_intervention", type=float, default=0.0)
    ap.add_argument("--max_chair_i_delta_vs_intervention", type=float, default=0.0)
    ap.add_argument("--max_chair_s_delta_vs_intervention", type=float, default=0.0)
    ap.add_argument("--selection_objective", type=str, default="f1", choices=["f1", "f1_minus_chairi", "teacher_precision", "target_precision"])
    args = ap.parse_args()

    rows = base.read_csv_rows(os.path.abspath(args.decision_rows_csv))
    with open(os.path.abspath(args.selected_policy_json), "r", encoding="utf-8") as f:
        policy = json.load(f)

    feature_specs = list(policy.get("feature_specs", []))
    feature_stats = {
        str(k): {"mean": float(v.get("mean", 0.0)), "std": float(v.get("std", 1.0))}
        for k, v in dict(policy.get("feature_stats", {})).items()
    }
    base_tau = float(policy.get("tau", 0.0))
    core_scores = build_scores(rows, feature_specs, feature_stats)
    aux_values = [base.maybe_float(row.get(str(args.aux_feature))) for row in rows]
    valid_aux = [float(v) for v in aux_values if v is not None and math.isfinite(float(v))]
    if not valid_aux:
        raise RuntimeError(f"Aux feature not found or empty: {args.aux_feature}")

    tau_offsets = [float(x) for x in str(args.tau_offsets).split(",") if str(x).strip()]
    aux_quantiles = [float(x) for x in str(args.aux_quantiles).split(",") if str(x).strip()]
    aux_taus = [float(x) for x in base.quantiles_to_thresholds(valid_aux, aux_quantiles)]

    intervention = base.aggregate_routes(rows, ["method"] * len(rows))
    sweep_rows: List[Dict[str, Any]] = []
    best_summary: Optional[Dict[str, Any]] = None
    best_routes: List[str] = []
    best_scores: List[Optional[float]] = []

    for tau_offset in tau_offsets:
        tau = float(base_tau + float(tau_offset))
        for aux_tau in aux_taus:
            routes: List[str] = []
            aux_pass_count = 0
            for score, aux_val in zip(core_scores, aux_values):
                core_flag = score is not None and float(score) >= float(tau)
                aux_flag = aux_pass(aux_val, direction=str(args.aux_direction), tau=float(aux_tau))
                if aux_flag:
                    aux_pass_count += 1
                routes.append("baseline" if (core_flag and aux_flag) else "method")

            summary = base.aggregate_routes(rows, routes)
            summary.pop("decision_rows", None)
            summary.update(target_stats(rows, routes, str(args.target_col)))
            summary.update(teacher_stats(rows, routes))
            summary["tau"] = float(tau)
            summary["tau_offset"] = float(tau_offset)
            summary["aux_feature"] = str(args.aux_feature)
            summary["aux_direction"] = str(args.aux_direction)
            summary["aux_tau"] = float(aux_tau)
            summary["aux_pass_rate"] = base.safe_div(float(aux_pass_count), float(max(1, len(rows))))
            summary["delta_f1_vs_int"] = float(summary["mean_f1"] - intervention["mean_f1"])
            summary["delta_chair_i_vs_int"] = float(summary["mean_chair_i"] - intervention["mean_chair_i"])
            summary["delta_chair_s_vs_int"] = float(summary["mean_chair_s"] - intervention["mean_chair_s"])
            summary["feasible"] = int(
                float(summary["baseline_rate"]) >= float(args.min_baseline_rate)
                and float(summary["baseline_rate"]) <= float(args.max_baseline_rate)
                and float(summary["delta_f1_vs_int"]) > float(args.min_f1_gain_vs_intervention)
                and float(summary["delta_chair_i_vs_int"]) <= float(args.max_chair_i_delta_vs_intervention)
                and float(summary["delta_chair_s_vs_int"]) <= float(args.max_chair_s_delta_vs_intervention)
            )
            sweep_rows.append(dict(summary))
            if int(summary["feasible"]) != 1:
                continue
            if best_summary is None or compare_key(summary, str(args.selection_objective)) > compare_key(best_summary, str(args.selection_objective)):
                best_summary = dict(summary)
                best_routes = list(routes)
                best_scores = list(core_scores)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sweep_csv = os.path.join(out_dir, "calibration_sweep.csv")
    decision_csv = os.path.join(out_dir, "decision_rows.csv")
    selected_policy_json = os.path.join(out_dir, "selected_policy.json")
    summary_json = os.path.join(out_dir, "summary.json")
    base.write_csv(sweep_csv, sweep_rows)

    if best_summary is None:
        base.write_json(
            summary_json,
            {
                "inputs": {
                    "decision_rows_csv": os.path.abspath(args.decision_rows_csv),
                    "selected_policy_json": os.path.abspath(args.selected_policy_json),
                    "aux_feature": str(args.aux_feature),
                    "aux_direction": str(args.aux_direction),
                    "tau_offsets": tau_offsets,
                    "aux_quantiles": aux_quantiles,
                },
                "baseline": {k: v for k, v in base.aggregate_routes(rows, ["baseline"] * len(rows)).items() if k != "decision_rows"},
                "intervention": {k: v for k, v in intervention.items() if k != "decision_rows"},
                "outputs": {
                    "calibration_sweep_csv": sweep_csv,
                },
                "error": "No feasible calibrated proxy policy found.",
            },
        )
        raise RuntimeError("No feasible calibrated proxy policy found.")

    decision_rows: List[Dict[str, Any]] = []
    for row, score, aux_val, route in zip(rows, best_scores, aux_values, best_routes):
        out = dict(row)
        out["core_score"] = score
        out["aux_value"] = aux_val
        out["route"] = route
        decision_rows.append(out)
    base.write_csv(decision_csv, decision_rows)

    selected_policy = {
        "policy_type": "generative_route_proxy_calibrated_v1",
        "source_decision_rows_csv": os.path.abspath(args.decision_rows_csv),
        "base_proxy_policy_json": os.path.abspath(args.selected_policy_json),
        "feature_specs": feature_specs,
        "feature_stats": feature_stats,
        "base_tau": float(base_tau),
        "tau": float(best_summary["tau"]),
        "tau_offset": float(best_summary["tau_offset"]),
        "aux_feature": str(args.aux_feature),
        "aux_direction": str(args.aux_direction),
        "aux_tau": float(best_summary["aux_tau"]),
    }
    base.write_json(selected_policy_json, selected_policy)

    summary = {
        "inputs": {
            "decision_rows_csv": os.path.abspath(args.decision_rows_csv),
            "selected_policy_json": os.path.abspath(args.selected_policy_json),
            "aux_feature": str(args.aux_feature),
            "aux_direction": str(args.aux_direction),
            "tau_offsets": tau_offsets,
            "aux_quantiles": aux_quantiles,
            "target_col": str(args.target_col),
            "min_baseline_rate": float(args.min_baseline_rate),
            "max_baseline_rate": float(args.max_baseline_rate),
            "min_f1_gain_vs_intervention": float(args.min_f1_gain_vs_intervention),
            "max_chair_i_delta_vs_intervention": float(args.max_chair_i_delta_vs_intervention),
            "max_chair_s_delta_vs_intervention": float(args.max_chair_s_delta_vs_intervention),
            "selection_objective": str(args.selection_objective),
        },
        "baseline": {k: v for k, v in base.aggregate_routes(rows, ["baseline"] * len(rows)).items() if k != "decision_rows"},
        "intervention": {k: v for k, v in intervention.items() if k != "decision_rows"},
        "best_policy": best_summary,
        "outputs": {
            "calibration_sweep_csv": sweep_csv,
            "decision_rows_csv": decision_csv,
            "selected_policy_json": selected_policy_json,
        },
    }
    base.write_json(summary_json, summary)
    print("[saved]", sweep_csv)
    print("[saved]", decision_csv)
    print("[saved]", selected_policy_json)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
