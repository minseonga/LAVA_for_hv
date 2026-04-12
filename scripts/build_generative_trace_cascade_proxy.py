#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build_generative_b_c_meta_controller as base


DEFAULT_ANCHOR_FEATURES = [
    "probe_n_unique_object_mentions",
    "probe_n_object_mentions",
    "probe_last_object_pos_frac",
    "probe_tail_tokens_after_last_object",
    "probe_tail_after_last_object_eos_margin_mean_real",
    "probe_tail_after_last_object_entropy_mean_real",
    "probe_object_token_fraction",
    "probe_n_content_tokens",
]

DEFAULT_GATE_FEATURES = [
    "probe_object_token_gap_min_real",
    "probe_object_token_lp_min_real",
    "probe_object_token_entropy_max_real",
    "probe_tail_after_last_object_gap_min_real",
    "probe_tail_after_last_object_lp_mean_real",
    "probe_tail_after_last_object_entropy_mean_real",
    "probe_tail_after_last_object_eos_margin_mean_real",
    "probe_last4_eos_margin_mean_real",
    "probe_entropy_tail_minus_head_real",
    "probe_gap_tail_minus_head_real",
    "probe_lp_tail_minus_head_real",
]


def parse_list(spec: str, default: Sequence[str]) -> List[str]:
    if str(spec or "").strip().lower() in {"", "default"}:
        return [str(x) for x in default]
    return [part.strip() for part in str(spec).split(",") if part.strip()]


def parse_quantiles(spec: str) -> List[float]:
    vals: List[float] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(min(1.0, max(0.0, float(s))))
    return sorted(set(vals)) or [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]


def direction_for(rows: Sequence[Dict[str, Any]], feature: str, target_col: str) -> Optional[Dict[str, Any]]:
    return base.evaluate_feature(rows, feature, f"{target_col}:binary:0")


def oriented_values(rows: Sequence[Dict[str, Any]], feature: str, direction: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for row in rows:
        x = base.maybe_float(row.get(feature))
        if x is None:
            out.append(None)
        else:
            out.append(float(x) if direction == "high" else -float(x))
    return out


def pass_threshold(value: Optional[float], tau: float) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) >= float(tau)


def binary_stats(rows: Sequence[Dict[str, Any]], routes: Sequence[str], target_col: str) -> Dict[str, float]:
    selected = [idx for idx, route in enumerate(routes) if route == "baseline"]
    labels = [int(base.maybe_int(row.get(target_col)) or 0) for row in rows]
    n_pos = sum(labels)
    tp = sum(labels[idx] for idx in selected)
    return {
        "target_precision": base.safe_div(float(tp), float(max(1, len(selected)))),
        "target_recall": base.safe_div(float(tp), float(max(1, n_pos))),
        "target_rate": base.safe_div(float(n_pos), float(max(1, len(rows)))),
    }


def route_by_cascade(
    anchor_scores: Sequence[Optional[float]],
    gate_scores: Sequence[Optional[float]],
    anchor_tau: float,
    gate_tau: float,
) -> List[str]:
    return [
        "baseline" if pass_threshold(a, anchor_tau) and pass_threshold(g, gate_tau) else "method"
        for a, g in zip(anchor_scores, gate_scores)
    ]


def compare_key(row: Dict[str, Any], objective: str) -> Tuple[float, ...]:
    if objective == "recall":
        primary = float(row["mean_recall"])
    elif objective == "recall_minus_chairi":
        primary = float(row["mean_recall"] - row["mean_chair_i"])
    elif objective == "f1_minus_chairi":
        primary = float(row["mean_f1_minus_chairi"])
    else:
        primary = float(row["mean_f1"])
    return (
        primary,
        float(row["target_precision"]),
        float(row["target_recall"]),
        -float(row["mean_chair_i"]),
        -float(row["mean_chair_s"]),
        -float(row["baseline_rate"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a two-stage intervention-trace cascade proxy: omission anchor AND cost gate.")
    ap.add_argument("--decision_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_col", default="v46_safe_target")
    ap.add_argument("--anchor_target_col", default="proxy_route")
    ap.add_argument("--anchor_positive_value", default="baseline")
    ap.add_argument("--anchor_features", default="default")
    ap.add_argument("--gate_features", default="default")
    ap.add_argument("--tau_quantiles", default="0.0,0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.0")
    ap.add_argument("--min_anchor_target_recall", type=float, default=0.5)
    ap.add_argument("--max_anchor_rate", type=float, default=0.4)
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=0.08)
    ap.add_argument("--min_recall_gain_vs_intervention", type=float, default=0.0)
    ap.add_argument("--min_f1_gain_vs_intervention", type=float, default=-1.0)
    ap.add_argument("--max_chair_i_delta_vs_intervention", type=float, default=0.005)
    ap.add_argument("--max_chair_s_delta_vs_intervention", type=float, default=0.005)
    ap.add_argument("--selection_objective", choices=["f1", "f1_minus_chairi", "recall", "recall_minus_chairi"], default="recall_minus_chairi")
    args = ap.parse_args()

    rows = base.read_csv_rows(os.path.abspath(args.decision_rows_csv))
    anchor_features = parse_list(args.anchor_features, DEFAULT_ANCHOR_FEATURES)
    gate_features = parse_list(args.gate_features, DEFAULT_GATE_FEATURES)
    quantiles = parse_quantiles(args.tau_quantiles)

    anchor_rows: List[Dict[str, Any]] = []
    rows_for_anchor = []
    for row in rows:
        item = dict(row)
        item["_anchor_target"] = int(str(row.get(str(args.anchor_target_col), "")).strip() == str(args.anchor_positive_value))
        rows_for_anchor.append(item)

    anchor_metrics: Dict[str, Dict[str, Any]] = {}
    gate_metrics: Dict[str, Dict[str, Any]] = {}
    for feature in anchor_features:
        result = direction_for(rows_for_anchor, feature, "_anchor_target")
        if result is not None:
            anchor_metrics[feature] = dict(result)
    for feature in gate_features:
        result = direction_for(rows, feature, str(args.target_col))
        if result is not None:
            gate_metrics[feature] = dict(result)

    intervention = base.aggregate_routes(rows, ["method"] * len(rows))
    intervention.pop("decision_rows", None)

    all_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_routes: List[str] = []

    for anchor_feature, anchor_info in anchor_metrics.items():
        anchor_direction = str(anchor_info["direction"])
        anchor_scores = oriented_values(rows, anchor_feature, anchor_direction)
        anchor_valid = [float(x) for x in anchor_scores if x is not None and math.isfinite(float(x))]
        for anchor_tau in base.quantiles_to_thresholds(anchor_valid, quantiles):
            anchor_pass = [pass_threshold(score, float(anchor_tau)) for score in anchor_scores]
            anchor_routes = ["baseline" if flag else "method" for flag in anchor_pass]
            anchor_rate = base.safe_div(float(sum(anchor_pass)), float(max(1, len(anchor_pass))))
            anchor_target_stats = binary_stats(rows_for_anchor, anchor_routes, "_anchor_target")
            if float(anchor_rate) > float(args.max_anchor_rate):
                continue
            if float(anchor_target_stats["target_recall"]) < float(args.min_anchor_target_recall):
                continue
            for gate_feature, gate_info in gate_metrics.items():
                gate_direction = str(gate_info["direction"])
                gate_scores = oriented_values(rows, gate_feature, gate_direction)
                gate_valid = [float(x) for x in gate_scores if x is not None and math.isfinite(float(x))]
                for gate_tau in base.quantiles_to_thresholds(gate_valid, quantiles):
                    routes = route_by_cascade(anchor_scores, gate_scores, float(anchor_tau), float(gate_tau))
                    summary = base.aggregate_routes(rows, routes)
                    summary.pop("decision_rows", None)
                    target = binary_stats(rows, routes, str(args.target_col))
                    summary.update(target)
                    summary.update(
                        {
                            "anchor_feature": str(anchor_feature),
                            "anchor_direction": str(anchor_direction),
                            "anchor_tau": float(anchor_tau),
                            "anchor_auroc": float(anchor_info["auroc"]),
                            "anchor_target_precision": float(anchor_target_stats["target_precision"]),
                            "anchor_target_recall": float(anchor_target_stats["target_recall"]),
                            "anchor_target_rate": float(anchor_target_stats["target_rate"]),
                            "anchor_rate": float(anchor_rate),
                            "gate_feature": str(gate_feature),
                            "gate_direction": str(gate_direction),
                            "gate_tau": float(gate_tau),
                            "gate_auroc": float(gate_info["auroc"]),
                            "delta_f1_vs_int": float(summary["mean_f1"] - intervention["mean_f1"]),
                            "delta_recall_vs_int": float(summary["mean_recall"] - intervention["mean_recall"]),
                            "delta_chair_i_vs_int": float(summary["mean_chair_i"] - intervention["mean_chair_i"]),
                            "delta_chair_s_vs_int": float(summary["mean_chair_s"] - intervention["mean_chair_s"]),
                        }
                    )
                    summary["feasible"] = int(
                        float(summary["baseline_rate"]) >= float(args.min_baseline_rate)
                        and float(summary["baseline_rate"]) <= float(args.max_baseline_rate)
                        and float(summary["delta_recall_vs_int"]) >= float(args.min_recall_gain_vs_intervention)
                        and float(summary["delta_f1_vs_int"]) >= float(args.min_f1_gain_vs_intervention)
                        and float(summary["delta_chair_i_vs_int"]) <= float(args.max_chair_i_delta_vs_intervention)
                        and float(summary["delta_chair_s_vs_int"]) <= float(args.max_chair_s_delta_vs_intervention)
                    )
                    all_rows.append(dict(summary))
                    if int(summary["feasible"]) != 1:
                        continue
                    if best is None or compare_key(summary, str(args.selection_objective)) > compare_key(best, str(args.selection_objective)):
                        best = dict(summary)
                        best_routes = list(routes)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cascade_results_csv = os.path.join(out_dir, "cascade_results.csv")
    selected_policy_json = os.path.join(out_dir, "selected_policy.json")
    decision_rows_csv = os.path.join(out_dir, "decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    all_rows.sort(key=lambda row: (-int(row.get("feasible") or 0), -float(row.get("mean_recall") or -1e9), -float(row.get("target_precision") or 0.0)))
    base.write_csv(cascade_results_csv, all_rows)

    payload: Dict[str, Any] = {
        "inputs": {
            "decision_rows_csv": os.path.abspath(args.decision_rows_csv),
            "target_col": str(args.target_col),
            "anchor_target_col": str(args.anchor_target_col),
            "anchor_positive_value": str(args.anchor_positive_value),
            "anchor_features": anchor_features,
            "gate_features": gate_features,
            "min_anchor_target_recall": float(args.min_anchor_target_recall),
            "max_anchor_rate": float(args.max_anchor_rate),
            "min_baseline_rate": float(args.min_baseline_rate),
            "max_baseline_rate": float(args.max_baseline_rate),
            "min_recall_gain_vs_intervention": float(args.min_recall_gain_vs_intervention),
            "max_chair_i_delta_vs_intervention": float(args.max_chair_i_delta_vs_intervention),
            "max_chair_s_delta_vs_intervention": float(args.max_chair_s_delta_vs_intervention),
            "selection_objective": str(args.selection_objective),
        },
        "counts": {
            "n_rows": int(len(rows)),
            "n_anchor_features": int(len(anchor_metrics)),
            "n_gate_features": int(len(gate_metrics)),
            "n_candidates": int(len(all_rows)),
            "n_feasible": int(sum(int(row.get("feasible") or 0) for row in all_rows)),
        },
        "baseline": {k: v for k, v in base.aggregate_routes(rows, ["baseline"] * len(rows)).items() if k != "decision_rows"},
        "intervention": intervention,
        "best_policy": best,
        "outputs": {
            "cascade_results_csv": cascade_results_csv,
        },
    }

    if best is None:
        base.write_json(summary_json, {**payload, "error": "No feasible cascade proxy policy found."})
        raise RuntimeError("No feasible cascade proxy policy found.")

    policy = {
        "policy_type": "generative_trace_cascade_proxy_v1",
        "source_decision_rows_csv": os.path.abspath(args.decision_rows_csv),
        "target_col": str(args.target_col),
        "anchor_feature": str(best["anchor_feature"]),
        "anchor_direction": str(best["anchor_direction"]),
        "anchor_tau": float(best["anchor_tau"]),
        "gate_feature": str(best["gate_feature"]),
        "gate_direction": str(best["gate_direction"]),
        "gate_tau": float(best["gate_tau"]),
    }
    base.write_json(selected_policy_json, policy)

    decision_rows: List[Dict[str, Any]] = []
    for row, route in zip(rows, best_routes):
        out = dict(row)
        out["route"] = route
        out["target_match"] = int((route == "baseline") == (int(base.maybe_int(row.get(str(args.target_col))) or 0) == 1))
        decision_rows.append(out)
    base.write_csv(decision_rows_csv, decision_rows)
    payload["outputs"].update({"selected_policy_json": selected_policy_json, "decision_rows_csv": decision_rows_csv})
    base.write_json(summary_json, payload)
    print("[saved]", cascade_results_csv)
    print("[saved]", selected_policy_json)
    print("[saved]", decision_rows_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
