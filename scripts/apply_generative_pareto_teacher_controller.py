#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import build_generative_b_c_meta_controller as base


def build_scores(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Dict[str, Any]],
    feature_stats: Dict[str, Dict[str, float]],
) -> List[float]:
    scores: List[float] = []
    for row in rows:
        total = 0.0
        for spec in feature_specs:
            feature = str(spec["feature"])
            direction = str(spec["direction"])
            weight = float(spec.get("weight", 0.0))
            stats = dict(feature_stats.get(feature, {}))
            mu = float(stats.get("mean", 0.0))
            sd = float(stats.get("std", 1.0))
            if sd == 0.0:
                sd = 1.0
            x = base.maybe_float(row.get(feature))
            if x is None:
                z = 0.0
            else:
                oriented = float(x) if direction == "high" else -float(x)
                z = (oriented - mu) / sd
            total += weight * z
        scores.append(float(total))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a frozen generative Pareto-teacher linear controller to held-out rows.")
    ap.add_argument("--claim_table_csv", type=str, required=True)
    ap.add_argument("--chair_table_csv", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--selected_policy_json", type=str, required=True)
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

    feature_specs = list(policy.get("feature_specs", []))
    feature_stats = {
        str(k): {"mean": float(v.get("mean", 0.0)), "std": float(v.get("std", 1.0))}
        for k, v in dict(policy.get("feature_stats", {})).items()
    }
    tau = float(policy.get("tau", 0.0))

    scores = build_scores(rows, feature_specs, feature_stats)
    routes: List[str] = [base.route_by_score(score, float(tau)) for score in scores]
    summary = base.aggregate_routes(rows, routes)

    decision_rows: List[Dict[str, Any]] = []
    for row, score, route in zip(rows, scores, routes):
        out = dict(row)
        out["pareto_teacher_score"] = float(score)
        out["route"] = str(route)
        decision_rows.append(out)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    decision_csv = os.path.join(out_dir, "decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    base.write_csv(decision_csv, decision_rows)
    base.write_json(
        summary_json,
        {
            "inputs": {
                "claim_table_csv": os.path.abspath(args.claim_table_csv),
                "chair_table_csv": os.path.abspath(args.chair_table_csv),
                "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
                "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
                "selected_policy_json": os.path.abspath(args.selected_policy_json),
            },
            "policy": {
                "policy_type": str(policy.get("policy_type", "")),
                "teacher_mode": policy.get("teacher_mode"),
                "min_f1_gain": policy.get("min_f1_gain"),
                "min_recall_gain": policy.get("min_recall_gain"),
                "constraint_mode": policy.get("constraint_mode"),
                "chair_eps": policy.get("chair_eps"),
                "selection_objective": policy.get("selection_objective"),
                "feature_specs": feature_specs,
                "tau": float(tau),
            },
            "baseline": {k: v for k, v in base.aggregate_routes(rows, ["baseline"] * len(rows)).items() if k != "decision_rows"},
            "intervention": {k: v for k, v in base.aggregate_routes(rows, ["method"] * len(rows)).items() if k != "decision_rows"},
            "evaluation": {k: v for k, v in summary.items() if k != "decision_rows"},
            "outputs": {
                "decision_rows_csv": decision_csv,
            },
        },
    )
    print("[saved]", decision_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
