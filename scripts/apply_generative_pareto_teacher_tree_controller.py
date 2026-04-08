#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import build_generative_b_c_meta_controller as base
import build_generative_pareto_teacher_tree_controller as tree


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a frozen generative Pareto-teacher tree controller to held-out rows.")
    ap.add_argument("--claim_table_csv", type=str, required=True)
    ap.add_argument("--chair_table_csv", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--selected_tree_json", type=str, required=True)
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

    with open(os.path.abspath(args.selected_tree_json), "r", encoding="utf-8") as f:
        policy = json.load(f)

    means = {str(k): float(v) for k, v in dict(policy.get("feature_means", {})).items()}
    built_tree = dict(policy.get("tree", {}))
    tau = float(policy.get("tau", 0.0))

    probs: List[float] = [tree.predict_tree_row(row, built_tree, means) for row in rows]
    routes: List[str] = [base.route_by_score(float(prob), float(tau)) for prob in probs]
    summary = base.aggregate_routes(rows, routes)

    decision_rows: List[Dict[str, Any]] = []
    for row, prob, route in zip(rows, probs, routes):
        out = dict(row)
        out["tree_fallback_prob"] = float(prob)
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
                "selected_tree_json": os.path.abspath(args.selected_tree_json),
            },
            "policy": {
                "policy_type": str(policy.get("policy_type", "")),
                "teacher_mode": policy.get("teacher_mode"),
                "min_f1_gain": policy.get("min_f1_gain"),
                "constraint_mode": policy.get("constraint_mode"),
                "chair_eps": policy.get("chair_eps"),
                "selection_objective": policy.get("selection_objective"),
                "feature_names": policy.get("feature_names", []),
                "tau": float(tau),
                "feature_usage": policy.get("feature_usage", {}),
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
