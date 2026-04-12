#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import build_generative_b_c_meta_controller as base
from apply_generative_route_proxy import build_routes


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a frozen generative route distill proxy to merged held-out rows.")
    ap.add_argument("--rows_csv", required=True)
    ap.add_argument("--selected_policy_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_col", default="distill_target")
    args = ap.parse_args()

    rows = base.read_csv_rows(os.path.abspath(args.rows_csv))
    with open(os.path.abspath(args.selected_policy_json), "r", encoding="utf-8") as f:
        policy: Dict[str, Any] = json.load(f)
    routes, scores, aux_values = build_routes(rows, policy)

    summary = base.aggregate_routes(rows, routes)
    summary.pop("decision_rows", None)
    baseline_summary = base.aggregate_routes(rows, ["baseline"] * len(rows))
    baseline_summary.pop("decision_rows", None)
    intervention_summary = base.aggregate_routes(rows, ["method"] * len(rows))
    intervention_summary.pop("decision_rows", None)

    selected_idx = [idx for idx, route in enumerate(routes) if route == "baseline"]
    targets = [int(base.maybe_int(row.get(str(args.target_col))) or 0) for row in rows]
    n_pos = sum(targets)
    tp = sum(targets[idx] for idx in selected_idx)
    summary.update(
        {
            "delta_f1_vs_int": float(summary["mean_f1"] - intervention_summary["mean_f1"]),
            "delta_recall_vs_int": float(summary["mean_recall"] - intervention_summary["mean_recall"]),
            "delta_chair_i_vs_int": float(summary["mean_chair_i"] - intervention_summary["mean_chair_i"]),
            "delta_chair_s_vs_int": float(summary["mean_chair_s"] - intervention_summary["mean_chair_s"]),
            "target_precision": base.safe_div(float(tp), float(max(1, len(selected_idx)))),
            "target_recall": base.safe_div(float(tp), float(max(1, n_pos))),
            "target_rate": base.safe_div(float(n_pos), float(max(1, len(rows)))),
        }
    )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    decision_rows: List[Dict[str, Any]] = []
    policy_type = str(policy.get("policy_type", "")).strip()
    aux_feature = str(policy.get("aux_feature") or "").strip()
    for row, route, score, aux_value in zip(rows, routes, scores, aux_values):
        out = dict(row)
        out["route"] = route
        out["distill_proxy_score"] = score
        if aux_feature:
            out["aux_feature"] = aux_feature
            out["aux_value"] = aux_value
        out["target_match"] = int((route == "baseline") == (int(base.maybe_int(row.get(str(args.target_col))) or 0) == 1))
        decision_rows.append(out)

    decision_rows_csv = os.path.join(out_dir, "decision_rows.csv")
    selected_policy_json = os.path.join(out_dir, "selected_policy.json")
    summary_json = os.path.join(out_dir, "summary.json")
    base.write_csv(decision_rows_csv, decision_rows)
    base.write_json(selected_policy_json, policy)
    base.write_json(
        summary_json,
        {
            "inputs": {
                "rows_csv": os.path.abspath(args.rows_csv),
                "selected_policy_json": os.path.abspath(args.selected_policy_json),
                "target_col": str(args.target_col),
            },
            "policy": {
                "policy_type": policy_type,
                "feature_specs": list(policy.get("feature_specs", [])),
                "tau": policy.get("tau"),
                "aux_feature": aux_feature,
                "aux_tau": policy.get("aux_tau"),
            },
            "baseline": baseline_summary,
            "intervention": intervention_summary,
            "evaluation": summary,
            "outputs": {
                "decision_rows_csv": decision_rows_csv,
                "selected_policy_json": selected_policy_json,
                "summary_json": summary_json,
            },
        },
    )
    print("[saved]", decision_rows_csv)
    print("[saved]", selected_policy_json)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
