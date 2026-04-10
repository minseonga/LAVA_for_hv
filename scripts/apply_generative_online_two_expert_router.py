#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import build_generative_b_c_meta_controller as base
import combine_generative_teacher_experts as union


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


def score_rows_with_policy(
    rows: Sequence[Dict[str, Any]],
    policy: Dict[str, Any],
    prefix: str,
) -> List[Dict[str, Any]]:
    feature_specs = list(policy.get("feature_specs", []))
    feature_stats = {
        str(k): {"mean": float(v.get("mean", 0.0)), "std": float(v.get("std", 1.0))}
        for k, v in dict(policy.get("feature_stats", {})).items()
    }
    scores = build_scores(rows, feature_specs, feature_stats)
    out: List[Dict[str, Any]] = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item[f"_{prefix}_score"] = float(score) if score is not None else 0.0
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a frozen online two-expert generative router to held-out rows.")
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
    if str(policy.get("policy_type", "")).strip() != "generative_teacher_union_v2":
        raise ValueError("selected_policy_json must be a generative_teacher_union_v2 policy")

    expert_a_policy = dict(policy.get("expert_a_policy", {}))
    expert_b_policy = dict(policy.get("expert_b_policy", {}))
    if not expert_a_policy or not expert_b_policy:
        raise ValueError("selected_policy_json must embed expert_a_policy and expert_b_policy")

    expert_a_rows = score_rows_with_policy(rows, expert_a_policy, "expert_a")
    expert_b_rows = score_rows_with_policy(rows, expert_b_policy, "expert_b")
    merged_rows: List[Dict[str, Any]] = []
    for a_row, b_row in zip(expert_a_rows, expert_b_rows):
        merged = dict(a_row)
        merged["_expert_b_score"] = b_row["_expert_b_score"]
        merged_rows.append(merged)

    final_routes, decision_rows, counts = union.combine_routes(
        merged_rows,
        union_mode=str(policy.get("union_mode", "priority_a")),
        expert_a_gate=str(policy.get("expert_a_gate", "none")),
        expert_b_gate=str(policy.get("expert_b_gate", "none")),
        expert_a_tau=float(base.maybe_float(expert_a_policy.get("tau")) or 0.0),
        expert_b_tau=float(base.maybe_float(expert_b_policy.get("tau")) or 0.0),
        expert_a_offset=float(base.maybe_float(policy.get("expert_a_tau_offset")) or 0.0),
        expert_b_offset=float(base.maybe_float(policy.get("expert_b_tau_offset")) or 0.0),
        expert_b_aux_feature=str(policy.get("expert_b_aux_feature") or expert_b_policy.get("aux_feature") or ""),
        expert_b_aux_direction=str(policy.get("expert_b_aux_direction") or expert_b_policy.get("aux_direction") or "high"),
        expert_b_aux_tau=base.maybe_float(policy.get("expert_b_aux_tau")),
    )

    baseline_summary = union.aggregate_without_rows(rows, ["baseline"] * len(rows))
    intervention_summary = union.aggregate_without_rows(rows, ["method"] * len(rows))
    summary = union.route_summary(rows, final_routes)

    labeled_teacher_rows = [union.teacher_label(row) for row in rows]
    teacher_labels_available = any(label is not None for label in labeled_teacher_rows)
    if teacher_labels_available:
        teacher_routes = ["baseline" if int(label or 0) == 1 else "method" for label in labeled_teacher_rows]
        teacher_summary = union.route_summary(rows, teacher_routes)
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

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    decision_rows_csv = os.path.join(out_dir, "decision_rows.csv")
    selected_policy_json = os.path.join(out_dir, "selected_policy.json")
    summary_json = os.path.join(out_dir, "summary.json")

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
            },
            "counts": {
                "n_rows": int(len(rows)),
                "teacher_positive_rate": teacher_summary.get("teacher_rate"),
                **counts,
            },
            "baseline": baseline_summary,
            "intervention": intervention_summary,
            "teacher_oracle": teacher_summary,
            "union_policy": {
                "union_mode": str(policy.get("union_mode", "priority_a")),
                "expert_a_tau_offset": float(base.maybe_float(policy.get("expert_a_tau_offset")) or 0.0),
                "expert_b_tau_offset": float(base.maybe_float(policy.get("expert_b_tau_offset")) or 0.0),
                "expert_b_aux_feature": str(policy.get("expert_b_aux_feature") or expert_b_policy.get("aux_feature") or ""),
                "expert_b_aux_direction": str(policy.get("expert_b_aux_direction") or expert_b_policy.get("aux_direction") or "high"),
                "expert_b_aux_tau": base.maybe_float(policy.get("expert_b_aux_tau")),
                **counts,
                **summary,
            },
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
