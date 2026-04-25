#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import build_posthoc_b_c_fusion_controller as base
import build_pcp_c_d_controller as pcp


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def choose_policy(bundle: Dict[str, Any], family: str) -> Dict[str, Any]:
    best_results = bundle.get("best_results") or {}
    if family == "selected":
        policy = bundle.get("selected_policy")
        if not policy:
            raise RuntimeError("selected_policy is missing from policy_json")
        return policy
    policy = best_results.get(family)
    if not policy:
        raise RuntimeError(f"family={family!r} is unavailable in policy_json")
    return policy


def compute_score(
    row: Dict[str, Any],
    *,
    c_features: List[Dict[str, Any]],
    d_features: List[Dict[str, Any]],
    family: str,
    alpha: float,
) -> Optional[float]:
    c_score = pcp.mean_z_score(row, c_features)
    d_score = pcp.mean_z_score(row, d_features)
    if family == "c_only":
        return c_score
    if family == "d_only":
        return d_score
    if c_score is None or d_score is None:
        return None
    return float((1.0 - float(alpha)) * float(c_score) + float(alpha) * float(d_score))


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a calibrated PCP C/D controller to held-out rows.")
    ap.add_argument("--rows_csv", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--family", type=str, default="selected", choices=["selected", "c_only", "d_only", "cd_fusion"])
    ap.add_argument("--derive_decision_kl", type=parse_bool, default=True)
    args = ap.parse_args()

    rows = pcp.load_rows(os.path.abspath(args.rows_csv), derive_decision_kl=bool(args.derive_decision_kl))
    with open(os.path.abspath(args.policy_json), "r", encoding="utf-8") as f:
        bundle: Dict[str, Any] = json.load(f)
    policy = choose_policy(bundle, str(args.family))

    c_features = list(bundle.get("selected_c_features") or [])
    d_features = list(bundle.get("selected_d_features") or [])
    family = str(policy["family"])
    alpha = float(policy["alpha"])
    tau = float(policy["tau"])

    route_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    for row in rows:
        score = compute_score(
            row,
            c_features=c_features,
            d_features=d_features,
            family=family,
            alpha=alpha,
        )
        route = "method"
        if score is not None and float(score) >= float(tau):
            route = "baseline"
        final_text = str(row.get("intervention_text", ""))
        final_source = "method"
        if route == "baseline":
            final_text = str(row.get("baseline_text", "")) or final_text
            final_source = "baseline_cached" if str(row.get("baseline_text", "")).strip() else "method_missing_baseline"
        route_row = {
            "id": str(row.get("id", "")),
            "image": str(row.get("image", "")),
            "question": str(row.get("question", "")),
            "route": route,
            "family": family,
            "alpha": alpha,
            "tau": tau,
            "score": score,
            "c_score": pcp.mean_z_score(row, c_features),
            "d_score": pcp.mean_z_score(row, d_features),
            "harm": int(base.maybe_int(row.get("harm")) or 0),
            "help": int(base.maybe_int(row.get("help")) or 0),
            "baseline_correct": row.get("baseline_correct"),
            "intervention_correct": row.get("intervention_correct"),
            "final_source": final_source,
            "final_text": final_text,
        }
        route_rows.append(route_row)
        pred_rows.append(
            {
                "question_id": str(row.get("id", "")),
                "id": str(row.get("id", "")),
                "image": str(row.get("image", "")),
                "text": final_text,
                "route": route,
                "family": family,
                "source": final_source,
            }
        )

    evaluation = pcp.evaluate_policy(
        rows,
        c_features=c_features,
        d_features=d_features,
        family=family,
        alpha=alpha,
        tau=tau,
    )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    route_rows_csv = os.path.join(out_dir, "pcp_route_rows.csv")
    pred_jsonl = os.path.join(out_dir, "pred_pcp_cd.jsonl")
    summary_json = os.path.join(out_dir, "summary.json")
    base.write_csv(route_rows_csv, route_rows)
    with open(pred_jsonl, "w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        summary_json,
        {
            "mode": "apply_pcp_c_d",
            "inputs": {
                "rows_csv": os.path.abspath(args.rows_csv),
                "policy_json": os.path.abspath(args.policy_json),
                "family": str(args.family),
                "derive_decision_kl": bool(args.derive_decision_kl),
            },
            "policy": {
                "selected_c_features": c_features,
                "selected_d_features": d_features,
                "applied_policy": policy,
            },
            "evaluation_from_cached_labels": evaluation,
            "outputs": {
                "pcp_route_rows_csv": route_rows_csv,
                "pred_jsonl": pred_jsonl,
            },
        },
    )
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
