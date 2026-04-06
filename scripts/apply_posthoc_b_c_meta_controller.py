#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import build_posthoc_b_c_fusion_controller as base
import build_posthoc_b_c_meta_controller as meta


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a calibrated post-hoc B/C meta-controller to held-out rows.")
    ap.add_argument("--scores_csv", type=str, required=True)
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--policy_bundle_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    rows = base.load_merged_rows(os.path.abspath(args.scores_csv), os.path.abspath(args.features_csv))
    with open(args.policy_bundle_json, "r", encoding="utf-8") as f:
        bundle: Dict[str, Any] = json.load(f)

    best_b_feature = bundle.get("best_b_feature")
    selected_c_features = bundle.get("selected_c_features") or []
    best_experts = bundle.get("best_experts") or {}
    best_meta_policy = bundle.get("best_meta_policy") or {}

    if not best_experts or not best_meta_policy:
        raise RuntimeError("policy bundle is missing best_experts or best_meta_policy")

    score_map = meta.build_score_maps(
        rows,
        best_b_feature,
        selected_c_features,
        best_experts.get("fusion"),
    )
    result = meta.evaluate_meta(
        rows,
        score_map,
        best_experts.get("b_only"),
        best_experts.get("c_only"),
        best_experts.get("fusion"),
        delta=float(best_meta_policy["delta"]),
        mode=str(best_meta_policy["mode"]),
    )
    route_rows = result.pop("route_rows")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    route_rows_csv = os.path.join(out_dir, "meta_route_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    base.write_csv(route_rows_csv, route_rows)
    write_json(
        summary_json,
        {
            "mode": "apply",
            "inputs": {
                "scores_csv": os.path.abspath(args.scores_csv),
                "features_csv": os.path.abspath(args.features_csv),
                "policy_bundle_json": os.path.abspath(args.policy_bundle_json),
            },
            "policy_bundle": {
                "best_b_feature": best_b_feature,
                "selected_c_features": selected_c_features,
                "best_experts": best_experts,
                "best_meta_policy": best_meta_policy,
            },
            "evaluation": result,
            "outputs": {
                "meta_route_rows_csv": os.path.abspath(route_rows_csv),
            },
        },
    )
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
