#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pnp_deploy.discriminative_meta import (  # noqa: E402
    MetaStrongController,
    compare_route_rows,
    merge_score_feature_rows,
    read_csv_rows,
    read_json,
    write_csv,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply the deployment-oriented discriminative meta_strong controller.")
    ap.add_argument("--scores_csv", type=str, required=True)
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--policy_bundle_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--reference_route_rows_csv", type=str, default="")
    args = ap.parse_args()

    rows = merge_score_feature_rows(os.path.abspath(args.scores_csv), os.path.abspath(args.features_csv))
    bundle: Dict[str, Any] = read_json(os.path.abspath(args.policy_bundle_json))
    controller = MetaStrongController.from_bundle(bundle)
    route_rows, evaluation = controller.evaluate(rows)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    route_csv = os.path.join(out_dir, "meta_route_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(route_csv, route_rows)

    parity = None
    if str(args.reference_route_rows_csv or "").strip():
        parity = compare_route_rows(route_rows, read_csv_rows(os.path.abspath(args.reference_route_rows_csv)))

    write_json(
        summary_json,
        {
            "mode": "deploy_apply",
            "inputs": {
                "scores_csv": os.path.abspath(args.scores_csv),
                "features_csv": os.path.abspath(args.features_csv),
                "policy_bundle_json": os.path.abspath(args.policy_bundle_json),
                "reference_route_rows_csv": (
                    os.path.abspath(args.reference_route_rows_csv)
                    if str(args.reference_route_rows_csv or "").strip()
                    else ""
                ),
            },
            "evaluation": evaluation,
            "parity": parity,
            "outputs": {
                "meta_route_rows_csv": route_csv,
            },
        },
    )
    print("[saved]", route_csv, flush=True)
    print("[saved]", summary_json, flush=True)
    if parity is not None:
        print("[parity]", parity, flush=True)


if __name__ == "__main__":
    main()

