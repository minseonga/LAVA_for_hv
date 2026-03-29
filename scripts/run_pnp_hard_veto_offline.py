#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pnp_controller.adapters.eazy import EAZYOfflineAdapter
from pnp_controller.adapters.offline_csv import GenericOfflineCsvAdapter
from pnp_controller.adapters.vga import VGAOfflineAdapter
from pnp_controller.adapters.vista import VISTAOfflineAdapter
from pnp_controller.core.controller import run_offline_hard_veto
from pnp_controller.core.schemas import HardVetoConfig, OfflineTableSchema, ThresholdCalibrationConfig


def write_csv(path, rows):
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def build_adapter(args):
    if args.backend == "vga":
        return VGAOfflineAdapter(args.per_case_csv, args.features_csv)
    if args.backend == "vista":
        return VISTAOfflineAdapter(args.per_case_csv, args.features_csv)
    if args.backend == "eazy":
        return EAZYOfflineAdapter(args.per_case_csv, args.features_csv)
    schema = OfflineTableSchema(
        id_col_per_case=args.id_col_per_case,
        id_col_feature=args.id_col_feature,
        gt_col=args.gt_col,
        baseline_col=args.baseline_col,
        method_col=args.method_col,
        case_col=args.case_col,
    )
    return GenericOfflineCsvAdapter(args.per_case_csv, args.features_csv, schema=schema)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run backend-agnostic offline hard-veto controller.")
    ap.add_argument("--backend", type=str, default="generic", choices=["generic", "vga", "vista", "eazy"])
    ap.add_argument("--per_case_csv", type=str, required=True)
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--id_col_per_case", type=str, default="id")
    ap.add_argument("--id_col_feature", type=str, default="id")
    ap.add_argument("--gt_col", type=str, default="gt")
    ap.add_argument("--baseline_col", type=str, default="pred_baseline")
    ap.add_argument("--method_col", type=str, default="pred_vga")
    ap.add_argument("--case_col", type=str, default="case_type")

    ap.add_argument("--frg_col", type=str, default="faithful_minus_global_attn")
    ap.add_argument("--gmi_col", type=str, default="guidance_mismatch_score")
    ap.add_argument("--improvement_case_value", type=str, default="vga_improvement")
    ap.add_argument("--regression_case_value", type=str, default="vga_regression")
    ap.add_argument("--fallback_when_missing_feature", type=str, default="method", choices=["baseline", "method"])
    ap.add_argument("--tau_frg", type=float, default=None)
    ap.add_argument("--tau_gmi", type=float, default=None)
    ap.add_argument("--calib_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_improvement", type=float, default=1.0)
    ap.add_argument("--max_wrong_veto_rate", type=float, default=0.35)
    ap.add_argument("--q_grid", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    q_grid = [float(x.strip()) for x in str(args.q_grid).split(",") if x.strip()]

    controller_cfg = HardVetoConfig(
        frg_col=args.frg_col,
        gmi_col=args.gmi_col,
        improvement_case_value=args.improvement_case_value,
        regression_case_value=args.regression_case_value,
        fallback_when_missing_feature=args.fallback_when_missing_feature,
        tau_frg=args.tau_frg,
        tau_gmi=args.tau_gmi,
        calibration=ThresholdCalibrationConfig(
            calib_ratio=args.calib_ratio,
            seed=args.seed,
            lambda_improvement=args.lambda_improvement,
            max_wrong_veto_rate=args.max_wrong_veto_rate,
            q_grid=q_grid,
        ),
    )

    adapter = build_adapter(args)
    merged = adapter.load_merged(controller_cfg=controller_cfg)
    schema = adapter.schema
    df, summary = run_offline_hard_veto(merged_df=merged, schema=schema, controller_cfg=controller_cfg)

    per_id_cols = [
        schema.id_col_per_case,
        schema.gt_col,
        schema.baseline_col,
        schema.method_col,
        schema.case_col,
        "__FRG__",
        "__GMI__",
        "veto",
        "route",
        "pred_controller",
    ]
    per_id_cols = [c for c in per_id_cols if c in df.columns]
    per_id_csv = os.path.join(args.out_dir, "per_id_controller.csv")
    df[per_id_cols].to_csv(per_id_csv, index=False)

    summary["adapter"] = {
        "backend": args.backend,
        "adapter_class": adapter.__class__.__name__,
        "per_case_csv": os.path.abspath(args.per_case_csv),
        "features_csv": os.path.abspath(args.features_csv),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", per_id_csv)
    print("[saved]", summary_path)
    print(
        "[summary]",
        json.dumps(
            {
                "backend": args.backend,
                "base_acc": summary["metrics"]["baseline"]["acc"],
                "method_acc": summary["metrics"]["method"]["acc"],
                "controller_acc": summary["metrics"]["controller"]["acc"],
                "delta_vs_method": summary["metrics"]["controller"]["acc"] - summary["metrics"]["method"]["acc"],
                "veto_rate": summary["counts"]["veto_rate"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
