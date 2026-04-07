#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs: List[float]) -> float:
    return sum(xs) / float(len(xs)) if xs else 0.0


def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / float(len(xs)))


def summarize_variant(variant_dir: str) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(variant_dir)):
        if not (name.startswith("metrics_seed") and name.endswith(".json")):
            continue
        seed = name[len("metrics_seed") : -len(".json")]
        metrics_path = os.path.join(variant_dir, name)
        compare_path = os.path.join(variant_dir, f"compare_seed{seed}.json")
        metrics_obj = read_json(metrics_path)
        metrics = dict(metrics_obj.get("metrics", {}))
        row: Dict[str, Any] = {
            "seed": int(seed),
            "metrics": metrics,
        }
        if os.path.isfile(compare_path):
            cmp_obj = read_json(compare_path)
            row["delta_acc_vs_baseline"] = float(
                cmp_obj.get("overall", {}).get("delta", {}).get("acc", 0.0)
            )
            row["change_counts"] = dict(cmp_obj.get("change_counts", {}))
        runs.append(row)

    accs = [float(r.get("metrics", {}).get("acc", 0.0)) for r in runs]
    f1s = [float(r.get("metrics", {}).get("f1", 0.0)) for r in runs]
    precs = [float(r.get("metrics", {}).get("precision", 0.0)) for r in runs]
    recs = [float(r.get("metrics", {}).get("recall", 0.0)) for r in runs]
    yes_ratios = [float(r.get("metrics", {}).get("yes_ratio", 0.0)) for r in runs]
    deltas = [float(r.get("delta_acc_vs_baseline", 0.0)) for r in runs if "delta_acc_vs_baseline" in r]
    harms = [
        float(r.get("change_counts", {}).get("harm", 0.0))
        for r in runs
        if "change_counts" in r
    ]
    gains = [
        float(r.get("change_counts", {}).get("gain", 0.0))
        for r in runs
        if "change_counts" in r
    ]
    nets = [
        float(r.get("change_counts", {}).get("net_gain", 0.0))
        for r in runs
        if "change_counts" in r
    ]

    return {
        "n_runs": len(runs),
        "runs": runs,
        "aggregate": {
            "acc_mean": mean(accs),
            "acc_std": std(accs),
            "f1_mean": mean(f1s),
            "f1_std": std(f1s),
            "precision_mean": mean(precs),
            "recall_mean": mean(recs),
            "yes_ratio_mean": mean(yes_ratios),
            "yes_ratio_std": std(yes_ratios),
            "delta_acc_vs_baseline_mean": mean(deltas),
            "delta_acc_vs_baseline_std": std(deltas),
            "harm_mean": mean(harms),
            "gain_mean": mean(gains),
            "net_gain_mean": mean(nets),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize VCD wrapper stability audit outputs.")
    ap.add_argument("--audit_dir", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    audit_dir = os.path.abspath(args.audit_dir)
    runs_root = os.path.join(audit_dir, "runs")
    baseline_metrics_path = os.path.join(audit_dir, "baseline_subset_metrics.json")
    subset_summary_path = os.path.join(audit_dir, "subset", "summary.json")

    baseline_metrics = read_json(baseline_metrics_path) if os.path.isfile(baseline_metrics_path) else {}
    subset_summary = read_json(subset_summary_path) if os.path.isfile(subset_summary_path) else {}

    variants: Dict[str, Any] = {}
    if os.path.isdir(runs_root):
        for variant in sorted(os.listdir(runs_root)):
            variant_dir = os.path.join(runs_root, variant)
            if os.path.isdir(variant_dir):
                variants[variant] = summarize_variant(variant_dir)

    base_acc = float(baseline_metrics.get("metrics", {}).get("acc", 0.0))
    base_yes = float(baseline_metrics.get("metrics", {}).get("yes_ratio", 0.0))
    flags: List[str] = []

    cd_sample = variants.get("cd_sample", {}).get("aggregate", {})
    cd_greedy = variants.get("cd_greedy", {}).get("aggregate", {})
    nocd_sample = variants.get("nocd_sample", {}).get("aggregate", {})
    nocd_greedy = variants.get("nocd_greedy", {}).get("aggregate", {})

    if cd_sample:
        if float(cd_sample.get("acc_mean", 0.0)) + 1e-9 < base_acc:
            flags.append("cd_sample_below_baseline")
        if abs(float(cd_sample.get("yes_ratio_mean", 0.0)) - base_yes) >= 0.05:
            flags.append("cd_sample_large_yes_ratio_shift_vs_baseline")
        if float(cd_sample.get("acc_std", 0.0)) >= 0.005:
            flags.append("cd_sample_seed_instability_high")
    if nocd_sample and cd_sample:
        if float(nocd_sample.get("acc_mean", 0.0)) - float(cd_sample.get("acc_mean", 0.0)) >= 0.01:
            flags.append("no_cd_sample_outperforms_cd_sample")
    if cd_greedy and cd_sample:
        if float(cd_greedy.get("acc_mean", 0.0)) - float(cd_sample.get("acc_mean", 0.0)) >= 0.01:
            flags.append("cd_greedy_outperforms_cd_sample")
    if nocd_greedy and nocd_sample:
        if float(nocd_greedy.get("acc_mean", 0.0)) - float(nocd_sample.get("acc_mean", 0.0)) >= 0.01:
            flags.append("no_cd_greedy_outperforms_no_cd_sample")

    out = {
        "inputs": {
            "audit_dir": audit_dir,
        },
        "subset": subset_summary,
        "baseline_subset_metrics": baseline_metrics,
        "variants": variants,
        "flags": flags,
    }

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[saved]", out_path)


if __name__ == "__main__":
    main()
