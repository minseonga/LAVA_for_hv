#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def feature_names(items: List[Dict[str, Any]]) -> List[str]:
    return [str(x.get("feature", "")) for x in items]


def metric_subset(summary: Dict[str, Any]) -> Dict[str, Any]:
    best = dict(summary.get("best_tau", {}))
    keys = [
        "lambda_harm",
        "tau",
        "method_rate",
        "baseline_rate",
        "final_acc",
        "baseline_acc",
        "intervention_acc",
        "delta_vs_baseline",
        "delta_vs_intervention",
        "source_balanced_utility",
        "worst_source_utility",
        "selected_count",
        "applied_help",
        "applied_harm",
        "applied_neutral",
        "applied_help_precision",
        "applied_harm_precision",
        "applied_help_recall",
        "applied_harm_recall",
        "veto_count",
        "veto_harm",
        "veto_help",
        "veto_neutral",
        "veto_harm_precision",
        "veto_help_precision",
        "veto_harm_recall",
    ]
    return {k: best.get(k) for k in keys if k in best}


def numeric_diffs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, av in a.items():
        bv = b.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[key] = float(av) - float(bv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare semantic v3 discovery summaries between mix and POPE-only runs.")
    ap.add_argument("--mix_summary_json", type=str, required=True)
    ap.add_argument("--pope_only_summary_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    mix = read_json(args.mix_summary_json)
    pope = read_json(args.pope_only_summary_json)

    mix_help = feature_names(list(mix.get("selected_help_features", [])))
    mix_harm = feature_names(list(mix.get("selected_harm_features", [])))
    pope_help = feature_names(list(pope.get("selected_help_features", [])))
    pope_harm = feature_names(list(pope.get("selected_harm_features", [])))

    mix_metrics = metric_subset(mix)
    pope_metrics = metric_subset(pope)

    out = {
        "inputs": {
            "mix_summary_json": os.path.abspath(args.mix_summary_json),
            "pope_only_summary_json": os.path.abspath(args.pope_only_summary_json),
        },
        "feature_overlap": {
            "help_mix": mix_help,
            "help_pope_only": pope_help,
            "help_intersection": sorted(set(mix_help) & set(pope_help)),
            "harm_mix": mix_harm,
            "harm_pope_only": pope_harm,
            "harm_intersection": sorted(set(mix_harm) & set(pope_harm)),
        },
        "best_tau_metrics": {
            "mix": mix_metrics,
            "pope_only": pope_metrics,
            "mix_minus_pope_only": numeric_diffs(mix_metrics, pope_metrics),
        },
    }
    write_json(args.out_json, out)
    print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
