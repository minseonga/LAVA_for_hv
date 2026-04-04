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


def discovery_core(summary: Dict[str, Any]) -> Dict[str, Any]:
    best = dict(summary.get("best_tau", {}))
    keys = [
        "lambda_harm",
        "tau",
        "method_rate",
        "baseline_rate",
        "final_acc",
        "baseline_acc",
        "intervention_acc",
        "oracle_posthoc_acc",
        "delta_vs_baseline",
        "delta_vs_intervention",
        "gap_to_oracle_posthoc",
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
        "harm_vs_help_auroc",
        "help_vs_harm_auroc",
        "veto_count",
        "veto_harm",
        "veto_help",
        "veto_neutral",
        "veto_harm_precision",
        "veto_help_precision",
        "veto_harm_recall",
    ]
    return {
        "sources": summary.get("sources"),
        "selected_help_features": feature_names(list(summary.get("selected_help_features", []))),
        "selected_harm_features": feature_names(list(summary.get("selected_harm_features", []))),
        "best_tau": {k: best.get(k) for k in keys if k in best},
    }


def apply_core(summary: Dict[str, Any]) -> Dict[str, Any]:
    ev = dict(summary.get("evaluation", {}))
    keys = [
        "n_eval",
        "method_rate",
        "baseline_rate",
        "baseline_acc",
        "intervention_acc",
        "oracle_posthoc_acc",
        "pregate_acc",
        "delta_vs_baseline",
        "delta_vs_intervention",
        "gap_to_oracle_posthoc",
        "applied_harm",
        "applied_help",
        "applied_neutral",
        "applied_help_precision",
        "applied_harm_precision",
        "veto_count",
        "veto_harm",
        "veto_help",
        "veto_neutral",
        "veto_harm_precision",
        "veto_help_precision",
        "veto_harm_recall",
    ]
    return {k: ev.get(k) for k in keys if k in ev}


def comparison_core(summary: Dict[str, Any]) -> Dict[str, Any]:
    overlap = dict(summary.get("feature_overlap", {}))
    metrics = dict(summary.get("best_tau_metrics", {}))
    return {
        "feature_overlap": overlap,
        "best_tau_metrics": metrics,
    }


def classify_summary(summary: Dict[str, Any]) -> str:
    if "best_tau" in summary:
        return "discovery_controller"
    if "evaluation" in summary:
        return "heldout_apply"
    if "best_tau_metrics" in summary:
        return "analysis_compare"
    return "unknown"


def compact_entry(root: str, path: str) -> Dict[str, Any]:
    summary = read_json(path)
    kind = classify_summary(summary)
    relpath = os.path.relpath(path, root)
    if kind == "discovery_controller":
        core = discovery_core(summary)
    elif kind == "heldout_apply":
        core = apply_core(summary)
    elif kind == "analysis_compare":
        core = comparison_core(summary)
    else:
        core = {"keys": sorted(summary.keys())}
    return {
        "path": relpath,
        "kind": kind,
        "core": core,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect compact key metrics from v3 pregate summary.json files.")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_json = os.path.abspath(args.out_json) if args.out_json else os.path.join(root, "summary_compact.json")

    entries: List[Dict[str, Any]] = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name != "summary.json":
                continue
            path = os.path.abspath(os.path.join(dirpath, name))
            if path == out_json:
                continue
            entries.append(compact_entry(root, path))
    entries.sort(key=lambda x: str(x["path"]))

    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        grouped[str(entry["path"])] = {
            "kind": entry["kind"],
            "core": entry["core"],
        }

    out = {
        "root": root,
        "n_summaries": len(entries),
        "entries": entries,
        "grouped": grouped,
    }
    write_json(out_json, out)
    print("[saved]", out_json)


if __name__ == "__main__":
    main()
