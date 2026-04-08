#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: object) -> Optional[int]:
    v = maybe_float(value)
    if v is None:
        return None
    return int(round(v))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def quantiles(values: Sequence[float], probs: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {}
    out: Dict[str, float] = {}
    n = len(vals)
    for p in probs:
        pp = min(1.0, max(0.0, float(p)))
        pos = pp * float(n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            q = vals[lo]
        else:
            w = pos - float(lo)
            q = (1.0 - w) * vals[lo] + w * vals[hi]
        out[f"q{int(round(pp * 100)):02d}"] = float(q)
    return out


def top_k(rows: Sequence[Dict[str, Any]], key: str, k: int) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=True)[: int(k)]


def feasible(row: Dict[str, str], intervention: Dict[str, Any], mode: str, eps: float) -> bool:
    ci = float(maybe_float(row.get("mean_chair_i")) or 0.0)
    cs = float(maybe_float(row.get("mean_chair_s")) or 0.0)
    int_ci = float(intervention.get("mean_chair_i", 0.0))
    int_cs = float(intervention.get("mean_chair_s", 0.0))
    if mode == "none":
        return True
    if mode == "chairi":
        return ci <= int_ci + float(eps)
    if mode == "chairs":
        return cs <= int_cs + float(eps)
    if mode == "both":
        return ci <= int_ci + float(eps) and cs <= int_cs + float(eps)
    return False


def selection_tuple(row: Dict[str, str], objective: str) -> List[float]:
    f1 = float(maybe_float(row.get("mean_f1")) or 0.0)
    ci = float(maybe_float(row.get("mean_chair_i")) or 0.0)
    br = float(maybe_float(row.get("baseline_rate")) or 0.0)
    util = float(maybe_float(row.get("mean_claim_utility")) or 0.0)
    f1_minus_ci = float(maybe_float(row.get("mean_f1_minus_chairi")) or (f1 - ci))
    if objective == "f1_minus_chairi":
        return [f1_minus_ci, f1, -br]
    if objective == "neg_chairi":
        return [-ci, f1, -br]
    if objective == "claim_utility":
        return [util, f1, -br]
    return [f1, -ci, -br]


def summarize_discovery(tree_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(tree_dir, "summary.json")
    tau_csv = os.path.join(tree_dir, "tau_sweep.csv")
    feat_csv = os.path.join(tree_dir, "feature_metrics.csv")
    decision_csv = os.path.join(tree_dir, "decision_rows.csv")

    summary = json.load(open(summary_path, "r", encoding="utf-8"))
    intervention = dict(summary.get("intervention", {}))
    inputs = dict(summary.get("inputs", {}))
    constraint_mode = str(inputs.get("constraint_mode", "both"))
    chair_eps = float(inputs.get("chair_eps", 0.0))
    objective = str(inputs.get("selection_objective", "f1"))

    feat_rows = read_csv_rows(feat_csv) if os.path.exists(feat_csv) else []
    tau_rows = read_csv_rows(tau_csv) if os.path.exists(tau_csv) else []
    decision_rows = read_csv_rows(decision_csv) if os.path.exists(decision_csv) else []

    probs = [float(maybe_float(r.get("tree_fallback_prob")) or 0.0) for r in decision_rows]
    teacher = [int(maybe_int(r.get("teacher_fallback")) or 0) for r in decision_rows]
    routes = [str(r.get("route", "")) for r in decision_rows]

    route_counts: Dict[str, int] = {}
    teacher_route_counts: Dict[str, int] = {
        "teacher1_route_baseline": 0,
        "teacher1_route_method": 0,
        "teacher0_route_baseline": 0,
        "teacher0_route_method": 0,
    }
    for y, route in zip(teacher, routes):
        route_counts[route] = int(route_counts.get(route, 0)) + 1
        key = f"teacher{int(y)}_route_{route}"
        if key in teacher_route_counts:
            teacher_route_counts[key] += 1

    feasible_rows = [r for r in tau_rows if feasible(r, intervention, constraint_mode, chair_eps)]
    best_feasible = None
    if feasible_rows:
        best_feasible = max(feasible_rows, key=lambda r: tuple(selection_tuple(r, objective)))
    best_all = None
    if tau_rows:
        best_all = max(tau_rows, key=lambda r: tuple(selection_tuple(r, objective)))

    no_op_rows = [r for r in tau_rows if abs(float(maybe_float(r.get("baseline_rate")) or 0.0) - 0.0) <= 1e-12]
    full_baseline_rows = [r for r in tau_rows if abs(float(maybe_float(r.get("baseline_rate")) or 0.0) - 1.0) <= 1e-12]

    return {
        "summary_json": os.path.abspath(summary_path),
        "counts": {
            "n_feature_rows": len(feat_rows),
            "n_tau_rows": len(tau_rows),
            "n_decision_rows": len(decision_rows),
            "n_feasible_tau_rows": len(feasible_rows),
            "n_no_op_rows": len(no_op_rows),
            "n_all_baseline_rows": len(full_baseline_rows),
        },
        "teacher": {
            "teacher_positive_rate": mean(teacher),
            "teacher_positive_count": int(sum(teacher)),
            "teacher_negative_count": int(len(teacher) - sum(teacher)),
            "teacher_route_counts": teacher_route_counts,
        },
        "tree_prob_distribution": {
            "mean": mean(probs),
            "quantiles": quantiles(probs, [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
        },
        "route_counts": route_counts,
        "features_top_auroc": top_k(
            [
                {
                    "feature": r.get("feature", ""),
                    "direction": r.get("direction", ""),
                    "auroc": float(maybe_float(r.get("auroc")) or 0.0),
                    "average_precision": float(maybe_float(r.get("average_precision")) or 0.0),
                }
                for r in feat_rows
            ],
            "auroc",
            10,
        ),
        "tau_sweep": {
            "constraint_mode": constraint_mode,
            "chair_eps": chair_eps,
            "selection_objective": objective,
            "best_feasible": best_feasible,
            "best_unconstrained": best_all,
            "best_no_op": max(no_op_rows, key=lambda r: tuple(selection_tuple(r, objective))) if no_op_rows else None,
            "best_all_baseline": max(full_baseline_rows, key=lambda r: tuple(selection_tuple(r, objective))) if full_baseline_rows else None,
            "baseline_rate_distribution_feasible": quantiles(
                [float(maybe_float(r.get("baseline_rate")) or 0.0) for r in feasible_rows],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ),
            "f1_distribution_feasible": quantiles(
                [float(maybe_float(r.get("mean_f1")) or 0.0) for r in feasible_rows],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            ),
        },
    }


def summarize_apply(apply_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(apply_dir, "summary.json")
    decision_csv = os.path.join(apply_dir, "decision_rows.csv")
    summary = json.load(open(summary_path, "r", encoding="utf-8"))
    decision_rows = read_csv_rows(decision_csv) if os.path.exists(decision_csv) else []
    probs = [float(maybe_float(r.get("tree_fallback_prob")) or 0.0) for r in decision_rows]
    routes = [str(r.get("route", "")) for r in decision_rows]
    route_counts: Dict[str, int] = {}
    for route in routes:
        route_counts[route] = int(route_counts.get(route, 0)) + 1
    return {
        "summary_json": os.path.abspath(summary_path),
        "route_counts": route_counts,
        "tree_prob_distribution": {
            "mean": mean(probs),
            "quantiles": quantiles(probs, [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
        },
        "baseline": summary.get("baseline", {}),
        "intervention": summary.get("intervention", {}),
        "evaluation": summary.get("evaluation", {}),
        "policy": summary.get("policy", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize discovery/test distributions for generative tree runs.")
    ap.add_argument("--tree_controller_dir", type=str, default="")
    ap.add_argument("--tree_apply_dir", type=str, default="")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    result: Dict[str, Any] = {}
    if args.tree_controller_dir:
        result["discovery"] = summarize_discovery(os.path.abspath(args.tree_controller_dir))
    if args.tree_apply_dir:
        result["test_apply"] = summarize_apply(os.path.abspath(args.tree_apply_dir))

    if args.out_json:
        write_json(os.path.abspath(args.out_json), result)
        print("[saved]", os.path.abspath(args.out_json))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
