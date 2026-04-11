#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build_generative_b_c_meta_controller as base


def infer_numeric_probe_features(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    out: List[str] = []
    for feature in base.infer_probe_feature_cols(rows):
        s = str(feature)
        if not s.startswith("probe_"):
            continue
        out.append(s)
    return out


def add_distill_target(
    rows: Sequence[Dict[str, Any]],
    *,
    route_col: str,
    positive_route: str,
    route_source_col: str,
    positive_route_source: str,
    target_col: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pos = str(positive_route)
    pos_source = str(positive_route_source).strip()
    for row in rows:
        item = dict(row)
        is_pos = str(row.get(route_col, "")).strip() == pos
        if pos_source:
            is_pos = is_pos and (str(row.get(route_source_col, "")).strip() == pos_source)
        item[target_col] = 1 if is_pos else 0
        out.append(item)
    return out


def parse_quantiles(spec: str) -> List[float]:
    vals: List[float] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            vals.append(float(s))
        except Exception:
            continue
    if not vals:
        vals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    out = sorted({min(1.0, max(0.0, float(v))) for v in vals})
    return out


def compute_feature_stats(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Tuple[str, str]],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for feature, direction in feature_specs:
        oriented_vals: List[float] = []
        for row in rows:
            x = base.maybe_float(row.get(feature))
            if x is None:
                continue
            oriented_vals.append(float(x) if direction == "high" else -float(x))
        stats[str(feature)] = {
            "mean": float(base.mean(oriented_vals) if oriented_vals else 0.0),
            "std": float(base.std(oriented_vals) if oriented_vals else 1.0),
        }
    return stats


def build_scores(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Tuple[str, str]],
    feature_stats: Dict[str, Dict[str, float]],
) -> List[Optional[float]]:
    scores: List[Optional[float]] = []
    for row in rows:
        zvals: List[float] = []
        ok = True
        for feature, direction in feature_specs:
            x = base.maybe_float(row.get(feature))
            if x is None:
                ok = False
                break
            stats = dict(feature_stats.get(str(feature), {}))
            mu = float(stats.get("mean", 0.0))
            sd = float(stats.get("std", 1.0))
            if sd == 0.0:
                sd = 1.0
            oriented = float(x) if direction == "high" else -float(x)
            zvals.append(float((oriented - mu) / sd))
        scores.append(float(base.mean(zvals)) if ok and zvals else None)
    return scores


def target_stats(rows: Sequence[Dict[str, Any]], routes: Sequence[str], target_col: str) -> Dict[str, float]:
    n_selected = sum(1 for route in routes if route == "baseline")
    tp = sum(
        int(base.maybe_int(row.get(target_col)) or 0)
        for row, route in zip(rows, routes)
        if route == "baseline"
    )
    n_pos = sum(int(base.maybe_int(row.get(target_col)) or 0) for row in rows)
    return {
        "target_precision": base.safe_div(float(tp), float(max(1, n_selected))),
        "target_recall": base.safe_div(float(tp), float(max(1, n_pos))),
        "target_rate": base.safe_div(float(n_pos), float(max(1, len(rows)))),
        "target_match_rate": base.safe_div(
            float(sum(1 for row, route in zip(rows, routes) if (route == "baseline") == (int(base.maybe_int(row.get(target_col)) or 0) == 1))),
            float(max(1, len(rows))),
        ),
    }


def teacher_stats(rows: Sequence[Dict[str, Any]], routes: Sequence[str]) -> Dict[str, float]:
    n_selected = sum(1 for route in routes if route == "baseline")
    tp = sum(
        int(base.maybe_int(row.get("teacher_fallback")) or 0)
        for row, route in zip(rows, routes)
        if route == "baseline"
    )
    n_pos = sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows)
    return {
        "teacher_precision": base.safe_div(float(tp), float(max(1, n_selected))),
        "teacher_recall": base.safe_div(float(tp), float(max(1, n_pos))),
        "teacher_rate": base.safe_div(float(n_pos), float(max(1, len(rows)))),
    }


def compare_key(summary: Dict[str, Any], objective: str) -> Tuple[float, ...]:
    if str(objective) == "recall":
        primary = float(summary["mean_recall"])
    elif str(objective) == "recall_minus_chairi":
        primary = float(summary["mean_recall"] - summary["mean_chair_i"])
    elif str(objective) == "f1_minus_chairi":
        primary = float(summary["mean_f1_minus_chairi"])
    else:
        primary = float(summary["mean_f1"])
    tie_metric = float(summary["mean_recall"] if str(objective).startswith("recall") else summary["mean_f1"])
    return (
        primary,
        tie_metric,
        -float(summary["mean_chair_i"]),
        -float(summary["mean_chair_s"]),
        float(summary.get("target_precision") or 0.0),
        float(summary.get("target_recall") or 0.0),
        -float(summary["baseline_rate"]),
    )


def evaluate_combo(
    rows: Sequence[Dict[str, Any]],
    *,
    combo: Sequence[Tuple[str, str]],
    target_col: str,
    tau_quantiles: Sequence[float],
    min_baseline_rate: float,
    max_baseline_rate: float,
    min_f1_gain_vs_intervention: float,
    min_recall_gain_vs_intervention: Optional[float],
    max_chair_i_delta_vs_intervention: float,
    max_chair_s_delta_vs_intervention: float,
    selection_objective: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    intervention = base.aggregate_routes(rows, ["method"] * len(rows))
    feature_stats = compute_feature_stats(rows, combo)
    scores = build_scores(rows, combo, feature_stats)
    valid_scores = [float(s) for s in scores if s is not None and math.isfinite(float(s))]
    thresholds = base.quantiles_to_thresholds(valid_scores, tau_quantiles)

    best_summary: Optional[Dict[str, Any]] = None
    sweep_rows: List[Dict[str, Any]] = []

    for tau in thresholds:
        routes = [base.route_by_score(score, float(tau)) for score in scores]
        summary = base.aggregate_routes(rows, routes)
        summary.pop("decision_rows", None)
        summary.update(target_stats(rows, routes, target_col))
        summary.update(teacher_stats(rows, routes))
        summary["tau"] = float(tau)
        summary["features"] = " + ".join(feature for feature, _ in combo)
        summary["n_features"] = int(len(combo))
        summary["distill_target_col"] = str(target_col)
        summary["delta_f1_vs_int"] = float(summary["mean_f1"] - intervention["mean_f1"])
        summary["delta_recall_vs_int"] = float(summary["mean_recall"] - intervention["mean_recall"])
        summary["delta_chair_i_vs_int"] = float(summary["mean_chair_i"] - intervention["mean_chair_i"])
        summary["delta_chair_s_vs_int"] = float(summary["mean_chair_s"] - intervention["mean_chair_s"])
        if min_recall_gain_vs_intervention is None:
            gain_ok = float(summary["delta_f1_vs_int"]) > float(min_f1_gain_vs_intervention)
        else:
            gain_ok = float(summary["delta_recall_vs_int"]) > float(min_recall_gain_vs_intervention)
        summary["feasible"] = int(
            float(summary["baseline_rate"]) >= float(min_baseline_rate)
            and float(summary["baseline_rate"]) <= float(max_baseline_rate)
            and gain_ok
            and float(summary["delta_chair_i_vs_int"]) <= float(max_chair_i_delta_vs_intervention)
            and float(summary["delta_chair_s_vs_int"]) <= float(max_chair_s_delta_vs_intervention)
        )
        sweep_rows.append(dict(summary))
        if int(summary["feasible"]) != 1:
            continue
        if best_summary is None or compare_key(summary, selection_objective) > compare_key(best_summary, selection_objective):
            best_summary = dict(summary)

    return best_summary, sweep_rows, feature_stats


def combo_label(feature_specs: Sequence[Tuple[str, str]]) -> str:
    return "__".join(f"{feature}:{direction}" for feature, direction in feature_specs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill a pairwise generative route into an intervention-only cheap proxy.")
    ap.add_argument("--decision_rows_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--route_col", type=str, default="route")
    ap.add_argument("--positive_route", type=str, default="baseline")
    ap.add_argument("--route_source_col", type=str, default="route_source")
    ap.add_argument("--positive_route_source", type=str, default="")
    ap.add_argument("--target_col", type=str, default="distill_target")
    ap.add_argument("--feature_cols", type=str, default="auto")
    ap.add_argument("--feature_prefix", type=str, default="probe_")
    ap.add_argument("--top_n_features", type=int, default=8)
    ap.add_argument("--max_combo_size", type=int, default=2)
    ap.add_argument("--tau_quantiles", type=str, default="0.0,0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.0")
    ap.add_argument("--min_baseline_rate", type=float, default=0.02)
    ap.add_argument("--max_baseline_rate", type=float, default=0.08)
    ap.add_argument("--min_f1_gain_vs_intervention", type=float, default=0.0)
    ap.add_argument("--min_recall_gain_vs_intervention", type=float, default=None)
    ap.add_argument("--max_chair_i_delta_vs_intervention", type=float, default=0.0)
    ap.add_argument("--max_chair_s_delta_vs_intervention", type=float, default=0.0)
    ap.add_argument("--selection_objective", type=str, default="f1", choices=["f1", "f1_minus_chairi", "recall", "recall_minus_chairi"])
    args = ap.parse_args()

    raw_rows = base.read_csv_rows(os.path.abspath(args.decision_rows_csv))
    rows = add_distill_target(
        raw_rows,
        route_col=str(args.route_col),
        positive_route=str(args.positive_route),
        route_source_col=str(args.route_source_col),
        positive_route_source=str(args.positive_route_source),
        target_col=str(args.target_col),
    )
    target_spec = f"{args.target_col}:binary:0"

    if str(args.feature_cols).strip().lower() == "auto":
        feature_cols = infer_numeric_probe_features(rows)
    else:
        feature_cols = [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]
    if str(args.feature_prefix).strip():
        feature_cols = [col for col in feature_cols if str(col).startswith(str(args.feature_prefix).strip())]

    feature_metrics: List[Dict[str, Any]] = []
    directions: Dict[str, str] = {}
    quantiles = parse_quantiles(args.tau_quantiles)
    best_single_by_feature: Dict[str, Dict[str, Any]] = {}

    for feature in feature_cols:
        result = base.evaluate_feature(rows, feature, target_spec)
        if result is None:
            continue
        direction = str(result["direction"])
        directions[str(feature)] = direction
        combo = [(str(feature), direction)]
        best_single, _, _ = evaluate_combo(
            rows,
            combo=combo,
            target_col=str(args.target_col),
            tau_quantiles=quantiles,
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_f1_gain_vs_intervention=float(args.min_f1_gain_vs_intervention),
            min_recall_gain_vs_intervention=args.min_recall_gain_vs_intervention,
            max_chair_i_delta_vs_intervention=float(args.max_chair_i_delta_vs_intervention),
            max_chair_s_delta_vs_intervention=float(args.max_chair_s_delta_vs_intervention),
            selection_objective=str(args.selection_objective),
        )
        if best_single is not None:
            best_single_by_feature[str(feature)] = dict(best_single)
        out = {
            "feature": str(feature),
            "direction": direction,
            "auroc": float(result["auroc"]),
            "average_precision": float(result["average_precision"]) if result.get("average_precision") is not None else None,
            "n": int(result["n"]),
            "n_pos": int(result["n_pos"]),
            "has_feasible_single": int(best_single is not None),
        }
        if best_single is not None:
            for key in [
                "tau",
                "baseline_rate",
                "mean_chair_i",
                "mean_chair_s",
                "mean_recall",
                "mean_precision",
                "mean_f1",
                "mean_f1_minus_chairi",
                "mean_claim_utility",
                "delta_f1_vs_int",
                "delta_recall_vs_int",
                "delta_chair_i_vs_int",
                "delta_chair_s_vs_int",
                "target_precision",
                "target_recall",
                "target_match_rate",
                "teacher_precision",
                "teacher_recall",
            ]:
                out[f"single_{key}"] = best_single.get(key)
        feature_metrics.append(out)

    single_sort_key = "single_mean_recall" if str(args.selection_objective).startswith("recall") else "single_mean_f1"
    feature_metrics.sort(
        key=lambda r: (
            -int(r.get("has_feasible_single") or 0),
            -float(r.get(single_sort_key) or -1e9),
            -float(r.get("auroc") or 0.0),
            -float(r.get("average_precision") or 0.0),
            str(r.get("feature") or ""),
        )
    )

    top_features = [str(row["feature"]) for row in feature_metrics[: max(1, int(args.top_n_features))]]
    candidate_combos: List[List[Tuple[str, str]]] = []
    max_combo_size = max(1, int(args.max_combo_size))
    for size in range(1, max_combo_size + 1):
        for combo_features in itertools.combinations(top_features, size):
            specs = [(feature, directions[str(feature)]) for feature in combo_features]
            candidate_combos.append(specs)

    combo_results: List[Dict[str, Any]] = []
    best_combo_summary: Optional[Dict[str, Any]] = None
    best_combo_specs: List[Tuple[str, str]] = []
    best_combo_stats: Dict[str, Dict[str, float]] = {}
    best_combo_sweep: List[Dict[str, Any]] = []

    for combo in candidate_combos:
        best_summary, sweep_rows, feature_stats = evaluate_combo(
            rows,
            combo=combo,
            target_col=str(args.target_col),
            tau_quantiles=quantiles,
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_f1_gain_vs_intervention=float(args.min_f1_gain_vs_intervention),
            min_recall_gain_vs_intervention=args.min_recall_gain_vs_intervention,
            max_chair_i_delta_vs_intervention=float(args.max_chair_i_delta_vs_intervention),
            max_chair_s_delta_vs_intervention=float(args.max_chair_s_delta_vs_intervention),
            selection_objective=str(args.selection_objective),
        )
        row: Dict[str, Any] = {
            "features": " + ".join(feature for feature, _ in combo),
            "combo_label": combo_label(combo),
            "n_features": int(len(combo)),
            "feature_directions": json.dumps([{ "feature": feature, "direction": direction } for feature, direction in combo], ensure_ascii=False),
            "has_feasible_policy": int(best_summary is not None),
        }
        if best_summary is not None:
            row.update(best_summary)
            if best_combo_summary is None or compare_key(best_summary, str(args.selection_objective)) > compare_key(best_combo_summary, str(args.selection_objective)):
                best_combo_summary = dict(best_summary)
                best_combo_specs = list(combo)
                best_combo_stats = dict(feature_stats)
                best_combo_sweep = list(sweep_rows)
        combo_results.append(row)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    feature_metrics_csv = os.path.join(out_dir, "feature_metrics.csv")
    combo_results_csv = os.path.join(out_dir, "combo_results.csv")
    selected_feature_metrics_csv = os.path.join(out_dir, "feature_metrics_selected.csv")
    tau_sweep_csv = os.path.join(out_dir, "tau_sweep.csv")
    selected_policy_json = os.path.join(out_dir, "selected_policy.json")
    decision_rows_csv = os.path.join(out_dir, "decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    base.write_csv(feature_metrics_csv, feature_metrics)
    base.write_csv(combo_results_csv, combo_results)

    if best_combo_summary is None:
        base.write_json(
            summary_json,
            {
                "inputs": {
                    "decision_rows_csv": os.path.abspath(args.decision_rows_csv),
                "route_col": str(args.route_col),
                "positive_route": str(args.positive_route),
                "route_source_col": str(args.route_source_col),
                "positive_route_source": str(args.positive_route_source),
                "target_col": str(args.target_col),
                    "feature_prefix": str(args.feature_prefix),
                    "top_n_features": int(args.top_n_features),
                    "max_combo_size": int(args.max_combo_size),
                    "selection_objective": str(args.selection_objective),
                },
                "counts": {
                    "n_rows": int(len(rows)),
                    "target_rate": base.safe_div(
                        float(sum(int(base.maybe_int(row.get(args.target_col)) or 0) for row in rows)),
                        float(max(1, len(rows))),
                    ),
                },
                "baseline": {k: v for k, v in base.aggregate_routes(rows, ["baseline"] * len(rows)).items() if k != "decision_rows"},
                "intervention": {k: v for k, v in base.aggregate_routes(rows, ["method"] * len(rows)).items() if k != "decision_rows"},
                "outputs": {
                    "feature_metrics_csv": feature_metrics_csv,
                    "combo_results_csv": combo_results_csv,
                },
                "error": "No feasible distill proxy policy found.",
            },
        )
        raise RuntimeError("No feasible distill proxy policy found.")

    selected_features = [str(feature) for feature, _ in best_combo_specs]
    selected_rows = [row for row in feature_metrics if str(row.get("feature")) in set(selected_features)]
    base.write_csv(selected_feature_metrics_csv, selected_rows)
    base.write_csv(tau_sweep_csv, best_combo_sweep)

    policy = {
        "policy_type": "generative_route_distill_proxy_v1",
        "source_decision_rows_csv": os.path.abspath(args.decision_rows_csv),
        "route_col": str(args.route_col),
        "positive_route": str(args.positive_route),
        "route_source_col": str(args.route_source_col),
        "positive_route_source": str(args.positive_route_source),
        "target_col": str(args.target_col),
        "feature_specs": [
            {"feature": str(feature), "direction": str(direction), "weight": 1.0}
            for feature, direction in best_combo_specs
        ],
        "feature_stats": best_combo_stats,
        "tau": float(best_combo_summary["tau"]),
    }
    base.write_json(selected_policy_json, policy)

    scores = build_scores(rows, best_combo_specs, best_combo_stats)
    routes = [base.route_by_score(score, float(best_combo_summary["tau"])) for score in scores]
    decision_rows: List[Dict[str, Any]] = []
    for row, score, route in zip(rows, scores, routes):
        out = dict(row)
        out["distill_proxy_score"] = score
        out["route"] = route
        out["distill_target"] = int(base.maybe_int(row.get(str(args.target_col))) or 0)
        out["target_match"] = int((route == "baseline") == (int(base.maybe_int(row.get(str(args.target_col))) or 0) == 1))
        decision_rows.append(out)
    base.write_csv(decision_rows_csv, decision_rows)

    summary = {
        "inputs": {
            "decision_rows_csv": os.path.abspath(args.decision_rows_csv),
            "route_col": str(args.route_col),
            "positive_route": str(args.positive_route),
            "route_source_col": str(args.route_source_col),
            "positive_route_source": str(args.positive_route_source),
            "target_col": str(args.target_col),
            "feature_prefix": str(args.feature_prefix),
            "top_n_features": int(args.top_n_features),
            "max_combo_size": int(args.max_combo_size),
            "tau_quantiles": quantiles,
            "min_baseline_rate": float(args.min_baseline_rate),
            "max_baseline_rate": float(args.max_baseline_rate),
            "min_f1_gain_vs_intervention": float(args.min_f1_gain_vs_intervention),
            "min_recall_gain_vs_intervention": args.min_recall_gain_vs_intervention,
            "max_chair_i_delta_vs_intervention": float(args.max_chair_i_delta_vs_intervention),
            "max_chair_s_delta_vs_intervention": float(args.max_chair_s_delta_vs_intervention),
            "selection_objective": str(args.selection_objective),
        },
        "counts": {
            "n_rows": int(len(rows)),
            "target_rate": base.safe_div(
                float(sum(int(base.maybe_int(row.get(str(args.target_col))) or 0) for row in rows)),
                float(max(1, len(rows))),
            ),
            "n_probe_features": int(len(feature_metrics)),
        },
        "baseline": {k: v for k, v in base.aggregate_routes(rows, ["baseline"] * len(rows)).items() if k != "decision_rows"},
        "intervention": {k: v for k, v in base.aggregate_routes(rows, ["method"] * len(rows)).items() if k != "decision_rows"},
        "best_policy": {k: v for k, v in best_combo_summary.items() if k != "decision_rows"},
        "outputs": {
            "feature_metrics_csv": feature_metrics_csv,
            "combo_results_csv": combo_results_csv,
            "feature_metrics_selected_csv": selected_feature_metrics_csv,
            "tau_sweep_csv": tau_sweep_csv,
            "selected_policy_json": selected_policy_json,
            "decision_rows_csv": decision_rows_csv,
        },
    }
    base.write_json(summary_json, summary)
    print("[saved]", feature_metrics_csv)
    print("[saved]", combo_results_csv)
    print("[saved]", selected_feature_metrics_csv)
    print("[saved]", tau_sweep_csv)
    print("[saved]", selected_policy_json)
    print("[saved]", decision_rows_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
