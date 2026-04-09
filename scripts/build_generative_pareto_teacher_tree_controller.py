#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build_generative_b_c_meta_controller as base
import build_generative_pareto_teacher_controller as teacher


def parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


def parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except Exception:
            continue
    return out


def gini(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    p = float(sum(int(y) for y in labels)) / float(len(labels))
    return float(1.0 - p * p - (1.0 - p) * (1.0 - p))


def quantiles_to_thresholds(values: Sequence[float], quantiles: Sequence[float]) -> List[float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return []
    if len(vals) == 1:
        v = float(vals[0])
        return [v - 1e-6, v, v + 1e-6]
    out = {float(vals[0]) - 1e-6, float(vals[-1]) + 1e-6}
    n = len(vals)
    for q in quantiles:
        qq = min(1.0, max(0.0, float(q)))
        pos = qq * float(n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.add(vals[lo])
        else:
            w = pos - float(lo)
            out.add((1.0 - w) * vals[lo] + w * vals[hi])
    return sorted(out)


def compute_feature_means(rows: Sequence[Dict[str, Any]], feature_names: Sequence[str]) -> Dict[str, float]:
    means: Dict[str, float] = {}
    for feat in feature_names:
        vals = [float(base.maybe_float(row.get(feat))) for row in rows if base.maybe_float(row.get(feat)) is not None]
        means[feat] = base.mean(vals)
    return means


def imputed_value(row: Dict[str, Any], feature: str, means: Dict[str, float]) -> float:
    v = base.maybe_float(row.get(feature))
    if v is None:
        return float(means.get(feature, 0.0))
    return float(v)


def candidate_thresholds(
    rows: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    feature: str,
    means: Dict[str, float],
    threshold_quantiles: Sequence[float],
) -> List[float]:
    vals = [imputed_value(rows[i], feature, means) for i in indices]
    return quantiles_to_thresholds(vals, threshold_quantiles)


def best_split(
    rows: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    feature_names: Sequence[str],
    labels: Sequence[int],
    means: Dict[str, float],
    *,
    min_leaf: int,
    threshold_quantiles: Sequence[float],
) -> Optional[Dict[str, Any]]:
    parent_labels = [int(labels[i]) for i in indices]
    parent_impurity = gini(parent_labels)
    if parent_impurity <= 1e-12:
        return None
    best: Optional[Dict[str, Any]] = None
    n = len(indices)
    for feat in feature_names:
        for tau in candidate_thresholds(rows, indices, feat, means, threshold_quantiles):
            left = [i for i in indices if imputed_value(rows[i], feat, means) <= float(tau)]
            right = [i for i in indices if imputed_value(rows[i], feat, means) > float(tau)]
            if len(left) < int(min_leaf) or len(right) < int(min_leaf):
                continue
            left_imp = gini([int(labels[i]) for i in left])
            right_imp = gini([int(labels[i]) for i in right])
            gain = parent_impurity - (
                (float(len(left)) / float(n)) * left_imp + (float(len(right)) / float(n)) * right_imp
            )
            cand = {
                "feature": feat,
                "tau": float(tau),
                "gain": float(gain),
                "left": left,
                "right": right,
            }
            if best is None or float(cand["gain"]) > float(best["gain"]):
                best = cand
    if best is None or float(best["gain"]) <= 1e-9:
        return None
    return best


def build_tree(
    rows: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    feature_names: Sequence[str],
    labels: Sequence[int],
    means: Dict[str, float],
    *,
    depth: int,
    max_depth: int,
    min_leaf: int,
    threshold_quantiles: Sequence[float],
) -> Dict[str, Any]:
    node_labels = [int(labels[i]) for i in indices]
    pos_rate = float(sum(node_labels)) / float(max(1, len(node_labels)))
    node: Dict[str, Any] = {
        "is_leaf": True,
        "depth": int(depth),
        "n": int(len(indices)),
        "pos_rate": float(pos_rate),
    }
    if depth >= int(max_depth) or len(indices) < int(2 * min_leaf):
        return node
    split = best_split(
        rows,
        indices,
        feature_names,
        labels,
        means,
        min_leaf=int(min_leaf),
        threshold_quantiles=threshold_quantiles,
    )
    if split is None:
        return node
    node.update(
        {
            "is_leaf": False,
            "feature": str(split["feature"]),
            "tau": float(split["tau"]),
            "gain": float(split["gain"]),
            "left": build_tree(
                rows,
                split["left"],
                feature_names,
                labels,
                means,
                depth=depth + 1,
                max_depth=int(max_depth),
                min_leaf=int(min_leaf),
                threshold_quantiles=threshold_quantiles,
            ),
            "right": build_tree(
                rows,
                split["right"],
                feature_names,
                labels,
                means,
                depth=depth + 1,
                max_depth=int(max_depth),
                min_leaf=int(min_leaf),
                threshold_quantiles=threshold_quantiles,
            ),
        }
    )
    return node


def predict_tree_row(row: Dict[str, Any], tree: Dict[str, Any], means: Dict[str, float]) -> float:
    node = tree
    while not bool(node.get("is_leaf", True)):
        feat = str(node["feature"])
        tau = float(node["tau"])
        x = imputed_value(row, feat, means)
        node = node["left"] if x <= tau else node["right"]
    return float(node.get("pos_rate", 0.0))


def collect_feature_usage(tree: Dict[str, Any], counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    if counts is None:
        counts = {}
    if bool(tree.get("is_leaf", True)):
        return counts
    feat = str(tree.get("feature"))
    counts[feat] = int(counts.get(feat, 0)) + 1
    collect_feature_usage(tree["left"], counts)
    collect_feature_usage(tree["right"], counts)
    return counts


def feasible(summary: Dict[str, Any], intervention: Dict[str, Any], mode: str, chair_eps: float) -> bool:
    return teacher.feasible_under_constraints(summary, intervention, mode, chair_eps)


def selection_key(row: Dict[str, Any], objective: str) -> Tuple[float, float, float]:
    return teacher.selection_key(row, objective)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a shallow non-linear tree controller against a Pareto-teacher target.")
    ap.add_argument("--claim_table_csv", type=str, required=True)
    ap.add_argument("--chair_table_csv", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--teacher_mode", type=str, default="strict_pareto", choices=["strict_pareto", "chairi_pareto", "f1_only"])
    ap.add_argument("--min_f1_gain", type=float, default=0.0)
    ap.add_argument("--feature_cols", type=str, default="auto")
    ap.add_argument("--feature_cols_file", type=str, default="")
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_n_features", type=int, default=8)
    ap.add_argument("--feature_family_mode", type=str, default="overall", choices=["overall", "balanced", "probe_only", "pair_only"])
    ap.add_argument("--top_n_probe_features", type=int, default=8)
    ap.add_argument("--top_n_pair_features", type=int, default=8)
    ap.add_argument("--max_depth_values", type=str, default="1,2,3")
    ap.add_argument("--min_leaf_values", type=str, default="3,5,8,10")
    ap.add_argument("--split_quantiles", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--tau_quantiles", type=str, default="0.0,0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.0")
    ap.add_argument("--constraint_mode", type=str, default="both", choices=["none", "chairi", "chairs", "both"])
    ap.add_argument("--chair_eps", type=float, default=0.0)
    ap.add_argument("--selection_objective", type=str, default="f1", choices=["f1", "f1_minus_chairi", "neg_chairi", "claim_utility"])
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    args = ap.parse_args()

    claim_rows = base.read_csv_rows(os.path.abspath(args.claim_table_csv))
    chair_rows = base.read_csv_rows(os.path.abspath(args.chair_table_csv))
    rows = base.build_master_rows(
        claim_rows,
        chair_rows,
        os.path.abspath(args.baseline_chair_json),
        os.path.abspath(args.intervention_chair_json),
    )
    rows = teacher.attach_teacher_labels(rows, str(args.teacher_mode), float(args.min_f1_gain))

    feature_cols = teacher.resolve_feature_cols(rows, str(args.feature_cols), str(args.feature_cols_file))
    feature_metrics_all: List[Dict[str, Any]] = []
    for feat in feature_cols:
        res = teacher.evaluate_feature(rows, feat)
        if res is None:
            continue
        feature_metrics_all.append(res)
    feature_metrics_all = teacher.sort_feature_metrics(feature_metrics_all)
    feature_metrics = teacher.select_feature_metrics(
        feature_metrics_all,
        min_feature_auroc=float(args.min_feature_auroc),
        top_n_features=int(args.top_n_features),
        feature_family_mode=str(args.feature_family_mode),
        top_n_probe_features=int(args.top_n_probe_features),
        top_n_pair_features=int(args.top_n_pair_features),
    )
    if not feature_metrics:
        raise RuntimeError("No feasible features for tree controller.")

    selected = feature_metrics[: max(1, int(args.top_n_features))]
    feature_names = [str(r["feature"]) for r in selected]
    means = compute_feature_means(rows, feature_names)
    labels = [int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows]
    intervention_summary = base.aggregate_routes(rows, ["method"] * len(rows))
    baseline_summary = base.aggregate_routes(rows, ["baseline"] * len(rows))
    teacher_routes = ["baseline" if int(y) == 1 else "method" for y in labels]
    teacher_summary = teacher.route_summary(rows, teacher_routes)
    teacher_summary["teacher_rate"] = base.safe_div(float(sum(labels)), float(max(1, len(labels))))

    max_depth_values = parse_int_list(args.max_depth_values)
    min_leaf_values = parse_int_list(args.min_leaf_values)
    split_quantiles = parse_float_list(args.split_quantiles)
    tau_quantiles = parse_float_list(args.tau_quantiles)

    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_tree: Optional[Dict[str, Any]] = None
    best_probs: List[float] = []
    best_routes: List[str] = []

    for max_depth in max_depth_values:
        for min_leaf in min_leaf_values:
            tree = build_tree(
                rows,
                list(range(len(rows))),
                feature_names,
                labels,
                means,
                depth=0,
                max_depth=int(max_depth),
                min_leaf=int(min_leaf),
                threshold_quantiles=split_quantiles,
            )
            probs = [predict_tree_row(row, tree, means) for row in rows]
            fit_auc = base.binary_auroc(probs, labels)
            fit_ap = base.binary_average_precision(probs, labels)
            tau_grid = quantiles_to_thresholds(probs, tau_quantiles)
            for tau in tau_grid:
                routes = [base.route_by_score(float(prob), float(tau)) for prob in probs]
                summary = teacher.route_summary(rows, routes)
                if float(summary["baseline_rate"]) < float(args.min_baseline_rate):
                    continue
                if float(summary["baseline_rate"]) > float(args.max_baseline_rate):
                    continue
                if not feasible(summary, intervention_summary, str(args.constraint_mode), float(args.chair_eps)):
                    continue
                row = {
                    "max_depth": int(max_depth),
                    "min_leaf": int(min_leaf),
                    "tau": float(tau),
                    "teacher_auroc": None if fit_auc is None else float(fit_auc),
                    "teacher_ap": None if fit_ap is None else float(fit_ap),
                    **{k: v for k, v in summary.items() if k != "decision_rows"},
                }
                sweep_rows.append(row)
                if best is None or selection_key(row, str(args.selection_objective)) > selection_key(best, str(args.selection_objective)):
                    best = dict(row)
                    best_tree = tree
                    best_probs = probs
                    best_routes = routes

    if best is None or best_tree is None:
        raise RuntimeError("No feasible tree controller satisfied the requested constraints.")

    decision_rows: List[Dict[str, Any]] = []
    for row, prob, route in zip(rows, best_probs, best_routes):
        out = dict(row)
        out["tree_fallback_prob"] = float(prob)
        out["route"] = route
        decision_rows.append(out)

    os.makedirs(args.out_dir, exist_ok=True)
    base.write_csv(os.path.join(args.out_dir, "feature_metrics.csv"), feature_metrics_all)
    base.write_csv(os.path.join(args.out_dir, "feature_metrics_selected.csv"), feature_metrics)
    base.write_csv(os.path.join(args.out_dir, "tau_sweep.csv"), sweep_rows)
    base.write_csv(os.path.join(args.out_dir, "decision_rows.csv"), decision_rows)
    base.write_json(
        os.path.join(args.out_dir, "selected_tree.json"),
        {
            "policy_type": "generative_pareto_teacher_tree_v1",
            "teacher_mode": str(args.teacher_mode),
            "min_f1_gain": float(args.min_f1_gain),
            "constraint_mode": str(args.constraint_mode),
            "chair_eps": float(args.chair_eps),
            "selection_objective": str(args.selection_objective),
            "feature_family_mode": str(args.feature_family_mode),
            "top_features": selected,
            "feature_names": feature_names,
            "feature_means": means,
            "tree": best_tree,
            "tau": float(best["tau"]),
            "feature_usage": collect_feature_usage(best_tree),
        },
    )
    base.write_json(
        os.path.join(args.out_dir, "summary.json"),
        {
            "inputs": {
                "claim_table_csv": os.path.abspath(args.claim_table_csv),
                "chair_table_csv": os.path.abspath(args.chair_table_csv),
                "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
                "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
                "teacher_mode": str(args.teacher_mode),
                "min_f1_gain": float(args.min_f1_gain),
                "constraint_mode": str(args.constraint_mode),
                "chair_eps": float(args.chair_eps),
                "selection_objective": str(args.selection_objective),
                "feature_family_mode": str(args.feature_family_mode),
                "feature_cols": feature_names,
                "feature_cols_file": os.path.abspath(str(args.feature_cols_file)) if str(args.feature_cols_file).strip() else "",
                "min_feature_auroc": float(args.min_feature_auroc),
                "top_n_features": int(args.top_n_features),
                "top_n_probe_features": int(args.top_n_probe_features),
                "top_n_pair_features": int(args.top_n_pair_features),
                "max_depth_values": max_depth_values,
                "min_leaf_values": min_leaf_values,
                "split_quantiles": split_quantiles,
                "tau_quantiles": tau_quantiles,
            },
            "counts": {
                "n_rows": int(len(rows)),
                "teacher_positive_rate": base.safe_div(float(sum(labels)), float(max(1, len(labels)))),
            },
            "baseline": {k: v for k, v in baseline_summary.items() if k != "decision_rows"},
            "intervention": {k: v for k, v in intervention_summary.items() if k != "decision_rows"},
            "teacher_oracle": {k: v for k, v in teacher_summary.items() if k != "decision_rows"},
            "best_policy": best,
            "outputs": {
                "feature_metrics_csv": os.path.abspath(os.path.join(args.out_dir, "feature_metrics.csv")),
                "feature_metrics_selected_csv": os.path.abspath(os.path.join(args.out_dir, "feature_metrics_selected.csv")),
                "tau_sweep_csv": os.path.abspath(os.path.join(args.out_dir, "tau_sweep.csv")),
                "decision_rows_csv": os.path.abspath(os.path.join(args.out_dir, "decision_rows.csv")),
                "selected_tree_json": os.path.abspath(os.path.join(args.out_dir, "selected_tree.json")),
            },
        },
    )
    print("[saved]", os.path.abspath(os.path.join(args.out_dir, "summary.json")))
    print("[saved]", os.path.abspath(os.path.join(args.out_dir, "decision_rows.csv")))


if __name__ == "__main__":
    main()
