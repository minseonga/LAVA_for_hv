#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build_generative_b_c_meta_controller as base


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


def parse_feature_cols_spec(spec: str) -> List[str]:
    out: List[str] = []
    for part in str(spec or "").replace("\n", ",").split(","):
        s = part.strip()
        if s:
            out.append(s)
    return out


def read_feature_cols_file(path: str) -> List[str]:
    cols: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cols.extend(parse_feature_cols_spec(stripped))
    deduped: List[str] = []
    seen = set()
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        deduped.append(col)
    return deduped


def resolve_feature_cols(
    rows: Sequence[Dict[str, Any]],
    feature_cols: str,
    feature_cols_file: str,
) -> List[str]:
    if str(feature_cols_file or "").strip():
        return read_feature_cols_file(os.path.abspath(str(feature_cols_file)))
    if str(feature_cols).strip() == "auto":
        return base.infer_probe_feature_cols(rows)
    return parse_feature_cols_spec(str(feature_cols))


def teacher_label(row: Dict[str, Any], mode: str, min_f1_gain: float = 0.0) -> int:
    bf = float(row["base_f1"])
    if1 = float(row["int_f1"])
    bci = float(row["base_chair_i"])
    ici = float(row["int_chair_i"])
    bcs = float(row["base_chair_s"])
    ics = float(row["int_chair_s"])

    if mode == "strict_pareto":
        choose = (
            (bf > (if1 + float(min_f1_gain)))
            and (bci <= ici)
            and (bcs <= ics)
            and ((bf - if1) > 1e-12 or (ici - bci) > 1e-12 or (ics - bcs) > 1e-12)
        )
        return int(choose)
    if mode == "chairi_pareto":
        choose = (
            (bf > (if1 + float(min_f1_gain)))
            and (bci <= ici)
            and ((bf - if1) > 1e-12 or (ici - bci) > 1e-12)
        )
        return int(choose)
    if mode == "f1_only":
        return int(bf > (if1 + float(min_f1_gain)))
    raise ValueError(f"Unsupported teacher mode: {mode}")


def attach_teacher_labels(
    rows: Sequence[Dict[str, Any]],
    teacher_mode: str,
    min_f1_gain: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["teacher_fallback"] = teacher_label(item, teacher_mode, min_f1_gain=float(min_f1_gain))
        out.append(item)
    return out


def evaluate_feature(rows: Sequence[Dict[str, Any]], feature: str) -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = base.maybe_float(row.get(feature))
        y = base.maybe_int(row.get("teacher_fallback"))
        if x is None or y not in {0, 1}:
            continue
        xs.append(float(x))
        ys.append(int(y))
    if len(xs) < 2:
        return None
    auc_high = base.binary_auroc(xs, ys)
    auc_low = base.binary_auroc([-x for x in xs], ys)
    if auc_high is None or auc_low is None:
        return None
    direction = "high" if auc_high >= auc_low else "low"
    oriented = xs if direction == "high" else [-x for x in xs]
    ap = base.binary_average_precision(oriented, ys)
    return {
        "feature": feature,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "average_precision": None if ap is None else float(ap),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
    }


def feature_family(feature: str) -> str:
    name = str(feature)
    if name.startswith("pair_"):
        return "pair"
    if name.startswith("probe_"):
        return "probe"
    return "other"


def sort_feature_metrics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(row) for row in rows]
    out.sort(key=lambda r: (-float(r["auroc"]), -float(r.get("average_precision") or 0.0), str(r["feature"])))
    return out


def select_feature_metrics(
    feature_metrics_all: Sequence[Dict[str, Any]],
    *,
    min_feature_auroc: float,
    top_n_features: int,
    feature_family_mode: str,
    top_n_probe_features: int,
    top_n_pair_features: int,
) -> List[Dict[str, Any]]:
    filtered = [
        dict(row)
        for row in feature_metrics_all
        if float(row["auroc"]) >= float(min_feature_auroc)
    ]
    filtered = sort_feature_metrics(filtered)
    if not filtered:
        return []

    mode = str(feature_family_mode)
    top_n = max(1, int(top_n_features))
    if mode == "overall":
        return filtered[:top_n]
    if mode == "probe_only":
        return [row for row in filtered if feature_family(str(row["feature"])) == "probe"][:top_n]
    if mode == "pair_only":
        return [row for row in filtered if feature_family(str(row["feature"])) == "pair"][:top_n]
    if mode != "balanced":
        raise ValueError(f"Unsupported feature_family_mode: {feature_family_mode}")

    selected: List[Dict[str, Any]] = []
    used = set()
    probe_rows = [row for row in filtered if feature_family(str(row["feature"])) == "probe"]
    pair_rows = [row for row in filtered if feature_family(str(row["feature"])) == "pair"]

    for row in probe_rows[: max(0, int(top_n_probe_features))]:
        feat = str(row["feature"])
        if feat not in used:
            used.add(feat)
            selected.append(row)
    for row in pair_rows[: max(0, int(top_n_pair_features))]:
        feat = str(row["feature"])
        if feat not in used:
            used.add(feat)
            selected.append(row)
    for row in filtered:
        if len(selected) >= top_n:
            break
        feat = str(row["feature"])
        if feat in used:
            continue
        used.add(feat)
        selected.append(row)
    return selected


def compute_feature_stats(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Tuple[str, str]],
) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for feature, direction in feature_specs:
        vals: List[float] = []
        for row in rows:
            x = base.maybe_float(row.get(feature))
            if x is None:
                continue
            vals.append(float(x) if direction == "high" else -float(x))
        stats[feature] = (base.mean(vals), base.std(vals))
    return stats


def build_z_matrix(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Tuple[str, str]],
    stats: Dict[str, Tuple[float, float]],
) -> List[List[float]]:
    matrix: List[List[float]] = []
    for row in rows:
        zvals: List[float] = []
        for feature, direction in feature_specs:
            x = base.maybe_float(row.get(feature))
            if x is None:
                zvals.append(0.0)
                continue
            oriented = float(x) if direction == "high" else -float(x)
            mu, sd = stats[feature]
            zvals.append((oriented - mu) / sd)
        matrix.append(zvals)
    return matrix


def combine_scores(matrix: Sequence[Sequence[float]], weights: Sequence[float]) -> List[float]:
    scores: List[float] = []
    for row in matrix:
        scores.append(float(sum(float(w) * float(v) for w, v in zip(weights, row))))
    return scores


def fit_linear_weights(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Tuple[str, str]],
    weight_grid: Sequence[float],
    num_passes: int,
) -> Tuple[List[float], Dict[str, Tuple[float, float]], List[List[float]], Dict[str, Any]]:
    stats = compute_feature_stats(rows, feature_specs)
    matrix = build_z_matrix(rows, feature_specs, stats)
    labels = [int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows]
    weights = [0.0 for _ in feature_specs]
    history: List[Dict[str, Any]] = []
    best_auc = -1.0

    for pass_idx in range(int(num_passes)):
        changed = False
        for j in range(len(feature_specs)):
            current_best = weights[j]
            current_auc = -1.0
            for cand in weight_grid:
                trial = list(weights)
                trial[j] = float(cand)
                scores = combine_scores(matrix, trial)
                auc = base.binary_auroc(scores, labels)
                if auc is None:
                    continue
                key = (float(auc), abs(float(cand)))
                best_key = (float(current_auc), abs(float(current_best)))
                if key > best_key:
                    current_best = float(cand)
                    current_auc = float(auc)
            if current_best != weights[j]:
                changed = True
                weights[j] = float(current_best)
            history.append(
                {
                    "pass": int(pass_idx),
                    "feature": feature_specs[j][0],
                    "weight": float(weights[j]),
                    "teacher_auroc": float(current_auc),
                }
            )
            best_auc = max(best_auc, float(current_auc))
        if not changed:
            break

    scores = combine_scores(matrix, weights)
    final_auc = base.binary_auroc(scores, labels)
    final_ap = base.binary_average_precision(scores, labels)
    fit_summary = {
        "teacher_auroc": None if final_auc is None else float(final_auc),
        "teacher_ap": None if final_ap is None else float(final_ap),
        "history": history,
    }
    return weights, stats, matrix, fit_summary


def route_summary(rows: Sequence[Dict[str, Any]], routes: Sequence[str]) -> Dict[str, Any]:
    summary = base.aggregate_routes(rows, routes)
    summary["teacher_precision"] = base.safe_div(
        float(sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row, route in zip(rows, routes) if route == "baseline")),
        float(max(1, sum(1 for route in routes if route == "baseline"))),
    )
    summary["teacher_recall"] = base.safe_div(
        float(sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row, route in zip(rows, routes) if route == "baseline")),
        float(max(1, sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows))),
    )
    return summary


def feasible_under_constraints(
    candidate: Dict[str, Any],
    intervention: Dict[str, Any],
    constraint_mode: str,
    chair_eps: float,
) -> bool:
    if constraint_mode == "none":
        return True
    if constraint_mode == "chairi":
        return float(candidate["mean_chair_i"]) <= float(intervention["mean_chair_i"]) + float(chair_eps)
    if constraint_mode == "chairs":
        return float(candidate["mean_chair_s"]) <= float(intervention["mean_chair_s"]) + float(chair_eps)
    if constraint_mode == "both":
        return (
            float(candidate["mean_chair_i"]) <= float(intervention["mean_chair_i"]) + float(chair_eps)
            and float(candidate["mean_chair_s"]) <= float(intervention["mean_chair_s"]) + float(chair_eps)
        )
    raise ValueError(f"Unsupported constraint mode: {constraint_mode}")


def selection_key(row: Dict[str, Any], objective: str) -> Tuple[float, float, float]:
    if objective == "f1_minus_chairi":
        return (
            float(row["mean_f1_minus_chairi"]),
            float(row["mean_f1"]),
            -float(row["baseline_rate"]),
        )
    if objective == "neg_chairi":
        return (
            -float(row["mean_chair_i"]),
            float(row["mean_f1"]),
            -float(row["baseline_rate"]),
        )
    if objective == "claim_utility":
        return (
            float(row["mean_claim_utility"]),
            float(row["mean_f1"]),
            -float(row["baseline_rate"]),
        )
    return (
        float(row["mean_f1"]),
        -float(row["mean_chair_i"]),
        -float(row["baseline_rate"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a shallow generative post-hoc controller against a Pareto-teacher fallback target.")
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
    ap.add_argument("--top_n_features", type=int, default=6)
    ap.add_argument("--feature_family_mode", type=str, default="overall", choices=["overall", "balanced", "probe_only", "pair_only"])
    ap.add_argument("--top_n_probe_features", type=int, default=6)
    ap.add_argument("--top_n_pair_features", type=int, default=6)
    ap.add_argument("--weight_grid", type=str, default="0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--num_passes", type=int, default=3)
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
    rows = attach_teacher_labels(rows, str(args.teacher_mode), float(args.min_f1_gain))
    feature_cols = resolve_feature_cols(rows, str(args.feature_cols), str(args.feature_cols_file))
    feature_metrics_all: List[Dict[str, Any]] = []
    for feat in feature_cols:
        res = evaluate_feature(rows, feat)
        if res is None:
            continue
        feature_metrics_all.append(res)
    feature_metrics_all = sort_feature_metrics(feature_metrics_all)
    feature_metrics = select_feature_metrics(
        feature_metrics_all,
        min_feature_auroc=float(args.min_feature_auroc),
        top_n_features=int(args.top_n_features),
        feature_family_mode=str(args.feature_family_mode),
        top_n_probe_features=int(args.top_n_probe_features),
        top_n_pair_features=int(args.top_n_pair_features),
    )
    if not feature_metrics:
        raise RuntimeError("No feasible features for Pareto-teacher controller.")

    selected = feature_metrics[: max(1, int(args.top_n_features))]
    feature_specs = [(str(r["feature"]), str(r["direction"])) for r in selected]
    weight_grid = parse_float_list(args.weight_grid)
    tau_quantiles = parse_float_list(args.tau_quantiles)

    weights, stats, matrix, fit_summary = fit_linear_weights(
        rows,
        feature_specs,
        weight_grid=weight_grid,
        num_passes=int(args.num_passes),
    )
    scores = combine_scores(matrix, weights)
    tau_grid = base.quantiles_to_thresholds(scores, tau_quantiles)
    baseline_summary = base.aggregate_routes(rows, ["baseline"] * len(rows))
    intervention_summary = base.aggregate_routes(rows, ["method"] * len(rows))

    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_routes: List[str] = []
    for tau in tau_grid:
        routes = [base.route_by_score(score, float(tau)) for score in scores]
        summary = route_summary(rows, routes)
        if float(summary["baseline_rate"]) < float(args.min_baseline_rate):
            continue
        if float(summary["baseline_rate"]) > float(args.max_baseline_rate):
            continue
        if not feasible_under_constraints(summary, intervention_summary, str(args.constraint_mode), float(args.chair_eps)):
            continue
        row = {
            "tau": float(tau),
            "teacher_mode": str(args.teacher_mode),
            "constraint_mode": str(args.constraint_mode),
            "selection_objective": str(args.selection_objective),
            **{k: v for k, v in summary.items() if k != "decision_rows"},
        }
        sweep_rows.append(row)
        if best is None or selection_key(row, str(args.selection_objective)) > selection_key(best, str(args.selection_objective)):
            best = dict(row)
            best_routes = routes

    if best is None:
        raise RuntimeError("No feasible controller satisfied the requested constraints.")

    decision_rows: List[Dict[str, Any]] = []
    for row, score, route in zip(rows, scores, best_routes):
        out = dict(row)
        out["pareto_teacher_score"] = float(score)
        out["route"] = route
        decision_rows.append(out)

    teacher_rows = route_summary(rows, ["baseline" if int(base.maybe_int(row.get("teacher_fallback")) or 0) == 1 else "method" for row in rows])
    teacher_rows["teacher_rate"] = base.safe_div(float(sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows)), float(max(1, len(rows))))

    os.makedirs(args.out_dir, exist_ok=True)
    base.write_csv(os.path.join(args.out_dir, "feature_metrics.csv"), feature_metrics_all)
    base.write_csv(os.path.join(args.out_dir, "feature_metrics_selected.csv"), feature_metrics)
    base.write_csv(os.path.join(args.out_dir, "tau_sweep.csv"), sweep_rows)
    base.write_csv(os.path.join(args.out_dir, "decision_rows.csv"), decision_rows)
    base.write_json(
        os.path.join(args.out_dir, "selected_policy.json"),
        {
            "policy_type": "generative_pareto_teacher_linear_v1",
            "teacher_mode": str(args.teacher_mode),
            "min_f1_gain": float(args.min_f1_gain),
            "constraint_mode": str(args.constraint_mode),
            "chair_eps": float(args.chair_eps),
            "selection_objective": str(args.selection_objective),
            "feature_family_mode": str(args.feature_family_mode),
            "feature_specs": [{"feature": f, "direction": d, "weight": w} for (f, d), w in zip(feature_specs, weights)],
            "feature_stats": {
                str(f): {"mean": float(stats[f][0]), "std": float(stats[f][1])}
                for f, _d in feature_specs
            },
            "tau": float(best["tau"]),
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
                "feature_cols": feature_cols,
                "feature_cols_file": os.path.abspath(str(args.feature_cols_file)) if str(args.feature_cols_file).strip() else "",
                "min_feature_auroc": float(args.min_feature_auroc),
                "top_n_features": int(args.top_n_features),
                "top_n_probe_features": int(args.top_n_probe_features),
                "top_n_pair_features": int(args.top_n_pair_features),
                "weight_grid": weight_grid,
                "num_passes": int(args.num_passes),
                "tau_quantiles": tau_quantiles,
            },
            "counts": {
                "n_rows": int(len(rows)),
                "teacher_positive_rate": base.safe_div(float(sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows)), float(max(1, len(rows)))),
            },
            "baseline": {k: v for k, v in baseline_summary.items() if k != "decision_rows"},
            "intervention": {k: v for k, v in intervention_summary.items() if k != "decision_rows"},
            "teacher_oracle": {k: v for k, v in teacher_rows.items() if k != "decision_rows"},
            "fit_summary": fit_summary,
            "best_policy": best,
            "outputs": {
                "feature_metrics_csv": os.path.abspath(os.path.join(args.out_dir, "feature_metrics.csv")),
                "feature_metrics_selected_csv": os.path.abspath(os.path.join(args.out_dir, "feature_metrics_selected.csv")),
                "tau_sweep_csv": os.path.abspath(os.path.join(args.out_dir, "tau_sweep.csv")),
                "decision_rows_csv": os.path.abspath(os.path.join(args.out_dir, "decision_rows.csv")),
            },
        },
    )
    print("[saved]", os.path.abspath(os.path.join(args.out_dir, "summary.json")))
    print("[saved]", os.path.abspath(os.path.join(args.out_dir, "decision_rows.csv")))


if __name__ == "__main__":
    main()
