#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def maybe_int(value: object) -> Optional[int]:
    v = maybe_float(value)
    if v is None:
        return None
    return int(round(v))


def mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 1.0
    mu = mean(seq)
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(max(math.sqrt(max(var, 0.0)), 1e-6))


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(int(y) for y in labels)
    n_neg = len(labels) - n_pos
    if len(scores) != len(labels) or n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = sum(rank for rank, y in zip(ranks, labels) if int(y) == 1)
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def binary_average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    if not pairs:
        return None
    n_pos = sum(y for _, y in pairs)
    if n_pos == 0:
        return None
    pairs.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    ap = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            ap += float(tp) / float(rank)
    return float(ap / float(n_pos))


def parse_target_spec(spec: str) -> Tuple[str, str, float]:
    parts = [x.strip() for x in str(spec).split(":")]
    if len(parts) == 1:
        return parts[0], "binary", 0.0
    if len(parts) != 3:
        raise ValueError(f"Invalid target spec: {spec}")
    return parts[0], parts[1], float(parts[2])


def target_value(row: Dict[str, str], spec: str) -> Optional[int]:
    col, op, threshold = parse_target_spec(spec)
    raw = maybe_float(row.get(col))
    if raw is None:
        return None
    if op == "binary":
        if int(round(raw)) in {0, 1}:
            return int(round(raw))
        return None
    if op == "lt":
        return int(float(raw) < float(threshold))
    if op == "le":
        return int(float(raw) <= float(threshold))
    if op == "gt":
        return int(float(raw) > float(threshold))
    if op == "ge":
        return int(float(raw) >= float(threshold))
    if op == "eq":
        return int(float(raw) == float(threshold))
    raise ValueError(f"Unsupported target op: {op}")


def zscore_oriented(values: Sequence[float], direction: str) -> List[float]:
    oriented = [float(v) if direction == "high" else -float(v) for v in values]
    mu = mean(oriented)
    sd = std(oriented)
    return [(float(v) - mu) / sd for v in oriented]


def evaluate_feature(rows: Sequence[Dict[str, str]], feature: str, target_spec: str) -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = maybe_float(row.get(feature))
        y = target_value(row, target_spec)
        if x is None or y not in {0, 1}:
            continue
        xs.append(float(x))
        ys.append(int(y))
    if len(xs) < 2:
        return None
    auc_high = binary_auroc(xs, ys)
    auc_low = binary_auroc([-x for x in xs], ys)
    if auc_high is None or auc_low is None:
        return None
    direction = "high" if auc_high >= auc_low else "low"
    oriented = xs if direction == "high" else [-x for x in xs]
    ap = binary_average_precision(oriented, ys)
    return {
        "feature": feature,
        "target_spec": target_spec,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "average_precision": None if ap is None else float(ap),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
        "positive_rate": float(sum(ys) / float(max(1, len(ys)))),
    }


def build_composite_scores(
    rows: Sequence[Dict[str, str]],
    feature_specs: Sequence[Tuple[str, str]],
) -> Tuple[List[int], List[float], List[int], List[int], List[float], List[float]]:
    labels_harm: List[int] = []
    labels_help: List[int] = []
    ids: List[int] = []
    matrix: List[List[float]] = []
    base_utils: List[float] = []
    int_utils: List[float] = []
    for idx, row in enumerate(rows):
        vals: List[float] = []
        ok = True
        for feature, _direction in feature_specs:
            x = maybe_float(row.get(feature))
            if x is None:
                ok = False
                break
            vals.append(float(x))
        base_u = maybe_float(row.get("baseline_claim_utility"))
        int_u = maybe_float(row.get("intervention_claim_utility"))
        harm = maybe_int(row.get("harm"))
        help_ = maybe_int(row.get("help"))
        if not ok or base_u is None or int_u is None or harm not in {0, 1} or help_ not in {0, 1}:
            continue
        ids.append(idx)
        matrix.append(vals)
        labels_harm.append(int(harm))
        labels_help.append(int(help_))
        base_utils.append(float(base_u))
        int_utils.append(float(int_u))
    if not matrix:
        return [], [], [], [], [], []
    cols = list(zip(*matrix))
    zcols: List[List[float]] = []
    for (_feature, direction), col in zip(feature_specs, cols):
        zcols.append(zscore_oriented(col, direction))
    scores = [mean(zvals) for zvals in zip(*zcols)]
    return ids, scores, labels_harm, labels_help, base_utils, int_utils


def quantiles_to_thresholds(values: Sequence[float], quantiles: Sequence[float]) -> List[float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return [0.0]
    if len(vals) == 1:
        return [vals[0]]
    out = {vals[0] - 1e-6, vals[-1] + 1e-6}
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


def evaluate_tau(
    scores: Sequence[float],
    labels_harm: Sequence[int],
    labels_help: Sequence[int],
    base_utils: Sequence[float],
    int_utils: Sequence[float],
    tau: float,
) -> Dict[str, Any]:
    n = len(scores)
    veto = [float(s) >= float(tau) for s in scores]
    veto_count = sum(1 for x in veto if x)
    final_utils = [float(bu) if v else float(iu) for bu, iu, v in zip(base_utils, int_utils, veto)]
    final_utility = mean(final_utils)
    intervention_utility = mean(int_utils)
    baseline_utility = mean(base_utils)
    selected_harm = sum(int(h) for h, v in zip(labels_harm, veto) if v)
    selected_help = sum(int(h) for h, v in zip(labels_help, veto) if v)
    selected_neutral = int(veto_count - selected_harm - selected_help)
    total_harm = sum(int(x) for x in labels_harm)
    total_help = sum(int(x) for x in labels_help)
    return {
        "tau": float(tau),
        "n_eval": int(n),
        "baseline_rate": safe_div(float(veto_count), float(max(1, n))),
        "method_rate": safe_div(float(n - veto_count), float(max(1, n))),
        "final_claim_utility": float(final_utility),
        "baseline_claim_utility": float(baseline_utility),
        "intervention_claim_utility": float(intervention_utility),
        "delta_vs_intervention": float(final_utility - intervention_utility),
        "delta_vs_baseline": float(final_utility - baseline_utility),
        "selected_count": int(veto_count),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "selected_harm_precision": safe_div(float(selected_harm), float(max(1, veto_count))),
        "selected_help_precision": safe_div(float(selected_help), float(max(1, veto_count))),
        "selected_harm_recall": safe_div(float(selected_harm), float(max(1, total_harm))),
        "selected_help_recall": safe_div(float(selected_help), float(max(1, total_help))),
    }


def selection_key(result: Dict[str, Any], objective: str) -> Sequence[float]:
    if objective == "delta_vs_intervention":
        return (
            float(result["delta_vs_intervention"]),
            float(result["final_claim_utility"]),
            float(result["selected_harm_precision"]),
            -float(result["baseline_rate"]),
        )
    return (
        float(result["final_claim_utility"]),
        float(result["delta_vs_intervention"]),
        float(result["selected_harm_precision"]),
        -float(result["baseline_rate"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a generative post-hoc risk controller from claim-aware tables.")
    ap.add_argument("--discovery_table_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--target_spec", type=str, default="delta_supported_recall:lt:0")
    ap.add_argument("--feature_cols", type=str, required=True)
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k_values", type=str, default="1,2,3,4,5")
    ap.add_argument("--tau_quantiles", type=str, default="0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99")
    ap.add_argument("--tau_objective", type=str, default="final_claim_utility", choices=["final_claim_utility", "delta_vs_intervention"])
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    ap.add_argument("--min_selected_count", type=int, default=0)
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    for path in args.discovery_table_csvs:
        rows.extend(read_csv_rows(path))

    feature_cols = [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]
    top_k_values = sorted({max(1, int(x.strip())) for x in str(args.top_k_values).split(",") if x.strip()})
    tau_quantiles = [float(x.strip()) for x in str(args.tau_quantiles).split(",") if x.strip()]

    feature_metrics: List[Dict[str, Any]] = []
    for feature in feature_cols:
        result = evaluate_feature(rows, feature, args.target_spec)
        if result is None:
            continue
        if float(result["auroc"]) < float(args.min_feature_auroc):
            continue
        feature_metrics.append(result)
    feature_metrics.sort(key=lambda r: (-float(r["auroc"]), -float(r["average_precision"] or 0.0), str(r["feature"])))
    if not feature_metrics:
        raise RuntimeError("No feasible feature passed the AUROC threshold.")

    composite_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_policy: Optional[Dict[str, Any]] = None

    for k in top_k_values:
        selected = feature_metrics[: min(int(k), len(feature_metrics))]
        feature_specs = [(str(r["feature"]), str(r["direction"])) for r in selected]
        _ids, scores, labels_harm, labels_help, base_utils, int_utils = build_composite_scores(rows, feature_specs)
        if not scores:
            continue
        tau_grid = quantiles_to_thresholds(scores, tau_quantiles)
        score_auc = binary_auroc(scores, labels_harm)
        score_ap = binary_average_precision(scores, labels_harm)
        for tau in tau_grid:
            result = evaluate_tau(scores, labels_harm, labels_help, base_utils, int_utils, tau)
            if float(result["baseline_rate"]) < float(args.min_baseline_rate):
                continue
            if float(result["baseline_rate"]) > float(args.max_baseline_rate):
                continue
            if int(result["selected_count"]) < int(args.min_selected_count):
                continue
            row = {
                "k": int(k),
                "features": ",".join(f for f, _ in feature_specs),
                "directions": ",".join(d for _, d in feature_specs),
                "score_harm_auroc": None if score_auc is None else float(score_auc),
                "score_harm_ap": None if score_ap is None else float(score_ap),
                **result,
            }
            composite_rows.append(row)
            if best is None or selection_key(row, str(args.tau_objective)) > selection_key(best, str(args.tau_objective)):
                best = row
                best_policy = {
                    "policy_type": "generative_posthoc_composite_v1",
                    "target_spec": str(args.target_spec),
                    "feature_specs": [{"feature": f, "direction": d} for f, d in feature_specs],
                    "tau": float(tau),
                    "tau_objective": str(args.tau_objective),
                    "route_policy": "baseline if posthoc_risk_score >= tau else method",
                    "min_baseline_rate": float(args.min_baseline_rate),
                    "max_baseline_rate": float(args.max_baseline_rate),
                    "min_selected_count": int(args.min_selected_count),
                }

    if best is None or best_policy is None:
        raise RuntimeError("No feasible generative post-hoc controller found.")

    os.makedirs(args.out_dir, exist_ok=True)
    feat_csv = os.path.join(args.out_dir, "feature_metrics.csv")
    comp_csv = os.path.join(args.out_dir, "composite_tau_sweep.csv")
    policy_json = os.path.join(args.out_dir, "selected_policy.json")
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_csv(feat_csv, feature_metrics)
    write_csv(comp_csv, composite_rows)
    write_json(policy_json, best_policy)
    write_json(summary_json, {
        "inputs": {
            "discovery_table_csvs": [os.path.abspath(x) for x in args.discovery_table_csvs],
            "target_spec": str(args.target_spec),
            "feature_cols": feature_cols,
            "min_feature_auroc": float(args.min_feature_auroc),
            "top_k_values": top_k_values,
            "tau_quantiles": tau_quantiles,
            "tau_objective": str(args.tau_objective),
            "min_baseline_rate": float(args.min_baseline_rate),
            "max_baseline_rate": float(args.max_baseline_rate),
            "min_selected_count": int(args.min_selected_count),
        },
        "n_rows": int(len(rows)),
        "best_feature": feature_metrics[0],
        "best_tau": best,
        "outputs": {
            "feature_metrics_csv": os.path.abspath(feat_csv),
            "composite_tau_sweep_csv": os.path.abspath(comp_csv),
            "policy_json": os.path.abspath(policy_json),
        },
    })
    print("[saved]", os.path.abspath(feat_csv))
    print("[saved]", os.path.abspath(comp_csv))
    print("[saved]", os.path.abspath(policy_json))
    print("[saved]", os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
