#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import evaluate_chair_pairwise_proxy as proxy
import evaluate_chair_pairwise_rollback as oracle


def parse_csv_floats(spec: str, default: Sequence[float]) -> List[float]:
    values: List[float] = []
    for part in str(spec or "").split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    return values or [float(x) for x in default]


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(x) for x in values) / float(len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(sum((float(x) - mu) ** 2 for x in values) / float(len(values))))


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return None
    qq = min(1.0, max(0.0, float(q)))
    pos = qq * float(len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    w = pos - float(lo)
    return float((1.0 - w) * vals[lo] + w * vals[hi])


def feature_scores(
    rows: Sequence[Dict[str, Any]],
    feature: str,
    direction: str,
) -> List[Tuple[int, Dict[str, Any], float]]:
    out: List[Tuple[int, Dict[str, Any], float]] = []
    for idx, row in enumerate(rows):
        value = proxy.maybe_float(row.get(feature))
        if value is None:
            continue
        oriented = float(value) if direction == "high" else -float(value)
        out.append((idx, row, oriented))
    return out


def top_budget_eval(
    rows: Sequence[Dict[str, Any]],
    scored: Sequence[Tuple[int, Dict[str, Any], float]],
    budget: float,
) -> Dict[str, Any]:
    n = len(rows)
    k = int(round(float(n) * float(budget)))
    if float(budget) > 0.0:
        k = max(1, k)
    k = min(n, max(0, k))
    selected_ids = {id(row) for _, row, _ in sorted(scored, key=lambda item: item[2], reverse=True)[:k]}
    labels = [int(row.get("teacher_positive") or 0) for _, row, _ in scored if id(row) in selected_ids]
    result = oracle.aggregate_counts(
        rows,
        lambda row, selected_ids=selected_ids: "baseline" if id(row) in selected_ids else "method",
    )
    intervention = oracle.aggregate_counts(rows, lambda _: "method")
    result.update(
        {
            "budget": float(budget),
            "n_selected": int(k),
            "teacher_precision": oracle.safe_div(float(sum(labels)), float(k)),
            "teacher_recall": oracle.safe_div(
                float(sum(labels)),
                float(sum(int(row.get("teacher_positive") or 0) for row in rows)),
            ),
            "delta_chair_i_vs_int": float(result["chair_i"] - intervention["chair_i"]),
            "delta_chair_s_vs_int": float(result["chair_s"] - intervention["chair_s"]),
            "delta_recall_vs_int": float(result["recall"] - intervention["recall"]),
            "delta_f1_vs_int": float(result["f1"] - intervention["f1"]),
        }
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose why GT-free CHAIR pairwise rollback proxies miss the oracle-positive samples."
    )
    ap.add_argument("--baseline_chair_json", required=True)
    ap.add_argument("--intervention_chair_json", required=True)
    ap.add_argument("--extra_features_csv", action="append", default=[])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chair_i_eps", type=float, default=0.01)
    ap.add_argument("--chair_s_eps", type=float, default=0.0)
    ap.add_argument("--min_recall_gain", type=float, default=0.0)
    ap.add_argument("--require_f1_nondecrease", action="store_true")
    ap.add_argument("--feature", action="append", default=[])
    ap.add_argument("--feature_prefix", action="append", default=[])
    ap.add_argument("--top_n", type=int, default=40)
    ap.add_argument("--budgets", type=str, default="0.01,0.02,0.05,0.08")
    args = ap.parse_args()

    rows = oracle.build_rows(
        os.path.abspath(args.baseline_chair_json),
        os.path.abspath(args.intervention_chair_json),
        chair_i_eps=float(args.chair_i_eps),
        chair_s_eps=float(args.chair_s_eps),
        min_recall_gain=float(args.min_recall_gain),
        require_f1_nondecrease=bool(args.require_f1_nondecrease),
    )
    feature_names = proxy.add_caption_pair_proxy_features(rows)
    extra_names = proxy.add_extra_numeric_features(rows, args.extra_features_csv)
    feature_names.extend([name for name in extra_names if name not in set(feature_names)])
    rollback_names = proxy.add_rollback_gain_cost_features(rows)
    feature_names.extend([name for name in rollback_names if name not in set(feature_names)])

    requested = [str(x) for x in args.feature]
    prefixes = [str(x) for x in args.feature_prefix]
    if requested:
        candidates = [name for name in requested if name in set(feature_names)]
    elif prefixes:
        candidates = [name for name in feature_names if any(str(name).startswith(prefix) for prefix in prefixes)]
    else:
        candidates = [name for name in feature_names if str(name).startswith("pair_claimdelta_rollback_")]

    budgets = parse_csv_floats(args.budgets, default=[0.01, 0.02, 0.05, 0.08])
    metrics: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    labels_all = [int(row.get("teacher_positive") or 0) for row in rows]

    for feature in candidates:
        direction_metric = proxy.feature_direction(rows, feature)
        if direction_metric is None:
            continue
        direction = str(direction_metric["direction"])
        scored = feature_scores(rows, feature, direction)
        if not scored:
            continue
        pos_vals = [score for _, row, score in scored if int(row.get("teacher_positive") or 0) == 1]
        neg_vals = [score for _, row, score in scored if int(row.get("teacher_positive") or 0) == 0]
        row_out: Dict[str, Any] = dict(direction_metric)
        row_out.update(
            {
                "n_valid": int(len(scored)),
                "n_pos": int(sum(labels_all)),
                "pos_mean_oriented": mean(pos_vals),
                "neg_mean_oriented": mean(neg_vals),
                "pos_std_oriented": std(pos_vals),
                "neg_std_oriented": std(neg_vals),
                "pos_q50_oriented": quantile(pos_vals, 0.5),
                "neg_q50_oriented": quantile(neg_vals, 0.5),
                "pos_q90_oriented": quantile(pos_vals, 0.9),
                "neg_q90_oriented": quantile(neg_vals, 0.9),
                "mean_gap_oriented": mean(pos_vals) - mean(neg_vals),
            }
        )
        metrics.append(row_out)
        for budget in budgets:
            budget_result = top_budget_eval(rows, scored, budget)
            budget_result.update(
                {
                    "feature": feature,
                    "direction": direction,
                    "teacher_auroc": row_out.get("teacher_auroc"),
                    "teacher_ap": row_out.get("teacher_ap"),
                }
            )
            budget_rows.append(budget_result)

    metrics.sort(
        key=lambda row: (
            -float(row.get("teacher_auroc") or 0.0),
            -float(row.get("teacher_ap") or 0.0),
            str(row.get("feature") or ""),
        )
    )
    budget_rows.sort(
        key=lambda row: (
            -float(row.get("delta_f1_vs_int") or -1e9),
            -float(row.get("delta_recall_vs_int") or -1e9),
            float(row.get("delta_chair_i_vs_int") or 1e9),
            str(row.get("feature") or ""),
        )
    )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    oracle.write_csv(os.path.join(out_dir, "feature_gap_metrics.csv"), metrics)
    oracle.write_csv(os.path.join(out_dir, "budget_eval.csv"), budget_rows)
    summary = {
        "inputs": {
            "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
            "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
            "extra_features_csv": [os.path.abspath(p) for p in args.extra_features_csv],
            "chair_i_eps": float(args.chair_i_eps),
            "chair_s_eps": float(args.chair_s_eps),
            "min_recall_gain": float(args.min_recall_gain),
            "require_f1_nondecrease": bool(args.require_f1_nondecrease),
            "feature": requested,
            "feature_prefix": prefixes,
            "budgets": budgets,
        },
        "counts": {
            "n_rows": int(len(rows)),
            "teacher_positive_rate": oracle.safe_div(float(sum(labels_all)), float(len(rows))),
            "n_features_evaluated": int(len(metrics)),
            "n_extra_features": int(len(extra_names)),
            "n_rollback_gain_cost_features": int(len(rollback_names)),
        },
        "baseline": oracle.aggregate_counts(rows, lambda _: "baseline"),
        "intervention": oracle.aggregate_counts(rows, lambda _: "method"),
        "teacher_oracle": oracle.aggregate_counts(
            rows,
            lambda row: "baseline" if int(row["teacher_positive"]) == 1 else "method",
        ),
        "top_features_by_auroc": metrics[: min(20, len(metrics))],
        "top_budget_policies_by_f1": budget_rows[: min(20, len(budget_rows))],
        "outputs": {
            "feature_gap_metrics_csv": os.path.join(out_dir, "feature_gap_metrics.csv"),
            "budget_eval_csv": os.path.join(out_dir, "budget_eval.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    oracle.write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
