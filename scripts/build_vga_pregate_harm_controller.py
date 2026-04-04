#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable


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


def mean(seq: Sequence[float]) -> float:
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std(seq: Sequence[float]) -> float:
    if len(seq) <= 1:
        return 1.0
    mu = mean(seq)
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(max(math.sqrt(max(var, 0.0)), 1e-6))


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def feature_cols(rows: Sequence[Dict[str, str]], allowlist: Optional[Sequence[str]] = None) -> List[str]:
    reserved = {
        "id",
        "benchmark",
        "split",
        "question",
        "image",
        "image_id",
        "category",
        "gt_label",
        "baseline_text",
        "intervention_text",
        "baseline_label",
        "intervention_label",
        "baseline_correct",
        "intervention_correct",
        "harm",
        "help",
        "utility",
        "oracle_route",
        "oracle_correct",
    }
    allow = set(str(x) for x in allowlist) if allowlist else None
    cols: List[str] = []
    if not rows:
        return cols
    for key in rows[0].keys():
        if key in reserved:
            continue
        if allow is not None and key not in allow:
            continue
        if maybe_float(rows[0].get(key)) is not None:
            cols.append(key)
    return cols


def evaluate_feature(rows: Sequence[Dict[str, str]], feature: str, target: str = "harm") -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = maybe_float(row.get(feature))
        y = maybe_int(row.get(target))
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
    return {
        "feature": feature,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
    }


def oriented_value(raw: float, direction: str) -> float:
    return float(raw) if direction == "high" else float(-raw)


def build_score_row(row: Dict[str, str], features: Sequence[Dict[str, Any]]) -> Optional[float]:
    zs: List[float] = []
    for feat in features:
        raw = maybe_float(row.get(str(feat["feature"])))
        if raw is None:
            return None
        oriented = oriented_value(raw, str(feat["direction"]))
        mu = float(feat["mu"])
        sd = max(float(feat["sd"]), 1e-6)
        zs.append((oriented - mu) / sd)
    if not zs:
        return None
    return float(sum(zs) / float(len(zs)))


def evaluate_tau(rows: Sequence[Dict[str, str]], features: Sequence[Dict[str, Any]], tau: float) -> Dict[str, Any]:
    n = 0
    route_baseline = 0
    final_correct = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0
    total_harm = 0
    total_help = 0
    for row in rows:
        score = build_score_row(row, features)
        if score is None:
            continue
        n += 1
        baseline_correct = int(maybe_int(row.get("baseline_correct")) or 0)
        intervention_correct = int(maybe_int(row.get("intervention_correct")) or 0)
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        baseline_correct_total += baseline_correct
        intervention_correct_total += intervention_correct
        total_harm += harm
        total_help += help_
        use_baseline = bool(score >= tau)
        if use_baseline:
            route_baseline += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int((harm == 0) and (help_ == 0))
            final_correct += baseline_correct
        else:
            final_correct += intervention_correct
    baseline_rate = float(route_baseline / float(max(1, n)))
    acc = float(final_correct / float(max(1, n)))
    precision = safe_div(float(selected_harm), float(route_baseline))
    recall = safe_div(float(selected_harm), float(total_harm))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    delta_vs_intervention = safe_div(float(final_correct - intervention_correct_total), float(max(1, n)))
    return {
        "tau": float(tau),
        "n_eval": int(n),
        "baseline_rate": baseline_rate,
        "method_rate": float(1.0 - baseline_rate),
        "final_acc": acc,
        "baseline_acc": safe_div(float(baseline_correct_total), float(max(1, n))),
        "intervention_acc": safe_div(float(intervention_correct_total), float(max(1, n))),
        "delta_vs_intervention": delta_vs_intervention,
        "selected_count": int(route_baseline),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "selected_harm_precision": precision,
        "selected_help_precision": safe_div(float(selected_help), float(route_baseline)),
        "selected_harm_recall": recall,
        "selected_harm_f1": f1,
    }


def selection_key(result: Dict[str, Any], objective: str) -> Sequence[float]:
    if objective == "harm_f1":
        return (
            float(result["selected_harm_f1"]),
            float(result["selected_harm_precision"]),
            float(result["delta_vs_intervention"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_precision":
        return (
            float(result["selected_harm_precision"]),
            float(result["selected_harm_recall"]),
            float(result["delta_vs_intervention"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_recall":
        return (
            float(result["selected_harm_recall"]),
            float(result["selected_harm_precision"]),
            float(result["delta_vs_intervention"]),
            -float(result["baseline_rate"]),
        )
    return (
        float(result["final_acc"]),
        float(result["delta_vs_intervention"]),
        float(result["selected_harm_precision"]),
        -float(result["baseline_rate"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a unified pre-intervention VGA harm gate from discovery tables.")
    ap.add_argument("--discovery_table_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--target_label", type=str, default="harm", choices=["harm"])
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--min_selected_count", type=int, default=0)
    ap.add_argument(
        "--tau_objective",
        type=str,
        default="final_acc",
        choices=["final_acc", "harm_f1", "harm_precision", "harm_recall"],
    )
    ap.add_argument("--feature_cols", type=str, default="")
    args = ap.parse_args()

    allowlist = [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]

    all_rows: List[Dict[str, str]] = []
    source_rows: Dict[str, List[Dict[str, str]]] = {}
    for path in tqdm(args.discovery_table_csvs, desc="controller-load", unit="table"):
        rows = read_csv_rows(path)
        if not rows:
            continue
        source = str(rows[0].get("benchmark", os.path.basename(path)))
        source_rows[source] = rows
        all_rows.extend(rows)
    feats = feature_cols(all_rows, allowlist=allowlist if allowlist else None)

    per_source_metrics: List[Dict[str, Any]] = []
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    for source, rows in tqdm(source_rows.items(), total=len(source_rows), desc="controller-source", unit="source"):
        for feat in tqdm(feats, desc=f"controller-features:{source}", unit="feature", leave=False):
            result = evaluate_feature(rows, feat, target=args.target_label)
            if result is None:
                continue
            result["source"] = source
            per_source_metrics.append(result)
            by_feature.setdefault(feat, []).append(result)

    candidates: List[Dict[str, Any]] = []
    for feat, rows in by_feature.items():
        if len(rows) != len(source_rows):
            continue
        directions = {str(r["direction"]) for r in rows}
        min_auc = min(float(r["auroc"]) for r in rows)
        mean_auc = mean([float(r["auroc"]) for r in rows])
        if len(directions) != 1 or min_auc < float(args.min_feature_auroc):
            continue
        direction = next(iter(directions))
        oriented_vals = [
            oriented_value(float(maybe_float(row.get(feat))), direction)
            for row in all_rows
            if maybe_float(row.get(feat)) is not None
        ]
        candidates.append(
            {
                "feature": feat,
                "direction": direction,
                "mean_auroc": mean_auc,
                "min_auroc": min_auc,
                "mu": mean(oriented_vals),
                "sd": std(oriented_vals),
            }
        )
    candidates.sort(key=lambda r: (-float(r["mean_auroc"]), -float(r["min_auroc"]), str(r["feature"])))
    selected = candidates[: max(1, int(args.top_k))]

    score_candidates: List[float] = []
    for row in all_rows:
        score = build_score_row(row, selected)
        if score is not None:
            score_candidates.append(float(score))
    uniq_scores = sorted(set(score_candidates))
    tau_grid = [float("-inf")] + uniq_scores + [float("inf")]
    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for tau in tqdm(tau_grid, desc="controller-tau", unit="tau"):
        result = evaluate_tau(all_rows, selected, tau)
        if float(result["baseline_rate"]) > float(args.max_baseline_rate):
            continue
        if float(result["baseline_rate"]) < float(args.min_baseline_rate):
            continue
        if int(result["selected_count"]) < int(args.min_selected_count):
            continue
        sweep_rows.append(result)
        if best is None:
            best = result
            continue
        cand_key = selection_key(result, str(args.tau_objective))
        best_key = selection_key(best, str(args.tau_objective))
        if cand_key > best_key:
            best = result
    if best is None:
        raise RuntimeError("No feasible tau found for the discovery tables.")

    policy = {
        "policy_type": "composite_harm_gate",
        "target_label": args.target_label,
        "features": selected,
        "tau": float(best["tau"]),
        "tau_objective": str(args.tau_objective),
        "min_baseline_rate": float(args.min_baseline_rate),
        "max_baseline_rate": float(args.max_baseline_rate),
        "min_selected_count": int(args.min_selected_count),
        "route_policy": "baseline if harm_score >= tau else method",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_csv = os.path.join(args.out_dir, "feature_direction_metrics.csv")
    candidates_csv = os.path.join(args.out_dir, "candidate_features.csv")
    sweep_csv = os.path.join(args.out_dir, "tau_sweep.csv")
    policy_json = os.path.join(args.out_dir, "selected_policy.json")
    summary_json = os.path.join(args.out_dir, "summary.json")

    write_csv(metrics_csv, per_source_metrics)
    write_csv(candidates_csv, candidates)
    write_csv(sweep_csv, sweep_rows)
    write_json(policy_json, policy)
    summary = {
        "inputs": {
            "discovery_table_csvs": [os.path.abspath(x) for x in args.discovery_table_csvs],
            "target_label": args.target_label,
            "min_feature_auroc": float(args.min_feature_auroc),
            "top_k": int(args.top_k),
            "tau_objective": str(args.tau_objective),
            "min_baseline_rate": float(args.min_baseline_rate),
            "max_baseline_rate": float(args.max_baseline_rate),
            "min_selected_count": int(args.min_selected_count),
            "feature_cols": allowlist,
        },
        "n_rows": int(len(all_rows)),
        "sources": sorted(source_rows.keys()),
        "selected_features": selected,
        "best_tau": best,
        "outputs": {
            "feature_direction_metrics_csv": os.path.abspath(metrics_csv),
            "candidate_features_csv": os.path.abspath(candidates_csv),
            "tau_sweep_csv": os.path.abspath(sweep_csv),
            "policy_json": os.path.abspath(policy_json),
        },
    }
    write_json(summary_json, summary)
    print("[saved]", os.path.abspath(metrics_csv))
    print("[saved]", os.path.abspath(candidates_csv))
    print("[saved]", os.path.abspath(sweep_csv))
    print("[saved]", os.path.abspath(policy_json))
    print("[saved]", os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
