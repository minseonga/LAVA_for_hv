#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def oriented_value(raw: float, direction: str) -> float:
    return float(raw) if direction == "high" else float(-raw)


def evaluate_feature(rows: Sequence[Dict[str, str]], feature: str, target: str) -> Optional[Dict[str, Any]]:
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
        "target": target,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
    }


def select_family(
    source_rows: Dict[str, List[Dict[str, str]]],
    features: Sequence[str],
    target: str,
    min_feature_auroc: float,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_source_metrics: List[Dict[str, Any]] = []
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    all_rows = [row for rows in source_rows.values() for row in rows]
    for source, rows in tqdm(source_rows.items(), total=len(source_rows), desc=f"select:{target}", unit="source"):
        for feat in tqdm(features, desc=f"{target}-features:{source}", unit="feature", leave=False):
            result = evaluate_feature(rows, feat, target=target)
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
        if len(directions) != 1 or min_auc < float(min_feature_auroc):
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
                "target": target,
                "direction": direction,
                "mean_auroc": mean_auc,
                "min_auroc": min_auc,
                "mu": mean(oriented_vals),
                "sd": std(oriented_vals),
            }
        )
    candidates.sort(key=lambda r: (-float(r["mean_auroc"]), -float(r["min_auroc"]), str(r["feature"])))
    return per_source_metrics, candidates[: max(1, int(top_k))]


def build_family_score(row: Dict[str, str], features: Sequence[Dict[str, Any]]) -> Optional[float]:
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


def build_net_score(row: Dict[str, str], harm_features: Sequence[Dict[str, Any]], help_features: Sequence[Dict[str, Any]], lam: float) -> Optional[Tuple[float, float, float]]:
    harm_score = build_family_score(row, harm_features)
    help_score = build_family_score(row, help_features)
    if harm_score is None or help_score is None:
        return None
    net = float(harm_score - float(lam) * help_score)
    return float(harm_score), float(help_score), net


def evaluate_tau(
    source_rows: Dict[str, List[Dict[str, str]]],
    harm_features: Sequence[Dict[str, Any]],
    help_features: Sequence[Dict[str, Any]],
    lam: float,
    tau: float,
) -> Dict[str, Any]:
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
    per_source: Dict[str, Dict[str, float]] = {}

    for source, rows in source_rows.items():
        src = {
            "n": 0,
            "route_baseline": 0,
            "baseline_correct_total": 0,
            "intervention_correct_total": 0,
            "final_correct": 0,
            "selected_harm": 0,
            "selected_help": 0,
            "selected_neutral": 0,
            "total_harm": 0,
            "total_help": 0,
        }
        for row in rows:
            scores = build_net_score(row, harm_features, help_features, lam)
            if scores is None:
                continue
            _, _, net_score = scores
            src["n"] += 1
            n += 1
            baseline_correct = int(maybe_int(row.get("baseline_correct")) or 0)
            intervention_correct = int(maybe_int(row.get("intervention_correct")) or 0)
            harm = int(maybe_int(row.get("harm")) or 0)
            help_ = int(maybe_int(row.get("help")) or 0)
            use_baseline = bool(net_score >= tau)

            baseline_correct_total += baseline_correct
            intervention_correct_total += intervention_correct
            total_harm += harm
            total_help += help_
            src["baseline_correct_total"] += baseline_correct
            src["intervention_correct_total"] += intervention_correct
            src["total_harm"] += harm
            src["total_help"] += help_

            if use_baseline:
                route_baseline += 1
                src["route_baseline"] += 1
                selected_harm += harm
                selected_help += help_
                selected_neutral += int((harm == 0) and (help_ == 0))
                src["selected_harm"] += harm
                src["selected_help"] += help_
                src["selected_neutral"] += int((harm == 0) and (help_ == 0))
                final_correct += baseline_correct
                src["final_correct"] += baseline_correct
            else:
                final_correct += intervention_correct
                src["final_correct"] += intervention_correct

        src_n = int(src["n"])
        src_route = int(src["route_baseline"])
        src["baseline_rate"] = safe_div(float(src_route), float(max(1, src_n)))
        src["final_acc"] = safe_div(float(src["final_correct"]), float(max(1, src_n)))
        src["intervention_acc"] = safe_div(float(src["intervention_correct_total"]), float(max(1, src_n)))
        src["delta_vs_intervention"] = safe_div(
            float(src["final_correct"] - src["intervention_correct_total"]),
            float(max(1, src_n)),
        )
        src["selected_harm_precision"] = safe_div(float(src["selected_harm"]), float(max(1, src_route)))
        src["selected_harm_recall"] = safe_div(float(src["selected_harm"]), float(max(1, int(src["total_harm"]))))
        per_source[source] = src

    baseline_rate = safe_div(float(route_baseline), float(max(1, n)))
    precision = safe_div(float(selected_harm), float(max(1, route_baseline)))
    recall = safe_div(float(selected_harm), float(max(1, total_harm)))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    source_balanced_utility = mean([float(v["delta_vs_intervention"]) for v in per_source.values()]) if per_source else 0.0

    return {
        "lambda_help": float(lam),
        "tau": float(tau),
        "n_eval": int(n),
        "baseline_rate": baseline_rate,
        "method_rate": float(1.0 - baseline_rate),
        "final_acc": safe_div(float(final_correct), float(max(1, n))),
        "baseline_acc": safe_div(float(baseline_correct_total), float(max(1, n))),
        "intervention_acc": safe_div(float(intervention_correct_total), float(max(1, n))),
        "delta_vs_intervention": safe_div(float(final_correct - intervention_correct_total), float(max(1, n))),
        "source_balanced_utility": float(source_balanced_utility),
        "selected_count": int(route_baseline),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "selected_harm_precision": precision,
        "selected_help_precision": safe_div(float(selected_help), float(max(1, route_baseline))),
        "selected_harm_recall": recall,
        "selected_harm_f1": f1,
        "per_source": per_source,
    }


def selection_key(result: Dict[str, Any], objective: str) -> Sequence[float]:
    if objective == "harm_f1":
        return (
            float(result["selected_harm_f1"]),
            float(result["selected_harm_precision"]),
            float(result["source_balanced_utility"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_precision":
        return (
            float(result["selected_harm_precision"]),
            float(result["selected_harm_recall"]),
            float(result["source_balanced_utility"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_recall":
        return (
            float(result["selected_harm_recall"]),
            float(result["selected_harm_precision"]),
            float(result["source_balanced_utility"]),
            -float(result["baseline_rate"]),
        )
    if objective == "balanced_utility":
        return (
            float(result["source_balanced_utility"]),
            float(result["delta_vs_intervention"]),
            float(result["selected_harm_precision"]),
            -float(result["baseline_rate"]),
        )
    return (
        float(result["final_acc"]),
        float(result["source_balanced_utility"]),
        float(result["selected_harm_precision"]),
        -float(result["baseline_rate"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a net susceptibility pre-gate controller from discovery tables.")
    ap.add_argument("--discovery_table_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--harm_feature_cols", type=str, required=True)
    ap.add_argument("--help_feature_cols", type=str, required=True)
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k_harm", type=int, default=3)
    ap.add_argument("--top_k_help", type=int, default=1)
    ap.add_argument("--lambda_values", type=str, default="0.5,1.0,1.5,2.0")
    ap.add_argument("--tau_objective", type=str, default="balanced_utility", choices=["balanced_utility", "final_acc", "harm_f1", "harm_precision", "harm_recall"])
    ap.add_argument("--min_baseline_rate", type=float, default=0.01)
    ap.add_argument("--max_baseline_rate", type=float, default=0.05)
    ap.add_argument("--min_selected_count", type=int, default=30)
    ap.add_argument("--min_selected_per_source", type=int, default=0)
    args = ap.parse_args()

    harm_features_raw = [x.strip() for x in str(args.harm_feature_cols).split(",") if x.strip()]
    help_features_raw = [x.strip() for x in str(args.help_feature_cols).split(",") if x.strip()]
    lambda_values = [float(x.strip()) for x in str(args.lambda_values).split(",") if x.strip()]

    source_rows: Dict[str, List[Dict[str, str]]] = {}
    all_rows: List[Dict[str, str]] = []
    for path in tqdm(args.discovery_table_csvs, desc="net-load", unit="table"):
        rows = read_csv_rows(path)
        if not rows:
            continue
        source = str(rows[0].get("benchmark", os.path.basename(path)))
        source_rows[source] = rows
        all_rows.extend(rows)

    harm_metrics, selected_harm = select_family(
        source_rows=source_rows,
        features=harm_features_raw,
        target="harm",
        min_feature_auroc=float(args.min_feature_auroc),
        top_k=int(args.top_k_harm),
    )
    help_metrics, selected_help = select_family(
        source_rows=source_rows,
        features=help_features_raw,
        target="help",
        min_feature_auroc=float(args.min_feature_auroc),
        top_k=int(args.top_k_help),
    )
    if not selected_harm:
        raise RuntimeError("No eligible harm-family features survived.")
    if not selected_help:
        raise RuntimeError("No eligible help-family features survived.")

    score_candidates: List[Tuple[float, float, float, float]] = []
    lambda_to_scores: Dict[float, List[float]] = {}
    for lam in lambda_values:
        vals: List[float] = []
        for row in all_rows:
            built = build_net_score(row, selected_harm, selected_help, lam)
            if built is None:
                continue
            _, _, net_score = built
            vals.append(float(net_score))
        lambda_to_scores[lam] = vals

    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for lam in tqdm(lambda_values, desc="net-lambda", unit="lambda"):
        uniq_scores = sorted(set(lambda_to_scores.get(lam, [])))
        tau_grid = [float("-inf")] + uniq_scores + [float("inf")]
        for tau in tqdm(tau_grid, desc=f"net-tau:lam={lam}", unit="tau", leave=False):
            result = evaluate_tau(source_rows, selected_harm, selected_help, lam, tau)
            if float(result["baseline_rate"]) > float(args.max_baseline_rate):
                continue
            if float(result["baseline_rate"]) < float(args.min_baseline_rate):
                continue
            if int(result["selected_count"]) < int(args.min_selected_count):
                continue
            if int(args.min_selected_per_source) > 0:
                ok = True
                for src_stats in result["per_source"].values():
                    if int(src_stats["route_baseline"]) < int(args.min_selected_per_source):
                        ok = False
                        break
                if not ok:
                    continue
            sweep_rows.append(result)
            if best is None or selection_key(result, str(args.tau_objective)) > selection_key(best, str(args.tau_objective)):
                best = result
    if best is None:
        raise RuntimeError("No feasible net-score controller found for the discovery tables.")

    policy = {
        "policy_type": "composite_net_susceptibility_gate",
        "target_label": "harm_vs_help_net",
        "harm_features": selected_harm,
        "help_features": selected_help,
        "lambda_help": float(best["lambda_help"]),
        "tau": float(best["tau"]),
        "tau_objective": str(args.tau_objective),
        "min_baseline_rate": float(args.min_baseline_rate),
        "max_baseline_rate": float(args.max_baseline_rate),
        "min_selected_count": int(args.min_selected_count),
        "min_selected_per_source": int(args.min_selected_per_source),
        "route_policy": "baseline if net_score >= tau else method",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    harm_metrics_csv = os.path.join(args.out_dir, "harm_feature_direction_metrics.csv")
    help_metrics_csv = os.path.join(args.out_dir, "help_feature_direction_metrics.csv")
    harm_candidates_csv = os.path.join(args.out_dir, "harm_selected_features.csv")
    help_candidates_csv = os.path.join(args.out_dir, "help_selected_features.csv")
    sweep_csv = os.path.join(args.out_dir, "tau_sweep.csv")
    policy_json = os.path.join(args.out_dir, "selected_policy.json")
    summary_json = os.path.join(args.out_dir, "summary.json")

    write_csv(harm_metrics_csv, harm_metrics)
    write_csv(help_metrics_csv, help_metrics)
    write_csv(harm_candidates_csv, selected_harm)
    write_csv(help_candidates_csv, selected_help)
    write_csv(sweep_csv, sweep_rows)
    write_json(policy_json, policy)
    summary = {
        "inputs": {
            "discovery_table_csvs": [os.path.abspath(x) for x in args.discovery_table_csvs],
            "harm_feature_cols": harm_features_raw,
            "help_feature_cols": help_features_raw,
            "min_feature_auroc": float(args.min_feature_auroc),
            "top_k_harm": int(args.top_k_harm),
            "top_k_help": int(args.top_k_help),
            "lambda_values": lambda_values,
            "tau_objective": str(args.tau_objective),
            "min_baseline_rate": float(args.min_baseline_rate),
            "max_baseline_rate": float(args.max_baseline_rate),
            "min_selected_count": int(args.min_selected_count),
            "min_selected_per_source": int(args.min_selected_per_source),
        },
        "n_rows": int(len(all_rows)),
        "sources": sorted(source_rows.keys()),
        "selected_harm_features": selected_harm,
        "selected_help_features": selected_help,
        "best_tau": best,
        "outputs": {
            "harm_feature_direction_metrics_csv": os.path.abspath(harm_metrics_csv),
            "help_feature_direction_metrics_csv": os.path.abspath(help_metrics_csv),
            "harm_selected_features_csv": os.path.abspath(harm_candidates_csv),
            "help_selected_features_csv": os.path.abspath(help_candidates_csv),
            "tau_sweep_csv": os.path.abspath(sweep_csv),
            "policy_json": os.path.abspath(policy_json),
        },
    }
    write_json(summary_json, summary)
    print("[saved]", os.path.abspath(harm_metrics_csv))
    print("[saved]", os.path.abspath(help_metrics_csv))
    print("[saved]", os.path.abspath(harm_candidates_csv))
    print("[saved]", os.path.abspath(help_candidates_csv))
    print("[saved]", os.path.abspath(sweep_csv))
    print("[saved]", os.path.abspath(policy_json))
    print("[saved]", os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
