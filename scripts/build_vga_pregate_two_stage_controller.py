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


def select_features(
    source_rows: Dict[str, List[Dict[str, str]]],
    feature_cols: Sequence[str],
    *,
    target_name: str,
    label_fn,
    min_feature_auroc: float,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_source_metrics: List[Dict[str, Any]] = []
    by_feature: Dict[str, List[Dict[str, Any]]] = {}

    for source, rows in tqdm(source_rows.items(), total=len(source_rows), desc=f"select:{target_name}", unit="source"):
        for feat in tqdm(feature_cols, desc=f"{target_name}-features:{source}", unit="feature", leave=False):
            xs: List[float] = []
            ys: List[int] = []
            for row in rows:
                x = maybe_float(row.get(feat))
                y = label_fn(row)
                if x is None or y is None:
                    continue
                xs.append(float(x))
                ys.append(int(y))
            if len(xs) < 2:
                continue
            auc_high = binary_auroc(xs, ys)
            auc_low = binary_auroc([-x for x in xs], ys)
            if auc_high is None or auc_low is None:
                continue
            direction = "high" if auc_high >= auc_low else "low"
            metric = {
                "feature": feat,
                "target": target_name,
                "direction": direction,
                "auroc": max(float(auc_high), float(auc_low)),
                "n": int(len(xs)),
                "n_pos": int(sum(ys)),
                "source": source,
            }
            per_source_metrics.append(metric)
            by_feature.setdefault(feat, []).append(metric)

    all_rows = [row for rows in source_rows.values() for row in rows]
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
            if maybe_float(row.get(feat)) is not None and label_fn(row) is not None
        ]
        candidates.append(
            {
                "feature": feat,
                "target": target_name,
                "direction": direction,
                "mean_auroc": mean_auc,
                "min_auroc": min_auc,
                "mu": mean(oriented_vals),
                "sd": std(oriented_vals),
            }
        )
    candidates.sort(key=lambda r: (-float(r["mean_auroc"]), -float(r["min_auroc"]), str(r["feature"])))
    return per_source_metrics, candidates[: max(1, int(top_k))]


def build_score(row: Dict[str, str], features: Sequence[Dict[str, Any]]) -> Optional[float]:
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


def evaluate_two_stage(
    source_rows: Dict[str, List[Dict[str, str]]],
    p1_features: Sequence[Dict[str, Any]],
    p2_features: Sequence[Dict[str, Any]],
    tau_p1: float,
    tau_p2: float,
) -> Dict[str, Any]:
    total_n = 0
    baseline_count = 0
    method_count = 0
    final_correct = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    oracle_correct_total = 0

    sensitive_true_total = 0
    predicted_sensitive_total = 0
    p1_tp = 0
    p1_fp = 0
    p1_fn = 0

    blocked_harm = 0
    blocked_help = 0
    blocked_neutral = 0
    method_harm = 0
    method_help = 0
    method_neutral = 0
    total_harm = 0
    total_help = 0

    per_source: Dict[str, Dict[str, float]] = {}

    for source, rows in source_rows.items():
        src = {
            "n": 0,
            "baseline_count": 0,
            "method_count": 0,
            "baseline_correct_total": 0,
            "intervention_correct_total": 0,
            "final_correct": 0,
            "oracle_correct_total": 0,
            "sensitive_true_total": 0,
            "predicted_sensitive_total": 0,
            "p1_tp": 0,
            "p1_fp": 0,
            "p1_fn": 0,
            "blocked_harm": 0,
            "blocked_help": 0,
            "blocked_neutral": 0,
            "method_harm": 0,
            "method_help": 0,
            "method_neutral": 0,
            "total_harm": 0,
            "total_help": 0,
        }
        for row in rows:
            p1_score = build_score(row, p1_features)
            p2_score = build_score(row, p2_features)
            if p1_score is None or p2_score is None:
                continue
            src["n"] += 1
            total_n += 1

            baseline_correct = int(maybe_int(row.get("baseline_correct")) or 0)
            intervention_correct = int(maybe_int(row.get("intervention_correct")) or 0)
            oracle_correct = int(maybe_int(row.get("oracle_correct")) or max(baseline_correct, intervention_correct))
            harm = int(maybe_int(row.get("harm")) or 0)
            help_ = int(maybe_int(row.get("help")) or 0)
            sensitive = int(harm or help_)

            src["baseline_correct_total"] += baseline_correct
            src["intervention_correct_total"] += intervention_correct
            src["oracle_correct_total"] += oracle_correct
            baseline_correct_total += baseline_correct
            intervention_correct_total += intervention_correct
            oracle_correct_total += oracle_correct

            src["total_harm"] += harm
            src["total_help"] += help_
            total_harm += harm
            total_help += help_

            src["sensitive_true_total"] += sensitive
            sensitive_true_total += sensitive

            predicted_sensitive = int(p1_score >= tau_p1)
            src["predicted_sensitive_total"] += predicted_sensitive
            predicted_sensitive_total += predicted_sensitive

            if predicted_sensitive and sensitive:
                src["p1_tp"] += 1
                p1_tp += 1
            elif predicted_sensitive and not sensitive:
                src["p1_fp"] += 1
                p1_fp += 1
            elif (not predicted_sensitive) and sensitive:
                src["p1_fn"] += 1
                p1_fn += 1

            # Route VGA only if susceptible and help-leaning.
            run_method = bool(predicted_sensitive and (p2_score < tau_p2))
            if run_method:
                src["method_count"] += 1
                method_count += 1
                src["final_correct"] += intervention_correct
                final_correct += intervention_correct
                if harm:
                    src["method_harm"] += 1
                    method_harm += 1
                elif help_:
                    src["method_help"] += 1
                    method_help += 1
                else:
                    src["method_neutral"] += 1
                    method_neutral += 1
            else:
                src["baseline_count"] += 1
                baseline_count += 1
                src["final_correct"] += baseline_correct
                final_correct += baseline_correct
                if harm:
                    src["blocked_harm"] += 1
                    blocked_harm += 1
                elif help_:
                    src["blocked_help"] += 1
                    blocked_help += 1
                else:
                    src["blocked_neutral"] += 1
                    blocked_neutral += 1

        n_src = int(src["n"])
        src["baseline_rate"] = safe_div(float(src["baseline_count"]), float(max(1, n_src)))
        src["method_rate"] = safe_div(float(src["method_count"]), float(max(1, n_src)))
        src["final_acc"] = safe_div(float(src["final_correct"]), float(max(1, n_src)))
        src["intervention_acc"] = safe_div(float(src["intervention_correct_total"]), float(max(1, n_src)))
        src["delta_vs_intervention"] = safe_div(float(src["final_correct"] - src["intervention_correct_total"]), float(max(1, n_src)))
        src["p1_sensitive_precision"] = safe_div(float(src["p1_tp"]), float(max(1, int(src["predicted_sensitive_total"]))))
        src["p1_sensitive_recall"] = safe_div(float(src["p1_tp"]), float(max(1, int(src["sensitive_true_total"]))))
        src["method_help_precision"] = safe_div(float(src["method_help"]), float(max(1, int(src["method_count"]))))
        src["method_harm_precision"] = safe_div(float(src["method_harm"]), float(max(1, int(src["method_count"]))))
        per_source[source] = src

    p1_prec = safe_div(float(p1_tp), float(max(1, predicted_sensitive_total)))
    p1_rec = safe_div(float(p1_tp), float(max(1, sensitive_true_total)))
    p1_f1 = safe_div(2.0 * p1_prec * p1_rec, p1_prec + p1_rec)
    source_balanced_utility = mean([float(v["delta_vs_intervention"]) for v in per_source.values()]) if per_source else 0.0
    worst_source_utility = min([float(v["delta_vs_intervention"]) for v in per_source.values()]) if per_source else 0.0

    return {
        "tau_p1": float(tau_p1),
        "tau_p2": float(tau_p2),
        "n_eval": int(total_n),
        "baseline_rate": safe_div(float(baseline_count), float(max(1, total_n))),
        "method_rate": safe_div(float(method_count), float(max(1, total_n))),
        "final_acc": safe_div(float(final_correct), float(max(1, total_n))),
        "baseline_acc": safe_div(float(baseline_correct_total), float(max(1, total_n))),
        "intervention_acc": safe_div(float(intervention_correct_total), float(max(1, total_n))),
        "oracle_posthoc_acc": safe_div(float(oracle_correct_total), float(max(1, total_n))),
        "delta_vs_intervention": safe_div(float(final_correct - intervention_correct_total), float(max(1, total_n))),
        "source_balanced_utility": float(source_balanced_utility),
        "worst_source_utility": float(worst_source_utility),
        "p1_predicted_sensitive_count": int(predicted_sensitive_total),
        "p1_predicted_sensitive_rate": safe_div(float(predicted_sensitive_total), float(max(1, total_n))),
        "p1_true_sensitive_count": int(sensitive_true_total),
        "p1_sensitive_precision": p1_prec,
        "p1_sensitive_recall": p1_rec,
        "p1_sensitive_f1": p1_f1,
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "blocked_harm": int(blocked_harm),
        "blocked_help": int(blocked_help),
        "blocked_neutral": int(blocked_neutral),
        "method_harm": int(method_harm),
        "method_help": int(method_help),
        "method_neutral": int(method_neutral),
        "method_help_precision": safe_div(float(method_help), float(max(1, method_count))),
        "method_harm_precision": safe_div(float(method_harm), float(max(1, method_count))),
        "per_source": per_source,
    }


def selection_key(result: Dict[str, Any], objective: str) -> Sequence[float]:
    if objective == "final_acc":
        return (
            float(result["final_acc"]),
            float(result["source_balanced_utility"]),
            float(result["worst_source_utility"]),
            float(result["method_help_precision"]),
            -float(result["method_harm_precision"]),
        )
    if objective == "p1_sensitive_f1":
        return (
            float(result["p1_sensitive_f1"]),
            float(result["source_balanced_utility"]),
            float(result["worst_source_utility"]),
            float(result["method_help_precision"]),
            -float(result["method_harm_precision"]),
        )
    return (
        float(result["source_balanced_utility"]),
        float(result["worst_source_utility"]),
        float(result["delta_vs_intervention"]),
        float(result["method_help_precision"]),
        -float(result["method_harm_precision"]),
    )


def parse_quantiles(text: str) -> List[float]:
    out = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a two-stage pre-intervention VGA controller (P1 susceptible miner + P2 harm/help sign classifier).")
    ap.add_argument("--discovery_table_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--p1_feature_cols", type=str, required=True)
    ap.add_argument("--p2_feature_cols", type=str, required=True)
    ap.add_argument("--min_feature_auroc_p1", type=float, default=0.55)
    ap.add_argument("--min_feature_auroc_p2", type=float, default=0.55)
    ap.add_argument("--top_k_p1", type=int, default=5)
    ap.add_argument("--top_k_p2", type=int, default=3)
    ap.add_argument("--tau_quantiles_p1", type=str, default="0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.95,0.97,0.98,0.99")
    ap.add_argument("--tau_quantiles_p2", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--objective", type=str, default="balanced_utility", choices=["balanced_utility", "final_acc", "p1_sensitive_f1"])
    ap.add_argument("--min_sensitive_rate", type=float, default=0.0)
    ap.add_argument("--max_sensitive_rate", type=float, default=1.0)
    ap.add_argument("--min_sensitive_count", type=int, default=0)
    args = ap.parse_args()

    p1_feature_cols = [x.strip() for x in str(args.p1_feature_cols).split(",") if x.strip()]
    p2_feature_cols = [x.strip() for x in str(args.p2_feature_cols).split(",") if x.strip()]
    q1 = parse_quantiles(args.tau_quantiles_p1)
    q2 = parse_quantiles(args.tau_quantiles_p2)

    source_rows: Dict[str, List[Dict[str, str]]] = {}
    all_rows: List[Dict[str, str]] = []
    for path in tqdm(args.discovery_table_csvs, desc="two-stage-load", unit="table"):
        rows = read_csv_rows(path)
        if not rows:
            continue
        source = str(rows[0].get("benchmark", os.path.basename(path)))
        source_rows[source] = rows
        all_rows.extend(rows)

    def p1_label(row: Dict[str, str]) -> Optional[int]:
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        return int(harm or help_)

    def p2_label(row: Dict[str, str]) -> Optional[int]:
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        if not (harm or help_):
            return None
        return int(harm)

    p1_metrics, p1_selected = select_features(
        source_rows,
        p1_feature_cols,
        target_name="sensitive",
        label_fn=p1_label,
        min_feature_auroc=float(args.min_feature_auroc_p1),
        top_k=int(args.top_k_p1),
    )
    p2_metrics, p2_selected = select_features(
        source_rows,
        p2_feature_cols,
        target_name="harm_vs_help",
        label_fn=p2_label,
        min_feature_auroc=float(args.min_feature_auroc_p2),
        top_k=int(args.top_k_p2),
    )
    if not p1_selected:
        raise RuntimeError("No eligible P1 sensitive features survived.")
    if not p2_selected:
        raise RuntimeError("No eligible P2 sign features survived.")

    p1_scores = [build_score(row, p1_selected) for row in all_rows]
    p2_scores = [build_score(row, p2_selected) for row in all_rows if p2_label(row) is not None]
    tau_grid_p1 = quantiles_to_thresholds([x for x in p1_scores if x is not None], q1)
    tau_grid_p2 = quantiles_to_thresholds([x for x in p2_scores if x is not None], q2)

    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for tau_p1 in tqdm(tau_grid_p1, desc="two-stage-p1", unit="tau"):
        for tau_p2 in tqdm(tau_grid_p2, desc=f"two-stage-p2@{tau_p1:.3f}", unit="tau", leave=False):
            result = evaluate_two_stage(source_rows, p1_selected, p2_selected, tau_p1, tau_p2)
            if float(result["p1_predicted_sensitive_rate"]) < float(args.min_sensitive_rate):
                continue
            if float(result["p1_predicted_sensitive_rate"]) > float(args.max_sensitive_rate):
                continue
            if int(result["p1_predicted_sensitive_count"]) < int(args.min_sensitive_count):
                continue
            sweep_rows.append(result)
            if best is None or selection_key(result, str(args.objective)) > selection_key(best, str(args.objective)):
                best = result

    if best is None:
        raise RuntimeError("No feasible two-stage controller found for the discovery tables.")

    policy = {
        "policy_type": "two_stage_sensitive_then_sign",
        "route_policy": "run VGA only if p1_sensitive_score >= tau_p1 and p2_harm_score < tau_p2",
        "p1": {
            "target": "sensitive = harm or help",
            "features": p1_selected,
            "tau": float(best["tau_p1"]),
        },
        "p2": {
            "target": "harm vs help within sensitive pool",
            "features": p2_selected,
            "tau": float(best["tau_p2"]),
        },
        "objective": str(args.objective),
        "min_sensitive_rate": float(args.min_sensitive_rate),
        "max_sensitive_rate": float(args.max_sensitive_rate),
        "min_sensitive_count": int(args.min_sensitive_count),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    p1_metrics_csv = os.path.join(args.out_dir, "p1_feature_direction_metrics.csv")
    p2_metrics_csv = os.path.join(args.out_dir, "p2_feature_direction_metrics.csv")
    p1_selected_csv = os.path.join(args.out_dir, "p1_selected_features.csv")
    p2_selected_csv = os.path.join(args.out_dir, "p2_selected_features.csv")
    sweep_csv = os.path.join(args.out_dir, "tau_sweep.csv")
    policy_json = os.path.join(args.out_dir, "selected_policy.json")
    summary_json = os.path.join(args.out_dir, "summary.json")

    write_csv(p1_metrics_csv, p1_metrics)
    write_csv(p2_metrics_csv, p2_metrics)
    write_csv(p1_selected_csv, p1_selected)
    write_csv(p2_selected_csv, p2_selected)
    write_csv(sweep_csv, sweep_rows)
    write_json(policy_json, policy)
    write_json(
        summary_json,
        {
            "inputs": {
                "discovery_table_csvs": [os.path.abspath(x) for x in args.discovery_table_csvs],
                "p1_feature_cols": p1_feature_cols,
                "p2_feature_cols": p2_feature_cols,
                "min_feature_auroc_p1": float(args.min_feature_auroc_p1),
                "min_feature_auroc_p2": float(args.min_feature_auroc_p2),
                "top_k_p1": int(args.top_k_p1),
                "top_k_p2": int(args.top_k_p2),
                "tau_quantiles_p1": q1,
                "tau_quantiles_p2": q2,
                "objective": str(args.objective),
                "min_sensitive_rate": float(args.min_sensitive_rate),
                "max_sensitive_rate": float(args.max_sensitive_rate),
                "min_sensitive_count": int(args.min_sensitive_count),
            },
            "n_rows": int(len(all_rows)),
            "sources": sorted(source_rows.keys()),
            "selected_p1_features": p1_selected,
            "selected_p2_features": p2_selected,
            "best_tau": best,
            "outputs": {
                "p1_feature_direction_metrics_csv": os.path.abspath(p1_metrics_csv),
                "p2_feature_direction_metrics_csv": os.path.abspath(p2_metrics_csv),
                "p1_selected_features_csv": os.path.abspath(p1_selected_csv),
                "p2_selected_features_csv": os.path.abspath(p2_selected_csv),
                "tau_sweep_csv": os.path.abspath(sweep_csv),
                "policy_json": os.path.abspath(policy_json),
            },
        },
    )
    print("[saved]", os.path.abspath(p1_metrics_csv))
    print("[saved]", os.path.abspath(p2_metrics_csv))
    print("[saved]", os.path.abspath(p1_selected_csv))
    print("[saved]", os.path.abspath(p2_selected_csv))
    print("[saved]", os.path.abspath(sweep_csv))
    print("[saved]", os.path.abspath(policy_json))
    print("[saved]", os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
