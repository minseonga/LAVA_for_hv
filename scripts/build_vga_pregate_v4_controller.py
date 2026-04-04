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


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


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


def build_z_vector(row: Dict[str, str], features: Sequence[Dict[str, Any]]) -> Optional[List[float]]:
    xs: List[float] = []
    for feat in features:
        raw = maybe_float(row.get(str(feat["feature"])))
        if raw is None:
            return None
        oriented = oriented_value(raw, str(feat["direction"]))
        mu = float(feat["mu"])
        sd = max(float(feat["sd"]), 1e-6)
        xs.append((oriented - mu) / sd)
    return xs


def fit_monotone_logistic(
    rows: Sequence[Dict[str, str]],
    target: str,
    features: Sequence[Dict[str, Any]],
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> Dict[str, Any]:
    xs: List[List[float]] = []
    ys: List[int] = []
    for row in rows:
        x = build_z_vector(row, features)
        y = maybe_int(row.get(target))
        if x is None or y not in {0, 1}:
            continue
        xs.append(x)
        ys.append(int(y))
    if not xs:
        raise RuntimeError(f"No usable rows for monotone logistic target={target}.")

    n = len(xs)
    d = len(xs[0])
    pos = sum(ys)
    neg = n - pos
    if pos == 0 or neg == 0:
        raise RuntimeError(f"Degenerate labels for monotone logistic target={target}.")

    bias = math.log((pos + 1e-3) / (neg + 1e-3))
    weights = [0.1 for _ in range(d)]
    pos_w = float(n) / float(2 * pos)
    neg_w = float(n) / float(2 * neg)

    for _ in tqdm(range(max(1, int(epochs))), desc=f"fit:{target}", unit="epoch", leave=False):
        grad_b = 0.0
        grad_w = [0.0 for _ in range(d)]
        total_w = 0.0
        for x, y in zip(xs, ys):
            logit = bias + sum(w * xi for w, xi in zip(weights, x))
            prob = sigmoid(max(min(logit, 30.0), -30.0))
            sample_w = pos_w if y == 1 else neg_w
            diff = (prob - float(y)) * sample_w
            grad_b += diff
            for i, xi in enumerate(x):
                grad_w[i] += diff * xi
            total_w += sample_w
        denom = max(total_w, 1e-6)
        grad_b /= denom
        bias -= float(lr) * grad_b
        for i in range(d):
            grad_w[i] = grad_w[i] / denom + float(l2) * weights[i]
            weights[i] = max(0.0, weights[i] - float(lr) * grad_w[i])

    probs: List[float] = []
    for x in xs:
        probs.append(sigmoid(bias + sum(w * xi for w, xi in zip(weights, x))))
    auroc = binary_auroc(probs, ys)
    return {
        "target": target,
        "features": list(features),
        "weights": [float(w) for w in weights],
        "bias": float(bias),
        "train_auroc": None if auroc is None else float(auroc),
        "n_train": int(n),
        "n_pos": int(pos),
        "fit_type": "monotone_logistic",
    }


def score_head(row: Dict[str, str], model: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    z = build_z_vector(row, list(model["features"]))
    if z is None:
        return None
    bias = float(model["bias"])
    weights = [float(x) for x in list(model["weights"])]
    logit = bias + sum(w * zi for w, zi in zip(weights, z))
    prob = sigmoid(max(min(logit, 30.0), -30.0))
    return float(logit), float(prob)


def build_apply_score(row: Dict[str, str], help_model: Dict[str, Any], harm_model: Dict[str, Any], lam: float) -> Optional[Tuple[float, float, float, float, float]]:
    help_pack = score_head(row, help_model)
    harm_pack = score_head(row, harm_model)
    if help_pack is None or harm_pack is None:
        return None
    help_logit, help_prob = help_pack
    harm_logit, harm_prob = harm_pack
    apply_score = float(help_prob - float(lam) * harm_prob)
    return float(help_logit), float(help_prob), float(harm_logit), float(harm_prob), float(apply_score)


def evaluate_tau(
    source_rows: Dict[str, List[Dict[str, str]]],
    help_model: Dict[str, Any],
    harm_model: Dict[str, Any],
    lam: float,
    tau: float,
) -> Dict[str, Any]:
    n = 0
    method_count = 0
    final_correct = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    oracle_correct_total = 0
    applied_harm = 0
    applied_help = 0
    applied_neutral = 0
    total_harm = 0
    total_help = 0
    per_source: Dict[str, Dict[str, float]] = {}
    overall_harm_help_scores: List[float] = []
    overall_harm_help_labels_harm: List[int] = []

    for source, rows in source_rows.items():
        src = {
            "n": 0,
            "method_count": 0,
            "baseline_correct_total": 0,
            "intervention_correct_total": 0,
            "oracle_correct_total": 0,
            "final_correct": 0,
            "applied_harm": 0,
            "applied_help": 0,
            "applied_neutral": 0,
            "total_harm": 0,
            "total_help": 0,
        }
        src_scores: List[float] = []
        src_labels_harm: List[int] = []
        for row in rows:
            built = build_apply_score(row, help_model, harm_model, lam)
            if built is None:
                continue
            _, _, _, _, apply_score = built
            baseline_correct = int(maybe_int(row.get("baseline_correct")) or 0)
            intervention_correct = int(maybe_int(row.get("intervention_correct")) or 0)
            oracle_correct = int(maybe_int(row.get("oracle_correct")) or max(baseline_correct, intervention_correct))
            harm = int(maybe_int(row.get("harm")) or 0)
            help_ = int(maybe_int(row.get("help")) or 0)
            use_method = bool(apply_score > tau)

            src["n"] += 1
            n += 1
            baseline_correct_total += baseline_correct
            intervention_correct_total += intervention_correct
            oracle_correct_total += oracle_correct
            total_harm += harm
            total_help += help_
            src["baseline_correct_total"] += baseline_correct
            src["intervention_correct_total"] += intervention_correct
            src["oracle_correct_total"] += oracle_correct
            src["total_harm"] += harm
            src["total_help"] += help_
            if harm or help_:
                src_scores.append(float(-apply_score))
                src_labels_harm.append(int(harm))
                overall_harm_help_scores.append(float(-apply_score))
                overall_harm_help_labels_harm.append(int(harm))

            if use_method:
                method_count += 1
                src["method_count"] += 1
                applied_harm += harm
                applied_help += help_
                applied_neutral += int((harm == 0) and (help_ == 0))
                src["applied_harm"] += harm
                src["applied_help"] += help_
                src["applied_neutral"] += int((harm == 0) and (help_ == 0))
                final_correct += intervention_correct
                src["final_correct"] += intervention_correct
            else:
                final_correct += baseline_correct
                src["final_correct"] += baseline_correct

        src_n = int(src["n"])
        src_method = int(src["method_count"])
        src_veto = int(src_n - src_method)
        src_total_neutral = int(src_n - int(src["total_harm"]) - int(src["total_help"]))
        src_veto_harm = int(int(src["total_harm"]) - int(src["applied_harm"]))
        src_veto_help = int(int(src["total_help"]) - int(src["applied_help"]))
        src_veto_neutral = int(src_total_neutral - int(src["applied_neutral"]))
        src["method_rate"] = safe_div(float(src_method), float(max(1, src_n)))
        src["baseline_rate"] = float(1.0 - float(src["method_rate"]))
        src["final_acc"] = safe_div(float(src["final_correct"]), float(max(1, src_n)))
        src["baseline_acc"] = safe_div(float(src["baseline_correct_total"]), float(max(1, src_n)))
        src["intervention_acc"] = safe_div(float(src["intervention_correct_total"]), float(max(1, src_n)))
        src["oracle_posthoc_acc"] = safe_div(float(src["oracle_correct_total"]), float(max(1, src_n)))
        src["delta_vs_baseline"] = safe_div(float(src["final_correct"] - src["baseline_correct_total"]), float(max(1, src_n)))
        src["delta_vs_intervention"] = safe_div(float(src["final_correct"] - src["intervention_correct_total"]), float(max(1, src_n)))
        src["veto_count"] = src_veto
        src["veto_harm"] = src_veto_harm
        src["veto_help"] = src_veto_help
        src["veto_neutral"] = src_veto_neutral
        src["applied_help_precision"] = safe_div(float(src["applied_help"]), float(max(1, src_method)))
        src["applied_harm_precision"] = safe_div(float(src["applied_harm"]), float(max(1, src_method)))
        src["applied_help_recall"] = safe_div(float(src["applied_help"]), float(max(1, int(src["total_help"]))))
        src["applied_harm_recall"] = safe_div(float(src["applied_harm"]), float(max(1, int(src["total_harm"]))))
        src["veto_harm_precision"] = safe_div(float(src_veto_harm), float(max(1, src_veto)))
        src["veto_help_precision"] = safe_div(float(src_veto_help), float(max(1, src_veto)))
        src["veto_harm_recall"] = safe_div(float(src_veto_harm), float(max(1, int(src["total_harm"]))))
        src_auc = binary_auroc(src_scores, src_labels_harm)
        src["harm_vs_help_auroc"] = None if src_auc is None else float(src_auc)
        src["help_vs_harm_auroc"] = None if src_auc is None else float(src_auc)
        per_source[source] = src

    method_rate = safe_div(float(method_count), float(max(1, n)))
    baseline_rate = float(1.0 - float(method_rate))
    veto_count = int(n - method_count)
    total_neutral = int(n - total_harm - total_help)
    veto_harm = int(total_harm - applied_harm)
    veto_help = int(total_help - applied_help)
    veto_neutral = int(total_neutral - applied_neutral)
    source_balanced_utility = mean([float(v["delta_vs_baseline"]) for v in per_source.values()]) if per_source else 0.0
    worst_source_utility = min((float(v["delta_vs_baseline"]) for v in per_source.values()), default=0.0)
    overall_auc = binary_auroc(overall_harm_help_scores, overall_harm_help_labels_harm)

    return {
        "lambda_harm": float(lam),
        "tau": float(tau),
        "n_eval": int(n),
        "method_rate": method_rate,
        "baseline_rate": baseline_rate,
        "final_acc": safe_div(float(final_correct), float(max(1, n))),
        "baseline_acc": safe_div(float(baseline_correct_total), float(max(1, n))),
        "intervention_acc": safe_div(float(intervention_correct_total), float(max(1, n))),
        "oracle_posthoc_acc": safe_div(float(oracle_correct_total), float(max(1, n))),
        "delta_vs_baseline": safe_div(float(final_correct - baseline_correct_total), float(max(1, n))),
        "delta_vs_intervention": safe_div(float(final_correct - intervention_correct_total), float(max(1, n))),
        "gap_to_oracle_posthoc": safe_div(float(oracle_correct_total - final_correct), float(max(1, n))),
        "source_balanced_utility": float(source_balanced_utility),
        "worst_source_utility": float(worst_source_utility),
        "selected_count": int(method_count),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "applied_harm": int(applied_harm),
        "applied_help": int(applied_help),
        "applied_neutral": int(applied_neutral),
        "applied_help_precision": safe_div(float(applied_help), float(max(1, method_count))),
        "applied_harm_precision": safe_div(float(applied_harm), float(max(1, method_count))),
        "applied_help_recall": safe_div(float(applied_help), float(max(1, total_help))),
        "applied_harm_recall": safe_div(float(applied_harm), float(max(1, total_harm))),
        "harm_vs_help_auroc": None if overall_auc is None else float(overall_auc),
        "help_vs_harm_auroc": None if overall_auc is None else float(overall_auc),
        "veto_count": int(veto_count),
        "veto_harm": int(veto_harm),
        "veto_help": int(veto_help),
        "veto_neutral": int(veto_neutral),
        "veto_harm_precision": safe_div(float(veto_harm), float(max(1, veto_count))),
        "veto_help_precision": safe_div(float(veto_help), float(max(1, veto_count))),
        "veto_harm_recall": safe_div(float(veto_harm), float(max(1, total_harm))),
        "per_source": per_source,
    }


def selection_key(result: Dict[str, Any], objective: str) -> Sequence[float]:
    if objective == "help_precision":
        return (
            float(result["applied_help_precision"]) - float(result["applied_harm_precision"]),
            float(result["source_balanced_utility"]),
            float(result["worst_source_utility"]),
            float(result["harm_vs_help_auroc"] or 0.0),
            float(result["final_acc"]),
        )
    if objective == "final_acc":
        return (
            float(result["final_acc"]),
            float(result["source_balanced_utility"]),
            float(result["worst_source_utility"]),
            float(result["harm_vs_help_auroc"] or 0.0),
        )
    return (
        float(result["source_balanced_utility"]),
        float(result["worst_source_utility"]),
        float(result["harm_vs_help_auroc"] or 0.0),
        float(result["applied_help_precision"]) - float(result["applied_harm_precision"]),
        float(result["final_acc"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified pre-gating v4 controller with monotone logistic help/harm heads.")
    ap.add_argument("--discovery_table_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--help_feature_cols", type=str, required=True)
    ap.add_argument("--harm_feature_cols", type=str, required=True)
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k_help", type=int, default=3)
    ap.add_argument("--top_k_harm", type=int, default=3)
    ap.add_argument("--lambda_values", type=str, default="0.5,1.0,1.5,2.0")
    ap.add_argument("--tau_quantiles", type=str, default="0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99")
    ap.add_argument("--tau_objective", type=str, default="balanced_utility", choices=["balanced_utility", "final_acc", "help_precision"])
    ap.add_argument("--min_method_rate", type=float, default=0.0)
    ap.add_argument("--max_method_rate", type=float, default=1.0)
    ap.add_argument("--min_selected_count", type=int, default=0)
    ap.add_argument("--fit_epochs", type=int, default=300)
    ap.add_argument("--fit_lr", type=float, default=0.05)
    ap.add_argument("--fit_l2", type=float, default=1e-3)
    args = ap.parse_args()

    help_features_raw = [x.strip() for x in str(args.help_feature_cols).split(",") if x.strip()]
    harm_features_raw = [x.strip() for x in str(args.harm_feature_cols).split(",") if x.strip()]
    lambda_values = [float(x.strip()) for x in str(args.lambda_values).split(",") if x.strip()]
    tau_quantiles = [float(x.strip()) for x in str(args.tau_quantiles).split(",") if x.strip()]

    source_rows: Dict[str, List[Dict[str, str]]] = {}
    all_rows: List[Dict[str, str]] = []
    for path in tqdm(args.discovery_table_csvs, desc="v4-load", unit="table"):
        rows = read_csv_rows(path)
        if not rows:
            continue
        source = str(rows[0].get("benchmark", os.path.basename(path)))
        source_rows[source] = rows
        all_rows.extend(rows)

    help_metrics, selected_help = select_family(source_rows, help_features_raw, "help", float(args.min_feature_auroc), int(args.top_k_help))
    harm_metrics, selected_harm = select_family(source_rows, harm_features_raw, "harm", float(args.min_feature_auroc), int(args.top_k_harm))
    if not selected_help:
        raise RuntimeError("No eligible help-family features survived.")
    if not selected_harm:
        raise RuntimeError("No eligible harm-family features survived.")

    help_model = fit_monotone_logistic(
        all_rows,
        "help",
        selected_help,
        epochs=int(args.fit_epochs),
        lr=float(args.fit_lr),
        l2=float(args.fit_l2),
    )
    harm_model = fit_monotone_logistic(
        all_rows,
        "harm",
        selected_harm,
        epochs=int(args.fit_epochs),
        lr=float(args.fit_lr),
        l2=float(args.fit_l2),
    )

    lambda_to_scores: Dict[float, List[float]] = {}
    for lam in lambda_values:
        vals: List[float] = []
        for row in all_rows:
            built = build_apply_score(row, help_model, harm_model, lam)
            if built is None:
                continue
            vals.append(float(built[-1]))
        lambda_to_scores[lam] = vals

    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for lam in tqdm(lambda_values, desc="v4-lambda", unit="lambda"):
        tau_grid = quantiles_to_thresholds(lambda_to_scores.get(lam, []), tau_quantiles)
        for tau in tqdm(tau_grid, desc=f"v4-tau:lam={lam}", unit="tau", leave=False):
            result = evaluate_tau(source_rows, help_model, harm_model, lam, tau)
            if float(result["method_rate"]) < float(args.min_method_rate):
                continue
            if float(result["method_rate"]) > float(args.max_method_rate):
                continue
            if int(result["selected_count"]) < int(args.min_selected_count):
                continue
            sweep_rows.append(result)
            if best is None or selection_key(result, str(args.tau_objective)) > selection_key(best, str(args.tau_objective)):
                best = result
    if best is None:
        raise RuntimeError("No feasible v4 controller found for the discovery tables.")

    policy = {
        "policy_type": "baseline_default_optin_v4_monotone_logistic",
        "target_label": "help_minus_harm_apply_score",
        "help_model": help_model,
        "harm_model": harm_model,
        "lambda_harm": float(best["lambda_harm"]),
        "tau": float(best["tau"]),
        "tau_objective": str(args.tau_objective),
        "min_method_rate": float(args.min_method_rate),
        "max_method_rate": float(args.max_method_rate),
        "min_selected_count": int(args.min_selected_count),
        "route_policy": "run method if apply_score > tau else baseline",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    help_metrics_csv = os.path.join(args.out_dir, "help_feature_direction_metrics.csv")
    harm_metrics_csv = os.path.join(args.out_dir, "harm_feature_direction_metrics.csv")
    help_selected_csv = os.path.join(args.out_dir, "help_selected_features.csv")
    harm_selected_csv = os.path.join(args.out_dir, "harm_selected_features.csv")
    sweep_csv = os.path.join(args.out_dir, "tau_sweep.csv")
    policy_json = os.path.join(args.out_dir, "selected_policy.json")
    summary_json = os.path.join(args.out_dir, "summary.json")

    write_csv(help_metrics_csv, help_metrics)
    write_csv(harm_metrics_csv, harm_metrics)
    write_csv(help_selected_csv, selected_help)
    write_csv(harm_selected_csv, selected_harm)
    write_csv(sweep_csv, sweep_rows)
    write_json(policy_json, policy)
    summary = {
        "inputs": {
            "discovery_table_csvs": [os.path.abspath(x) for x in args.discovery_table_csvs],
            "help_feature_cols": help_features_raw,
            "harm_feature_cols": harm_features_raw,
            "min_feature_auroc": float(args.min_feature_auroc),
            "top_k_help": int(args.top_k_help),
            "top_k_harm": int(args.top_k_harm),
            "lambda_values": lambda_values,
            "tau_quantiles": tau_quantiles,
            "tau_objective": str(args.tau_objective),
            "min_method_rate": float(args.min_method_rate),
            "max_method_rate": float(args.max_method_rate),
            "min_selected_count": int(args.min_selected_count),
            "fit_epochs": int(args.fit_epochs),
            "fit_lr": float(args.fit_lr),
            "fit_l2": float(args.fit_l2),
        },
        "n_rows": int(len(all_rows)),
        "sources": sorted(source_rows.keys()),
        "selected_help_features": selected_help,
        "selected_harm_features": selected_harm,
        "help_model": help_model,
        "harm_model": harm_model,
        "best_tau": best,
        "outputs": {
            "help_feature_direction_metrics_csv": os.path.abspath(help_metrics_csv),
            "harm_feature_direction_metrics_csv": os.path.abspath(harm_metrics_csv),
            "help_selected_features_csv": os.path.abspath(help_selected_csv),
            "harm_selected_features_csv": os.path.abspath(harm_selected_csv),
            "tau_sweep_csv": os.path.abspath(sweep_csv),
            "policy_json": os.path.abspath(policy_json),
        },
    }
    write_json(summary_json, summary)
    print("[saved]", os.path.abspath(help_metrics_csv))
    print("[saved]", os.path.abspath(harm_metrics_csv))
    print("[saved]", os.path.abspath(help_selected_csv))
    print("[saved]", os.path.abspath(harm_selected_csv))
    print("[saved]", os.path.abspath(sweep_csv))
    print("[saved]", os.path.abspath(policy_json))
    print("[saved]", os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
