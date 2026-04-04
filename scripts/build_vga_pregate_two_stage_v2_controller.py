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


def evaluate_feature(rows: Sequence[Dict[str, str]], feature: str, target_name: str, label_fn) -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = maybe_float(row.get(feature))
        y = label_fn(row)
        if x is None or y is None:
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
        "target": target_name,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
    }


def select_family(
    source_rows: Dict[str, List[Dict[str, str]]],
    features: Sequence[str],
    *,
    target_name: str,
    label_fn,
    min_feature_auroc: float,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_source_metrics: List[Dict[str, Any]] = []
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    all_rows = [row for rows in source_rows.values() for row in rows]
    for source, rows in tqdm(source_rows.items(), total=len(source_rows), desc=f"select:{target_name}", unit="source"):
        for feat in tqdm(features, desc=f"{target_name}-features:{source}", unit="feature", leave=False):
            result = evaluate_feature(rows, feat, target_name, label_fn)
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


def build_base_z(row: Dict[str, str], features: Sequence[Dict[str, Any]]) -> Optional[List[float]]:
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


def build_interaction_pairs(size: int, mode: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if mode == "none":
        return pairs
    for i in range(size):
        for j in range(i + 1, size):
            pairs.append((i, j))
    return pairs


def expand_with_pairs(zs: Sequence[float], pairs: Sequence[Tuple[int, int]]) -> List[float]:
    out = list(zs)
    for i, j in pairs:
        out.append(float(zs[i] * zs[j]))
    return out


def fit_nonnegative_logistic(
    rows: Sequence[Dict[str, str]],
    *,
    label_fn,
    features: Sequence[Dict[str, Any]],
    pair_mode: str,
    epochs: int,
    lr: float,
    l2: float,
    target_name: str,
) -> Dict[str, Any]:
    pairs = build_interaction_pairs(len(features), pair_mode)
    xs: List[List[float]] = []
    ys: List[int] = []
    for row in rows:
        zs = build_base_z(row, features)
        y = label_fn(row)
        if zs is None or y is None:
            continue
        xs.append(expand_with_pairs(zs, pairs))
        ys.append(int(y))
    if not xs:
        raise RuntimeError(f"No usable rows for target={target_name}")
    n = len(xs)
    d = len(xs[0])
    pos = sum(ys)
    neg = n - pos
    if pos == 0 or neg == 0:
        raise RuntimeError(f"Degenerate labels for target={target_name}")

    bias = math.log((pos + 1e-3) / (neg + 1e-3))
    weights = [0.1 for _ in range(d)]
    pos_w = float(n) / float(2 * pos)
    neg_w = float(n) / float(2 * neg)

    for _ in tqdm(range(max(1, int(epochs))), desc=f"fit:{target_name}", unit="epoch", leave=False):
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
    auc = binary_auroc(probs, ys)
    pair_names = [f"{features[i]['feature']}*{features[j]['feature']}" for i, j in pairs]
    return {
        "target": target_name,
        "features": list(features),
        "pair_mode": pair_mode,
        "pair_names": pair_names,
        "weights": [float(w) for w in weights],
        "bias": float(bias),
        "train_auroc": None if auc is None else float(auc),
        "n_train": int(n),
        "n_pos": int(pos),
        "fit_type": "nonnegative_logistic",
    }


def score_model(row: Dict[str, str], model: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    zs = build_base_z(row, list(model["features"]))
    if zs is None:
        return None
    pair_mode = str(model.get("pair_mode", "none"))
    pairs = build_interaction_pairs(len(zs), pair_mode)
    x = expand_with_pairs(zs, pairs)
    bias = float(model["bias"])
    weights = [float(w) for w in list(model["weights"])]
    logit = bias + sum(w * xi for w, xi in zip(weights, x))
    prob = sigmoid(max(min(logit, 30.0), -30.0))
    return float(logit), float(prob)


def label_sensitive(row: Dict[str, str]) -> Optional[int]:
    harm = int(maybe_int(row.get("harm")) or 0)
    help_ = int(maybe_int(row.get("help")) or 0)
    return int(harm or help_)


def label_harm_vs_help(row: Dict[str, str]) -> Optional[int]:
    harm = int(maybe_int(row.get("harm")) or 0)
    help_ = int(maybe_int(row.get("help")) or 0)
    if not (harm or help_):
        return None
    return int(harm)


def evaluate_two_stage(
    source_rows: Dict[str, List[Dict[str, str]]],
    p1_model: Dict[str, Any],
    p2_model: Dict[str, Any],
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

    veto_harm = 0
    veto_help = 0
    veto_neutral = 0
    method_harm = 0
    method_help = 0
    method_neutral = 0
    total_harm = 0
    total_help = 0
    per_source: Dict[str, Dict[str, float]] = {}
    overall_p2_scores: List[float] = []
    overall_p2_labels_harm: List[int] = []

    for source, rows in source_rows.items():
        src = {
            "n": 0,
            "baseline_count": 0,
            "method_count": 0,
            "baseline_correct_total": 0,
            "intervention_correct_total": 0,
            "oracle_correct_total": 0,
            "final_correct": 0,
            "sensitive_true_total": 0,
            "predicted_sensitive_total": 0,
            "p1_tp": 0,
            "p1_fp": 0,
            "p1_fn": 0,
            "veto_harm": 0,
            "veto_help": 0,
            "veto_neutral": 0,
            "method_harm": 0,
            "method_help": 0,
            "method_neutral": 0,
            "total_harm": 0,
            "total_help": 0,
        }
        src_p2_scores: List[float] = []
        src_p2_labels_harm: List[int] = []
        for row in rows:
            p1_pack = score_model(row, p1_model)
            p2_pack = score_model(row, p2_model)
            if p1_pack is None or p2_pack is None:
                continue
            _, p1_prob = p1_pack
            _, p2_harm_prob = p2_pack
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
            if sensitive:
                src_p2_scores.append(float(p2_harm_prob))
                src_p2_labels_harm.append(int(harm))
                overall_p2_scores.append(float(p2_harm_prob))
                overall_p2_labels_harm.append(int(harm))

            predicted_sensitive = int(p1_prob >= tau_p1)
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

            run_method = bool(predicted_sensitive and (p2_harm_prob < tau_p2))
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
                    src["veto_harm"] += 1
                    veto_harm += 1
                elif help_:
                    src["veto_help"] += 1
                    veto_help += 1
                else:
                    src["veto_neutral"] += 1
                    veto_neutral += 1

        n_src = int(src["n"])
        src["baseline_rate"] = safe_div(float(src["baseline_count"]), float(max(1, n_src)))
        src["method_rate"] = safe_div(float(src["method_count"]), float(max(1, n_src)))
        src["final_acc"] = safe_div(float(src["final_correct"]), float(max(1, n_src)))
        src["baseline_acc"] = safe_div(float(src["baseline_correct_total"]), float(max(1, n_src)))
        src["intervention_acc"] = safe_div(float(src["intervention_correct_total"]), float(max(1, n_src)))
        src["oracle_posthoc_acc"] = safe_div(float(src["oracle_correct_total"]), float(max(1, n_src)))
        src["delta_vs_baseline"] = safe_div(float(src["final_correct"] - src["baseline_correct_total"]), float(max(1, n_src)))
        src["delta_vs_intervention"] = safe_div(float(src["final_correct"] - src["intervention_correct_total"]), float(max(1, n_src)))
        src["p1_sensitive_precision"] = safe_div(float(src["p1_tp"]), float(max(1, int(src["predicted_sensitive_total"]))))
        src["p1_sensitive_recall"] = safe_div(float(src["p1_tp"]), float(max(1, int(src["sensitive_true_total"]))))
        src["method_help_precision"] = safe_div(float(src["method_help"]), float(max(1, int(src["method_count"]))))
        src["method_harm_precision"] = safe_div(float(src["method_harm"]), float(max(1, int(src["method_count"]))))
        src["veto_count"] = int(src["baseline_count"])
        src["veto_harm_precision"] = safe_div(float(src["veto_harm"]), float(max(1, int(src["baseline_count"]))))
        src["veto_help_precision"] = safe_div(float(src["veto_help"]), float(max(1, int(src["baseline_count"]))))
        src["veto_harm_recall"] = safe_div(float(src["veto_harm"]), float(max(1, int(src["total_harm"]))))
        src_auc = binary_auroc(src_p2_scores, src_p2_labels_harm)
        src["p2_harm_vs_help_auroc"] = None if src_auc is None else float(src_auc)
        per_source[source] = src

    p1_prec = safe_div(float(p1_tp), float(max(1, predicted_sensitive_total)))
    p1_rec = safe_div(float(p1_tp), float(max(1, sensitive_true_total)))
    p1_f1 = safe_div(2.0 * p1_prec * p1_rec, p1_prec + p1_rec)
    source_balanced_utility = mean([float(v["delta_vs_intervention"]) for v in per_source.values()]) if per_source else 0.0
    worst_source_utility = min((float(v["delta_vs_intervention"]) for v in per_source.values()), default=0.0)
    p2_auc = binary_auroc(overall_p2_scores, overall_p2_labels_harm)

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
        "delta_vs_baseline": safe_div(float(final_correct - baseline_correct_total), float(max(1, total_n))),
        "delta_vs_intervention": safe_div(float(final_correct - intervention_correct_total), float(max(1, total_n))),
        "source_balanced_utility": float(source_balanced_utility),
        "worst_source_utility": float(worst_source_utility),
        "p1_predicted_sensitive_count": int(predicted_sensitive_total),
        "p1_predicted_sensitive_rate": safe_div(float(predicted_sensitive_total), float(max(1, total_n))),
        "p1_true_sensitive_count": int(sensitive_true_total),
        "p1_sensitive_precision": p1_prec,
        "p1_sensitive_recall": p1_rec,
        "p1_sensitive_f1": p1_f1,
        "p2_harm_vs_help_auroc": None if p2_auc is None else float(p2_auc),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "veto_count": int(baseline_count),
        "veto_harm": int(veto_harm),
        "veto_help": int(veto_help),
        "veto_neutral": int(veto_neutral),
        "veto_harm_precision": safe_div(float(veto_harm), float(max(1, baseline_count))),
        "veto_help_precision": safe_div(float(veto_help), float(max(1, baseline_count))),
        "veto_harm_recall": safe_div(float(veto_harm), float(max(1, total_harm))),
        "method_harm": int(method_harm),
        "method_help": int(method_help),
        "method_neutral": int(method_neutral),
        "method_help_precision": safe_div(float(method_help), float(max(1, method_count))),
        "method_harm_precision": safe_div(float(method_harm), float(max(1, method_count))),
        "per_source": per_source,
    }


def selection_key(result: Dict[str, Any], objective: str) -> Sequence[float]:
    if objective == "p2_sign":
        return (
            float(result["p2_harm_vs_help_auroc"] or 0.0),
            float(result["veto_harm_precision"]) - float(result["veto_help_precision"]),
            float(result["source_balanced_utility"]),
            float(result["worst_source_utility"]),
            float(result["delta_vs_intervention"]),
        )
    return (
        float(result["source_balanced_utility"]),
        float(result["worst_source_utility"]),
        float(result["p2_harm_vs_help_auroc"] or 0.0),
        float(result["veto_harm_precision"]) - float(result["veto_help_precision"]),
        float(result["delta_vs_intervention"]),
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
    ap = argparse.ArgumentParser(description="Build semantic two-stage v2 controller with stronger P2 sign model.")
    ap.add_argument("--discovery_table_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--p1_feature_cols", type=str, required=True)
    ap.add_argument("--p2_feature_cols", type=str, required=True)
    ap.add_argument("--min_feature_auroc_p1", type=float, default=0.55)
    ap.add_argument("--min_feature_auroc_p2", type=float, default=0.55)
    ap.add_argument("--top_k_p1", type=int, default=5)
    ap.add_argument("--top_k_p2", type=int, default=5)
    ap.add_argument("--tau_quantiles_p1", type=str, default="0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--tau_quantiles_p2", type=str, default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--objective", type=str, default="balanced_utility", choices=["balanced_utility", "p2_sign"])
    ap.add_argument("--min_sensitive_rate", type=float, default=0.05)
    ap.add_argument("--max_sensitive_rate", type=float, default=0.50)
    ap.add_argument("--min_sensitive_count", type=int, default=50)
    ap.add_argument("--p1_pair_mode", type=str, default="none", choices=["none", "pairwise"])
    ap.add_argument("--p2_pair_mode", type=str, default="pairwise", choices=["none", "pairwise"])
    ap.add_argument("--fit_epochs", type=int, default=300)
    ap.add_argument("--fit_lr", type=float, default=0.05)
    ap.add_argument("--fit_l2", type=float, default=1e-3)
    args = ap.parse_args()

    p1_feature_cols = [x.strip() for x in str(args.p1_feature_cols).split(",") if x.strip()]
    p2_feature_cols = [x.strip() for x in str(args.p2_feature_cols).split(",") if x.strip()]
    q1 = parse_quantiles(args.tau_quantiles_p1)
    q2 = parse_quantiles(args.tau_quantiles_p2)

    source_rows: Dict[str, List[Dict[str, str]]] = {}
    all_rows: List[Dict[str, str]] = []
    for path in tqdm(args.discovery_table_csvs, desc="two-stage-v2-load", unit="table"):
        rows = read_csv_rows(path)
        if not rows:
            continue
        source = str(rows[0].get("benchmark", os.path.basename(path)))
        source_rows[source] = rows
        all_rows.extend(rows)

    p1_metrics, p1_selected = select_family(
        source_rows,
        p1_feature_cols,
        target_name="sensitive",
        label_fn=label_sensitive,
        min_feature_auroc=float(args.min_feature_auroc_p1),
        top_k=int(args.top_k_p1),
    )
    p2_metrics, p2_selected = select_family(
        source_rows,
        p2_feature_cols,
        target_name="harm_vs_help",
        label_fn=label_harm_vs_help,
        min_feature_auroc=float(args.min_feature_auroc_p2),
        top_k=int(args.top_k_p2),
    )
    if not p1_selected:
        raise RuntimeError("No eligible P1 features survived.")
    if not p2_selected:
        raise RuntimeError("No eligible P2 features survived.")

    p1_model = fit_nonnegative_logistic(
        all_rows,
        label_fn=label_sensitive,
        features=p1_selected,
        pair_mode=str(args.p1_pair_mode),
        epochs=int(args.fit_epochs),
        lr=float(args.fit_lr),
        l2=float(args.fit_l2),
        target_name="p1_sensitive",
    )
    p2_model = fit_nonnegative_logistic(
        all_rows,
        label_fn=label_harm_vs_help,
        features=p2_selected,
        pair_mode=str(args.p2_pair_mode),
        epochs=int(args.fit_epochs),
        lr=float(args.fit_lr),
        l2=float(args.fit_l2),
        target_name="p2_harm_vs_help",
    )

    p1_probs = [score_model(row, p1_model)[1] for row in all_rows if score_model(row, p1_model) is not None]
    p2_probs = [score_model(row, p2_model)[1] for row in all_rows if label_harm_vs_help(row) is not None and score_model(row, p2_model) is not None]
    tau_grid_p1 = quantiles_to_thresholds(p1_probs, q1)
    tau_grid_p2 = quantiles_to_thresholds(p2_probs, q2)

    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for tau_p1 in tqdm(tau_grid_p1, desc="two-stage-v2-p1", unit="tau"):
        for tau_p2 in tqdm(tau_grid_p2, desc=f"two-stage-v2-p2@{tau_p1:.3f}", unit="tau", leave=False):
            result = evaluate_two_stage(source_rows, p1_model, p2_model, tau_p1, tau_p2)
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
        raise RuntimeError("No feasible semantic two-stage v2 controller found.")

    policy = {
        "policy_type": "semantic_two_stage_v2",
        "route_policy": "run VGA only if p1_sensitive_prob >= tau_p1 and p2_harm_prob < tau_p2",
        "p1_model": p1_model,
        "p2_model": p2_model,
        "tau_p1": float(best["tau_p1"]),
        "tau_p2": float(best["tau_p2"]),
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
                "p1_pair_mode": str(args.p1_pair_mode),
                "p2_pair_mode": str(args.p2_pair_mode),
                "fit_epochs": int(args.fit_epochs),
                "fit_lr": float(args.fit_lr),
                "fit_l2": float(args.fit_l2),
            },
            "n_rows": int(len(all_rows)),
            "sources": sorted(source_rows.keys()),
            "selected_p1_features": p1_selected,
            "selected_p2_features": p2_selected,
            "p1_model": p1_model,
            "p2_model": p2_model,
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
