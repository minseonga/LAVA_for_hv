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
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


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


def target_value(row: Dict[str, Any], spec: str) -> Optional[int]:
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


def parse_term_count(value: object) -> int:
    s = str(value if value is not None else "").strip()
    if not s:
        return 0
    return len([x for x in s.split(" || ") if x.strip()])


def load_chair_metric_map(path: str) -> Dict[str, Dict[str, float]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    out: Dict[str, Dict[str, float]] = {}
    for row in obj.get("sentences", []):
        image_id = str(row.get("image_id", "")).strip()
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        if not image_id:
            continue
        out[image_id] = {
            "chair_s": float(maybe_float(metrics.get("CHAIRs")) or 0.0),
            "chair_i": float(maybe_float(metrics.get("CHAIRi")) or 0.0),
            "recall": float(maybe_float(metrics.get("Recall")) or 0.0),
        }
    return out


def build_master_rows(
    claim_rows: Sequence[Dict[str, str]],
    chair_rows: Sequence[Dict[str, str]],
    baseline_chair_json: str,
    intervention_chair_json: str,
) -> List[Dict[str, Any]]:
    claim_map = {str(r.get("id", "")).strip(): r for r in claim_rows}
    chair_map = {str(r.get("id", "")).strip(): r for r in chair_rows}
    base_chair_map = load_chair_metric_map(baseline_chair_json)
    int_chair_map = load_chair_metric_map(intervention_chair_json)

    rows: List[Dict[str, Any]] = []
    for sid, claim in claim_map.items():
        chair = chair_map.get(sid)
        if chair is None:
            continue
        image_id = str(claim.get("image_id", chair.get("image_id", ""))).strip()
        base_json = base_chair_map.get(image_id, {})
        int_json = int_chair_map.get(image_id, {})

        n_base_supported = int(maybe_int(claim.get("n_base_supported")) or 0)
        n_int_supported = int(maybe_int(claim.get("n_int_supported")) or 0)
        n_base_hall = int(maybe_int(claim.get("n_base_hall")) or 0)
        n_int_hall = int(maybe_int(claim.get("n_int_hall")) or 0)

        base_precision = safe_div(float(n_base_supported), float(max(1, n_base_supported + n_base_hall)))
        int_precision = safe_div(float(n_int_supported), float(max(1, n_int_supported + n_int_hall)))
        base_recall = float(maybe_float(claim.get("base_supported_recall")) or 0.0)
        int_recall = float(maybe_float(claim.get("int_supported_recall")) or 0.0)
        base_f1 = safe_div(2.0 * base_precision * base_recall, base_precision + base_recall)
        int_f1 = safe_div(2.0 * int_precision * int_recall, int_precision + int_recall)

        row: Dict[str, Any] = dict(claim)
        for key, value in chair.items():
            if key not in row:
                row[key] = value
        row["chair_harm"] = int(maybe_int(chair.get("harm")) or 0)
        row["chair_help"] = int(maybe_int(chair.get("help")) or 0)
        row["base_chair_s"] = float(maybe_float(chair.get("baseline_chair_s")) or base_json.get("chair_s", 0.0))
        row["int_chair_s"] = float(maybe_float(chair.get("intervention_chair_s")) or int_json.get("chair_s", 0.0))
        row["base_chair_i"] = float(maybe_float(chair.get("baseline_chair")) or base_json.get("chair_i", 0.0))
        row["int_chair_i"] = float(maybe_float(chair.get("intervention_chair")) or int_json.get("chair_i", 0.0))
        row["base_recall"] = base_recall
        row["int_recall"] = int_recall
        row["base_precision"] = base_precision
        row["int_precision"] = int_precision
        row["base_f1"] = base_f1
        row["int_f1"] = int_f1
        row["base_f1_minus_chairi"] = float(base_f1 - row["base_chair_i"])
        row["int_f1_minus_chairi"] = float(int_f1 - row["int_chair_i"])
        rows.append(row)
    rows.sort(key=lambda r: int(str(r.get("id", "0"))))
    return rows


def infer_probe_feature_cols(rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    out: List[str] = []
    seen = set()
    all_keys: List[str] = []
    for row in rows:
        for key in row.keys():
            skey = str(key)
            if skey in seen:
                continue
            seen.add(skey)
            all_keys.append(skey)
    for key in all_keys:
        skey = str(key)
        if not (skey.startswith("probe_") or skey.startswith("pair_")):
            continue
        ok = False
        for row in rows:
            if maybe_float(row.get(key)) is not None:
                ok = True
                break
        if ok:
            out.append(str(key))
    return out


def evaluate_feature(rows: Sequence[Dict[str, Any]], feature: str, target_spec: str) -> Optional[Dict[str, Any]]:
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
    }


def build_composite_scores(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Tuple[str, str]],
) -> List[Optional[float]]:
    valid_vals: Dict[str, List[float]] = {}
    stats: Dict[str, Tuple[float, float]] = {}
    for feature, direction in feature_specs:
        oriented_vals: List[float] = []
        for row in rows:
            x = maybe_float(row.get(feature))
            if x is not None:
                oriented_vals.append(float(x) if direction == "high" else float(-x))
        stats[feature] = (mean(oriented_vals), std(oriented_vals))

    scores: List[Optional[float]] = []
    for row in rows:
        zvals: List[float] = []
        ok = True
        for feature, direction in feature_specs:
            x = maybe_float(row.get(feature))
            if x is None:
                ok = False
                break
            val = float(x) if direction == "high" else float(-x)
            mu, sd = stats[feature]
            zvals.append((val - mu) / sd)
        scores.append(mean(zvals) if ok and zvals else None)
    return scores


def quantiles_to_thresholds(values: Sequence[float], quantiles: Sequence[float]) -> List[float]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
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


def route_by_score(score: Optional[float], tau: float) -> str:
    if score is None:
        return "method"
    return "baseline" if float(score) >= float(tau) else "method"


def route_metrics(row: Dict[str, Any], route: str) -> Dict[str, float]:
    use_base = route == "baseline"
    prefix = "base_" if use_base else "int_"
    chair_i = float(row[f"{prefix}chair_i"])
    chair_s = float(row[f"{prefix}chair_s"])
    recall = float(row[f"{prefix}recall"])
    precision = float(row[f"{prefix}precision"])
    f1 = float(row[f"{prefix}f1"])
    claim_utility = float(row["baseline_claim_utility"] if use_base else row["intervention_claim_utility"])
    return {
        "chair_i": chair_i,
        "chair_s": chair_s,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "f1_minus_chairi": float(f1 - chair_i),
        "claim_utility": claim_utility,
    }


def objective_value(metrics: Dict[str, float], objective: str) -> float:
    if objective == "f1":
        return float(metrics["mean_f1"])
    if objective == "recall":
        return float(metrics["mean_recall"])
    if objective == "recall_minus_chairi":
        return float(metrics["mean_recall"] - metrics["mean_chair_i"])
    if objective == "neg_chairi":
        return float(-metrics["mean_chair_i"])
    if objective == "claim_utility":
        return float(metrics["mean_claim_utility"])
    return float(metrics["mean_f1_minus_chairi"])


def aggregate_routes(
    rows: Sequence[Dict[str, Any]],
    routes: Sequence[str],
    *,
    b_scores: Optional[Sequence[Optional[float]]] = None,
    c_scores: Optional[Sequence[Optional[float]]] = None,
    f_scores: Optional[Sequence[Optional[float]]] = None,
    experts: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    chair_i_vals: List[float] = []
    chair_s_vals: List[float] = []
    recall_vals: List[float] = []
    precision_vals: List[float] = []
    f1_vals: List[float] = []
    utility_vals: List[float] = []
    decision_rows: List[Dict[str, Any]] = []
    baseline_count = 0

    for idx, (row, route) in enumerate(zip(rows, routes)):
        mets = route_metrics(row, route)
        if route == "baseline":
            baseline_count += 1
        chair_i_vals.append(mets["chair_i"])
        chair_s_vals.append(mets["chair_s"])
        recall_vals.append(mets["recall"])
        precision_vals.append(mets["precision"])
        f1_vals.append(mets["f1"])
        utility_vals.append(mets["claim_utility"])
        out = {
            "id": row.get("id"),
            "image_id": row.get("image_id"),
            "route": route,
            "final_chair_i": mets["chair_i"],
            "final_chair_s": mets["chair_s"],
            "final_recall": mets["recall"],
            "final_precision": mets["precision"],
            "final_f1": mets["f1"],
            "final_f1_minus_chairi": mets["f1_minus_chairi"],
            "final_claim_utility": mets["claim_utility"],
            "base_chair_i": row.get("base_chair_i"),
            "int_chair_i": row.get("int_chair_i"),
            "base_f1": row.get("base_f1"),
            "int_f1": row.get("int_f1"),
            "chair_harm": row.get("chair_harm"),
            "claim_harm": row.get("harm"),
        }
        if b_scores is not None:
            out["b_score"] = b_scores[idx]
        if c_scores is not None:
            out["c_score"] = c_scores[idx]
        if f_scores is not None:
            out["f_score"] = f_scores[idx]
        if experts is not None:
            out["expert"] = experts[idx]
        decision_rows.append(out)

    return {
        "n_eval": int(len(rows)),
        "baseline_rate": safe_div(float(baseline_count), float(max(1, len(rows)))),
        "method_rate": safe_div(float(len(rows) - baseline_count), float(max(1, len(rows)))),
        "mean_chair_i": mean(chair_i_vals),
        "mean_chair_s": mean(chair_s_vals),
        "mean_recall": mean(recall_vals),
        "mean_precision": mean(precision_vals),
        "mean_f1": mean(f1_vals),
        "mean_f1_minus_chairi": mean([float(f - c) for f, c in zip(f1_vals, chair_i_vals)]),
        "mean_claim_utility": mean(utility_vals),
        "decision_rows": decision_rows,
    }


def compare_key(summary: Dict[str, Any], objective: str) -> Tuple[float, float, float]:
    tie_metric = "mean_recall" if str(objective).startswith("recall") else "mean_f1"
    return (
        objective_value(summary, objective),
        float(summary[tie_metric]),
        -float(summary["baseline_rate"]),
    )


def choose_expert(
    b_score: Optional[float],
    c_score: Optional[float],
    delta: float,
    mode: str,
) -> str:
    b_ok = b_score is not None
    c_ok = c_score is not None
    if b_ok and not c_ok:
        return "b_only"
    if c_ok and not b_ok:
        return "c_only"
    if not b_ok and not c_ok:
        return "fusion"

    assert b_score is not None and c_score is not None
    abs_b = abs(float(b_score))
    abs_c = abs(float(c_score))
    if abs_b - abs_c >= float(delta):
        return "b_only"
    if abs_c - abs_b >= float(delta):
        return "c_only"
    if mode == "delta_then_fusion":
        return "fusion"
    if mode == "delta_then_stronger":
        return "b_only" if abs_b >= abs_c else "c_only"
    if mode == "agree_fusion_else_stronger":
        b_sign = 0 if b_score == 0 else (1 if float(b_score) > 0 else -1)
        c_sign = 0 if c_score == 0 else (1 if float(c_score) > 0 else -1)
        if b_sign == c_sign:
            return "fusion"
        return "b_only" if abs_b >= abs_c else "c_only"
    return "fusion"


def select_candidate_features(
    rows: Sequence[Dict[str, Any]],
    target_spec: str,
    feature_cols: Sequence[str],
    min_auroc: float,
) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for feature in feature_cols:
        result = evaluate_feature(rows, feature, target_spec)
        if result is None:
            continue
        if float(result["auroc"]) < float(min_auroc):
            continue
        metrics.append(result)
    metrics.sort(key=lambda r: (-float(r["auroc"]), -float(r["average_precision"] or 0.0), str(r["feature"])))
    return metrics


def search_single_family(
    rows: Sequence[Dict[str, Any]],
    feature_metrics: Sequence[Dict[str, Any]],
    *,
    top_k_values: Sequence[int],
    tau_quantiles: Sequence[float],
    objective: str,
    min_baseline_rate: float,
    max_baseline_rate: float,
    family_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Optional[float]]]:
    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_scores: List[Optional[float]] = []

    for k in top_k_values:
        selected = feature_metrics[: min(int(k), len(feature_metrics))]
        if not selected:
            continue
        feature_specs = [(str(r["feature"]), str(r["direction"])) for r in selected]
        scores = build_composite_scores(rows, feature_specs)
        tau_grid = quantiles_to_thresholds([float(s) for s in scores if s is not None], tau_quantiles)
        valid_scores: List[float] = []
        valid_labels: List[int] = []
        target_spec = str(selected[0]["target_spec"])
        for row, score in zip(rows, scores):
            y = target_value(row, target_spec)
            if score is None or y not in {0, 1}:
                continue
            valid_scores.append(float(score))
            valid_labels.append(int(y))
        score_auc = binary_auroc(valid_scores, valid_labels)
        score_ap = binary_average_precision(valid_scores, valid_labels)

        for tau in tau_grid:
            routes = [route_by_score(score, float(tau)) for score in scores]
            summary = aggregate_routes(rows, routes, b_scores=scores if family_name == "b_only" else None, c_scores=scores if family_name == "c_only" else None)
            if float(summary["baseline_rate"]) < float(min_baseline_rate):
                continue
            if float(summary["baseline_rate"]) > float(max_baseline_rate):
                continue
            row = {
                "family": family_name,
                "k": int(k),
                "features": ",".join(f for f, _ in feature_specs),
                "directions": ",".join(d for _, d in feature_specs),
                "score_target_auroc": None if score_auc is None else float(score_auc),
                "score_target_ap": None if score_ap is None else float(score_ap),
                "tau": float(tau),
                **{k2: v2 for k2, v2 in summary.items() if k2 != "decision_rows"},
            }
            sweep_rows.append(row)
            if best is None or compare_key(row, objective) > compare_key(best, objective):
                best = dict(row)
                best["feature_specs"] = [{"feature": f, "direction": d} for f, d in feature_specs]
                best_scores = scores

    if best is None:
        raise RuntimeError(f"No feasible policy found for {family_name}.")
    return best, sweep_rows, best_scores


def search_fusion(
    rows: Sequence[Dict[str, Any]],
    b_policy: Dict[str, Any],
    c_policy: Dict[str, Any],
    b_scores: Sequence[Optional[float]],
    c_scores: Sequence[Optional[float]],
    *,
    weight_grid: Sequence[float],
    tau_quantiles: Sequence[float],
    objective: str,
    min_baseline_rate: float,
    max_baseline_rate: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Optional[float]]]:
    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_scores: List[Optional[float]] = []

    for w_b in weight_grid:
        for w_c in weight_grid:
            if float(w_b) <= 0.0 or float(w_c) <= 0.0:
                continue
            scores: List[Optional[float]] = []
            for b, c in zip(b_scores, c_scores):
                if b is None or c is None:
                    scores.append(None)
                else:
                    scores.append(float(w_b) * float(b) + float(w_c) * float(c))
            tau_grid = quantiles_to_thresholds([float(s) for s in scores if s is not None], tau_quantiles)
            for tau in tau_grid:
                routes = [route_by_score(score, float(tau)) for score in scores]
                summary = aggregate_routes(rows, routes, b_scores=b_scores, c_scores=c_scores, f_scores=scores)
                if float(summary["baseline_rate"]) < float(min_baseline_rate):
                    continue
                if float(summary["baseline_rate"]) > float(max_baseline_rate):
                    continue
                row = {
                    "family": "fusion",
                    "w_b": float(w_b),
                    "w_c": float(w_c),
                    "tau": float(tau),
                    **{k: v for k, v in summary.items() if k != "decision_rows"},
                }
                sweep_rows.append(row)
                if best is None or compare_key(row, objective) > compare_key(best, objective):
                    best = dict(row)
                    best_scores = scores

    if best is None:
        raise RuntimeError("No feasible fusion policy found.")
    best["b_policy_ref"] = {
        "features": b_policy["features"],
        "directions": b_policy["directions"],
        "tau": b_policy["tau"],
    }
    best["c_policy_ref"] = {
        "features": c_policy["features"],
        "directions": c_policy["directions"],
        "tau": c_policy["tau"],
    }
    return best, sweep_rows, best_scores


def search_meta(
    rows: Sequence[Dict[str, Any]],
    b_policy: Dict[str, Any],
    c_policy: Dict[str, Any],
    fusion_policy: Dict[str, Any],
    b_scores: Sequence[Optional[float]],
    c_scores: Sequence[Optional[float]],
    f_scores: Sequence[Optional[float]],
    *,
    delta_grid: Sequence[float],
    meta_modes: Sequence[str],
    objective: str,
    min_baseline_rate: float,
    max_baseline_rate: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for delta in delta_grid:
        for mode in meta_modes:
            routes: List[str] = []
            experts: List[str] = []
            for b_score, c_score, f_score in zip(b_scores, c_scores, f_scores):
                expert = choose_expert(b_score, c_score, float(delta), str(mode))
                if expert == "b_only":
                    route = route_by_score(b_score, float(b_policy["tau"]))
                elif expert == "c_only":
                    route = route_by_score(c_score, float(c_policy["tau"]))
                else:
                    route = route_by_score(f_score, float(fusion_policy["tau"]))
                    expert = "fusion"
                experts.append(expert)
                routes.append(route)

            summary = aggregate_routes(rows, routes, b_scores=b_scores, c_scores=c_scores, f_scores=f_scores, experts=experts)
            if float(summary["baseline_rate"]) < float(min_baseline_rate):
                continue
            if float(summary["baseline_rate"]) > float(max_baseline_rate):
                continue
            n = float(max(1, len(rows)))
            row = {
                "family": "meta",
                "delta": float(delta),
                "mode": str(mode),
                "expert_b_only_rate": safe_div(float(sum(1 for x in experts if x == "b_only")), n),
                "expert_c_only_rate": safe_div(float(sum(1 for x in experts if x == "c_only")), n),
                "expert_fusion_rate": safe_div(float(sum(1 for x in experts if x == "fusion")), n),
                **{k: v for k, v in summary.items() if k != "decision_rows"},
            }
            sweep_rows.append(row)
            if best is None or compare_key(row, objective) > compare_key(best, objective):
                best = dict(row)
                best["decision_rows"] = summary["decision_rows"]

    if best is None:
        raise RuntimeError("No feasible meta policy found.")
    best["b_policy_ref"] = {"tau": b_policy["tau"], "features": b_policy["features"]}
    best["c_policy_ref"] = {"tau": c_policy["tau"], "features": c_policy["features"]}
    best["fusion_policy_ref"] = {"tau": fusion_policy["tau"], "w_b": fusion_policy["w_b"], "w_c": fusion_policy["w_c"]}
    return best, sweep_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a generative B/C/meta controller and evaluate CHAIR/F1 tradeoffs.")
    ap.add_argument("--claim_table_csv", type=str, required=True)
    ap.add_argument("--chair_table_csv", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--b_target_spec", type=str, default="delta_supported_recall:lt:0")
    ap.add_argument("--c_target_spec", type=str, default="chair_harm")
    ap.add_argument("--feature_cols", type=str, default="auto")
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k_values", type=str, default="1,2,3,4,5")
    ap.add_argument("--tau_quantiles", type=str, default="0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99")
    ap.add_argument("--weight_grid", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--delta_grid", type=str, default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--meta_modes", type=str, default="delta_then_fusion,delta_then_stronger,agree_fusion_else_stronger")
    ap.add_argument("--selection_objective", type=str, default="f1_minus_chairi", choices=["f1_minus_chairi", "f1", "neg_chairi", "claim_utility", "recall", "recall_minus_chairi"])
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    args = ap.parse_args()

    claim_rows = read_csv_rows(os.path.abspath(args.claim_table_csv))
    chair_rows = read_csv_rows(os.path.abspath(args.chair_table_csv))
    rows = build_master_rows(
        claim_rows,
        chair_rows,
        os.path.abspath(args.baseline_chair_json),
        os.path.abspath(args.intervention_chair_json),
    )
    feature_cols = infer_probe_feature_cols(rows) if str(args.feature_cols) == "auto" else [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]
    top_k_values = sorted({max(1, int(x.strip())) for x in str(args.top_k_values).split(",") if x.strip()})
    tau_quantiles = [float(x.strip()) for x in str(args.tau_quantiles).split(",") if x.strip()]
    weight_grid = [float(x.strip()) for x in str(args.weight_grid).split(",") if x.strip()]
    delta_grid = [float(x.strip()) for x in str(args.delta_grid).split(",") if x.strip()]
    meta_modes = [x.strip() for x in str(args.meta_modes).split(",") if x.strip()]

    b_metrics = select_candidate_features(rows, str(args.b_target_spec), feature_cols, float(args.min_feature_auroc))
    c_metrics = select_candidate_features(rows, str(args.c_target_spec), feature_cols, float(args.min_feature_auroc))
    if not b_metrics:
        raise RuntimeError("No feasible B features passed the AUROC threshold.")
    if not c_metrics:
        raise RuntimeError("No feasible C features passed the AUROC threshold.")

    b_policy, b_sweep, b_scores = search_single_family(
        rows,
        b_metrics,
        top_k_values=top_k_values,
        tau_quantiles=tau_quantiles,
        objective=str(args.selection_objective),
        min_baseline_rate=float(args.min_baseline_rate),
        max_baseline_rate=float(args.max_baseline_rate),
        family_name="b_only",
    )
    c_policy, c_sweep, c_scores = search_single_family(
        rows,
        c_metrics,
        top_k_values=top_k_values,
        tau_quantiles=tau_quantiles,
        objective=str(args.selection_objective),
        min_baseline_rate=float(args.min_baseline_rate),
        max_baseline_rate=float(args.max_baseline_rate),
        family_name="c_only",
    )
    fusion_policy, fusion_sweep, f_scores = search_fusion(
        rows,
        b_policy,
        c_policy,
        b_scores,
        c_scores,
        weight_grid=weight_grid,
        tau_quantiles=tau_quantiles,
        objective=str(args.selection_objective),
        min_baseline_rate=float(args.min_baseline_rate),
        max_baseline_rate=float(args.max_baseline_rate),
    )
    meta_policy, meta_sweep = search_meta(
        rows,
        b_policy,
        c_policy,
        fusion_policy,
        b_scores,
        c_scores,
        f_scores,
        delta_grid=delta_grid,
        meta_modes=meta_modes,
        objective=str(args.selection_objective),
        min_baseline_rate=float(args.min_baseline_rate),
        max_baseline_rate=float(args.max_baseline_rate),
    )

    baseline_summary = aggregate_routes(rows, ["baseline"] * len(rows))
    intervention_summary = aggregate_routes(rows, ["method"] * len(rows))

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "merged_rows.csv"), rows)
    write_csv(os.path.join(args.out_dir, "b_feature_metrics.csv"), b_metrics)
    write_csv(os.path.join(args.out_dir, "c_feature_metrics.csv"), c_metrics)
    write_csv(os.path.join(args.out_dir, "b_tau_sweep.csv"), b_sweep)
    write_csv(os.path.join(args.out_dir, "c_tau_sweep.csv"), c_sweep)
    write_csv(os.path.join(args.out_dir, "fusion_tau_sweep.csv"), fusion_sweep)
    write_csv(os.path.join(args.out_dir, "meta_sweep.csv"), meta_sweep)
    write_csv(os.path.join(args.out_dir, "selected_meta_rows.csv"), meta_policy["decision_rows"])
    write_json(
        os.path.join(args.out_dir, "selected_meta_policy.json"),
        {
            "policy_type": "generative_b_c_meta_v1",
            "selection_objective": str(args.selection_objective),
            "b_target_spec": str(args.b_target_spec),
            "c_target_spec": str(args.c_target_spec),
            "b_policy": {k: v for k, v in b_policy.items() if k not in {"decision_rows"}},
            "c_policy": {k: v for k, v in c_policy.items() if k not in {"decision_rows"}},
            "fusion_policy": {k: v for k, v in fusion_policy.items() if k not in {"decision_rows"}},
            "meta_policy": {k: v for k, v in meta_policy.items() if k not in {"decision_rows"}},
        },
    )
    write_json(
        os.path.join(args.out_dir, "summary.json"),
        {
            "inputs": {
                "claim_table_csv": os.path.abspath(args.claim_table_csv),
                "chair_table_csv": os.path.abspath(args.chair_table_csv),
                "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
                "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
                "b_target_spec": str(args.b_target_spec),
                "c_target_spec": str(args.c_target_spec),
                "selection_objective": str(args.selection_objective),
                "feature_cols": feature_cols,
                "min_feature_auroc": float(args.min_feature_auroc),
                "top_k_values": top_k_values,
                "tau_quantiles": tau_quantiles,
                "weight_grid": weight_grid,
                "delta_grid": delta_grid,
                "meta_modes": meta_modes,
            },
            "counts": {
                "n_rows": int(len(rows)),
            },
            "baseline": {k: v for k, v in baseline_summary.items() if k != "decision_rows"},
            "intervention": {k: v for k, v in intervention_summary.items() if k != "decision_rows"},
            "best_b_only": {k: v for k, v in b_policy.items() if k != "decision_rows"},
            "best_c_only": {k: v for k, v in c_policy.items() if k != "decision_rows"},
            "best_fusion": {k: v for k, v in fusion_policy.items() if k != "decision_rows"},
            "best_meta": {k: v for k, v in meta_policy.items() if k != "decision_rows"},
            "outputs": {
                "merged_rows_csv": os.path.abspath(os.path.join(args.out_dir, "merged_rows.csv")),
                "selected_meta_rows_csv": os.path.abspath(os.path.join(args.out_dir, "selected_meta_rows.csv")),
                "selected_meta_policy_json": os.path.abspath(os.path.join(args.out_dir, "selected_meta_policy.json")),
            },
        },
    )
    print("[saved]", os.path.abspath(os.path.join(args.out_dir, "summary.json")))
    print("[saved]", os.path.abspath(os.path.join(args.out_dir, "selected_meta_rows.csv")))


if __name__ == "__main__":
    main()
