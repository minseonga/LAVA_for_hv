#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def parse_yes_no(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip().lower() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def safe_id(value: object) -> str:
    return str(value or "").strip()


def parse_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def parse_int01(value: object) -> Optional[int]:
    if value in {1, 1.0, True, "1", "true", "True"}:
        return 1
    if value in {0, 0.0, False, "0", "false", "False"}:
        return 0
    return None


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def binary_entropy(p: float) -> float:
    eps = 1e-12
    p = min(1.0 - eps, max(eps, float(p)))
    q = 1.0 - p
    return float(-(p * math.log(p) + q * math.log(q)))


def binary_kl_to_uniform(p: float) -> float:
    eps = 1e-12
    p = min(1.0 - eps, max(eps, float(p)))
    q = 1.0 - p
    return float(p * math.log(2.0 * p) + q * math.log(2.0 * q))


def add_decision_kl_features(row: Dict[str, Any]) -> None:
    """Derive no-extra-forward decision KL/entropy features from existing CSV scalars."""
    candidate_p = parse_float(row.get("cheap_decision_candidate_prob_binary"))
    yes_p = parse_float(row.get("cheap_decision_yes_prob_binary"))
    no_p = parse_float(row.get("cheap_decision_no_prob_binary"))
    margin = parse_float(row.get("cheap_decision_candidate_minus_alt"))

    if candidate_p is not None:
        p = min(1.0 - 1e-12, max(1e-12, float(candidate_p)))
        row["cheap_decision_candidate_kl_uniform"] = binary_kl_to_uniform(p)
        row["cheap_decision_candidate_entropy"] = binary_entropy(p)
        row["cheap_decision_candidate_conf_abs"] = abs(p - 0.5)
        row["cheap_decision_candidate_neg_entropy"] = -binary_entropy(p)

    if yes_p is not None:
        p = min(1.0 - 1e-12, max(1e-12, float(yes_p)))
        row["cheap_decision_yesno_kl_uniform"] = binary_kl_to_uniform(p)
        row["cheap_decision_yesno_entropy"] = binary_entropy(p)
        row["cheap_decision_yesno_conf_abs"] = abs(p - 0.5)
        row["cheap_decision_yesno_neg_entropy"] = -binary_entropy(p)

    if no_p is not None:
        p = min(1.0 - 1e-12, max(1e-12, float(no_p)))
        row["cheap_decision_no_kl_uniform"] = binary_kl_to_uniform(p)
        row["cheap_decision_no_entropy"] = binary_entropy(p)
        row["cheap_decision_no_conf_abs"] = abs(p - 0.5)

    if margin is not None:
        row["cheap_decision_margin_abs_log1p"] = math.log1p(abs(float(margin)))
        row["cheap_decision_margin_signed_kl_proxy"] = math.copysign(
            binary_kl_to_uniform(candidate_p if candidate_p is not None else 0.5),
            float(margin),
        )


def load_gt(path: str, id_col: str, label_col: str, group_col: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            sid = safe_id(row.get(id_col))
            label = safe_id(row.get(label_col)).lower()
            if not sid or label not in {"yes", "no"}:
                continue
            out[sid] = {
                "gt_label": label,
                "category": safe_id(row.get(group_col)) if group_col else "",
                "question": safe_id(row.get("question")),
            }
    return out


def load_pred(path: str, pred_key: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = safe_id(row.get("question_id", row.get("id")))
            if not sid or sid.lower() in {"none", "null", "nan"}:
                continue
            if pred_key == "auto":
                text = row.get("text", "") or row.get("output", "") or row.get("answer", "")
            else:
                text = row.get(pred_key, "")
            out[sid] = {"text": str(text or ""), "label": parse_yes_no(str(text or ""))}
    return out


def load_csv_rows(path: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            sid = safe_id(row.get("id", row.get("question_id")))
            if sid:
                rows[sid] = dict(row)
    return rows


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def pstdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return float(math.sqrt(max(0.0, var))) or 1.0


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = float(i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def infer_feature_specs(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_prefixes: Sequence[str],
    top_k: int,
    min_present_rate: float,
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    candidate_cols: List[str] = []
    for key in rows[0].keys():
        if any(str(key).startswith(prefix) for prefix in feature_prefixes):
            n_present = sum(1 for row in rows if parse_float(row.get(key)) is not None)
            if n_present >= max(5, int(float(min_present_rate) * len(rows))):
                candidate_cols.append(str(key))

    labels = [int(row.get("harm", 0)) for row in rows]
    specs: List[Dict[str, Any]] = []
    for col in candidate_cols:
        paired = [(parse_float(row.get(col)), int(row.get("harm", 0))) for row in rows]
        paired = [(x, y) for x, y in paired if x is not None]
        if len(paired) < 5:
            continue
        vals = [float(x) for x, _ in paired]
        ys = [int(y) for _, y in paired]
        auc_high = auroc(vals, ys)
        if auc_high is None:
            continue
        auc_low = 1.0 - auc_high
        direction = "high" if auc_high >= auc_low else "low"
        oriented = vals if direction == "high" else [-x for x in vals]
        specs.append(
            {
                "feature": col,
                "direction": direction,
                "auroc": max(float(auc_high), float(auc_low)),
                "mu": mean(oriented),
                "sd": pstdev(oriented),
            }
        )

    specs.sort(key=lambda x: float(x["auroc"]), reverse=True)
    return specs[: int(top_k)]


def score_row(row: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> Optional[float]:
    vals: List[float] = []
    for spec in specs:
        x = parse_float(row.get(str(spec["feature"])))
        if x is None:
            return None
        if str(spec.get("direction")) == "low":
            x = -x
        vals.append((float(x) - float(spec["mu"])) / float(spec["sd"] or 1.0))
    return mean(vals)


def orient_value(value: float, direction: str) -> float:
    return float(value) if str(direction) == "high" else -float(value)


def z_value(row: Mapping[str, Any], spec: Mapping[str, Any]) -> Optional[float]:
    x = parse_float(row.get(str(spec["feature"])))
    if x is None:
        return None
    x = orient_value(float(x), str(spec.get("direction", "high")))
    return float((x - float(spec["mu"])) / float(spec["sd"] or 1.0))


def infer_single_feature_specs(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    min_present_rate: float,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for col in feature_names:
        n_present = sum(1 for row in rows if parse_float(row.get(col)) is not None)
        if n_present < max(5, int(float(min_present_rate) * len(rows))):
            continue
        paired = [(parse_float(row.get(col)), int(row.get("harm", 0))) for row in rows]
        paired = [(x, y) for x, y in paired if x is not None]
        vals = [float(x) for x, _ in paired]
        ys = [int(y) for _, y in paired]
        auc_high = auroc(vals, ys)
        if auc_high is None:
            continue
        auc_low = 1.0 - auc_high
        direction = "high" if auc_high >= auc_low else "low"
        oriented = vals if direction == "high" else [-x for x in vals]
        specs.append(
            {
                "feature": col,
                "direction": direction,
                "auroc": max(float(auc_high), float(auc_low)),
                "mu": mean(oriented),
                "sd": pstdev(oriented),
            }
        )
    specs.sort(key=lambda x: float(x["auroc"]), reverse=True)
    return specs


def choose_threshold(rows: Sequence[Mapping[str, Any]], scores: Mapping[str, float]) -> Optional[float]:
    pairs = [
        (float(scores[str(row["id"])]), row)
        for row in rows
        if str(row["id"]) in scores
    ]
    if not pairs:
        return None

    # Exact threshold search in O(n log n). The previous implementation tried
    # every unique threshold and rescanned all rows, which becomes prohibitive
    # for interaction sweeps over thousands of feature combinations.
    pairs.sort(key=lambda item: item[0], reverse=True)
    base_correct = sum(int(row["intervention_correct"]) for row in rows)
    best_key = (base_correct / len(rows) if rows else 0.0, 0, 0)
    best_tau = float(pairs[0][0]) + 1e-9

    selected = harm_fixed = help_lost = correct_delta = 0
    i = 0
    while i < len(pairs):
        score_value = float(pairs[i][0])
        j = i
        while j < len(pairs) and float(pairs[j][0]) == score_value:
            row = pairs[j][1]
            selected += 1
            harm_fixed += int(row["harm"])
            help_lost += int(row["help"])
            correct_delta += int(row["baseline_correct"]) - int(row["intervention_correct"])
            j += 1
        correct = base_correct + correct_delta
        final_acc = correct / len(rows) if rows else 0.0
        net = harm_fixed - help_lost
        key = (final_acc, net, -selected)
        if key > best_key:
            best_key = key
            best_tau = score_value
        i = j
    return best_tau


def evaluate_scores(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    *,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    tau = threshold if threshold is not None else choose_threshold(rows, scores)
    selected = harm_fixed = help_lost = neutral = correct = 0
    if tau is None:
        correct = sum(int(row["intervention_correct"]) for row in rows)
        return {
            "threshold": None,
            "selected": 0,
            "selected_harm": 0,
            "selected_help": 0,
            "selected_neutral": 0,
            "net": 0,
            "final_acc": correct / len(rows) if rows else 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    for row in rows:
        sid = str(row["id"])
        selected_here = sid in scores and float(scores[sid]) >= float(tau)
        if selected_here:
            selected += 1
            harm_fixed += int(row["harm"])
            help_lost += int(row["help"])
            if not int(row["harm"]) and not int(row["help"]):
                neutral += 1
            correct += int(row["baseline_correct"])
        else:
            correct += int(row["intervention_correct"])
    n_harm = sum(int(row["harm"]) for row in rows)
    return {
        "threshold": tau,
        "selected": selected,
        "selected_harm": harm_fixed,
        "selected_help": help_lost,
        "selected_neutral": neutral,
        "net": harm_fixed - help_lost,
        "final_acc": correct / len(rows) if rows else 0.0,
        "precision": harm_fixed / selected if selected else 0.0,
        "recall": harm_fixed / n_harm if n_harm else 0.0,
    }


def make_interaction_scores(
    rows: Sequence[Mapping[str, Any]],
    token_spec: Mapping[str, Any],
    aux_spec: Mapping[str, Any],
    *,
    mode: str,
    weight: float,
    gate_tau: Optional[float] = None,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for row in rows:
        tz = z_value(row, token_spec)
        az = z_value(row, aux_spec)
        if tz is None or az is None:
            continue
        if mode == "add":
            score = float(tz) + float(weight) * float(az)
        elif mode == "mul":
            score = float(tz) * float(az)
        elif mode == "min":
            score = min(float(tz), float(az))
        elif mode == "max":
            score = max(float(tz), float(az))
        elif mode == "gate":
            if gate_tau is None or float(az) < float(gate_tau):
                continue
            score = float(tz)
        elif mode == "gate_add":
            if gate_tau is None or float(az) < float(gate_tau):
                continue
            score = float(tz) + float(weight) * float(az)
        else:
            raise ValueError(f"Unsupported interaction mode: {mode}")
        scores[str(row["id"])] = float(score)
    return scores


def quantile_values(values: Sequence[float], quantiles: Sequence[float]) -> List[float]:
    if not values:
        return []
    vals = sorted(float(v) for v in values)
    out: List[float] = []
    for q in quantiles:
        idx = min(len(vals) - 1, max(0, int(round(float(q) * (len(vals) - 1)))))
        out.append(vals[idx])
    return sorted(set(out))


def run_interaction_sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    token_features: Sequence[str],
    aux_features: Sequence[str],
    weights: Sequence[float],
    min_present_rate: float,
    top_k_each: int,
) -> Dict[str, Any]:
    token_specs = infer_single_feature_specs(rows, feature_names=token_features, min_present_rate=min_present_rate)
    aux_specs = infer_single_feature_specs(rows, feature_names=aux_features, min_present_rate=min_present_rate)
    token_specs = token_specs[: int(top_k_each)]
    aux_specs = aux_specs[: int(top_k_each)]

    results: List[Dict[str, Any]] = []
    for token_spec in token_specs:
        base_scores = {
            str(row["id"]): float(z)
            for row in rows
            for z in [z_value(row, token_spec)]
            if z is not None
        }
        base_eval = evaluate_scores(rows, base_scores)
        results.append(
            {
                "mode": "token_only",
                "weight": 0.0,
                "gate_tau": None,
                "token_feature": token_spec["feature"],
                "token_direction": token_spec["direction"],
                "token_auroc": token_spec["auroc"],
                "aux_feature": "",
                "aux_direction": "",
                "aux_auroc": 0.0,
                **base_eval,
            }
        )

        for aux_spec in aux_specs:
            aux_zs = [z_value(row, aux_spec) for row in rows]
            aux_vals = [float(x) for x in aux_zs if x is not None]
            gate_taus = quantile_values(aux_vals, [0.25, 0.5, 0.75, 0.9])
            for mode in ("add", "mul", "min", "max"):
                for w in weights if mode == "add" else [0.0]:
                    scores = make_interaction_scores(rows, token_spec, aux_spec, mode=mode, weight=float(w))
                    metric = evaluate_scores(rows, scores)
                    results.append(
                        {
                            "mode": mode,
                            "weight": float(w),
                            "gate_tau": None,
                            "token_feature": token_spec["feature"],
                            "token_direction": token_spec["direction"],
                            "token_auroc": token_spec["auroc"],
                            "aux_feature": aux_spec["feature"],
                            "aux_direction": aux_spec["direction"],
                            "aux_auroc": aux_spec["auroc"],
                            **metric,
                        }
                    )
            for gate_tau in gate_taus:
                for mode in ("gate", "gate_add"):
                    for w in weights if mode == "gate_add" else [0.0]:
                        scores = make_interaction_scores(
                            rows,
                            token_spec,
                            aux_spec,
                            mode=mode,
                            weight=float(w),
                            gate_tau=float(gate_tau),
                        )
                        metric = evaluate_scores(rows, scores)
                        results.append(
                            {
                                "mode": mode,
                                "weight": float(w),
                                "gate_tau": float(gate_tau),
                                "token_feature": token_spec["feature"],
                                "token_direction": token_spec["direction"],
                                "token_auroc": token_spec["auroc"],
                                "aux_feature": aux_spec["feature"],
                                "aux_direction": aux_spec["direction"],
                                "aux_auroc": aux_spec["auroc"],
                                **metric,
                            }
                        )

    results.sort(
        key=lambda r: (
            float(r.get("final_acc", 0.0)),
            int(r.get("net", 0)),
            float(r.get("precision", 0.0)),
            -int(r.get("selected", 0)),
        ),
        reverse=True,
    )
    return {
        "token_specs": token_specs,
        "aux_specs": aux_specs,
        "top_results": results[:50],
    }


def count_outcomes(rows: Iterable[Mapping[str, Any]], route_key: Optional[str] = None) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    changed: Counter[str] = Counter()
    by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    missed_harm: Counter[str] = Counter()
    caught_harm: Counter[str] = Counter()

    for row in rows:
        b = str(row.get("baseline_label", ""))
        m = str(row.get("intervention_label", ""))
        out = str(row.get("outcome", ""))
        cat = str(row.get("category", ""))
        trans = f"{b}->{m}"
        counts[out] += 1
        transitions[f"{trans}:{out}"] += 1
        by_category[cat][out] += 1
        if b != m:
            changed[f"{trans}:{out}"] += 1
        if route_key:
            route = str(row.get(route_key, ""))
            if int(row.get("harm", 0)) == 1:
                if route == "baseline":
                    caught_harm[trans] += 1
                else:
                    missed_harm[trans] += 1

    return {
        "outcome_counts": dict(counts),
        "transition_outcome_counts": dict(transitions),
        "changed_transition_counts": dict(changed),
        "by_category": {k: dict(v) for k, v in sorted(by_category.items())},
        "caught_harm_by_transition": dict(caught_harm),
        "missed_harm_by_transition": dict(missed_harm),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose C-controller caught/missed harm by POPE transition.")
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--feature_rows_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--label_col", default="answer")
    ap.add_argument("--group_col", default="category")
    ap.add_argument("--baseline_pred_key", default="auto")
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--feature_prefixes", default="cheap_")
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min_present_rate", type=float, default=0.8)
    ap.add_argument("--feature_rows_only", type=parse_bool, default=False)
    ap.add_argument("--derive_decision_kl_features", type=parse_bool, default=False)
    ap.add_argument("--interaction_sweep", type=parse_bool, default=False)
    ap.add_argument(
        "--interaction_token_features",
        default=(
            "cheap_target_gap_content_min,cheap_lp_content_min,cheap_lp_content_std,"
            "cheap_lp_content_tail_gap,cheap_lp_content_min_len_corr,cheap_lp_all_mean"
        ),
    )
    ap.add_argument(
        "--interaction_aux_features",
        default=(
            "cheap_decision_candidate_prob_binary,cheap_decision_candidate_minus_alt,"
            "cheap_decision_candidate_label_lp,cheap_decision_alt_label_lp,"
            "cheap_decision_candidate_kl_uniform,cheap_decision_candidate_entropy,"
            "cheap_decision_margin_signed_kl_proxy,cheap_decision_yesno_kl_uniform,"
            "cheap_decision_yesno_entropy,cheap_decision_yesno_conf_abs"
        ),
    )
    ap.add_argument("--interaction_weights", default="-1.0,-0.5,-0.25,0.25,0.5,1.0")
    ap.add_argument("--interaction_top_k_each", type=int, default=8)
    ap.add_argument("--max_examples", type=int, default=20)
    args = ap.parse_args()

    gt = load_gt(args.gt_csv, args.id_col, args.label_col, args.group_col)
    baseline = load_pred(args.baseline_pred_jsonl, args.baseline_pred_key)
    intervention = load_pred(args.intervention_pred_jsonl, args.intervention_pred_key)
    feature_rows = load_csv_rows(args.feature_rows_csv)

    rows: List[Dict[str, Any]] = []
    for sid, g in gt.items():
        if bool(args.feature_rows_only) and sid not in feature_rows:
            continue
        b = baseline.get(sid, {})
        m = intervention.get(sid, {})
        baseline_label = b.get("label", "")
        intervention_label = m.get("label", "")
        if baseline_label not in {"yes", "no"} or intervention_label not in {"yes", "no"}:
            continue
        bc = int(baseline_label == g["gt_label"])
        ic = int(intervention_label == g["gt_label"])
        if bc and ic:
            outcome = "both_correct"
        elif (not bc) and (not ic):
            outcome = "both_wrong"
        elif (not bc) and ic:
            outcome = "help"
        else:
            outcome = "harm"
        row: Dict[str, Any] = {
            "id": sid,
            "gt_label": g["gt_label"],
            "category": g["category"],
            "question": g["question"],
            "baseline_label": baseline_label,
            "intervention_label": intervention_label,
            "baseline_text": b.get("text", ""),
            "intervention_text": m.get("text", ""),
            "baseline_correct": bc,
            "intervention_correct": ic,
            "harm": int(outcome == "harm"),
            "help": int(outcome == "help"),
            "outcome": outcome,
        }
        row.update(feature_rows.get(sid, {}))
        row["id"] = sid
        if bool(args.derive_decision_kl_features):
            add_decision_kl_features(row)
        rows.append(row)

    feature_prefixes = [x.strip() for x in str(args.feature_prefixes).split(",") if x.strip()]
    specs = infer_feature_specs(
        rows,
        feature_prefixes=feature_prefixes,
        top_k=int(args.top_k),
        min_present_rate=float(args.min_present_rate),
    )
    scores = {str(row["id"]): score_row(row, specs) for row in rows}
    scores = {k: float(v) for k, v in scores.items() if v is not None}
    threshold = float(args.threshold) if args.threshold is not None else choose_threshold(rows, scores)

    selected_count = selected_harm = selected_help = selected_neutral = final_correct = 0
    missed_examples: List[Dict[str, Any]] = []
    caught_examples: List[Dict[str, Any]] = []
    for row in rows:
        sid = str(row["id"])
        score = scores.get(sid)
        route = "method"
        if score is not None and threshold is not None and float(score) >= float(threshold):
            route = "baseline"
            selected_count += 1
            selected_harm += int(row["harm"])
            selected_help += int(row["help"])
            if not int(row["harm"]) and not int(row["help"]):
                selected_neutral += 1
            final_correct += int(row["baseline_correct"])
        else:
            final_correct += int(row["intervention_correct"])
        row["diagnostic_c_score"] = score
        row["diagnostic_route"] = route

        if int(row["harm"]) == 1:
            example = {
                "id": sid,
                "category": row["category"],
                "transition": f"{row['baseline_label']}->{row['intervention_label']}",
                "score": score,
                "gt_label": row["gt_label"],
                "question": row["question"],
                "baseline_text": row["baseline_text"],
                "intervention_text": row["intervention_text"],
            }
            for spec in specs:
                feat = str(spec["feature"])
                example[feat] = row.get(feat)
            if route == "baseline" and len(caught_examples) < int(args.max_examples):
                caught_examples.append(example)
            if route != "baseline" and len(missed_examples) < int(args.max_examples):
                missed_examples.append(example)

    base_correct = sum(int(row["baseline_correct"]) for row in rows)
    method_correct = sum(int(row["intervention_correct"]) for row in rows)
    n = len(rows)
    summary = {
        "n": n,
        "baseline_acc": base_correct / n if n else 0.0,
        "intervention_acc": method_correct / n if n else 0.0,
        "diagnostic_final_acc": final_correct / n if n else 0.0,
        "diagnostic_delta_vs_intervention": (final_correct - method_correct) / n if n else 0.0,
        "selected": selected_count,
        "selected_harm": selected_harm,
        "selected_help": selected_help,
        "selected_neutral": selected_neutral,
        "missed_harm": sum(int(row["harm"]) == 1 and row["diagnostic_route"] != "baseline" for row in rows),
        "feature_specs": specs,
        "threshold": threshold,
    }
    analysis = count_outcomes(rows, route_key="diagnostic_route")
    out = {
        "inputs": vars(args),
        "summary": summary,
        "analysis": analysis,
        "missed_harm_examples": missed_examples,
        "caught_harm_examples": caught_examples,
    }
    if bool(args.interaction_sweep):
        token_features = [x.strip() for x in str(args.interaction_token_features).split(",") if x.strip()]
        aux_features = [x.strip() for x in str(args.interaction_aux_features).split(",") if x.strip()]
        weights = [float(x.strip()) for x in str(args.interaction_weights).split(",") if x.strip()]
        out["interaction_sweep"] = run_interaction_sweep(
            rows,
            token_features=token_features,
            aux_features=aux_features,
            weights=weights,
            min_present_rate=float(args.min_present_rate),
            top_k_each=int(args.interaction_top_k_each),
        )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("n", summary["n"])
    print("baseline_acc", summary["baseline_acc"])
    print("intervention_acc", summary["intervention_acc"])
    print("diagnostic_final_acc", summary["diagnostic_final_acc"])
    print("delta_vs_intervention", summary["diagnostic_delta_vs_intervention"])
    print("selected", selected_count, "harm", selected_harm, "help", selected_help, "neutral", selected_neutral)
    print("missed_harm", summary["missed_harm"])
    print("threshold", threshold)
    print("feature_specs")
    for spec in specs:
        print(spec["feature"], "dir", spec["direction"], "auroc", round(float(spec["auroc"]), 6))
    print("missed_harm_by_transition", analysis["missed_harm_by_transition"])
    print("caught_harm_by_transition", analysis["caught_harm_by_transition"])
    if bool(args.interaction_sweep):
        top = out.get("interaction_sweep", {}).get("top_results", [])
        print("interaction_sweep_top")
        for row in top[:10]:
            print(
                "acc",
                round(float(row.get("final_acc", 0.0)), 6),
                "net",
                row.get("net"),
                "prec",
                round(float(row.get("precision", 0.0)), 4),
                "sel",
                row.get("selected"),
                "harm",
                row.get("selected_harm"),
                "help",
                row.get("selected_help"),
                "mode",
                row.get("mode"),
                "w",
                row.get("weight"),
                "gate",
                row.get("gate_tau"),
                "tok",
                row.get("token_feature"),
                row.get("token_direction"),
                "aux",
                row.get("aux_feature"),
                row.get("aux_direction"),
            )
    print("[saved]", args.out_json)


if __name__ == "__main__":
    main()
