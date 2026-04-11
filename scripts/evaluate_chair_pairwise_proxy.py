#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import evaluate_chair_pairwise_rollback as oracle
import extract_generative_pairwise_features as pairfeat


DEFAULT_TAU_QUANTILES = [
    0.0,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.98,
    0.99,
    1.0,
]


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / float(len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mu = mean(values)
    var = sum((float(x) - mu) ** 2 for x in values) / float(len(values))
    return float(max(math.sqrt(max(var, 0.0)), 1e-6))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pairs = sorted(zip(scores, labels), key=lambda x: float(x[0]), reverse=True)
    n_pos = sum(int(y) for _, y in pairs)
    if n_pos <= 0:
        return None
    tp = 0
    total = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if int(label) == 1:
            tp += 1
            total += float(tp) / float(rank)
    return float(total / float(n_pos))


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_id_candidates(row: Dict[str, Any]) -> List[str]:
    vals: List[str] = []
    for key in ["image_id", "id", "question_id"]:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        vals.append(value)
        try:
            vals.append(str(int(value)))
        except Exception:
            pass
    out: List[str] = []
    seen = set()
    for value in vals:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def add_extra_numeric_features(rows: Sequence[Dict[str, Any]], paths: Sequence[str]) -> List[str]:
    added: List[str] = []
    seen = set()
    if not paths:
        return added
    for path in paths:
        table = read_csv_rows(os.path.abspath(path))
        feature_map: Dict[str, Dict[str, str]] = {}
        for item in table:
            for sid in row_id_candidates(item):
                feature_map[sid] = item
        for row in rows:
            matched = None
            for sid in row_id_candidates(row):
                matched = feature_map.get(sid)
                if matched is not None:
                    break
            if matched is None:
                continue
            for key, value in matched.items():
                if key in {"id", "image_id", "question_id", "image", "question"}:
                    continue
                x = maybe_float(value)
                if x is None:
                    continue
                out_key = str(key)
                row[out_key] = float(x)
                if out_key not in seen:
                    seen.add(out_key)
                    added.append(out_key)
    return added


def feature_direction(rows: Sequence[Dict[str, Any]], feature: str) -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = maybe_float(row.get(feature))
        if x is None:
            continue
        xs.append(float(x))
        ys.append(int(row.get("teacher_positive") or 0))
    if len(xs) < 2 or sum(ys) <= 0 or sum(ys) >= len(ys):
        return None
    auc_high = oracle.auroc(xs, ys)
    auc_low = oracle.auroc([-x for x in xs], ys)
    if auc_high is None or auc_low is None:
        return None
    direction = "high" if float(auc_high) >= float(auc_low) else "low"
    oriented = xs if direction == "high" else [-x for x in xs]
    return {
        "feature": feature,
        "direction": direction,
        "teacher_auroc": float(max(float(auc_high), float(auc_low))),
        "teacher_ap": average_precision(oriented, ys),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
    }


def add_caption_pair_proxy_features(rows: Sequence[Dict[str, Any]]) -> List[str]:
    feature_names: List[str] = []
    seen = set()
    for row in rows:
        feats = pairfeat.build_pair_features(
            str(row.get("base_caption", "")),
            str(row.get("int_caption", "")),
        )
        pairfeat.add_cross_risk_features(feats)

        object_preserve = float(feats.get("pair_object_overlap_base_frac") or 0.0)
        content_drop = float(feats.get("pair_unique_content_drop_rate") or 0.0)
        content_add = float(feats.get("pair_unique_content_add_rate") or 0.0)
        object_drop = float(feats.get("pair_object_drop_rate") or 0.0)
        object_add = float(feats.get("pair_object_add_rate") or 0.0)
        relation_drop = float(feats.get("pair_relation_drop_rate") or 0.0)
        attr_drop = float(feats.get("pair_attr_drop_rate") or 0.0)
        count_drop = float(feats.get("pair_count_drop_rate") or 0.0)
        word_shorter = float(feats.get("pair_word_shorter_frac") or 0.0)
        content_shorter = float(feats.get("pair_content_shorter_frac") or 0.0)

        # These composites are still GT-free: they only use the baseline/intervention text pair.
        feats.update(
            {
                "pair_proxy_object_preserved_content_drop": object_preserve * content_drop,
                "pair_proxy_object_preserved_relation_drop": object_preserve * relation_drop,
                "pair_proxy_object_preserved_detail_drop": object_preserve
                * mean([relation_drop, attr_drop, count_drop, content_drop]),
                "pair_proxy_recall_safe_drop_score": (content_drop + object_drop + content_shorter)
                / (1.0 + content_add + object_add),
                "pair_proxy_structure_preserved_compression": object_preserve
                * mean([word_shorter, content_shorter, content_drop]),
                "pair_proxy_relation_detail_drop_per_object_drop": mean([relation_drop, attr_drop, count_drop])
                / (1.0 + object_drop),
            }
        )

        for key, value in feats.items():
            x = maybe_float(value)
            if x is None:
                continue
            out_key = f"proxy_{key}"
            row[out_key] = float(x)
            if out_key not in seen:
                seen.add(out_key)
                feature_names.append(out_key)

        # CHAIR generated-object counts come from caption text extraction, not per-image GT labels.
        base_words = float(row.get("base_n_words") or 0.0)
        int_words = float(row.get("int_n_words") or 0.0)
        base_gen = float(row.get("base_n_generated_instances") or 0.0)
        int_gen = float(row.get("int_n_generated_instances") or 0.0)
        base_unique = float(row.get("base_n_generated_unique") or 0.0)
        int_unique = float(row.get("int_n_generated_unique") or 0.0)
        base_dup = float(row.get("base_n_duplicate_object_mentions") or 0.0)
        int_dup = float(row.get("int_n_duplicate_object_mentions") or 0.0)
        gen_proxy = {
            "proxy_chairgen_base_n_words": base_words,
            "proxy_chairgen_int_n_words": int_words,
            "proxy_chairgen_word_drop_rate": max(0.0, base_words - int_words) / max(1.0, base_words),
            "proxy_chairgen_word_ratio_int_to_base": int_words / max(1.0, base_words),
            "proxy_chairgen_base_generated_instances": base_gen,
            "proxy_chairgen_int_generated_instances": int_gen,
            "proxy_chairgen_generated_instance_drop": max(0.0, base_gen - int_gen),
            "proxy_chairgen_generated_instance_drop_rate": max(0.0, base_gen - int_gen) / max(1.0, base_gen),
            "proxy_chairgen_base_generated_unique": base_unique,
            "proxy_chairgen_int_generated_unique": int_unique,
            "proxy_chairgen_generated_unique_drop": max(0.0, base_unique - int_unique),
            "proxy_chairgen_generated_unique_drop_rate": max(0.0, base_unique - int_unique) / max(1.0, base_unique),
            "proxy_chairgen_generated_unique_ratio_int_to_base": int_unique / max(1.0, base_unique),
            "proxy_chairgen_base_object_density": base_unique / max(1.0, base_words),
            "proxy_chairgen_int_object_density": int_unique / max(1.0, int_words),
            "proxy_chairgen_object_density_drop": max(0.0, base_unique / max(1.0, base_words) - int_unique / max(1.0, int_words)),
            "proxy_chairgen_base_duplicate_rate": base_dup / max(1.0, base_gen),
            "proxy_chairgen_int_duplicate_rate": int_dup / max(1.0, int_gen),
            "proxy_chairgen_duplicate_rate_increase": max(0.0, int_dup / max(1.0, int_gen) - base_dup / max(1.0, base_gen)),
            "proxy_chairgen_unique_drop_x_int_dup_increase": (
                max(0.0, base_unique - int_unique) / max(1.0, base_unique)
            )
            * max(0.0, int_dup / max(1.0, int_gen) - base_dup / max(1.0, base_gen)),
        }
        for key, value in gen_proxy.items():
            row[key] = float(value)
            if key not in seen:
                seen.add(key)
                feature_names.append(key)
    return feature_names


def parse_quantiles(spec: str) -> List[float]:
    values: List[float] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        values.append(float(s))
    if not values:
        values = DEFAULT_TAU_QUANTILES
    return sorted({min(1.0, max(0.0, float(v))) for v in values})


def build_scores(
    rows: Sequence[Dict[str, Any]],
    specs: Sequence[Tuple[str, str]],
) -> Tuple[List[Optional[float]], Dict[str, Dict[str, float]]]:
    stats: Dict[str, Dict[str, float]] = {}
    for feature, direction in specs:
        vals: List[float] = []
        for row in rows:
            x = maybe_float(row.get(feature))
            if x is None:
                continue
            vals.append(float(x) if direction == "high" else -float(x))
        stats[feature] = {"mean": mean(vals), "std": std(vals)}

    scores: List[Optional[float]] = []
    for row in rows:
        zvals: List[float] = []
        ok = True
        for feature, direction in specs:
            x = maybe_float(row.get(feature))
            if x is None:
                ok = False
                break
            oriented = float(x) if direction == "high" else -float(x)
            mu = float(stats[feature]["mean"])
            sd = float(stats[feature]["std"] or 1.0)
            zvals.append((oriented - mu) / sd)
        scores.append(mean(zvals) if ok and zvals else None)
    return scores, stats


def evaluate_specs(
    rows: Sequence[Dict[str, Any]],
    specs: Sequence[Tuple[str, str]],
    quantiles: Sequence[float],
    max_baseline_rate: float,
    chair_i_eps_vs_int: float,
    chair_s_eps_vs_int: float,
    min_recall_gain_vs_int: float,
    require_f1_nondecrease_vs_int: bool,
    min_teacher_precision: float,
    min_teacher_recall: float,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    scores, stats = build_scores(rows, specs)
    valid_scores = [float(s) for s in scores if s is not None and math.isfinite(float(s))]
    thresholds = oracle.quantile_thresholds(valid_scores, quantiles)
    labels = [int(row.get("teacher_positive") or 0) for row in rows]
    n_pos = sum(labels)
    intervention = oracle.aggregate_counts(rows, lambda _: "method")

    sweep: List[Dict[str, Any]] = []
    for tau in thresholds:
        routes = ["baseline" if score is not None and float(score) >= float(tau) else "method" for score in scores]
        selected_idx = [idx for idx, route in enumerate(routes) if route == "baseline"]
        route_by_obj_id = {id(row): route for row, route in zip(rows, routes)}
        result = oracle.aggregate_counts(rows, lambda row, route_by_obj_id=route_by_obj_id: route_by_obj_id[id(row)])
        tp = sum(labels[idx] for idx in selected_idx)
        teacher_precision = oracle.safe_div(float(tp), float(len(selected_idx)))
        teacher_recall = oracle.safe_div(float(tp), float(n_pos))
        result.update(
            {
                "features": " + ".join(feature for feature, _ in specs),
                "feature_specs": json.dumps(
                    [{"feature": feature, "direction": direction} for feature, direction in specs],
                    ensure_ascii=False,
                ),
                "n_features": int(len(specs)),
                "tau": float(tau),
                "teacher_precision": teacher_precision,
                "teacher_recall": teacher_recall,
                "delta_chair_i_vs_int": float(result["chair_i"] - intervention["chair_i"]),
                "delta_chair_s_vs_int": float(result["chair_s"] - intervention["chair_s"]),
                "delta_recall_vs_int": float(result["recall"] - intervention["recall"]),
                "delta_f1_vs_int": float(result["f1"] - intervention["f1"]),
                "feasible": int(
                    result["baseline_rate"] <= float(max_baseline_rate)
                    and result["chair_i"] <= float(intervention["chair_i"] + chair_i_eps_vs_int)
                    and result["chair_s"] <= float(intervention["chair_s"] + chair_s_eps_vs_int)
                    and result["recall"] >= float(intervention["recall"] + min_recall_gain_vs_int)
                    and (not require_f1_nondecrease_vs_int or result["f1"] >= float(intervention["f1"] - 1e-12))
                    and teacher_precision >= float(min_teacher_precision)
                    and teacher_recall >= float(min_teacher_recall)
                ),
            }
        )
        sweep.append(result)

    feasible = [row for row in sweep if int(row["feasible"]) == 1]
    if not feasible:
        return sweep, None, stats
    best = sorted(
        feasible,
        key=lambda row: (
            float(row["delta_recall_vs_int"]),
            float(row["delta_f1_vs_int"]),
            -float(row["delta_chair_i_vs_int"]),
            -float(row["baseline_rate"]),
        ),
        reverse=True,
    )[0]
    return sweep, best, stats


def routes_for_policy(
    rows: Sequence[Dict[str, Any]],
    specs: Sequence[Tuple[str, str]],
    stats: Dict[str, Dict[str, float]],
    tau: float,
) -> Tuple[List[str], List[Optional[float]]]:
    scores: List[Optional[float]] = []
    routes: List[str] = []
    for row in rows:
        zvals: List[float] = []
        ok = True
        for feature, direction in specs:
            x = maybe_float(row.get(feature))
            if x is None:
                ok = False
                break
            oriented = float(x) if direction == "high" else -float(x)
            mu = float(stats.get(feature, {}).get("mean", 0.0))
            sd = float(stats.get(feature, {}).get("std", 1.0) or 1.0)
            zvals.append((oriented - mu) / sd)
        score = mean(zvals) if ok and zvals else None
        scores.append(score)
        routes.append("baseline" if score is not None and float(score) >= float(tau) else "method")
    return routes, scores


def score_with_policy_stats(
    row: Dict[str, Any],
    specs: Sequence[Tuple[str, str]],
    stats: Dict[str, Dict[str, float]],
) -> Optional[float]:
    zvals: List[float] = []
    for feature, direction in specs:
        x = maybe_float(row.get(feature))
        if x is None:
            return None
        oriented = float(x) if direction == "high" else -float(x)
        mu = float(stats.get(feature, {}).get("mean", 0.0))
        sd = float(stats.get(feature, {}).get("std", 1.0) or 1.0)
        zvals.append((oriented - mu) / sd)
    return mean(zvals) if zvals else None


def evaluate_gated_specs(
    rows: Sequence[Dict[str, Any]],
    *,
    anchor_spec: Tuple[str, str],
    gate_spec: Optional[Tuple[str, str]],
    quantiles: Sequence[float],
    max_baseline_rate: float,
    chair_i_eps_vs_int: float,
    chair_s_eps_vs_int: float,
    min_recall_gain_vs_int: float,
    require_f1_nondecrease_vs_int: bool,
    min_teacher_precision: float,
    min_teacher_recall: float,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    labels = [int(row.get("teacher_positive") or 0) for row in rows]
    n_pos = sum(labels)
    intervention = oracle.aggregate_counts(rows, lambda _: "method")

    anchor_scores, anchor_stats = build_scores(rows, [anchor_spec])
    anchor_valid = [float(s) for s in anchor_scores if s is not None and math.isfinite(float(s))]
    anchor_thresholds = oracle.quantile_thresholds(anchor_valid, quantiles)
    if gate_spec is None:
        gate_scores = [0.0 for _ in rows]
        gate_stats: Dict[str, Dict[str, float]] = {}
        gate_thresholds = [0.0]
    else:
        gate_scores, gate_stats = build_scores(rows, [gate_spec])
        gate_valid = [float(s) for s in gate_scores if s is not None and math.isfinite(float(s))]
        gate_thresholds = oracle.quantile_thresholds(gate_valid, quantiles)

    sweep: List[Dict[str, Any]] = []
    for anchor_tau in anchor_thresholds:
        for gate_tau in gate_thresholds:
            routes: List[str] = []
            selected_idx: List[int] = []
            for idx, (a_score, g_score) in enumerate(zip(anchor_scores, gate_scores)):
                selected = a_score is not None and float(a_score) >= float(anchor_tau)
                if gate_spec is not None:
                    selected = selected and g_score is not None and float(g_score) >= float(gate_tau)
                route = "baseline" if selected else "method"
                routes.append(route)
                if selected:
                    selected_idx.append(idx)
            route_by_obj_id = {id(row): route for row, route in zip(rows, routes)}
            result = oracle.aggregate_counts(rows, lambda row, route_by_obj_id=route_by_obj_id: route_by_obj_id[id(row)])
            tp = sum(labels[idx] for idx in selected_idx)
            teacher_precision = oracle.safe_div(float(tp), float(len(selected_idx)))
            teacher_recall = oracle.safe_div(float(tp), float(n_pos))
            result.update(
                {
                    "anchor_feature": anchor_spec[0],
                    "anchor_direction": anchor_spec[1],
                    "gate_feature": "" if gate_spec is None else gate_spec[0],
                    "gate_direction": "" if gate_spec is None else gate_spec[1],
                    "anchor_tau": float(anchor_tau),
                    "gate_tau": None if gate_spec is None else float(gate_tau),
                    "teacher_precision": teacher_precision,
                    "teacher_recall": teacher_recall,
                    "delta_chair_i_vs_int": float(result["chair_i"] - intervention["chair_i"]),
                    "delta_chair_s_vs_int": float(result["chair_s"] - intervention["chair_s"]),
                    "delta_recall_vs_int": float(result["recall"] - intervention["recall"]),
                    "delta_f1_vs_int": float(result["f1"] - intervention["f1"]),
                    "feasible": int(
                        result["baseline_rate"] <= float(max_baseline_rate)
                        and result["chair_i"] <= float(intervention["chair_i"] + chair_i_eps_vs_int)
                        and result["chair_s"] <= float(intervention["chair_s"] + chair_s_eps_vs_int)
                        and result["recall"] >= float(intervention["recall"] + min_recall_gain_vs_int)
                        and (not require_f1_nondecrease_vs_int or result["f1"] >= float(intervention["f1"] - 1e-12))
                        and teacher_precision >= float(min_teacher_precision)
                        and teacher_recall >= float(min_teacher_recall)
                    ),
                    "anchor_feature_stats": json.dumps(anchor_stats, ensure_ascii=False),
                    "gate_feature_stats": json.dumps(gate_stats, ensure_ascii=False),
                }
            )
            sweep.append(result)

    feasible = [row for row in sweep if int(row["feasible"]) == 1]
    if not feasible:
        return sweep, None
    best = sorted(
        feasible,
        key=lambda row: (
            float(row["delta_recall_vs_int"]),
            float(row["delta_f1_vs_int"]),
            -float(row["delta_chair_i_vs_int"]),
            -float(row["delta_chair_s_vs_int"]),
            -float(row["baseline_rate"]),
        ),
        reverse=True,
    )[0]
    return sweep, best


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Distill a CHAIR pairwise rollback oracle into GT-free caption-pair proxy features."
    )
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--chair_i_eps", type=float, default=0.01)
    ap.add_argument("--chair_s_eps", type=float, default=0.0)
    ap.add_argument("--min_recall_gain", type=float, default=0.0)
    ap.add_argument("--require_f1_nondecrease", action="store_true")
    ap.add_argument("--max_baseline_rate", type=float, default=0.08)
    ap.add_argument("--top_n_features", type=int, default=12)
    ap.add_argument("--max_combo_size", type=int, default=2)
    ap.add_argument("--feature", action="append", default=[])
    ap.add_argument("--extra_features_csv", action="append", default=[])
    ap.add_argument("--tau_quantiles", type=str, default=",".join(str(x) for x in DEFAULT_TAU_QUANTILES))
    ap.add_argument("--fit_gated", action="store_true")
    ap.add_argument("--anchor_feature", type=str, default="proxy_chairgen_generated_unique_drop")
    ap.add_argument("--gate_feature", action="append", default=[])
    ap.add_argument("--gate_feature_prefix", action="append", default=[])
    ap.add_argument(
        "--min_teacher_precision",
        type=float,
        default=0.0,
        help="Minimum teacher precision required when fitting a policy on a labeled split.",
    )
    ap.add_argument(
        "--min_teacher_recall",
        type=float,
        default=0.0,
        help="Minimum teacher recall required when fitting a policy on a labeled split.",
    )
    ap.add_argument(
        "--selected_policy_json",
        type=str,
        default="",
        help="Apply an already selected caption-pair proxy policy instead of fitting on this split.",
    )
    args = ap.parse_args()

    rows = oracle.build_rows(
        os.path.abspath(args.baseline_chair_json),
        os.path.abspath(args.intervention_chair_json),
        chair_i_eps=float(args.chair_i_eps),
        chair_s_eps=float(args.chair_s_eps),
        min_recall_gain=float(args.min_recall_gain),
        require_f1_nondecrease=bool(args.require_f1_nondecrease),
    )
    if not rows:
        raise RuntimeError("No overlapping CHAIR rows.")

    feature_names = add_caption_pair_proxy_features(rows)
    extra_feature_names = add_extra_numeric_features(rows, args.extra_features_csv)
    feature_names.extend([name for name in extra_feature_names if name not in set(feature_names)])
    if str(args.selected_policy_json).strip():
        policy = json.load(open(os.path.abspath(args.selected_policy_json), "r", encoding="utf-8"))
        if str(policy.get("policy_type")) == "chair_pairwise_caption_gated_proxy_v1":
            anchor_spec = (str(policy["anchor_feature"]), str(policy["anchor_direction"]))
            gate_feature = str(policy.get("gate_feature") or "").strip()
            gate_spec = None
            if gate_feature:
                gate_spec = (gate_feature, str(policy["gate_direction"]))
            anchor_stats = {
                str(feature): {"mean": float(values.get("mean", 0.0)), "std": float(values.get("std", 1.0) or 1.0)}
                for feature, values in dict(policy.get("anchor_feature_stats", {})).items()
            }
            gate_stats = {
                str(feature): {"mean": float(values.get("mean", 0.0)), "std": float(values.get("std", 1.0) or 1.0)}
                for feature, values in dict(policy.get("gate_feature_stats", {})).items()
            }
            anchor_tau = float(policy["anchor_tau"])
            gate_tau = None if policy.get("gate_tau") is None else float(policy["gate_tau"])
            routes = []
            scores = []
            for row in rows:
                anchor_score = score_with_policy_stats(row, [anchor_spec], anchor_stats)
                gate_score = None if gate_spec is None else score_with_policy_stats(row, [gate_spec], gate_stats)
                selected = anchor_score is not None and anchor_score >= anchor_tau
                if gate_spec is not None:
                    selected = selected and gate_score is not None and gate_tau is not None and gate_score >= gate_tau
                routes.append("baseline" if selected else "method")
                scores.append(anchor_score)
        else:
            specs = [
                (str(item["feature"]), str(item["direction"]))
                for item in policy.get("feature_specs", [])
            ]
            stats = {
                str(feature): {"mean": float(values.get("mean", 0.0)), "std": float(values.get("std", 1.0) or 1.0)}
                for feature, values in dict(policy.get("feature_stats", {})).items()
            }
            tau = float(policy["tau"])
            routes, scores = routes_for_policy(rows, specs, stats, tau)
        evaluation = oracle.aggregate_counts(
            rows,
            lambda row, route_by_obj_id={id(row): route for row, route in zip(rows, routes)}: route_by_obj_id[id(row)],
        )
        intervention = oracle.aggregate_counts(rows, lambda _: "method")
        selected_idx = [idx for idx, route in enumerate(routes) if route == "baseline"]
        labels = [int(row.get("teacher_positive") or 0) for row in rows]
        evaluation.update(
            {
                "delta_chair_i_vs_int": float(evaluation["chair_i"] - intervention["chair_i"]),
                "delta_chair_s_vs_int": float(evaluation["chair_s"] - intervention["chair_s"]),
                "delta_recall_vs_int": float(evaluation["recall"] - intervention["recall"]),
                "delta_f1_vs_int": float(evaluation["f1"] - intervention["f1"]),
                "teacher_precision": oracle.safe_div(
                    float(sum(labels[idx] for idx in selected_idx)),
                    float(len(selected_idx)),
                ),
                "teacher_recall": oracle.safe_div(
                    float(sum(labels[idx] for idx in selected_idx)),
                    float(sum(labels)),
                ),
            }
        )

        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        decision_rows: List[Dict[str, Any]] = []
        for row, route, score in zip(rows, routes, scores):
            out = dict(row)
            out["proxy_route"] = route
            out["proxy_score"] = score
            out["proxy_target_match"] = int((route == "baseline") == (int(row["teacher_positive"]) == 1))
            decision_rows.append(out)
        oracle.write_csv(os.path.join(out_dir, "decision_rows.csv"), decision_rows)
        summary = {
            "inputs": {
                "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
                "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
                "selected_policy_json": os.path.abspath(args.selected_policy_json),
                "chair_i_eps": float(args.chair_i_eps),
                "chair_s_eps": float(args.chair_s_eps),
                "min_recall_gain": float(args.min_recall_gain),
                "require_f1_nondecrease": bool(args.require_f1_nondecrease),
            },
            "counts": {
                "n_rows": int(len(rows)),
                "teacher_positive_rate": oracle.safe_div(
                    float(sum(int(r["teacher_positive"]) for r in rows)),
                    float(len(rows)),
                ),
                "n_proxy_features": int(len(feature_names)),
            },
            "baseline": oracle.aggregate_counts(rows, lambda _: "baseline"),
            "intervention": intervention,
            "teacher_oracle": oracle.aggregate_counts(
                rows,
                lambda row: "baseline" if int(row["teacher_positive"]) == 1 else "method",
            ),
            "applied_proxy_policy": {
                "policy_type": policy.get("policy_type"),
                "feature_specs": policy.get("feature_specs", []),
                "anchor_feature": policy.get("anchor_feature"),
                "anchor_direction": policy.get("anchor_direction"),
                "gate_feature": policy.get("gate_feature"),
                "gate_direction": policy.get("gate_direction"),
                "tau": policy.get("tau"),
                "anchor_tau": policy.get("anchor_tau"),
                "gate_tau": policy.get("gate_tau"),
                **evaluation,
            },
            "outputs": {
                "decision_rows_csv": os.path.join(out_dir, "decision_rows.csv"),
                "summary_json": os.path.join(out_dir, "summary.json"),
            },
        }
        oracle.write_json(os.path.join(out_dir, "summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if bool(args.fit_gated):
        anchor_metric = feature_direction(rows, str(args.anchor_feature))
        if anchor_metric is None:
            raise RuntimeError(f"Anchor feature is not usable: {args.anchor_feature}")
        anchor_spec = (str(args.anchor_feature), str(anchor_metric["direction"]))

        if args.gate_feature:
            gate_candidates = [str(x) for x in args.gate_feature]
        else:
            gate_prefixes = [str(x) for x in args.gate_feature_prefix]
            if gate_prefixes:
                gate_candidates = [
                    name for name in feature_names
                    if any(str(name).startswith(prefix) for prefix in gate_prefixes)
                ]
            else:
                gate_candidates = [name for name in feature_names if name != str(args.anchor_feature)]

        gate_metrics: List[Dict[str, Any]] = []
        for feature in gate_candidates:
            metric = feature_direction(rows, feature)
            if metric is not None:
                gate_metrics.append(metric)
        gate_metrics.sort(
            key=lambda row: (
                -float(row.get("teacher_auroc") or 0.0),
                -float(row.get("teacher_ap") or 0.0),
                str(row.get("feature") or ""),
            )
        )
        gate_metrics = gate_metrics[: max(0, int(args.top_n_features))]

        quantiles = parse_quantiles(args.tau_quantiles)
        gated_results: List[Dict[str, Any]] = []
        all_sweeps: List[Dict[str, Any]] = []
        best_policy: Optional[Dict[str, Any]] = None
        best_sweep: List[Dict[str, Any]] = []

        candidates: List[Optional[Dict[str, Any]]] = [None] + gate_metrics
        for gate_metric in candidates:
            gate_spec = None
            if gate_metric is not None:
                gate_spec = (str(gate_metric["feature"]), str(gate_metric["direction"]))
            sweep, best = evaluate_gated_specs(
                rows,
                anchor_spec=anchor_spec,
                gate_spec=gate_spec,
                quantiles=quantiles,
                max_baseline_rate=float(args.max_baseline_rate),
                chair_i_eps_vs_int=float(args.chair_i_eps),
                chair_s_eps_vs_int=float(args.chair_s_eps),
                min_recall_gain_vs_int=float(args.min_recall_gain),
                require_f1_nondecrease_vs_int=bool(args.require_f1_nondecrease),
                min_teacher_precision=float(args.min_teacher_precision),
                min_teacher_recall=float(args.min_teacher_recall),
            )
            all_sweeps.extend(sweep)
            result: Dict[str, Any] = {
                "anchor_feature": anchor_spec[0],
                "anchor_direction": anchor_spec[1],
                "gate_feature": "" if gate_spec is None else gate_spec[0],
                "gate_direction": "" if gate_spec is None else gate_spec[1],
                "anchor_teacher_auroc": anchor_metric["teacher_auroc"],
                "anchor_teacher_ap": anchor_metric["teacher_ap"],
                "gate_teacher_auroc": None if gate_metric is None else gate_metric["teacher_auroc"],
                "gate_teacher_ap": None if gate_metric is None else gate_metric["teacher_ap"],
                "has_feasible_policy": int(best is not None),
            }
            if best is not None:
                result.update(best)
                if best_policy is None or (
                    float(best["delta_recall_vs_int"]),
                    float(best["delta_f1_vs_int"]),
                    -float(best["delta_chair_i_vs_int"]),
                    -float(best["delta_chair_s_vs_int"]),
                    -float(best["baseline_rate"]),
                ) > (
                    float(best_policy["delta_recall_vs_int"]),
                    float(best_policy["delta_f1_vs_int"]),
                    -float(best_policy["delta_chair_i_vs_int"]),
                    -float(best_policy["delta_chair_s_vs_int"]),
                    -float(best_policy["baseline_rate"]),
                ):
                    best_policy = dict(best)
                    best_sweep = list(sweep)
            gated_results.append(result)

        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        oracle.write_csv(os.path.join(out_dir, "gate_feature_metrics.csv"), gate_metrics)
        oracle.write_csv(os.path.join(out_dir, "gated_results.csv"), gated_results)
        oracle.write_csv(os.path.join(out_dir, "gated_tau_sweep_all.csv"), all_sweeps)

        outputs = {
            "gate_feature_metrics_csv": os.path.join(out_dir, "gate_feature_metrics.csv"),
            "gated_results_csv": os.path.join(out_dir, "gated_results.csv"),
            "gated_tau_sweep_all_csv": os.path.join(out_dir, "gated_tau_sweep_all.csv"),
        }
        if best_policy is not None:
            anchor_stats = json.loads(str(best_policy.get("anchor_feature_stats") or "{}"))
            gate_stats = json.loads(str(best_policy.get("gate_feature_stats") or "{}"))
            selected = {
                "policy_type": "chair_pairwise_caption_gated_proxy_v1",
                "source": "caption-pair recall expert with optional GT-free/extra-feature safety gate; CHAIR GT used only for teacher/evaluation",
                "anchor_feature": str(best_policy["anchor_feature"]),
                "anchor_direction": str(best_policy["anchor_direction"]),
                "anchor_feature_stats": anchor_stats,
                "anchor_tau": float(best_policy["anchor_tau"]),
                "gate_feature": str(best_policy.get("gate_feature") or ""),
                "gate_direction": str(best_policy.get("gate_direction") or ""),
                "gate_feature_stats": gate_stats,
                "gate_tau": best_policy.get("gate_tau"),
            }
            oracle.write_json(os.path.join(out_dir, "selected_policy.json"), selected)
            oracle.write_csv(os.path.join(out_dir, "gated_tau_sweep.csv"), best_sweep)
            outputs.update(
                {
                    "selected_policy_json": os.path.join(out_dir, "selected_policy.json"),
                    "gated_tau_sweep_csv": os.path.join(out_dir, "gated_tau_sweep.csv"),
                }
            )

        summary = {
            "inputs": {
                "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
                "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
                "extra_features_csv": [os.path.abspath(p) for p in args.extra_features_csv],
                "chair_i_eps": float(args.chair_i_eps),
                "chair_s_eps": float(args.chair_s_eps),
                "min_recall_gain": float(args.min_recall_gain),
                "require_f1_nondecrease": bool(args.require_f1_nondecrease),
                "max_baseline_rate": float(args.max_baseline_rate),
                "anchor_feature": str(args.anchor_feature),
                "gate_feature": [str(x) for x in args.gate_feature],
                "gate_feature_prefix": [str(x) for x in args.gate_feature_prefix],
                "top_n_features": int(args.top_n_features),
                "min_teacher_precision": float(args.min_teacher_precision),
                "min_teacher_recall": float(args.min_teacher_recall),
                "tau_quantiles": quantiles,
            },
            "counts": {
                "n_rows": int(len(rows)),
                "teacher_positive_rate": oracle.safe_div(
                    float(sum(int(r["teacher_positive"]) for r in rows)),
                    float(len(rows)),
                ),
                "n_proxy_features": int(len(feature_names)),
                "n_extra_features": int(len(extra_feature_names)),
                "n_gate_candidates": int(len(gate_metrics)),
            },
            "baseline": oracle.aggregate_counts(rows, lambda _: "baseline"),
            "intervention": oracle.aggregate_counts(rows, lambda _: "method"),
            "teacher_oracle": oracle.aggregate_counts(
                rows,
                lambda row: "baseline" if int(row["teacher_positive"]) == 1 else "method",
            ),
            "best_gated_policy": best_policy,
            "outputs": outputs,
        }
        oracle.write_json(os.path.join(out_dir, "summary.json"), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if best_policy is None:
            raise RuntimeError("No feasible gated proxy policy found.")
        return

    if args.feature:
        requested = set(str(x) for x in args.feature)
        feature_names = [name for name in feature_names if name in requested]

    feature_metrics: List[Dict[str, Any]] = []
    directions: Dict[str, str] = {}
    single_best: Dict[str, Dict[str, Any]] = {}
    quantiles = parse_quantiles(args.tau_quantiles)

    all_single_sweeps: List[Dict[str, Any]] = []
    for feature in feature_names:
        metric = feature_direction(rows, feature)
        if metric is None:
            continue
        direction = str(metric["direction"])
        directions[feature] = direction
        sweep, best, _ = evaluate_specs(
            rows,
            [(feature, direction)],
            quantiles,
            max_baseline_rate=float(args.max_baseline_rate),
            chair_i_eps_vs_int=float(args.chair_i_eps),
            chair_s_eps_vs_int=float(args.chair_s_eps),
            min_recall_gain_vs_int=float(args.min_recall_gain),
            require_f1_nondecrease_vs_int=bool(args.require_f1_nondecrease),
            min_teacher_precision=float(args.min_teacher_precision),
            min_teacher_recall=float(args.min_teacher_recall),
        )
        all_single_sweeps.extend(sweep)
        out = dict(metric)
        out["has_feasible_single"] = int(best is not None)
        if best is not None:
            single_best[feature] = dict(best)
            for key in [
                "tau",
                "baseline_rate",
                "chair_s",
                "chair_i",
                "recall",
                "precision",
                "f1",
                "delta_chair_i_vs_int",
                "delta_chair_s_vs_int",
                "delta_recall_vs_int",
                "delta_f1_vs_int",
                "teacher_precision",
                "teacher_recall",
            ]:
                out[f"single_{key}"] = best.get(key)
        feature_metrics.append(out)

    feature_metrics.sort(
        key=lambda row: (
            -int(row.get("has_feasible_single") or 0),
            -float(row.get("single_delta_recall_vs_int") or -1e9),
            -float(row.get("teacher_auroc") or 0.0),
            -float(row.get("teacher_ap") or 0.0),
            str(row.get("feature") or ""),
        )
    )

    top_features = [str(row["feature"]) for row in feature_metrics[: max(1, int(args.top_n_features))]]
    combo_results: List[Dict[str, Any]] = []
    best_combo: Optional[Dict[str, Any]] = None
    best_specs: List[Tuple[str, str]] = []
    best_stats: Dict[str, Dict[str, float]] = {}
    best_sweep: List[Dict[str, Any]] = []
    max_combo_size = max(1, int(args.max_combo_size))

    for size in range(1, max_combo_size + 1):
        for combo_features in itertools.combinations(top_features, size):
            specs = [(feature, directions[feature]) for feature in combo_features]
            sweep, best, stats = evaluate_specs(
                rows,
                specs,
                quantiles,
                max_baseline_rate=float(args.max_baseline_rate),
                chair_i_eps_vs_int=float(args.chair_i_eps),
                chair_s_eps_vs_int=float(args.chair_s_eps),
                min_recall_gain_vs_int=float(args.min_recall_gain),
                require_f1_nondecrease_vs_int=bool(args.require_f1_nondecrease),
                min_teacher_precision=float(args.min_teacher_precision),
                min_teacher_recall=float(args.min_teacher_recall),
            )
            row: Dict[str, Any] = {
                "features": " + ".join(combo_features),
                "feature_specs": json.dumps(
                    [{"feature": feature, "direction": direction} for feature, direction in specs],
                    ensure_ascii=False,
                ),
                "n_features": int(len(specs)),
                "has_feasible_policy": int(best is not None),
            }
            if best is not None:
                row.update(best)
                if best_combo is None or (
                    float(best["delta_recall_vs_int"]),
                    float(best["delta_f1_vs_int"]),
                    -float(best["delta_chair_i_vs_int"]),
                    -float(best["baseline_rate"]),
                ) > (
                    float(best_combo["delta_recall_vs_int"]),
                    float(best_combo["delta_f1_vs_int"]),
                    -float(best_combo["delta_chair_i_vs_int"]),
                    -float(best_combo["baseline_rate"]),
                ):
                    best_combo = dict(best)
                    best_specs = list(specs)
                    best_stats = dict(stats)
                    best_sweep = list(sweep)
            combo_results.append(row)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    oracle.write_csv(os.path.join(out_dir, "feature_metrics.csv"), feature_metrics)
    oracle.write_csv(os.path.join(out_dir, "combo_results.csv"), combo_results)
    oracle.write_csv(os.path.join(out_dir, "single_tau_sweep.csv"), all_single_sweeps)

    baseline = oracle.aggregate_counts(rows, lambda _: "baseline")
    intervention = oracle.aggregate_counts(rows, lambda _: "method")
    teacher_oracle = oracle.aggregate_counts(rows, lambda row: "baseline" if int(row["teacher_positive"]) == 1 else "method")

    outputs = {
        "feature_metrics_csv": os.path.join(out_dir, "feature_metrics.csv"),
        "combo_results_csv": os.path.join(out_dir, "combo_results.csv"),
        "single_tau_sweep_csv": os.path.join(out_dir, "single_tau_sweep.csv"),
    }
    if best_combo is not None:
        oracle.write_csv(os.path.join(out_dir, "tau_sweep.csv"), best_sweep)
        policy = {
            "policy_type": "chair_pairwise_caption_proxy_v1",
            "source": "caption_pair_only; CHAIR GT used only for teacher/evaluation",
            "feature_specs": [
                {"feature": feature, "direction": direction, "weight": 1.0}
                for feature, direction in best_specs
            ],
            "feature_stats": best_stats,
            "tau": float(best_combo["tau"]),
        }
        oracle.write_json(os.path.join(out_dir, "selected_policy.json"), policy)
        routes, scores = routes_for_policy(rows, best_specs, best_stats, float(best_combo["tau"]))
        decision_rows: List[Dict[str, Any]] = []
        for row, route, score in zip(rows, routes, scores):
            out = dict(row)
            out["proxy_route"] = route
            out["proxy_score"] = score
            out["proxy_target_match"] = int((route == "baseline") == (int(row["teacher_positive"]) == 1))
            decision_rows.append(out)
        oracle.write_csv(os.path.join(out_dir, "decision_rows.csv"), decision_rows)
        outputs.update(
            {
                "tau_sweep_csv": os.path.join(out_dir, "tau_sweep.csv"),
                "selected_policy_json": os.path.join(out_dir, "selected_policy.json"),
                "decision_rows_csv": os.path.join(out_dir, "decision_rows.csv"),
            }
        )

    summary = {
        "inputs": {
            "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
            "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
            "chair_i_eps": float(args.chair_i_eps),
            "chair_s_eps": float(args.chair_s_eps),
            "min_recall_gain": float(args.min_recall_gain),
            "require_f1_nondecrease": bool(args.require_f1_nondecrease),
            "max_baseline_rate": float(args.max_baseline_rate),
            "top_n_features": int(args.top_n_features),
            "max_combo_size": int(args.max_combo_size),
            "tau_quantiles": quantiles,
            "min_teacher_precision": float(args.min_teacher_precision),
            "min_teacher_recall": float(args.min_teacher_recall),
        },
        "counts": {
            "n_rows": int(len(rows)),
            "teacher_positive_rate": oracle.safe_div(
                float(sum(int(r["teacher_positive"]) for r in rows)),
                float(len(rows)),
            ),
            "n_proxy_features": int(len(feature_metrics)),
        },
        "baseline": baseline,
        "intervention": intervention,
        "teacher_oracle": teacher_oracle,
        "best_proxy_policy": best_combo,
        "outputs": outputs,
    }
    oracle.write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if best_combo is None:
        raise RuntimeError("No feasible caption-pair proxy policy found.")


if __name__ == "__main__":
    main()
