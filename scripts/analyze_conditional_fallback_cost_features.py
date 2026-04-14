#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TARGET_COL = "oracle_recall_gain_f1_nondecrease_ci_unique_noworse"
DEFAULT_EXCLUDE_SUBSTRINGS = (
    "_names",
    "_details_json",
    "_error",
    "_text",
    "caption",
    "image",
)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("id") or row.get("image_id") or row.get("question_id") or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def flag(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / float(len(values))) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    idx = q * (len(vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def auc_high(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for a in pos:
        for b in neg:
            total += 1
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / float(total) if total else None


def average_precision(items: Sequence[Tuple[int, float]]) -> Optional[float]:
    ranked = sorted(items, key=lambda item: item[1], reverse=True)
    n_pos = sum(label for label, _ in ranked)
    if n_pos <= 0:
        return None
    hits = 0
    total = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label:
            hits += 1
            total += hits / float(rank)
    return total / float(n_pos)


def precision_at(items: Sequence[Tuple[int, float]], k: int) -> Optional[float]:
    top = sorted(items, key=lambda item: item[1], reverse=True)[: int(k)]
    if not top:
        return None
    return sum(label for label, _ in top) / float(len(top))


def is_numeric_feature(rows: Sequence[Dict[str, Any]], feature: str) -> bool:
    seen = 0
    unique: set[float] = set()
    for row in rows:
        value = safe_float(row.get(feature))
        if value is None:
            continue
        seen += 1
        unique.add(round(value, 12))
        if seen >= 10 and len(unique) >= 3:
            return True
    return False


def numeric_features(rows: Sequence[Dict[str, Any]], target_col: str) -> List[str]:
    if not rows:
        return []
    out: List[str] = []
    for feature in rows[0].keys():
        if feature == target_col or feature in {"id", "image_id", "question_id"}:
            continue
        if feature.startswith("audit_"):
            continue
        if any(part in feature for part in DEFAULT_EXCLUDE_SUBSTRINGS):
            continue
        if is_numeric_feature(rows, feature):
            out.append(feature)
    return out


def feature_metric(rows: Sequence[Dict[str, Any]], target_col: str, feature: str, min_valid_frac: float) -> Optional[Dict[str, Any]]:
    pairs: List[Tuple[int, float]] = []
    for row in rows:
        value = safe_float(row.get(feature))
        if value is None:
            continue
        pairs.append((flag(row.get(target_col)), value))
    if len(pairs) < max(8, int(float(min_valid_frac) * len(rows))):
        return None
    if len({round(score, 12) for _, score in pairs}) < 3:
        return None
    pos = [score for label, score in pairs if label]
    neg = [score for label, score in pairs if not label]
    auc = auc_high(pos, neg)
    if auc is None:
        return None
    direction = "high" if auc >= 0.5 else "low"
    oriented = [(label, score if direction == "high" else -score) for label, score in pairs]
    return {
        "feature": feature,
        "direction": direction,
        "auroc": max(float(auc), float(1.0 - auc)),
        "auroc_high": float(auc),
        "ap": average_precision(oriented),
        "p_at_10": precision_at(oriented, 10),
        "p_at_25": precision_at(oriented, 25),
        "p_at_50": precision_at(oriented, 50),
        "n": len(pairs),
        "n_pos": sum(label for label, _ in pairs),
        "target_rate": sum(label for label, _ in pairs) / float(len(pairs)) if pairs else 0.0,
        "pos_mean": mean(pos),
        "neg_mean": mean(neg),
    }


def parse_gate(spec: str) -> Tuple[str, str, str, float]:
    parts = str(spec).split("::")
    if len(parts) != 4:
        raise ValueError(f"Gate spec must be name::feature::direction::threshold, got: {spec}")
    name, feature, direction, threshold = parts
    direction = direction.strip().lower()
    if direction not in {"high", "low"}:
        raise ValueError(f"Gate direction must be high or low, got: {direction}")
    return name.strip(), feature.strip(), direction, float(threshold)


def make_gate_fn(feature: str, direction: str, threshold: float) -> Callable[[Dict[str, Any]], bool]:
    def gate(row: Dict[str, Any]) -> bool:
        value = safe_float(row.get(feature))
        if value is None:
            return False
        if direction == "high":
            return value >= threshold
        return value <= threshold

    return gate


def default_gates(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, Callable[[Dict[str, Any]], bool], Dict[str, Any]]]:
    gates: List[Tuple[str, Callable[[Dict[str, Any]], bool], Dict[str, Any]]] = [
        ("all", lambda row: True, {"feature": "", "direction": "", "threshold": ""}),
    ]
    fixed = [
        ("base_only_ge1", "capobjyn_base_only_object_count", "high", 1.0),
        ("base_only_ge2", "capobjyn_base_only_object_count", "high", 2.0),
        ("verified_base_only_ge1", "capobjyn_verified_base_only_count", "high", 1.0),
        ("has_verified_base_no_verified_int", "capobjyn_has_verified_base_only_no_verified_int_only", "high", 1.0),
    ]
    for name, feature, direction, threshold in fixed:
        if rows and feature in rows[0]:
            gates.append(
                (
                    name,
                    make_gate_fn(feature, direction, threshold),
                    {"feature": feature, "direction": direction, "threshold": threshold},
                )
            )
    quantile_specs = [
        ("jaccard_gap_q70", "capobjyn_base_only_x_jaccard_gap", "high", 0.70),
        ("jaccard_gap_q80", "capobjyn_base_only_x_jaccard_gap", "high", 0.80),
        ("rollback_gain_q70", "capobjyn_rollback_gain", "high", 0.70),
        ("rollback_gain_q80", "capobjyn_rollback_gain", "high", 0.80),
    ]
    for name, feature, direction, q in quantile_specs:
        values = [safe_float(row.get(feature)) for row in rows]
        vals = [float(v) for v in values if v is not None]
        if len(vals) < 10:
            continue
        threshold = percentile(vals, q)
        gates.append(
            (
                name,
                make_gate_fn(feature, direction, threshold),
                {"feature": feature, "direction": direction, "threshold": threshold, "quantile": q},
            )
        )
    return gates


def selected_stats(rows: Sequence[Dict[str, Any]], target_col: str, all_pos: int) -> Dict[str, Any]:
    n = len(rows)
    n_pos = sum(flag(row.get(target_col)) for row in rows)
    return {
        "n_selected": n,
        "n_target": n_pos,
        "target_precision": n_pos / float(n) if n else 0.0,
        "target_recall": n_pos / float(all_pos) if all_pos else 0.0,
    }


def rule_candidates(
    rows: Sequence[Dict[str, Any]],
    target_col: str,
    gate_name: str,
    features: Sequence[str],
    quantiles: Sequence[float],
    min_selected: int,
    max_selected: int,
    min_valid_frac: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    all_pos = sum(flag(row.get(target_col)) for row in rows)
    for feature in features:
        metric = feature_metric(rows, target_col, feature, min_valid_frac)
        if metric is None:
            continue
        direction = str(metric["direction"])
        vals = [safe_float(row.get(feature)) for row in rows]
        values = [float(v) for v in vals if v is not None]
        if len(values) < max(8, int(min_valid_frac * len(rows))):
            continue
        qs = [1.0 - q if direction == "high" else q for q in quantiles]
        for q, raw_q in zip(quantiles, qs):
            tau = percentile(values, raw_q)
            if direction == "high":
                selected = [row for row in rows if (safe_float(row.get(feature)) is not None and float(safe_float(row.get(feature))) >= tau)]
            else:
                selected = [row for row in rows if (safe_float(row.get(feature)) is not None and float(safe_float(row.get(feature))) <= tau)]
            if len(selected) < int(min_selected) or len(selected) > int(max_selected):
                continue
            stats = selected_stats(selected, target_col, all_pos)
            candidates.append(
                {
                    "gate": gate_name,
                    "feature": feature,
                    "direction": direction,
                    "tau": tau,
                    "quantile": q,
                    "feature_auroc_in_gate": metric["auroc"],
                    "feature_ap_in_gate": metric["ap"],
                    **stats,
                    "selected_ids": "|".join(safe_id(row) for row in selected[:200]),
                }
            )
    candidates.sort(
        key=lambda row: (
            float(row.get("target_precision", 0.0)),
            float(row.get("target_recall", 0.0)),
            float(row.get("feature_auroc_in_gate", 0.0)),
        ),
        reverse=True,
    )
    return candidates


def attach_oracle_fields(rows: List[Dict[str, Any]], oracle_rows_csv: str) -> None:
    if not str(oracle_rows_csv or "").strip() or not os.path.isfile(os.path.abspath(oracle_rows_csv)):
        return
    oracle = {safe_id(row): row for row in read_csv_rows(oracle_rows_csv)}
    for row in rows:
        o = oracle.get(safe_id(row))
        if not o:
            continue
        for key in (
            "n_base_only_supported_unique",
            "n_int_only_supported_unique",
            "n_base_only_hallucinated_unique",
            "n_int_only_hallucinated_unique",
            "delta_recall_base_minus_int",
            "delta_f1_unique_base_minus_int",
            "delta_ci_unique_base_minus_int",
            "failure_type",
        ):
            if key in o:
                row[f"audit_{key}"] = o[key]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze cost/net-benefit features after a benefit candidate gate.")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--target_col", default=DEFAULT_TARGET_COL)
    ap.add_argument("--oracle_rows_csv", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gate", action="append", default=[], help="name::feature::high|low::threshold")
    ap.add_argument("--quantiles", default="0.05,0.08,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--min_selected", type=int, default=5)
    ap.add_argument("--max_selected", type=int, default=120)
    ap.add_argument("--min_valid_frac", type=float, default=0.8)
    args = ap.parse_args()

    rows = read_csv_rows(args.features_csv)
    attach_oracle_fields(rows, args.oracle_rows_csv)
    target_col = str(args.target_col)
    features = numeric_features(rows, target_col)
    all_pos = sum(flag(row.get(target_col)) for row in rows)
    all_target_rate = all_pos / float(max(1, len(rows)))

    gates = default_gates(rows)
    for spec in args.gate:
        name, feature, direction, threshold = parse_gate(spec)
        gates.append(
            (
                name,
                make_gate_fn(feature, direction, threshold),
                {"feature": feature, "direction": direction, "threshold": threshold},
            )
        )

    gate_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    quantiles = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]

    for gate_name, gate_fn, gate_meta in gates:
        selected = [row for row in rows if gate_fn(row)]
        stats = selected_stats(selected, target_col, all_pos)
        gate_summary = {
            "gate": gate_name,
            **gate_meta,
            **stats,
            "target_lift_vs_all": stats["target_precision"] / all_target_rate if all_target_rate else 0.0,
        }
        if selected and "audit_n_base_only_supported_unique" in selected[0]:
            has_gold = [
                row
                for row in selected
                if (safe_float(row.get("audit_n_base_only_supported_unique")) or 0.0) > 0.0
            ]
            gate_summary["audit_has_base_only_supported_rate"] = len(has_gold) / float(len(selected))
            gate_summary["audit_mean_delta_f1_unique_base_minus_int"] = mean(
                [
                    safe_float(row.get("audit_delta_f1_unique_base_minus_int")) or 0.0
                    for row in selected
                ]
            )
            gate_summary["audit_mean_delta_ci_unique_base_minus_int"] = mean(
                [
                    safe_float(row.get("audit_delta_ci_unique_base_minus_int")) or 0.0
                    for row in selected
                ]
            )
        gate_rows.append(gate_summary)

        for feature in features:
            metric = feature_metric(selected, target_col, feature, float(args.min_valid_frac))
            if metric is None:
                continue
            metric_rows.append({"gate": gate_name, **gate_meta, **metric})

        candidate_rows.extend(
            rule_candidates(
                selected,
                target_col,
                gate_name,
                features,
                quantiles,
                min_selected=int(args.min_selected),
                max_selected=int(args.max_selected),
                min_valid_frac=float(args.min_valid_frac),
            )
        )

    metric_rows.sort(
        key=lambda row: (
            str(row.get("gate", "")) != "all",
            float(row.get("auroc", 0.0)),
            float(row.get("ap") or 0.0),
        ),
        reverse=True,
    )
    candidate_rows.sort(
        key=lambda row: (
            float(row.get("target_precision", 0.0)),
            float(row.get("target_recall", 0.0)),
            float(row.get("feature_auroc_in_gate", 0.0)),
        ),
        reverse=True,
    )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    gate_csv = os.path.join(out_dir, "conditional_gate_summary.csv")
    metrics_csv = os.path.join(out_dir, "conditional_feature_metrics.csv")
    candidates_csv = os.path.join(out_dir, "conditional_rule_candidates.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(gate_csv, gate_rows)
    write_csv(metrics_csv, metric_rows)
    write_csv(candidates_csv, candidate_rows)
    summary = {
        "inputs": {
            "features_csv": os.path.abspath(args.features_csv),
            "target_col": target_col,
            "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv) if str(args.oracle_rows_csv or "").strip() else "",
            "quantiles": quantiles,
            "min_selected": int(args.min_selected),
            "max_selected": int(args.max_selected),
            "min_valid_frac": float(args.min_valid_frac),
        },
        "counts": {
            "n_rows": len(rows),
            "n_target": all_pos,
            "target_rate": all_target_rate,
            "n_numeric_features": len(features),
            "n_gates": len(gates),
            "n_conditional_metrics": len(metric_rows),
            "n_rule_candidates": len(candidate_rows),
        },
        "top_gates": gate_rows[:20],
        "top_conditional_metrics": metric_rows[:40],
        "top_rule_candidates": candidate_rows[:40],
        "outputs": {
            "gate_summary_csv": gate_csv,
            "feature_metrics_csv": metrics_csv,
            "rule_candidates_csv": candidates_csv,
            "summary_json": summary_json,
        },
    }
    write_json(summary_json, summary)
    print("[saved]", gate_csv)
    print("[saved]", metrics_csv)
    print("[saved]", candidates_csv)
    print("[saved]", summary_json)
    for row in metric_rows[:15]:
        print(
            "[metric]",
            "gate=",
            row.get("gate"),
            row.get("feature"),
            "dir=",
            row.get("direction"),
            "auc=",
            row.get("auroc"),
            "ap=",
            row.get("ap"),
        )
    for row in candidate_rows[:10]:
        print(
            "[rule]",
            row.get("gate"),
            row.get("feature"),
            row.get("direction"),
            row.get("tau"),
            "n=",
            row.get("n_selected"),
            "prec=",
            row.get("target_precision"),
            "rec=",
            row.get("target_recall"),
        )


if __name__ == "__main__":
    main()
