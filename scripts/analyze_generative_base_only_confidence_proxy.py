#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from extract_generative_semantic_pairwise_features import (
    normalize_token,
    read_prediction_map,
    sorted_preview,
    unit_summary,
    write_csv,
    write_json,
)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def min_or_zero(values: Sequence[float]) -> float:
    return float(min(values)) if values else 0.0


def max_or_zero(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    idx = min(len(xs) - 1, max(0, int(round(float(q) * (len(xs) - 1)))))
    return float(xs[idx])


def load_trace_rows(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row)
        if sid:
            out[sid] = row
    return out


def parse_word_trace(row: Dict[str, str]) -> List[Dict[str, Any]]:
    raw = str(row.get("probe_content_word_trace_json", "")).strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        word = normalize_token(str(item.get("word", "")))
        if not word:
            continue
        lp = safe_float(item.get("lp_min"))
        gap = safe_float(item.get("gap_min"))
        ent = safe_float(item.get("ent_max"))
        if lp is None or gap is None or ent is None:
            continue
        out.append({"word": word, "pos": item.get("pos", 0), "lp": lp, "gap": gap, "ent": ent})
    return out


def auc_high(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    good = 0.0
    total = 0
    for a in pos:
        for b in neg:
            total += 1
            if a > b:
                good += 1.0
            elif a == b:
                good += 0.5
    return good / float(total) if total else None


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
    ranked = sorted(items, key=lambda item: item[1], reverse=True)[: int(k)]
    if not ranked:
        return None
    return sum(label for label, _ in ranked) / float(len(ranked))


def feature_metrics(rows: Sequence[Dict[str, Any]], target_col: str) -> List[Dict[str, Any]]:
    if not rows:
        return []
    excluded = {"id", "image", "base_text", "int_text", "bo_token_units", "bo_trace_words"}
    metrics: List[Dict[str, Any]] = []
    for feature in rows[0]:
        if feature in excluded:
            continue
        pairs: List[Tuple[int, float]] = []
        for row in rows:
            value = safe_float(row.get(feature))
            if value is not None:
                pairs.append((flag(row.get(target_col)), value))
        if len(pairs) < max(10, int(0.8 * len(rows))):
            continue
        if len({round(score, 12) for _, score in pairs}) < 3:
            continue
        pos = [score for label, score in pairs if label]
        neg = [score for label, score in pairs if not label]
        auc = auc_high(pos, neg)
        if auc is None:
            continue
        direction = "high" if auc >= 0.5 else "low"
        oriented = [(label, score if direction == "high" else -score) for label, score in pairs]
        metrics.append(
            {
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
                "pos_mean": mean(pos),
                "neg_mean": mean(neg),
            }
        )
    metrics.sort(key=lambda row: (float(row.get("auroc") or 0.0), float(row.get("ap") or 0.0)), reverse=True)
    return metrics


def add_trace_stats(row: Dict[str, Any], prefix: str, traces: Sequence[Dict[str, Any]]) -> None:
    lp = [float(t["lp"]) for t in traces]
    gap = [float(t["gap"]) for t in traces]
    ent = [float(t["ent"]) for t in traces]
    row[f"{prefix}_trace_count"] = len(traces)
    row[f"{prefix}_lp_mean"] = mean(lp)
    row[f"{prefix}_lp_min"] = min_or_zero(lp)
    row[f"{prefix}_lp_q10"] = quantile(lp, 0.10)
    row[f"{prefix}_gap_mean"] = mean(gap)
    row[f"{prefix}_gap_min"] = min_or_zero(gap)
    row[f"{prefix}_gap_q10"] = quantile(gap, 0.10)
    row[f"{prefix}_ent_mean"] = mean(ent)
    row[f"{prefix}_ent_max"] = max_or_zero(ent)
    row[f"{prefix}_ent_q90"] = quantile(ent, 0.90)
    row[f"{prefix}_high_gap_count_ge_000"] = sum(1 for v in gap if v >= 0.0)
    row[f"{prefix}_high_gap_count_ge_025"] = sum(1 for v in gap if v >= 0.25)
    row[f"{prefix}_high_gap_count_ge_050"] = sum(1 for v in gap if v >= 0.50)
    row[f"{prefix}_high_lp_count_ge_m1"] = sum(1 for v in lp if v >= -1.0)
    row[f"{prefix}_high_lp_count_ge_m2"] = sum(1 for v in lp if v >= -2.0)
    row[f"{prefix}_low_ent_count_le_200"] = sum(1 for v in ent if v <= 2.0)
    row[f"{prefix}_low_ent_count_le_250"] = sum(1 for v in ent if v <= 2.5)
    row[f"{prefix}_confident_count_gap000_ent250"] = sum(1 for g, e in zip(gap, ent) if g >= 0.0 and e <= 2.5)
    row[f"{prefix}_confident_count_gap025_ent250"] = sum(1 for g, e in zip(gap, ent) if g >= 0.25 and e <= 2.5)
    row[f"{prefix}_confidence_score_gap_pos"] = sum(max(0.0, v) for v in gap)
    row[f"{prefix}_confidence_score_lp_plus_gap"] = sum(max(0.0, g) + max(0.0, lpv + 2.0) for lpv, g in zip(lp, gap))
    row[f"{prefix}_risk_score_low_gap_high_ent"] = sum(max(0.0, -g) + max(0.0, e - 3.0) for g, e in zip(gap, ent))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze whether baseline-only caption content has high baseline token confidence."
    )
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--baseline_trace_csv", required=True)
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_feature_metrics_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--preview_limit", type=int, default=80)
    args = ap.parse_args()

    baseline = read_prediction_map(args.baseline_pred_jsonl)
    intervention = read_prediction_map(args.intervention_pred_jsonl)
    traces = load_trace_rows(args.baseline_trace_csv)
    oracle = {safe_id(row): row for row in read_csv_rows(args.oracle_rows_csv)}

    ids = sorted(
        set(baseline.keys()) & set(intervention.keys()) & set(traces.keys()) & set(oracle.keys()),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        b = unit_summary(baseline[sid]["text"])
        i = unit_summary(intervention[sid]["text"])
        base_tokens = set(b["unique_token_units"])
        int_tokens = set(i["unique_token_units"])
        base_only_tokens = base_tokens - int_tokens
        shared_tokens = base_tokens & int_tokens
        word_trace = parse_word_trace(traces[sid])
        base_only_trace = [t for t in word_trace if str(t["word"]) in base_only_tokens]
        shared_trace = [t for t in word_trace if str(t["word"]) in shared_tokens]

        row: Dict[str, Any] = {
            "id": sid,
            "image": baseline[sid].get("image") or intervention[sid].get("image") or "",
            args.target_col: flag(oracle[sid].get(args.target_col)),
            "base_text": baseline[sid]["text"],
            "int_text": intervention[sid]["text"],
            "bo_token_units": sorted_preview(base_only_tokens, int(args.preview_limit)),
            "bo_trace_words": sorted_preview([t["word"] for t in base_only_trace], int(args.preview_limit)),
            "bo_token_count": len(base_only_tokens),
            "shared_token_count": len(shared_tokens),
            "base_token_count": len(base_tokens),
            "int_token_count": len(int_tokens),
            "bo_trace_coverage_rate": len({t["word"] for t in base_only_trace}) / float(max(1, len(base_only_tokens))),
            "bo_count_x_trace_coverage": len(base_only_tokens)
            * (len({t["word"] for t in base_only_trace}) / float(max(1, len(base_only_tokens)))),
        }
        add_trace_stats(row, "bo_conf", base_only_trace)
        add_trace_stats(row, "shared_conf", shared_trace)
        row["bo_minus_shared_lp_mean"] = float(row["bo_conf_lp_mean"] - row["shared_conf_lp_mean"])
        row["bo_minus_shared_gap_mean"] = float(row["bo_conf_gap_mean"] - row["shared_conf_gap_mean"])
        row["bo_minus_shared_ent_mean"] = float(row["bo_conf_ent_mean"] - row["shared_conf_ent_mean"])
        row["bo_high_gap_count_x_bo_count"] = float(row["bo_conf_high_gap_count_ge_000"] * len(base_only_tokens))
        row["bo_confidence_score_x_bo_count"] = float(row["bo_conf_confidence_score_gap_pos"] * len(base_only_tokens))
        rows.append(row)

    metrics = feature_metrics(rows, args.target_col)
    n_target = sum(flag(row.get(args.target_col)) for row in rows)
    summary = {
        "inputs": {
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "baseline_trace_csv": os.path.abspath(args.baseline_trace_csv),
            "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
            "target_col": args.target_col,
        },
        "counts": {
            "n_rows": len(rows),
            "n_target": int(n_target),
            "target_rate": float(n_target / float(max(1, len(rows)))),
            "n_feature_metrics": len(metrics),
        },
        "top_feature_metrics": metrics[:30],
        "outputs": {
            "features_csv": os.path.abspath(args.out_csv),
            "feature_metrics_csv": os.path.abspath(args.out_feature_metrics_csv),
            "summary_json": os.path.abspath(args.out_summary_json),
        },
    }

    write_csv(args.out_csv, rows)
    write_csv(args.out_feature_metrics_csv, metrics)
    write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_feature_metrics_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))
    for metric in metrics[:10]:
        print(
            "[metric]",
            metric["feature"],
            "dir=",
            metric["direction"],
            "auc=",
            metric["auroc"],
            "ap=",
            metric["ap"],
        )


if __name__ == "__main__":
    main()
