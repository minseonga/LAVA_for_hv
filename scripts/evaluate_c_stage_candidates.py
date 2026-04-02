#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_FEATURES = [
    "faithful_minus_global_attn",
    "harmful_minus_faithful",
    "faithful_head_attn_mean",
    "faithful_head_attn_topkmean",
    "guidance_mismatch_score",
    "harmful_minus_global_attn",
    "harmful_head_attn_mean",
    "harmful_head_attn_topkmean",
    "early_attn_mean",
    "late_attn_mean",
    "late_uplift",
    "U_bad",
    "U_unified",
    "stage_a_score",
    "stage_a_gap_mean",
    "stage_a_faithful_mean",
    "stage_a_harmful_mean",
    "stage_a_faithful_std",
]


def parse_bool(text: object) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def maybe_float(value: object) -> Optional[float]:
    s = str(value or "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: object) -> Optional[int]:
    f = maybe_float(value)
    if f is None:
        return None
    return int(round(f))


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
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


def infer_case_type(base_correct: Optional[int], int_correct: Optional[int]) -> str:
    if base_correct is None or int_correct is None:
        return "unknown"
    if int(base_correct) == 1 and int(int_correct) == 0:
        return "vga_regression"
    if int(base_correct) == 0 and int(int_correct) == 1:
        return "vga_improvement"
    if int(base_correct) == 1 and int(int_correct) == 1:
        return "both_correct"
    if int(base_correct) == 0 and int(int_correct) == 0:
        return "both_wrong"
    return "unknown"


def assign_avg_ranks(scores: Sequence[float]) -> List[float]:
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and float(scores[order[j]]) == float(scores[order[i]]):
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pos = sum(1 for y in labels if int(y) == 1)
    neg = sum(1 for y in labels if int(y) == 0)
    if pos == 0 or neg == 0:
        return None
    ranks = assign_avg_ranks(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if int(y) == 1)
    u = rank_sum_pos - float(pos * (pos + 1)) / 2.0
    return float(u / float(pos * neg))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pos = sum(1 for y in labels if int(y) == 1)
    if pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    tp = 0
    ap = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(labels[idx]) == 1:
            tp += 1
            ap += float(tp) / float(rank)
    return float(ap / float(pos))


def mean(values: Iterable[float]) -> Optional[float]:
    seq = [float(v) for v in values]
    if not seq:
        return None
    return float(sum(seq) / float(len(seq)))


def std(values: Iterable[float]) -> Optional[float]:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0 if seq else None
    mu = mean(seq)
    if mu is None:
        return None
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(math.sqrt(max(0.0, var)))


def parse_percents(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text or "").split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    return out or [0.5, 1.0, 2.0, 5.0]


def merge_rows(
    score_rows: Sequence[Dict[str, str]],
    taxonomy_rows: Optional[Sequence[Dict[str, str]]],
    feature_rows: Optional[Sequence[Dict[str, str]]],
) -> List[Dict[str, Any]]:
    tax_map = {str(row.get("id", "")).strip(): row for row in (taxonomy_rows or [])}
    feat_map = {str(row.get("id", "")).strip(): row for row in (feature_rows or [])}
    merged: List[Dict[str, Any]] = []
    for row in score_rows:
        sid = str(row.get("id", "")).strip()
        tax = tax_map.get(sid, {})
        feat = feat_map.get(sid, {})
        base_correct = maybe_int(row.get("baseline_correct"))
        if base_correct is None:
            base_correct = maybe_int(tax.get("baseline_ok"))
        int_correct = maybe_int(row.get("intervention_correct"))
        if int_correct is None:
            int_correct = maybe_int(tax.get("vga_ok"))
        case_type = str(tax.get("case_type", "")).strip() or infer_case_type(base_correct, int_correct)

        merged_row: Dict[str, Any] = dict(row)
        merged_row["id"] = sid
        merged_row["baseline_correct"] = base_correct
        merged_row["intervention_correct"] = int_correct
        merged_row["case_type"] = case_type
        for key, value in tax.items():
            if key not in merged_row or merged_row[key] in {"", None}:
                merged_row[key] = value
        for key, value in feat.items():
            if key == "id":
                continue
            if key not in merged_row or merged_row[key] in {"", None}:
                merged_row[key] = value
        merged.append(merged_row)
    return merged


def subset_by_bottom_percent(rows: Sequence[Dict[str, Any]], percent: float) -> List[Dict[str, Any]]:
    valid = [
        row
        for row in rows
        if maybe_float(row.get("stage_b_score")) is not None
        and row.get("baseline_correct") is not None
        and row.get("intervention_correct") is not None
        and str(row.get("case_type", "")) != "unknown"
    ]
    ordered = sorted(valid, key=lambda row: float(row["stage_b_score"]))
    if not ordered:
        return []
    k = max(1, int(math.ceil(float(percent) / 100.0 * float(len(ordered)))))
    return ordered[:k]


def select_feature_names(rows: Sequence[Dict[str, Any]], feature_cols: str) -> List[str]:
    if str(feature_cols or "").strip():
        names = [part.strip() for part in str(feature_cols).split(",") if part.strip()]
    else:
        names = list(DEFAULT_FEATURES)
    available = []
    if not rows:
        return available
    keys = set(rows[0].keys())
    for name in names:
        if name in keys:
            available.append(name)
    return available


def threshold_grid(values: Sequence[float]) -> List[float]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return [0.0]
    if len(finite) == 1:
        return [finite[0]]
    quantiles = {finite[0], finite[-1]}
    for q in [i / 100.0 for i in range(1, 100)]:
        pos = q * float(len(finite) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            quantiles.add(finite[lo])
        else:
            w = pos - float(lo)
            quantiles.add((1.0 - w) * finite[lo] + w * finite[hi])
    return sorted(quantiles)


def orient_scores(values: Sequence[float], labels_regression: Sequence[int], labels_improvement: Sequence[int]) -> Tuple[str, List[float], Optional[float], Optional[float]]:
    high_auc_imp = binary_auc(values, labels_improvement)
    low_auc_imp = binary_auc([-float(v) for v in values], labels_improvement)
    high_auc_reg = binary_auc(values, labels_regression)
    low_auc_reg = binary_auc([-float(v) for v in values], labels_regression)

    high_key = (
        -1.0 if high_auc_imp is None else high_auc_imp,
        -1.0 if high_auc_reg is None else high_auc_reg,
    )
    low_key = (
        -1.0 if low_auc_imp is None else low_auc_imp,
        -1.0 if low_auc_reg is None else low_auc_reg,
    )
    if low_key > high_key:
        return "low", [-float(v) for v in values], low_auc_reg, low_auc_imp
    return "high", list(values), high_auc_reg, high_auc_imp


def zscore(values: Sequence[float]) -> List[float]:
    mu = mean(values)
    sigma = std(values)
    if mu is None:
        return [0.0 for _ in values]
    if sigma is None or sigma <= 1e-12:
        return [0.0 for _ in values]
    return [(float(v) - float(mu)) / float(sigma) for v in values]


def subset_summary_row(name: str, rows: Sequence[Dict[str, Any]], percent: float) -> Dict[str, Any]:
    n = len(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[str(row["case_type"])] += 1
    return {
        "subset": name,
        "bottom_percent": float(percent),
        "n": int(n),
        "n_regression": int(counts["vga_regression"]),
        "n_improvement": int(counts["vga_improvement"]),
        "n_both_wrong": int(counts["both_wrong"]),
        "n_both_correct": int(counts["both_correct"]),
        "regression_share": (None if n == 0 else float(counts["vga_regression"] / float(n))),
        "improvement_share": (None if n == 0 else float(counts["vga_improvement"] / float(n))),
        "both_wrong_share": (None if n == 0 else float(counts["both_wrong"] / float(n))),
        "both_correct_share": (None if n == 0 else float(counts["both_correct"] / float(n))),
        "stage_b_score_mean": mean(float(row["stage_b_score"]) for row in rows),
        "stage_b_score_std": std(float(row["stage_b_score"]) for row in rows),
    }


def evaluate_policy(
    all_rows: Sequence[Dict[str, Any]],
    subset_ids: Sequence[str],
    rescue_ids: Sequence[str],
) -> Dict[str, Any]:
    subset_set = set(subset_ids)
    rescue_set = set(rescue_ids)
    n_total = 0
    intervention_correct_total = 0
    final_correct = 0
    rescue_count = 0
    subset_count = 0
    subset_regression_total = 0
    rescued_regression = 0
    rescued_improvement = 0
    rescued_both_wrong = 0
    rescued_both_correct = 0

    for row in all_rows:
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        if bc is None or ic is None:
            continue
        sid = str(row["id"])
        n_total += 1
        intervention_correct_total += int(ic)
        in_subset = sid in subset_set
        if in_subset:
            subset_count += 1
            if row["case_type"] == "vga_regression":
                subset_regression_total += 1
        if sid in rescue_set:
            rescue_count += 1
            final_correct += int(bc)
            if row["case_type"] == "vga_regression":
                rescued_regression += 1
            elif row["case_type"] == "vga_improvement":
                rescued_improvement += 1
            elif row["case_type"] == "both_wrong":
                rescued_both_wrong += 1
            elif row["case_type"] == "both_correct":
                rescued_both_correct += 1
        else:
            final_correct += int(ic)

    rescue_precision = None if rescue_count == 0 else float(rescued_regression / float(rescue_count))
    return {
        "n_eval": int(n_total),
        "subset_size": int(subset_count),
        "rescue_count": int(rescue_count),
        "subset_rate": (None if n_total == 0 else float(subset_count / float(n_total))),
        "rescue_rate": (None if n_total == 0 else float(rescue_count / float(n_total))),
        "intervention_acc": (None if n_total == 0 else float(intervention_correct_total / float(n_total))),
        "final_acc": (None if n_total == 0 else float(final_correct / float(n_total))),
        "delta_vs_intervention": (None if n_total == 0 else float((final_correct - intervention_correct_total) / float(n_total))),
        "rescue_precision_regression": rescue_precision,
        "subset_regression_recall": (None if subset_regression_total == 0 else float(rescued_regression / float(subset_regression_total))),
        "rescued_improvement_share": (None if rescue_count == 0 else float(rescued_improvement / float(rescue_count))),
        "rescued_both_wrong_share": (None if rescue_count == 0 else float(rescued_both_wrong / float(rescue_count))),
        "rescued_both_correct_share": (None if rescue_count == 0 else float(rescued_both_correct / float(rescue_count))),
    }


def evaluate_single_features(
    all_rows: Sequence[Dict[str, Any]],
    subset_rows: Sequence[Dict[str, Any]],
    subset_name: str,
    bottom_percent: float,
    features: Sequence[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    subset_ids = [str(row["id"]) for row in subset_rows]

    for feat in features:
        feat_rows = [row for row in subset_rows if maybe_float(row.get(feat)) is not None]
        if len(feat_rows) < 4:
            continue
        values = [float(row[feat]) for row in feat_rows]
        labels_reg = [1 if row["case_type"] == "vga_regression" else 0 for row in feat_rows]
        imp_rows = [row for row in feat_rows if row["case_type"] in {"vga_regression", "vga_improvement"}]
        labels_imp = [1 if row["case_type"] == "vga_regression" else 0 for row in imp_rows]
        direction, oriented_values, auc_reg, auc_imp = orient_scores(
            values=values,
            labels_regression=labels_reg,
            labels_improvement=labels_imp,
        )
        oriented_imp_values = [
            score
            for row, score in zip(feat_rows, oriented_values)
            if row["case_type"] in {"vga_regression", "vga_improvement"}
        ]

        thresholds = threshold_grid(oriented_values)
        best: Optional[Dict[str, Any]] = None
        for tau in thresholds:
            rescue_ids = [str(row["id"]) for row, score in zip(feat_rows, oriented_values) if float(score) >= float(tau)]
            policy = evaluate_policy(all_rows=all_rows, subset_ids=subset_ids, rescue_ids=rescue_ids)
            row_out = {
                "subset": subset_name,
                "bottom_percent": float(bottom_percent),
                "method": "single_feature",
                "feature": feat,
                "direction_for_regression": direction,
                "tau_c": float(tau),
                "auroc_reg_vs_non_rescue": auc_reg,
                "auroc_reg_vs_improvement": (None if len(oriented_imp_values) == 0 else binary_auc(oriented_imp_values, labels_imp)),
                **policy,
            }
            if best is None:
                best = row_out
                continue
            cand_key = (
                -1.0 if row_out["final_acc"] is None else float(row_out["final_acc"]),
                -1.0 if row_out["rescue_precision_regression"] is None else float(row_out["rescue_precision_regression"]),
                -float(row_out["rescue_rate"] or 0.0),
            )
            best_key = (
                -1.0 if best["final_acc"] is None else float(best["final_acc"]),
                -1.0 if best["rescue_precision_regression"] is None else float(best["rescue_precision_regression"]),
                -float(best["rescue_rate"] or 0.0),
            )
            if cand_key > best_key:
                best = row_out
        if best is not None:
            out.append(best)
    return out


def evaluate_pair_features(
    all_rows: Sequence[Dict[str, Any]],
    subset_rows: Sequence[Dict[str, Any]],
    subset_name: str,
    bottom_percent: float,
    single_rows: Sequence[Dict[str, Any]],
    topn_for_pairs: int,
) -> List[Dict[str, Any]]:
    top_features = sorted(
        single_rows,
        key=lambda row: (
            -1.0 if row["auroc_reg_vs_improvement"] is None else float(row["auroc_reg_vs_improvement"]),
            -1.0 if row["auroc_reg_vs_non_rescue"] is None else float(row["auroc_reg_vs_non_rescue"]),
        ),
        reverse=True,
    )[:topn_for_pairs]
    if len(top_features) < 2:
        return []

    feat_meta = {(row["feature"], row["direction_for_regression"]) for row in top_features}
    out: List[Dict[str, Any]] = []
    subset_ids = [str(row["id"]) for row in subset_rows]

    for (feat_a, dir_a), (feat_b, dir_b) in combinations(sorted(feat_meta), 2):
        pair_rows = [
            row
            for row in subset_rows
            if maybe_float(row.get(feat_a)) is not None and maybe_float(row.get(feat_b)) is not None
        ]
        if len(pair_rows) < 4:
            continue
        vals_a = [float(row[feat_a]) for row in pair_rows]
        vals_b = [float(row[feat_b]) for row in pair_rows]
        if dir_a == "low":
            vals_a = [-float(v) for v in vals_a]
        if dir_b == "low":
            vals_b = [-float(v) for v in vals_b]
        z_a = zscore(vals_a)
        z_b = zscore(vals_b)
        combined = [(a + b) / 2.0 for a, b in zip(z_a, z_b)]
        labels_reg = [1 if row["case_type"] == "vga_regression" else 0 for row in pair_rows]
        imp_rows = [row for row in pair_rows if row["case_type"] in {"vga_regression", "vga_improvement"}]
        imp_combined = [
            score
            for row, score in zip(pair_rows, combined)
            if row["case_type"] in {"vga_regression", "vga_improvement"}
        ]
        imp_labels = [1 if row["case_type"] == "vga_regression" else 0 for row in imp_rows]
        auc_reg = binary_auc(combined, labels_reg)
        auc_imp = binary_auc(imp_combined, imp_labels) if imp_labels else None

        thresholds = threshold_grid(combined)
        best: Optional[Dict[str, Any]] = None
        for tau in thresholds:
            rescue_ids = [str(row["id"]) for row, score in zip(pair_rows, combined) if float(score) >= float(tau)]
            policy = evaluate_policy(all_rows=all_rows, subset_ids=subset_ids, rescue_ids=rescue_ids)
            row_out = {
                "subset": subset_name,
                "bottom_percent": float(bottom_percent),
                "method": "pair_feature",
                "feature": f"{feat_a}+{feat_b}",
                "direction_for_regression": f"{dir_a}+{dir_b}",
                "tau_c": float(tau),
                "auroc_reg_vs_non_rescue": auc_reg,
                "auroc_reg_vs_improvement": auc_imp,
                **policy,
            }
            if best is None:
                best = row_out
                continue
            cand_key = (
                -1.0 if row_out["final_acc"] is None else float(row_out["final_acc"]),
                -1.0 if row_out["rescue_precision_regression"] is None else float(row_out["rescue_precision_regression"]),
                -float(row_out["rescue_rate"] or 0.0),
            )
            best_key = (
                -1.0 if best["final_acc"] is None else float(best["final_acc"]),
                -1.0 if best["rescue_precision_regression"] is None else float(best["rescue_precision_regression"]),
                -float(best["rescue_rate"] or 0.0),
            )
            if cand_key > best_key:
                best = row_out
        if best is not None:
            out.append(best)
    return out


def top_rows_by_subset(rows: Sequence[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subset"])].append(row)
    out: List[Dict[str, Any]] = []
    for subset, group_rows in grouped.items():
        ranked = sorted(
            group_rows,
            key=lambda row: (
                -1.0 if row["final_acc"] is None else float(row["final_acc"]),
                -1.0 if row["rescue_precision_regression"] is None else float(row["rescue_precision_regression"]),
                -float(row["rescue_rate"] or 0.0),
            ),
            reverse=True,
        )[:topk]
        out.extend(ranked)
    return out


def build_baseline_rows(all_rows: Sequence[Dict[str, Any]], subset_rows: Sequence[Dict[str, Any]], subset_name: str, bottom_percent: float) -> List[Dict[str, Any]]:
    subset_ids = [str(row["id"]) for row in subset_rows]
    rescue_all = evaluate_policy(all_rows=all_rows, subset_ids=subset_ids, rescue_ids=subset_ids)
    intervention_only = evaluate_policy(all_rows=all_rows, subset_ids=subset_ids, rescue_ids=[])
    out = [
        {"subset": subset_name, "bottom_percent": float(bottom_percent), "method": "intervention_only", **intervention_only},
        {"subset": subset_name, "bottom_percent": float(bottom_percent), "method": "rescue_all_in_subset", **rescue_all},
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate C-stage rescue candidates inside B-low subsets.")
    ap.add_argument("--scores_csv", type=str, required=True)
    ap.add_argument("--taxonomy_csv", type=str, default="")
    ap.add_argument("--features_csv", type=str, default="")
    ap.add_argument("--subset_percents", type=str, default="0.5,1,2,5")
    ap.add_argument("--feature_cols", type=str, default="")
    ap.add_argument("--pair_feature_topn", type=int, default=8)
    ap.add_argument("--topk_rows", type=int, default=8)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    score_rows = load_csv_rows(args.scores_csv)
    taxonomy_rows = load_csv_rows(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else None
    feature_rows = load_csv_rows(args.features_csv) if str(args.features_csv).strip() else None
    merged_rows = merge_rows(score_rows=score_rows, taxonomy_rows=taxonomy_rows, feature_rows=feature_rows)
    feature_names = select_feature_names(merged_rows, feature_cols=args.feature_cols)
    os.makedirs(args.out_dir, exist_ok=True)

    subset_summary_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    single_rows_all: List[Dict[str, Any]] = []
    pair_rows_all: List[Dict[str, Any]] = []

    for percent in parse_percents(args.subset_percents):
        subset_name = f"bottom_{str(percent).replace('.', 'p')}pct"
        subset_rows = subset_by_bottom_percent(merged_rows, percent=percent)
        subset_summary_rows.append(subset_summary_row(subset_name, subset_rows, percent))
        baseline_rows.extend(build_baseline_rows(merged_rows, subset_rows, subset_name, percent))
        single_rows = evaluate_single_features(
            all_rows=merged_rows,
            subset_rows=subset_rows,
            subset_name=subset_name,
            bottom_percent=percent,
            features=feature_names,
        )
        single_rows_all.extend(single_rows)
        pair_rows_all.extend(
            evaluate_pair_features(
                all_rows=merged_rows,
                subset_rows=subset_rows,
                subset_name=subset_name,
                bottom_percent=percent,
                single_rows=single_rows,
                topn_for_pairs=int(args.pair_feature_topn),
            )
        )

    single_top = top_rows_by_subset(single_rows_all, topk=int(args.topk_rows))
    pair_top = top_rows_by_subset(pair_rows_all, topk=int(args.topk_rows))

    subset_csv = os.path.join(args.out_dir, "subset_summary.csv")
    baseline_csv = os.path.join(args.out_dir, "baseline_policies.csv")
    single_csv = os.path.join(args.out_dir, "single_feature_results.csv")
    single_top_csv = os.path.join(args.out_dir, "single_feature_top.csv")
    pair_csv = os.path.join(args.out_dir, "pair_feature_results.csv")
    pair_top_csv = os.path.join(args.out_dir, "pair_feature_top.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")

    write_csv(subset_csv, subset_summary_rows)
    write_csv(baseline_csv, baseline_rows)
    write_csv(single_csv, single_rows_all)
    write_csv(single_top_csv, single_top)
    write_csv(pair_csv, pair_rows_all)
    write_csv(pair_top_csv, pair_top)

    summary = {
        "inputs": {
            "scores_csv": os.path.abspath(args.scores_csv),
            "taxonomy_csv": (os.path.abspath(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else ""),
            "features_csv": (os.path.abspath(args.features_csv) if str(args.features_csv).strip() else ""),
            "subset_percents": parse_percents(args.subset_percents),
            "feature_cols": feature_names,
            "pair_feature_topn": int(args.pair_feature_topn),
        },
        "counts": {
            "n_rows": int(len(merged_rows)),
            "n_features": int(len(feature_names)),
            "n_single_results": int(len(single_rows_all)),
            "n_pair_results": int(len(pair_rows_all)),
        },
        "outputs": {
            "subset_csv": os.path.abspath(subset_csv),
            "baseline_csv": os.path.abspath(baseline_csv),
            "single_csv": os.path.abspath(single_csv),
            "single_top_csv": os.path.abspath(single_top_csv),
            "pair_csv": os.path.abspath(pair_csv),
            "pair_top_csv": os.path.abspath(pair_top_csv),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", subset_csv)
    print("[saved]", baseline_csv)
    print("[saved]", single_csv)
    print("[saved]", single_top_csv)
    print("[saved]", pair_csv)
    print("[saved]", pair_top_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
