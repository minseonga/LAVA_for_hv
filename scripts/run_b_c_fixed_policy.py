#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CHEAP_FEATURES = [
    "cheap_lp_content_min",
    "cheap_lp_content_tail_gap",
    "cheap_lp_content_tail_z",
    "cheap_lp_content_q10",
    "cheap_lp_content_min_len_corr",
    "cheap_target_gap_content_min",
    "cheap_lp_content_std",
    "cheap_entropy_content_mean",
    "cheap_margin_content_mean",
    "cheap_target_gap_content_std",
    "cheap_conflict_lp_minus_entropy",
]


def maybe_float(value: object) -> Optional[float]:
    s = str(value or "").strip()
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
    f = maybe_float(value)
    if f is None:
        return None
    return int(round(f))


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


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def parse_percents(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text or "").split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    return out or [1.0, 2.0, 5.0]


def select_feature_names(rows: Sequence[Dict[str, Any]], feature_cols: str) -> List[str]:
    if str(feature_cols or "").strip():
        names = [part.strip() for part in str(feature_cols).split(",") if part.strip()]
    else:
        names = list(DEFAULT_CHEAP_FEATURES)
    if not rows:
        return []
    keys = set(rows[0].keys())
    return [name for name in names if name in keys]


def threshold_grid(values: Sequence[float]) -> List[float]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return [0.0]
    if len(finite) == 1:
        return [finite[0]]
    out = {finite[0], finite[-1]}
    for q in [i / 100.0 for i in range(1, 100)]:
        pos = q * float(len(finite) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.add(finite[lo])
        else:
            w = pos - float(lo)
            out.add((1.0 - w) * finite[lo] + w * finite[hi])
    return sorted(out)


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


def orient_single_feature(
    rows: Sequence[Dict[str, Any]],
    feature: str,
) -> Tuple[str, Optional[float], Optional[float]]:
    feat_rows = [row for row in rows if maybe_float(row.get(feature)) is not None]
    values = [float(row[feature]) for row in feat_rows]
    imp_rows = [row for row in feat_rows if row["case_type"] in {"vga_regression", "vga_improvement"}]
    imp_values = [float(row[feature]) for row in imp_rows]
    imp_labels = [1 if row["case_type"] == "vga_regression" else 0 for row in imp_rows]
    reg_labels = [1 if row["case_type"] == "vga_regression" else 0 for row in feat_rows]

    high_auc_imp = binary_auc(imp_values, imp_labels) if imp_rows else None
    low_auc_imp = binary_auc([-float(v) for v in imp_values], imp_labels) if imp_rows else None
    high_auc_reg = binary_auc(values, reg_labels)
    low_auc_reg = binary_auc([-float(v) for v in values], reg_labels)
    high_key = (-1.0 if high_auc_imp is None else high_auc_imp, -1.0 if high_auc_reg is None else high_auc_reg)
    low_key = (-1.0 if low_auc_imp is None else low_auc_imp, -1.0 if low_auc_reg is None else low_auc_reg)
    if low_key > high_key:
        return "low", low_auc_reg, low_auc_imp
    return "high", high_auc_reg, high_auc_imp


def subset_by_tau_b(rows: Sequence[Dict[str, Any]], tau_b: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        score = maybe_float(row.get("stage_b_score"))
        if score is None:
            continue
        if float(score) <= float(tau_b):
            out.append(row)
    return out


def tau_b_from_percent(rows: Sequence[Dict[str, Any]], percent: float) -> Optional[float]:
    valid = [
        row for row in rows
        if maybe_float(row.get("stage_b_score")) is not None
        and row.get("baseline_correct") is not None
        and row.get("intervention_correct") is not None
        and str(row.get("case_type", "")) != "unknown"
    ]
    ordered = sorted(valid, key=lambda row: float(row["stage_b_score"]))
    if not ordered:
        return None
    k = max(1, int(math.ceil(float(percent) / 100.0 * float(len(ordered)))))
    return float(ordered[k - 1]["stage_b_score"])


def evaluate_policy(
    all_rows: Sequence[Dict[str, Any]],
    tau_b: float,
    feature: str,
    direction: str,
    tau_c: float,
) -> Dict[str, Any]:
    n_total = 0
    intervention_correct_total = 0
    final_correct = 0
    subset_count = 0
    rescue_count = 0
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
        n_total += 1
        intervention_correct_total += int(ic)

        score_b = maybe_float(row.get("stage_b_score"))
        feat_val = maybe_float(row.get(feature))
        in_subset = score_b is not None and float(score_b) <= float(tau_b)
        if in_subset:
            subset_count += 1
            if row["case_type"] == "vga_regression":
                subset_regression_total += 1

        rescue = False
        if in_subset and feat_val is not None:
            oriented = -float(feat_val) if direction == "low" else float(feat_val)
            rescue = float(oriented) >= float(tau_c)

        if rescue:
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


def calibrate_policy(args: argparse.Namespace) -> None:
    score_rows = load_csv_rows(args.scores_csv)
    taxonomy_rows = load_csv_rows(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else None
    feature_rows = load_csv_rows(args.features_csv) if str(args.features_csv).strip() else None
    merged_rows = merge_rows(score_rows, taxonomy_rows, feature_rows)
    feature_names = select_feature_names(merged_rows, args.feature_cols)

    candidate_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for percent in parse_percents(args.subset_percents):
        tau_b = tau_b_from_percent(merged_rows, percent)
        if tau_b is None:
            continue
        subset_rows = subset_by_tau_b(merged_rows, tau_b)
        for feature in feature_names:
            feat_subset = [row for row in subset_rows if maybe_float(row.get(feature)) is not None]
            if len(feat_subset) < 4:
                continue
            direction, auc_reg, auc_imp = orient_single_feature(feat_subset, feature)
            oriented_values = []
            for row in feat_subset:
                value = float(row[feature])
                oriented_values.append(-value if direction == "low" else value)
            for tau_c in threshold_grid(oriented_values):
                policy = evaluate_policy(
                    all_rows=merged_rows,
                    tau_b=float(tau_b),
                    feature=feature,
                    direction=direction,
                    tau_c=float(tau_c),
                )
                row_out = {
                    "subset_percent": float(percent),
                    "tau_b": float(tau_b),
                    "feature": feature,
                    "direction_for_regression": direction,
                    "tau_c": float(tau_c),
                    "auroc_reg_vs_non_rescue": auc_reg,
                    "auroc_reg_vs_improvement": auc_imp,
                    **policy,
                }
                if policy["rescue_rate"] is not None and float(policy["rescue_rate"]) > float(args.max_rescue_rate):
                    candidate_rows.append(row_out)
                    continue
                candidate_rows.append(row_out)
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

    if best is None:
        raise RuntimeError("No feasible calibration candidate found.")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    candidates_csv = os.path.join(out_dir, "calibration_candidates.csv")
    selected_json = os.path.join(out_dir, "selected_policy.json")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(candidates_csv, candidate_rows)
    write_json(selected_json, {
        "subset_percent": float(best["subset_percent"]),
        "tau_b": float(best["tau_b"]),
        "feature": str(best["feature"]),
        "direction_for_regression": str(best["direction_for_regression"]),
        "tau_c": float(best["tau_c"]),
    })
    write_json(summary_json, {
        "mode": "calibrate",
        "inputs": {
            "scores_csv": os.path.abspath(args.scores_csv),
            "taxonomy_csv": (os.path.abspath(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else ""),
            "features_csv": (os.path.abspath(args.features_csv) if str(args.features_csv).strip() else ""),
            "subset_percents": parse_percents(args.subset_percents),
            "feature_cols": feature_names,
            "max_rescue_rate": float(args.max_rescue_rate),
        },
        "selected_policy": {
            "subset_percent": float(best["subset_percent"]),
            "tau_b": float(best["tau_b"]),
            "feature": str(best["feature"]),
            "direction_for_regression": str(best["direction_for_regression"]),
            "tau_c": float(best["tau_c"]),
            "calibration_metrics": {
                k: best[k]
                for k in [
                    "final_acc",
                    "delta_vs_intervention",
                    "subset_rate",
                    "rescue_rate",
                    "rescue_precision_regression",
                    "subset_regression_recall",
                    "rescued_improvement_share",
                    "rescued_both_wrong_share",
                    "rescued_both_correct_share",
                    "auroc_reg_vs_non_rescue",
                    "auroc_reg_vs_improvement",
                ]
            },
        },
        "outputs": {
            "candidates_csv": os.path.abspath(candidates_csv),
            "selected_policy_json": os.path.abspath(selected_json),
        },
    })
    print("[saved]", candidates_csv)
    print("[saved]", selected_json)
    print("[saved]", summary_json)


def apply_policy(args: argparse.Namespace) -> None:
    score_rows = load_csv_rows(args.scores_csv)
    taxonomy_rows = load_csv_rows(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else None
    feature_rows = load_csv_rows(args.features_csv) if str(args.features_csv).strip() else None
    merged_rows = merge_rows(score_rows, taxonomy_rows, feature_rows)

    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)

    tau_b = float(policy["tau_b"])
    feature = str(policy["feature"])
    direction = str(policy["direction_for_regression"])
    tau_c = float(policy["tau_c"])

    summary = evaluate_policy(
        all_rows=merged_rows,
        tau_b=tau_b,
        feature=feature,
        direction=direction,
        tau_c=tau_c,
    )

    decision_rows: List[Dict[str, Any]] = []
    for row in merged_rows:
        score_b = maybe_float(row.get("stage_b_score"))
        feat_val = maybe_float(row.get(feature))
        in_subset = score_b is not None and float(score_b) <= float(tau_b)
        oriented = None if feat_val is None else (-float(feat_val) if direction == "low" else float(feat_val))
        rescue = bool(in_subset and oriented is not None and float(oriented) >= float(tau_c))
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        final_correct = None
        if bc is not None and ic is not None:
            final_correct = int(bc) if rescue else int(ic)
        decision_rows.append({
            "id": row.get("id"),
            "case_type": row.get("case_type"),
            "stage_b_score": score_b,
            "b_flag": int(in_subset),
            "feature": feature,
            "feature_raw_value": feat_val,
            "feature_oriented_value": oriented,
            "tau_b": tau_b,
            "tau_c": tau_c,
            "c_flag": (None if oriented is None else int(float(oriented) >= float(tau_c))),
            "rescue": int(rescue),
            "baseline_correct": bc,
            "intervention_correct": ic,
            "final_correct": final_correct,
        })

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    decisions_csv = os.path.join(out_dir, "decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(decisions_csv, decision_rows)
    write_json(summary_json, {
        "mode": "apply",
        "inputs": {
            "scores_csv": os.path.abspath(args.scores_csv),
            "taxonomy_csv": (os.path.abspath(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else ""),
            "features_csv": (os.path.abspath(args.features_csv) if str(args.features_csv).strip() else ""),
            "policy_json": os.path.abspath(args.policy_json),
        },
        "policy": {
            "subset_percent": float(policy.get("subset_percent", "nan")),
            "tau_b": tau_b,
            "feature": feature,
            "direction_for_regression": direction,
            "tau_c": tau_c,
        },
        "evaluation": summary,
        "outputs": {
            "decision_rows_csv": os.path.abspath(decisions_csv),
        },
    })
    print("[saved]", decisions_csv)
    print("[saved]", summary_json)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate/apply a fixed B+C rescue policy.")
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_cal = sub.add_parser("calibrate")
    ap_cal.add_argument("--scores_csv", type=str, required=True)
    ap_cal.add_argument("--taxonomy_csv", type=str, default="")
    ap_cal.add_argument("--features_csv", type=str, default="")
    ap_cal.add_argument("--subset_percents", type=str, default="1,2,5")
    ap_cal.add_argument("--feature_cols", type=str, default="")
    ap_cal.add_argument("--max_rescue_rate", type=float, default=0.03)
    ap_cal.add_argument("--out_dir", type=str, required=True)

    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--scores_csv", type=str, required=True)
    ap_apply.add_argument("--taxonomy_csv", type=str, default="")
    ap_apply.add_argument("--features_csv", type=str, default="")
    ap_apply.add_argument("--policy_json", type=str, required=True)
    ap_apply.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()
    if args.mode == "calibrate":
        calibrate_policy(args)
    elif args.mode == "apply":
        apply_policy(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
