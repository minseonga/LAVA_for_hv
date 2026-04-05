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


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mu = mean(values)
    var = sum((float(v) - mu) ** 2 for v in values) / float(len(values))
    return float(max(math.sqrt(max(var, 0.0)), 1e-6))


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def infer_case_type(base_correct: Optional[int], int_correct: Optional[int]) -> str:
    if base_correct is None or int_correct is None:
        return "unknown"
    if int(base_correct) == 1 and int(int_correct) == 0:
        return "regression"
    if int(base_correct) == 0 and int(int_correct) == 1:
        return "improvement"
    if int(base_correct) == 1 and int(int_correct) == 1:
        return "both_correct"
    if int(base_correct) == 0 and int(int_correct) == 0:
        return "both_wrong"
    return "unknown"


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


def load_merged_rows(scores_csv: str, features_csv: str) -> List[Dict[str, Any]]:
    score_rows = read_csv_rows(scores_csv)
    feat_rows = read_csv_rows(features_csv)
    feat_map = {str(r.get("id", "")).strip(): r for r in feat_rows}
    merged: List[Dict[str, Any]] = []
    for row in score_rows:
        sid = str(row.get("id", "")).strip()
        feat = feat_map.get(sid, {})
        bc = maybe_int(row.get("baseline_correct"))
        ic = maybe_int(row.get("intervention_correct"))
        case_type = infer_case_type(bc, ic)
        merged_row: Dict[str, Any] = dict(row)
        merged_row["id"] = sid
        merged_row["baseline_correct"] = bc
        merged_row["intervention_correct"] = ic
        merged_row["case_type"] = case_type
        merged_row["harm"] = int(case_type == "regression")
        merged_row["help"] = int(case_type == "improvement")
        for key, value in feat.items():
            if key == "id":
                continue
            merged_row[key] = value
        merged.append(merged_row)
    return merged


def orient_feature(rows: Sequence[Dict[str, Any]], feature: str, target: str = "harm") -> Optional[Dict[str, Any]]:
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
    auc_low = binary_auroc([-float(x) for x in xs], ys)
    if auc_high is None or auc_low is None:
        return None
    direction = "high" if auc_high >= auc_low else "low"
    oriented = [float(x) if direction == "high" else float(-x) for x in xs]
    return {
        "feature": feature,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "mu": mean(oriented),
        "sd": std(oriented),
        "n": int(len(xs)),
        "n_pos": int(sum(ys)),
    }


def oriented_z(row: Dict[str, Any], feat: Dict[str, Any]) -> Optional[float]:
    raw = maybe_float(row.get(str(feat["feature"])))
    if raw is None:
        return None
    oriented = float(raw) if str(feat["direction"]) == "high" else float(-raw)
    mu = float(feat["mu"])
    sd = max(float(feat["sd"]), 1e-6)
    return float((oriented - mu) / sd)


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


def evaluate_tau(
    rows: Sequence[Dict[str, Any]],
    b_feat: Optional[Dict[str, Any]],
    c_feats: Sequence[Dict[str, Any]],
    w_b: float,
    w_c: float,
    tau: float,
) -> Dict[str, Any]:
    n = 0
    selected = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    final_correct = 0
    total_harm = 0
    total_help = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0
    score_values: List[float] = []

    for row in rows:
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        if bc is None or ic is None:
            continue

        b_score = None if b_feat is None else oriented_z(row, b_feat)
        c_zs = [oriented_z(row, feat) for feat in c_feats]
        if any(v is None for v in c_zs):
            c_score = None
        else:
            c_score = None if not c_zs else float(sum(float(v) for v in c_zs if v is not None) / float(len(c_zs)))
        if w_b > 0.0 and b_score is None:
            continue
        if w_c > 0.0 and c_score is None:
            continue

        risk = float(w_b * float(b_score or 0.0) + w_c * float(c_score or 0.0))
        score_values.append(risk)
        n += 1
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        total_harm += harm
        total_help += help_
        baseline_correct_total += int(bc)
        intervention_correct_total += int(ic)

        use_baseline = bool(risk >= float(tau))
        if use_baseline:
            selected += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int((harm == 0) and (help_ == 0))
            final_correct += int(bc)
        else:
            final_correct += int(ic)

    baseline_rate = safe_div(float(selected), float(max(1, n)))
    precision = safe_div(float(selected_harm), float(max(1, selected)))
    recall = safe_div(float(selected_harm), float(max(1, total_harm)))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tau": float(tau),
        "n_eval": int(n),
        "baseline_rate": baseline_rate,
        "method_rate": float(1.0 - baseline_rate),
        "final_acc": safe_div(float(final_correct), float(max(1, n))),
        "baseline_acc": safe_div(float(baseline_correct_total), float(max(1, n))),
        "intervention_acc": safe_div(float(intervention_correct_total), float(max(1, n))),
        "delta_vs_intervention": safe_div(float(final_correct - intervention_correct_total), float(max(1, n))),
        "selected_count": int(selected),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "selected_harm_precision": precision,
        "selected_help_precision": safe_div(float(selected_help), float(max(1, selected))),
        "selected_harm_recall": recall,
        "selected_harm_f1": f1,
    }


def selection_key(result: Dict[str, Any], objective: str) -> Tuple[float, float, float, float]:
    if objective == "harm_f1":
        return (
            float(result["selected_harm_f1"]),
            float(result["selected_harm_precision"]),
            float(result["delta_vs_intervention"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_precision":
        return (
            float(result["selected_harm_precision"]),
            float(result["selected_harm_recall"]),
            float(result["delta_vs_intervention"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_recall":
        return (
            float(result["selected_harm_recall"]),
            float(result["selected_harm_precision"]),
            float(result["delta_vs_intervention"]),
            -float(result["baseline_rate"]),
        )
    return (
        float(result["final_acc"]),
        float(result["delta_vs_intervention"]),
        float(result["selected_harm_precision"]),
        -float(result["baseline_rate"]),
    )


def search_family(
    rows: Sequence[Dict[str, Any]],
    b_feat: Optional[Dict[str, Any]],
    c_feats: Sequence[Dict[str, Any]],
    family: str,
    weight_grid: Sequence[float],
    objective: str,
    min_baseline_rate: float,
    max_baseline_rate: float,
    min_selected_count: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    if family == "b_only":
        weight_pairs = [(float(w), 0.0) for w in weight_grid if float(w) > 0.0]
    elif family == "c_only":
        weight_pairs = [(0.0, float(w)) for w in weight_grid if float(w) > 0.0]
    else:
        weight_pairs = [(float(wb), float(wc)) for wb in weight_grid for wc in weight_grid if float(wb) > 0.0 and float(wc) > 0.0]

    for w_b, w_c in weight_pairs:
        scores: List[float] = []
        for row in rows:
            b_score = None if b_feat is None else oriented_z(row, b_feat)
            c_zs = [oriented_z(row, feat) for feat in c_feats]
            if any(v is None for v in c_zs):
                c_score = None
            else:
                c_score = None if not c_zs else float(sum(float(v) for v in c_zs if v is not None) / float(len(c_zs)))
            if w_b > 0.0 and b_score is None:
                continue
            if w_c > 0.0 and c_score is None:
                continue
            scores.append(float(w_b * float(b_score or 0.0) + w_c * float(c_score or 0.0)))
        for tau in threshold_grid(scores):
            result = evaluate_tau(rows, b_feat=b_feat, c_feats=c_feats, w_b=w_b, w_c=w_c, tau=tau)
            result.update(
                {
                    "family": family,
                    "w_b": float(w_b),
                    "w_c": float(w_c),
                    "b_feature": None if b_feat is None else str(b_feat["feature"]),
                    "c_features": ",".join(str(f["feature"]) for f in c_feats),
                    "n_c_features": int(len(c_feats)),
                }
            )
            candidates.append(result)
            if int(result["selected_count"]) < int(min_selected_count):
                continue
            if float(result["baseline_rate"]) < float(min_baseline_rate):
                continue
            if float(result["baseline_rate"]) > float(max_baseline_rate):
                continue
            if best is None or selection_key(result, objective) > selection_key(best, objective):
                best = result
    return best, candidates


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a soft post-hoc B/C fusion controller from stage-B scores and cheap features.")
    ap.add_argument("--scores_csv", type=str, required=True)
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--b_feature_cols", type=str, default="stage_b_score")
    ap.add_argument(
        "--c_feature_cols",
        type=str,
        default="cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy",
    )
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k_c", type=int, default=3)
    ap.add_argument("--weight_grid", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--tau_objective", type=str, default="final_acc", choices=["final_acc", "harm_f1", "harm_precision", "harm_recall"])
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    ap.add_argument("--min_selected_count", type=int, default=0)
    args = ap.parse_args()

    rows = load_merged_rows(os.path.abspath(args.scores_csv), os.path.abspath(args.features_csv))
    b_feature_names = [x.strip() for x in str(args.b_feature_cols).split(",") if x.strip()]
    c_feature_names = [x.strip() for x in str(args.c_feature_cols).split(",") if x.strip()]
    weight_grid = [float(x.strip()) for x in str(args.weight_grid).split(",") if x.strip()]

    b_metrics: List[Dict[str, Any]] = []
    for feat in b_feature_names:
        result = orient_feature(rows, feat, target="harm")
        if result is not None:
            b_metrics.append(result)
    b_metrics.sort(key=lambda r: (-float(r["auroc"]), str(r["feature"])))
    best_b = b_metrics[0] if b_metrics else None

    c_metrics: List[Dict[str, Any]] = []
    for feat in c_feature_names:
        result = orient_feature(rows, feat, target="harm")
        if result is not None:
            c_metrics.append(result)
    c_metrics.sort(key=lambda r: (-float(r["auroc"]), str(r["feature"])))
    selected_c = [r for r in c_metrics if float(r["auroc"]) >= float(args.min_feature_auroc)]
    if int(args.top_k_c) > 0:
        selected_c = selected_c[: int(args.top_k_c)]

    all_candidates: List[Dict[str, Any]] = []
    best_results: Dict[str, Dict[str, Any]] = {}

    if best_b is not None:
        best, cand = search_family(
            rows,
            b_feat=best_b,
            c_feats=[],
            family="b_only",
            weight_grid=weight_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
        )
        all_candidates.extend(cand)
        if best is not None:
            best_results["b_only"] = best

    if selected_c:
        best, cand = search_family(
            rows,
            b_feat=None,
            c_feats=selected_c,
            family="c_only",
            weight_grid=weight_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
        )
        all_candidates.extend(cand)
        if best is not None:
            best_results["c_only"] = best

    if best_b is not None and selected_c:
        best, cand = search_family(
            rows,
            b_feat=best_b,
            c_feats=selected_c,
            family="fusion",
            weight_grid=weight_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
        )
        all_candidates.extend(cand)
        if best is not None:
            best_results["fusion"] = best

    candidates_csv = os.path.join(args.out_dir, "fusion_candidates.csv")
    b_metrics_csv = os.path.join(args.out_dir, "b_feature_metrics.csv")
    c_metrics_csv = os.path.join(args.out_dir, "c_feature_metrics.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")

    write_csv(candidates_csv, all_candidates)
    write_csv(b_metrics_csv, b_metrics)
    write_csv(c_metrics_csv, c_metrics)
    write_json(
        summary_json,
        {
            "inputs": {
                "scores_csv": os.path.abspath(args.scores_csv),
                "features_csv": os.path.abspath(args.features_csv),
                "b_feature_cols": b_feature_names,
                "c_feature_cols": c_feature_names,
                "min_feature_auroc": float(args.min_feature_auroc),
                "top_k_c": int(args.top_k_c),
                "weight_grid": weight_grid,
                "tau_objective": str(args.tau_objective),
                "min_baseline_rate": float(args.min_baseline_rate),
                "max_baseline_rate": float(args.max_baseline_rate),
                "min_selected_count": int(args.min_selected_count),
            },
            "counts": {
                "n_rows": int(len(rows)),
                "n_harm": int(sum(int(row.get("harm", 0) or 0) for row in rows)),
                "n_help": int(sum(int(row.get("help", 0) or 0) for row in rows)),
            },
            "best_b_feature": best_b,
            "selected_c_features": selected_c,
            "best_results": best_results,
            "outputs": {
                "fusion_candidates_csv": os.path.abspath(candidates_csv),
                "b_feature_metrics_csv": os.path.abspath(b_metrics_csv),
                "c_feature_metrics_csv": os.path.abspath(c_metrics_csv),
            },
        },
    )
    print("[saved]", os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
