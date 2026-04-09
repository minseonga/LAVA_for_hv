#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import build_generative_b_c_meta_controller as base
import build_generative_pareto_teacher_controller as linear


def parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


def parse_budget_list(spec: str) -> List[float]:
    vals = [float(x) for x in parse_float_list(spec) if float(x) > 0.0]
    out = sorted({min(1.0, max(0.0, v)) for v in vals})
    return out or [0.02, 0.05, 0.1]


def build_routes_from_budget(scores: Sequence[Optional[float]], budget_rate: float) -> List[str]:
    valid = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None and math.isfinite(float(score))]
    n = len(scores)
    k = int(round(float(budget_rate) * float(n)))
    k = max(0, min(n, k))
    routes = ["method"] * n
    if k <= 0 or not valid:
        return routes
    valid.sort(key=lambda x: x[1], reverse=True)
    chosen = {idx for idx, _ in valid[:k]}
    for idx in chosen:
        routes[idx] = "baseline"
    return routes


def teacher_stats(rows: Sequence[Dict[str, Any]], routes: Sequence[str]) -> Dict[str, float]:
    n_selected = sum(1 for route in routes if route == "baseline")
    tp = sum(
        int(base.maybe_int(row.get("teacher_fallback")) or 0)
        for row, route in zip(rows, routes)
        if route == "baseline"
    )
    n_pos = sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows)
    return {
        "teacher_precision": base.safe_div(float(tp), float(max(1, n_selected))),
        "teacher_recall": base.safe_div(float(tp), float(max(1, n_pos))),
        "baseline_rate": base.safe_div(float(n_selected), float(max(1, len(rows)))),
    }


def oriented_scores(rows: Sequence[Dict[str, Any]], feature: str, direction: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for row in rows:
        val = base.maybe_float(row.get(feature))
        if val is None:
            out.append(None)
            continue
        out.append(float(val) if direction == "high" else -float(val))
    return out


def add_budget_metrics(
    out_row: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    scores: Sequence[Optional[float]],
    budgets: Sequence[float],
    objective: str,
) -> None:
    intervention = base.aggregate_routes(rows, ["method"] * len(rows))
    for budget in budgets:
        key = f"b{int(round(float(budget) * 100.0)):02d}"
        routes = build_routes_from_budget(scores, budget)
        stats = teacher_stats(rows, routes)
        summary = base.aggregate_routes(rows, routes)
        out_row[f"{key}_baseline_rate"] = float(summary["baseline_rate"])
        out_row[f"{key}_teacher_precision"] = float(stats["teacher_precision"])
        out_row[f"{key}_teacher_recall"] = float(stats["teacher_recall"])
        out_row[f"{key}_mean_chair_i"] = float(summary["mean_chair_i"])
        out_row[f"{key}_mean_chair_s"] = float(summary["mean_chair_s"])
        out_row[f"{key}_mean_recall"] = float(summary["mean_recall"])
        out_row[f"{key}_mean_precision"] = float(summary["mean_precision"])
        out_row[f"{key}_mean_f1"] = float(summary["mean_f1"])
        out_row[f"{key}_mean_f1_minus_chairi"] = float(summary["mean_f1_minus_chairi"])
        out_row[f"{key}_mean_claim_utility"] = float(summary["mean_claim_utility"])
        out_row[f"{key}_delta_chair_i_vs_int"] = float(summary["mean_chair_i"] - intervention["mean_chair_i"])
        out_row[f"{key}_delta_chair_s_vs_int"] = float(summary["mean_chair_s"] - intervention["mean_chair_s"])
        out_row[f"{key}_delta_f1_vs_int"] = float(summary["mean_f1"] - intervention["mean_f1"])
        out_row[f"{key}_delta_f1_minus_chairi_vs_int"] = float(
            summary["mean_f1_minus_chairi"] - intervention["mean_f1_minus_chairi"]
        )
        out_row[f"{key}_delta_claim_utility_vs_int"] = float(
            summary["mean_claim_utility"] - intervention["mean_claim_utility"]
        )
        out_row[f"{key}_objective"] = float(base.objective_value(summary, objective))


def sort_rows(rows: Sequence[Dict[str, Any]], budgets: Sequence[float]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    primary = f"b{int(round(float(budgets[0]) * 100.0)):02d}_teacher_precision"
    secondary = f"b{int(round(float(budgets[0]) * 100.0)):02d}_delta_f1_minus_chairi_vs_int"
    out = [dict(row) for row in rows]
    out.sort(
        key=lambda r: (
            -float(r.get(primary) or 0.0),
            -float(r.get(secondary) or 0.0),
            -float(r.get("auroc") or 0.0),
            str(r.get("feature") or ""),
        )
    )
    return out


def top_rows(rows: Sequence[Dict[str, Any]], family: Optional[str], k: int) -> List[Dict[str, Any]]:
    subset = [dict(row) for row in rows if family is None or str(row.get("family")) == family]
    return subset[: max(0, int(k))]


def budget_key(budget: float) -> str:
    return f"b{int(round(float(budget) * 100.0)):02d}"


def select_candidate_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    budget: float,
    min_precision: float,
    min_recall: float,
    min_delta_f1_minus_chairi: float,
    max_delta_chair_i: float,
    max_delta_chair_s: float,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    key = budget_key(budget)
    selected: List[Dict[str, Any]] = []
    for row in rows:
        p = float(row.get(f"{key}_teacher_precision") or 0.0)
        r = float(row.get(f"{key}_teacher_recall") or 0.0)
        df = float(row.get(f"{key}_delta_f1_minus_chairi_vs_int") or 0.0)
        dci = float(row.get(f"{key}_delta_chair_i_vs_int") or 0.0)
        dcs = float(row.get(f"{key}_delta_chair_s_vs_int") or 0.0)
        if p < float(min_precision):
            continue
        if r < float(min_recall):
            continue
        if df < float(min_delta_f1_minus_chairi):
            continue
        if dci > float(max_delta_chair_i):
            continue
        if dcs > float(max_delta_chair_s):
            continue
        selected.append(dict(row))
        if int(max_candidates) > 0 and len(selected) >= int(max_candidates):
            break
    return selected


def write_candidate_features(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row["feature"]).strip())
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine generative fallback features by small-budget precision and aggregate lift.")
    ap.add_argument("--claim_table_csv", type=str, required=True)
    ap.add_argument("--chair_table_csv", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--teacher_mode", type=str, default="chairi_pareto", choices=["strict_pareto", "chairi_pareto", "f1_only"])
    ap.add_argument("--min_f1_gain", type=float, default=0.0)
    ap.add_argument("--feature_cols", type=str, default="auto")
    ap.add_argument("--feature_prefix", type=str, default="")
    ap.add_argument("--budgets", type=str, default="0.02,0.05,0.10,0.15")
    ap.add_argument("--selection_objective", type=str, default="f1_minus_chairi", choices=["f1", "neg_chairi", "claim_utility", "f1_minus_chairi"])
    ap.add_argument("--top_k_summary", type=int, default=20)
    ap.add_argument("--candidate_budget", type=float, default=0.05)
    ap.add_argument("--min_candidate_precision", type=float, default=0.0)
    ap.add_argument("--min_candidate_recall", type=float, default=0.0)
    ap.add_argument("--min_candidate_delta_f1_minus_chairi", type=float, default=-1e9)
    ap.add_argument("--max_candidate_delta_chair_i", type=float, default=1e9)
    ap.add_argument("--max_candidate_delta_chair_s", type=float, default=1e9)
    ap.add_argument("--max_candidates", type=int, default=0)
    ap.add_argument("--candidate_out_txt", type=str, default="")
    ap.add_argument("--candidate_out_csv", type=str, default="")
    args = ap.parse_args()

    claim_rows = base.read_csv_rows(args.claim_table_csv)
    chair_rows = base.read_csv_rows(args.chair_table_csv)
    rows = base.build_master_rows(
        claim_rows=claim_rows,
        chair_rows=chair_rows,
        baseline_chair_json=args.baseline_chair_json,
        intervention_chair_json=args.intervention_chair_json,
    )
    rows = linear.attach_teacher_labels(rows, teacher_mode=args.teacher_mode, min_f1_gain=float(args.min_f1_gain))

    feature_cols: List[str]
    if str(args.feature_cols).strip().lower() == "auto":
        feature_cols = base.infer_probe_feature_cols(rows)
    else:
        feature_cols = [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]
    if str(args.feature_prefix).strip():
        feature_cols = [col for col in feature_cols if str(col).startswith(str(args.feature_prefix).strip())]

    budgets = parse_budget_list(args.budgets)
    mined_rows: List[Dict[str, Any]] = []
    for feature in feature_cols:
        result = linear.evaluate_feature(rows, feature)
        if result is None:
            continue
        direction = str(result["direction"])
        scores = oriented_scores(rows, feature, direction)
        out_row: Dict[str, Any] = {
            "feature": str(feature),
            "family": linear.feature_family(str(feature)),
            "direction": direction,
            "auroc": float(result["auroc"]),
            "average_precision": float(result["average_precision"]) if result.get("average_precision") is not None else None,
            "n": int(result["n"]),
            "n_pos": int(result["n_pos"]),
        }
        add_budget_metrics(out_row, rows, scores, budgets, objective=str(args.selection_objective))
        mined_rows.append(out_row)

    mined_rows = sort_rows(mined_rows, budgets)
    base.write_csv(args.out_csv, mined_rows)
    print(f"[saved] {os.path.abspath(args.out_csv)}")

    candidate_rows = select_candidate_rows(
        mined_rows,
        budget=float(args.candidate_budget),
        min_precision=float(args.min_candidate_precision),
        min_recall=float(args.min_candidate_recall),
        min_delta_f1_minus_chairi=float(args.min_candidate_delta_f1_minus_chairi),
        max_delta_chair_i=float(args.max_candidate_delta_chair_i),
        max_delta_chair_s=float(args.max_candidate_delta_chair_s),
        max_candidates=int(args.max_candidates),
    )
    if str(args.candidate_out_csv or "").strip():
        base.write_csv(args.candidate_out_csv, candidate_rows)
        print(f"[saved] {os.path.abspath(args.candidate_out_csv)}")
    if str(args.candidate_out_txt or "").strip():
        write_candidate_features(args.candidate_out_txt, candidate_rows)
        print(f"[saved] {os.path.abspath(args.candidate_out_txt)}")

    if str(args.out_summary_json or "").strip():
        intervention = base.aggregate_routes(rows, ["method"] * len(rows))
        summary = {
            "inputs": {
                "claim_table_csv": os.path.abspath(args.claim_table_csv),
                "chair_table_csv": os.path.abspath(args.chair_table_csv),
                "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
                "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
                "teacher_mode": args.teacher_mode,
                "min_f1_gain": float(args.min_f1_gain),
                "feature_cols": feature_cols,
                "feature_prefix": str(args.feature_prefix or ""),
                "budgets": budgets,
                "selection_objective": args.selection_objective,
                "candidate_budget": float(args.candidate_budget),
                "min_candidate_precision": float(args.min_candidate_precision),
                "min_candidate_recall": float(args.min_candidate_recall),
                "min_candidate_delta_f1_minus_chairi": float(args.min_candidate_delta_f1_minus_chairi),
                "max_candidate_delta_chair_i": float(args.max_candidate_delta_chair_i),
                "max_candidate_delta_chair_s": float(args.max_candidate_delta_chair_s),
                "max_candidates": int(args.max_candidates),
            },
            "counts": {
                "n_rows": int(len(rows)),
                "teacher_positive_rate": base.safe_div(
                    float(sum(int(base.maybe_int(row.get("teacher_fallback")) or 0) for row in rows)),
                    float(max(1, len(rows))),
                ),
                "n_features": int(len(mined_rows)),
                "n_candidates": int(len(candidate_rows)),
            },
            "intervention": {
                "mean_chair_i": float(intervention["mean_chair_i"]),
                "mean_chair_s": float(intervention["mean_chair_s"]),
                "mean_recall": float(intervention["mean_recall"]),
                "mean_precision": float(intervention["mean_precision"]),
                "mean_f1": float(intervention["mean_f1"]),
                "mean_f1_minus_chairi": float(intervention["mean_f1_minus_chairi"]),
                "mean_claim_utility": float(intervention["mean_claim_utility"]),
            },
            "top_overall": top_rows(mined_rows, None, int(args.top_k_summary)),
            "top_probe": top_rows([row for row in mined_rows if row["family"] == "probe"], None, int(args.top_k_summary)),
            "top_pair": top_rows([row for row in mined_rows if row["family"] == "pair"], None, int(args.top_k_summary)),
            "candidate_rows": candidate_rows,
        }
        base.write_json(args.out_summary_json, summary)
        print(f"[saved] {os.path.abspath(args.out_summary_json)}")


if __name__ == "__main__":
    main()
