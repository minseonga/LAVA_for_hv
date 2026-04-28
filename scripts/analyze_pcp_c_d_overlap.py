#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set

import build_posthoc_b_c_fusion_controller as base
import build_pcp_c_d_controller as pcp


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def selected_ids_for_family(
    rows: List[Dict[str, Any]],
    *,
    c_features: List[Dict[str, Any]],
    d_features: List[Dict[str, Any]],
    family: str,
    alpha: float,
    tau: float,
    candidate_filter: str,
) -> Set[str]:
    out: Set[str] = set()
    for row in rows:
        sid = str(row.get("id", "")).strip()
        if not sid:
            continue
        if not pcp.is_route_candidate(row, candidate_filter):
            continue
        score = None
        if family == "c_only":
            score = pcp.mean_z_score(row, c_features)
        elif family == "d_only":
            score = pcp.mean_z_score(row, d_features)
        elif family == "cd_fusion":
            c_score = pcp.mean_z_score(row, c_features)
            d_score = pcp.mean_z_score(row, d_features)
            if c_score is not None and d_score is not None:
                score = float((1.0 - float(alpha)) * float(c_score) + float(alpha) * float(d_score))
        else:
            raise ValueError(f"unsupported family={family!r}")
        if score is not None and float(score) >= float(tau):
            out.add(sid)
    return out


def evaluate_selected_ids(rows: Iterable[Dict[str, Any]], selected_ids: Set[str]) -> Dict[str, Any]:
    selected = 0
    harm_fixed = 0
    help_lost = 0
    neutral = 0
    correct = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    n_eval = 0
    total_harm = 0
    total_help = 0

    for row in rows:
        bc = base.maybe_int(row.get("baseline_correct"))
        ic = base.maybe_int(row.get("intervention_correct"))
        if bc is None or ic is None:
            continue
        sid = str(row.get("id", "")).strip()
        harm = int(base.maybe_int(row.get("harm")) or 0)
        help_ = int(base.maybe_int(row.get("help")) or 0)
        total_harm += harm
        total_help += help_
        baseline_correct_total += int(bc)
        intervention_correct_total += int(ic)
        n_eval += 1
        if sid in selected_ids:
            selected += 1
            harm_fixed += harm
            help_lost += help_
            neutral += int((harm == 0) and (help_ == 0))
            correct += int(bc)
        else:
            correct += int(ic)

    precision = base.safe_div(float(harm_fixed), float(max(1, selected)))
    recall = base.safe_div(float(harm_fixed), float(max(1, total_harm)))
    f1 = base.safe_div(2.0 * precision * recall, precision + recall)
    return {
        "n_eval": int(n_eval),
        "selected_count": int(selected),
        "baseline_rate": base.safe_div(float(selected), float(max(1, n_eval))),
        "final_acc": base.safe_div(float(correct), float(max(1, n_eval))),
        "baseline_acc": base.safe_div(float(baseline_correct_total), float(max(1, n_eval))),
        "intervention_acc": base.safe_div(float(intervention_correct_total), float(max(1, n_eval))),
        "delta_vs_intervention": base.safe_div(float(correct - intervention_correct_total), float(max(1, n_eval))),
        "selected_harm": int(harm_fixed),
        "selected_help": int(help_lost),
        "selected_neutral": int(neutral),
        "net": int(harm_fixed - help_lost),
        "selected_harm_precision": precision,
        "selected_harm_recall": recall,
        "selected_harm_f1": f1,
        "total_harm": int(total_harm),
        "total_help": int(total_help),
    }


def region_stats(rows: Iterable[Dict[str, Any]], selected_ids: Set[str]) -> Dict[str, Any]:
    n = harm = help_ = neutral = both_correct = both_wrong = 0
    ids: List[str] = []
    for row in rows:
        sid = str(row.get("id", "")).strip()
        if sid not in selected_ids:
            continue
        ids.append(sid)
        n += 1
        harm += int(base.maybe_int(row.get("harm")) or 0)
        help_ += int(base.maybe_int(row.get("help")) or 0)
        neutral += int((int(base.maybe_int(row.get("harm")) or 0) == 0) and (int(base.maybe_int(row.get("help")) or 0) == 0))
        bc = base.maybe_int(row.get("baseline_correct"))
        ic = base.maybe_int(row.get("intervention_correct"))
        if bc == 1 and ic == 1:
            both_correct += 1
        elif bc == 0 and ic == 0:
            both_wrong += 1
    return {
        "count": int(n),
        "harm": int(harm),
        "help": int(help_),
        "neutral": int(neutral),
        "net": int(harm - help_),
        "harm_precision": base.safe_div(float(harm), float(max(1, n))),
        "both_correct": int(both_correct),
        "both_wrong": int(both_wrong),
        "sample_ids_preview": ids[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze overlap between PCP C-only and D-only selected sets.")
    ap.add_argument("--rows_csv", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--derive_decision_kl", type=pcp.parse_bool, default=True)
    args = ap.parse_args()

    rows = pcp.load_rows(os.path.abspath(args.rows_csv), derive_decision_kl=bool(args.derive_decision_kl))
    with open(os.path.abspath(args.policy_json), "r", encoding="utf-8") as f:
        bundle: Dict[str, Any] = json.load(f)

    candidate_filter = str(bundle.get("candidate_filter") or "all")
    c_features = list(bundle.get("selected_c_features") or [])
    d_features = list(bundle.get("selected_d_features") or [])
    best_results = bundle.get("best_results") or {}
    c_policy = best_results.get("c_only")
    d_policy = best_results.get("d_only")
    cd_policy = best_results.get("cd_fusion")
    if not c_policy or not d_policy:
        raise RuntimeError("policy_json must contain both c_only and d_only results")

    c_ids = selected_ids_for_family(
        rows,
        c_features=c_features,
        d_features=d_features,
        family="c_only",
        alpha=float(c_policy["alpha"]),
        tau=float(c_policy["tau"]),
        candidate_filter=candidate_filter,
    )
    d_ids = selected_ids_for_family(
        rows,
        c_features=c_features,
        d_features=d_features,
        family="d_only",
        alpha=float(d_policy["alpha"]),
        tau=float(d_policy["tau"]),
        candidate_filter=candidate_filter,
    )
    both_ids = c_ids & d_ids
    union_ids = c_ids | d_ids
    c_only_ids = c_ids - d_ids
    d_only_ids = d_ids - c_ids
    neither_ids = {str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()} - union_ids

    out = {
        "inputs": {
            "rows_csv": os.path.abspath(args.rows_csv),
            "policy_json": os.path.abspath(args.policy_json),
            "derive_decision_kl": bool(args.derive_decision_kl),
            "candidate_filter": candidate_filter,
        },
        "policies": {
            "c_only": c_policy,
            "d_only": d_policy,
            "cd_fusion": cd_policy,
        },
        "set_counts": {
            "c_selected": int(len(c_ids)),
            "d_selected": int(len(d_ids)),
            "intersection": int(len(both_ids)),
            "union": int(len(union_ids)),
            "c_only_exclusive": int(len(c_only_ids)),
            "d_only_exclusive": int(len(d_only_ids)),
            "neither": int(len(neither_ids)),
            "jaccard": base.safe_div(float(len(both_ids)), float(max(1, len(union_ids)))),
            "overlap_over_c": base.safe_div(float(len(both_ids)), float(max(1, len(c_ids)))),
            "overlap_over_d": base.safe_div(float(len(both_ids)), float(max(1, len(d_ids)))),
        },
        "regions": {
            "c_only_exclusive": region_stats(rows, c_only_ids),
            "d_only_exclusive": region_stats(rows, d_only_ids),
            "intersection": region_stats(rows, both_ids),
            "union": region_stats(rows, union_ids),
            "neither": region_stats(rows, neither_ids),
        },
        "counterfactual_policies": {
            "c_only": evaluate_selected_ids(rows, c_ids),
            "d_only": evaluate_selected_ids(rows, d_ids),
            "union_or": evaluate_selected_ids(rows, union_ids),
            "intersection_and": evaluate_selected_ids(rows, both_ids),
        },
    }
    write_json(os.path.abspath(args.out_json), out)
    print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
