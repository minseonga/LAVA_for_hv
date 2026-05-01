#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_apply_layered_d_family_controller import (  # noqa: E402
    candidate_score_distribution,
    empirical_cdf,
    evaluate_scores,
    group_by_layer,
    maybe_float,
    maybe_int,
    percentile_scores,
    read_rows as read_d_rows,
    write_csv,
    write_json,
)
from build_apply_cvis_layered_d_fusion_controller import (  # noqa: E402
    build_fusion_scores,
    filter_c_rows,
    fusion_name,
    index_rows_by_id,
    is_candidate,
    load_or_calibrate_d_policy,
    load_or_calibrate_object_policy,
    merge_rows,
    parse_fusion_specs,
    score_c_rows,
    score_d_rows,
    score_object_rows,
    orient_c_feature,
    read_csv_rows,
)


AXIS_ORDER = ("C", "D", "O")


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(max(0.0, sum((x - mu) ** 2 for x in values) / float(len(values)))))


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = float(q) * float(len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return float(vals[lo] * (1.0 - weight) + vals[hi] * weight)


def ks_distance(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    aa = sorted(float(x) for x in a if math.isfinite(float(x)))
    bb = sorted(float(x) for x in b if math.isfinite(float(x)))
    if not aa or not bb:
        return None
    i = 0
    j = 0
    best = 0.0
    values = sorted(set(aa + bb))
    for value in values:
        while i < len(aa) and aa[i] <= value:
            i += 1
        while j < len(bb) and bb[j] <= value:
            j += 1
        best = max(best, abs(i / len(aa) - j / len(bb)))
    return float(best)


def row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id", row.get("question_id", ""))).strip()


def transition(row: Mapping[str, Any]) -> str:
    base = str(row.get("baseline_label", "")).strip().lower()
    intervention = str(row.get("intervention_label", "")).strip().lower()
    if base in {"yes", "no"} and intervention in {"yes", "no"}:
        return f"{base}->{intervention}"
    return "unknown"


def candidate_counts(rows: Sequence[Mapping[str, Any]], candidate_filter: str) -> Dict[str, Any]:
    n = 0
    harm = 0
    help_ = 0
    neutral = 0
    transitions: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for row in rows:
        if not is_candidate(row, candidate_filter):
            continue
        h = int(maybe_int(row.get("harm")) or 0)
        hp = int(maybe_int(row.get("help")) or 0)
        n += 1
        harm += h
        help_ += hp
        neutral += int(h == 0 and hp == 0)
        transitions[transition(row)] += 1
        categories[str(row.get("category", "") or "unknown")] += 1
    return {
        "n_route_candidates": int(n),
        "n_route_candidate_harm": int(harm),
        "n_route_candidate_help": int(help_),
        "n_route_candidate_neutral": int(neutral),
        "route_candidate_harm_rate": float(harm / max(1, n)),
        "route_candidate_help_rate": float(help_ / max(1, n)),
        "transition_counts": dict(transitions),
        "category_counts": dict(categories),
    }


def score_distribution(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    *,
    candidate_filter: str,
) -> Dict[str, Any]:
    vals: List[float] = []
    ys: List[int] = []
    for row in rows:
        sid = row_id(row)
        if sid not in scores or not is_candidate(row, candidate_filter):
            continue
        h = maybe_int(row.get("harm"))
        hp = maybe_int(row.get("help"))
        if h not in {0, 1} or hp not in {0, 1}:
            continue
        if int(h) == 0 and int(hp) == 0:
            continue
        vals.append(float(scores[sid]))
        ys.append(int(h))
    harm_vals = [x for x, y in zip(vals, ys) if y == 1]
    help_vals = [x for x, y in zip(vals, ys) if y == 0]
    out: Dict[str, Any] = {
        "score_n": int(len(vals)),
        "score_harm": int(sum(ys)),
        "score_help": int(len(ys) - sum(ys)),
        "score_mean": mean(vals) if vals else "",
        "score_std": std(vals) if vals else "",
        "score_harm_mean": mean(harm_vals) if harm_vals else "",
        "score_help_mean": mean(help_vals) if help_vals else "",
    }
    for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
        key = f"score_q{int(q * 100):02d}"
        out[key] = quantile(vals, q)
    return out


def routed_eval(
    rows: Sequence[Mapping[str, Any]],
    raw_scores: Mapping[str, float],
    *,
    candidate_filter: str,
    score_space: str,
    tau_raw: float,
    tau_percentile: float,
    calibration_cdf: Sequence[float],
    min_selected_count: int,
    oracle: bool,
) -> Tuple[Dict[str, Any], Dict[str, float], Optional[float]]:
    score_space = str(score_space or "raw").strip().lower()
    if score_space == "percentile":
        score_space = "discovery_percentile"
    if score_space == "raw":
        route_scores = dict(raw_scores)
        tau = None if oracle else float(tau_raw)
    elif score_space == "discovery_percentile":
        route_scores = percentile_scores(raw_scores, calibration_cdf)
        tau = None if oracle else float(tau_percentile)
    elif score_space == "batch_percentile":
        batch_cdf = candidate_score_distribution(rows, raw_scores, candidate_filter=candidate_filter)
        route_scores = percentile_scores(raw_scores, batch_cdf)
        tau = None if oracle else float(tau_percentile)
    else:
        raise ValueError(f"Unsupported score_space={score_space!r}")
    result, _ = evaluate_scores(
        rows,
        route_scores,
        candidate_filter=candidate_filter,
        tau=tau,
        min_selected_count=min_selected_count,
    )
    return result, route_scores, tau


def eval_axis_candidate(
    *,
    name: str,
    axis_set: Sequence[str],
    fusion_spec: Optional[Mapping[str, Any]],
    required_streams: Optional[int],
    cal_rows: Sequence[Mapping[str, Any]],
    apply_rows: Sequence[Mapping[str, Any]],
    cal_score_maps: Mapping[str, Mapping[str, float]],
    apply_score_maps: Mapping[str, Mapping[str, float]],
    candidate_filter: str,
    score_space: str,
    min_selected_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float], List[float]]:
    axis_set = tuple(axis_set)
    if len(axis_set) == 1:
        cal_raw = dict(cal_score_maps[axis_set[0]])
        apply_raw = dict(apply_score_maps[axis_set[0]])
        spec_name = "single"
    else:
        assert fusion_spec is not None
        maps_cal = [cal_score_maps[x] for x in axis_set]
        maps_apply = [apply_score_maps[x] for x in axis_set]
        cal_raw = build_fusion_scores(maps_cal, fusion_spec, required_streams=required_streams)
        apply_raw = build_fusion_scores(maps_apply, fusion_spec, required_streams=required_streams)
        spec_name = fusion_name(fusion_spec)

    cal_cdf = candidate_score_distribution(cal_rows, cal_raw, candidate_filter=candidate_filter)
    cal_best_raw, _ = evaluate_scores(
        cal_rows,
        cal_raw,
        candidate_filter=candidate_filter,
        min_selected_count=min_selected_count,
    )
    tau_raw = float(cal_best_raw["tau"])
    tau_percentile = empirical_cdf(tau_raw, cal_cdf)

    cal_locked, _, cal_tau = routed_eval(
        cal_rows,
        cal_raw,
        candidate_filter=candidate_filter,
        score_space=score_space,
        tau_raw=tau_raw,
        tau_percentile=tau_percentile,
        calibration_cdf=cal_cdf,
        min_selected_count=min_selected_count,
        oracle=False,
    )
    apply_locked, _, apply_tau = routed_eval(
        apply_rows,
        apply_raw,
        candidate_filter=candidate_filter,
        score_space=score_space,
        tau_raw=tau_raw,
        tau_percentile=tau_percentile,
        calibration_cdf=cal_cdf,
        min_selected_count=min_selected_count,
        oracle=False,
    )
    apply_oracle, _, _ = routed_eval(
        apply_rows,
        apply_raw,
        candidate_filter=candidate_filter,
        score_space=score_space,
        tau_raw=tau_raw,
        tau_percentile=tau_percentile,
        calibration_cdf=cal_cdf,
        min_selected_count=min_selected_count,
        oracle=True,
    )

    base = {
        "axis_name": name,
        "axis_set": "+".join(axis_set),
        "fusion_name": spec_name,
        "fusion": json.dumps(dict(fusion_spec or {"mode": "single"}), sort_keys=True),
        "required_streams": required_streams if required_streams is not None else len(axis_set),
        "score_space": score_space,
        "tau_raw": tau_raw,
        "tau_percentile": tau_percentile,
        "calibration_score_count": len(cal_cdf),
    }
    rows_out = []
    for stage, result, tau_used in (
        ("calibration_locked", cal_locked, cal_tau),
        ("apply_locked", apply_locked, apply_tau),
        ("apply_oracle", apply_oracle, None),
    ):
        rows_out.append(
            {
                **base,
                "stage": stage,
                "route_tau": tau_used if tau_used is not None else result.get("tau"),
                **dict(result),
            }
        )
    return rows_out, cal_raw, apply_raw, cal_cdf


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare C_vis, layered-D, object-D, pairwise fusion, and triple fusion under suppression scope."
    )
    ap.add_argument("--calibration_c_rows_csv", required=True)
    ap.add_argument("--calibration_d_trajectory_long_csv", required=True)
    ap.add_argument("--calibration_object_trajectory_long_csv", default="")
    ap.add_argument("--apply_c_rows_csv", default="")
    ap.add_argument("--apply_d_trajectory_long_csv", default="")
    ap.add_argument("--apply_object_trajectory_long_csv", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_filter", default="yes_to_no", choices=["all", "changed_answer", "yes_to_no"])
    ap.add_argument("--c_feature", default="abl_black_rel_delta_orig_minus_blind__cheap_decision_margin_abs")
    ap.add_argument("--c_layer", default="")
    ap.add_argument("--c_direction", default="auto", choices=["auto", "high", "low"])
    ap.add_argument("--d_policy_json", default="")
    ap.add_argument("--d_layer_grid", default="all")
    ap.add_argument("--object_feature", default="obj_target_gap_mean")
    ap.add_argument("--object_layer_grid", default="late")
    ap.add_argument("--object_direction", default="auto", choices=["auto", "high", "low"])
    ap.add_argument("--fusion_modes", default="mean,min,max")
    ap.add_argument("--alpha_grid", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--score_space", default="raw", choices=["raw", "percentile", "discovery_percentile", "batch_percentile"])
    ap.add_argument("--min_selected_count", type=int, default=5)
    ap.add_argument("--include_triple_req2", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cal_c_rows = read_csv_rows(os.path.abspath(args.calibration_c_rows_csv))
    cal_d_rows = read_d_rows(os.path.abspath(args.calibration_d_trajectory_long_csv))
    cal_object_rows = (
        read_csv_rows(os.path.abspath(args.calibration_object_trajectory_long_csv))
        if str(args.calibration_object_trajectory_long_csv or "").strip()
        else []
    )
    apply_c_rows = (
        read_csv_rows(os.path.abspath(args.apply_c_rows_csv))
        if str(args.apply_c_rows_csv or "").strip()
        else list(cal_c_rows)
    )
    apply_d_rows = (
        read_d_rows(os.path.abspath(args.apply_d_trajectory_long_csv))
        if str(args.apply_d_trajectory_long_csv or "").strip()
        else list(cal_d_rows)
    )
    apply_object_rows = (
        read_csv_rows(os.path.abspath(args.apply_object_trajectory_long_csv))
        if str(args.apply_object_trajectory_long_csv or "").strip()
        else list(cal_object_rows)
    )

    cal_c_rows_f = filter_c_rows(cal_c_rows, str(args.c_layer))
    apply_c_rows_f = filter_c_rows(apply_c_rows, str(args.c_layer))
    cal_c_rows_by_id = index_rows_by_id(cal_c_rows_f)
    apply_c_rows_by_id = index_rows_by_id(apply_c_rows_f)
    c_metric = orient_c_feature(
        list(cal_c_rows_by_id.values()),
        str(args.c_feature),
        str(args.candidate_filter),
        str(args.c_direction),
    )
    cal_c_scores = score_c_rows(cal_c_rows_by_id, c_metric)
    apply_c_scores = score_c_rows(apply_c_rows_by_id, c_metric)

    d_policy = load_or_calibrate_d_policy(
        d_policy_json=str(args.d_policy_json),
        calibration_d_rows=cal_d_rows,
        layer_grid_spec=str(args.d_layer_grid),
        candidate_filter=str(args.candidate_filter),
        min_selected_count=int(args.min_selected_count),
    )
    cal_d_rows_by_id, cal_d_scores = score_d_rows(cal_d_rows, d_policy)
    apply_d_rows_by_id, apply_d_scores = score_d_rows(apply_d_rows, d_policy)

    object_policy = None
    cal_object_rows_by_id: Dict[str, Dict[str, Any]] = {}
    apply_object_rows_by_id: Dict[str, Dict[str, Any]] = {}
    cal_object_scores: Dict[str, float] = {}
    apply_object_scores: Dict[str, float] = {}
    if cal_object_rows:
        object_policy = load_or_calibrate_object_policy(
            object_rows=cal_object_rows,
            object_feature=str(args.object_feature),
            object_layer_grid=str(args.object_layer_grid),
            object_direction=str(args.object_direction),
            candidate_filter=str(args.candidate_filter),
            min_selected_count=int(args.min_selected_count),
        )
        cal_object_rows_by_id, cal_object_scores = score_object_rows(cal_object_rows, object_policy)
        apply_object_rows_by_id, apply_object_scores = score_object_rows(apply_object_rows, object_policy)

    cal_rows_by_id = merge_rows(cal_c_rows_by_id, cal_d_rows_by_id, cal_object_rows_by_id)
    apply_rows_by_id = merge_rows(apply_c_rows_by_id, apply_d_rows_by_id, apply_object_rows_by_id)
    cal_rows = [cal_rows_by_id[sid] for sid in sorted(cal_rows_by_id, key=lambda x: (len(str(x)), str(x)))]
    apply_rows = [apply_rows_by_id[sid] for sid in sorted(apply_rows_by_id, key=lambda x: (len(str(x)), str(x)))]

    cal_score_maps: Dict[str, Mapping[str, float]] = {"C": cal_c_scores, "D": cal_d_scores}
    apply_score_maps: Dict[str, Mapping[str, float]] = {"C": apply_c_scores, "D": apply_d_scores}
    if object_policy:
        cal_score_maps["O"] = cal_object_scores
        apply_score_maps["O"] = apply_object_scores

    specs = parse_fusion_specs(str(args.fusion_modes), str(args.alpha_grid))
    candidates: List[Tuple[str, Tuple[str, ...], Optional[Mapping[str, Any]], Optional[int]]] = [
        ("C", ("C",), None, 1),
        ("D", ("D",), None, 1),
    ]
    if object_policy:
        candidates.append(("O", ("O",), None, 1))
    for axes in (("C", "D"), ("D", "O"), ("C", "O"), ("C", "D", "O")):
        if any(axis not in cal_score_maps for axis in axes):
            continue
        for spec in specs:
            if str(spec.get("mode")) == "alpha" and len(axes) != 2:
                continue
            required = len(axes)
            candidates.append(("+".join(axes), tuple(axes), spec, required))
    if object_policy and bool(args.include_triple_req2):
        for spec in specs:
            if str(spec.get("mode")) != "alpha":
                candidates.append(("C+D+O_req2", ("C", "D", "O"), spec, 2))

    comparison_rows: List[Dict[str, Any]] = []
    distribution_rows: List[Dict[str, Any]] = []
    for name, axes, spec, required in candidates:
        rows_out, cal_raw, apply_raw, cal_cdf = eval_axis_candidate(
            name=name,
            axis_set=axes,
            fusion_spec=spec,
            required_streams=required,
            cal_rows=cal_rows,
            apply_rows=apply_rows,
            cal_score_maps=cal_score_maps,
            apply_score_maps=apply_score_maps,
            candidate_filter=str(args.candidate_filter),
            score_space=str(args.score_space),
            min_selected_count=int(args.min_selected_count),
        )
        comparison_rows.extend(rows_out)
        cal_dist = score_distribution(cal_rows, cal_raw, candidate_filter=str(args.candidate_filter))
        apply_dist = score_distribution(apply_rows, apply_raw, candidate_filter=str(args.candidate_filter))
        cal_values = candidate_score_distribution(cal_rows, cal_raw, candidate_filter=str(args.candidate_filter))
        apply_values = candidate_score_distribution(apply_rows, apply_raw, candidate_filter=str(args.candidate_filter))
        distribution_rows.append(
            {
                "axis_name": name,
                "axis_set": "+".join(axes),
                "fusion_name": "single" if spec is None else fusion_name(spec),
                "required_streams": required,
                "calibration_candidates": len(cal_values),
                "apply_candidates": len(apply_values),
                "apply_score_ks_vs_calibration": ks_distance(cal_values, apply_values),
                "apply_q50_minus_calibration_q50": (
                    (quantile(apply_values, 0.50) or 0.0) - (quantile(cal_values, 0.50) or 0.0)
                    if cal_values and apply_values
                    else ""
                ),
                "calibration": json.dumps(cal_dist, sort_keys=True),
                "apply": json.dumps(apply_dist, sort_keys=True),
            }
        )

    comparison_rows.sort(
        key=lambda row: (
            str(row.get("stage")),
            -int(row.get("net", 0)),
            -float(row.get("selected_harm_precision", 0.0)),
            str(row.get("axis_name")),
            str(row.get("fusion_name")),
        )
    )
    write_csv(os.path.join(out_dir, "axis_fusion_comparison.csv"), comparison_rows)
    write_csv(os.path.join(out_dir, "score_distribution_shift.csv"), distribution_rows)
    summary = {
        "mode": "cvis_layered_d_object_axis_fusion_audit",
        "candidate_filter": str(args.candidate_filter),
        "score_space": str(args.score_space),
        "inputs": {
            "calibration_c_rows_csv": os.path.abspath(args.calibration_c_rows_csv),
            "calibration_d_trajectory_long_csv": os.path.abspath(args.calibration_d_trajectory_long_csv),
            "calibration_object_trajectory_long_csv": os.path.abspath(args.calibration_object_trajectory_long_csv)
            if str(args.calibration_object_trajectory_long_csv or "").strip()
            else "",
            "apply_c_rows_csv": os.path.abspath(args.apply_c_rows_csv) if str(args.apply_c_rows_csv or "").strip() else "",
            "apply_d_trajectory_long_csv": os.path.abspath(args.apply_d_trajectory_long_csv)
            if str(args.apply_d_trajectory_long_csv or "").strip()
            else "",
            "apply_object_trajectory_long_csv": os.path.abspath(args.apply_object_trajectory_long_csv)
            if str(args.apply_object_trajectory_long_csv or "").strip()
            else "",
        },
        "axis_policy": {
            "c_metric": c_metric,
            "d_policy": {
                "selected_layer": d_policy.get("selected_layer"),
                "selected_d_features": d_policy.get("selected_d_features"),
                "selected_policy": d_policy.get("selected_policy"),
            },
            "object_policy": None
            if not object_policy
            else {
                "selected_layer": object_policy.get("selected_layer"),
                "object_feature": object_policy.get("object_feature"),
                "object_metric": object_policy.get("object_metric"),
                "selected_policy": object_policy.get("selected_policy"),
            },
        },
        "calibration_counts": candidate_counts(cal_rows, str(args.candidate_filter)),
        "apply_counts": candidate_counts(apply_rows, str(args.candidate_filter)),
        "best_apply_locked": comparison_rows[:10],
        "outputs": {
            "axis_fusion_comparison_csv": os.path.join(out_dir, "axis_fusion_comparison.csv"),
            "score_distribution_shift_csv": os.path.join(out_dir, "score_distribution_shift.csv"),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print("[saved]", os.path.join(out_dir, "axis_fusion_comparison.csv"))
    print("[saved]", os.path.join(out_dir, "score_distribution_shift.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
