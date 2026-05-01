#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import build_pcp_c_d_controller as pcp  # noqa: E402
import build_posthoc_b_c_fusion_controller as base  # noqa: E402
from build_apply_layered_d_family_controller import (  # noqa: E402
    calibrate as calibrate_layered_d,
    candidate_score_distribution,
    empirical_cdf,
    group_by_layer,
    maybe_float,
    maybe_int,
    parse_layer_grid,
    percentile_scores,
    read_rows as read_d_rows,
    score_row as score_d_row,
)


LABEL_KEYS = (
    "image",
    "question",
    "text",
    "category",
    "baseline_text",
    "intervention_text",
    "baseline_label",
    "intervention_label",
    "baseline_correct",
    "intervention_correct",
    "harm",
    "help",
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def read_json(path: str) -> Dict[str, Any]:
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id", row.get("question_id", ""))).strip()


def index_by_id(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = row_id(row)
        if sid and sid not in out:
            out[sid] = dict(row)
    return out


def overlay_labels(
    d_rows: Sequence[Dict[str, Any]],
    c_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in d_rows:
        sid = row_id(row)
        merged = dict(row)
        c_row = c_rows_by_id.get(sid)
        if c_row:
            for key in LABEL_KEYS:
                if str(merged.get(key, "")).strip() == "" and key in c_row:
                    merged[key] = c_row.get(key)
        merged["id"] = sid
        out.append(merged)
    return out


def sorted_ids(ids: Sequence[str]) -> List[str]:
    return sorted((str(x) for x in ids), key=lambda x: (len(x), x))


def score_c_rows(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    c_features: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sid, row in rows_by_id.items():
        score = pcp.mean_z_score(dict(row), list(c_features))
        if score is not None:
            out[str(sid)] = float(score)
    return out


def load_or_fit_c_features(
    *,
    c_policy_json: str,
    fit_rows: Sequence[Dict[str, Any]],
    c_feature_cols: Sequence[str],
    min_present_rate: float,
    min_feature_auroc: float,
    top_k_c: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if str(c_policy_json or "").strip():
        policy = read_json(c_policy_json)
        selected = list(policy.get("selected_c_features") or [])
        if not selected:
            raise RuntimeError(f"No selected_c_features in c_policy_json={c_policy_json!r}.")
        return [], selected
    metrics, selected = pcp.orient_feature_list(
        fit_rows,
        c_feature_cols,
        target="harm",
        min_present_rate=float(min_present_rate),
        min_feature_auroc=float(min_feature_auroc),
        top_k=int(top_k_c),
    )
    if not selected:
        raise RuntimeError("No C features were selected from the calibration rows.")
    return metrics, selected


def load_or_fit_d_policy(
    *,
    d_policy_json: str,
    d_rows: Sequence[Dict[str, Any]],
    d_layer_grid: str,
    candidate_filter: str,
    min_selected_count: int,
    score_space: str,
) -> Dict[str, Any]:
    if str(d_policy_json or "").strip():
        return read_json(d_policy_json)
    available_layers = sorted(group_by_layer(d_rows))
    if not available_layers:
        raise RuntimeError("No layer_index values found in calibration D trajectory rows.")
    layer_grid = parse_layer_grid(str(d_layer_grid), available_layers)
    return calibrate_layered_d(
        d_rows,
        layer_grid=layer_grid,
        candidate_filter=str(candidate_filter),
        min_selected_count=int(min_selected_count),
        score_space=str(score_space),
    )


def score_d_rows_from_policy(
    d_rows: Sequence[Dict[str, Any]],
    d_policy: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    layer = int(d_policy["selected_layer"])
    metrics = list(d_policy.get("selected_d_features") or [])
    layer_rows = group_by_layer(d_rows).get(layer, [])
    rows_by_id = index_by_id(layer_rows)
    scores: Dict[str, float] = {}
    for sid, row in rows_by_id.items():
        score = score_d_row(row, metrics)
        if score is not None:
            scores[str(sid)] = float(score)
    return rows_by_id, scores


def fusion_score(c_score: Optional[float], d_score: Optional[float], family: str, alpha: float) -> Optional[float]:
    if family == "c_only":
        return c_score
    if family == "d_only":
        return d_score
    if c_score is None or d_score is None:
        return None
    return float((1.0 - float(alpha)) * float(c_score) + float(alpha) * float(d_score))


def route_scores_for_space(
    raw_scores: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_filter: str,
    score_space: str,
    calibration_cdf: Sequence[float],
) -> Tuple[Dict[str, float], List[float]]:
    score_space = str(score_space or "raw").strip().lower()
    if score_space == "percentile":
        score_space = "discovery_percentile"
    if score_space == "raw":
        return dict(raw_scores), sorted(float(x) for x in calibration_cdf)
    if score_space == "discovery_percentile":
        cdf = sorted(float(x) for x in calibration_cdf if math.isfinite(float(x)))
        if not cdf:
            raise RuntimeError("discovery_percentile requires a non-empty calibration CDF.")
        return percentile_scores(raw_scores, cdf), cdf
    if score_space == "batch_percentile":
        cdf = candidate_score_distribution(rows, raw_scores, candidate_filter=candidate_filter)
        if not cdf:
            raise RuntimeError("batch_percentile requires a non-empty apply candidate score distribution.")
        return percentile_scores(raw_scores, cdf), cdf
    raise ValueError(f"Unsupported score_space={score_space!r}.")


def evaluate_route_scores(
    rows: Sequence[Mapping[str, Any]],
    route_scores: Mapping[str, float],
    *,
    candidate_filter: str,
    tau: Optional[float] = None,
    min_selected_count: int = 0,
    objective: str = "final_acc",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidate_values = [
        float(route_scores[row_id(row)])
        for row in rows
        if row_id(row) in route_scores and pcp.is_route_candidate(dict(row), candidate_filter)
    ]
    taus = [float(tau)] if tau is not None else pcp.threshold_grid(candidate_values)
    best: Optional[Dict[str, Any]] = None
    sweep: List[Dict[str, Any]] = []
    for tau_value in taus:
        n_eval = 0
        n_route_candidates = 0
        selected = 0
        baseline_correct_total = 0
        intervention_correct_total = 0
        final_correct_total = 0
        total_harm = 0
        total_help = 0
        route_candidate_harm = 0
        route_candidate_help = 0
        route_candidate_neutral = 0
        selected_harm = 0
        selected_help = 0
        selected_neutral = 0
        for row in rows:
            sid = row_id(row)
            if sid not in route_scores:
                continue
            bc = maybe_int(row.get("baseline_correct"))
            ic = maybe_int(row.get("intervention_correct"))
            if bc is None or ic is None:
                continue
            harm = int(maybe_int(row.get("harm")) or 0)
            help_ = int(maybe_int(row.get("help")) or 0)
            can_route = pcp.is_route_candidate(dict(row), candidate_filter)
            use_baseline = bool(can_route and float(route_scores[sid]) >= float(tau_value))
            n_eval += 1
            baseline_correct_total += int(bc)
            intervention_correct_total += int(ic)
            total_harm += harm
            total_help += help_
            if can_route:
                n_route_candidates += 1
                route_candidate_harm += harm
                route_candidate_help += help_
                route_candidate_neutral += int(harm == 0 and help_ == 0)
            if use_baseline:
                selected += 1
                selected_harm += harm
                selected_help += help_
                selected_neutral += int(harm == 0 and help_ == 0)
                final_correct_total += int(bc)
            else:
                final_correct_total += int(ic)
        precision = base.safe_div(float(selected_harm), float(max(1, selected)))
        recall = base.safe_div(float(selected_harm), float(max(1, total_harm)))
        result = {
            "tau": float(tau_value),
            "n_eval": int(n_eval),
            "baseline_rate": base.safe_div(float(selected), float(max(1, n_eval))),
            "method_rate": 1.0 - base.safe_div(float(selected), float(max(1, n_eval))),
            "baseline_acc": base.safe_div(float(baseline_correct_total), float(max(1, n_eval))),
            "intervention_acc": base.safe_div(float(intervention_correct_total), float(max(1, n_eval))),
            "final_acc": base.safe_div(float(final_correct_total), float(max(1, n_eval))),
            "delta_vs_intervention": base.safe_div(
                float(final_correct_total - intervention_correct_total),
                float(max(1, n_eval)),
            ),
            "selected_count": int(selected),
            "total_harm": int(total_harm),
            "total_help": int(total_help),
            "n_route_candidates": int(n_route_candidates),
            "n_route_candidate_harm": int(route_candidate_harm),
            "n_route_candidate_help": int(route_candidate_help),
            "n_route_candidate_neutral": int(route_candidate_neutral),
            "route_candidate_baseline_rate": base.safe_div(float(selected), float(max(1, n_route_candidates))),
            "selected_harm": int(selected_harm),
            "selected_help": int(selected_help),
            "selected_neutral": int(selected_neutral),
            "net": int(selected_harm - selected_help),
            "selected_harm_precision": precision,
            "selected_help_precision": base.safe_div(float(selected_help), float(max(1, selected))),
            "selected_harm_recall": recall,
            "selected_harm_recall_in_scope": base.safe_div(float(selected_harm), float(max(1, route_candidate_harm))),
            "selected_help_recall_in_scope": base.safe_div(float(selected_help), float(max(1, route_candidate_help))),
            "selected_harm_f1": base.safe_div(2.0 * precision * recall, precision + recall),
        }
        sweep.append(result)
        if int(result["selected_count"]) < int(min_selected_count):
            continue
        if best is None or pcp.selection_key(result, objective) > pcp.selection_key(best, objective):
            best = result
    return best or (sweep[0] if sweep else {}), sweep


def build_raw_scores(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    c_scores: Mapping[str, float],
    d_scores: Mapping[str, float],
    *,
    family: str,
    alpha: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sid in rows_by_id:
        score = fusion_score(c_scores.get(sid), d_scores.get(sid), family, alpha)
        if score is not None:
            out[str(sid)] = float(score)
    return out


def calibrate_policy(
    *,
    c_rows: Sequence[Dict[str, Any]],
    d_rows: Sequence[Dict[str, Any]],
    c_policy_json: str,
    d_policy_json: str,
    c_feature_cols: Sequence[str],
    d_layer_grid: str,
    candidate_filter: str,
    min_present_rate: float,
    min_feature_auroc: float,
    top_k_c: int,
    min_selected_count: int,
    alpha_grid: Sequence[float],
    score_space: str,
    tau_objective: str,
) -> Dict[str, Any]:
    c_rows_by_id = index_by_id(c_rows)
    d_rows_labeled = overlay_labels(d_rows, c_rows_by_id)
    fit_rows = [row for row in c_rows if pcp.is_route_candidate(row, candidate_filter)]
    if not fit_rows:
        raise RuntimeError(f"No calibration C rows remain after candidate_filter={candidate_filter!r}.")

    c_metrics, selected_c = load_or_fit_c_features(
        c_policy_json=c_policy_json,
        fit_rows=fit_rows,
        c_feature_cols=c_feature_cols,
        min_present_rate=min_present_rate,
        min_feature_auroc=min_feature_auroc,
        top_k_c=top_k_c,
    )
    c_scores = score_c_rows(c_rows_by_id, selected_c)

    d_policy = load_or_fit_d_policy(
        d_policy_json=d_policy_json,
        d_rows=d_rows_labeled,
        d_layer_grid=d_layer_grid,
        candidate_filter=candidate_filter,
        min_selected_count=min_selected_count,
        score_space=score_space,
    )
    d_rows_by_id, d_scores = score_d_rows_from_policy(d_rows_labeled, d_policy)

    merged_rows_by_id: Dict[str, Dict[str, Any]] = {}
    for sid in sorted_ids(set(c_rows_by_id) | set(d_rows_by_id)):
        row: Dict[str, Any] = {}
        if sid in d_rows_by_id:
            row.update(d_rows_by_id[sid])
        if sid in c_rows_by_id:
            row.update(c_rows_by_id[sid])
        row["id"] = sid
        merged_rows_by_id[sid] = row

    family_specs: List[Tuple[str, float]] = [("c_only", 0.0), ("d_only", 1.0)]
    family_specs.extend(("cd_fusion", float(alpha)) for alpha in alpha_grid if 0.0 < float(alpha) < 1.0)

    family_results: Dict[str, Dict[str, Any]] = {}
    family_sweeps: List[Dict[str, Any]] = []
    family_cdfs: Dict[str, List[float]] = {}
    for family, alpha in family_specs:
        raw_scores = build_raw_scores(merged_rows_by_id, c_scores, d_scores, family=family, alpha=alpha)
        eval_rows = [merged_rows_by_id[sid] for sid in sorted_ids(raw_scores)]
        calibration_cdf = candidate_score_distribution(eval_rows, raw_scores, candidate_filter=candidate_filter)
        if not calibration_cdf:
            continue
        route_scores, _ = route_scores_for_space(
            raw_scores,
            eval_rows,
            candidate_filter=candidate_filter,
            score_space=score_space,
            calibration_cdf=calibration_cdf,
        )
        best, sweep = evaluate_route_scores(
            eval_rows,
            route_scores,
            candidate_filter=candidate_filter,
            min_selected_count=min_selected_count,
            objective=tau_objective,
        )
        if not best:
            continue
        route_tau = maybe_float(best.get("tau"))
        if route_tau is None:
            continue
        if str(score_space) == "raw":
            tau_raw = float(route_tau)
            tau_percentile = empirical_cdf(tau_raw, calibration_cdf)
        else:
            tau_percentile = float(route_tau)
            # First raw score whose discovery percentile reaches the selected route tau.
            sorted_cdf = sorted(float(x) for x in calibration_cdf)
            idx = min(max(0, math.ceil(float(tau_percentile) * len(sorted_cdf)) - 1), len(sorted_cdf) - 1)
            tau_raw = float(sorted_cdf[idx])
        key = "cd_fusion" if family == "cd_fusion" else family
        if family == "cd_fusion":
            key = f"cd_fusion_a{alpha:.3f}".replace(".", "p")
        result = {
            **best,
            "family": family,
            "alpha": float(alpha),
            "tau_raw": float(tau_raw),
            "tau_percentile": float(tau_percentile),
            "score_space": str(score_space),
            "route_tau": float(route_tau),
            "calibration_score_count": int(len(calibration_cdf)),
        }
        family_results[key] = result
        family_cdfs[key] = sorted(float(x) for x in calibration_cdf)
        for row in sweep:
            family_sweeps.append({**row, "family": family, "alpha": float(alpha), "family_key": key})

    selected_key: Optional[str] = None
    selected_policy: Optional[Dict[str, Any]] = None
    for key, result in family_results.items():
        if selected_policy is None or pcp.selection_key(result, tau_objective) > pcp.selection_key(selected_policy, tau_objective):
            selected_key = key
            selected_policy = result
    if not selected_policy or selected_key is None:
        raise RuntimeError("No viable old-C + layered-D policy was calibrated.")

    return {
        "mode": "oldc_layered_d_fusion_controller",
        "candidate_filter": candidate_filter,
        "score_space": str(score_space),
        "c_feature_cols": list(c_feature_cols),
        "selected_c_features": selected_c,
        "c_feature_metrics": c_metrics,
        "d_policy": {
            "selected_layer": d_policy.get("selected_layer"),
            "selected_d_features": d_policy.get("selected_d_features"),
            "selected_policy": d_policy.get("selected_policy"),
            "layer_grid": d_policy.get("layer_grid"),
            "layer_candidates": d_policy.get("layer_candidates"),
        },
        "selected_key": selected_key,
        "selected_policy": selected_policy,
        "selected_calibration_score_cdf": family_cdfs[selected_key],
        "family_results": family_results,
        "family_score_cdfs": family_cdfs,
        "tau_sweep": family_sweeps,
        "counts": {
            "n_c_rows": int(len(c_rows)),
            "n_d_rows": int(len(d_rows)),
            "n_c_scored": int(len(c_scores)),
            "n_d_scored": int(len(d_scores)),
            "n_common_scored": int(len(set(c_scores) & set(d_scores))),
        },
    }


def apply_policy(
    *,
    c_rows: Sequence[Dict[str, Any]],
    d_rows: Sequence[Dict[str, Any]],
    policy: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    candidate_filter = str(policy.get("candidate_filter") or "changed_answer")
    selected_policy = dict(policy["selected_policy"])
    selected_key = str(policy.get("selected_key") or "")
    family = str(selected_policy["family"])
    alpha = float(selected_policy.get("alpha", 0.0))
    score_space = str(selected_policy.get("score_space") or policy.get("score_space") or "raw")
    route_tau = float(selected_policy.get("route_tau", selected_policy.get("tau")))

    c_rows_by_id = index_by_id(c_rows)
    d_rows_labeled = overlay_labels(d_rows, c_rows_by_id)
    d_rows_by_id, d_scores = score_d_rows_from_policy(d_rows_labeled, policy["d_policy"])
    c_scores = score_c_rows(c_rows_by_id, list(policy.get("selected_c_features") or []))

    merged_rows_by_id: Dict[str, Dict[str, Any]] = {}
    for sid in sorted_ids(set(c_rows_by_id) | set(d_rows_by_id)):
        row: Dict[str, Any] = {}
        if sid in d_rows_by_id:
            row.update(d_rows_by_id[sid])
        if sid in c_rows_by_id:
            row.update(c_rows_by_id[sid])
        row["id"] = sid
        merged_rows_by_id[sid] = row
    raw_scores = build_raw_scores(merged_rows_by_id, c_scores, d_scores, family=family, alpha=alpha)
    eval_rows = [merged_rows_by_id[sid] for sid in sorted_ids(raw_scores)]
    cdf = list(policy.get("family_score_cdfs", {}).get(selected_key) or policy.get("selected_calibration_score_cdf") or [])
    route_scores, route_cdf = route_scores_for_space(
        raw_scores,
        eval_rows,
        candidate_filter=candidate_filter,
        score_space=score_space,
        calibration_cdf=cdf,
    )
    evaluation, _ = evaluate_route_scores(eval_rows, route_scores, candidate_filter=candidate_filter, tau=route_tau)
    evaluation = {
        **evaluation,
        "family": family,
        "alpha": alpha,
        "selected_key": selected_key,
        "score_space": score_space,
        "tau_raw": selected_policy.get("tau_raw"),
        "tau_percentile": selected_policy.get("tau_percentile"),
        "route_tau": route_tau,
        "route_score_count": int(len(route_scores)),
        "route_cdf_count": int(len(route_cdf)),
        "c_scored_count": int(len(c_scores)),
        "d_scored_count": int(len(d_scores)),
        "common_scored_count": int(len(set(c_scores) & set(d_scores))),
    }

    route_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        sid = row_id(row)
        route_score = route_scores.get(sid)
        raw_score = raw_scores.get(sid)
        can_route = pcp.is_route_candidate(dict(row), candidate_filter)
        route = "baseline" if can_route and route_score is not None and float(route_score) >= route_tau else "method"
        final_text = str(row.get("intervention_text", ""))
        final_source = "method"
        if route == "baseline":
            final_text = str(row.get("baseline_text", "")) or final_text
            final_source = "baseline_cached" if str(row.get("baseline_text", "")).strip() else "method_missing_baseline"
        route_rows.append(
            {
                "id": sid,
                "image": str(row.get("image", row.get("image_id", ""))),
                "question": str(row.get("question", row.get("text", ""))),
                "category": str(row.get("category", "")),
                "route": route,
                "family": family,
                "alpha": alpha,
                "selected_key": selected_key,
                "score_space": score_space,
                "tau": route_tau,
                "tau_raw": selected_policy.get("tau_raw"),
                "tau_percentile": selected_policy.get("tau_percentile"),
                "score": route_score,
                "raw_score": raw_score,
                "c_score": c_scores.get(sid),
                "d_score": d_scores.get(sid),
                "route_candidate": int(can_route),
                "harm": int(maybe_int(row.get("harm")) or 0),
                "help": int(maybe_int(row.get("help")) or 0),
                "baseline_correct": row.get("baseline_correct"),
                "intervention_correct": row.get("intervention_correct"),
                "baseline_label": row.get("baseline_label"),
                "intervention_label": row.get("intervention_label"),
                "final_source": final_source,
                "final_text": final_text,
            }
        )
        pred_rows.append(
            {
                "question_id": sid,
                "id": sid,
                "image": str(row.get("image", row.get("image_id", ""))),
                "text": final_text,
                "route": route,
                "family": family,
                "source": final_source,
            }
        )
    return route_rows, pred_rows, evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate/apply old compact C with layered-D replacement.")
    ap.add_argument("--calibration_c_rows_csv", default="")
    ap.add_argument("--calibration_d_trajectory_long_csv", default="")
    ap.add_argument("--apply_c_rows_csv", default="")
    ap.add_argument("--apply_d_trajectory_long_csv", default="")
    ap.add_argument("--policy_json", default="")
    ap.add_argument("--c_policy_json", default="", help="Optional old compact policy JSON to reuse selected_c_features.")
    ap.add_argument("--d_policy_json", default="", help="Optional layered-D policy JSON to reuse selected layer/features.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_filter", default="changed_answer", choices=["all", "changed_answer", "yes_to_no"])
    ap.add_argument(
        "--c_feature_cols",
        default="cheap_target_gap_content_min,cheap_lp_content_min,cheap_lp_content_std",
    )
    ap.add_argument("--derive_decision_kl", type=parse_bool, default=True)
    ap.add_argument("--min_present_rate", type=float, default=0.8)
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k_c", type=int, default=3)
    ap.add_argument("--d_layer_grid", default="quartiles")
    ap.add_argument("--alpha_grid", default="0.1,0.25,0.5,0.75,0.9")
    ap.add_argument("--min_selected_count", type=int, default=5)
    ap.add_argument("--score_space", default="raw", choices=["raw", "percentile", "discovery_percentile", "batch_percentile"])
    ap.add_argument(
        "--tau_objective",
        default="final_acc",
        choices=["final_acc", "net", "harm_precision", "harm_recall", "harm_f1"],
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if str(args.policy_json or "").strip():
        policy = read_json(args.policy_json)
    else:
        if not str(args.calibration_c_rows_csv or "").strip() or not str(args.calibration_d_trajectory_long_csv or "").strip():
            raise RuntimeError("--calibration_c_rows_csv and --calibration_d_trajectory_long_csv are required without --policy_json.")
        calibration_c_rows = pcp.load_rows(
            os.path.abspath(args.calibration_c_rows_csv),
            derive_decision_kl=bool(args.derive_decision_kl),
        )
        calibration_d_rows = read_d_rows(os.path.abspath(args.calibration_d_trajectory_long_csv))
        policy = calibrate_policy(
            c_rows=calibration_c_rows,
            d_rows=calibration_d_rows,
            c_policy_json=str(args.c_policy_json),
            d_policy_json=str(args.d_policy_json),
            c_feature_cols=[x.strip() for x in str(args.c_feature_cols).split(",") if x.strip()],
            d_layer_grid=str(args.d_layer_grid),
            candidate_filter=str(args.candidate_filter),
            min_present_rate=float(args.min_present_rate),
            min_feature_auroc=float(args.min_feature_auroc),
            top_k_c=int(args.top_k_c),
            min_selected_count=int(args.min_selected_count),
            alpha_grid=[float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()],
            score_space=("discovery_percentile" if str(args.score_space) == "percentile" else str(args.score_space)),
            tau_objective=str(args.tau_objective),
        )
        policy["inputs"] = {
            "calibration_c_rows_csv": os.path.abspath(args.calibration_c_rows_csv),
            "calibration_d_trajectory_long_csv": os.path.abspath(args.calibration_d_trajectory_long_csv),
            "c_policy_json": os.path.abspath(args.c_policy_json) if str(args.c_policy_json or "").strip() else "",
            "d_policy_json": os.path.abspath(args.d_policy_json) if str(args.d_policy_json or "").strip() else "",
            "candidate_filter": str(args.candidate_filter),
            "c_feature_cols": [x.strip() for x in str(args.c_feature_cols).split(",") if x.strip()],
            "d_layer_grid": str(args.d_layer_grid),
            "alpha_grid": [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()],
            "min_selected_count": int(args.min_selected_count),
            "score_space": str(args.score_space),
            "tau_objective": str(args.tau_objective),
        }
        write_json(os.path.join(out_dir, "selected_policy.json"), policy)
        base.write_csv(os.path.join(out_dir, "c_feature_metrics.csv"), policy.get("c_feature_metrics") or [])
        base.write_csv(os.path.join(out_dir, "tau_sweep.csv"), policy.get("tau_sweep") or [])
        base.write_csv(
            os.path.join(out_dir, "family_grid_summary.csv"),
            [
                {
                    "family_key": key,
                    "family": row.get("family"),
                    "alpha": row.get("alpha"),
                    "tau": row.get("tau"),
                    "tau_raw": row.get("tau_raw"),
                    "tau_percentile": row.get("tau_percentile"),
                    "score_space": row.get("score_space"),
                    "calibration_score_count": row.get("calibration_score_count"),
                    "n_eval": row.get("n_eval"),
                    "n_route_candidates": row.get("n_route_candidates"),
                    "n_route_candidate_harm": row.get("n_route_candidate_harm"),
                    "n_route_candidate_help": row.get("n_route_candidate_help"),
                    "selected_count": row.get("selected_count"),
                    "selected_harm": row.get("selected_harm"),
                    "selected_help": row.get("selected_help"),
                    "net": row.get("net"),
                    "selected_harm_precision": row.get("selected_harm_precision"),
                    "selected_harm_recall": row.get("selected_harm_recall"),
                    "selected_harm_recall_in_scope": row.get("selected_harm_recall_in_scope"),
                    "delta_vs_intervention": row.get("delta_vs_intervention"),
                }
                for key, row in sorted((policy.get("family_results") or {}).items())
            ],
        )
        print("[saved]", os.path.join(out_dir, "selected_policy.json"))

    if str(args.apply_c_rows_csv or "").strip() and str(args.apply_d_trajectory_long_csv or "").strip():
        apply_c_rows = pcp.load_rows(os.path.abspath(args.apply_c_rows_csv), derive_decision_kl=bool(args.derive_decision_kl))
        apply_d_rows = read_d_rows(os.path.abspath(args.apply_d_trajectory_long_csv))
        route_rows, pred_rows, evaluation = apply_policy(c_rows=apply_c_rows, d_rows=apply_d_rows, policy=policy)
        route_path = os.path.join(out_dir, "pcp_route_rows.csv")
        pred_path = os.path.join(out_dir, "pred_pcp_cd.jsonl")
        base.write_csv(route_path, route_rows)
        write_jsonl(pred_path, pred_rows)
        summary = {
            "mode": "apply_oldc_layered_d_fusion_controller",
            "inputs": {
                "apply_c_rows_csv": os.path.abspath(args.apply_c_rows_csv),
                "apply_d_trajectory_long_csv": os.path.abspath(args.apply_d_trajectory_long_csv),
                "policy_json": os.path.abspath(args.policy_json) if str(args.policy_json or "").strip() else os.path.join(out_dir, "selected_policy.json"),
            },
            "policy": {
                "selected_key": policy.get("selected_key"),
                "selected_policy": policy.get("selected_policy"),
                "selected_c_features": policy.get("selected_c_features"),
                "d_policy": policy.get("d_policy"),
            },
            "evaluation_from_cached_labels": evaluation,
            "outputs": {
                "pcp_route_rows_csv": route_path,
                "pred_jsonl": pred_path,
            },
        }
        write_json(os.path.join(out_dir, "summary.json"), summary)
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
