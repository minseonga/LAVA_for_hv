#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


D_FEATURES = (
    "candidate_minus_alt",
    "candidate_prob_binary",
    "candidate_kl_uniform",
)


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
    value_f = maybe_float(value)
    if value_f is None:
        return None
    return int(round(value_f))


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mu = mean(values)
    return float(max((sum((x - mu) ** 2 for x in values) / len(values)) ** 0.5, 1e-6))


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(scores) != len(labels):
        return None
    n_pos = sum(int(y) for y in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if int(pairs[k][1]) == 1:
                rank_sum += avg_rank
        i = j
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def kl_uniform(p: float) -> float:
    eps = 1e-12
    p = min(1.0 - eps, max(eps, float(p)))
    q = 1.0 - p
    return float(p * math.log(2.0 * p) + q * math.log(2.0 * q))


def read_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out: Dict[str, Any] = dict(row)
            p = maybe_float(out.get("candidate_prob_binary"))
            if p is not None:
                out["candidate_kl_uniform"] = kl_uniform(float(p))
            rows.append(out)
    return rows


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_candidate(row: Mapping[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    base = str(row.get("baseline_label", "")).strip().lower()
    intervention = str(row.get("intervention_label", "")).strip().lower()
    if mode == "changed_answer":
        return base in {"yes", "no"} and intervention in {"yes", "no"} and base != intervention
    if mode == "yes_to_no":
        return base == "yes" and intervention == "no"
    raise ValueError(f"Unsupported candidate_filter={mode!r}")


def orient_feature(rows: Sequence[Mapping[str, Any]], feature: str) -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = maybe_float(row.get(feature))
        y = maybe_int(row.get("harm"))
        help_ = maybe_int(row.get("help"))
        if x is None or y not in {0, 1} or help_ not in {0, 1}:
            continue
        if int(y) == 0 and int(help_) == 0:
            continue
        xs.append(float(x))
        ys.append(int(y))
    if len(xs) < 2:
        return None
    auc_high = binary_auroc(xs, ys)
    auc_low = binary_auroc([-x for x in xs], ys)
    if auc_high is None or auc_low is None:
        return None
    direction = "high" if auc_high >= auc_low else "low"
    oriented = [x if direction == "high" else -x for x in xs]
    return {
        "feature": feature,
        "direction": direction,
        "auroc": max(float(auc_high), float(auc_low)),
        "raw_auroc_high": float(auc_high),
        "mu": mean(oriented),
        "sd": std(oriented),
        "n": len(xs),
        "n_pos": sum(ys),
    }


def oriented_z(row: Mapping[str, Any], metric: Mapping[str, Any]) -> Optional[float]:
    raw = maybe_float(row.get(str(metric["feature"])))
    if raw is None:
        return None
    oriented = raw if str(metric["direction"]) == "high" else -raw
    return float((oriented - float(metric["mu"])) / max(float(metric["sd"]), 1e-6))


def score_row(row: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> Optional[float]:
    vals: List[float] = []
    for metric in metrics:
        z = oriented_z(row, metric)
        if z is None:
            return None
        vals.append(float(z))
    return mean(vals) if vals else None


def threshold_grid(values: Sequence[float]) -> List[float]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return [0.0]
    return sorted(set(finite))


def empirical_cdf(value: float, sorted_values: Sequence[float]) -> float:
    if not sorted_values:
        return 0.0
    return float(bisect.bisect_right(sorted_values, float(value)) / float(len(sorted_values)))


def candidate_score_distribution(
    rows: Sequence[Mapping[str, Any]],
    scores_by_id: Mapping[str, float],
    *,
    candidate_filter: str,
) -> List[float]:
    values = [
        float(scores_by_id[str(row.get("id", ""))])
        for row in rows
        if str(row.get("id", "")) in scores_by_id and is_candidate(row, candidate_filter)
    ]
    return sorted(float(v) for v in values if math.isfinite(float(v)))


def percentile_scores(
    scores_by_id: Mapping[str, float],
    calibration_cdf: Sequence[float],
) -> Dict[str, float]:
    sorted_cdf = sorted(float(v) for v in calibration_cdf if math.isfinite(float(v)))
    return {str(sid): empirical_cdf(float(score), sorted_cdf) for sid, score in scores_by_id.items()}


def group_by_layer(rows: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer = maybe_int(row.get("layer_index"))
        if layer is not None:
            out[int(layer)].append(dict(row))
    return dict(out)


def default_layer_grid(layers: Sequence[int]) -> List[int]:
    if not layers:
        return []
    final_layer = max(int(x) for x in layers)
    raw = [final_layer // 4, final_layer // 2, (3 * final_layer) // 4, final_layer]
    available = sorted(set(int(x) for x in layers))
    grid: List[int] = []
    for target in raw:
        closest = min(available, key=lambda x: (abs(x - target), x))
        if closest not in grid:
            grid.append(closest)
    return grid


def parse_layer_grid(spec: str, layers: Sequence[int]) -> List[int]:
    spec = str(spec or "quartiles").strip().lower()
    available = sorted(set(int(x) for x in layers))
    if spec in {"quartiles", "default"}:
        return default_layer_grid(available)
    if spec == "all":
        return available
    requested = [int(x.strip()) for x in spec.split(",") if x.strip()]
    missing = [x for x in requested if x not in available]
    if missing:
        raise RuntimeError(f"Requested layers are unavailable: {missing}; available={available}")
    return requested


def evaluate_scores(
    rows: Sequence[Mapping[str, Any]],
    scores_by_id: Mapping[str, float],
    *,
    candidate_filter: str,
    tau: Optional[float] = None,
    min_selected_count: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidate_scores = [
        float(scores_by_id[str(row.get("id", ""))])
        for row in rows
        if str(row.get("id", "")) in scores_by_id and is_candidate(row, candidate_filter)
    ]
    taus = [float(tau)] if tau is not None else threshold_grid(candidate_scores)
    best: Optional[Dict[str, Any]] = None
    sweep: List[Dict[str, Any]] = []
    for value in taus:
        n_eval = 0
        n_route_candidates = 0
        total_harm = 0
        total_help = 0
        route_candidate_harm = 0
        route_candidate_help = 0
        route_candidate_neutral = 0
        selected = 0
        selected_harm = 0
        selected_help = 0
        selected_neutral = 0
        baseline_correct_total = 0
        intervention_correct_total = 0
        final_correct_total = 0
        for row in rows:
            sid = str(row.get("id", ""))
            if sid not in scores_by_id:
                continue
            bc = maybe_int(row.get("baseline_correct"))
            ic = maybe_int(row.get("intervention_correct"))
            if bc is None or ic is None:
                continue
            harm = int(maybe_int(row.get("harm")) or 0)
            help_ = int(maybe_int(row.get("help")) or 0)
            can_route = is_candidate(row, candidate_filter)
            route_baseline = can_route and float(scores_by_id[sid]) >= float(value)
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
            if route_baseline:
                selected += 1
                selected_harm += harm
                selected_help += help_
                selected_neutral += int(harm == 0 and help_ == 0)
                final_correct_total += int(bc)
            else:
                final_correct_total += int(ic)
        result = {
            "tau": float(value),
            "n_eval": int(n_eval),
            "baseline_rate": float(selected / max(1, n_eval)),
            "method_rate": float(1.0 - selected / max(1, n_eval)),
            "baseline_acc": float(baseline_correct_total / max(1, n_eval)),
            "intervention_acc": float(intervention_correct_total / max(1, n_eval)),
            "final_acc": float(final_correct_total / max(1, n_eval)),
            "delta_vs_intervention": float((final_correct_total - intervention_correct_total) / max(1, n_eval)),
            "selected_count": int(selected),
            "total_harm": int(total_harm),
            "total_help": int(total_help),
            "n_route_candidates": int(n_route_candidates),
            "n_route_candidate_harm": int(route_candidate_harm),
            "n_route_candidate_help": int(route_candidate_help),
            "n_route_candidate_neutral": int(route_candidate_neutral),
            "route_candidate_baseline_rate": float(selected / max(1, n_route_candidates)),
            "selected_harm": int(selected_harm),
            "selected_help": int(selected_help),
            "selected_neutral": int(selected_neutral),
            "net": int(selected_harm - selected_help),
            "selected_harm_precision": float(selected_harm / max(1, selected)),
            "selected_harm_recall": float(selected_harm / max(1, total_harm)),
            "selected_harm_recall_in_scope": float(selected_harm / max(1, route_candidate_harm)),
            "selected_help_recall_in_scope": float(selected_help / max(1, route_candidate_help)),
        }
        sweep.append(result)
        if selected < int(min_selected_count):
            continue
        if best is None or (
            int(result["net"]),
            int(result["selected_harm"]),
            -int(result["selected_help"]),
        ) > (
            int(best["net"]),
            int(best["selected_harm"]),
            -int(best["selected_help"]),
        ):
            best = result
    return best or (sweep[0] if sweep else {}), sweep


def calibrate(
    rows: Sequence[Dict[str, Any]],
    *,
    layer_grid: Sequence[int],
    candidate_filter: str,
    min_selected_count: int,
    score_space: str,
) -> Dict[str, Any]:
    score_space = str(score_space or "raw").strip().lower()
    if score_space not in {"raw", "percentile", "discovery_percentile", "batch_percentile"}:
        raise ValueError(
            f"Unsupported score_space={score_space!r}; expected raw, discovery_percentile, percentile, or batch_percentile"
        )
    if score_space == "percentile":
        score_space = "discovery_percentile"
    by_layer = group_by_layer(rows)
    candidates: List[Dict[str, Any]] = []
    for layer in layer_grid:
        layer_rows = by_layer.get(int(layer), [])
        fit_rows = [row for row in layer_rows if is_candidate(row, candidate_filter)]
        metrics = [m for feature in D_FEATURES if (m := orient_feature(fit_rows, feature)) is not None]
        scores = {
            str(row.get("id", "")): score
            for row in layer_rows
            if (score := score_row(row, metrics)) is not None
        }
        best, sweep = evaluate_scores(
            layer_rows,
            scores,
            candidate_filter=candidate_filter,
            min_selected_count=min_selected_count,
        )
        calibration_cdf = candidate_score_distribution(layer_rows, scores, candidate_filter=candidate_filter)
        raw_tau = maybe_float(best.get("tau")) if best else None
        tau_percentile = None if raw_tau is None else empirical_cdf(float(raw_tau), calibration_cdf)
        if best:
            best = {
                **best,
                "tau_raw": float(raw_tau),
                "tau_percentile": tau_percentile,
                "score_space": score_space,
                "calibration_score_count": int(len(calibration_cdf)),
            }
        sweep = [
            {
                **row,
                "tau_raw": row.get("tau"),
                "tau_percentile": empirical_cdf(float(row["tau"]), calibration_cdf) if "tau" in row else None,
            }
            for row in sweep
        ]
        candidates.append(
            {
                "layer": int(layer),
                "selected_d_features": metrics,
                "best": best,
                "sweep": sweep,
                "calibration_score_cdf": calibration_cdf,
            }
        )
    viable = [x for x in candidates if x.get("best")]
    if not viable:
        raise RuntimeError("No viable layer candidates were calibrated.")
    selected = max(
        viable,
        key=lambda x: (
            int(x["best"]["net"]),
            int(x["best"]["selected_harm"]),
            -int(x["best"]["selected_help"]),
            -int(x["layer"]),
        ),
    )
    selected_policy = {
        "family": "layered_d",
        "layer": int(selected["layer"]),
        **selected["best"],
    }
    if score_space in {"discovery_percentile", "batch_percentile"}:
        selected_policy["route_tau"] = float(selected_policy["tau_percentile"])
    else:
        selected_policy["route_tau"] = float(selected_policy["tau_raw"])

    return {
        "mode": "layered_d_family_controller",
        "candidate_filter": candidate_filter,
        "score_space": score_space,
        "layer_grid": [int(x) for x in layer_grid],
        "selected_layer": int(selected["layer"]),
        "selected_d_features": selected["selected_d_features"],
        "selected_policy": selected_policy,
        "calibration_score_cdf": selected.get("calibration_score_cdf", []),
        "layer_candidates": [
            {
                "layer": int(x["layer"]),
                "selected_d_features": x["selected_d_features"],
                "best": x["best"],
                "calibration_score_count": int(len(x.get("calibration_score_cdf", []))),
            }
            for x in candidates
        ],
    }


def apply_policy(
    rows: Sequence[Dict[str, Any]],
    policy: Mapping[str, Any],
    *,
    candidate_filter: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    layer = int(policy["selected_layer"])
    metrics = list(policy.get("selected_d_features") or [])
    selected_policy = dict(policy.get("selected_policy") or {})
    score_space = str(selected_policy.get("score_space") or policy.get("score_space") or "raw").strip().lower()
    if score_space == "percentile":
        score_space = "discovery_percentile"
    if score_space not in {"raw", "discovery_percentile", "batch_percentile"}:
        raise ValueError(f"Unsupported score_space={score_space!r}; expected raw, discovery_percentile, or batch_percentile")
    tau_raw = maybe_float(selected_policy.get("tau_raw", selected_policy.get("tau")))
    tau_percentile = maybe_float(selected_policy.get("tau_percentile"))
    if tau_raw is None:
        raise RuntimeError("Policy is missing a raw tau.")
    layer_rows = group_by_layer(rows).get(layer, [])
    raw_scores = {
        str(row.get("id", "")): score
        for row in layer_rows
        if (score := score_row(row, metrics)) is not None
    }
    if score_space == "discovery_percentile":
        calibration_cdf = policy.get("calibration_score_cdf") or selected_policy.get("calibration_score_cdf") or []
        if not calibration_cdf:
            raise RuntimeError("discovery_percentile score_space requires calibration_score_cdf in the policy.")
        route_scores = percentile_scores(raw_scores, calibration_cdf)
        tau_route = float(tau_percentile if tau_percentile is not None else empirical_cdf(float(tau_raw), sorted(calibration_cdf)))
    elif score_space == "batch_percentile":
        if tau_percentile is None:
            calibration_cdf = policy.get("calibration_score_cdf") or selected_policy.get("calibration_score_cdf") or []
            if not calibration_cdf:
                raise RuntimeError("batch_percentile score_space requires tau_percentile or calibration_score_cdf in the policy.")
            tau_percentile = empirical_cdf(float(tau_raw), sorted(calibration_cdf))
        batch_cdf = candidate_score_distribution(layer_rows, raw_scores, candidate_filter=candidate_filter)
        route_scores = percentile_scores(raw_scores, batch_cdf)
        tau_route = float(tau_percentile)
    else:
        route_scores = dict(raw_scores)
        tau_route = float(tau_raw)
    evaluation, _ = evaluate_scores(layer_rows, route_scores, candidate_filter=candidate_filter, tau=tau_route)
    evaluation = {
        **evaluation,
        "score_space": score_space,
        "tau_raw": float(tau_raw),
        "tau_percentile": tau_percentile,
        "route_tau": float(tau_route),
    }
    route_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    for row in layer_rows:
        sid = str(row.get("id", ""))
        raw_score = raw_scores.get(sid)
        route_score = route_scores.get(sid)
        can_route = is_candidate(row, candidate_filter)
        route = "baseline" if can_route and route_score is not None and float(route_score) >= tau_route else "method"
        final_text = str(row.get("intervention_text", ""))
        final_source = "method"
        if route == "baseline":
            final_text = str(row.get("baseline_text", "")) or final_text
            final_source = "baseline_cached" if str(row.get("baseline_text", "")).strip() else "method_missing_baseline"
        route_rows.append(
            {
                "id": sid,
                "image": str(row.get("image", "")),
                "question": str(row.get("question", "")),
                "route": route,
                "family": "layered_d",
                "layer": layer,
                "score_space": score_space,
                "tau": tau_route,
                "tau_raw": tau_raw,
                "tau_percentile": tau_percentile,
                "score": route_score,
                "raw_score": raw_score,
                "score_percentile": route_score if score_space in {"discovery_percentile", "batch_percentile"} else None,
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
                "image": str(row.get("image", "")),
                "text": final_text,
                "route": route,
                "family": "layered_d",
                "layer": layer,
                "score_space": score_space,
                "source": final_source,
            }
        )
    return route_rows, pred_rows, evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate and apply layer-grid D-family RAPIC from trajectory long CSVs.")
    ap.add_argument("--calibration_trajectory_long_csv", default="")
    ap.add_argument("--apply_trajectory_long_csv", default="")
    ap.add_argument("--policy_json", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_filter", default="changed_answer", choices=["all", "changed_answer", "yes_to_no"])
    ap.add_argument("--layer_grid", default="quartiles", help="'quartiles', 'all', or comma-separated layer indices.")
    ap.add_argument("--min_selected_count", type=int, default=5)
    ap.add_argument(
        "--score_space",
        default="raw",
        choices=["raw", "percentile", "discovery_percentile", "batch_percentile"],
        help=(
            "Route with raw scores, discovery-CDF percentiles, or apply-batch percentiles. "
            "Calibration objective is unchanged. 'percentile' is kept as an alias for discovery_percentile."
        ),
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    policy: Dict[str, Any]
    if str(args.policy_json or "").strip():
        with open(os.path.abspath(args.policy_json), "r", encoding="utf-8") as f:
            policy = json.load(f)
    else:
        if not str(args.calibration_trajectory_long_csv or "").strip():
            raise RuntimeError("--calibration_trajectory_long_csv is required when --policy_json is not provided.")
        calibration_rows = read_rows(os.path.abspath(args.calibration_trajectory_long_csv))
        available_layers = sorted(group_by_layer(calibration_rows))
        layer_grid = parse_layer_grid(str(args.layer_grid), available_layers)
        policy = calibrate(
            calibration_rows,
            layer_grid=layer_grid,
            candidate_filter=str(args.candidate_filter),
            min_selected_count=int(args.min_selected_count),
            score_space=str(args.score_space),
        )
        policy["inputs"] = {
            "calibration_trajectory_long_csv": os.path.abspath(args.calibration_trajectory_long_csv),
            "candidate_filter": str(args.candidate_filter),
            "layer_grid": str(args.layer_grid),
            "min_selected_count": int(args.min_selected_count),
            "score_space": str(args.score_space),
        }
        write_json(os.path.join(out_dir, "selected_policy.json"), policy)
        write_csv(
            os.path.join(out_dir, "layer_grid_summary.csv"),
            [
                {
                    "layer": row["layer"],
                    "tau": row["best"].get("tau"),
                    "tau_raw": row["best"].get("tau_raw"),
                    "tau_percentile": row["best"].get("tau_percentile"),
                    "score_space": row["best"].get("score_space"),
                    "calibration_score_count": row.get("calibration_score_count"),
                    "selected_count": row["best"].get("selected_count"),
                    "selected_harm": row["best"].get("selected_harm"),
                    "selected_help": row["best"].get("selected_help"),
                    "net": row["best"].get("net"),
                    "selected_harm_precision": row["best"].get("selected_harm_precision"),
                    "selected_harm_recall": row["best"].get("selected_harm_recall"),
                    "feature_aurocs": ";".join(
                        f"{m['feature']}={float(m['auroc']):.6f}:{m['direction']}"
                        for m in row.get("selected_d_features", [])
                    ),
                }
                for row in policy.get("layer_candidates", [])
            ],
        )
        print("[saved]", os.path.join(out_dir, "selected_policy.json"))

    if str(args.apply_trajectory_long_csv or "").strip():
        apply_rows = read_rows(os.path.abspath(args.apply_trajectory_long_csv))
        route_rows, pred_rows, evaluation = apply_policy(
            apply_rows,
            policy,
            candidate_filter=str(policy.get("candidate_filter") or args.candidate_filter),
        )
        route_path = os.path.join(out_dir, "pcp_route_rows.csv")
        pred_path = os.path.join(out_dir, "pred_pcp_cd.jsonl")
        write_csv(route_path, route_rows)
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "mode": "apply_layered_d_family_controller",
            "inputs": {
                "apply_trajectory_long_csv": os.path.abspath(args.apply_trajectory_long_csv),
                "policy_json": os.path.abspath(args.policy_json) if str(args.policy_json or "").strip() else os.path.join(out_dir, "selected_policy.json"),
            },
            "policy": {
                "selected_layer": policy.get("selected_layer"),
                "selected_d_features": policy.get("selected_d_features"),
                "selected_policy": policy.get("selected_policy"),
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
