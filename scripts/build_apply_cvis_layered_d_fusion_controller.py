#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_apply_layered_d_family_controller import (  # noqa: E402
    calibrate as calibrate_layered_d,
    candidate_score_distribution,
    empirical_cdf,
    evaluate_scores,
    group_by_layer,
    is_candidate,
    maybe_float,
    maybe_int,
    mean,
    parse_layer_grid,
    percentile_scores,
    read_rows as read_d_rows,
    score_row as score_d_row,
    std,
    write_csv,
    write_json,
)


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json(path: str) -> Dict[str, Any]:
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id", row.get("question_id", ""))).strip()


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(int(y) for y in labels)
    n_neg = len(labels) - n_pos
    if len(scores) != len(labels) or n_pos == 0 or n_neg == 0:
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


def filter_c_rows(rows: Sequence[Dict[str, Any]], c_layer: str) -> List[Dict[str, Any]]:
    layer_spec = str(c_layer or "").strip().lower()
    if layer_spec == "":
        return [dict(row) for row in rows]
    available = sorted({int(maybe_int(row.get("layer_index"))) for row in rows if maybe_int(row.get("layer_index")) is not None})
    if not available:
        raise RuntimeError("--c_layer was provided, but c_rows_csv has no layer_index column.")
    if layer_spec == "final":
        target = max(available)
    else:
        target = int(layer_spec)
    return [dict(row) for row in rows if maybe_int(row.get("layer_index")) == target]


def index_rows_by_id(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = row_id(row)
        if sid and sid not in out:
            out[sid] = dict(row)
    return out


def orient_c_feature(rows: Sequence[Mapping[str, Any]], feature: str, candidate_filter: str) -> Dict[str, Any]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        if not is_candidate(row, candidate_filter):
            continue
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
        raise RuntimeError(f"No usable rows to fit C feature={feature!r}.")
    auc_high = binary_auroc(xs, ys)
    auc_low = binary_auroc([-x for x in xs], ys)
    if auc_high is None or auc_low is None:
        raise RuntimeError(f"C feature={feature!r} has no positive/negative labels.")
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


def score_c_row(row: Mapping[str, Any], metric: Mapping[str, Any]) -> Optional[float]:
    raw = maybe_float(row.get(str(metric["feature"])))
    if raw is None:
        return None
    oriented = raw if str(metric["direction"]) == "high" else -raw
    return float((oriented - float(metric["mu"])) / max(float(metric["sd"]), 1e-6))


def score_c_rows(rows_by_id: Mapping[str, Mapping[str, Any]], metric: Mapping[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for sid, row in rows_by_id.items():
        score = score_c_row(row, metric)
        if score is not None:
            scores[str(sid)] = float(score)
    return scores


def load_or_calibrate_d_policy(
    *,
    d_policy_json: str,
    calibration_d_rows: Sequence[Dict[str, Any]],
    layer_grid_spec: str,
    candidate_filter: str,
    min_selected_count: int,
) -> Dict[str, Any]:
    if str(d_policy_json or "").strip():
        return read_json(d_policy_json)
    available_layers = sorted(group_by_layer(calibration_d_rows))
    if not available_layers:
        raise RuntimeError("No layer_index values were found in calibration D trajectory CSV.")
    layer_grid = parse_layer_grid(layer_grid_spec, available_layers)
    return calibrate_layered_d(
        calibration_d_rows,
        layer_grid=layer_grid,
        candidate_filter=candidate_filter,
        min_selected_count=min_selected_count,
        score_space="raw",
    )


def score_d_rows(rows: Sequence[Dict[str, Any]], d_policy: Mapping[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    layer = int(d_policy["selected_layer"])
    metrics = list(d_policy.get("selected_d_features") or [])
    layer_rows = group_by_layer(rows).get(layer, [])
    rows_by_id = index_rows_by_id(layer_rows)
    scores = {
        str(sid): float(score)
        for sid, row in rows_by_id.items()
        if (score := score_d_row(row, metrics)) is not None
    }
    return rows_by_id, scores


def default_object_layer_grid(layers: Sequence[int]) -> List[int]:
    if not layers:
        return []
    final_layer = max(int(x) for x in layers)
    raw = [(3 * final_layer) // 4, (7 * final_layer) // 8, final_layer]
    available = sorted(set(int(x) for x in layers))
    grid: List[int] = []
    for target in raw:
        closest = min(available, key=lambda x: (abs(x - target), x))
        if closest not in grid:
            grid.append(closest)
    return grid


def parse_object_layer_grid(spec: str, layers: Sequence[int]) -> List[int]:
    spec = str(spec or "late").strip().lower()
    available = sorted(set(int(x) for x in layers))
    if spec in {"late", "late_semantic", "object_default"}:
        return default_object_layer_grid(available)
    if spec == "all":
        return available
    return parse_layer_grid(spec, available)


def load_or_calibrate_object_policy(
    *,
    object_rows: Sequence[Dict[str, Any]],
    object_feature: str,
    object_layer_grid: str,
    candidate_filter: str,
    min_selected_count: int,
) -> Optional[Dict[str, Any]]:
    if not object_rows:
        return None
    by_layer = group_by_layer(object_rows)
    available_layers = sorted(by_layer)
    if not available_layers:
        raise RuntimeError("No layer_index values were found in object trajectory CSV.")
    layer_grid = parse_object_layer_grid(object_layer_grid, available_layers)
    candidates: List[Dict[str, Any]] = []
    for layer in layer_grid:
        layer_rows = by_layer.get(int(layer), [])
        rows_by_id = index_rows_by_id(layer_rows)
        metric = orient_c_feature(list(rows_by_id.values()), object_feature, candidate_filter)
        scores = score_c_rows(rows_by_id, metric)
        best, _ = evaluate_scores(
            layer_rows,
            scores,
            candidate_filter=candidate_filter,
            min_selected_count=min_selected_count,
        )
        candidates.append(
            {
                "layer": int(layer),
                "object_metric": metric,
                "best": best,
            }
        )
    viable = [x for x in candidates if x.get("best")]
    if not viable:
        raise RuntimeError("No viable object-layer candidates were calibrated.")
    selected = max(
        viable,
        key=lambda x: (
            int(x["best"]["net"]),
            int(x["best"]["selected_harm"]),
            -int(x["best"]["selected_help"]),
            -int(x["layer"]),
        ),
    )
    return {
        "family": "object_layer_d",
        "selected_layer": int(selected["layer"]),
        "object_feature": object_feature,
        "object_metric": selected["object_metric"],
        "selected_policy": {
            "family": "object_layer_d",
            "layer": int(selected["layer"]),
            **dict(selected["best"]),
        },
        "layer_grid": [int(x) for x in layer_grid],
        "layer_candidates": candidates,
    }


def score_object_rows(
    rows: Sequence[Dict[str, Any]],
    object_policy: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    if not object_policy:
        return {}, {}
    layer = int(object_policy["selected_layer"])
    metric = dict(object_policy["object_metric"])
    layer_rows = group_by_layer(rows).get(layer, [])
    rows_by_id = index_rows_by_id(layer_rows)
    return rows_by_id, score_c_rows(rows_by_id, metric)


def merge_rows(
    c_rows_by_id: Mapping[str, Mapping[str, Any]],
    d_rows_by_id: Mapping[str, Mapping[str, Any]],
    object_rows_by_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    object_rows_by_id = object_rows_by_id or {}
    ids = sorted(set(c_rows_by_id) | set(d_rows_by_id) | set(object_rows_by_id), key=lambda x: (len(str(x)), str(x)))
    merged: Dict[str, Dict[str, Any]] = {}
    for sid in ids:
        row: Dict[str, Any] = {}
        if sid in d_rows_by_id:
            row.update(dict(d_rows_by_id[sid]))
        if sid in object_rows_by_id:
            for key, value in object_rows_by_id[sid].items():
                if key not in row or str(row.get(key, "")).strip() == "":
                    row[key] = value
        if sid in c_rows_by_id:
            for key, value in c_rows_by_id[sid].items():
                if key not in row or str(row.get(key, "")).strip() == "":
                    row[key] = value
        row["id"] = sid
        merged[sid] = row
    return merged


def parse_fusion_specs(modes: str, alpha_grid: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    requested = [x.strip().lower() for x in str(modes or "mean").split(",") if x.strip()]
    for mode in requested:
        if mode in {"mean", "min", "max"}:
            specs.append({"mode": mode})
        elif mode in {"alpha", "alpha_grid"}:
            for token in str(alpha_grid or "").split(","):
                token = token.strip()
                if not token:
                    continue
                alpha = float(token)
                if alpha < 0.0 or alpha > 1.0:
                    raise ValueError(f"alpha must be in [0,1], got {alpha}")
                specs.append({"mode": "alpha", "c_weight": alpha})
        else:
            raise ValueError(f"Unsupported fusion mode={mode!r}")
    if not specs:
        specs.append({"mode": "mean"})
    return specs


def fusion_score(values: Sequence[float], spec: Mapping[str, Any]) -> float:
    vals = [float(x) for x in values]
    if not vals:
        raise ValueError("Cannot fuse an empty score list.")
    mode = str(spec.get("mode", "mean"))
    if mode == "mean":
        return float(sum(vals) / len(vals))
    if mode == "min":
        return float(min(vals))
    if mode == "max":
        return float(max(vals))
    if mode == "alpha":
        if len(vals) != 2:
            raise ValueError("alpha fusion is only defined for two score streams: C and D.")
        c_weight = float(spec.get("c_weight", 0.5))
        return float(c_weight * vals[0] + (1.0 - c_weight) * vals[1])
    raise ValueError(f"Unsupported fusion mode={mode!r}")


def fusion_name(spec: Mapping[str, Any]) -> str:
    mode = str(spec.get("mode", "mean"))
    if mode == "alpha":
        return f"alpha_c{float(spec.get('c_weight', 0.5)):.3f}".replace(".", "p")
    return mode


def build_fusion_scores(
    score_maps: Sequence[Mapping[str, float]],
    spec: Mapping[str, Any],
    *,
    required_streams: Optional[int] = None,
) -> Dict[str, float]:
    if not score_maps:
        return {}
    n_required = len(score_maps) if required_streams is None else int(required_streams)
    n_required = max(1, min(n_required, len(score_maps)))
    common = set(score_maps[0])
    for scores in score_maps[1:n_required]:
        common &= set(scores)
    out: Dict[str, float] = {}
    for sid in sorted(common, key=lambda x: (len(str(x)), str(x))):
        values = [float(scores[sid]) for scores in score_maps if sid in scores]
        out[str(sid)] = fusion_score(values, spec)
    return out


def add_tau_metadata(
    best: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    scores_by_id: Mapping[str, float],
    *,
    candidate_filter: str,
    score_space: str,
) -> Tuple[Dict[str, Any], List[float]]:
    cdf = candidate_score_distribution(calibration_rows, scores_by_id, candidate_filter=candidate_filter)
    raw_tau = maybe_float(best.get("tau"))
    if raw_tau is None:
        raise RuntimeError("Selected policy is missing tau.")
    tau_percentile = empirical_cdf(float(raw_tau), cdf)
    out = {
        **dict(best),
        "tau_raw": float(raw_tau),
        "tau_percentile": float(tau_percentile),
        "score_space": score_space,
        "calibration_score_count": int(len(cdf)),
        "route_tau": float(tau_percentile if score_space in {"discovery_percentile", "batch_percentile"} else raw_tau),
    }
    return out, cdf


def calibrate_fusion(
    *,
    c_rows: Sequence[Dict[str, Any]],
    d_rows: Sequence[Dict[str, Any]],
    object_rows: Sequence[Dict[str, Any]],
    c_feature: str,
    c_layer: str,
    d_policy_json: str,
    d_layer_grid: str,
    object_feature: str,
    object_layer_grid: str,
    candidate_filter: str,
    min_selected_count: int,
    score_space: str,
    fusion_modes: str,
    alpha_grid: str,
) -> Dict[str, Any]:
    score_space = str(score_space or "raw").strip().lower()
    if score_space == "percentile":
        score_space = "discovery_percentile"
    if score_space not in {"raw", "discovery_percentile", "batch_percentile"}:
        raise ValueError(f"Unsupported score_space={score_space!r}")

    c_rows_f = filter_c_rows(c_rows, c_layer)
    c_rows_by_id = index_rows_by_id(c_rows_f)
    c_metric = orient_c_feature(list(c_rows_by_id.values()), c_feature, candidate_filter)
    c_scores = score_c_rows(c_rows_by_id, c_metric)

    d_policy = load_or_calibrate_d_policy(
        d_policy_json=d_policy_json,
        calibration_d_rows=d_rows,
        layer_grid_spec=d_layer_grid,
        candidate_filter=candidate_filter,
        min_selected_count=min_selected_count,
    )
    d_rows_by_id, d_scores = score_d_rows(d_rows, d_policy)

    object_policy = load_or_calibrate_object_policy(
        object_rows=object_rows,
        object_feature=object_feature,
        object_layer_grid=object_layer_grid,
        candidate_filter=candidate_filter,
        min_selected_count=min_selected_count,
    )
    object_rows_by_id, object_scores = score_object_rows(object_rows, object_policy)

    score_maps: List[Mapping[str, float]] = [c_scores, d_scores]
    if object_policy:
        score_maps.append(object_scores)
    common_ids = set(c_scores) & set(d_scores)
    merged_rows_by_id = merge_rows(c_rows_by_id, d_rows_by_id, object_rows_by_id)
    eval_rows = [merged_rows_by_id[sid] for sid in sorted(common_ids, key=lambda x: (len(str(x)), str(x)))]

    candidates: List[Dict[str, Any]] = []
    for spec in parse_fusion_specs(fusion_modes, alpha_grid):
        if object_policy and str(spec.get("mode")) == "alpha":
            continue
        try:
            scores = build_fusion_scores(score_maps, spec, required_streams=2)
        except ValueError:
            continue
        best, sweep = evaluate_scores(
            eval_rows,
            scores,
            candidate_filter=candidate_filter,
            min_selected_count=min_selected_count,
        )
        selected_policy, cdf = add_tau_metadata(
            best,
            eval_rows,
            scores,
            candidate_filter=candidate_filter,
            score_space=score_space,
        )
        candidates.append(
            {
                "fusion": spec,
                "fusion_name": fusion_name(spec),
                "selected_policy": selected_policy,
                "calibration_score_cdf": cdf,
                "sweep": sweep,
            }
        )
    if not candidates:
        raise RuntimeError("No viable fusion candidates. alpha_grid is only available for two score streams.")

    selected = max(
        candidates,
        key=lambda x: (
            int(x["selected_policy"]["net"]),
            int(x["selected_policy"]["selected_harm"]),
            -int(x["selected_policy"]["selected_help"]),
        ),
    )
    selected_policy = {
        "family": "cvis_layered_d_object_fusion" if object_policy else "cvis_layered_d_fusion",
        "fusion": selected["fusion"],
        "fusion_name": selected["fusion_name"],
        **selected["selected_policy"],
    }

    return {
        "mode": "cvis_layered_d_fusion_controller",
        "candidate_filter": candidate_filter,
        "score_space": score_space,
        "c_feature": c_feature,
        "c_layer": c_layer,
        "c_metric": c_metric,
        "d_policy": {
            "selected_layer": d_policy.get("selected_layer"),
            "selected_d_features": d_policy.get("selected_d_features"),
            "selected_policy": d_policy.get("selected_policy"),
            "layer_grid": d_policy.get("layer_grid"),
        },
        "object_policy": (
            None
            if not object_policy
            else {
                "selected_layer": object_policy.get("selected_layer"),
                "object_feature": object_policy.get("object_feature"),
                "object_metric": object_policy.get("object_metric"),
                "selected_policy": object_policy.get("selected_policy"),
                "layer_grid": object_policy.get("layer_grid"),
            }
        ),
        "selected_policy": selected_policy,
        "calibration_score_cdf": selected["calibration_score_cdf"],
        "fusion_candidates": [
            {
                "fusion": x["fusion"],
                "fusion_name": x["fusion_name"],
                "selected_policy": x["selected_policy"],
                "calibration_score_count": int(len(x["calibration_score_cdf"])),
            }
            for x in candidates
        ],
    }


def routed_scores(
    raw_scores: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    candidate_filter: str,
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    selected_policy = dict(policy["selected_policy"])
    score_space = str(selected_policy.get("score_space") or policy.get("score_space") or "raw").strip().lower()
    if score_space == "percentile":
        score_space = "discovery_percentile"
    tau_raw = maybe_float(selected_policy.get("tau_raw", selected_policy.get("tau")))
    tau_percentile = maybe_float(selected_policy.get("tau_percentile"))
    if tau_raw is None:
        raise RuntimeError("Policy is missing tau_raw.")
    if score_space == "raw":
        return dict(raw_scores), float(tau_raw), {"tau_raw": tau_raw, "tau_percentile": tau_percentile, "score_space": score_space}
    if tau_percentile is None:
        cdf = policy.get("calibration_score_cdf") or []
        tau_percentile = empirical_cdf(float(tau_raw), cdf)
    if score_space == "discovery_percentile":
        cdf = policy.get("calibration_score_cdf") or []
        if not cdf:
            raise RuntimeError("discovery_percentile requires calibration_score_cdf.")
        return percentile_scores(raw_scores, cdf), float(tau_percentile), {
            "tau_raw": tau_raw,
            "tau_percentile": tau_percentile,
            "score_space": score_space,
        }
    if score_space == "batch_percentile":
        cdf = candidate_score_distribution(rows, raw_scores, candidate_filter=candidate_filter)
        return percentile_scores(raw_scores, cdf), float(tau_percentile), {
            "tau_raw": tau_raw,
            "tau_percentile": tau_percentile,
            "score_space": score_space,
            "batch_score_count": len(cdf),
        }
    raise ValueError(f"Unsupported score_space={score_space!r}")


def apply_fusion(
    *,
    c_rows: Sequence[Dict[str, Any]],
    d_rows: Sequence[Dict[str, Any]],
    object_rows: Sequence[Dict[str, Any]],
    policy: Mapping[str, Any],
    candidate_filter: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    c_rows_f = filter_c_rows(c_rows, str(policy.get("c_layer", "")))
    c_rows_by_id = index_rows_by_id(c_rows_f)
    c_scores = score_c_rows(c_rows_by_id, policy["c_metric"])
    d_rows_by_id, d_scores = score_d_rows(d_rows, policy["d_policy"])
    object_policy = policy.get("object_policy")
    object_rows_by_id, object_scores = score_object_rows(object_rows, object_policy)
    merged_rows_by_id = merge_rows(c_rows_by_id, d_rows_by_id, object_rows_by_id)

    spec = dict(policy["selected_policy"]["fusion"])
    score_maps: List[Mapping[str, float]] = [c_scores, d_scores]
    if object_policy:
        score_maps.append(object_scores)
    raw_scores = build_fusion_scores(score_maps, spec, required_streams=2)
    eval_rows = [merged_rows_by_id[sid] for sid in sorted(raw_scores, key=lambda x: (len(str(x)), str(x)))]
    route_scores, tau_route, tau_meta = routed_scores(raw_scores, eval_rows, policy, candidate_filter=candidate_filter)
    evaluation, _ = evaluate_scores(eval_rows, route_scores, candidate_filter=candidate_filter, tau=tau_route)
    evaluation = {
        **evaluation,
        **tau_meta,
        "route_tau": float(tau_route),
        "fusion": spec,
        "fusion_name": policy["selected_policy"].get("fusion_name"),
    }

    route_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        sid = row_id(row)
        route_score = route_scores.get(sid)
        raw_score = raw_scores.get(sid)
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
                "image": str(row.get("image", row.get("image_id", ""))),
                "question": str(row.get("question", row.get("text", ""))),
                "category": str(row.get("category", "")),
                "route": route,
                "family": "cvis_layered_d_object_fusion" if object_policy else "cvis_layered_d_fusion",
                "fusion_name": policy["selected_policy"].get("fusion_name"),
                "score_space": tau_meta.get("score_space"),
                "tau": tau_route,
                "tau_raw": tau_meta.get("tau_raw"),
                "tau_percentile": tau_meta.get("tau_percentile"),
                "score": route_score,
                "raw_score": raw_score,
                "c_score": c_scores.get(sid),
                "d_score": d_scores.get(sid),
                "object_score": object_scores.get(sid) if object_policy else None,
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
                "family": "cvis_layered_d_object_fusion" if object_policy else "cvis_layered_d_fusion",
                "fusion_name": policy["selected_policy"].get("fusion_name"),
                "source": final_source,
            }
        )
    return route_rows, pred_rows, evaluation


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate/apply fixed C_vis + layered-D fusion controller.")
    ap.add_argument("--calibration_c_rows_csv", default="")
    ap.add_argument("--calibration_d_trajectory_long_csv", default="")
    ap.add_argument("--calibration_object_trajectory_long_csv", default="")
    ap.add_argument("--apply_c_rows_csv", default="")
    ap.add_argument("--apply_d_trajectory_long_csv", default="")
    ap.add_argument("--apply_object_trajectory_long_csv", default="")
    ap.add_argument("--policy_json", default="")
    ap.add_argument("--d_policy_json", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_filter", default="changed_answer", choices=["all", "changed_answer", "yes_to_no"])
    ap.add_argument("--c_feature", default="abl_black_rel_delta_orig_minus_blind__cheap_decision_margin_abs")
    ap.add_argument("--c_layer", default="", help="Optional layer_index for layer-wise C rows; use 'final' for max layer.")
    ap.add_argument("--d_layer_grid", default="quartiles", help="'quartiles', 'all', or comma-separated layer indices.")
    ap.add_argument("--object_feature", default="obj_target_gap_mean")
    ap.add_argument(
        "--object_layer_grid",
        default="late",
        help="'late' maps to {3L/4,7L/8,L}; also accepts 'all' or comma-separated layer indices.",
    )
    ap.add_argument("--min_selected_count", type=int, default=5)
    ap.add_argument("--score_space", default="raw", choices=["raw", "percentile", "discovery_percentile", "batch_percentile"])
    ap.add_argument("--fusion_modes", default="mean", help="Comma list of mean,min,max,alpha_grid.")
    ap.add_argument("--alpha_grid", default="0,0.25,0.5,0.75,1")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if str(args.policy_json or "").strip():
        policy = read_json(args.policy_json)
    else:
        if not str(args.calibration_c_rows_csv or "").strip() or not str(args.calibration_d_trajectory_long_csv or "").strip():
            raise RuntimeError("--calibration_c_rows_csv and --calibration_d_trajectory_long_csv are required without --policy_json.")
        c_rows = read_csv_rows(os.path.abspath(args.calibration_c_rows_csv))
        d_rows = read_d_rows(os.path.abspath(args.calibration_d_trajectory_long_csv))
        object_rows = (
            read_csv_rows(os.path.abspath(args.calibration_object_trajectory_long_csv))
            if str(args.calibration_object_trajectory_long_csv or "").strip()
            else []
        )
        policy = calibrate_fusion(
            c_rows=c_rows,
            d_rows=d_rows,
            object_rows=object_rows,
            c_feature=str(args.c_feature),
            c_layer=str(args.c_layer),
            d_policy_json=str(args.d_policy_json),
            d_layer_grid=str(args.d_layer_grid),
            object_feature=str(args.object_feature),
            object_layer_grid=str(args.object_layer_grid),
            candidate_filter=str(args.candidate_filter),
            min_selected_count=int(args.min_selected_count),
            score_space=str(args.score_space),
            fusion_modes=str(args.fusion_modes),
            alpha_grid=str(args.alpha_grid),
        )
        policy["inputs"] = {
            "calibration_c_rows_csv": os.path.abspath(args.calibration_c_rows_csv),
            "calibration_d_trajectory_long_csv": os.path.abspath(args.calibration_d_trajectory_long_csv),
            "calibration_object_trajectory_long_csv": os.path.abspath(args.calibration_object_trajectory_long_csv)
            if str(args.calibration_object_trajectory_long_csv or "").strip()
            else "",
            "d_policy_json": os.path.abspath(args.d_policy_json) if str(args.d_policy_json or "").strip() else "",
            "candidate_filter": str(args.candidate_filter),
            "c_feature": str(args.c_feature),
            "c_layer": str(args.c_layer),
            "d_layer_grid": str(args.d_layer_grid),
            "object_feature": str(args.object_feature),
            "object_layer_grid": str(args.object_layer_grid),
            "min_selected_count": int(args.min_selected_count),
            "score_space": str(args.score_space),
            "fusion_modes": str(args.fusion_modes),
            "alpha_grid": str(args.alpha_grid),
        }
        write_json(os.path.join(out_dir, "selected_policy.json"), policy)
        write_csv(
            os.path.join(out_dir, "fusion_grid_summary.csv"),
            [
                {
                    "fusion_name": row["fusion_name"],
                    "fusion": json.dumps(row["fusion"], sort_keys=True),
                    "tau": row["selected_policy"].get("tau"),
                    "tau_raw": row["selected_policy"].get("tau_raw"),
                    "tau_percentile": row["selected_policy"].get("tau_percentile"),
                    "score_space": row["selected_policy"].get("score_space"),
                    "calibration_score_count": row.get("calibration_score_count"),
                    "n_route_candidates": row["selected_policy"].get("n_route_candidates"),
                    "n_route_candidate_harm": row["selected_policy"].get("n_route_candidate_harm"),
                    "n_route_candidate_help": row["selected_policy"].get("n_route_candidate_help"),
                    "selected_count": row["selected_policy"].get("selected_count"),
                    "selected_harm": row["selected_policy"].get("selected_harm"),
                    "selected_help": row["selected_policy"].get("selected_help"),
                    "net": row["selected_policy"].get("net"),
                    "selected_harm_precision": row["selected_policy"].get("selected_harm_precision"),
                    "selected_harm_recall": row["selected_policy"].get("selected_harm_recall"),
                    "selected_harm_recall_in_scope": row["selected_policy"].get("selected_harm_recall_in_scope"),
                }
                for row in policy.get("fusion_candidates", [])
            ],
        )
        print("[saved]", os.path.join(out_dir, "selected_policy.json"))

    if str(args.apply_c_rows_csv or "").strip() and str(args.apply_d_trajectory_long_csv or "").strip():
        apply_c_rows = read_csv_rows(os.path.abspath(args.apply_c_rows_csv))
        apply_d_rows = read_d_rows(os.path.abspath(args.apply_d_trajectory_long_csv))
        if policy.get("object_policy") and not str(args.apply_object_trajectory_long_csv or "").strip():
            raise RuntimeError("--apply_object_trajectory_long_csv is required by this policy.")
        apply_object_rows = (
            read_csv_rows(os.path.abspath(args.apply_object_trajectory_long_csv))
            if str(args.apply_object_trajectory_long_csv or "").strip()
            else []
        )
        route_rows, pred_rows, evaluation = apply_fusion(
            c_rows=apply_c_rows,
            d_rows=apply_d_rows,
            object_rows=apply_object_rows,
            policy=policy,
            candidate_filter=str(policy.get("candidate_filter") or args.candidate_filter),
        )
        route_path = os.path.join(out_dir, "pcp_route_rows.csv")
        pred_path = os.path.join(out_dir, "pred_pcp_cd.jsonl")
        write_csv(route_path, route_rows)
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "mode": "apply_cvis_layered_d_fusion_controller",
            "inputs": {
                "apply_c_rows_csv": os.path.abspath(args.apply_c_rows_csv),
                "apply_d_trajectory_long_csv": os.path.abspath(args.apply_d_trajectory_long_csv),
                "apply_object_trajectory_long_csv": os.path.abspath(args.apply_object_trajectory_long_csv)
                if str(args.apply_object_trajectory_long_csv or "").strip()
                else "",
                "policy_json": os.path.abspath(args.policy_json) if str(args.policy_json or "").strip() else os.path.join(out_dir, "selected_policy.json"),
            },
            "policy": {
                "c_feature": policy.get("c_feature"),
                "c_metric": policy.get("c_metric"),
                "d_policy": policy.get("d_policy"),
                "object_policy": policy.get("object_policy"),
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
