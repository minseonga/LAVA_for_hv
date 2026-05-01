#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


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
    v = maybe_float(value)
    if v is None:
        return None
    return int(round(v))


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


def layer_score(row: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]) -> Optional[float]:
    vals: List[float] = []
    for metric in metrics:
        z = oriented_z(row, metric)
        if z is None:
            return None
        vals.append(float(z))
    return mean(vals)


def threshold_grid(values: Sequence[float]) -> List[float]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return [0.0]
    return sorted(set(finite))


def evaluate_scores(
    rows: Sequence[Mapping[str, Any]],
    scores_by_id: Mapping[str, float],
    *,
    candidate_filter: str,
    min_selected_count: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidate_scores = [
        float(scores_by_id[str(row.get("id", ""))])
        for row in rows
        if str(row.get("id", "")) in scores_by_id and is_candidate(row, candidate_filter)
    ]
    best: Optional[Dict[str, Any]] = None
    sweep: List[Dict[str, Any]] = []
    for tau in threshold_grid(candidate_scores):
        n_eval = 0
        total_harm = 0
        total_help = 0
        selected = 0
        selected_harm = 0
        selected_help = 0
        for row in rows:
            sid = str(row.get("id", ""))
            if sid not in scores_by_id:
                continue
            harm = int(maybe_int(row.get("harm")) or 0)
            help_ = int(maybe_int(row.get("help")) or 0)
            n_eval += 1
            total_harm += harm
            total_help += help_
            if is_candidate(row, candidate_filter) and float(scores_by_id[sid]) >= float(tau):
                selected += 1
                selected_harm += harm
                selected_help += help_
        result = {
            "tau": float(tau),
            "n_eval": int(n_eval),
            "selected_count": int(selected),
            "total_harm": int(total_harm),
            "total_help": int(total_help),
            "selected_harm": int(selected_harm),
            "selected_help": int(selected_help),
            "net": int(selected_harm - selected_help),
            "delta_vs_intervention_changed": float((selected_harm - selected_help) / max(1, total_harm + total_help)),
            "selected_harm_precision": float(selected_harm / max(1, selected)),
            "selected_harm_recall": float(selected_harm / max(1, total_harm)),
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
    return best or sweep[0], sweep


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze fixed 3-feature D-family scores from layer trajectory rows.")
    ap.add_argument("--trajectory_long_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_filter", default="changed_answer", choices=["all", "changed_answer", "yes_to_no"])
    ap.add_argument("--min_selected_count", type=int, default=5)
    args = ap.parse_args()

    rows = read_rows(os.path.abspath(args.trajectory_long_csv))
    by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer = maybe_int(row.get("layer_index"))
        if layer is not None:
            by_layer[int(layer)].append(row)

    per_layer_rows: List[Dict[str, Any]] = []
    for layer in sorted(by_layer):
        layer_rows = by_layer[layer]
        fit_rows = [row for row in layer_rows if is_candidate(row, str(args.candidate_filter))]
        metrics = [m for feat in D_FEATURES if (m := orient_feature(fit_rows, feat)) is not None]
        scores: Dict[str, float] = {}
        for row in layer_rows:
            score = layer_score(row, metrics)
            if score is not None:
                scores[str(row.get("id", ""))] = score
        best, _ = evaluate_scores(
            layer_rows,
            scores,
            candidate_filter=str(args.candidate_filter),
            min_selected_count=int(args.min_selected_count),
        )
        per_layer_rows.append(
            {
                "layer": int(layer),
                "n_features": len(metrics),
                "feature_aurocs": ";".join(f"{m['feature']}={float(m['auroc']):.6f}:{m['direction']}" for m in metrics),
                **best,
            }
        )

    out_dir = os.path.abspath(args.out_dir)
    write_csv(os.path.join(out_dir, "per_layer_d_family3_summary.csv"), per_layer_rows)
    best_net = max(per_layer_rows, key=lambda r: (int(r["net"]), int(r["selected_harm"]), -int(r["selected_help"])))
    best_precision = max(
        per_layer_rows,
        key=lambda r: (float(r["selected_harm_precision"]), int(r["net"]), int(r["selected_harm"])),
    )
    summary = {
        "inputs": {
            "trajectory_long_csv": os.path.abspath(args.trajectory_long_csv),
            "candidate_filter": str(args.candidate_filter),
            "d_features": list(D_FEATURES),
        },
        "best_by_net": best_net,
        "best_by_precision": best_precision,
        "outputs": {
            "per_layer_csv": os.path.join(out_dir, "per_layer_d_family3_summary.csv"),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print("[saved]", os.path.join(out_dir, "per_layer_d_family3_summary.csv"))
    print(json.dumps(summary["best_by_net"], indent=2))


if __name__ == "__main__":
    main()
