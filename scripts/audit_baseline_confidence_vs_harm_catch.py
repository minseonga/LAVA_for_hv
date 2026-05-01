#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_FEATURES = (
    "candidate_prob_binary",
    "candidate_minus_alt",
    "candidate_label_lp",
    "margin_abs",
    "yes_prob_binary",
    "no_prob_binary",
    "yes_minus_no",
)


def safe_str(value: Any) -> str:
    return str("" if value is None else value).strip()


def maybe_float(value: Any) -> Optional[float]:
    s = safe_str(value)
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def row_id(row: Mapping[str, Any]) -> str:
    return safe_str(row.get("id") or row.get("question_id"))


def maybe_int(value: Any) -> Optional[int]:
    x = maybe_float(value)
    return None if x is None else int(round(x))


def parse_layers(spec: str, available: Sequence[int]) -> List[int]:
    vals = sorted(set(int(x) for x in available))
    raw = safe_str(spec).lower()
    if raw in {"", "all"}:
        return vals
    if raw in {"final", "last"}:
        return [max(vals)] if vals else []
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token in {"final", "last"}:
            out.append(max(vals))
        else:
            out.append(int(token))
    missing = [x for x in out if x not in vals]
    if missing:
        raise RuntimeError(f"Requested layers unavailable: {missing}; available={vals}")
    return sorted(set(out))


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def stats(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
        }
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": mu,
        "std": sd,
        "q10": quantile(vals, 0.10),
        "q25": quantile(vals, 0.25),
        "q50": quantile(vals, 0.50),
        "q75": quantile(vals, 0.75),
        "q90": quantile(vals, 0.90),
    }


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(scores) != len(labels):
        return None
    pos = [float(x) for x, y in zip(scores, labels) if int(y) == 1]
    neg = [float(x) for x, y in zip(scores, labels) if int(y) == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(len(pos) * len(neg))


def load_route_rows(path: str) -> Dict[str, Dict[str, Any]]:
    if not safe_str(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv_rows(path):
        sid = row_id(row)
        if sid:
            out[sid] = dict(row)
    return out


def catch_bucket(row: Mapping[str, Any]) -> str:
    outcome = safe_str(row.get("outcome"))
    route = safe_str(row.get("route")).lower()
    selected = route == "baseline"
    if outcome == "harm":
        return "caught_harm" if selected else "missed_harm"
    if outcome == "help":
        return "selected_help" if selected else "preserved_help"
    if outcome == "neutral":
        return "selected_neutral" if selected else "unselected_neutral"
    return "missing"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare baseline replay confidence between caught and missed harm cases."
    )
    ap.add_argument("--changed_rows_csv", required=True, help="changed_rows.csv from analyze_pope_harm_help_segments.py")
    ap.add_argument("--baseline_trajectory_long_csv", required=True)
    ap.add_argument("--route_rows_csv", default="", help="Optional pcp_route_rows.csv. Overrides route in changed_rows_csv.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", default="all", help="'all', 'final', or comma-separated layer indices.")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    args = ap.parse_args()

    changed = {row_id(row): dict(row) for row in read_csv_rows(args.changed_rows_csv) if row_id(row)}
    routes = load_route_rows(args.route_rows_csv)
    traj_rows = read_csv_rows(args.baseline_trajectory_long_csv)
    available_layers = sorted(
        set(int(x) for row in traj_rows if (x := maybe_int(row.get("layer_index"))) is not None)
    )
    layers = parse_layers(str(args.layers), available_layers)
    features = [x.strip() for x in str(args.features).split(",") if x.strip()]

    audit_rows: List[Dict[str, Any]] = []
    for row in traj_rows:
        sid = row_id(row)
        layer = maybe_int(row.get("layer_index"))
        if not sid or sid not in changed or layer is None or int(layer) not in layers:
            continue
        base = dict(changed[sid])
        route = routes.get(sid)
        if route:
            base["route"] = safe_str(route.get("route")).lower()
            for col in ("score", "raw_score", "c_score", "d_score", "object_score", "fusion_score"):
                if col in route:
                    base[f"route_{col}"] = route.get(col)
        base["catch_bucket"] = catch_bucket(base)
        base["baseline_conf_layer"] = int(layer)
        for feature in features:
            if feature in row:
                base[f"baseline_{feature}"] = row.get(feature)
        audit_rows.append(base)

    summary_rows: List[Dict[str, Any]] = []
    for layer in layers:
        layer_rows = [row for row in audit_rows if int(row.get("baseline_conf_layer", -1)) == int(layer)]
        for feature in features:
            col = f"baseline_{feature}"
            if not any(maybe_float(row.get(col)) is not None for row in layer_rows):
                continue
            groups: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
            for row in layer_rows:
                value = maybe_float(row.get(col))
                if value is None:
                    continue
                key = (
                    safe_str(row.get("catch_bucket")),
                    safe_str(row.get("error_type")),
                    safe_str(row.get("category")),
                )
                groups[key].append(float(value))
            for (bucket, error_type, category), vals in groups.items():
                payload = stats(vals)
                summary_rows.append(
                    {
                        "layer": int(layer),
                        "feature": feature,
                        "catch_bucket": bucket,
                        "error_type": error_type,
                        "category": category,
                        **payload,
                    }
                )

    auc_rows: List[Dict[str, Any]] = []
    for layer in layers:
        layer_rows = [
            row
            for row in audit_rows
            if int(row.get("baseline_conf_layer", -1)) == int(layer)
            and safe_str(row.get("outcome")) == "harm"
            and safe_str(row.get("catch_bucket")) in {"caught_harm", "missed_harm"}
        ]
        for feature in features:
            col = f"baseline_{feature}"
            xs: List[float] = []
            ys: List[int] = []
            for row in layer_rows:
                value = maybe_float(row.get(col))
                if value is None:
                    continue
                xs.append(float(value))
                ys.append(1 if safe_str(row.get("catch_bucket")) == "caught_harm" else 0)
            auc_high = binary_auroc(xs, ys)
            if auc_high is None:
                continue
            auc_low = binary_auroc([-x for x in xs], ys) or 0.0
            auc_rows.append(
                {
                    "layer": int(layer),
                    "feature": feature,
                    "n_harm": len(xs),
                    "n_caught": sum(ys),
                    "n_missed": len(ys) - sum(ys),
                    "caught_vs_missed_auroc": max(float(auc_high), float(auc_low)),
                    "direction_for_caught": "high" if auc_high >= auc_low else "low",
                    "raw_auroc_high": float(auc_high),
                }
            )

    summary = {
        "inputs": {
            "changed_rows_csv": os.path.abspath(args.changed_rows_csv),
            "baseline_trajectory_long_csv": os.path.abspath(args.baseline_trajectory_long_csv),
            "route_rows_csv": os.path.abspath(args.route_rows_csv) if safe_str(args.route_rows_csv) else "",
            "layers": layers,
            "features": features,
        },
        "counts": {
            "n_changed_ids": len(changed),
            "n_route_ids": len(routes),
            "n_audit_rows": len(audit_rows),
            "catch_bucket_counts": dict(Counter(row.get("catch_bucket") for row in audit_rows)),
        },
    }

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "baseline_confidence_audit_rows.csv"), audit_rows)
    write_csv(os.path.join(out_dir, "baseline_confidence_bucket_summary.csv"), summary_rows)
    write_csv(os.path.join(out_dir, "caught_vs_missed_harm_auc.csv"), auc_rows)
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print("[saved]", os.path.join(out_dir, "summary.json"))
    print("[saved]", os.path.join(out_dir, "baseline_confidence_bucket_summary.csv"))
    print("[saved]", os.path.join(out_dir, "caught_vs_missed_harm_auc.csv"))


if __name__ == "__main__":
    main()
