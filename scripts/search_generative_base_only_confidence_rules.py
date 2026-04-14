#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("id") or row.get("image_id") or row.get("question_id") or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def target_flag(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def load_joined(feature_csv: str, oracle_rows_csv: str) -> List[Dict[str, str]]:
    features = {safe_id(row): row for row in read_csv_rows(feature_csv)}
    rows: List[Dict[str, str]] = []
    for oracle in read_csv_rows(oracle_rows_csv):
        sid = safe_id(oracle)
        if not sid or sid not in features:
            continue
        merged = dict(oracle)
        for key, value in features[sid].items():
            if key in {"image_id", "question_id"}:
                continue
            merged[key] = value
        merged["id"] = sid
        rows.append(merged)
    return rows


def aggregate_route(rows: Sequence[Dict[str, str]], selected_ids: Set[str], target_col: str) -> Dict[str, Any]:
    n = len(rows)
    chair_s = 0
    gen_u = 0
    hall_u = 0
    supp = 0
    gt = 0
    gen_i = 0
    hall_i = 0
    n_target = 0
    for row in rows:
        route = "base" if row["id"] in selected_ids else "int"
        chair_s += int(float(row[f"{route}_chair_s"]))
        gen_u += int(float(row[f"{route}_n_generated_unique"]))
        hall_u += int(float(row[f"{route}_n_hallucinated_unique"]))
        supp += int(float(row[f"{route}_n_supported_unique"]))
        gt += int(float(row[f"{route}_n_gt_objects"]))
        gen_i += int(float(row[f"{route}_n_generated_inst"]))
        hall_i += int(float(row[f"{route}_n_hallucinated_inst"]))
        if row["id"] in selected_ids:
            n_target += target_flag(row.get(target_col))
    precision = safe_div(float(supp), float(gen_u))
    recall = safe_div(float(supp), float(gt))
    return {
        "n_selected": len(selected_ids),
        "n_target": int(n_target),
        "target_precision": safe_div(float(n_target), float(len(selected_ids))),
        "target_recall": safe_div(float(n_target), float(sum(target_flag(row.get(target_col)) for row in rows))),
        "chair_s": safe_div(float(chair_s), float(n)),
        "ci_unique": safe_div(float(hall_u), float(gen_u)),
        "ci_inst": safe_div(float(hall_i), float(gen_i)),
        "precision_unique": precision,
        "recall": recall,
        "f1_unique": f1(precision, recall),
    }


def score_candidate(candidate: Dict[str, Any], base: Dict[str, Any]) -> float:
    return (
        float(candidate["delta_f1_unique"])
        + 0.5 * float(candidate["delta_recall"])
        - 0.5 * max(0.0, float(candidate["delta_ci_unique"]))
        - 0.1 * max(0.0, float(candidate["delta_chair_s"]))
    )


def add_deltas(candidate: Dict[str, Any], base: Dict[str, Any]) -> None:
    for key in ("f1_unique", "recall", "ci_unique", "ci_inst", "chair_s", "precision_unique"):
        candidate[f"delta_{key}"] = float(candidate[key]) - float(base[key])
    candidate["score"] = score_candidate(candidate, base)


def load_metric_features(feature_metrics_csv: str, max_features: int, prefixes: Sequence[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in read_csv_rows(feature_metrics_csv):
        feature = str(row.get("feature", ""))
        if prefixes and not any(feature.startswith(prefix) for prefix in prefixes):
            continue
        out.append(row)
        if len(out) >= int(max_features):
            break
    return out


def select_topk(
    rows: Sequence[Dict[str, str]],
    feature: str,
    direction: str,
    k: int,
) -> Set[str]:
    ranked: List[Tuple[str, float]] = []
    for row in rows:
        val = safe_float(row.get(feature))
        if val is not None:
            ranked.append((row["id"], val))
    ranked.sort(key=lambda item: item[1], reverse=(direction == "high"))
    return {sid for sid, _ in ranked[: int(k)]}


def threshold_for(values: Sequence[float], q: float, direction: str) -> float:
    vals = sorted(float(v) for v in values)
    if direction == "high":
        idx = int(round(float(q) * (len(vals) - 1)))
    else:
        idx = int(round((1.0 - float(q)) * (len(vals) - 1)))
    return vals[min(len(vals) - 1, max(0, idx))]


def select_threshold(rows: Sequence[Dict[str, str]], specs: Sequence[Tuple[str, str, float]]) -> Set[str]:
    selected: Set[str] = set()
    for row in rows:
        ok = True
        for feature, direction, tau in specs:
            val = safe_float(row.get(feature))
            if val is None:
                ok = False
                break
            if direction == "high" and val < tau:
                ok = False
                break
            if direction == "low" and val > tau:
                ok = False
                break
        if ok:
            selected.add(row["id"])
    return selected


def evaluate_rule(
    rows: Sequence[Dict[str, str]],
    selected_ids: Set[str],
    target_col: str,
    base: Dict[str, Any],
    rule: str,
) -> Dict[str, Any]:
    out = aggregate_route(rows, selected_ids, target_col)
    out["rule"] = rule
    out["selected_ids"] = "|".join(sorted(selected_ids, key=lambda value: int(value) if value.isdigit() else value))
    add_deltas(out, base)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Search conservative rules over v61 base-only confidence features.")
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--feature_metrics_csv", required=True)
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_features", type=int, default=20)
    ap.add_argument("--topk", default="3,5,8,10,12,15,20,25,30,40,50,60,75,100")
    ap.add_argument("--threshold_quantiles", default="0.80,0.85,0.90,0.92,0.95,0.97")
    ap.add_argument("--prefix", action="append", default=["bo_conf", "bo_"])
    ap.add_argument("--min_selected", type=int, default=3)
    ap.add_argument("--max_selected", type=int, default=100)
    ap.add_argument("--max_delta_ci_unique", type=float, default=0.005)
    ap.add_argument("--max_delta_chair_s", type=float, default=0.02)
    ap.add_argument("--min_delta_f1", type=float, default=0.0)
    ap.add_argument("--min_delta_recall", type=float, default=0.0)
    args = ap.parse_args()

    rows = load_joined(args.feature_csv, args.oracle_rows_csv)
    base = aggregate_route(rows, set(), args.target_col)
    metric_features = load_metric_features(args.feature_metrics_csv, int(args.max_features), args.prefix)
    topks = [int(x) for x in str(args.topk).split(",") if x.strip()]
    qs = [float(x) for x in str(args.threshold_quantiles).split(",") if x.strip()]

    candidates: List[Dict[str, Any]] = []
    for metric in metric_features:
        feature = str(metric["feature"])
        direction = str(metric["direction"])
        for k in topks:
            selected = select_topk(rows, feature, direction, k)
            if int(args.min_selected) <= len(selected) <= int(args.max_selected):
                candidates.append(
                    evaluate_rule(rows, selected, args.target_col, base, f"top{k}:{feature}:{direction}")
                )

    for left, right in itertools.combinations(metric_features[: min(10, len(metric_features))], 2):
        f1_name, d1 = str(left["feature"]), str(left["direction"])
        f2_name, d2 = str(right["feature"]), str(right["direction"])
        v1 = [safe_float(row.get(f1_name)) for row in rows]
        v2 = [safe_float(row.get(f2_name)) for row in rows]
        v1 = [x for x in v1 if x is not None]
        v2 = [x for x in v2 if x is not None]
        if len(v1) < max(10, int(0.8 * len(rows))) or len(v2) < max(10, int(0.8 * len(rows))):
            continue
        for q in qs:
            t1 = threshold_for(v1, q, d1)
            t2 = threshold_for(v2, q, d2)
            selected = select_threshold(rows, [(f1_name, d1, t1), (f2_name, d2, t2)])
            if int(args.min_selected) <= len(selected) <= int(args.max_selected):
                candidates.append(
                    evaluate_rule(
                        rows,
                        selected,
                        args.target_col,
                        base,
                        f"{f1_name} {d1} {t1:.6g} & {f2_name} {d2} {t2:.6g}",
                    )
                )

    valid = [
        row
        for row in candidates
        if int(row["n_selected"]) >= int(args.min_selected)
        and float(row["delta_f1_unique"]) >= float(args.min_delta_f1)
        and float(row["delta_recall"]) >= float(args.min_delta_recall)
        and float(row["delta_ci_unique"]) <= float(args.max_delta_ci_unique)
        and float(row["delta_chair_s"]) <= float(args.max_delta_chair_s)
    ]
    candidates.sort(key=lambda row: (float(row["score"]), float(row["delta_f1_unique"])), reverse=True)
    valid.sort(key=lambda row: (float(row["score"]), float(row["delta_f1_unique"])), reverse=True)

    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    candidates_csv = os.path.join(args.out_dir, "rule_candidates.csv")
    valid_csv = os.path.join(args.out_dir, "rule_candidates_valid.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_csv(candidates_csv, candidates)
    write_csv(valid_csv, valid)
    write_json(
        summary_json,
        {
            "inputs": {
                "feature_csv": os.path.abspath(args.feature_csv),
                "feature_metrics_csv": os.path.abspath(args.feature_metrics_csv),
                "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
                "target_col": args.target_col,
                "max_features": int(args.max_features),
                "prefix": list(args.prefix),
            },
            "baseline_intervention": base,
            "counts": {
                "n_rows": len(rows),
                "n_target": sum(target_flag(row.get(args.target_col)) for row in rows),
                "n_candidates": len(candidates),
                "n_valid": len(valid),
            },
            "best": valid[0] if valid else None,
            "best_any": candidates[0] if candidates else None,
            "outputs": {
                "rule_candidates_csv": os.path.abspath(candidates_csv),
                "rule_candidates_valid_csv": os.path.abspath(valid_csv),
                "summary_json": os.path.abspath(summary_json),
            },
        },
    )
    print("[saved]", candidates_csv)
    print("[saved]", valid_csv)
    print("[saved]", summary_json)
    if valid:
        print("[best]", json.dumps(valid[0], ensure_ascii=False, sort_keys=True))
    elif candidates:
        print("[best_any]", json.dumps(candidates[0], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
