#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TARGET_COL = "oracle_recall_gain_f1_nondecrease_ci_unique_noworse"


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


def flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def auc_high(values: Sequence[Optional[float]], labels: Sequence[bool]) -> Optional[Tuple[float, str, float, int, int]]:
    pairs = [(float(v), bool(label)) for v, label in zip(values, labels) if v is not None]
    pos = [value for value, label in pairs if label]
    neg = [value for value, label in pairs if not label]
    if not pos or not neg:
        return None
    score = 0.0
    total = 0
    for pval in pos:
        for nval in neg:
            if pval > nval:
                score += 1.0
            elif pval == nval:
                score += 0.5
            total += 1
    raw = score / float(total)
    return max(raw, 1.0 - raw), ("high" if raw >= 0.5 else "low"), raw, len(pos), len(neg)


def rankdata(values: Sequence[float]) -> List[float]:
    pairs = sorted((float(value), idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = rank
        i = j + 1
    return ranks


def corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    ma = sum(a) / float(len(a))
    mb = sum(b) / float(len(b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if not va or not vb:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def is_intervention_only_feature(source: str, feature: str) -> bool:
    key = feature.lower()
    if any(block in key for block in ("oracle", "audit", "target_col")):
        return False
    if any(block in key for block in ("base_", "baseonly", "base_only", "delta_", "shared", "jaccard", "similarity", "caption_pair")):
        return False
    if "int_only" in key or "intonly" in key:
        return False
    if source == "v74":
        return feature.startswith("capobj_int_") or feature.startswith("capcost_int_")
    return (
        "__sem_int_" in feature
        or "__sem_trace_int_" in feature
        or feature.startswith("v48_trace__probe_")
        or feature.startswith("v48b_trace__probe_")
        or feature
        in {
            "v60_self_inventory__inv_tok_int_caption_unit_count",
            "v60_self_inventory__inv_all_int_caption_unit_count",
        }
    )


def infer_source(path: str) -> str:
    name = os.path.basename(path).lower()
    parent = os.path.dirname(path).lower()
    if "v74" in parent or "object_divergence_cost" in parent:
        return "v74"
    if "nonleak" in parent or "joined_features" in name:
        return "nonleak"
    return "generic"


def build_labels(oracle_rows: Sequence[Dict[str, str]], target_col: str) -> Dict[str, Dict[str, Any]]:
    labels: Dict[str, Dict[str, Any]] = {}
    for row in oracle_rows:
        sid = safe_id(row)
        delta_f1 = safe_float(row.get("delta_f1_unique_base_minus_int")) or 0.0
        delta_recall = safe_float(row.get("delta_recall_base_minus_int")) or 0.0
        delta_ci = safe_float(row.get("delta_ci_unique_base_minus_int")) or 0.0
        delta_chair_s = safe_float(row.get("delta_chair_s_base_minus_int")) or 0.0
        gain = flag(row.get(target_col))
        harm = delta_f1 < -1e-12 or delta_ci > 1e-12 or delta_chair_s > 1e-12
        utility = delta_f1 + 0.25 * delta_recall - max(0.0, delta_ci) - 0.2 * max(0.0, delta_chair_s)
        labels[sid] = {
            "gain": gain,
            "harm": harm,
            "same": (not gain and not harm),
            "utility": utility,
            "delta_f1": delta_f1,
            "delta_recall": delta_recall,
            "delta_ci_unique": delta_ci,
            "delta_chair_s": delta_chair_s,
        }
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Screen intervention-only inference/caption features against fallback gain/same/harm outcomes."
    )
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--feature_csv", action="append", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_col", default=DEFAULT_TARGET_COL)
    ap.add_argument("--min_valid_count", type=int, default=400)
    args = ap.parse_args()

    labels = build_labels(read_csv_rows(args.oracle_rows_csv), args.target_col)
    features: Dict[str, Dict[str, float]] = defaultdict(dict)
    for path in args.feature_csv:
        source = infer_source(path)
        for row in read_csv_rows(path):
            sid = safe_id(row)
            if sid not in labels:
                continue
            for key, value in row.items():
                if key in {"id", "image_id", "question_id"}:
                    continue
                val = safe_float(value)
                if val is None or not is_intervention_only_feature(source, key):
                    continue
                features[sid][f"{source}__{key}"] = val

    ids = sorted(labels, key=lambda value: int(value) if value.isdigit() else value)
    all_features = sorted({key for sid in ids for key in features.get(sid, {})})
    feature_cols = [
        key
        for key in all_features
        if sum(key in features.get(sid, {}) for sid in ids) >= int(args.min_valid_count)
    ]

    table_rows: List[Dict[str, Any]] = []
    for sid in ids:
        row = {
            "id": sid,
            "fallback_gain": int(labels[sid]["gain"]),
            "fallback_harm": int(labels[sid]["harm"]),
            "fallback_same": int(labels[sid]["same"]),
            "fallback_utility": labels[sid]["utility"],
            "delta_f1": labels[sid]["delta_f1"],
            "delta_recall": labels[sid]["delta_recall"],
            "delta_ci_unique": labels[sid]["delta_ci_unique"],
            "delta_chair_s": labels[sid]["delta_chair_s"],
        }
        for col in feature_cols:
            row[col] = features.get(sid, {}).get(col, "")
        table_rows.append(row)

    metrics: List[Dict[str, Any]] = []
    for col in feature_cols:
        values = [safe_float(row.get(col)) for row in table_rows]
        for label_name in ("gain", "harm"):
            labs = [bool(labels[row["id"]][label_name]) for row in table_rows]
            res = auc_high(values, labs)
            if res is None:
                continue
            auc, direction, raw, n_pos, n_neg = res
            pos_values = [v for v, lab in zip(values, labs) if lab and v is not None]
            neg_values = [v for v, lab in zip(values, labs) if (not lab) and v is not None]
            metric = {
                "comparison": f"{label_name}_vs_rest",
                "feature": col,
                "direction": direction,
                "score": auc,
                "auc_high": raw,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "pos_mean": sum(pos_values) / float(len(pos_values)),
                "neg_mean": sum(neg_values) / float(len(neg_values)),
            }
            for group in ("gain", "same", "harm"):
                group_values = [
                    safe_float(row.get(col))
                    for row in table_rows
                    if bool(labels[row["id"]][group]) and safe_float(row.get(col)) is not None
                ]
                metric[f"{group}_mean"] = sum(group_values) / float(len(group_values)) if group_values else ""
                metric[f"{group}_n"] = len(group_values)
            metrics.append(metric)

        extreme = [
            (value, bool(labels[row["id"]]["gain"]))
            for value, row in zip(values, table_rows)
            if value is not None and (labels[row["id"]]["gain"] or labels[row["id"]]["harm"])
        ]
        res = auc_high([v for v, _ in extreme], [lab for _, lab in extreme])
        if res is not None:
            auc, direction, raw, n_pos, n_neg = res
            metrics.append(
                {
                    "comparison": "gain_vs_harm",
                    "feature": col,
                    "direction": direction,
                    "score": auc,
                    "auc_high": raw,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
            )

        utility_pairs = [
            (value, float(labels[row["id"]]["utility"]))
            for value, row in zip(values, table_rows)
            if value is not None
        ]
        if len(utility_pairs) >= 4:
            spearman = corr(
                rankdata([value for value, _ in utility_pairs]),
                rankdata([utility for _, utility in utility_pairs]),
            )
            metrics.append(
                {
                    "comparison": "utility_spearman",
                    "feature": col,
                    "direction": "high" if spearman >= 0.0 else "low",
                    "score": abs(spearman),
                    "auc_high": spearman,
                    "n_pos": "",
                    "n_neg": "",
                }
            )

    metrics = sorted(metrics, key=lambda row: float(row["score"]), reverse=True)

    out_dir = os.path.abspath(args.out_dir)
    features_csv = os.path.join(out_dir, "intervention_only_features.csv")
    metrics_csv = os.path.join(out_dir, "intervention_only_feature_metrics.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(features_csv, table_rows)
    write_csv(metrics_csv, metrics)
    write_json(
        summary_json,
        {
            "counts": {
                "n_rows": len(table_rows),
                "n_features": len(feature_cols),
                "n_gain": sum(int(row["fallback_gain"]) for row in table_rows),
                "n_harm": sum(int(row["fallback_harm"]) for row in table_rows),
                "n_same": sum(int(row["fallback_same"]) for row in table_rows),
            },
            "inputs": {
                "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
                "feature_csv": [os.path.abspath(path) for path in args.feature_csv],
                "target_col": args.target_col,
                "min_valid_count": args.min_valid_count,
            },
            "outputs": {
                "features_csv": features_csv,
                "metrics_csv": metrics_csv,
                "summary_json": summary_json,
            },
            "top_gain": [row for row in metrics if row["comparison"] == "gain_vs_rest"][:20],
            "top_harm": [row for row in metrics if row["comparison"] == "harm_vs_rest"][:20],
            "top_gain_vs_harm": [row for row in metrics if row["comparison"] == "gain_vs_harm"][:20],
            "top_utility_spearman": [row for row in metrics if row["comparison"] == "utility_spearman"][:20],
        },
    )
    print(f"[saved] {features_csv}")
    print(f"[saved] {metrics_csv}")
    print(f"[saved] {summary_json}")


if __name__ == "__main__":
    main()
