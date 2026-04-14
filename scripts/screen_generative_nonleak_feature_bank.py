#!/usr/bin/env python3
"""Screen non-CHAIR/GT generative features against a diagnostic oracle target.

The oracle labels may be built from CHAIR/GT, but this script only consumes
feature CSVs that are supplied by the caller. It keeps numeric feature columns
and drops obvious labels, identifiers, and text columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ID_KEYS = ("id", "image_id", "question_id")

DROP_EXACT = {
    "id",
    "image",
    "image_id",
    "question_id",
}

DROP_PREFIXES = (
    "oracle_",
    "target_",
    "net_",
)

DROP_SUBSTRINGS = (
    "caption",
    "decoded_text",
    "text",
    "words",
    "tokens",
    "units",
    "selected_ids",
    "policy",
)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_id(row: Dict[str, str]) -> str:
    for key in ID_KEYS:
        val = row.get(key, "")
        if val not in ("", None):
            text = str(val).strip()
            if text.endswith(".0") and text[:-2].isdigit():
                text = text[:-2]
            return text
    image = row.get("image", "")
    if image:
        m = re.search(r"(\d+)", os.path.basename(image))
        if m:
            return str(int(m.group(1)))
    raise KeyError(f"row has no id-like key: {row.keys()}")


def as_float(val: str) -> Optional[float]:
    if val is None:
        return None
    text = str(val).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def should_drop_col(name: str) -> bool:
    low = name.lower()
    if low in DROP_EXACT:
        return True
    if any(low.startswith(prefix) for prefix in DROP_PREFIXES):
        return True
    if any(part in low for part in DROP_SUBSTRINGS):
        # Keep numeric counts/rates that happen to include "token" or "unit".
        allow = (
            low.endswith("_count")
            or low.endswith("_rate")
            or low.endswith("_mean")
            or low.endswith("_min")
            or low.endswith("_max")
            or low.endswith("_std")
            or low.endswith("_q10")
            or low.endswith("_q90")
            or low.endswith("_frac")
            or low.endswith("_jaccard")
            or "_count_" in low
            or "_rate_" in low
            or "unit_count" in low
            or "token_count" in low
        )
        if not allow:
            return True
    return False


def rank_auc(labels: Sequence[int], values: Sequence[float]) -> Optional[float]:
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(labels) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: Sequence[int], values: Sequence[float]) -> Optional[float]:
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    order = sorted(range(len(labels)), key=lambda i: values[i], reverse=True)
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            hits += 1
            total += hits / rank
    return total / n_pos


def precision_at(labels: Sequence[int], values: Sequence[float], k: int) -> float:
    if not labels or k <= 0:
        return 0.0
    k = min(k, len(labels))
    order = sorted(range(len(labels)), key=lambda i: values[i], reverse=True)[:k]
    return sum(labels[i] for i in order) / k


def mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def stdev(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def evaluate_feature(
    name: str,
    labels: Sequence[int],
    values: Sequence[float],
) -> Optional[Dict[str, object]]:
    auc_high = rank_auc(labels, values)
    if auc_high is None:
        return None
    direction = "high" if auc_high >= 0.5 else "low"
    oriented = list(values) if direction == "high" else [-x for x in values]
    auc = max(auc_high, 1.0 - auc_high)
    ap = average_precision(labels, oriented)
    pos_vals = [v for y, v in zip(labels, values) if y == 1]
    neg_vals = [v for y, v in zip(labels, values) if y == 0]
    return {
        "feature": name,
        "direction": direction,
        "auroc": auc,
        "auroc_high": auc_high,
        "ap": ap if ap is not None else 0.0,
        "p_at_5": precision_at(labels, oriented, 5),
        "p_at_10": precision_at(labels, oriented, 10),
        "p_at_25": precision_at(labels, oriented, 25),
        "p_at_40": precision_at(labels, oriented, 40),
        "p_at_75": precision_at(labels, oriented, 75),
        "n": len(labels),
        "n_pos": sum(labels),
        "pos_mean": mean(pos_vals),
        "neg_mean": mean(neg_vals),
        "pos_std": stdev(pos_vals),
        "neg_std": stdev(neg_vals),
    }


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def parse_feature_spec(spec: str) -> Tuple[str, str]:
    if "::" not in spec:
        path = spec
        stem = os.path.splitext(os.path.basename(path))[0]
        return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_"), path
    name, path = spec.split("::", 1)
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_"), path


def load_feature_bank(
    specs: Sequence[str],
    ids: Sequence[str],
    min_valid_frac: float,
) -> Tuple[Dict[str, List[Optional[float]]], Dict[str, object]]:
    id_set = set(ids)
    features: Dict[str, List[Optional[float]]] = {}
    stats = {"sources": []}

    for spec in specs:
        source, path = parse_feature_spec(spec)
        rows = read_csv(path)
        by_id = {safe_id(row): row for row in rows if safe_id(row) in id_set}
        fieldnames = list(rows[0].keys()) if rows else []
        kept = 0
        dropped = 0
        for col in fieldnames:
            if should_drop_col(col):
                dropped += 1
                continue
            vals: List[Optional[float]] = []
            n_valid = 0
            for sample_id in ids:
                row = by_id.get(sample_id)
                val = as_float(row.get(col, "")) if row else None
                vals.append(val)
                if val is not None:
                    n_valid += 1
            if n_valid < max(1, int(len(ids) * min_valid_frac)):
                dropped += 1
                continue
            # Impute missing numeric values with the column mean.
            valid_vals = [v for v in vals if v is not None]
            fill = mean(valid_vals)
            out_vals = [fill if v is None else v for v in vals]
            out_name = f"{source}__{col}"
            if out_name in features:
                suffix = 2
                while f"{out_name}_{suffix}" in features:
                    suffix += 1
                out_name = f"{out_name}_{suffix}"
            features[out_name] = out_vals
            kept += 1
        stats["sources"].append(
            {
                "source": source,
                "path": path,
                "rows": len(rows),
                "matched_rows": len(by_id),
                "kept_numeric_features": kept,
                "dropped_columns": dropped,
            }
        )

    return features, stats


def zscore(vals: Sequence[float]) -> List[float]:
    m = mean(vals)
    s = stdev(vals)
    if s == 0:
        return [0.0 for _ in vals]
    return [(v - m) / s for v in vals]


def build_combo_metrics(
    labels: Sequence[int],
    features: Dict[str, List[Optional[float]]],
    single_rows: Sequence[Dict[str, object]],
    max_combo_features: int,
) -> List[Dict[str, object]]:
    top = list(single_rows)[:max_combo_features]
    oriented_z: Dict[str, List[float]] = {}
    for row in top:
        name = str(row["feature"])
        vals = [float(v) for v in features[name]]
        if row["direction"] == "low":
            vals = [-v for v in vals]
        oriented_z[name] = zscore(vals)

    combos: List[Dict[str, object]] = []
    for a, b in combinations([str(r["feature"]) for r in top], 2):
        va = oriented_z[a]
        vb = oriented_z[b]
        sum_vals = [x + y for x, y in zip(va, vb)]
        min_vals = [min(x, y) for x, y in zip(va, vb)]
        max_vals = [max(x, y) for x, y in zip(va, vb)]
        for kind, vals in (
            ("sum_z", sum_vals),
            ("min_z", min_vals),
            ("max_z", max_vals),
        ):
            metric = evaluate_feature(f"{kind}::{a}::{b}", labels, vals)
            if metric is not None:
                metric["combo_kind"] = kind
                metric["feature_a"] = a
                metric["feature_b"] = b
                combos.append(metric)
    combos.sort(key=lambda r: (float(r["auroc"]), float(r["ap"])), reverse=True)
    return combos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_rows_csv", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--feature_csv", action="append", required=True, help="NAME::PATH")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_valid_frac", type=float, default=0.95)
    parser.add_argument("--max_combo_features", type=int, default=60)
    parser.add_argument("--write_joined", action="store_true")
    args = parser.parse_args()

    oracle_rows = read_csv(args.oracle_rows_csv)
    ids = [safe_id(row) for row in oracle_rows]
    labels = []
    for row in oracle_rows:
        val = as_float(row.get(args.target_col, ""))
        labels.append(1 if val and val > 0 else 0)

    features, load_stats = load_feature_bank(args.feature_csv, ids, args.min_valid_frac)
    single_rows: List[Dict[str, object]] = []
    for name, vals in features.items():
        metric = evaluate_feature(name, labels, [float(v) for v in vals])
        if metric is not None:
            single_rows.append(metric)
    single_rows.sort(key=lambda r: (float(r["auroc"]), float(r["ap"])), reverse=True)

    combo_rows = build_combo_metrics(labels, features, single_rows, args.max_combo_features)

    single_fields = [
        "feature",
        "direction",
        "auroc",
        "auroc_high",
        "ap",
        "p_at_5",
        "p_at_10",
        "p_at_25",
        "p_at_40",
        "p_at_75",
        "n",
        "n_pos",
        "pos_mean",
        "neg_mean",
        "pos_std",
        "neg_std",
    ]
    combo_fields = [
        "feature",
        "combo_kind",
        "feature_a",
        "feature_b",
        "direction",
        "auroc",
        "auroc_high",
        "ap",
        "p_at_5",
        "p_at_10",
        "p_at_25",
        "p_at_40",
        "p_at_75",
        "n",
        "n_pos",
        "pos_mean",
        "neg_mean",
        "pos_std",
        "neg_std",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    single_csv = os.path.join(args.out_dir, "single_feature_metrics.csv")
    combo_csv = os.path.join(args.out_dir, "combo_feature_metrics.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_csv(single_csv, single_rows, single_fields)
    write_csv(combo_csv, combo_rows, combo_fields)

    joined_csv = ""
    if args.write_joined:
        joined_csv = os.path.join(args.out_dir, "joined_features.csv")
        fields = ["id", args.target_col] + list(features.keys())
        joined_rows = []
        for i, sample_id in enumerate(ids):
            row = {"id": sample_id, args.target_col: labels[i]}
            for name, vals in features.items():
                row[name] = vals[i]
            joined_rows.append(row)
        write_csv(joined_csv, joined_rows, fields)

    summary = {
        "inputs": {
            "oracle_rows_csv": args.oracle_rows_csv,
            "target_col": args.target_col,
            "feature_csv": args.feature_csv,
            "min_valid_frac": args.min_valid_frac,
            "max_combo_features": args.max_combo_features,
        },
        "counts": {
            "n_rows": len(labels),
            "n_pos": sum(labels),
            "n_features": len(features),
            "n_single_metrics": len(single_rows),
            "n_combo_metrics": len(combo_rows),
        },
        "load_stats": load_stats,
        "top_single": single_rows[:20],
        "top_combo": combo_rows[:20],
        "outputs": {
            "single_feature_metrics_csv": single_csv,
            "combo_feature_metrics_csv": combo_csv,
            "joined_features_csv": joined_csv,
            "summary_json": summary_json,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[saved] {single_csv}")
    print(f"[saved] {combo_csv}")
    if joined_csv:
        print(f"[saved] {joined_csv}")
    print(f"[saved] {summary_json}")
    print(f"[summary] n={len(labels)} n_pos={sum(labels)} n_features={len(features)}")
    print("[top single]")
    for row in single_rows[:10]:
        print(
            f"{row['feature']} dir={row['direction']} auc={float(row['auroc']):.4f} "
            f"ap={float(row['ap']):.4f} p@25={float(row['p_at_25']):.3f}"
        )
    print("[top combo]")
    for row in combo_rows[:10]:
        print(
            f"{row['combo_kind']} {row['feature_a']} + {row['feature_b']} "
            f"auc={float(row['auroc']):.4f} ap={float(row['ap']):.4f} "
            f"p@25={float(row['p_at_25']):.3f}"
        )


if __name__ == "__main__":
    main()
