#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Sequence


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            cols.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def row_id(row: Dict[str, Any]) -> str:
    for key in ("id", "image_id", "question_id"):
        value = str(row.get(key, "")).strip()
        if value:
            try:
                return str(int(value))
            except Exception:
                return value
    return ""


def maybe_float(value: Any) -> float | None:
    s = str(value if value is not None else "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def add_missing_utility(row: Dict[str, Any]) -> None:
    base_f1 = maybe_float(row.get("base_f1"))
    base_chair_i = maybe_float(row.get("base_chair_i"))
    int_f1 = maybe_float(row.get("int_f1"))
    int_chair_i = maybe_float(row.get("int_chair_i"))
    if str(row.get("baseline_claim_utility", "")).strip() == "" and base_f1 is not None and base_chair_i is not None:
        row["baseline_claim_utility"] = float(base_f1 - base_chair_i)
    if str(row.get("intervention_claim_utility", "")).strip() == "" and int_f1 is not None and int_chair_i is not None:
        row["intervention_claim_utility"] = float(int_f1 - int_chair_i)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge pairwise route decision rows with intervention-only feature rows for route distillation."
    )
    ap.add_argument("--route_rows_csv", required=True)
    ap.add_argument("--feature_rows_csv", action="append", default=[])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument(
        "--feature_prefix",
        action="append",
        default=[],
        help="Only merge feature columns with this prefix. Repeatable. Empty means merge all non-conflicting columns.",
    )
    ap.add_argument("--teacher_positive_col", default="teacher_positive")
    ap.add_argument("--teacher_fallback_col", default="teacher_fallback")
    ap.add_argument("--route_col", default="", help="Optional route column used to create a binary distillation target.")
    ap.add_argument("--positive_route", default="baseline")
    ap.add_argument("--target_col", default="", help="Optional binary target column to create from route_col.")
    args = ap.parse_args()

    route_rows = read_csv_rows(os.path.abspath(args.route_rows_csv))
    merged_by_id: Dict[str, Dict[str, Any]] = {}
    for row in route_rows:
        sid = row_id(row)
        if not sid:
            continue
        item: Dict[str, Any] = dict(row)
        if str(item.get(args.teacher_fallback_col, "")).strip() == "" and str(item.get(args.teacher_positive_col, "")).strip() != "":
            item[str(args.teacher_fallback_col)] = item.get(args.teacher_positive_col)
        if str(args.route_col or "").strip() and str(args.target_col or "").strip():
            item[str(args.target_col)] = int(
                str(item.get(str(args.route_col), "")).strip() == str(args.positive_route).strip()
            )
        add_missing_utility(item)
        merged_by_id[sid] = item

    prefixes = [str(x) for x in args.feature_prefix if str(x).strip()]
    n_feature_rows = 0
    n_merged_feature_rows = 0
    n_feature_cols_added = 0
    for feature_csv in args.feature_rows_csv:
        feature_rows = read_csv_rows(os.path.abspath(feature_csv))
        for feature_row in feature_rows:
            n_feature_rows += 1
            sid = row_id(feature_row)
            if not sid or sid not in merged_by_id:
                continue
            n_merged_feature_rows += 1
            target = merged_by_id[sid]
            for key, value in feature_row.items():
                skey = str(key)
                if skey in {"id", "image", "image_id", "question", "question_id"}:
                    continue
                if prefixes and not any(skey.startswith(prefix) for prefix in prefixes):
                    continue
                if skey in target and str(target.get(skey, "")).strip() != "":
                    continue
                target[skey] = value
                n_feature_cols_added += 1

    rows = [merged_by_id[sid] for sid in sorted(merged_by_id.keys(), key=lambda x: int(x) if x.isdigit() else x)]
    write_csv(args.out_csv, rows)
    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "route_rows_csv": os.path.abspath(args.route_rows_csv),
                    "feature_rows_csv": [os.path.abspath(p) for p in args.feature_rows_csv],
                    "feature_prefix": prefixes,
                    "teacher_positive_col": str(args.teacher_positive_col),
                    "teacher_fallback_col": str(args.teacher_fallback_col),
                    "route_col": str(args.route_col),
                    "positive_route": str(args.positive_route),
                    "target_col": str(args.target_col),
                },
                "counts": {
                    "n_route_rows": int(len(route_rows)),
                    "n_output_rows": int(len(rows)),
                    "n_feature_rows_seen": int(n_feature_rows),
                    "n_feature_rows_merged": int(n_merged_feature_rows),
                    "n_feature_values_added": int(n_feature_cols_added),
                },
                "outputs": {"out_csv": os.path.abspath(args.out_csv)},
            },
        )
    print(f"[saved] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
