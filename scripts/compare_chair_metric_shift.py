#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

from summarize_chair_main_table import load_chair_metrics


METRIC_KEYS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def parse_pair(spec: str) -> Tuple[str, str, str]:
    parts = str(spec).split("::", 2)
    if len(parts) != 3:
        raise ValueError(f"pair must be name::normal_chair_json::fused_chair_json, got: {spec}")
    return parts[0].strip(), os.path.abspath(parts[1].strip()), os.path.abspath(parts[2].strip())


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare CHAIR metric shifts between normal and fused prompts.")
    ap.add_argument("--pair", action="append", default=[], help="name::normal_chair_json::fused_chair_json")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for spec in args.pair:
        name, normal_path, fused_path = parse_pair(spec)
        normal_metrics, normal_n = load_chair_metrics(normal_path)
        fused_metrics, fused_n = load_chair_metrics(fused_path)
        row: Dict[str, Any] = {
            "name": name,
            "normal_chair_json": normal_path,
            "fused_chair_json": fused_path,
            "normal_n": normal_n,
            "fused_n": fused_n,
        }
        for key in METRIC_KEYS:
            row[f"normal_{key}"] = normal_metrics[key]
            row[f"fused_{key}"] = fused_metrics[key]
            row[f"delta_fused_minus_normal_{key}"] = float(fused_metrics[key]) - float(normal_metrics[key])
        rows.append(row)

    write_csv(args.out_csv, rows)
    out = {
        "pairs": rows,
        "outputs": {
            "csv": os.path.abspath(args.out_csv),
            "json": os.path.abspath(args.out_json),
        },
    }
    write_json(args.out_json, out)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
