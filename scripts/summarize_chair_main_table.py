#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Sequence, Tuple


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_chair_metrics(path: str) -> Tuple[Dict[str, float], int]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    overall = obj.get("overall_metrics", {})
    sentences = obj.get("sentences", [])
    chair_s = float(overall.get("CHAIRs", 0.0))
    chair_i = float(overall.get("CHAIRi", 0.0))
    recall = float(overall.get("Recall", 0.0))
    length = float(overall.get("Len", 0.0))
    precision = 1.0 - chair_i
    denom = precision + recall
    f1 = 0.0 if denom <= 0.0 else (2.0 * precision * recall / denom)
    return (
        {
            "CHAIRs": chair_s,
            "CHAIRi": chair_i,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "Len": length,
        },
        int(len(sentences)),
    )


def parse_entry(spec: str) -> Tuple[str, str, str]:
    parts = str(spec).split("::", 2)
    if len(parts) != 3:
        raise ValueError(f"entry must be method::split::path, got: {spec}")
    method, split, path = parts
    return method.strip(), split.strip(), os.path.abspath(path.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize CHAIR main-table metrics from split-level json files.")
    ap.add_argument("--entry", action="append", default=[], help="method::split::chair_json")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    baseline_by_split: Dict[str, Dict[str, float]] = {}
    parsed_entries = [parse_entry(spec) for spec in args.entry]

    for method, split, path in parsed_entries:
        metrics, n = load_chair_metrics(path)
        row: Dict[str, Any] = {
            "method": method,
            "split": split,
            "n": int(n),
            "chair_json": path,
            **metrics,
        }
        rows.append(row)
        if method == "baseline":
            baseline_by_split[split] = metrics

    for row in rows:
        baseline = baseline_by_split.get(str(row["split"]))
        if baseline is None:
            continue
        for key in ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len"):
            row[f"delta_vs_baseline_{key}"] = float(row[key]) - float(baseline[key])

    rows.sort(key=lambda r: (str(r["split"]), str(r["method"])))
    write_csv(args.out_csv, rows)

    out_obj = {
        "entries": rows,
        "baseline_by_split": baseline_by_split,
        "outputs": {
            "csv": os.path.abspath(args.out_csv),
            "json": os.path.abspath(args.out_json),
        },
    }
    write_json(args.out_json, out_obj)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
