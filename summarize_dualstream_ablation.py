#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize dual-stream ablation (A/B/A+B).")
    ap.add_argument("--ablation_root", type=str, required=True)
    ap.add_argument("--beam6_summary_json", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    root = os.path.abspath(args.ablation_root)
    out_csv = os.path.abspath(args.out_csv) if args.out_csv else os.path.join(root, "ablation_table.csv")
    out_json = os.path.abspath(args.out_json) if args.out_json else os.path.join(root, "ablation_table.json")

    rows: List[Dict[str, Any]] = []
    methods = ["a_only", "b_only", "a_plus_b"]
    for m in methods:
        sj = os.path.join(root, m, "summary.json")
        obj = read_json(sj)
        if not obj:
            continue
        cnt = obj.get("counts", {})
        rows.append(
            {
                "method": m,
                "accuracy": cnt.get("accuracy"),
                "accuracy_strict": cnt.get("accuracy_strict"),
                "accuracy_heuristic": cnt.get("accuracy_heuristic"),
                "mean_latency_sec_per_sample": cnt.get("mean_latency_sec_per_sample"),
                "mean_generated_tokens": cnt.get("mean_generated_tokens"),
                "n_valid": cnt.get("n_valid"),
                "n_error": cnt.get("n_error"),
            }
        )

    beam_obj = read_json(args.beam6_summary_json) if args.beam6_summary_json else {}
    if beam_obj:
        cnt = beam_obj.get("counts", {})
        rows.append(
            {
                "method": "beam6_reference",
                "accuracy": cnt.get("accuracy"),
                "accuracy_strict": cnt.get("accuracy_strict"),
                "accuracy_heuristic": cnt.get("accuracy_heuristic"),
                "mean_latency_sec_per_sample": None,
                "mean_generated_tokens": None,
                "n_valid": cnt.get("n_valid"),
                "n_error": None,
            }
        )

    write_csv(out_csv, rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)
    print("[saved]", out_csv)
    print("[saved]", out_json)


if __name__ == "__main__":
    main()

