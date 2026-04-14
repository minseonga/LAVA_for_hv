#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Set

from extract_generative_semantic_pairwise_features import read_prediction_map


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_selected_ids(raw: str) -> Set[str]:
    out = set()
    for item in str(raw or "").replace(",", "|").split("|"):
        item = item.strip()
        if not item:
            continue
        try:
            item = str(int(item))
        except Exception:
            pass
        out.add(item)
    return out


def load_selected_ids(selected_ids: str, selected_rule_csv: str, rule_rank: int) -> Set[str]:
    if selected_ids.strip():
        return parse_selected_ids(selected_ids)
    rows = read_csv_rows(selected_rule_csv)
    if not rows:
        return set()
    idx = max(0, int(rule_rank))
    if idx >= len(rows):
        raise IndexError(f"rule_rank={rule_rank} out of range for {selected_rule_csv} ({len(rows)} rows)")
    return parse_selected_ids(rows[idx].get("selected_ids", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize routed captions from selected fallback sample IDs.")
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--selected_ids", default="")
    ap.add_argument("--selected_rule_csv", default="")
    ap.add_argument("--rule_rank", type=int, default=0)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--caption_key", default="output")
    args = ap.parse_args()

    selected = load_selected_ids(args.selected_ids, args.selected_rule_csv, int(args.rule_rank))
    baseline = read_prediction_map(args.baseline_pred_jsonl)
    intervention = read_prediction_map(args.intervention_pred_jsonl)
    ids = sorted(
        set(baseline) & set(intervention),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        use_baseline = sid in selected
        src = baseline[sid] if use_baseline else intervention[sid]
        rows.append(
            {
                "question_id": sid,
                "id": sid,
                "image_id": sid,
                "image": src.get("image", ""),
                str(args.caption_key): src.get("text", ""),
                "route": "baseline" if use_baseline else "intervention",
                "route_source": "selected_fallback",
            }
        )

    write_jsonl(args.out_jsonl, rows)
    summary = {
        "inputs": {
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "selected_rule_csv": os.path.abspath(args.selected_rule_csv) if args.selected_rule_csv else "",
            "rule_rank": int(args.rule_rank),
        },
        "counts": {
            "n_rows": len(rows),
            "n_selected_ids": len(selected),
            "n_baseline_routes": sum(1 for row in rows if row["route"] == "baseline"),
            "n_intervention_routes": sum(1 for row in rows if row["route"] == "intervention"),
        },
        "outputs": {"routed_jsonl": os.path.abspath(args.out_jsonl)},
    }
    if args.out_summary_json:
        write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if args.out_summary_json:
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
