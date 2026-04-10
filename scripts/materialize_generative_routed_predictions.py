#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize final routed caption predictions from generative decision rows.")
    ap.add_argument("--decision_rows_csv", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--caption_key", type=str, default="text")
    ap.add_argument("--image_id_key", type=str, default="image_id")
    ap.add_argument("--question_id_key", type=str, default="question_id")
    args = ap.parse_args()

    rows = read_csv_rows(os.path.abspath(args.decision_rows_csv))
    out_rows: List[Dict[str, Any]] = []
    n_baseline = 0
    n_method = 0
    n_missing = 0

    for row in rows:
        route = str(row.get("route", "method")).strip()
        use_baseline = route == "baseline"
        final_text = str(row.get("baseline_text" if use_baseline else "intervention_text", "")).strip()
        if not final_text:
            n_missing += 1
        image = str(row.get("image", "")).strip()
        image_id = str(row.get("image_id", "")).strip()
        sample_id = str(row.get("id", "")).strip()
        out: Dict[str, Any] = {
            str(args.question_id_key): sample_id,
            "id": sample_id,
            "image": image,
            str(args.image_id_key): image_id,
            str(args.caption_key): final_text,
            "route": route,
            "route_source": str(row.get("route_source", "")).strip(),
        }
        out_rows.append(out)
        n_baseline += int(use_baseline)
        n_method += int(not use_baseline)

    write_jsonl(os.path.abspath(args.out_jsonl), out_rows)
    summary = {
        "inputs": {
            "decision_rows_csv": os.path.abspath(args.decision_rows_csv),
            "caption_key": str(args.caption_key),
            "image_id_key": str(args.image_id_key),
            "question_id_key": str(args.question_id_key),
        },
        "counts": {
            "n_rows": int(len(out_rows)),
            "n_baseline_routes": int(n_baseline),
            "n_method_routes": int(n_method),
            "n_missing_final_text": int(n_missing),
        },
        "outputs": {
            "final_predictions_jsonl": os.path.abspath(args.out_jsonl),
        },
    }
    if str(args.out_summary_json).strip():
        write_json(os.path.abspath(args.out_summary_json), summary)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if str(args.out_summary_json).strip():
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
