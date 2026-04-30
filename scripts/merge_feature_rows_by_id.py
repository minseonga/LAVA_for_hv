#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Sequence


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def sample_id(row: Dict[str, Any]) -> str:
    for key in ("question_id", "id", "qid"):
        raw = str(row.get(key, "")).strip()
        if raw:
            try:
                return str(int(float(raw)))
            except Exception:
                return raw
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Left-join feature CSVs by sample id.")
    ap.add_argument("--base_csv", required=True)
    ap.add_argument("--extra_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--extra_prefix", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_rows = read_csv(args.base_csv)
    extra_rows = read_csv(args.extra_csv)
    extra_by_id = {sample_id(row): row for row in extra_rows if sample_id(row)}

    merged: List[Dict[str, Any]] = []
    n_matched = 0
    copied_cols = set()
    skipped_conflicts = set()
    for base in base_rows:
        sid = sample_id(base)
        out = dict(base)
        extra = extra_by_id.get(sid)
        if extra is not None:
            n_matched += 1
            for key, value in extra.items():
                if key in {"question_id", "id", "qid"}:
                    continue
                out_key = f"{args.extra_prefix}{key}" if args.extra_prefix and key not in base else key
                if out_key in out and not args.overwrite:
                    skipped_conflicts.add(out_key)
                    continue
                out[out_key] = value
                copied_cols.add(out_key)
        merged.append(out)

    write_csv(args.out_csv, merged)
    summary = {
        "inputs": {
            "base_csv": os.path.abspath(args.base_csv),
            "extra_csv": os.path.abspath(args.extra_csv),
            "extra_prefix": args.extra_prefix,
            "overwrite": bool(args.overwrite),
        },
        "counts": {
            "base_rows": len(base_rows),
            "extra_rows": len(extra_rows),
            "matched_rows": n_matched,
            "copied_columns": len(copied_cols),
            "skipped_conflict_columns": len(skipped_conflicts),
        },
        "copied_columns": sorted(copied_cols),
        "skipped_conflict_columns": sorted(skipped_conflicts),
        "outputs": {
            "out_csv": os.path.abspath(args.out_csv),
            "out_summary_json": os.path.abspath(args.out_summary_json) if args.out_summary_json else "",
        },
    }
    if args.out_summary_json:
        write_json(args.out_summary_json, summary)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print("[saved]", os.path.abspath(args.out_csv))
    if args.out_summary_json:
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
