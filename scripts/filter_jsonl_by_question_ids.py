#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def safe_id(value: object) -> str:
    return str(value or "").strip()


def row_id(row: Dict[str, Any], key: str) -> str:
    if key != "auto":
        return safe_id(row.get(key))
    return safe_id(row.get("question_id", row.get("id")))


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter a jsonl file to the question ids in another jsonl file.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--question_id_key", type=str, default="auto")
    ap.add_argument("--row_id_key", type=str, default="auto")
    ap.add_argument("--missing_ok", action="store_true")
    args = ap.parse_args()

    questions = read_jsonl(os.path.abspath(args.question_file))
    if int(args.limit) > 0:
        questions = questions[: int(args.limit)]
    wanted = [row_id(row, args.question_id_key) for row in questions]
    wanted = [sid for sid in wanted if sid]

    rows_by_id = {row_id(row, args.row_id_key): row for row in read_jsonl(os.path.abspath(args.in_jsonl))}
    out: List[Dict[str, Any]] = []
    missing: List[str] = []
    for sid in wanted:
        row = rows_by_id.get(sid)
        if row is None:
            missing.append(sid)
            continue
        out.append(row)

    if missing and not bool(args.missing_ok):
        raise RuntimeError(f"Missing {len(missing)} requested ids in {args.in_jsonl}; first={missing[:5]}")

    write_jsonl(os.path.abspath(args.out_jsonl), out)
    print("[saved]", os.path.abspath(args.out_jsonl))
    print("[info] n_rows=", len(out), "missing=", len(missing))


if __name__ == "__main__":
    main()
