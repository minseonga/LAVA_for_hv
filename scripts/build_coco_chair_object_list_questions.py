#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


DEFAULT_PROMPT = (
    "List the salient visible objects and entities in this image. "
    "Answer only with a comma-separated list of nouns or short noun phrases. "
    "Do not write a sentence."
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert caption questions into auxiliary object-list prompts.")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default="chair_object_list_probe")
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        image_id = str(row.get("image_id") or row.get("question_id") or idx).strip()
        question_id = str(row.get("question_id") or image_id or idx).strip()
        out.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "image": row.get("image", ""),
                "question": args.prompt,
                "text": args.prompt,
                "label": "",
                "split": row.get("split", ""),
                "category": args.category,
            }
        )

    write_jsonl(args.out_jsonl, out)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if args.out_summary_json:
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "in_jsonl": os.path.abspath(args.in_jsonl),
                    "prompt": args.prompt,
                    "limit": int(args.limit),
                    "category": args.category,
                },
                "counts": {"n_rows": len(out)},
                "outputs": {"out_jsonl": os.path.abspath(args.out_jsonl)},
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
