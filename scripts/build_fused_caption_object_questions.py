#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


DEFAULT_FUSED_PROMPT = """Describe the image in a detailed caption. Then list only the concrete physical objects you mentioned in the caption.

Format your response exactly like this:
Caption: [your caption here]
Objects: [comma-separated singular physical object nouns only]

Rules for Objects:
- Include only concrete, visually verifiable physical objects explicitly mentioned in the caption.
- Do not include abstract concepts, actions, scene descriptions, emotions, states, relationships, attributes, colors, or numbers.
- Map visually concrete people roles to "person" (e.g., passenger, man, woman, child, player).
- Keep compound objects together (e.g., dining table, traffic light, cell phone).
- If no concrete physical object is mentioned, write "none"."""


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
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
    ap = argparse.ArgumentParser(description="Build fused caption+object-list generation questions.")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--prompt", default=DEFAULT_FUSED_PROMPT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default="fused_caption_object_generation")
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl, limit=int(args.limit))
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        image_id = str(row.get("image_id") or row.get("question_id") or row.get("id") or idx).strip()
        question_id = str(row.get("question_id") or image_id or idx).strip()
        out.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "image": row.get("image", ""),
                "question": str(args.prompt),
                "text": str(args.prompt),
                "label": "",
                "split": row.get("split", ""),
                "category": str(args.category),
            }
        )

    write_jsonl(args.out_jsonl, out)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "in_jsonl": os.path.abspath(args.in_jsonl),
                    "prompt": str(args.prompt),
                    "limit": int(args.limit),
                    "category": str(args.category),
                },
                "counts": {"n_rows": len(out)},
                "outputs": {"out_jsonl": os.path.abspath(args.out_jsonl)},
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
