#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Set


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_image_id(image_name: str) -> str:
    digits = re.findall(r"(\d+)", str(image_name))
    if not digits:
        return ""
    return str(int(digits[-1]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unique-image caption prompts from a POPE discovery question file.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--prompt", type=str, default="Please describe this image in detail.")
    args = ap.parse_args()

    rows = read_jsonl(args.question_file)
    out: List[Dict[str, Any]] = []
    seen_images: Set[str] = set()
    for row in rows:
        image = str(row.get("image", "")).strip()
        if not image or image in seen_images:
            continue
        seen_images.add(image)
        image_id = str(row.get("image_id", "")).strip() or parse_image_id(image)
        qid = image_id or str(len(out))
        out.append(
            {
                "question_id": qid,
                "image_id": image_id,
                "image": image,
                "question": args.prompt,
                "text": args.prompt,
                "label": "",
                "category": "caption_discovery",
            }
        )

    write_jsonl(args.out_jsonl, out)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "prompt": args.prompt,
                },
                "counts": {
                    "n_input_rows": int(len(rows)),
                    "n_unique_images": int(len(out)),
                },
                "outputs": {
                    "caption_question_jsonl": os.path.abspath(args.out_jsonl),
                },
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
