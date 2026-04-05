#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Sequence, Set


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


OBJECT_PATTERNS = [
    re.compile(r"^is there (?:a|an) (.+?) in the image\??$", re.IGNORECASE),
    re.compile(r"^is there (.+?) in the image\??$", re.IGNORECASE),
    re.compile(r"^are there (.+?) in the image\??$", re.IGNORECASE),
]


def normalize_object_term(value: object) -> str:
    s = str(value if value is not None else "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    if s.startswith("any "):
        s = s[4:].strip()
    return s


def extract_object_terms(row: Dict[str, Any]) -> List[str]:
    raw = row.get("object")
    values: Sequence[object]
    if isinstance(raw, list):
        values = raw
    elif raw is None:
        values = []
    else:
        values = [raw]

    terms: List[str] = []
    for value in values:
        term = normalize_object_term(value)
        if term:
            terms.append(term)
    if terms:
        return terms

    question = str(row.get("question", row.get("text", ""))).strip()
    for pattern in OBJECT_PATTERNS:
        m = pattern.match(question)
        if m:
            term = normalize_object_term(m.group(1))
            if term:
                return [term]
    return []


def unique_keep_order(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        s = normalize_object_term(value)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unique-image caption prompts from a POPE discovery question file.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--prompt", type=str, default="Please describe this image in detail.")
    ap.add_argument("--include_objects", type=str, default="true")
    ap.add_argument("--max_objects_per_image", type=int, default=32)
    args = ap.parse_args()

    rows = read_jsonl(args.question_file)
    include_objects = str(args.include_objects).strip().lower() in {"1", "true", "yes", "y"}
    out: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image = str(row.get("image", "")).strip()
        if not image:
            continue
        bucket = grouped.get(image)
        if bucket is None:
            image_id = str(row.get("image_id", "")).strip() or parse_image_id(image)
            bucket = {
                "image": image,
                "image_id": image_id,
                "question_id": image_id or str(len(grouped)),
                "objects": [],
            }
            grouped[image] = bucket
        bucket["objects"].extend(extract_object_terms(row))

    images_with_objects = 0
    total_objects = 0
    for bucket in grouped.values():
        objects = unique_keep_order(bucket.get("objects", []))
        if int(args.max_objects_per_image) > 0:
            objects = objects[: int(args.max_objects_per_image)]
        row = {
            "question_id": bucket["question_id"],
            "image_id": bucket["image_id"],
            "image": bucket["image"],
            "question": args.prompt,
            "text": args.prompt,
            "label": "",
            "category": "caption_discovery",
        }
        if include_objects and objects:
            row["object"] = objects
            images_with_objects += 1
            total_objects += len(objects)
        out.append(row)

    write_jsonl(args.out_jsonl, out)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "prompt": args.prompt,
                    "include_objects": bool(include_objects),
                    "max_objects_per_image": int(args.max_objects_per_image),
                },
                "counts": {
                    "n_input_rows": int(len(rows)),
                    "n_unique_images": int(len(out)),
                    "n_images_with_objects": int(images_with_objects),
                    "mean_objects_per_image": (
                        0.0 if len(out) == 0 else float(total_objects / float(max(1, len(out))))
                    ),
                },
                "outputs": {
                    "caption_question_jsonl": os.path.abspath(args.out_jsonl),
                },
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
