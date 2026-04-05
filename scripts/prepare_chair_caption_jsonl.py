#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".json":
        obj = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(obj, list):
            return [dict(x) for x in obj if isinstance(x, dict)]
        raise ValueError(f"Expected list json in {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(dict(obj))
    return rows


def parse_image_id(image_name: str) -> str:
    digits = re.findall(r"(\d+)", str(image_name))
    if not digits:
        return ""
    return str(int(digits[-1]))


def normalize_image_id(row: Dict[str, Any], image_id_key: str, image_key: str) -> Optional[int]:
    raw_image_id = str(row.get(image_id_key, "")).strip()
    if raw_image_id:
        digits = re.findall(r"(\d+)", raw_image_id)
        if digits:
            return int(digits[-1])
    image = str(row.get(image_key, "")).strip()
    if image:
        parsed = parse_image_id(image)
        if parsed:
            return int(parsed)
    qid = str(row.get("question_id", row.get("id", ""))).strip()
    if qid:
        digits = re.findall(r"(\d+)", qid)
        if digits:
            return int(digits[-1])
    return None


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize caption predictions for EAZY CHAIR eval.")
    ap.add_argument("--in_file", type=str, required=True)
    ap.add_argument("--out_file", type=str, required=True)
    ap.add_argument("--image_id_key", type=str, default="image_id")
    ap.add_argument("--image_key", type=str, default="image")
    ap.add_argument("--drop_missing", action="store_true")
    args = ap.parse_args()

    rows = read_json_or_jsonl(os.path.abspath(args.in_file))
    out: List[Dict[str, Any]] = []
    missing = 0
    for row in rows:
        image_id = normalize_image_id(row, args.image_id_key, args.image_key)
        if image_id is None:
            missing += 1
            if args.drop_missing:
                continue
            row[args.image_id_key] = ""
        else:
            row[args.image_id_key] = int(image_id)
        out.append(row)

    write_jsonl(os.path.abspath(args.out_file), out)
    print("[saved]", os.path.abspath(args.out_file))
    print("[info] n_rows=", len(out), "missing_image_id=", missing)


if __name__ == "__main__":
    main()
