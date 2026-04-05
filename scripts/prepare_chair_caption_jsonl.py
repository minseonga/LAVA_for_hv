#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List


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
    args = ap.parse_args()

    rows = read_json_or_jsonl(os.path.abspath(args.in_file))
    out: List[Dict[str, Any]] = []
    for row in rows:
        image_id = str(row.get(args.image_id_key, "")).strip()
        if not image_id:
            image = str(row.get(args.image_key, "")).strip()
            image_id = parse_image_id(image)
        if image_id:
            row[args.image_id_key] = image_id
        out.append(row)

    write_jsonl(os.path.abspath(args.out_file), out)
    print("[saved]", os.path.abspath(args.out_file))


if __name__ == "__main__":
    main()
