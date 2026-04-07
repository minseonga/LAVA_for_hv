#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Sequence


def parse_image_id(image_name: str) -> int:
    digits = re.findall(r"(\d+)", str(image_name))
    if not digits:
        raise ValueError(f"could not parse image id from: {image_name}")
    return int(digits[-1])


def list_images(image_folder: str) -> List[str]:
    out: List[str] = []
    for name in os.listdir(image_folder):
        lower = name.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
            out.append(name)
    out.sort()
    return out


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_rows(images: Sequence[str], prompt: str, split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for image_name in images:
        image_id = parse_image_id(image_name)
        rows.append(
            {
                "question_id": str(image_id),
                "image_id": str(image_id),
                "image": image_name,
                "question": prompt,
                "text": prompt,
                "label": "",
                "split": split_name,
                "category": f"chair_{split_name}",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample random MS COCO image splits for CHAIR evaluation.")
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", type=str, default="Please describe this image in detail.")
    args = ap.parse_args()

    image_folder = os.path.abspath(args.image_folder)
    images = list_images(image_folder)
    need = int(args.n_val) + int(args.n_test)
    if len(images) < need:
        raise SystemExit(f"not enough images in {image_folder}: found {len(images)}, need {need}")

    rnd = random.Random(int(args.seed))
    sampled = list(images)
    rnd.shuffle(sampled)
    sampled = sampled[:need]

    val_images = sorted(sampled[: int(args.n_val)], key=parse_image_id)
    test_images = sorted(sampled[int(args.n_val) : need], key=parse_image_id)

    val_rows = build_rows(val_images, str(args.prompt), "val")
    test_rows = build_rows(test_images, str(args.prompt), "test")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    val_csv = os.path.join(out_dir, "val_images.csv")
    test_csv = os.path.join(out_dir, "test_images.csv")
    val_q = os.path.join(out_dir, "val_caption_q.jsonl")
    test_q = os.path.join(out_dir, "test_caption_q.jsonl")
    summary_json = os.path.join(out_dir, "summary.json")

    write_csv(
        val_csv,
        [{"split": "val", "image_id": parse_image_id(name), "image": name} for name in val_images],
    )
    write_csv(
        test_csv,
        [{"split": "test", "image_id": parse_image_id(name), "image": name} for name in test_images],
    )
    write_jsonl(val_q, val_rows)
    write_jsonl(test_q, test_rows)

    write_json(
        summary_json,
        {
            "inputs": {
                "image_folder": image_folder,
                "n_val": int(args.n_val),
                "n_test": int(args.n_test),
                "seed": int(args.seed),
                "prompt": str(args.prompt),
            },
            "counts": {
                "n_images_found": int(len(images)),
                "n_val": int(len(val_images)),
                "n_test": int(len(test_images)),
            },
            "outputs": {
                "val_images_csv": val_csv,
                "test_images_csv": test_csv,
                "val_caption_q_jsonl": val_q,
                "test_caption_q_jsonl": test_q,
                "summary_json": summary_json,
            },
        },
    )

    print("[saved]", val_csv)
    print("[saved]", test_csv)
    print("[saved]", val_q)
    print("[saved]", test_q)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
