#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple


NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "a",
    "an",
}

STOPWORDS = {
    "does",
    "is",
    "are",
    "there",
    "this",
    "image",
    "in",
    "the",
    "a",
    "an",
    "of",
    "and",
    "between",
    "direct",
    "contact",
    "with",
    "without",
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def image_stem(image_name: str) -> str:
    base = os.path.basename(str(image_name).strip())
    stem, _ = os.path.splitext(base)
    return stem


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_relation_objects(question: str) -> List[str]:
    q = question.strip().lower().rstrip("?")
    m = re.search(r"between the (.+?) and (.+)$", q)
    if not m:
        return []
    left = m.group(1).strip()
    right = m.group(2).strip()
    return unique_keep_order([left, right])


def parse_existence_object(question: str) -> List[str]:
    q = question.strip().lower().rstrip("?")
    m = re.match(r"^is there (.+?) in this image$", q)
    if not m:
        return []
    phrase = m.group(1).strip()
    words = phrase.split()
    while words and (words[0].isdigit() or words[0] in NUMBER_WORDS):
        words = words[1:]
    if not words:
        return []
    return [" ".join(words)]


def parse_attribute_object(question: str) -> List[str]:
    q = question.strip().lower().rstrip("?")
    m = re.match(r"^is there (.+?) in this image$", q)
    if m:
        phrase = m.group(1).strip()
        words = phrase.split()
        while words and (words[0].isdigit() or words[0] in NUMBER_WORDS):
            words = words[1:]
        if words:
            return [" ".join(words)]
    m = re.match(r"^are there (.+?) in this image$", q)
    if m:
        phrase = m.group(1).strip()
        words = phrase.split()
        while words and (words[0].isdigit() or words[0] in NUMBER_WORDS):
            words = words[1:]
        if words:
            return [" ".join(words)]
    m = re.match(r"^does the (.+?)\s+[a-z0-9-]+(?:\s+.+?)?\s+in this image$", q)
    if m:
        phrase = m.group(1).strip()
        if phrase:
            return [phrase]
    m = re.match(r"^is the (.+?) in this image$", q)
    if m:
        phrase = m.group(1).strip()
        words = phrase.split()
        if len(words) >= 2:
            return [" ".join(words[:-1])]
        if words:
            return [words[0]]
    return []


def fallback_objects(question: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    keep = [tok for tok in tokens if tok not in STOPWORDS and not tok.isdigit()]
    if not keep:
        return []
    return unique_keep_order(keep[:2])


def infer_object_list(question: str, ann_type: str) -> List[str]:
    if ann_type in {"discriminative-relation", "relation"}:
        out = parse_relation_objects(question)
    elif ann_type == "discriminative-hallucination":
        out = parse_existence_object(question)
    else:
        out = parse_attribute_object(question)
    if out:
        return out
    return fallback_objects(question)


def select_rows_by_images(rows: Sequence[Dict[str, Any]], image_set: set[str]) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("image", "")).strip() in image_set]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare AMBER discriminative assets in the POPE-style internal format.")
    ap.add_argument("--amber_root", type=str, required=True)
    ap.add_argument("--query_json", type=str, default="")
    ap.add_argument("--annotations_json", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--discovery_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    amber_root = os.path.abspath(args.amber_root)
    query_json = os.path.abspath(args.query_json or os.path.join(amber_root, "data", "query", "query_discriminative.json"))
    annotations_json = os.path.abspath(args.annotations_json or os.path.join(amber_root, "data", "annotations.json"))
    out_dir = os.path.abspath(args.out_dir)

    queries = load_json(query_json)
    annotations = load_json(annotations_json)
    ann_by_id = {int(row["id"]): row for row in annotations}

    q_rows: List[Dict[str, Any]] = []
    q_obj_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []
    type_counter: Counter[str] = Counter()

    for item in queries:
        qid = int(item["id"])
        ann = ann_by_id.get(qid)
        if ann is None:
            continue
        truth = str(ann.get("truth", "")).strip().lower()
        if truth not in {"yes", "no"}:
            continue
        ann_type = str(ann.get("type", "")).strip()
        question = str(item.get("query", "")).strip()
        image = str(item.get("image", "")).strip()
        if not question or not image:
            continue

        object_list = infer_object_list(question, ann_type)
        q_row = {
            "question_id": str(qid),
            "image": image,
            "text": question,
            "question": question,
        }
        q_obj_row = dict(q_row)
        q_obj_row["object"] = object_list
        q_obj_row["amber_type"] = ann_type

        gt_rows.append({
            "id": str(qid),
            "answer": truth,
            "category": ann_type,
            "image_id": image_stem(image),
            "question": question,
            "orig_question_id": str(qid),
        })
        q_rows.append(q_row)
        q_obj_rows.append(q_obj_row)
        type_counter[ann_type] += 1

    image_names = sorted({str(row["image"]).strip() for row in q_rows})
    rng = random.Random(int(args.seed))
    rng.shuffle(image_names)
    n_discovery_images = max(1, int(round(float(args.discovery_ratio) * float(len(image_names)))))
    n_discovery_images = min(n_discovery_images, max(1, len(image_names) - 1))
    discovery_images = set(image_names[:n_discovery_images])
    test_images = set(image_names[n_discovery_images:])

    discovery_q = select_rows_by_images(q_rows, discovery_images)
    discovery_q_obj = select_rows_by_images(q_obj_rows, discovery_images)
    discovery_gt = [row for row in gt_rows if f"{row['image_id']}.jpg" in discovery_images]

    test_q = select_rows_by_images(q_rows, test_images)
    test_q_obj = select_rows_by_images(q_obj_rows, test_images)
    test_gt = [row for row in gt_rows if f"{row['image_id']}.jpg" in test_images]

    all_assets = os.path.join(out_dir, "all", "assets")
    discovery_assets = os.path.join(out_dir, "discovery", "assets")
    test_assets = os.path.join(out_dir, "test", "assets")

    gt_cols = ["id", "answer", "category", "image_id", "question", "orig_question_id"]

    write_jsonl(os.path.join(all_assets, "amber_q.jsonl"), q_rows)
    write_jsonl(os.path.join(all_assets, "amber_q_with_object.jsonl"), q_obj_rows)
    write_csv(os.path.join(all_assets, "amber_gt.csv"), gt_rows, gt_cols)

    write_jsonl(os.path.join(discovery_assets, "discovery_q.jsonl"), discovery_q)
    write_jsonl(os.path.join(discovery_assets, "discovery_q_with_object.jsonl"), discovery_q_obj)
    write_csv(os.path.join(discovery_assets, "discovery_gt.csv"), discovery_gt, gt_cols)

    write_jsonl(os.path.join(test_assets, "test_q.jsonl"), test_q)
    write_jsonl(os.path.join(test_assets, "test_q_with_object.jsonl"), test_q_obj)
    write_csv(os.path.join(test_assets, "test_gt.csv"), test_gt, gt_cols)

    summary = {
        "inputs": {
            "amber_root": amber_root,
            "query_json": query_json,
            "annotations_json": annotations_json,
            "discovery_ratio": float(args.discovery_ratio),
            "seed": int(args.seed),
        },
        "counts": {
            "n_total_questions": int(len(q_rows)),
            "n_total_images": int(len(image_names)),
            "n_discovery_questions": int(len(discovery_q)),
            "n_test_questions": int(len(test_q)),
            "n_discovery_images": int(len(discovery_images)),
            "n_test_images": int(len(test_images)),
            "by_type": dict(type_counter),
        },
        "outputs": {
            "all_q_jsonl": os.path.join(all_assets, "amber_q.jsonl"),
            "all_q_with_object_jsonl": os.path.join(all_assets, "amber_q_with_object.jsonl"),
            "all_gt_csv": os.path.join(all_assets, "amber_gt.csv"),
            "discovery_q_jsonl": os.path.join(discovery_assets, "discovery_q.jsonl"),
            "discovery_q_with_object_jsonl": os.path.join(discovery_assets, "discovery_q_with_object.jsonl"),
            "discovery_gt_csv": os.path.join(discovery_assets, "discovery_gt.csv"),
            "test_q_jsonl": os.path.join(test_assets, "test_q.jsonl"),
            "test_q_with_object_jsonl": os.path.join(test_assets, "test_q_with_object.jsonl"),
            "test_gt_csv": os.path.join(test_assets, "test_gt.csv"),
        },
    }
    write_json(os.path.join(out_dir, "split_summary.json"), summary)

    print("[saved]", os.path.join(out_dir, "split_summary.json"))
    print("[saved]", os.path.join(discovery_assets, "discovery_q_with_object.jsonl"))
    print("[saved]", os.path.join(test_assets, "test_q_with_object.jsonl"))


if __name__ == "__main__":
    main()
