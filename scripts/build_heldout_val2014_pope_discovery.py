#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set


POPE_CATEGORIES = ("adversarial", "popular", "random")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(fieldnames))
        wr.writeheader()
        wr.writerows(rows)


def extract_image_id_token(value: str) -> str:
    groups = re.findall(r"\d+", str(value or ""))
    if not groups:
        return ""
    for group in reversed(groups):
        if len(group) == 12:
            return group
    return groups[-1]


def normalize_image_refs(value: Any) -> Set[str]:
    s = str("" if value is None else value).strip()
    if not s:
        return set()
    base = os.path.basename(s)
    out = {s, base}
    image_id_token = extract_image_id_token(base)
    if image_id_token:
        image_id = int(image_id_token)
        out.add(str(image_id))
        out.add(f"{image_id:012d}")
        out.add(f"COCO_val2014_{image_id:012d}.jpg")
        out.add(f"COCO_train2014_{image_id:012d}.jpg")
    return out


def collect_excluded_images(jsonl_paths: Sequence[str], csv_paths: Sequence[str]) -> Set[str]:
    excluded: Set[str] = set()
    for path in jsonl_paths:
        if not str(path).strip():
            continue
        for row in read_jsonl(os.path.abspath(path)):
            for key in ("image", "image_id", "file_name", "filename"):
                excluded.update(normalize_image_refs(row.get(key)))
    for path in csv_paths:
        if not str(path).strip():
            continue
        with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for key in ("image", "image_id", "file_name", "filename"):
                    excluded.update(normalize_image_refs(row.get(key)))
    return excluded


def image_is_excluded(image_id: int, file_name: str, excluded: Set[str]) -> bool:
    refs = set()
    refs.update(normalize_image_refs(image_id))
    refs.update(normalize_image_refs(file_name))
    return bool(refs & excluded)


def article_for(noun: str) -> str:
    s = str(noun or "").strip().lower()
    if s[:1] in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"


def question_for(category: str) -> str:
    return f"Is there {article_for(category)} {category} in the image?"


def build_cooccurrence(image_to_cats: Dict[int, Set[str]]) -> Counter[tuple[str, str]]:
    cooccur: Counter[tuple[str, str]] = Counter()
    for cats in image_to_cats.values():
        ordered = sorted(cats)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                cooccur[(a, b)] += 1
                cooccur[(b, a)] += 1
    return cooccur


def sample_negatives(
    *,
    mode: str,
    present: Set[str],
    all_categories: Sequence[str],
    global_frequency: Counter[str],
    cooccur: Counter[tuple[str, str]],
    n: int,
    rng: random.Random,
) -> List[str]:
    absent = [c for c in all_categories if c not in present]
    if len(absent) <= n:
        return list(absent)
    if mode == "random":
        rng.shuffle(absent)
        return absent[:n]
    if mode == "popular":
        return sorted(absent, key=lambda c: (-global_frequency[c], rng.random()))[:n]
    if mode == "adversarial":
        scores: Counter[str] = Counter()
        for pc in present:
            for ac in absent:
                scores[ac] += cooccur.get((pc, ac), 0)
        return sorted(absent, key=lambda c: (-scores[c], rng.random()))[:n]
    raise ValueError(f"Unsupported POPE category: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a POPE-style discovery set from COCO val2014 while excluding images "
            "used by evaluation assets. Outputs category JSONL plus unified q/gt assets."
        )
    )
    ap.add_argument("--ann_file", required=True, help="COCO instances_val2014.json")
    ap.add_argument("--image_folder", required=True, help="Folder containing val2014 images")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude_jsonl", action="append", default=[], help="Question/pred JSONL files whose images must be excluded")
    ap.add_argument("--exclude_csv", action="append", default=[], help="GT/subset CSV files whose images must be excluded")
    ap.add_argument("--n_images", type=int, default=500)
    ap.add_argument("--n_pos_per_image_per_category", type=int, default=1)
    ap.add_argument("--n_neg_per_image_per_category", type=int, default=1)
    ap.add_argument("--min_present_categories", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.abspath(args.ann_file), "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    all_categories = sorted(cat_id_to_name.values())
    all_category_set = set(all_categories)

    image_id_to_file = {int(img["id"]): str(img["file_name"]) for img in coco["images"]}
    image_to_cats: Dict[int, Set[str]] = defaultdict(set)
    for ann in coco["annotations"]:
        category = cat_id_to_name.get(int(ann["category_id"]))
        if category:
            image_to_cats[int(ann["image_id"])].add(category)

    global_frequency: Counter[str] = Counter()
    for cats in image_to_cats.values():
        global_frequency.update(cats)
    cooccur = build_cooccurrence(image_to_cats)
    excluded = collect_excluded_images(args.exclude_jsonl, args.exclude_csv)

    valid_image_ids: List[int] = []
    n_missing_file = 0
    n_excluded = 0
    for image_id, file_name in image_id_to_file.items():
        present = image_to_cats.get(image_id, set())
        if len(present) < int(args.min_present_categories):
            continue
        if not os.path.exists(os.path.join(os.path.abspath(args.image_folder), file_name)):
            n_missing_file += 1
            continue
        if image_is_excluded(image_id, file_name, excluded):
            n_excluded += 1
            continue
        valid_image_ids.append(image_id)

    if not valid_image_ids:
        raise RuntimeError("No valid val2014 images remain after filtering.")

    rng.shuffle(valid_image_ids)
    sampled_image_ids = valid_image_ids[: min(int(args.n_images), len(valid_image_ids))]

    category_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in POPE_CATEGORIES}
    q_rows: List[Dict[str, Any]] = []
    q_obj_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []
    subset_rows: List[Dict[str, Any]] = []

    qid = 0
    for image_id in sampled_image_ids:
        file_name = image_id_to_file[image_id]
        present = sorted(image_to_cats[image_id] & all_category_set)
        if not present:
            continue
        for pope_category in POPE_CATEGORIES:
            present_pool = list(present)
            rng.shuffle(present_pool)
            positives = present_pool[: min(int(args.n_pos_per_image_per_category), len(present_pool))]
            negatives = sample_negatives(
                mode=pope_category,
                present=set(present),
                all_categories=all_categories,
                global_frequency=global_frequency,
                cooccur=cooccur,
                n=int(args.n_neg_per_image_per_category),
                rng=rng,
            )
            for label, objects in (("yes", positives), ("no", negatives)):
                for obj in objects:
                    sid = str(qid)
                    text = question_for(obj)
                    base = {
                        "question_id": sid,
                        "id": sid,
                        "image": file_name,
                        "image_id": file_name,
                        "text": text,
                        "question": text,
                        "category": pope_category,
                    }
                    q_rows.append(dict(base))
                    q_obj_rows.append({**base, "object": [obj]})
                    gt_rows.append(
                        {
                            "id": sid,
                            "answer": label,
                            "category": pope_category,
                            "image_id": file_name,
                            "question": text,
                            "orig_question_id": sid,
                            "group": pope_category,
                        }
                    )
                    subset_rows.append({"id": sid, "group": pope_category})
                    category_rows[pope_category].append(
                        {
                            "question_id": sid,
                            "id": sid,
                            "image": file_name,
                            "text": text,
                            "question": text,
                            "label": label,
                            "category": pope_category,
                            "object": obj,
                        }
                    )
                    qid += 1

    for pope_category, rows in category_rows.items():
        write_jsonl(os.path.join(out_dir, f"discovery_{pope_category}.jsonl"), rows)

    write_jsonl(os.path.join(out_dir, "discovery_q.jsonl"), q_rows)
    write_jsonl(os.path.join(out_dir, "discovery_q_with_object.jsonl"), q_obj_rows)
    write_csv(
        os.path.join(out_dir, "discovery_gt.csv"),
        gt_rows,
        ["id", "answer", "category", "image_id", "question", "orig_question_id", "group"],
    )
    write_csv(os.path.join(out_dir, "discovery_subset_ids.csv"), subset_rows, ["id", "group"])

    summary = {
        "inputs": {
            "ann_file": os.path.abspath(args.ann_file),
            "image_folder": os.path.abspath(args.image_folder),
            "exclude_jsonl": [os.path.abspath(p) for p in args.exclude_jsonl],
            "exclude_csv": [os.path.abspath(p) for p in args.exclude_csv],
            "n_images": int(args.n_images),
            "n_pos_per_image_per_category": int(args.n_pos_per_image_per_category),
            "n_neg_per_image_per_category": int(args.n_neg_per_image_per_category),
            "min_present_categories": int(args.min_present_categories),
            "seed": int(args.seed),
        },
        "counts": {
            "n_coco_images": int(len(image_id_to_file)),
            "n_exclude_refs": int(len(excluded)),
            "n_excluded_images": int(n_excluded),
            "n_missing_files": int(n_missing_file),
            "n_valid_images_after_filter": int(len(valid_image_ids)),
            "n_sampled_images": int(len(sampled_image_ids)),
            "n_questions": int(len(q_rows)),
            "n_yes": int(sum(1 for r in gt_rows if r["answer"] == "yes")),
            "n_no": int(sum(1 for r in gt_rows if r["answer"] == "no")),
            "by_category": {
                category: {
                    "n": int(len(rows)),
                    "yes": int(sum(1 for r in rows if r["label"] == "yes")),
                    "no": int(sum(1 for r in rows if r["label"] == "no")),
                }
                for category, rows in category_rows.items()
            },
        },
        "outputs": {
            "q_jsonl": os.path.join(out_dir, "discovery_q.jsonl"),
            "q_with_object_jsonl": os.path.join(out_dir, "discovery_q_with_object.jsonl"),
            "gt_csv": os.path.join(out_dir, "discovery_gt.csv"),
            "subset_ids_csv": os.path.join(out_dir, "discovery_subset_ids.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
