#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def safe_str(x: Any) -> str:
    return str("" if x is None else x).strip()


def norm_image_id(x: Any) -> str:
    s = safe_str(x)
    if not s:
        return ""
    base = os.path.basename(s)
    stem = os.path.splitext(base)[0]
    if stem.isdigit():
        return str(int(stem))
    return stem


def norm_object_name(x: Any) -> str:
    s = safe_str(x).lower()
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def slug(x: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_")
    return s[:48] or "object"


def article_for(phrase: str) -> str:
    first = (phrase or "").strip().lower()[:1]
    return "an" if first in {"a", "e", "i", "o", "u"} else "a"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(fieldnames))
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fieldnames})


def load_excluded_image_ids(paths: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for raw_path in paths:
        path = safe_str(raw_path)
        if not path:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        lower = path.lower()
        if lower.endswith(".csv"):
            with open(path, "r", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                for row in rd:
                    for key in ("image_id", "image", "imageId"):
                        val = row.get(key)
                        if safe_str(val):
                            out.add(norm_image_id(val))
                            break
        elif lower.endswith(".jsonl"):
            for row in read_jsonl(path):
                for key in ("image_id", "image", "imageId"):
                    val = row.get(key)
                    if safe_str(val):
                        out.add(norm_image_id(val))
                        break
        else:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                iterable = payload.values()
            elif isinstance(payload, list):
                iterable = payload
            else:
                iterable = []
            for row in iterable:
                if isinstance(row, dict):
                    for key in ("image_id", "image", "imageId"):
                        val = row.get(key)
                        if safe_str(val):
                            out.add(norm_image_id(val))
                            break
    return {x for x in out if x}


def load_scene_graph_objects(path: str, min_object_len: int, max_object_words: int) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError("GQA scene graph JSON must be a dict keyed by image id.")

    image_to_objects: Dict[str, List[str]] = {}
    for raw_image_id, graph in payload.items():
        image_id = norm_image_id(raw_image_id)
        if not image_id or not isinstance(graph, dict):
            continue
        objects = graph.get("objects", {})
        names: Set[str] = set()
        if isinstance(objects, dict):
            obj_iter = objects.values()
        elif isinstance(objects, list):
            obj_iter = objects
        else:
            obj_iter = []
        for obj in obj_iter:
            if not isinstance(obj, dict):
                continue
            name = norm_object_name(obj.get("name", ""))
            if len(name) < int(min_object_len):
                continue
            if len(name.split()) > int(max_object_words):
                continue
            names.add(name)
        if names:
            image_to_objects[image_id] = sorted(names)
    return image_to_objects


def balanced_take(
    positives: Sequence[Tuple[str, str]],
    negatives: Sequence[Tuple[str, str]],
    target_n: int,
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    pos = list(positives)
    neg = list(negatives)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_yes = min(len(pos), target_n // 2)
    n_no = min(len(neg), target_n - n_yes)
    # If one side is short, fill from the other side.
    if n_yes + n_no < target_n:
        spare_yes = min(len(pos), target_n - n_no)
        spare_no = min(len(neg), target_n - spare_yes)
        n_yes, n_no = spare_yes, spare_no

    rows = [(image_id, obj, "yes") for image_id, obj in pos[:n_yes]]
    rows.extend((image_id, obj, "no") for image_id, obj in neg[:n_no])
    rng.shuffle(rows)
    return rows


def maybe_existing_image(image_folder: str, image_name: str) -> bool:
    if not image_folder:
        return True
    return os.path.exists(os.path.join(image_folder, image_name))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a POPE-style object-existence calibration bank from GQA train "
            "scene graphs. This creates labels from held-out train scene graphs; "
            "changed-candidate enrichment is done later from model predictions."
        )
    )
    ap.add_argument("--scene_graph_json", type=str, default="/home/kms/data/GQA/train_sceneGraphs.json")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--target_n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--image_ext", type=str, default=".jpg")
    ap.add_argument("--image_folder", type=str, default="")
    ap.add_argument("--require_images", action="store_true")
    ap.add_argument("--exclude_gt_csv", action="append", default=[])
    ap.add_argument("--exclude_question_jsonl", action="append", default=[])
    ap.add_argument("--min_object_len", type=int, default=2)
    ap.add_argument("--max_object_words", type=int, default=4)
    ap.add_argument("--max_pos_per_image", type=int, default=3)
    ap.add_argument("--max_neg_per_image", type=int, default=3)
    ap.add_argument("--qid_prefix", type=str, default="gqa_train_pope")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    excluded = load_excluded_image_ids(list(args.exclude_gt_csv) + list(args.exclude_question_jsonl))
    image_to_objects = load_scene_graph_objects(
        os.path.abspath(args.scene_graph_json),
        min_object_len=int(args.min_object_len),
        max_object_words=int(args.max_object_words),
    )
    if excluded:
        image_to_objects = {k: v for k, v in image_to_objects.items() if k not in excluded}
    if not image_to_objects:
        raise RuntimeError("No eligible GQA train images after filtering.")

    vocab = sorted({obj for objs in image_to_objects.values() for obj in objs})
    if len(vocab) < 2:
        raise RuntimeError("Object vocabulary is too small.")

    positives: List[Tuple[str, str]] = []
    negatives: List[Tuple[str, str]] = []
    missing_images = 0
    skipped_missing_images = 0

    image_ids = list(image_to_objects)
    rng.shuffle(image_ids)
    for image_id in image_ids:
        image_name = f"{image_id}{args.image_ext}"
        image_exists = maybe_existing_image(args.image_folder, image_name)
        if not image_exists:
            missing_images += 1
            if args.require_images:
                skipped_missing_images += 1
                continue

        present = set(image_to_objects[image_id])
        pos_objs = list(present)
        rng.shuffle(pos_objs)
        for obj in pos_objs[: max(0, int(args.max_pos_per_image))]:
            positives.append((image_id, obj))

        absent_pool = [obj for obj in vocab if obj not in present]
        rng.shuffle(absent_pool)
        for obj in absent_pool[: max(0, int(args.max_neg_per_image))]:
            negatives.append((image_id, obj))

    selected = balanced_take(positives, negatives, int(args.target_n), rng)
    if not selected:
        raise RuntimeError("No rows selected.")

    q_rows: List[Dict[str, Any]] = []
    q_obj_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []
    subset_rows: List[Dict[str, Any]] = []
    seen_qids: Dict[str, int] = defaultdict(int)

    for idx, (image_id, obj, answer) in enumerate(selected):
        image_name = f"{image_id}{args.image_ext}"
        question = f"Is there {article_for(obj)} {obj} in the image?"
        base_qid = f"{args.qid_prefix}_{image_id}_{answer}_{slug(obj)}"
        seen_qids[base_qid] += 1
        qid = base_qid if seen_qids[base_qid] == 1 else f"{base_qid}_{seen_qids[base_qid]}"
        row = {
            "question_id": qid,
            "id": qid,
            "image": image_name,
            "image_id": image_id,
            "text": question,
            "question": question,
        }
        q_rows.append(dict(row))
        q_obj_rows.append({**row, "object": [obj]})
        gt_rows.append(
            {
                "id": qid,
                "answer": answer,
                "category": "gqa_train",
                "image_id": image_name,
                "question": question,
                "orig_question_id": qid,
                "group": f"gqa_train_{answer}",
                "object": obj,
            }
        )
        subset_rows.append({"id": qid, "group": f"gqa_train_{answer}"})

    q_jsonl = os.path.join(out_dir, "gqa_train_pope_q.jsonl")
    q_obj_jsonl = os.path.join(out_dir, "gqa_train_pope_q_with_object.jsonl")
    gt_csv = os.path.join(out_dir, "gqa_train_pope_gt.csv")
    subset_csv = os.path.join(out_dir, "gqa_train_pope_subset_ids.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    write_jsonl(q_jsonl, q_rows)
    write_jsonl(q_obj_jsonl, q_obj_rows)
    write_csv(
        gt_csv,
        gt_rows,
        ["id", "answer", "category", "image_id", "question", "orig_question_id", "group", "object"],
    )
    write_csv(subset_csv, subset_rows, ["id", "group"])

    counts_by_answer = defaultdict(int)
    for row in gt_rows:
        counts_by_answer[row["answer"]] += 1
    summary = {
        "inputs": {
            "scene_graph_json": os.path.abspath(args.scene_graph_json),
            "target_n": int(args.target_n),
            "seed": int(args.seed),
            "image_ext": args.image_ext,
            "image_folder": os.path.abspath(args.image_folder) if args.image_folder else "",
            "require_images": bool(args.require_images),
            "exclude_gt_csv": [os.path.abspath(p) for p in args.exclude_gt_csv],
            "exclude_question_jsonl": [os.path.abspath(p) for p in args.exclude_question_jsonl],
            "min_object_len": int(args.min_object_len),
            "max_object_words": int(args.max_object_words),
            "max_pos_per_image": int(args.max_pos_per_image),
            "max_neg_per_image": int(args.max_neg_per_image),
            "qid_prefix": args.qid_prefix,
        },
        "counts": {
            "n_excluded_images": int(len(excluded)),
            "n_eligible_images": int(len(image_to_objects)),
            "n_vocab": int(len(vocab)),
            "n_positive_candidates": int(len(positives)),
            "n_negative_candidates": int(len(negatives)),
            "n_selected": int(len(gt_rows)),
            "n_selected_yes": int(counts_by_answer["yes"]),
            "n_selected_no": int(counts_by_answer["no"]),
            "n_missing_images_checked": int(missing_images),
            "n_skipped_missing_images": int(skipped_missing_images),
        },
        "outputs": {
            "q_jsonl": q_jsonl,
            "q_with_object_jsonl": q_obj_jsonl,
            "gt_csv": gt_csv,
            "subset_ids_csv": subset_csv,
            "summary_json": summary_json,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
