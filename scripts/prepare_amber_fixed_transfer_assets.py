#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.prepare_amber_discriminative_assets import image_stem, infer_object_list


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


def strip_object(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.pop("object", None)
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare AMBER assets for fixed-transfer evaluation.")
    ap.add_argument("--amber_root", type=str, required=True)
    ap.add_argument("--query_all_json", type=str, default="")
    ap.add_argument("--annotations_json", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    amber_root = os.path.abspath(args.amber_root)
    query_all_json = os.path.abspath(args.query_all_json or os.path.join(amber_root, "data", "query", "query_all.json"))
    annotations_json = os.path.abspath(args.annotations_json or os.path.join(amber_root, "data", "annotations.json"))
    out_dir = os.path.abspath(args.out_dir)

    queries = load_json(query_all_json)
    annotations = load_json(annotations_json)
    ann_by_id = {int(row["id"]): row for row in annotations}

    rows_all_with_object: List[Dict[str, Any]] = []
    rows_gen_with_object: List[Dict[str, Any]] = []
    rows_disc_with_object: List[Dict[str, Any]] = []
    type_counter: Counter[str] = Counter()

    for item in queries:
        qid = int(item["id"])
        ann = ann_by_id.get(qid)
        if ann is None:
            continue
        ann_type = str(ann.get("type", "")).strip()
        image = str(item.get("image", "")).strip()
        query = str(item.get("query", "")).strip()
        truth = ann.get("truth", "")

        row = {
            "question_id": str(qid),
            "id": str(qid),
            "image": image,
            "image_id": image_stem(image),
            "text": query,
            "question": query,
            "label": (str(truth).strip().lower() if isinstance(truth, str) else ""),
            "object": ([] if ann_type == "generative" else infer_object_list(query, ann_type)),
            "amber_type": ann_type,
        }
        rows_all_with_object.append(row)
        type_counter[ann_type] += 1
        if ann_type == "generative":
            rows_gen_with_object.append(row)
        else:
            rows_disc_with_object.append(row)

    rows_all = strip_object(rows_all_with_object)
    rows_gen = strip_object(rows_gen_with_object)
    rows_disc = strip_object(rows_disc_with_object)

    outputs = {
        "all_q_jsonl": os.path.join(out_dir, "all", "assets", "amber_all_q.jsonl"),
        "all_q_with_object_jsonl": os.path.join(out_dir, "all", "assets", "amber_all_q_with_object.jsonl"),
        "generative_q_jsonl": os.path.join(out_dir, "generative", "assets", "amber_generative_q.jsonl"),
        "generative_q_with_object_jsonl": os.path.join(out_dir, "generative", "assets", "amber_generative_q_with_object.jsonl"),
        "discriminative_q_jsonl": os.path.join(out_dir, "discriminative", "assets", "amber_discriminative_q.jsonl"),
        "discriminative_q_with_object_jsonl": os.path.join(out_dir, "discriminative", "assets", "amber_discriminative_q_with_object.jsonl"),
    }

    write_jsonl(outputs["all_q_jsonl"], rows_all)
    write_jsonl(outputs["all_q_with_object_jsonl"], rows_all_with_object)
    write_jsonl(outputs["generative_q_jsonl"], rows_gen)
    write_jsonl(outputs["generative_q_with_object_jsonl"], rows_gen_with_object)
    write_jsonl(outputs["discriminative_q_jsonl"], rows_disc)
    write_jsonl(outputs["discriminative_q_with_object_jsonl"], rows_disc_with_object)

    summary = {
        "inputs": {
            "amber_root": amber_root,
            "query_all_json": query_all_json,
            "annotations_json": annotations_json,
        },
        "counts": {
            "n_all": int(len(rows_all)),
            "n_generative": int(len(rows_gen)),
            "n_discriminative": int(len(rows_disc)),
            "by_type": dict(type_counter),
        },
        "outputs": outputs,
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print("[saved]", os.path.join(out_dir, "summary.json"))
    for key in (
        "all_q_with_object_jsonl",
        "generative_q_with_object_jsonl",
        "discriminative_q_with_object_jsonl",
    ):
        print("[saved]", outputs[key])


if __name__ == "__main__":
    main()
