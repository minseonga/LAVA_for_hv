#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple


def infer_category(path: str) -> str:
    name = os.path.basename(path).lower()
    for key in ("adversarial", "popular", "random"):
        if key in name:
            return key
    return "discovery"


def extract_object_phrase(question: str) -> str:
    q = (question or "").strip().lower().rstrip("?").strip()
    m = re.match(r"^is there (?:a|an)\s+(.+?)\s+in the image$", q)
    if m:
        return m.group(1).strip()
    m2 = re.match(r"^is there\s+(.+?)\s+in the image$", q)
    if m2:
        return m2.group(1).strip()
    return ""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GT/question/subset assets from a POPE-style discovery jsonl.")
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--category", type=str, default="")
    ap.add_argument("--group", type=str, default="")
    args = ap.parse_args()

    in_jsonl = os.path.abspath(args.in_jsonl)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows_in = read_jsonl(in_jsonl)
    category = str(args.category).strip() or infer_category(in_jsonl)
    group = str(args.group).strip() or category

    gt_rows: List[Dict[str, Any]] = []
    subset_rows: List[Dict[str, Any]] = []
    q_rows: List[Dict[str, Any]] = []
    q_obj_rows: List[Dict[str, Any]] = []

    for idx, rec in enumerate(rows_in):
        qid = str(rec.get("question_id", rec.get("id", idx))).strip()
        if qid == "":
            qid = str(idx)
        image = str(rec.get("image", "")).strip()
        text = str(rec.get("text", rec.get("question", ""))).strip()
        label = str(rec.get("label", rec.get("answer", ""))).strip().lower()
        if image == "" or text == "" or label not in {"yes", "no"}:
            continue

        obj = extract_object_phrase(text)

        gt_rows.append(
            {
                "id": qid,
                "answer": label,
                "category": category,
                "image_id": image,
                "question": text,
                "orig_question_id": qid,
                "group": group,
            }
        )
        subset_rows.append({"id": qid, "group": group})

        base_q = {
            "question_id": qid,
            "id": qid,
            "image": image,
            "text": text,
            "question": text,
        }
        q_rows.append(dict(base_q))
        if obj != "":
            q_obj_rows.append({**base_q, "object": [obj]})
        else:
            q_obj_rows.append(dict(base_q))

    gt_csv = os.path.join(out_dir, "discovery_gt.csv")
    subset_csv = os.path.join(out_dir, "discovery_subset_ids.csv")
    q_jsonl = os.path.join(out_dir, "discovery_q.jsonl")
    q_obj_jsonl = os.path.join(out_dir, "discovery_q_with_object.jsonl")
    summary_json = os.path.join(out_dir, "summary.json")

    write_csv(
        gt_csv,
        gt_rows,
        ["id", "answer", "category", "image_id", "question", "orig_question_id", "group"],
    )
    write_csv(subset_csv, subset_rows, ["id", "group"])
    write_jsonl(q_jsonl, q_rows)
    write_jsonl(q_obj_jsonl, q_obj_rows)

    summary = {
        "inputs": {
            "in_jsonl": in_jsonl,
            "category": category,
            "group": group,
        },
        "counts": {
            "n_rows": int(len(gt_rows)),
            "n_with_object": int(sum(1 for r in q_obj_rows if "object" in r)),
        },
        "outputs": {
            "gt_csv": gt_csv,
            "subset_ids_csv": subset_csv,
            "q_jsonl": q_jsonl,
            "q_with_object_jsonl": q_obj_jsonl,
            "summary_json": summary_json,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", gt_csv)
    print("[saved]", subset_csv)
    print("[saved]", q_jsonl)
    print("[saved]", q_obj_jsonl)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
