#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_TEMPLATE = """Describe the image in one accurate, concise caption.

The previous caption may have mentioned uncertain objects: {objects}.
Do not mention those objects unless they are clearly visible in the image.
Focus only on concrete physical objects that are visually grounded."""


def read_jsonl(path: str, *, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    import csv

    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("question_id") or row.get("image_id") or row.get("id") or "").strip()
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def split_objects(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in str(text or "").split("|"):
        obj = raw.strip()
        if not obj:
            continue
        key = obj.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build negative-object re-caption prompts from oracle or predicted risky objects.")
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_selected_ids_json", default="")
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--object_col", default="int_only_hallucinated_unique")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_objects", type=int, default=4)
    ap.add_argument("--selected_only", action="store_true")
    ap.add_argument("--empty_object_policy", choices=["original_prompt", "generic_recaption", "skip"], default="original_prompt")
    args = ap.parse_args()

    questions = read_jsonl(args.question_file, limit=int(args.limit))
    oracle_by_id = {safe_id(row): row for row in read_csv_rows(args.oracle_rows_csv)}
    out_rows: List[Dict[str, Any]] = []
    selected_ids: List[str] = []
    n_with_objects = 0
    n_missing_oracle = 0

    for idx, q in enumerate(questions):
        sid = safe_id(q) or str(idx)
        oracle = oracle_by_id.get(sid)
        if oracle is None:
            n_missing_oracle += 1
        objects = split_objects(oracle.get(args.object_col, "") if oracle is not None else "")[: int(args.max_objects)]
        if objects:
            n_with_objects += 1
            selected_ids.append(sid)
            object_text = ", ".join(objects)
            prompt = str(args.template).replace("{objects}", object_text)
        elif args.selected_only or args.empty_object_policy == "skip":
            continue
        elif args.empty_object_policy == "generic_recaption":
            prompt = "Describe the image in one accurate, concise caption. Focus only on concrete physical objects that are visually grounded."
        else:
            prompt = str(q.get("text", q.get("question", ""))).strip()

        out_rows.append(
            {
                "question_id": sid,
                "image_id": sid,
                "image": q.get("image", ""),
                "question": prompt,
                "text": prompt,
                "label": "",
                "split": q.get("split", ""),
                "category": "negative_object_recaption",
                "negative_objects": "|".join(objects),
                "negative_object_count": len(objects),
            }
        )

    write_jsonl(args.out_jsonl, out_rows)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if str(args.out_selected_ids_json or "").strip():
        write_json(args.out_selected_ids_json, {"selected_ids": selected_ids})
        print("[saved]", os.path.abspath(args.out_selected_ids_json))
    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
                    "object_col": str(args.object_col),
                    "limit": int(args.limit),
                    "max_objects": int(args.max_objects),
                    "selected_only": bool(args.selected_only),
                    "empty_object_policy": str(args.empty_object_policy),
                },
                "counts": {
                    "n_questions": len(questions),
                    "n_output_rows": len(out_rows),
                    "n_with_negative_objects": n_with_objects,
                    "n_missing_oracle": n_missing_oracle,
                },
                "outputs": {
                    "out_jsonl": os.path.abspath(args.out_jsonl),
                    "out_selected_ids_json": os.path.abspath(args.out_selected_ids_json) if args.out_selected_ids_json else "",
                },
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
