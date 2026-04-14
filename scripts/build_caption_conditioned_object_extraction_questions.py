#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from extract_generative_semantic_pairwise_features import read_prediction_map


DEFAULT_PROMPT_TEMPLATE = """Task: Extract ONLY the concrete, visually verifiable physical objects from the given image caption.
Rule 1: Return a comma-separated list of singular physical object nouns.
Rule 2: Only include objects explicitly mentioned in the caption.
Rule 3: DO NOT include abstract concepts, actions, scene descriptions, emotions, states, relationships, attributes, colors, or numbers.
Rule 4: Map visually concrete people roles to "person" (e.g., passenger, man, woman, child, player).
Rule 5: If an object is a compound noun, keep it together (e.g., dining table, traffic light, cell phone, brake pedal).
Rule 6: If no concrete physical object is mentioned, return "none".

Examples:
Caption: "An elegant glassware arrangement on a table."
Objects: glassware, table

Caption: "A passenger waiting beside a car near the brake pedal."
Objects: person, car, brake pedal

Caption: "A person playing a nintendo wii game with a remote on the couch."
Objects: person, remote, couch

Caption: "A beautiful scene with a sense of unity and togetherness."
Objects: none

Caption: "{caption}"
Objects:"""


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("question_id") or row.get("image_id") or row.get("id") or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def main() -> None:
    ap = argparse.ArgumentParser(description="Build caption-conditioned object extraction prompts.")
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    ap.add_argument("--pred_text_key", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default="caption_conditioned_object_extraction")
    args = ap.parse_args()

    questions = read_jsonl(args.question_file, limit=int(args.limit))
    pred = read_prediction_map(os.path.abspath(args.pred_jsonl), str(args.pred_text_key))
    out: List[Dict[str, Any]] = []
    n_missing = 0

    for idx, q in enumerate(questions):
        sid = safe_id(q) or str(idx)
        if sid not in pred:
            n_missing += 1
            caption = ""
        else:
            caption = str(pred[sid].get("text", "")).strip()
        prompt = str(args.prompt_template).replace("{caption}", caption)
        out.append(
            {
                "question_id": sid,
                "image_id": sid,
                "image": q.get("image", pred.get(sid, {}).get("image", "")),
                "question": prompt,
                "text": prompt,
                "label": "",
                "split": q.get("split", ""),
                "category": str(args.category),
            }
        )

    write_jsonl(args.out_jsonl, out)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "pred_jsonl": os.path.abspath(args.pred_jsonl),
                    "limit": int(args.limit),
                    "category": str(args.category),
                },
                "counts": {"n_rows": len(out), "n_missing_predictions": n_missing},
                "outputs": {"out_jsonl": os.path.abspath(args.out_jsonl)},
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
