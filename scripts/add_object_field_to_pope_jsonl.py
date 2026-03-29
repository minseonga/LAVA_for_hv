#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PATTERNS = [
    re.compile(r"^Is there an? (.+?) in the image\?$", re.IGNORECASE),
    re.compile(r"^Is there an? (.+?)\?$", re.IGNORECASE),
    re.compile(r"^Is the (.+?) in the image\?$", re.IGNORECASE),
    re.compile(r"^Are there (.+?) in the image\?$", re.IGNORECASE),
]


def extract_object_phrase(question: str):
    q = (question or "").strip()
    for pat in PATTERNS:
        m = pat.match(q)
        if m:
            phrase = m.group(1).strip()
            if phrase:
                return phrase
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Add VGA-compatible `object` field to POPE JSONL questions."
    )
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument(
        "--overwrite_object",
        action="store_true",
        help="Overwrite existing `object` field if present.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any question cannot be parsed for object phrase.",
    )
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    parsed = 0
    kept_existing = 0
    unparsed = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for ln, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1

            question = row.get("question", row.get("text", ""))
            if "question" not in row and question:
                row["question"] = question

            if (not args.overwrite_object) and ("object" in row) and row["object"]:
                kept_existing += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            phrase = extract_object_phrase(question)
            if phrase is None:
                unparsed += 1
                if args.strict:
                    raise ValueError(
                        f"Could not parse object from question at line {ln}: {question}"
                    )
                row["object"] = []
            else:
                parsed += 1
                row["object"] = [phrase]

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_jsonl": str(in_path),
                "output_jsonl": str(out_path),
                "total_rows": total,
                "parsed_rows": parsed,
                "kept_existing_rows": kept_existing,
                "unparsed_rows": unparsed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

