#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path


CATEGORY_OFFSETS = {
    "adversarial": (0, 3000),
    "popular": (3000, 6000),
    "random": (6000, 9000),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rebuild VISTA/EAZY pope_coco category jsonl from pope_9000_gt.csv."
    )
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument(
        "--category",
        type=str,
        required=True,
        choices=["adversarial", "popular", "random"],
    )
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument(
        "--backup_old",
        action="store_true",
        help="If out_jsonl exists, save <out_jsonl>.bak before overwrite.",
    )
    args = ap.parse_args()

    start, end = CATEGORY_OFFSETS[args.category]
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and args.backup_old:
        bak = Path(str(out_path) + ".bak")
        bak.write_bytes(out_path.read_bytes())
        print("[saved]", str(bak))

    rows = []
    with open(args.gt_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["id"])
            if qid < start or qid >= end:
                continue
            rows.append(
                {
                    "question_id": int(row.get("orig_question_id", qid - start + 1)),
                    "image": f"{row['image_id']}.jpg",
                    "text": row["question"],
                    "label": row["answer"].strip().lower(),
                }
            )

    rows.sort(key=lambda x: int(x["question_id"]))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "gt_csv": os.path.abspath(args.gt_csv),
        "category": args.category,
        "out_jsonl": str(out_path.resolve()),
        "count": len(rows),
        "question_id_min": rows[0]["question_id"] if rows else None,
        "question_id_max": rows[-1]["question_id"] if rows else None,
    }
    s_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", str(out_path))
    print("[saved]", str(s_path))
    print("[count]", len(rows))


if __name__ == "__main__":
    main()
