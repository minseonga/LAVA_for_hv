#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Dict, List


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_image_name(image_id: str) -> str:
    v = str(image_id).strip()
    if v.lower().endswith(".jpg"):
        return v
    if v.startswith("COCO_val2014_"):
        return f"{v}.jpg"
    # If numeric id arrives, keep COCO naming.
    if v.isdigit():
        return f"COCO_val2014_{int(v):012d}.jpg"
    return f"{v}.jpg"


def to_native_jsonl_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        qid = int(float(r["id"]))
        out.append(
            {
                "question_id": qid,
                "image": normalize_image_name(r["image_id"]),
                "text": str(r["question"]),
                "label": str(r["answer"]).strip().lower(),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strict 500 POPE subset (125 x 4 groups) + category jsonl files.")
    ap.add_argument("--subset_gt_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--per_group", type=int, default=125)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_csv(os.path.abspath(args.subset_gt_csv))
    rnd = random.Random(int(args.seed))

    by_group: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        g = str(r.get("group", "")).strip()
        by_group.setdefault(g, []).append(r)

    needed = int(args.per_group)
    selected: List[Dict[str, str]] = []
    for g, items in sorted(by_group.items(), key=lambda x: x[0]):
        if len(items) < needed:
            raise RuntimeError(f"group='{g}' has only {len(items)} rows (< {needed})")
        picked = list(items)
        rnd.shuffle(picked)
        selected.extend(picked[:needed])

    # Stable final order by id
    selected.sort(key=lambda r: int(float(r["id"])))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "subset_500_gt.csv")
    write_csv(out_csv, selected)

    # Build per-category jsonl for native POPE scripts.
    cat_map: Dict[str, List[Dict[str, str]]] = {"random": [], "popular": [], "adversarial": []}
    for r in selected:
        c = str(r.get("category", "")).strip().lower()
        if c in cat_map:
            cat_map[c].append(r)

    out_paths: Dict[str, str] = {}
    for c in ("random", "popular", "adversarial"):
        rows_c = cat_map[c]
        jsonl_rows = to_native_jsonl_rows(rows_c)
        p = os.path.join(out_dir, f"coco_pope_{c}_strict500.json")
        write_jsonl(p, jsonl_rows)
        out_paths[c] = p

    summary = {
        "inputs": {
            "subset_gt_csv": os.path.abspath(args.subset_gt_csv),
            "per_group": int(args.per_group),
            "seed": int(args.seed),
        },
        "counts": {
            "n_selected": len(selected),
            "by_group": {g: len([r for r in selected if str(r.get("group", "")).strip() == g]) for g in sorted(by_group)},
            "by_category": {c: len(cat_map[c]) for c in ("random", "popular", "adversarial")},
        },
        "outputs": {
            "subset_500_gt_csv": out_csv,
            "pope_random_jsonl": out_paths["random"],
            "pope_popular_jsonl": out_paths["popular"],
            "pope_adversarial_jsonl": out_paths["adversarial"],
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(summary["outputs"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_csv)
    print("[saved]", out_paths["random"])
    print("[saved]", out_paths["popular"])
    print("[saved]", out_paths["adversarial"])
    print("[saved]", summary["outputs"]["summary_json"])


if __name__ == "__main__":
    main()

