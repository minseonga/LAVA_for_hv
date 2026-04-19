#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_yes_no(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip().lower() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def sid(value: object) -> str:
    return str(value or "").strip()


def load_questions(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = sid(row.get("question_id", row.get("id")))
            if key:
                out[key] = row
    return out


def load_gt(path: str, id_col: str, label_col: str, group_col: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            key = sid(row.get(id_col))
            label = sid(row.get(label_col)).lower()
            if key and label in {"yes", "no"}:
                out[key] = {"label": label, "category": sid(row.get(group_col))}
    return out


def load_pred(path: str, pred_key: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = sid(row.get("question_id", row.get("id")))
            if not key or key.lower() in {"none", "null", "nan"}:
                continue
            if pred_key == "auto":
                text = row.get("text", "") or row.get("output", "") or row.get("answer", "")
            else:
                text = row.get(pred_key, "")
            out[key] = {"text": str(text or ""), "label": parse_yes_no(str(text or ""))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize a POPE diagnostic subset around method-vs-baseline transitions.")
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--label_col", default="answer")
    ap.add_argument("--group_col", default="category")
    ap.add_argument("--baseline_pred_key", default="auto")
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--mode", default="changed", choices=["changed", "changed_plus_neutral", "all_harm_balanced"])
    ap.add_argument("--max_per_bucket", type=int, default=0)
    ap.add_argument("--neutral_per_changed", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    questions = load_questions(args.question_file)
    gt = load_gt(args.gt_csv, args.id_col, args.label_col, args.group_col)
    baseline = load_pred(args.baseline_pred_jsonl, args.baseline_pred_key)
    intervention = load_pred(args.intervention_pred_jsonl, args.intervention_pred_key)

    buckets: Dict[str, List[str]] = defaultdict(list)
    rows: Dict[str, Dict[str, Any]] = {}
    for key, g in gt.items():
        if key not in questions:
            continue
        b = baseline.get(key, {})
        m = intervention.get(key, {})
        bl = b.get("label", "")
        ml = m.get("label", "")
        if bl not in {"yes", "no"} or ml not in {"yes", "no"}:
            continue
        bc = int(bl == g["label"])
        mc = int(ml == g["label"])
        if bc and mc:
            outcome = "both_correct"
        elif (not bc) and (not mc):
            outcome = "both_wrong"
        elif (not bc) and mc:
            outcome = "help"
        else:
            outcome = "harm"
        transition = f"{bl}->{ml}"
        rows[key] = {
            "transition": transition,
            "outcome": outcome,
            "category": g["category"],
        }
        buckets[f"{transition}:{outcome}"].append(key)

    selected: List[str] = []
    if args.mode == "changed":
        for bucket, ids in buckets.items():
            transition = bucket.split(":", 1)[0]
            if transition in {"yes->no", "no->yes"}:
                rng.shuffle(ids)
                selected.extend(ids[: int(args.max_per_bucket) or len(ids)])
    elif args.mode == "changed_plus_neutral":
        changed_ids: List[str] = []
        for bucket, ids in buckets.items():
            transition = bucket.split(":", 1)[0]
            if transition in {"yes->no", "no->yes"}:
                rng.shuffle(ids)
                take = ids[: int(args.max_per_bucket) or len(ids)]
                selected.extend(take)
                changed_ids.extend(take)
        neutral_pool = buckets.get("yes->yes:both_correct", []) + buckets.get("no->no:both_correct", [])
        rng.shuffle(neutral_pool)
        selected.extend(neutral_pool[: int(len(changed_ids) * float(args.neutral_per_changed))])
    else:
        for bucket, ids in buckets.items():
            outcome = bucket.split(":", 1)[1]
            if outcome in {"harm", "help"}:
                rng.shuffle(ids)
                selected.extend(ids[: int(args.max_per_bucket) or len(ids)])

    selected = sorted(set(selected), key=lambda x: int(x) if x.isdigit() else x)
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for key in selected:
            f.write(json.dumps(questions[key], ensure_ascii=False) + "\n")

    summary = {
        "n_selected": len(selected),
        "mode": args.mode,
        "selected_counts": dict(sorted((k, sum(1 for sid_ in selected if rows[sid_]["transition"] + ":" + rows[sid_]["outcome"] == k)) for k in buckets)),
        "source_counts": {k: len(v) for k, v in sorted(buckets.items())},
        "out_jsonl": args.out_jsonl,
    }
    if args.out_summary_json:
        Path(args.out_summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("n_selected", len(selected))
    print("out", args.out_jsonl)
    print("selected_counts")
    for key, value in summary["selected_counts"].items():
        if value:
            print(key, value)


if __name__ == "__main__":
    main()
