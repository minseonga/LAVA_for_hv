#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def load_gt(path_csv: str, id_col: str, label_col: str) -> Dict[str, dict]:
    out = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            qid = str(r.get(id_col, "")).strip()
            if not qid:
                continue
            out[qid] = r
    return out


def load_pred(path_jsonl: str, pred_text_key: str) -> Dict[str, str]:
    out = {}
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r.get("question_id", "")).strip()
            if not qid:
                continue
            mode = (pred_text_key or "auto").strip().lower()
            if mode == "text":
                txt = r.get("text", "")
            elif mode == "output":
                txt = r.get("output", "")
            else:
                txt = r.get("text", "")
                if not str(txt).strip():
                    txt = r.get("output", "")
                if not str(txt).strip():
                    txt = r.get("answer", "")
            out[qid] = parse_yes_no(txt)
    return out


def load_questions_jsonl(path_jsonl: str) -> Dict[str, dict]:
    out = {}
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r.get("question_id", r.get("id", ""))).strip()
            if not qid:
                continue
            out[qid] = r
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Build balanced 4-way subset from VCS-vs-VGA correctness relation."
    )
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--vcs_pred_jsonl", required=True)
    ap.add_argument("--vga_pred_jsonl", required=True)
    ap.add_argument("--vcs_pred_text_key", default="text", choices=["auto", "text", "output"])
    ap.add_argument("--vga_pred_text_key", default="output", choices=["auto", "text", "output"])
    ap.add_argument("--questions_jsonl", required=True, help="Base questions jsonl to subset.")
    ap.add_argument(
        "--questions_with_object_jsonl",
        default="",
        help="Optional object-augmented questions jsonl; if given, a matched subset is also saved.",
    )
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--label_col", default="answer")
    ap.add_argument("--subset_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    gt = load_gt(args.gt_csv, id_col=args.id_col, label_col=args.label_col)
    vcs = load_pred(args.vcs_pred_jsonl, pred_text_key=args.vcs_pred_text_key)
    vga = load_pred(args.vga_pred_jsonl, pred_text_key=args.vga_pred_text_key)

    groups = defaultdict(list)
    for qid, row in gt.items():
        y = str(row.get(args.label_col, "")).strip().lower()
        if y not in {"yes", "no"}:
            continue
        pv = vcs.get(qid)
        pg = vga.get(qid)
        if pv is None or pg is None:
            continue
        cv = pv == y
        cg = pg == y
        if cv and cg:
            g = "both_correct"
        elif (not cv) and cg:
            g = "vcs_wrong_vga_correct"
        elif cv and (not cg):
            g = "vcs_correct_vga_wrong"
        else:
            g = "both_wrong"
        groups[g].append(qid)

    target_per_group = int(args.subset_size // 4)
    if target_per_group <= 0:
        raise ValueError("subset_size must be >= 4.")

    required_groups = [
        "both_correct",
        "vcs_wrong_vga_correct",
        "vcs_correct_vga_wrong",
        "both_wrong",
    ]
    for g in required_groups:
        n = len(groups[g])
        if n < target_per_group:
            raise RuntimeError(f"Not enough samples in {g}: have {n}, need {target_per_group}")

    chosen = {}
    for g in required_groups:
        chosen[g] = rng.sample(groups[g], k=target_per_group)

    subset_ids = []
    for g in required_groups:
        subset_ids.extend(chosen[g])
    rng.shuffle(subset_ids)

    q_base = load_questions_jsonl(args.questions_jsonl)
    q_obj = (
        load_questions_jsonl(args.questions_with_object_jsonl)
        if args.questions_with_object_jsonl.strip()
        else None
    )

    # Save id+group
    id_rows = []
    id_to_group = {}
    for g, ids in chosen.items():
        for qid in ids:
            id_to_group[qid] = g
    for qid in subset_ids:
        id_rows.append({"id": qid, "group": id_to_group[qid]})

    ids_csv = os.path.join(args.out_dir, "subset_ids.csv")
    with open(ids_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "group"])
        wr.writeheader()
        wr.writerows(id_rows)

    # Save subset gt
    gt_out_csv = os.path.join(args.out_dir, "subset_gt.csv")
    with open(gt_out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(next(iter(gt.values())).keys())
        wr = csv.DictWriter(f, fieldnames=fieldnames + ["group"])
        wr.writeheader()
        for qid in subset_ids:
            row = dict(gt[qid])
            row["group"] = id_to_group[qid]
            wr.writerow(row)

    # Save subset questions
    q_out_jsonl = os.path.join(args.out_dir, "subset_questions.jsonl")
    with open(q_out_jsonl, "w", encoding="utf-8") as f:
        for qid in subset_ids:
            if qid in q_base:
                f.write(json.dumps(q_base[qid], ensure_ascii=False) + "\n")

    q_obj_out_jsonl = ""
    if q_obj is not None:
        q_obj_out_jsonl = os.path.join(args.out_dir, "subset_questions_with_object.jsonl")
        with open(q_obj_out_jsonl, "w", encoding="utf-8") as f:
            for qid in subset_ids:
                if qid in q_obj:
                    f.write(json.dumps(q_obj[qid], ensure_ascii=False) + "\n")

    summary = {
        "inputs": {
            "gt_csv": os.path.abspath(args.gt_csv),
            "vcs_pred_jsonl": os.path.abspath(args.vcs_pred_jsonl),
            "vga_pred_jsonl": os.path.abspath(args.vga_pred_jsonl),
            "questions_jsonl": os.path.abspath(args.questions_jsonl),
            "questions_with_object_jsonl": os.path.abspath(args.questions_with_object_jsonl)
            if args.questions_with_object_jsonl.strip()
            else "",
            "subset_size": int(args.subset_size),
            "seed": int(args.seed),
            "vcs_pred_text_key": args.vcs_pred_text_key,
            "vga_pred_text_key": args.vga_pred_text_key,
        },
        "available_counts": {g: len(groups[g]) for g in required_groups},
        "chosen_counts": {g: len(chosen[g]) for g in required_groups},
        "outputs": {
            "subset_ids_csv": ids_csv,
            "subset_gt_csv": gt_out_csv,
            "subset_questions_jsonl": q_out_jsonl,
            "subset_questions_with_object_jsonl": q_obj_out_jsonl,
            "summary_json": os.path.join(args.out_dir, "summary.json"),
        },
    }
    with open(summary["outputs"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", ids_csv)
    print("[saved]", gt_out_csv)
    print("[saved]", q_out_jsonl)
    if q_obj_out_jsonl:
        print("[saved]", q_obj_out_jsonl)
    print("[saved]", summary["outputs"]["summary_json"])


if __name__ == "__main__":
    main()

