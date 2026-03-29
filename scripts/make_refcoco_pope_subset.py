#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys: List[str] = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_image_id_int(x: Any) -> Optional[int]:
    s = str(x or "").strip()
    if s == "":
        return None
    m = re.findall(r"(\d+)", s)
    if not m:
        return None
    try:
        return int(m[-1])
    except Exception:
        return None


def normalize_ref_sets(x: str) -> List[str]:
    out: List[str] = []
    for t in str(x or "").split(","):
        v = t.strip()
        if v:
            out.append(v)
    return out


def load_refcoco_image_union(refcoco_root: str, ref_sets: Sequence[str]) -> Tuple[Set[int], Dict[int, str], Dict[str, int]]:
    union_ids: Set[int] = set()
    id_to_file: Dict[int, str] = {}
    per_set: Dict[str, int] = {}
    for ds in ref_sets:
        p = os.path.join(refcoco_root, ds, "instances.json")
        if not os.path.isfile(p):
            per_set[ds] = 0
            continue
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        images = obj.get("images", []) if isinstance(obj, dict) else []
        cur = 0
        for im in images:
            iid = extract_image_id_int(im.get("id"))
            if iid is None:
                iid = extract_image_id_int(im.get("file_name"))
            if iid is None:
                continue
            cur += 1
            union_ids.add(int(iid))
            if int(iid) not in id_to_file:
                id_to_file[int(iid)] = str(im.get("file_name") or "")
        per_set[ds] = int(cur)
    return union_ids, id_to_file, per_set


def sample_rows_by_qid(rows: Sequence[Dict[str, Any]], keep_qids: Set[str], qid_key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        qid = str(r.get(qid_key, "")).strip()
        if qid in keep_qids:
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build POPE∩RefCOCO subset; fallback to RefCOCO-only image subset if overlap is empty.")
    ap.add_argument("--pope_question_jsonl", type=str, required=True)
    ap.add_argument("--pope_gt_csv", type=str, default="")
    ap.add_argument("--pope_role_csv", type=str, default="")
    ap.add_argument("--pope_qid_key_jsonl", type=str, default="question_id")
    ap.add_argument("--pope_qid_key_gt", type=str, default="id")
    ap.add_argument("--pope_qid_key_role", type=str, default="id")
    ap.add_argument("--pope_image_key_jsonl", type=str, default="image")
    ap.add_argument("--refcoco_root", type=str, default="/home/kms/data/refcoco_data")
    ap.add_argument("--ref_sets", type=str, default="refcoco,refcoco+,refcocog")
    ap.add_argument("--subset_size", type=int, default=0, help="0 means all overlap samples.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fallback_refonly_size", type=int, default=1000)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(int(args.seed))

    q_rows = read_jsonl(args.pope_question_jsonl)
    ref_sets = normalize_ref_sets(args.ref_sets)
    ref_ids, ref_id_to_file, ref_per_set = load_refcoco_image_union(args.refcoco_root, ref_sets)

    # POPE question candidates by RefCOCO image overlap.
    overlap_q: List[Dict[str, Any]] = []
    for r in q_rows:
        iid = extract_image_id_int(r.get(args.pope_image_key_jsonl))
        if iid is None:
            continue
        if int(iid) in ref_ids:
            overlap_q.append(r)

    if int(args.subset_size) > 0 and len(overlap_q) > int(args.subset_size):
        overlap_q = rng.sample(overlap_q, k=int(args.subset_size))

    overlap_q = sorted(overlap_q, key=lambda x: str(x.get(args.pope_qid_key_jsonl, "")))
    keep_qids = set(str(r.get(args.pope_qid_key_jsonl, "")).strip() for r in overlap_q)

    out_q_jsonl = os.path.join(args.out_dir, "pope_refcoco_overlap_questions.jsonl")
    write_jsonl(out_q_jsonl, overlap_q)

    out_gt_csv = ""
    gt_rows: List[Dict[str, Any]] = []
    if str(args.pope_gt_csv).strip() and os.path.isfile(args.pope_gt_csv):
        gt_rows_all = read_csv(args.pope_gt_csv)
        gt_rows = sample_rows_by_qid(gt_rows_all, keep_qids=keep_qids, qid_key=str(args.pope_qid_key_gt))
        out_gt_csv = os.path.join(args.out_dir, "pope_refcoco_overlap_gt.csv")
        write_csv(out_gt_csv, gt_rows)

    out_role_csv = ""
    role_rows: List[Dict[str, Any]] = []
    if str(args.pope_role_csv).strip() and os.path.isfile(args.pope_role_csv):
        role_rows_all = read_csv(args.pope_role_csv)
        role_rows = sample_rows_by_qid(role_rows_all, keep_qids=keep_qids, qid_key=str(args.pope_qid_key_role))
        out_role_csv = os.path.join(args.out_dir, "pope_refcoco_overlap_role.csv")
        write_csv(out_role_csv, role_rows)

    ids_txt = os.path.join(args.out_dir, "pope_refcoco_overlap_ids.txt")
    with open(ids_txt, "w", encoding="utf-8") as f:
        for qid in sorted(keep_qids):
            f.write(f"{qid}\n")

    # Always save RefCOCO union ids for separate runs.
    union_csv = os.path.join(args.out_dir, "refcoco_union_image_ids.csv")
    union_rows = []
    for iid in sorted(ref_ids):
        union_rows.append({"image_id_int": int(iid), "file_name": str(ref_id_to_file.get(int(iid), ""))})
    write_csv(union_csv, union_rows)

    # Optional fallback subset when overlap is empty or too small.
    refonly_csv = ""
    refonly_rows: List[Dict[str, Any]] = []
    refonly_k = int(max(0, args.fallback_refonly_size))
    if refonly_k > 0:
        pool = list(sorted(ref_ids))
        if len(pool) > refonly_k:
            picks = rng.sample(pool, k=refonly_k)
        else:
            picks = pool
        for iid in sorted(picks):
            refonly_rows.append({"image_id_int": int(iid), "file_name": str(ref_id_to_file.get(int(iid), ""))})
        refonly_csv = os.path.join(args.out_dir, "refcoco_refonly_subset_image_ids.csv")
        write_csv(refonly_csv, refonly_rows)

    summary = {
        "inputs": vars(args),
        "ref_sets": ref_sets,
        "ref_per_set_counts": ref_per_set,
        "counts": {
            "n_pope_questions_in": int(len(q_rows)),
            "n_ref_union_images": int(len(ref_ids)),
            "n_overlap_questions": int(len(overlap_q)),
            "n_overlap_unique_qids": int(len(keep_qids)),
            "n_overlap_gt_rows": int(len(gt_rows)),
            "n_overlap_role_rows": int(len(role_rows)),
            "n_refonly_rows": int(len(refonly_rows)),
        },
        "outputs": {
            "overlap_questions_jsonl": out_q_jsonl,
            "overlap_gt_csv": out_gt_csv,
            "overlap_role_csv": out_role_csv,
            "overlap_ids_txt": ids_txt,
            "ref_union_image_ids_csv": union_csv,
            "refonly_subset_image_ids_csv": refonly_csv,
            "summary_json": os.path.join(args.out_dir, "summary.json"),
        },
        "notes": {
            "overlap_zero_means_current_pope_file_has_no_refcoco-image intersection": bool(len(overlap_q) == 0)
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_q_jsonl)
    if out_gt_csv:
        print("[saved]", out_gt_csv)
    if out_role_csv:
        print("[saved]", out_role_csv)
    print("[saved]", union_csv)
    if refonly_csv:
        print("[saved]", refonly_csv)
    print("[saved]", summary_path)


if __name__ == "__main__":
    main()

