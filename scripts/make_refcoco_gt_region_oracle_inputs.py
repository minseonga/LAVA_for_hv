#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_list(s: str) -> List[str]:
    out: List[str] = []
    for x in str(s or "").split(","):
        t = x.strip()
        if t:
            out.append(t)
    return out


def infer_refs_file(dataset: str, refcoco_root: str, prefer_split: str = "unc") -> str:
    base = os.path.join(refcoco_root, dataset)
    cands: List[str] = []
    if dataset in {"refcoco", "refcoco+"}:
        if prefer_split == "google":
            cands = [os.path.join(base, "refs(google).p"), os.path.join(base, "refs(unc).p")]
        else:
            cands = [os.path.join(base, "refs(unc).p"), os.path.join(base, "refs(google).p")]
    elif dataset == "refcocog":
        if prefer_split == "google":
            cands = [os.path.join(base, "refs(google).p"), os.path.join(base, "refs(umd).p")]
        else:
            cands = [os.path.join(base, "refs(umd).p"), os.path.join(base, "refs(google).p")]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    for p in cands:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No refs pickle found for {dataset} under {base}")


def load_ref_data(dataset: str, refcoco_root: str, refs_file: str = "", refs_split_pref: str = "unc") -> Dict[str, Any]:
    ds_dir = os.path.join(refcoco_root, dataset)
    inst_json = os.path.join(ds_dir, "instances.json")
    if not os.path.isfile(inst_json):
        raise FileNotFoundError(inst_json)
    if refs_file.strip() == "":
        refs_file = infer_refs_file(dataset, refcoco_root, refs_split_pref)
    if not os.path.isfile(refs_file):
        raise FileNotFoundError(refs_file)

    with open(inst_json, "r", encoding="utf-8") as f:
        inst = json.load(f)
    with open(refs_file, "rb") as f:
        refs = pickle.load(f, encoding="latin1")

    images = inst.get("images", []) if isinstance(inst, dict) else []
    anns = inst.get("annotations", []) if isinstance(inst, dict) else []
    cats = inst.get("categories", []) if isinstance(inst, dict) else []

    img_by_id: Dict[int, Dict[str, Any]] = {}
    for im in images:
        try:
            iid = int(im.get("id"))
        except Exception:
            continue
        img_by_id[iid] = im

    ann_by_id: Dict[int, Dict[str, Any]] = {}
    img_to_anns: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns:
        try:
            aid = int(a.get("id"))
            iid = int(a.get("image_id"))
        except Exception:
            continue
        ann_by_id[aid] = a
        img_to_anns.setdefault(iid, []).append(a)

    cat_by_id: Dict[int, str] = {}
    for c in cats:
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        cat_by_id[cid] = str(c.get("name") or f"cat_{cid}")

    img_to_catset: Dict[int, Set[int]] = {}
    for iid, arr in img_to_anns.items():
        img_to_catset[iid] = set(int(a.get("category_id")) for a in arr if a.get("category_id") is not None)

    return {
        "dataset": dataset,
        "instances_json": inst_json,
        "refs_file": refs_file,
        "refs": refs if isinstance(refs, list) else [],
        "img_by_id": img_by_id,
        "ann_by_id": ann_by_id,
        "img_to_anns": img_to_anns,
        "img_to_catset": img_to_catset,
        "cat_by_id": cat_by_id,
    }


def _cell_rect(gx: int, gy: int, cell_w: float, cell_h: float) -> Tuple[float, float, float, float]:
    x0 = gx * cell_w
    y0 = gy * cell_h
    x1 = (gx + 1) * cell_w
    y1 = (gy + 1) * cell_h
    return x0, y0, x1, y1


def _inter_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def bbox_to_patch_scores(
    bbox_xywh: Sequence[float],
    image_w: int,
    image_h: int,
    grid_size: int,
) -> Dict[int, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(image_w), x + w), min(float(image_h), y + h)
    if x1 <= x0 or y1 <= y0:
        return {}
    box = (x0, y0, x1, y1)
    cell_w = float(image_w) / float(grid_size)
    cell_h = float(image_h) / float(grid_size)
    out: Dict[int, float] = {}
    for gy in range(int(grid_size)):
        for gx in range(int(grid_size)):
            crect = _cell_rect(gx, gy, cell_w, cell_h)
            ia = _inter_area(box, crect)
            if ia <= 0.0:
                continue
            idx = int(gy * grid_size + gx)
            out[idx] = float(ia)
    return out


def merge_patch_scores(maps: Sequence[Dict[int, float]], mode: str = "sum") -> Dict[int, float]:
    out: Dict[int, float] = {}
    for m in maps:
        for k, v in m.items():
            if mode == "max":
                out[k] = max(float(out.get(k, 0.0)), float(v))
            else:
                out[k] = float(out.get(k, 0.0)) + float(v)
    return out


def topk_from_scores(scores: Dict[int, float], topk: int) -> List[int]:
    if not scores:
        return []
    arr = sorted(scores.items(), key=lambda x: (float(x[1]), -int(x[0])), reverse=True)
    if int(topk) > 0:
        arr = arr[: int(topk)]
    return [int(k) for k, _ in arr]


def sample_absent_category(cat_all: Set[int], cat_present: Set[int], rng: random.Random) -> Optional[int]:
    cand = list(sorted(set(cat_all) - set(cat_present)))
    if len(cand) == 0:
        return None
    return int(rng.choice(cand))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate RefCOCO yes/no QA + GT-region role CSV for oracle guidance.")
    ap.add_argument("--refcoco_root", type=str, default="/home/kms/data/refcoco_data")
    ap.add_argument("--dataset", type=str, default="refcoco", choices=["refcoco", "refcoco+", "refcocog"])
    ap.add_argument("--refs_file", type=str, default="")
    ap.add_argument("--refs_split_pref", type=str, default="unc", choices=["unc", "google", "umd"])
    ap.add_argument("--splits", type=str, default="val,testA,testB,test")
    ap.add_argument("--image_folder", type=str, default="/home/kms/data/images/mscoco/images/train2014")
    ap.add_argument("--ensure_image_exists", action="store_true")
    ap.add_argument("--subset_size", type=int, default=1000, help="Total questions (yes+no).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid_size", type=int, default=24)
    ap.add_argument("--supportive_topk", type=int, default=16)
    ap.add_argument("--assertive_topk", type=int, default=16)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(int(args.seed))
    data = load_ref_data(
        dataset=str(args.dataset),
        refcoco_root=str(args.refcoco_root),
        refs_file=str(args.refs_file),
        refs_split_pref=str(args.refs_split_pref),
    )

    refs: List[Dict[str, Any]] = data["refs"]
    img_by_id: Dict[int, Dict[str, Any]] = data["img_by_id"]
    ann_by_id: Dict[int, Dict[str, Any]] = data["ann_by_id"]
    img_to_anns: Dict[int, List[Dict[str, Any]]] = data["img_to_anns"]
    img_to_catset: Dict[int, Set[int]] = data["img_to_catset"]
    cat_by_id: Dict[int, str] = data["cat_by_id"]
    cat_all: Set[int] = set(cat_by_id.keys())
    split_set = set(parse_list(args.splits))

    # filter refs by requested splits + valid annotations
    cands: List[Dict[str, Any]] = []
    for r in refs:
        sp = str(r.get("split") or "")
        if split_set and sp not in split_set:
            continue
        try:
            ann_id = int(r.get("ann_id"))
            image_id = int(r.get("image_id"))
            ref_id = int(r.get("ref_id"))
            category_id = int(r.get("category_id"))
        except Exception:
            continue
        if ann_id not in ann_by_id or image_id not in img_by_id:
            continue
        im = img_by_id[image_id]
        file_name = str(im.get("file_name") or "")
        if file_name == "":
            continue
        if bool(args.ensure_image_exists):
            p = os.path.join(args.image_folder, file_name)
            if not os.path.isfile(p):
                continue
        cands.append(
            {
                "split": sp,
                "ann_id": ann_id,
                "image_id": image_id,
                "ref_id": ref_id,
                "category_id": category_id,
                "file_name": file_name,
            }
        )

    rng.shuffle(cands)
    target_total = int(max(1, args.subset_size))
    target_pairs = target_total // 2

    q_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []
    role_rows: List[Dict[str, Any]] = []
    used_qid: Set[str] = set()
    n_pairs = 0

    for c in cands:
        if n_pairs >= target_pairs:
            break
        ann_id = int(c["ann_id"])
        image_id = int(c["image_id"])
        ref_id = int(c["ref_id"])
        category_id = int(c["category_id"])
        file_name = str(c["file_name"])
        split = str(c["split"])

        ann = ann_by_id.get(ann_id)
        im = img_by_id.get(image_id)
        if ann is None or im is None:
            continue
        w = int(im.get("width") or 0)
        h = int(im.get("height") or 0)
        if w <= 0 or h <= 0:
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        present = img_to_catset.get(image_id, set())
        neg_cid = sample_absent_category(cat_all, present, rng)
        if neg_cid is None:
            continue

        pos_cat_name = str(cat_by_id.get(category_id, f"cat_{category_id}"))
        neg_cat_name = str(cat_by_id.get(int(neg_cid), f"cat_{neg_cid}"))
        q_pos = f"Is there a {pos_cat_name} in the image?"
        q_neg = f"Is there a {neg_cat_name} in the image?"

        qid_pos = f"{args.dataset}_{split}_{ref_id}_pos"
        qid_neg = f"{args.dataset}_{split}_{ref_id}_neg"
        if qid_pos in used_qid or qid_neg in used_qid:
            continue
        used_qid.add(qid_pos)
        used_qid.add(qid_neg)

        q_rows.append({"question_id": qid_pos, "image": file_name, "text": q_pos})
        q_rows.append({"question_id": qid_neg, "image": file_name, "text": q_neg})

        gt_rows.append(
            {
                "id": qid_pos,
                "answer": "yes",
                "dataset": args.dataset,
                "split": split,
                "image_id": image_id,
                "file_name": file_name,
                "ann_id": ann_id,
                "ref_id": ref_id,
                "category_id": category_id,
                "category_name": pos_cat_name,
                "question": q_pos,
            }
        )
        gt_rows.append(
            {
                "id": qid_neg,
                "answer": "no",
                "dataset": args.dataset,
                "split": split,
                "image_id": image_id,
                "file_name": file_name,
                "ann_id": "",
                "ref_id": ref_id,
                "category_id": int(neg_cid),
                "category_name": neg_cat_name,
                "question": q_neg,
            }
        )

        # supportive for positive: target bbox
        sup_scores = bbox_to_patch_scores(bbox_xywh=bbox, image_w=w, image_h=h, grid_size=int(args.grid_size))
        sup_idx = topk_from_scores(sup_scores, topk=int(args.supportive_topk))
        # assertive: non-target boxes for positive / all boxes for negative
        anns = img_to_anns.get(image_id, [])
        ass_boxes_pos = []
        ass_boxes_neg = []
        for a in anns:
            bb = a.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            aid = int(a.get("id"))
            if aid != ann_id:
                ass_boxes_pos.append(bb)
            ass_boxes_neg.append(bb)
        ass_scores_pos = merge_patch_scores(
            [bbox_to_patch_scores(bb, image_w=w, image_h=h, grid_size=int(args.grid_size)) for bb in ass_boxes_pos], mode="sum"
        )
        ass_scores_neg = merge_patch_scores(
            [bbox_to_patch_scores(bb, image_w=w, image_h=h, grid_size=int(args.grid_size)) for bb in ass_boxes_neg], mode="sum"
        )

        ass_idx_pos = [p for p in topk_from_scores(ass_scores_pos, topk=int(args.assertive_topk)) if p not in set(sup_idx)]
        ass_idx_neg = topk_from_scores(ass_scores_neg, topk=int(args.assertive_topk))

        for rk, p in enumerate(sup_idx):
            role_rows.append(
                {
                    "id": qid_pos,
                    "candidate_patch_idx": int(p),
                    "candidate_rank": int(rk),
                    "role_label": "supportive",
                    "image_id": image_id,
                    "ann_id": ann_id,
                    "source": "gt_bbox",
                }
            )
        for rk, p in enumerate(ass_idx_pos):
            role_rows.append(
                {
                    "id": qid_pos,
                    "candidate_patch_idx": int(p),
                    "candidate_rank": int(rk),
                    "role_label": "harmful",
                    "image_id": image_id,
                    "ann_id": ann_id,
                    "source": "nontarget_bbox",
                }
            )
        for rk, p in enumerate(ass_idx_neg):
            role_rows.append(
                {
                    "id": qid_neg,
                    "candidate_patch_idx": int(p),
                    "candidate_rank": int(rk),
                    "role_label": "harmful",
                    "image_id": image_id,
                    "ann_id": "",
                    "source": "all_bbox",
                }
            )
        n_pairs += 1

    # if odd target requested, append one extra positive if possible
    if target_total % 2 == 1 and len(cands) > 0 and len(q_rows) < target_total:
        for c in cands:
            ann_id = int(c["ann_id"])
            image_id = int(c["image_id"])
            ref_id = int(c["ref_id"])
            category_id = int(c["category_id"])
            file_name = str(c["file_name"])
            split = str(c["split"])
            qid_pos = f"{args.dataset}_{split}_{ref_id}_pos_extra"
            if qid_pos in used_qid:
                continue
            pos_cat_name = str(cat_by_id.get(category_id, f"cat_{category_id}"))
            q_pos = f"Is there a {pos_cat_name} in the image?"
            q_rows.append({"question_id": qid_pos, "image": file_name, "text": q_pos})
            gt_rows.append(
                {
                    "id": qid_pos,
                    "answer": "yes",
                    "dataset": args.dataset,
                    "split": split,
                    "image_id": image_id,
                    "file_name": file_name,
                    "ann_id": ann_id,
                    "ref_id": ref_id,
                    "category_id": category_id,
                    "category_name": pos_cat_name,
                    "question": q_pos,
                }
            )
            break

    if int(args.subset_size) > 0 and len(q_rows) > int(args.subset_size):
        q_rows = q_rows[: int(args.subset_size)]
        keep_ids = set(str(r.get("question_id")) for r in q_rows)
        gt_rows = [r for r in gt_rows if str(r.get("id")) in keep_ids]
        role_rows = [r for r in role_rows if str(r.get("id")) in keep_ids]

    out_q = os.path.join(args.out_dir, f"{args.dataset}_qa_{len(q_rows)}.jsonl")
    out_gt = os.path.join(args.out_dir, f"{args.dataset}_gt_{len(gt_rows)}.csv")
    out_role = os.path.join(args.out_dir, f"{args.dataset}_gt_region_role_{len(role_rows)}.csv")
    write_jsonl(out_q, q_rows)
    write_csv(out_gt, gt_rows)
    write_csv(out_role, role_rows)

    gt_yes = sum(1 for r in gt_rows if str(r.get("answer")).lower() == "yes")
    gt_no = sum(1 for r in gt_rows if str(r.get("answer")).lower() == "no")
    summary = {
        "inputs": vars(args),
        "resolved": {
            "instances_json": data["instances_json"],
            "refs_file": data["refs_file"],
        },
        "counts": {
            "n_refs_total": int(len(refs)),
            "n_candidates_after_filter": int(len(cands)),
            "n_questions": int(len(q_rows)),
            "n_gt_rows": int(len(gt_rows)),
            "n_role_rows": int(len(role_rows)),
            "n_yes": int(gt_yes),
            "n_no": int(gt_no),
        },
        "outputs": {
            "questions_jsonl": out_q,
            "gt_csv": out_gt,
            "gt_region_role_csv": out_role,
            "summary_json": os.path.join(args.out_dir, "summary.json"),
        },
    }
    out_summary = os.path.join(args.out_dir, "summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_q)
    print("[saved]", out_gt)
    print("[saved]", out_role)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()
