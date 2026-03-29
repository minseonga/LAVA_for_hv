#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def parse_bool(x: Any) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def norm_text(x: Any) -> str:
    s = str("" if x is None else x).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def singularize(x: str) -> str:
    s = norm_text(x)
    if s.endswith("ies") and len(s) > 3:
        return s[:-3] + "y"
    if s.endswith("es") and len(s) > 2:
        return s[:-2]
    if s.endswith("s") and len(s) > 1:
        return s[:-1]
    return s


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def parse_idx_json(x: Any, k: int) -> List[int]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    out: List[int] = []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            for z in arr:
                try:
                    out.append(int(z))
                except Exception:
                    pass
    except Exception:
        parts = [p for p in s.split("|") if p.strip() != ""]
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                pass
    if int(k) > 0:
        out = out[: int(k)]
    return out


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = area_a + area_b - inter
    if den <= 0.0:
        return 0.0
    return float(inter / den)


def patch_to_xyxy(pidx: int, grid: int, w: int, h: int) -> Tuple[float, float, float, float]:
    r = int(pidx) // int(grid)
    c = int(pidx) % int(grid)
    x0 = float(c) * float(w) / float(grid)
    x1 = float(c + 1) * float(w) / float(grid)
    y0 = float(r) * float(h) / float(grid)
    y1 = float(r + 1) * float(h) / float(grid)
    return x0, y0, x1, y1


def build_category_map(categories: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for c in categories:
        cid = int(c.get("id"))
        nm = norm_text(c.get("name"))
        if nm != "":
            name_to_id[nm] = cid
            name_to_id[singularize(nm)] = cid
            id_to_name[cid] = nm

    # common aliases used in POPE/GQA style object words
    aliases = {
        "bike": "bicycle",
        "motorbike": "motorcycle",
        "cellphone": "cell phone",
        "cell phone": "cell phone",
        "phone": "cell phone",
        "tv": "tv",
        "television": "tv",
        "sofa": "couch",
        "fridge": "refrigerator",
        "trafficlight": "traffic light",
        "stoplight": "traffic light",
        "pottedplant": "potted plant",
        "airplane": "airplane",
        "plane": "airplane",
    }
    for k, v in aliases.items():
        kk = norm_text(k)
        vv = norm_text(v)
        if vv in name_to_id:
            name_to_id[kk] = name_to_id[vv]
            name_to_id[singularize(kk)] = name_to_id[vv]
    return name_to_id, id_to_name


def object_phrase_to_cat_ids(obj_phrase: str, name_to_id: Dict[str, int]) -> List[int]:
    op = norm_text(obj_phrase)
    if op == "":
        return []
    cands = [op, singularize(op)]
    toks = [t for t in op.split(" ") if t != ""]
    if len(toks) > 0:
        cands.append(toks[-1])
        cands.append(singularize(toks[-1]))
    for c in cands:
        if c in name_to_id:
            return [int(name_to_id[c])]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze semantic identity of selected patches on POPE traces.")
    ap.add_argument("--trace_csv", type=str, required=True, help="per_layer_yes_trace.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--pos_col", type=str, default="is_fp_hallucination")
    ap.add_argument("--neg_col", type=str, default="is_tp_yes")
    ap.add_argument("--idx_col", type=str, default="yes_sim_objpatch_topk_idx_global_json")
    ap.add_argument("--object_col", type=str, default="object_phrase")
    ap.add_argument("--iou_thr_target", type=float, default=0.1)
    ap.add_argument("--iou_thr_distractor", type=float, default=0.1)
    ap.add_argument(
        "--instances_json",
        type=str,
        default="",
        help="COCO-format instances json (optional; needed only for box-overlap semantic identity).",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = read_csv(os.path.abspath(args.trace_csv))
    if len(rows) == 0:
        raise RuntimeError("No rows in trace_csv.")

    instances_raw = str(args.instances_json or "").strip()
    instances_path = os.path.abspath(instances_raw) if instances_raw != "" else ""
    use_instances = bool(instances_path != "" and os.path.isfile(instances_path))
    coco = {}
    img_meta_by_stem: Dict[str, Dict[str, Any]] = {}
    anns_by_imgid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}

    if use_instances:
        with open(instances_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        for im in coco.get("images", []):
            stem = str(im.get("file_name") or "")
            stem = os.path.splitext(stem)[0]
            if stem != "":
                img_meta_by_stem[stem] = im
        for ann in coco.get("annotations", []):
            try:
                imgid = int(ann.get("image_id"))
                anns_by_imgid[imgid].append(ann)
            except Exception:
                pass
        name_to_id, id_to_name = build_category_map(coco.get("categories", []))

    per_rows: List[Dict[str, Any]] = []
    for r in rows:
        li = safe_float(r.get("block_layer_idx"))
        if li is None or int(li) != int(args.layer):
            continue
        is_pos = parse_bool(r.get(args.pos_col))
        is_neg = parse_bool(r.get(args.neg_col))
        if not (is_pos or is_neg):
            continue
        image_id = str(r.get("image_id") or "").strip()
        sid = str(r.get("id") or "").strip()
        obj = str(r.get(args.object_col) or "").strip()
        idx = parse_idx_json(r.get(args.idx_col), int(args.topk))
        if sid == "" or image_id == "" or len(idx) == 0:
            continue

        nvis = int(safe_float(r.get("n_visual_tokens")) or 576)
        grid = int(round(math.sqrt(float(nvis))))
        if grid * grid != nvis:
            grid = 24

        row_out: Dict[str, Any] = {
            "id": sid,
            "image_id": image_id,
            "group": ("fp_hall" if is_pos else "tp_yes"),
            "object_phrase": obj,
            "topk_idx_json": json.dumps([int(x) for x in idx], ensure_ascii=False),
            "top1_idx": int(idx[0]),
            "grid": int(grid),
            "has_instances": bool(use_instances),
            "matched_category_id": None,
            "matched_category_name": None,
            "top1_iou_target_max": None,
            "top1_iou_distractor_max": None,
            "topk_iou_target_max": None,
            "topk_iou_distractor_max": None,
            "semantic_identity": "unknown",
        }

        if use_instances:
            im = img_meta_by_stem.get(image_id)
            if im is not None:
                w = int(im.get("width"))
                h = int(im.get("height"))
                imgid = int(im.get("id"))
                anns = anns_by_imgid.get(imgid, [])
                target_cat_ids = object_phrase_to_cat_ids(obj, name_to_id)
                if len(target_cat_ids) > 0:
                    row_out["matched_category_id"] = int(target_cat_ids[0])
                    row_out["matched_category_name"] = id_to_name.get(int(target_cat_ids[0]))
                target_boxes: List[Tuple[float, float, float, float]] = []
                other_boxes: List[Tuple[float, float, float, float]] = []
                for a in anns:
                    bbox = a.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue
                    x, y, bw, bh = [float(v) for v in bbox]
                    bxyxy = (x, y, x + bw, y + bh)
                    cid = int(a.get("category_id"))
                    if len(target_cat_ids) > 0 and cid in target_cat_ids:
                        target_boxes.append(bxyxy)
                    else:
                        other_boxes.append(bxyxy)

                top1_box = patch_to_xyxy(int(idx[0]), int(grid), int(w), int(h))
                topk_boxes = [patch_to_xyxy(int(p), int(grid), int(w), int(h)) for p in idx]
                t1_t = max([iou_xyxy(top1_box, b) for b in target_boxes], default=0.0)
                t1_o = max([iou_xyxy(top1_box, b) for b in other_boxes], default=0.0)
                tk_t = max([max([iou_xyxy(pb, b) for b in target_boxes], default=0.0) for pb in topk_boxes], default=0.0)
                tk_o = max([max([iou_xyxy(pb, b) for b in other_boxes], default=0.0) for pb in topk_boxes], default=0.0)
                row_out["top1_iou_target_max"] = float(t1_t)
                row_out["top1_iou_distractor_max"] = float(t1_o)
                row_out["topk_iou_target_max"] = float(tk_t)
                row_out["topk_iou_distractor_max"] = float(tk_o)

                if t1_t >= float(args.iou_thr_target) and t1_t >= t1_o:
                    row_out["semantic_identity"] = "target_evidence"
                elif t1_o >= float(args.iou_thr_distractor) and t1_o > t1_t:
                    row_out["semantic_identity"] = "distractor"
                else:
                    row_out["semantic_identity"] = "background_or_other"
            else:
                row_out["semantic_identity"] = "no_image_meta"

        per_rows.append(row_out)

    # Group aggregates
    group_rows: List[Dict[str, Any]] = []
    for g in ["fp_hall", "tp_yes"]:
        gg = [r for r in per_rows if str(r.get("group")) == g]
        if len(gg) == 0:
            continue
        c = Counter(str(r.get("semantic_identity")) for r in gg)
        t1_t = [safe_float(r.get("top1_iou_target_max")) for r in gg]
        t1_t = [float(v) for v in t1_t if v is not None]
        t1_o = [safe_float(r.get("top1_iou_distractor_max")) for r in gg]
        t1_o = [float(v) for v in t1_o if v is not None]
        tk_t = [safe_float(r.get("topk_iou_target_max")) for r in gg]
        tk_t = [float(v) for v in tk_t if v is not None]
        tk_o = [safe_float(r.get("topk_iou_distractor_max")) for r in gg]
        tk_o = [float(v) for v in tk_o if v is not None]

        row = {
            "group": g,
            "n": int(len(gg)),
            "semantic_target_evidence": int(c.get("target_evidence", 0)),
            "semantic_distractor": int(c.get("distractor", 0)),
            "semantic_background_or_other": int(c.get("background_or_other", 0)),
            "semantic_no_image_meta": int(c.get("no_image_meta", 0)),
            "mean_top1_iou_target": (None if len(t1_t) == 0 else float(sum(t1_t) / len(t1_t))),
            "mean_top1_iou_distractor": (None if len(t1_o) == 0 else float(sum(t1_o) / len(t1_o))),
            "mean_topk_iou_target": (None if len(tk_t) == 0 else float(sum(tk_t) / len(tk_t))),
            "mean_topk_iou_distractor": (None if len(tk_o) == 0 else float(sum(tk_o) / len(tk_o))),
        }
        group_rows.append(row)

    out_per = os.path.join(out_dir, "per_sample_semantic_identity.csv")
    out_group = os.path.join(out_dir, "group_semantic_stats.csv")
    out_summary = os.path.join(out_dir, "summary.json")
    write_csv(out_per, per_rows)
    write_csv(out_group, group_rows)

    summary = {
        "inputs": {
            "trace_csv": os.path.abspath(args.trace_csv),
            "layer": int(args.layer),
            "topk": int(args.topk),
            "idx_col": str(args.idx_col),
            "object_col": str(args.object_col),
            "instances_json": instances_path,
            "instances_found": bool(use_instances),
            "iou_thr_target": float(args.iou_thr_target),
            "iou_thr_distractor": float(args.iou_thr_distractor),
        },
        "counts": {
            "n_rows_input": int(len(rows)),
            "n_rows_used": int(len(per_rows)),
            "n_fp_hall": int(sum(1 for r in per_rows if str(r.get("group")) == "fp_hall")),
            "n_tp_yes": int(sum(1 for r in per_rows if str(r.get("group")) == "tp_yes")),
        },
        "outputs": {
            "per_sample_csv": out_per,
            "group_stats_csv": out_group,
            "summary_json": out_summary,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[saved]", out_per)
    print("[saved]", out_group)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()
