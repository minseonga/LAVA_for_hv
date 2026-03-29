#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
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
    s = re.sub(r"\s+", " ", s)
    return s


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


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _try_json_list(x: str) -> Optional[List[Any]]:
    try:
        v = json.loads(x)
    except Exception:
        return None
    if isinstance(v, list):
        return v
    return None


def parse_idx_list(x: Any, k: int) -> List[int]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    js = _try_json_list(s)
    vals: List[int] = []
    if js is not None:
        for z in js:
            try:
                vals.append(int(z))
            except Exception:
                pass
    else:
        # fallback: "1|2|3"
        parts = [p for p in s.split("|") if p.strip() != ""]
        for p in parts:
            try:
                vals.append(int(p))
            except Exception:
                pass
    if int(k) > 0:
        vals = vals[: int(k)]
    return vals


def parse_weight_list(x: Any, k: int) -> List[float]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    js = _try_json_list(s)
    vals: List[float] = []
    if js is not None:
        for z in js:
            v = safe_float(z)
            if v is not None:
                vals.append(float(v))
    else:
        parts = [p for p in s.split("|") if p.strip() != ""]
        for p in parts:
            v = safe_float(p)
            if v is not None:
                vals.append(float(v))
    if int(k) > 0:
        vals = vals[: int(k)]
    return vals


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    if len(sa) == 0 and len(sb) == 0:
        return 1.0
    uni = sa | sb
    if len(uni) == 0:
        return 0.0
    inter = sa & sb
    return float(len(inter) / len(uni))


def weighted_overlap(idx_a: Sequence[int], w_a: Sequence[float], idx_b: Sequence[int], w_b: Sequence[float]) -> float:
    if len(idx_a) == 0 or len(idx_b) == 0:
        return 0.0
    da: Dict[int, float] = {}
    db: Dict[int, float] = {}
    for i, p in enumerate(idx_a):
        ww = float(w_a[i]) if i < len(w_a) else 1.0
        da[int(p)] = max(0.0, float(ww))
    for i, p in enumerate(idx_b):
        ww = float(w_b[i]) if i < len(w_b) else 1.0
        db[int(p)] = max(0.0, float(ww))
    sa = sum(da.values())
    sb = sum(db.values())
    if sa <= 0.0:
        sa = float(len(da))
        da = {k: 1.0 for k in da.keys()}
    if sb <= 0.0:
        sb = float(len(db))
        db = {k: 1.0 for k in db.keys()}
    da = {k: v / sa for k, v in da.items()}
    db = {k: v / sb for k, v in db.items()}
    keys = set(da.keys()) | set(db.keys())
    return float(sum(min(da.get(k, 0.0), db.get(k, 0.0)) for k in keys))


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = [(int(labels[i]), float(scores[i])) for i in range(len(labels))]
    n_pos = int(sum(1 for y, _ in pairs if y == 1))
    n_neg = int(sum(1 for y, _ in pairs if y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    idxs = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(idxs):
        j = i + 1
        while j < len(idxs) and pairs[idxs[j]][1] == pairs[idxs[i]][1]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[idxs[k]] = float(avg_rank)
        i = j
    sum_pos = float(sum(ranks[i] for i in range(len(pairs)) if pairs[i][0] == 1))
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def ks_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pos = sorted(float(scores[i]) for i in range(len(scores)) if int(labels[i]) == 1)
    neg = sorted(float(scores[i]) for i in range(len(scores)) if int(labels[i]) == 0)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    support = sorted(set(pos + neg))
    i = 0
    j = 0
    dmax = 0.0
    for v in support:
        while i < n_pos and pos[i] <= v:
            i += 1
        while j < n_neg and neg[j] <= v:
            j += 1
        f_pos = float(i / n_pos)
        f_neg = float(j / n_neg)
        dmax = max(dmax, abs(f_pos - f_neg))
    return float(dmax)


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if len(xs) == 0:
        return None
    if len(xs) == 1:
        return float(xs[0])
    qq = min(1.0, max(0.0, float(q)))
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = float(pos - lo)
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def bootstrap_ci(metric_fn, labels: Sequence[int], scores: Sequence[float], n_boot: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    if int(n_boot) <= 0:
        return None, None
    n = int(len(labels))
    if n <= 1:
        return None, None
    rng = random.Random(int(seed))
    vals: List[float] = []
    for _ in range(int(n_boot)):
        ids = [rng.randrange(n) for _ in range(n)]
        lb = [int(labels[i]) for i in ids]
        sc = [float(scores[i]) for i in ids]
        m = metric_fn(lb, sc)
        if m is not None and math.isfinite(float(m)):
            vals.append(float(m))
    if len(vals) == 0:
        return None, None
    return quantile(vals, 0.025), quantile(vals, 0.975)


def entropy_norm(counts: Sequence[int]) -> float:
    xs = [int(v) for v in counts if int(v) > 0]
    if len(xs) <= 1:
        return 0.0
    n = float(sum(xs))
    ps = [float(v) / n for v in xs]
    h = -sum(p * math.log(p + 1e-12) for p in ps)
    return float(h / math.log(float(len(xs))))


def main() -> None:
    ap = argparse.ArgumentParser(description="POPE sink-vs-evidence analysis from per_layer_yes_trace.csv")
    ap.add_argument("--trace_csv", type=str, required=True, help="per_layer_yes_trace.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--image_col", type=str, default="image_id")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--question_col", type=str, default="question")
    ap.add_argument("--object_col", type=str, default="object_phrase")
    ap.add_argument("--pos_col", type=str, default="is_fp_hallucination")
    ap.add_argument("--neg_col", type=str, default="is_tp_yes")
    ap.add_argument("--top1_col", type=str, default="yes_sim_objpatch_argmax_idx_global")
    ap.add_argument("--topk_idx_col", type=str, default="yes_sim_objpatch_topk_idx_global_json")
    ap.add_argument("--topk_weight_col", type=str, default="yes_sim_objpatch_topk_weight_json")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--pair_filter", type=str, default="diff_object", choices=["all", "diff_object", "same_object"])
    ap.add_argument("--min_rows_per_image", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = read_csv(os.path.abspath(args.trace_csv))
    if len(rows) == 0:
        raise RuntimeError("No rows in trace_csv.")

    rows_l: List[Dict[str, Any]] = []
    for r in rows:
        li = safe_float(r.get("block_layer_idx"))
        if li is None or int(li) != int(args.layer):
            continue
        image_id = str(r.get(args.image_col) or "").strip()
        sid = str(r.get(args.id_col) or "").strip()
        q = str(r.get(args.question_col) or "").strip()
        if image_id == "" or sid == "":
            continue
        is_pos = parse_bool(r.get(args.pos_col))
        is_neg = parse_bool(r.get(args.neg_col))
        if not (is_pos or is_neg):
            continue
        idx_topk = parse_idx_list(r.get(args.topk_idx_col), int(args.topk))
        if len(idx_topk) == 0:
            continue
        if str(args.top1_col).strip() != "":
            top1 = safe_float(r.get(args.top1_col))
            top1_idx = int(top1) if top1 is not None else int(idx_topk[0])
        else:
            top1_idx = int(idx_topk[0])
        w_topk = parse_weight_list(r.get(args.topk_weight_col), int(args.topk))
        obj = norm_text(r.get(args.object_col))
        rows_l.append(
            {
                "id": sid,
                "image_id": image_id,
                "question": q,
                "object_phrase": obj,
                "is_pos": bool(is_pos),
                "is_neg": bool(is_neg),
                "top1_idx": int(top1_idx),
                "topk_idx": idx_topk,
                "topk_w": w_topk,
            }
        )

    by_img: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows_l:
        by_img[str(r["image_id"])].append(r)

    pair_rows: List[Dict[str, Any]] = []
    for image_id, rs in by_img.items():
        if len(rs) < int(args.min_rows_per_image):
            continue
        for a, b in itertools.combinations(rs, 2):
            obj_same = bool(a["object_phrase"] == b["object_phrase"])
            if args.pair_filter == "diff_object" and obj_same:
                continue
            if args.pair_filter == "same_object" and (not obj_same):
                continue

            qa = norm_text(a["question"])
            qb = norm_text(b["question"])
            same_q = bool(qa == qb)
            same_top1 = bool(int(a["top1_idx"]) == int(b["top1_idx"]))
            jac = float(jaccard(a["topk_idx"], b["topk_idx"]))
            wov = float(weighted_overlap(a["topk_idx"], a["topk_w"], b["topk_idx"], b["topk_w"]))

            if bool(a["is_pos"]) and bool(b["is_pos"]):
                pair_group = "pos_pos"
            elif bool(a["is_neg"]) and bool(b["is_neg"]):
                pair_group = "neg_neg"
            else:
                pair_group = "mixed"

            pair_rows.append(
                {
                    "image_id": image_id,
                    "id_a": a["id"],
                    "id_b": b["id"],
                    "pair_group": pair_group,
                    "same_question": bool(same_q),
                    "same_object": bool(obj_same),
                    "same_top1": bool(same_top1),
                    "jaccard_topk": jac,
                    "weighted_overlap_topk": wov,
                    "top1_a": int(a["top1_idx"]),
                    "top1_b": int(b["top1_idx"]),
                }
            )

    image_rows: List[Dict[str, Any]] = []
    for image_id, rs in by_img.items():
        for gname, pred in [
            ("pos", lambda z: bool(z["is_pos"])),
            ("neg", lambda z: bool(z["is_neg"])),
        ]:
            rr = [r for r in rs if pred(r)]
            if len(rr) < int(args.min_rows_per_image):
                continue
            c = Counter(int(r["top1_idx"]) for r in rr)
            n = int(len(rr))
            dom = int(max(c.values())) if len(c) > 0 else 0
            dom_share = float(dom / n) if n > 0 else 0.0
            entn = float(entropy_norm(list(c.values())))

            # Pair stats inside this image/group.
            same_top1_vals: List[int] = []
            jac_vals: List[float] = []
            wov_vals: List[float] = []
            for a, b in itertools.combinations(rr, 2):
                obj_same = bool(a["object_phrase"] == b["object_phrase"])
                if args.pair_filter == "diff_object" and obj_same:
                    continue
                if args.pair_filter == "same_object" and (not obj_same):
                    continue
                same_top1_vals.append(1 if int(a["top1_idx"]) == int(b["top1_idx"]) else 0)
                jac_vals.append(float(jaccard(a["topk_idx"], b["topk_idx"])))
                wov_vals.append(float(weighted_overlap(a["topk_idx"], a["topk_w"], b["topk_idx"], b["topk_w"])))

            image_rows.append(
                {
                    "image_id": image_id,
                    "group": gname,
                    "n_rows": n,
                    "n_unique_questions": int(len(set(norm_text(r["question"]) for r in rr))),
                    "n_unique_objects": int(len(set(norm_text(r["object_phrase"]) for r in rr))),
                    "dominant_top1_share": dom_share,
                    "top1_entropy_norm": entn,
                    "mean_pair_same_top1": (None if len(same_top1_vals) == 0 else float(sum(same_top1_vals) / len(same_top1_vals))),
                    "mean_pair_jaccard_topk": (None if len(jac_vals) == 0 else float(sum(jac_vals) / len(jac_vals))),
                    "mean_pair_weighted_overlap_topk": (None if len(wov_vals) == 0 else float(sum(wov_vals) / len(wov_vals))),
                    "n_pairs_used": int(len(jac_vals)),
                }
            )

    # Pair-level eval: pos_pos vs neg_neg only.
    eval_rows: List[Dict[str, Any]] = []
    eval_pairs = [r for r in pair_rows if str(r.get("pair_group")) in {"pos_pos", "neg_neg"}]
    if len(eval_pairs) > 0:
        labels = [1 if str(r.get("pair_group")) == "pos_pos" else 0 for r in eval_pairs]
        for metric in ["same_top1", "jaccard_topk", "weighted_overlap_topk"]:
            scores = [float(r.get(metric)) for r in eval_pairs]
            auc_h = auc_from_scores(labels, scores)
            auc_b = None if auc_h is None else max(float(auc_h), 1.0 - float(auc_h))
            direction = None
            if auc_h is not None:
                direction = "higher_in_pos_pos" if float(auc_h) >= 0.5 else "lower_in_pos_pos"
            ks_h = ks_from_scores(labels, scores)
            auc_lo, auc_hi = bootstrap_ci(auc_from_scores, labels, scores, int(args.bootstrap), int(args.seed))
            ks_lo, ks_hi = bootstrap_ci(ks_from_scores, labels, scores, int(args.bootstrap), int(args.seed))
            eval_rows.append(
                {
                    "metric": metric,
                    "comparison": "pos_pos_vs_neg_neg",
                    "n": int(len(scores)),
                    "auc_pos_high": auc_h,
                    "auc_best_dir": auc_b,
                    "direction": direction,
                    "ks_pos_high": ks_h,
                    "auc_ci95_lo": auc_lo,
                    "auc_ci95_hi": auc_hi,
                    "ks_ci95_lo": ks_lo,
                    "ks_ci95_hi": ks_hi,
                }
            )

    # Image-level eval: pos-image vs neg-image.
    pos_img = [r for r in image_rows if str(r.get("group")) == "pos"]
    neg_img = [r for r in image_rows if str(r.get("group")) == "neg"]
    if len(pos_img) > 0 and len(neg_img) > 0:
        metrics = [
            "dominant_top1_share",
            "top1_entropy_norm",
            "mean_pair_same_top1",
            "mean_pair_jaccard_topk",
            "mean_pair_weighted_overlap_topk",
        ]
        for metric in metrics:
            merged = []
            for r in pos_img:
                v = safe_float(r.get(metric))
                if v is not None:
                    merged.append((1, float(v)))
            for r in neg_img:
                v = safe_float(r.get(metric))
                if v is not None:
                    merged.append((0, float(v)))
            if len(merged) <= 1:
                continue
            labels = [int(y) for y, _ in merged]
            scores = [float(v) for _, v in merged]
            auc_h = auc_from_scores(labels, scores)
            auc_b = None if auc_h is None else max(float(auc_h), 1.0 - float(auc_h))
            direction = None
            if auc_h is not None:
                direction = "higher_in_pos_img" if float(auc_h) >= 0.5 else "lower_in_pos_img"
            ks_h = ks_from_scores(labels, scores)
            auc_lo, auc_hi = bootstrap_ci(auc_from_scores, labels, scores, int(args.bootstrap), int(args.seed))
            ks_lo, ks_hi = bootstrap_ci(ks_from_scores, labels, scores, int(args.bootstrap), int(args.seed))
            eval_rows.append(
                {
                    "metric": metric,
                    "comparison": "pos_img_vs_neg_img",
                    "n": int(len(scores)),
                    "auc_pos_high": auc_h,
                    "auc_best_dir": auc_b,
                    "direction": direction,
                    "ks_pos_high": ks_h,
                    "auc_ci95_lo": auc_lo,
                    "auc_ci95_hi": auc_hi,
                    "ks_ci95_lo": ks_lo,
                    "ks_ci95_hi": ks_hi,
                }
            )

    best = None
    if len(eval_rows) > 0:
        best = max(eval_rows, key=lambda r: float(r.get("auc_best_dir") or -1.0))

    out_pair = os.path.join(out_dir, "pair_sink_metrics.csv")
    out_image = os.path.join(out_dir, "image_sink_metrics.csv")
    out_eval = os.path.join(out_dir, "eval_sink_vs_evidence.csv")
    out_summary = os.path.join(out_dir, "summary.json")
    write_csv(out_pair, pair_rows)
    write_csv(out_image, image_rows)
    write_csv(out_eval, eval_rows)

    summary = {
        "inputs": {
            "trace_csv": os.path.abspath(args.trace_csv),
            "layer": int(args.layer),
            "image_col": str(args.image_col),
            "id_col": str(args.id_col),
            "question_col": str(args.question_col),
            "object_col": str(args.object_col),
            "pos_col": str(args.pos_col),
            "neg_col": str(args.neg_col),
            "top1_col": str(args.top1_col),
            "topk_idx_col": str(args.topk_idx_col),
            "topk_weight_col": str(args.topk_weight_col),
            "topk": int(args.topk),
            "pair_filter": str(args.pair_filter),
            "min_rows_per_image": int(args.min_rows_per_image),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "counts": {
            "n_rows_input": int(len(rows)),
            "n_rows_layer_filtered": int(len(rows_l)),
            "n_images": int(len(by_img)),
            "n_pair_rows": int(len(pair_rows)),
            "n_image_rows": int(len(image_rows)),
            "n_eval_rows": int(len(eval_rows)),
            "n_pos_pos_pairs": int(sum(1 for r in pair_rows if str(r.get("pair_group")) == "pos_pos")),
            "n_neg_neg_pairs": int(sum(1 for r in pair_rows if str(r.get("pair_group")) == "neg_neg")),
            "n_mixed_pairs": int(sum(1 for r in pair_rows if str(r.get("pair_group")) == "mixed")),
        },
        "best_eval": best,
        "outputs": {
            "pair_csv": out_pair,
            "image_csv": out_image,
            "eval_csv": out_eval,
            "summary_json": out_summary,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_pair)
    print("[saved]", out_image)
    print("[saved]", out_eval)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()

