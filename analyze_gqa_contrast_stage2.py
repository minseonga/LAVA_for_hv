#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def norm_text(x: Any) -> str:
    s = str("" if x is None else x).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
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


def summarize_binary(labels: List[int], scores: List[float]) -> Dict[str, Any]:
    auc = auc_from_scores(labels, scores)
    ks = ks_from_scores(labels, scores)
    return {
        "n": int(len(scores)),
        "auc_pos_high": auc,
        "auc_best_dir": (None if auc is None else float(max(float(auc), 1.0 - float(auc)))),
        "direction": (None if auc is None else ("higher_in_positive" if float(auc) >= 0.5 else "lower_in_positive")),
        "ks_pos_high": ks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 GQA contrast/consistency analysis with fixed AIS metric.")
    ap.add_argument("--per_id_csv", type=str, required=True, help="per_id_ers_ais_pcs.csv")
    ap.add_argument("--baseline_csv", type=str, required=True, help="GQA per_sample.csv with pred_answer_eval/is_success")
    ap.add_argument("--questions_json", type=str, default="/home/kms/data/gqa/testdev_balanced_questions.json")
    ap.add_argument("--metric", type=str, default="ais_mean")
    ap.add_argument(
        "--pair_modes",
        type=str,
        default="equivalent,entailed,local_flip",
        help="Comma-separated pair sources: equivalent, entailed, local_flip.",
    )
    ap.add_argument("--local_flip_same_image", type=parse_bool, default=True)
    ap.add_argument("--local_flip_limit_per_group", type=int, default=200)
    ap.add_argument("--min_pairs_for_eval", type=int, default=20)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    pair_modes = {x.strip().lower() for x in str(args.pair_modes).split(",") if x.strip() != ""}

    with open(os.path.abspath(args.questions_json), "r", encoding="utf-8") as f:
        qj = json.load(f)
    if not isinstance(qj, dict):
        raise RuntimeError("questions_json must be dict keyed by question id.")

    per_id_rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(args.per_id_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            per_id_rows.append(dict(r))
    if len(per_id_rows) == 0:
        raise RuntimeError("Empty per_id_csv.")

    base_rows: Dict[str, Dict[str, Any]] = {}
    with open(os.path.abspath(args.baseline_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "").strip()
            if sid != "":
                base_rows[sid] = dict(r)

    # Build sample map (id -> annotations + metric).
    samples: Dict[str, Dict[str, Any]] = {}
    missing_meta = 0
    for r in per_id_rows:
        sid = str(r.get("id") or "").strip()
        if sid == "":
            continue
        qr = qj.get(sid)
        if qr is None:
            missing_meta += 1
            continue
        mv = safe_float(r.get(args.metric))
        if mv is None:
            continue
        br = base_rows.get(sid, {})
        answer_gt = str(br.get("answer") or r.get("answer_gt") or "").strip()
        pred = str(br.get("pred_answer_eval") or r.get("answer_pred") or "").strip()
        is_success = parse_bool(br.get("is_success")) if sid in base_rows else bool(parse_bool(r.get("is_tp_yes")))
        samples[sid] = {
            "id": sid,
            "image_id": str(br.get("image_id") or r.get("image_id") or qr.get("imageId") or "").strip(),
            "answer_gt": answer_gt,
            "pred_answer": pred,
            "answer_gt_norm": norm_text(answer_gt),
            "pred_norm": norm_text(pred),
            "is_success": bool(is_success),
            "is_failure": bool(not is_success),
            "metric": float(mv),
            "local_group": str((qr.get("groups") or {}).get("local") or "").strip(),
            "equivalent": [str(x) for x in (qr.get("equivalent") or [])],
            "entailed": [str(x) for x in (qr.get("entailed") or [])],
        }

    if len(samples) == 0:
        raise RuntimeError("No usable samples after merge.")

    # Pair builders.
    pair_set_undir = set()
    pair_rows: List[Dict[str, Any]] = []

    def add_pair(a: str, b: str, ptype: str, directed: bool = False) -> None:
        if a == b:
            return
        if a not in samples or b not in samples:
            return
        if directed:
            key = (ptype, a, b, "dir")
            if key in pair_set_undir:
                return
            pair_set_undir.add(key)
        else:
            x, y = sorted([a, b])
            key = (ptype, x, y, "undir")
            if key in pair_set_undir:
                return
            pair_set_undir.add(key)

        sa = samples[a]
        sb = samples[b]
        gt_diff = bool(sa["answer_gt_norm"] != sb["answer_gt_norm"])
        pred_diff = bool(sa["pred_norm"] != sb["pred_norm"])
        one_wrong = bool(sa["is_success"] != sb["is_success"])
        pair_rows.append(
            {
                "pair_type": str(ptype),
                "is_directed": bool(directed),
                "id_a": a,
                "id_b": b,
                "image_a": sa["image_id"],
                "image_b": sb["image_id"],
                "same_image": bool(sa["image_id"] == sb["image_id"]),
                "answer_gt_a": sa["answer_gt"],
                "answer_gt_b": sb["answer_gt"],
                "pred_a": sa["pred_answer"],
                "pred_b": sb["pred_answer"],
                "answer_gt_diff": bool(gt_diff),
                "pred_diff": bool(pred_diff),
                "success_a": bool(sa["is_success"]),
                "success_b": bool(sb["is_success"]),
                "one_correct_one_wrong": bool(one_wrong),
                "both_correct": bool(sa["is_success"] and sb["is_success"]),
                "both_wrong": bool((not sa["is_success"]) and (not sb["is_success"])),
                "metric_a": float(sa["metric"]),
                "metric_b": float(sb["metric"]),
                "metric_mean": float(0.5 * (sa["metric"] + sb["metric"])),
                "metric_max": float(max(sa["metric"], sb["metric"])),
                "metric_absdiff": float(abs(sa["metric"] - sb["metric"])),
            }
        )

    if "equivalent" in pair_modes:
        for sid, s in samples.items():
            for t in s["equivalent"]:
                add_pair(sid, t, ptype="equivalent", directed=False)

    if "entailed" in pair_modes:
        for sid, s in samples.items():
            for t in s["entailed"]:
                add_pair(sid, t, ptype="entailed", directed=True)

    if "local_flip" in pair_modes:
        by_local: Dict[str, List[str]] = defaultdict(list)
        for sid, s in samples.items():
            lg = str(s.get("local_group") or "")
            if lg != "":
                by_local[lg].append(sid)
        lim = int(max(1, args.local_flip_limit_per_group))
        for lg, ids in by_local.items():
            m = len(ids)
            if m < 2:
                continue
            c = 0
            for i in range(m):
                si = samples[ids[i]]
                for j in range(i + 1, m):
                    sj = samples[ids[j]]
                    if bool(args.local_flip_same_image) and si["image_id"] != sj["image_id"]:
                        continue
                    if si["answer_gt_norm"] == sj["answer_gt_norm"]:
                        continue
                    add_pair(ids[i], ids[j], ptype="local_flip", directed=False)
                    c += 1
                    if c >= lim:
                        break
                if c >= lim:
                    break

    # Pair-level failure definitions.
    for r in pair_rows:
        ptype = str(r["pair_type"])
        gt_diff = bool(r["answer_gt_diff"])
        pred_diff = bool(r["pred_diff"])
        if ptype == "equivalent":
            # equivalent should preserve answer.
            r["pair_consistency_failure"] = bool(pred_diff)
            r["pair_expected_relation"] = "same_answer"
        elif ptype == "local_flip":
            # contrast/local_flip should change answer.
            r["pair_consistency_failure"] = bool(not pred_diff)
            r["pair_expected_relation"] = "different_answer"
        else:
            # entailed: keep as unknown consistency target; use correctness asymmetry only.
            r["pair_consistency_failure"] = None
            r["pair_expected_relation"] = "entailed_directional"
        r["pair_gt_diff"] = bool(gt_diff)

    # Directed "original correct -> contrast failure" events (for gt-different pairs).
    directed_events: List[Dict[str, Any]] = []
    for r in pair_rows:
        if not bool(r.get("answer_gt_diff")):
            continue
        sa = bool(r.get("success_a"))
        sb = bool(r.get("success_b"))
        ma = safe_float(r.get("metric_a"))
        mb = safe_float(r.get("metric_b"))
        if ma is None or mb is None:
            continue
        if sa and (not sb):
            directed_events.append(
                {
                    "pair_type": r["pair_type"],
                    "origin_id": r["id_a"],
                    "contrast_id": r["id_b"],
                    "origin_success": True,
                    "contrast_failure": True,
                    "metric_origin": float(ma),
                    "metric_contrast": float(mb),
                    "delta_contrast_minus_origin": float(mb - ma),
                }
            )
        if sb and (not sa):
            directed_events.append(
                {
                    "pair_type": r["pair_type"],
                    "origin_id": r["id_b"],
                    "contrast_id": r["id_a"],
                    "origin_success": True,
                    "contrast_failure": True,
                    "metric_origin": float(mb),
                    "metric_contrast": float(ma),
                    "delta_contrast_minus_origin": float(ma - mb),
                }
            )

    # Summaries by pair type.
    summary_rows: List[Dict[str, Any]] = []
    for ptype in sorted(set(str(r["pair_type"]) for r in pair_rows)):
        rs = [r for r in pair_rows if str(r["pair_type"]) == ptype]
        n = len(rs)
        n_gt_diff = sum(1 for r in rs if bool(r.get("pair_gt_diff")))
        n_one_wrong = sum(1 for r in rs if bool(r.get("one_correct_one_wrong")))
        n_both_correct = sum(1 for r in rs if bool(r.get("both_correct")))
        n_both_wrong = sum(1 for r in rs if bool(r.get("both_wrong")))
        cf_vals = [r for r in rs if r.get("pair_consistency_failure") is not None]
        n_cf = len(cf_vals)
        n_cf_fail = sum(1 for r in cf_vals if bool(r.get("pair_consistency_failure")))
        de = [x for x in directed_events if str(x["pair_type"]) == ptype]
        deltas = [float(x["delta_contrast_minus_origin"]) for x in de]
        summary_rows.append(
            {
                "pair_type": ptype,
                "n_pairs": int(n),
                "n_pairs_gt_diff": int(n_gt_diff),
                "n_one_correct_one_wrong": int(n_one_wrong),
                "rate_one_correct_one_wrong": (None if n == 0 else float(n_one_wrong / n)),
                "n_both_correct": int(n_both_correct),
                "n_both_wrong": int(n_both_wrong),
                "n_consistency_eval": int(n_cf),
                "n_consistency_fail": int(n_cf_fail),
                "consistency_fail_rate": (None if n_cf == 0 else float(n_cf_fail / n_cf)),
                "n_directed_orig_correct_contrast_fail": int(len(de)),
                "delta_contrast_minus_origin_mean": (None if len(deltas) == 0 else float(sum(deltas) / len(deltas))),
                "delta_contrast_minus_origin_pos_rate": (None if len(deltas) == 0 else float(sum(1 for d in deltas if d > 0.0) / len(deltas))),
            }
        )

    # Pair-level binary eval: consistency_fail vs not, scored by pair-level metric.
    eval_rows: List[Dict[str, Any]] = []
    score_fields = ["metric_mean", "metric_max", "metric_absdiff"]
    for ptype in sorted(set(str(r["pair_type"]) for r in pair_rows)):
        rs = [r for r in pair_rows if str(r["pair_type"]) == ptype and r.get("pair_consistency_failure") is not None]
        for sf in score_fields:
            labels: List[int] = []
            scores: List[float] = []
            for r in rs:
                v = safe_float(r.get(sf))
                if v is None:
                    continue
                labels.append(1 if bool(r.get("pair_consistency_failure")) else 0)
                scores.append(float(v))
            if len(scores) < int(args.min_pairs_for_eval):
                continue
            sm = summarize_binary(labels, scores)
            eval_rows.append(
                {
                    "pair_type": ptype,
                    "target": "pair_consistency_failure",
                    "score_field": sf,
                    **sm,
                }
            )

    out_manifest = os.path.join(out_dir, "pair_manifest.csv")
    out_summary_rows = os.path.join(out_dir, "pair_summary_by_type.csv")
    out_directed = os.path.join(out_dir, "directed_orig_correct_to_contrast_fail.csv")
    out_eval = os.path.join(out_dir, "pair_eval.csv")
    out_summary = os.path.join(out_dir, "summary.json")

    write_csv(out_manifest, pair_rows)
    write_csv(out_summary_rows, summary_rows)
    write_csv(out_directed, directed_events)
    write_csv(out_eval, eval_rows)

    # Key top-lines for requested stage-2 checks.
    def pick_row(ptype: str) -> Optional[Dict[str, Any]]:
        for r in summary_rows:
            if str(r.get("pair_type")) == ptype:
                return r
        return None

    key_local = pick_row("local_flip")
    key_equiv = pick_row("equivalent")

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "per_id_csv": os.path.abspath(args.per_id_csv),
                    "baseline_csv": os.path.abspath(args.baseline_csv),
                    "questions_json": os.path.abspath(args.questions_json),
                    "metric": str(args.metric),
                    "pair_modes": sorted(list(pair_modes)),
                    "local_flip_same_image": bool(args.local_flip_same_image),
                    "local_flip_limit_per_group": int(args.local_flip_limit_per_group),
                    "min_pairs_for_eval": int(args.min_pairs_for_eval),
                },
                "counts": {
                    "n_samples_merged": int(len(samples)),
                    "n_pairs_total": int(len(pair_rows)),
                    "n_directed_events": int(len(directed_events)),
                    "n_pair_eval_rows": int(len(eval_rows)),
                    "n_missing_meta_from_questions": int(missing_meta),
                },
                "stage2_key_checks": {
                    "original_correct_vs_contrast_failure_local_flip": key_local,
                    "paired_consistency_failure_equivalent": key_equiv,
                },
                "outputs": {
                    "pair_manifest_csv": out_manifest,
                    "pair_summary_csv": out_summary_rows,
                    "directed_events_csv": out_directed,
                    "pair_eval_csv": out_eval,
                    "summary_json": out_summary,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[saved]", out_manifest)
    print("[saved]", out_summary_rows)
    print("[saved]", out_directed)
    print("[saved]", out_eval)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()

