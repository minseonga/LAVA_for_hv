#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc GQA AIS stratification by structural/semantic type.")
    ap.add_argument("--per_id_csv", type=str, required=True, help="Output of analyze_ers_ais_pcs.py (per_id_ers_ais_pcs.csv).")
    ap.add_argument("--questions_json", type=str, default="/home/kms/data/gqa/testdev_balanced_questions.json")
    ap.add_argument("--metric", type=str, default="ais_mean")
    ap.add_argument("--min_group_n", type=int, default=30)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.abspath(args.questions_json), "r", encoding="utf-8") as f:
        qj = json.load(f)
    if not isinstance(qj, dict):
        raise RuntimeError("GQA questions_json must be dict keyed by question id.")

    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(args.per_id_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    if len(rows) == 0:
        raise RuntimeError("Empty per_id_csv.")

    merged: List[Dict[str, Any]] = []
    miss_meta = 0
    for r in rows:
        sid = str(r.get("id") or "").strip()
        qr = qj.get(sid)
        if qr is None:
            miss_meta += 1
            continue
        types = qr.get("types") or {}
        merged.append(
            {
                **r,
                "id": sid,
                "structural": str(types.get("structural") or ""),
                "semantic_type": str(types.get("semantic") or ""),
                "detailed_type": str(types.get("detailed") or ""),
            }
        )
    if len(merged) == 0:
        raise RuntimeError("No rows matched with GQA metadata.")

    metric = str(args.metric)

    def eval_by_group(group_key: str) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in merged:
            g = str(r.get(group_key) or "").strip()
            if g == "":
                continue
            groups.setdefault(g, []).append(r)

        out: List[Dict[str, Any]] = []
        for g, rs in groups.items():
            labels: List[int] = []
            scores: List[float] = []
            for r in rs:
                v = safe_float(r.get(metric))
                if v is None:
                    continue
                if parse_bool(r.get("is_fp_hallucination")):
                    labels.append(1)
                    scores.append(float(v))
                elif parse_bool(r.get("is_tp_yes")):
                    labels.append(0)
                    scores.append(float(v))
            n = len(scores)
            if n < int(args.min_group_n):
                continue
            auc = auc_from_scores(labels, scores)
            ks = ks_from_scores(labels, scores)
            n_pos = int(sum(1 for y in labels if y == 1))
            n_neg = int(sum(1 for y in labels if y == 0))
            if auc is None or ks is None or n_pos == 0 or n_neg == 0:
                continue
            out.append(
                {
                    "group_axis": group_key,
                    "group_value": g,
                    "metric": metric,
                    "n": n,
                    "n_failure": n_pos,
                    "n_success": n_neg,
                    "auc_hall_high": float(auc),
                    "auc_best_dir": float(max(auc, 1.0 - auc)),
                    "direction": "higher_in_hallucination" if auc >= 0.5 else "lower_in_hallucination",
                    "ks_hall_high": float(ks),
                    "score_mean": float(sum(scores) / n),
                }
            )
        out.sort(key=lambda x: float(x["auc_best_dir"]), reverse=True)
        return out

    rows_struct = eval_by_group("structural")
    rows_sem = eval_by_group("semantic_type")
    rows_detail = eval_by_group("detailed_type")

    all_rows = rows_struct + rows_sem + rows_detail
    best = all_rows[0] if len(all_rows) > 0 else None

    out_struct = os.path.join(out_dir, "stratified_structural.csv")
    out_sem = os.path.join(out_dir, "stratified_semantic.csv")
    out_detail = os.path.join(out_dir, "stratified_detailed.csv")
    out_summary = os.path.join(out_dir, "summary.json")
    write_csv(out_struct, rows_struct)
    write_csv(out_sem, rows_sem)
    write_csv(out_detail, rows_detail)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "per_id_csv": os.path.abspath(args.per_id_csv),
                    "questions_json": os.path.abspath(args.questions_json),
                    "metric": metric,
                    "min_group_n": int(args.min_group_n),
                },
                "counts": {
                    "n_per_id_rows": int(len(rows)),
                    "n_merged_rows": int(len(merged)),
                    "n_missing_meta": int(miss_meta),
                    "n_struct_groups": int(len(rows_struct)),
                    "n_semantic_groups": int(len(rows_sem)),
                    "n_detailed_groups": int(len(rows_detail)),
                },
                "best_group": best,
                "outputs": {
                    "structural_csv": out_struct,
                    "semantic_csv": out_sem,
                    "detailed_csv": out_detail,
                    "summary_json": out_summary,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[saved]", out_struct)
    print("[saved]", out_sem)
    print("[saved]", out_detail)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()

