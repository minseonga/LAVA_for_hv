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
    ap = argparse.ArgumentParser(description="Layer-wise phenomenon occurrence ratio (false vs true).")
    ap.add_argument("--per_layer_trace_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--metric", type=str, default="yes_sim_local_max")
    ap.add_argument("--layer_col", type=str, default="block_layer_idx")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--pos_col", type=str, default="is_fp_hallucination", help="Positive class (e.g., failure).")
    ap.add_argument("--neg_col", type=str, default="is_tp_yes", help="Negative class (e.g., success).")
    ap.add_argument(
        "--event_direction",
        type=str,
        default="high",
        choices=["high", "low"],
        help="Whether the phenomenon is larger ('high') or smaller ('low') in positive class.",
    )
    ap.add_argument(
        "--threshold_quantile",
        type=float,
        default=0.9,
        help="Threshold is derived from negative class distribution per layer.",
    )
    args = ap.parse_args()

    in_csv = os.path.abspath(args.per_layer_trace_csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with open(in_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    if len(rows) == 0:
        raise RuntimeError("Empty per-layer trace CSV.")

    # Deduplicate by (id, layer): keep first valid metric.
    by_layer: Dict[int, Dict[str, List[float]]] = {}
    seen = set()
    for r in rows:
        sid = str(r.get(args.id_col) or "")
        li = safe_float(r.get(args.layer_col))
        mv = safe_float(r.get(args.metric))
        if sid == "" or li is None or mv is None:
            continue
        layer = int(li)
        key = (sid, layer)
        if key in seen:
            continue
        seen.add(key)
        b = by_layer.setdefault(layer, {"pos": [], "neg": []})
        if parse_bool(r.get(args.pos_col)):
            b["pos"].append(float(mv))
        elif parse_bool(r.get(args.neg_col)):
            b["neg"].append(float(mv))

    rows_out: List[Dict[str, Any]] = []
    for layer in sorted(by_layer.keys()):
        pos = list(by_layer[layer]["pos"])
        neg = list(by_layer[layer]["neg"])
        if len(pos) == 0 or len(neg) == 0:
            continue

        labels = [1] * len(pos) + [0] * len(neg)
        scores = pos + neg
        auc_h = auc_from_scores(labels, scores)
        ks_h = ks_from_scores(labels, scores)
        auc_best = None if auc_h is None else float(max(float(auc_h), 1.0 - float(auc_h)))
        direction = None if auc_h is None else ("higher_in_pos" if float(auc_h) >= 0.5 else "lower_in_pos")

        q = float(min(1.0, max(0.0, args.threshold_quantile)))
        if str(args.event_direction) == "high":
            thr = quantile(neg, q)
            event_pos = [1 if v >= float(thr) else 0 for v in pos] if thr is not None else []
            event_neg = [1 if v >= float(thr) else 0 for v in neg] if thr is not None else []
        else:
            thr = quantile(neg, 1.0 - q)
            event_pos = [1 if v <= float(thr) else 0 for v in pos] if thr is not None else []
            event_neg = [1 if v <= float(thr) else 0 for v in neg] if thr is not None else []

        pos_rate = (None if len(event_pos) == 0 else float(sum(event_pos) / len(event_pos)))
        neg_rate = (None if len(event_neg) == 0 else float(sum(event_neg) / len(event_neg)))
        lift = None
        if pos_rate is not None and neg_rate is not None:
            lift = float(pos_rate / max(1e-9, neg_rate))

        rows_out.append(
            {
                "layer": int(layer),
                "metric": str(args.metric),
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "mean_pos": float(sum(pos) / len(pos)),
                "mean_neg": float(sum(neg) / len(neg)),
                "auc_pos_high": auc_h,
                "auc_best_dir": auc_best,
                "auc_direction": direction,
                "ks_pos_high": ks_h,
                "threshold_from_neg_q": q,
                "threshold": thr,
                "event_direction": str(args.event_direction),
                "event_rate_pos": pos_rate,
                "event_rate_neg": neg_rate,
                "event_rate_diff_pos_minus_neg": (
                    None if pos_rate is None or neg_rate is None else float(pos_rate - neg_rate)
                ),
                "event_rate_lift_pos_over_neg": lift,
            }
        )

    if len(rows_out) == 0:
        raise RuntimeError("No valid layer rows for ratio analysis.")

    rows_auc = [r for r in rows_out if r.get("auc_best_dir") is not None]
    rows_lift = [r for r in rows_out if r.get("event_rate_lift_pos_over_neg") is not None]
    best_auc = max(rows_auc, key=lambda r: float(r["auc_best_dir"])) if len(rows_auc) > 0 else None
    best_lift = max(rows_lift, key=lambda r: float(r["event_rate_lift_pos_over_neg"])) if len(rows_lift) > 0 else None

    out_csv = os.path.join(out_dir, "layer_occurrence_ratio.csv")
    out_summary = os.path.join(out_dir, "summary.json")
    write_csv(out_csv, rows_out)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "per_layer_trace_csv": in_csv,
                    "metric": str(args.metric),
                    "layer_col": str(args.layer_col),
                    "id_col": str(args.id_col),
                    "pos_col": str(args.pos_col),
                    "neg_col": str(args.neg_col),
                    "event_direction": str(args.event_direction),
                    "threshold_quantile": float(args.threshold_quantile),
                },
                "counts": {
                    "n_rows_input": int(len(rows)),
                    "n_layers_eval": int(len(rows_out)),
                },
                "best_auc_layer": best_auc,
                "best_lift_layer": best_lift,
                "outputs": {
                    "layer_occurrence_ratio_csv": out_csv,
                    "summary_json": out_summary,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[saved]", out_csv)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()

