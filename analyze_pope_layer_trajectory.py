#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_layers(s: str) -> List[int]:
    out: List[int] = []
    for t in str(s).split(","):
        tt = t.strip()
        if tt == "":
            continue
        out.append(int(tt))
    if len(out) == 0:
        raise RuntimeError("No valid layers parsed from --layers.")
    return out


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([])
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def mean(xs: Sequence[float]) -> Optional[float]:
    if len(xs) == 0:
        return None
    return float(sum(xs) / float(len(xs)))


def std(xs: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n <= 1:
        return 0.0 if n == 1 else None
    mu = float(sum(xs) / float(n))
    v = float(sum((x - mu) * (x - mu) for x in xs) / float(n - 1))
    return float(math.sqrt(max(0.0, v)))


def trapezoid_area(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    area = 0.0
    for i in range(len(xs) - 1):
        dx = float(xs[i + 1] - xs[i])
        area += 0.5 * dx * float(ys[i] + ys[i + 1])
    return float(area)


def auc_hall_high(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> Optional[float]:
    # Mann-Whitney U / (n_pos * n_neg), ties get 0.5
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return None
    total = 0.0
    for sp in scores_pos:
        lt = 0
        eq = 0
        for sn in scores_neg:
            if sp > sn:
                lt += 1
            elif sp == sn:
                eq += 1
        total += float(lt) + 0.5 * float(eq)
    return float(total / float(n_pos * n_neg))


def ks_stat(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> Optional[float]:
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return None
    vals = sorted(set(list(scores_pos) + list(scores_neg)))
    if len(vals) == 0:
        return None
    p_sorted = sorted(scores_pos)
    n_sorted = sorted(scores_neg)
    ip = 0
    ineg = 0
    np = len(p_sorted)
    nn = len(n_sorted)
    ks = 0.0
    for v in vals:
        while ip < np and p_sorted[ip] <= v:
            ip += 1
        while ineg < nn and n_sorted[ineg] <= v:
            ineg += 1
        cdf_p = float(ip) / float(np)
        cdf_n = float(ineg) / float(nn)
        d = abs(cdf_p - cdf_n)
        if d > ks:
            ks = d
    return float(ks)


def bootstrap_ci(
    scores_pos: Sequence[float],
    scores_neg: Sequence[float],
    fn,
    n_boot: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float]]:
    if n_boot <= 0:
        return None, None
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return None, None
    rng = random.Random(seed)
    vals: List[float] = []
    sp = list(scores_pos)
    sn = list(scores_neg)
    for _ in range(int(n_boot)):
        bp = [sp[rng.randrange(len(sp))] for _ in range(len(sp))]
        bn = [sn[rng.randrange(len(sn))] for _ in range(len(sn))]
        v = fn(bp, bn)
        if v is not None and math.isfinite(v):
            vals.append(float(v))
    if len(vals) == 0:
        return None, None
    vals = sorted(vals)
    lo_idx = int(max(0, math.floor(0.025 * (len(vals) - 1))))
    hi_idx = int(min(len(vals) - 1, math.floor(0.975 * (len(vals) - 1))))
    return float(vals[lo_idx]), float(vals[hi_idx])


def add_layer_feature_dict(
    out: Dict[str, Any],
    values_by_layer: Dict[int, float],
    layers: List[int],
    prefix: str = "v",
) -> None:
    for l in layers:
        v = values_by_layer.get(int(l))
        out[f"{prefix}_l{int(l)}"] = "" if v is None else float(v)


def build_features_for_sample(
    values_by_layer: Dict[int, float],
    layers: List[int],
) -> Dict[str, Any]:
    feat: Dict[str, Any] = {}
    add_layer_feature_dict(feat, values_by_layer, layers, prefix="v")

    vals_sel: List[float] = []
    xs_sel: List[float] = []
    for l in layers:
        v = values_by_layer.get(int(l))
        if v is None:
            continue
        xs_sel.append(float(l))
        vals_sel.append(float(v))

    if len(vals_sel) > 0:
        feat["v_mean_sel"] = float(sum(vals_sel) / float(len(vals_sel)))
        feat["v_std_sel"] = float(std(vals_sel) or 0.0)
        feat["v_peak_sel"] = float(max(vals_sel))
        argmax_i = max(range(len(vals_sel)), key=lambda i: vals_sel[i])
        feat["v_peak_layer_sel"] = int(xs_sel[argmax_i])
    else:
        feat["v_mean_sel"] = ""
        feat["v_std_sel"] = ""
        feat["v_peak_sel"] = ""
        feat["v_peak_layer_sel"] = ""

    # Pairwise deltas for specified list order.
    if len(layers) >= 2:
        for i in range(len(layers) - 1):
            l0 = int(layers[i])
            l1 = int(layers[i + 1])
            v0 = values_by_layer.get(l0)
            v1 = values_by_layer.get(l1)
            key = f"d_l{l0}_l{l1}"
            feat[key] = "" if (v0 is None or v1 is None) else float(v1 - v0)
        l0 = int(layers[0])
        l1 = int(layers[-1])
        v0 = values_by_layer.get(l0)
        v1 = values_by_layer.get(l1)
        feat[f"d_l{l0}_l{l1}"] = "" if (v0 is None or v1 is None) else float(v1 - v0)
        if v0 is None or v1 is None:
            feat[f"slope_l{l0}_l{l1}"] = ""
        else:
            feat[f"slope_l{l0}_l{l1}"] = float((v1 - v0) / float(max(1, l1 - l0)))
    if len(layers) >= 3:
        la = int(layers[0])
        lm = int(layers[1])
        lb = int(layers[-1])
        va = values_by_layer.get(la)
        vm = values_by_layer.get(lm)
        vb = values_by_layer.get(lb)
        if va is None or vm is None or vb is None:
            feat["mid_bump"] = ""
        else:
            feat["mid_bump"] = float(vm - 0.5 * (va + vb))

    area = trapezoid_area(xs_sel, vals_sel) if len(xs_sel) >= 2 else None
    feat["auc_trapz_sel"] = "" if area is None else float(area)
    return feat


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze per-sample layer trajectories (FP hall vs TP yes).")
    ap.add_argument("--per_layer_trace_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--metric", type=str, default="yes_sim_objpatch_max")
    ap.add_argument("--layers", type=str, default="10,17,24")
    ap.add_argument("--group_pos_col", type=str, default="is_fp_hallucination")
    ap.add_argument("--group_neg_col", type=str, default="is_tp_yes")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_csv = os.path.abspath(args.per_layer_trace_csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    layers = parse_layers(args.layers)
    metric = str(args.metric)

    rows: List[Dict[str, Any]] = []
    with open(in_csv, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    if len(rows) == 0:
        raise RuntimeError("Empty input csv.")

    # Aggregate by sample id.
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = str(r.get("id") or "")
        if sid == "":
            continue
        layer = safe_float(r.get("block_layer_idx"))
        val = safe_float(r.get(metric))
        if layer is None:
            continue
        li = int(layer)
        ent = by_id.setdefault(
            sid,
            {
                "id": sid,
                "image_id": str(r.get("image_id") or ""),
                "question": str(r.get("question") or ""),
                "answer_gt": str(r.get("answer_gt") or ""),
                "answer_pred": str(r.get("answer_pred") or ""),
                "pred_text": str(r.get("pred_text") or ""),
                "group": "other",
                "values_by_layer": {},
            },
        )
        if parse_bool(r.get(args.group_pos_col)):
            ent["group"] = "fp_hall"
        elif parse_bool(r.get(args.group_neg_col)) and ent.get("group") != "fp_hall":
            ent["group"] = "tp_yes"
        if val is not None:
            ent["values_by_layer"][li] = float(val)

    per_sample_layers: List[Dict[str, Any]] = []
    per_sample_feat: List[Dict[str, Any]] = []
    for sid, ent in by_id.items():
        vals = dict(ent.get("values_by_layer", {}))

        row_l: Dict[str, Any] = {
            "id": sid,
            "group": ent.get("group"),
            "image_id": ent.get("image_id"),
            "question": ent.get("question"),
            "answer_gt": ent.get("answer_gt"),
            "answer_pred": ent.get("answer_pred"),
            "pred_text": ent.get("pred_text"),
        }
        add_layer_feature_dict(row_l, vals, layers, prefix=metric)
        per_sample_layers.append(row_l)

        feat = build_features_for_sample(vals, layers=layers)
        row_f = dict(row_l)
        row_f.update(feat)
        per_sample_feat.append(row_f)

    # Group stats for all numeric features.
    group_stats: List[Dict[str, Any]] = []
    feature_keys_all = [
        k
        for k in per_sample_feat[0].keys()
        if k
        not in (
            "id",
            "group",
            "image_id",
            "question",
            "answer_gt",
            "answer_pred",
            "pred_text",
        )
    ]
    for g in ("fp_hall", "tp_yes"):
        sub = [r for r in per_sample_feat if str(r.get("group")) == g]
        for key in feature_keys_all:
            vals = [float(r[key]) for r in sub if safe_float(r.get(key)) is not None]
            group_stats.append(
                {
                    "group": g,
                    "feature": key,
                    "n": int(len(vals)),
                    "mean": "" if len(vals) == 0 else float(sum(vals) / float(len(vals))),
                    "std": "" if len(vals) <= 1 else float(std(vals) or 0.0),
                    "min": "" if len(vals) == 0 else float(min(vals)),
                    "max": "" if len(vals) == 0 else float(max(vals)),
                }
            )

    # Evaluate feature separability: fp_hall(positive) vs tp_yes(negative).
    eval_rows: List[Dict[str, Any]] = []
    numeric_keys = [
        k
        for k in per_sample_feat[0].keys()
        if k
        not in (
            "id",
            "group",
            "image_id",
            "question",
            "answer_gt",
            "answer_pred",
            "pred_text",
        )
    ]
    for key in numeric_keys:
        pos = [safe_float(r.get(key)) for r in per_sample_feat if str(r.get("group")) == "fp_hall"]
        neg = [safe_float(r.get(key)) for r in per_sample_feat if str(r.get("group")) == "tp_yes"]
        pos2 = [float(x) for x in pos if x is not None]
        neg2 = [float(x) for x in neg if x is not None]
        if len(pos2) == 0 or len(neg2) == 0:
            continue
        auc_h = auc_hall_high(pos2, neg2)
        ks_h = ks_stat(pos2, neg2)
        if auc_h is None or ks_h is None:
            continue
        auc_b = max(float(auc_h), float(1.0 - auc_h))
        direction = "higher_in_hallucination" if auc_h >= 0.5 else "lower_in_hallucination"
        auc_lo, auc_hi = bootstrap_ci(pos2, neg2, auc_hall_high, int(args.bootstrap), int(args.seed))
        ks_lo, ks_hi = bootstrap_ci(pos2, neg2, ks_stat, int(args.bootstrap), int(args.seed) + 997)
        eval_rows.append(
            {
                "feature": key,
                "n_pos_fp": int(len(pos2)),
                "n_neg_tp": int(len(neg2)),
                "auc_hall_high": float(auc_h),
                "auc_best_dir": float(auc_b),
                "direction": direction,
                "auc_ci95_lo": "" if auc_lo is None else float(auc_lo),
                "auc_ci95_hi": "" if auc_hi is None else float(auc_hi),
                "ks_hall_high": float(ks_h),
                "ks_ci95_lo": "" if ks_lo is None else float(ks_lo),
                "ks_ci95_hi": "" if ks_hi is None else float(ks_hi),
            }
        )
    eval_rows = sorted(eval_rows, key=lambda r: float(r.get("auc_best_dir") or 0.0), reverse=True)

    # Outputs
    out_layers = os.path.join(out_dir, "per_sample_layer_values.csv")
    out_feat = os.path.join(out_dir, "per_sample_trajectory_features.csv")
    out_gs = os.path.join(out_dir, "group_stats.csv")
    out_eval = os.path.join(out_dir, "feature_eval_fp_vs_tp.csv")
    out_summary = os.path.join(out_dir, "summary.json")

    write_csv(out_layers, per_sample_layers)
    write_csv(out_feat, per_sample_feat)
    write_csv(out_gs, group_stats)
    write_csv(out_eval, eval_rows)

    best = eval_rows[0] if len(eval_rows) > 0 else None
    summary = {
        "inputs": {
            "per_layer_trace_csv": in_csv,
            "metric": metric,
            "layers": [int(x) for x in layers],
            "group_pos_col": str(args.group_pos_col),
            "group_neg_col": str(args.group_neg_col),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "counts": {
            "n_rows_input": int(len(rows)),
            "n_unique_ids": int(len(by_id)),
            "n_fp_hall": int(sum(1 for v in by_id.values() if v.get("group") == "fp_hall")),
            "n_tp_yes": int(sum(1 for v in by_id.values() if v.get("group") == "tp_yes")),
            "n_features_eval": int(len(eval_rows)),
        },
        "best_eval": best,
        "outputs": {
            "per_sample_layer_values_csv": out_layers,
            "per_sample_trajectory_features_csv": out_feat,
            "group_stats_csv": out_gs,
            "feature_eval_csv": out_eval,
            "summary_json": out_summary,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", out_layers)
    print("[saved]", out_feat)
    print("[saved]", out_gs)
    print("[saved]", out_eval)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()
