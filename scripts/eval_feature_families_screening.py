#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    n = len(labels)
    if n == 0:
        return None
    pairs = [(int(labels[i]), float(scores[i])) for i in range(n)]
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    idx = sorted(range(n), key=lambda i: pairs[i][1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[idx[j]][1] == pairs[idx[i]][1]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[idx[k]] = avg_rank
        i = j
    sum_pos = sum(ranks[i] for i in range(n) if pairs[i][0] == 1)
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    n = len(labels)
    if n == 0:
        return None
    pairs = [(int(labels[i]), float(scores[i])) for i in range(n)]
    n_pos = sum(1 for y, _ in pairs if y == 1)
    if n_pos == 0:
        return None
    pairs.sort(key=lambda x: x[1], reverse=True)
    tp = 0
    sum_prec = 0.0
    for i, (y, _) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            sum_prec += tp / float(i)
    return float(sum_prec / float(n_pos))


def ks_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pos = sorted(float(scores[i]) for i in range(len(scores)) if int(labels[i]) == 1)
    neg = sorted(float(scores[i]) for i in range(len(scores)) if int(labels[i]) == 0)
    if len(pos) == 0 or len(neg) == 0:
        return None
    vals = sorted(set(pos + neg))
    i = j = 0
    n_pos = len(pos)
    n_neg = len(neg)
    dmax = 0.0
    for v in vals:
        while i < n_pos and pos[i] <= v:
            i += 1
        while j < n_neg and neg[j] <= v:
            j += 1
        d = abs(i / float(n_pos) - j / float(n_neg))
        if d > dmax:
            dmax = d
    return float(dmax)


def family_columns() -> Dict[str, List[str]]:
    A = [
        "obj_token_prob_max",
        "obj_token_prob_mean",
        "obj_token_prob_lse",
        "obj_token_prob_topkmean",
        "entropy_score",
        "G_entropy",
        "G_top1_mass",
        "G_top5_mass",
        "G_effective_support_size",
    ]
    B = [
        "early_attn_mean",
        "late_attn_mean",
        "late_uplift",
        "late_topk_persistence",
        "peak_layer_idx",
        "peak_before_final",
        "late_rank_std",
        "argmax_patch_flip_count",
        "persistence_after_peak",
    ]
    C = [
        "faithful_head_attn_mean",
        "faithful_head_attn_topkmean",
        "faithful_head_coverage",
        "faithful_minus_global_attn",
        "faithful_n_points",
    ]
    D = [
        "harmful_head_attn_mean",
        "harmful_head_attn_topkmean",
        "harmful_head_coverage",
        "harmful_minus_global_attn",
        "harmful_minus_faithful",
        "harmful_n_points",
    ]
    E = [
        "supportive_outside_G",
        "harmful_inside_G",
        "guidance_mismatch_score",
        "context_need_score",
        "G_overfocus",
        "faithful_on_G_mass",
        "faithful_on_nonG_mass",
        "harmful_on_G_mass",
        "harmful_on_nonG_mass",
        "faithful_G_alignment",
        "harmful_G_alignment",
    ]
    return {"A": A, "B": B, "C": C, "D": D, "E": E}


def build_label(row: Dict[str, Any], mode: str, custom_col: str) -> Optional[int]:
    if mode == "fp_vs_tp_yes":
        fp = safe_int(row.get("target_is_fp_hallucination"), None)
        tp = safe_int(row.get("target_is_tp_yes"), None)
        if fp is None or tp is None:
            return None
        if fp == 1:
            return 1
        if tp == 1:
            return 0
        return None
    if mode == "incorrect_vs_correct":
        c = safe_int(row.get("target_is_correct"), None)
        if c is None:
            return None
        return 0 if c == 1 else 1
    v = safe_int(row.get(custom_col), None)
    if v in {0, 1}:
        return int(v)
    return None


def apply_split_filter(rows: List[Dict[str, Any]], split_filter: str) -> List[Dict[str, Any]]:
    if split_filter == "all":
        return rows
    out = []
    for r in rows:
        sp = str(r.get("split", "")).strip().lower()
        if sp == split_filter:
            out.append(r)
    return out


def feature_eval_rows(rows: List[Dict[str, Any]], col: str, labels: List[int]) -> Tuple[List[int], List[float]]:
    ys: List[int] = []
    xs: List[float] = []
    for r, y in zip(rows, labels):
        v = safe_float(r.get(col), None)
        if v is None:
            continue
        ys.append(int(y))
        xs.append(float(v))
    return ys, xs


def zscore(vs: List[float], eps: float = 1e-6) -> List[float]:
    if len(vs) == 0:
        return []
    m = sum(vs) / float(len(vs))
    if len(vs) <= 1:
        return [0.0 for _ in vs]
    var = sum((x - m) * (x - m) for x in vs) / float(len(vs) - 1)
    s = math.sqrt(max(var, eps))
    return [(x - m) / s for x in vs]


def composite_score(rows: List[Dict[str, Any]], labels: List[int], cols: List[str]) -> Tuple[List[int], List[float], List[str]]:
    # Build per-feature sign by mean-difference direction on available points, then average signed z.
    valid_cols: List[str] = []
    per_col_values: Dict[str, List[Optional[float]]] = {}
    for c in cols:
        vals = [safe_float(r.get(c), None) for r in rows]
        if sum(1 for v in vals if v is not None) >= 10:
            valid_cols.append(c)
            per_col_values[c] = vals
    if len(valid_cols) == 0:
        return [], [], []

    # Precompute z per col (missing -> None)
    zvals_by_col: Dict[str, List[Optional[float]]] = {}
    sign_by_col: Dict[str, float] = {}
    for c in valid_cols:
        vals = per_col_values[c]
        idx = [i for i, v in enumerate(vals) if v is not None]
        arr = [float(vals[i]) for i in idx]
        zarr = zscore(arr)
        zfull: List[Optional[float]] = [None for _ in vals]
        for i, z in zip(idx, zarr):
            zfull[i] = float(z)
        zvals_by_col[c] = zfull

        pos = [float(vals[i]) for i in idx if labels[i] == 1]
        neg = [float(vals[i]) for i in idx if labels[i] == 0]
        if len(pos) == 0 or len(neg) == 0:
            sign_by_col[c] = 1.0
        else:
            sign_by_col[c] = 1.0 if (sum(pos) / len(pos)) >= (sum(neg) / len(neg)) else -1.0

    ys: List[int] = []
    xs: List[float] = []
    for i, y in enumerate(labels):
        parts = []
        for c in valid_cols:
            zc = zvals_by_col[c][i]
            if zc is not None:
                parts.append(sign_by_col[c] * float(zc))
        if len(parts) == 0:
            continue
        ys.append(int(y))
        xs.append(float(sum(parts) / float(len(parts))))
    return ys, xs, valid_cols


def main() -> None:
    ap = argparse.ArgumentParser(description="Family-wise screening for A/B/C/D/E features.")
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--target_mode", type=str, default="fp_vs_tp_yes", choices=["fp_vs_tp_yes", "incorrect_vs_correct", "custom_col"])
    ap.add_argument("--target_col", type=str, default="target_binary")
    ap.add_argument("--split_filter", type=str, default="all", choices=["all", "calib", "eval"])
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_csv(os.path.abspath(args.features_csv))
    rows = apply_split_filter(rows, str(args.split_filter))

    # build labels and aligned rows
    use_rows: List[Dict[str, Any]] = []
    labels: List[int] = []
    for r in rows:
        y = build_label(r, mode=str(args.target_mode), custom_col=str(args.target_col))
        if y is None:
            continue
        use_rows.append(r)
        labels.append(int(y))

    fam = family_columns()
    combos: Dict[str, List[str]] = {
        "A+E": fam["A"] + fam["E"],
        "C+D": fam["C"] + fam["D"],
        "A+C+D": fam["A"] + fam["C"] + fam["D"],
        "A+B+C+D+E": fam["A"] + fam["B"] + fam["C"] + fam["D"] + fam["E"],
    }

    single_rows: List[Dict[str, Any]] = []
    composite_rows: List[Dict[str, Any]] = []

    # 1) feature-wise inside each family
    for fk, cols in fam.items():
        for c in cols:
            ys, xs = feature_eval_rows(use_rows, c, labels)
            if len(ys) < 20:
                continue
            auc = auc_from_scores(ys, xs)
            apv = average_precision(ys, xs)
            ks = ks_from_scores(ys, xs)
            if auc is None:
                continue
            auc_best = max(float(auc), 1.0 - float(auc))
            # flip-based best AP
            ap_flip = average_precision(ys, [-v for v in xs])
            ap_best = max(float(apv or 0.0), float(ap_flip or 0.0))
            single_rows.append(
                {
                    "family": fk,
                    "feature": c,
                    "n": len(ys),
                    "auc": float(auc),
                    "auc_best_dir": float(auc_best),
                    "ks": None if ks is None else float(ks),
                    "ap": None if apv is None else float(apv),
                    "ap_best_dir": float(ap_best),
                }
            )

    # 2) family-only composite
    for fk, cols in list(fam.items()) + list(combos.items()):
        ys, xs, valid_cols = composite_score(use_rows, labels, cols)
        if len(ys) < 20:
            continue
        auc = auc_from_scores(ys, xs)
        apv = average_precision(ys, xs)
        ks = ks_from_scores(ys, xs)
        if auc is None:
            continue
        auc_best = max(float(auc), 1.0 - float(auc))
        ap_flip = average_precision(ys, [-v for v in xs])
        ap_best = max(float(apv or 0.0), float(ap_flip or 0.0))
        composite_rows.append(
            {
                "set_name": fk,
                "n": len(ys),
                "n_features_used": len(valid_cols),
                "features_used": "|".join(valid_cols),
                "auc": float(auc),
                "auc_best_dir": float(auc_best),
                "ks": None if ks is None else float(ks),
                "ap": None if apv is None else float(apv),
                "ap_best_dir": float(ap_best),
            }
        )

    out_single = os.path.join(args.out_dir, "family_single_feature_metrics.csv")
    out_comp = os.path.join(args.out_dir, "family_composite_metrics.csv")
    out_sum = os.path.join(args.out_dir, "summary.json")
    write_csv(out_single, single_rows)
    write_csv(out_comp, composite_rows)

    best_single = None
    if len(single_rows) > 0:
        best_single = sorted(single_rows, key=lambda r: float(r.get("auc_best_dir", 0.0)), reverse=True)[0]
    best_comp = None
    if len(composite_rows) > 0:
        best_comp = sorted(composite_rows, key=lambda r: float(r.get("auc_best_dir", 0.0)), reverse=True)[0]

    summary = {
        "inputs": {
            "features_csv": os.path.abspath(args.features_csv),
            "target_mode": str(args.target_mode),
            "target_col": str(args.target_col),
            "split_filter": str(args.split_filter),
        },
        "counts": {
            "n_rows_input": len(rows),
            "n_rows_labeled": len(use_rows),
            "n_pos": int(sum(labels)),
            "n_neg": int(len(labels) - sum(labels)),
        },
        "best_single_feature": best_single,
        "best_composite_set": best_comp,
        "outputs": {
            "family_single_feature_metrics_csv": out_single,
            "family_composite_metrics_csv": out_comp,
            "summary_json": out_sum,
        },
    }
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_single)
    print("[saved]", out_comp)
    print("[saved]", out_sum)


if __name__ == "__main__":
    main()

