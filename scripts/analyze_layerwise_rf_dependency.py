#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_json_int_list(x: Any) -> List[int]:
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            out: List[int] = []
            for v in arr:
                try:
                    out.append(int(v))
                except Exception:
                    continue
            return out
    except Exception:
        return []
    return []


def parse_json_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            out: List[float] = []
            for v in arr:
                try:
                    out.append(float(v))
                except Exception:
                    continue
            return out
    except Exception:
        return []
    return []


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def boolish(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def zscore_vec(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / sd


def rankdata(a: np.ndarray) -> np.ndarray:
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    arr = a[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum() - 1
    counts = np.bincount(dense)
    starts = np.r_[0, counts.cumsum()[:-1]]
    avg = starts + (counts - 1) / 2.0
    ranks = avg[dense] + 1.0
    return ranks[inv]


def auc_from_scores(y: np.ndarray, s: np.ndarray) -> float:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(s.astype(float))
    sum_r_pos = float(np.sum(r[y == 1]))
    auc = (sum_r_pos - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(auc)


def ks_stat(y: np.ndarray, s: np.ndarray) -> float:
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    vals = np.sort(np.unique(s))
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    cdf_pos = np.searchsorted(pos_sorted, vals, side="right") / float(len(pos_sorted))
    cdf_neg = np.searchsorted(neg_sorted, vals, side="right") / float(len(neg_sorted))
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    xm = xa - np.mean(xa)
    ym = ya - np.mean(ya)
    den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if den <= 0:
        return 0.0
    return float(np.sum(xm * ym) / den)


def load_group_map(samples_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(samples_csv)
    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        sid = str(r.get("id", "")).strip()
        if not sid:
            continue
        if boolish(r.get("is_fp_hallucination", 0)):
            out[sid] = "fp_hall"
        elif boolish(r.get("is_tp_yes", 0)):
            out[sid] = "tp_yes"
    return out


def load_role_sets(role_csv: Path) -> Tuple[Dict[str, set], Dict[str, set]]:
    sup: Dict[str, set] = defaultdict(set)
    harm: Dict[str, set] = defaultdict(set)
    with role_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id", "")).strip()
            if not sid:
                continue
            p = safe_int(r.get("candidate_patch_idx"), -1)
            if p < 0:
                continue
            label = str(r.get("role_label", "")).strip().lower()
            if label == "supportive":
                sup[sid].add(p)
            elif label in {"harmful", "assertive"}:
                harm[sid].add(p)
    return sup, harm


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer-wise dependency analysis for RF proxy features.")
    ap.add_argument("--trace_csv", type=str, required=True, help="per_layer_yes_trace.csv")
    ap.add_argument("--samples_csv", type=str, required=True, help="samples_from_baseline.csv (for fp/tp labels)")
    ap.add_argument("--role_csv", type=str, required=True, help="per_patch_role_effect.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_ids", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early_start", type=int, default=1)
    ap.add_argument("--early_end", type=int, default=4)
    ap.add_argument("--a_metric", type=str, default="yes_sim_objpatch_topk")
    ap.add_argument("--attn_idx_col", type=str, default="yes_attn_vis_topk_idx_json")
    ap.add_argument("--attn_w_col", type=str, default="yes_attn_vis_topk_weight_json")
    args = ap.parse_args()

    trace_csv = Path(args.trace_csv)
    samples_csv = Path(args.samples_csv)
    role_csv = Path(args.role_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_map = load_group_map(samples_csv)
    if not group_map:
        raise RuntimeError("No fp_hall/tp_yes ids from samples_csv")
    sup_sets, harm_sets = load_role_sets(role_csv)

    valid_ids = sorted(set(group_map.keys()))
    rng = random.Random(int(args.seed))
    rng.shuffle(valid_ids)
    sample_ids = valid_ids[: min(int(args.num_ids), len(valid_ids))]
    sample_set = set(sample_ids)

    trace = pd.read_csv(trace_csv)
    trace["id"] = trace["id"].astype(str)
    trace = trace[trace["id"].isin(sample_set)].copy()
    if trace.empty:
        raise RuntimeError("No trace rows matched sampled ids")

    recs: List[Dict[str, Any]] = []
    for _, r in trace.iterrows():
        sid = str(r["id"])
        group = group_map.get(sid, "")
        if group not in {"fp_hall", "tp_yes"}:
            continue

        li = safe_int(r.get("block_layer_idx"), -1)
        idxs = parse_json_int_list(r.get(args.attn_idx_col))
        ws = parse_json_float_list(r.get(args.attn_w_col))
        sup = sup_sets.get(sid, set())
        harm = harm_sets.get(sid, set())

        c = 0.0
        d = 0.0
        for i, p in enumerate(idxs):
            w = float(ws[i]) if i < len(ws) else 0.0
            if p in sup:
                c += w
            if p in harm:
                d += w
        a = safe_float(r.get(args.a_metric), 0.0)

        recs.append(
            {
                "id": sid,
                "group": group,
                "block_layer_idx": li,
                "C": float(c),
                "A": float(a),
                "D": float(d),
            }
        )

    feat_df = pd.DataFrame(recs)
    if feat_df.empty:
        raise RuntimeError("No feature rows built")

    e0, e1 = min(args.early_start, args.early_end), max(args.early_start, args.early_end)
    early_ref = (
        feat_df[(feat_df["block_layer_idx"] >= e0) & (feat_df["block_layer_idx"] <= e1)]
        .groupby("id", as_index=False)["C"]
        .mean()
        .rename(columns={"C": "C_early_mean"})
    )
    feat_df = feat_df.merge(early_ref, on="id", how="left")
    feat_df["C_early_mean"] = feat_df["C_early_mean"].fillna(0.0)
    feat_df["B"] = np.maximum(0.0, feat_df["C_early_mean"].values - feat_df["C"].values)

    u_vals = []
    for li, gdf in feat_df.groupby("block_layer_idx"):
        C = gdf["C"].to_numpy(dtype=float)
        A = gdf["A"].to_numpy(dtype=float)
        D = gdf["D"].to_numpy(dtype=float)
        B = gdf["B"].to_numpy(dtype=float)
        U = zscore_vec(C) + zscore_vec(A) - zscore_vec(D) - zscore_vec(B)
        tmp = gdf[["id", "block_layer_idx"]].copy()
        tmp["U"] = U
        u_vals.append(tmp)
    u_df = pd.concat(u_vals, ignore_index=True) if u_vals else pd.DataFrame(columns=["id", "block_layer_idx", "U"])
    feat_df = feat_df.merge(u_df, on=["id", "block_layer_idx"], how="left")

    eval_rows: List[Dict[str, Any]] = []
    for li, gdf in feat_df.groupby("block_layer_idx"):
        y = (gdf["group"].values == "fp_hall").astype(int)
        for m in ["C", "A", "D", "B", "U"]:
            s = gdf[m].to_numpy(dtype=float)
            auc_hi = auc_from_scores(y, s)
            auc_best = max(auc_hi, 1.0 - auc_hi) if math.isfinite(auc_hi) else float("nan")
            direction = "higher_in_fp_hall" if auc_hi >= 0.5 else "lower_in_fp_hall"
            ks = ks_stat(y, s)
            eval_rows.append(
                {
                    "block_layer_idx": int(li),
                    "metric": m,
                    "n": int(len(gdf)),
                    "n_fp": int(np.sum(y == 1)),
                    "n_tp": int(np.sum(y == 0)),
                    "auc_hall_high": float(auc_hi),
                    "auc_best_dir": float(auc_best),
                    "ks": float(ks),
                    "direction": direction,
                }
            )
    eval_df = pd.DataFrame(eval_rows).sort_values(["auc_best_dir", "ks"], ascending=[False, False])

    dep_rows: List[Dict[str, Any]] = []
    pairs = [("C", "A"), ("C", "D"), ("C", "B"), ("A", "D"), ("A", "B"), ("D", "B")]
    for (li, grp), gdf in feat_df.groupby(["block_layer_idx", "group"]):
        for x, y in pairs:
            dep_rows.append(
                {
                    "block_layer_idx": int(li),
                    "group": str(grp),
                    "x": x,
                    "y": y,
                    "pearson": float(pearson(gdf[x].values, gdf[y].values)),
                    "n": int(len(gdf)),
                }
            )
    dep_df = pd.DataFrame(dep_rows)

    mean_df = (
        feat_df.groupby(["block_layer_idx", "group"], as_index=False)[["C", "A", "D", "B", "U"]]
        .mean()
        .sort_values(["block_layer_idx", "group"])
    )

    feat_csv = out_dir / "layerwise_rf_proxy_features.csv"
    eval_csv = out_dir / "layerwise_rf_proxy_eval.csv"
    dep_csv = out_dir / "layerwise_rf_proxy_dependency.csv"
    mean_csv = out_dir / "layerwise_rf_proxy_group_means.csv"
    sampled_ids_csv = out_dir / "sampled_ids.csv"
    summary_json = out_dir / "summary.json"

    feat_df.to_csv(feat_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)
    dep_df.to_csv(dep_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    pd.DataFrame({"id": sample_ids}).to_csv(sampled_ids_csv, index=False)

    best = eval_df.iloc[0].to_dict() if len(eval_df) else {}
    summary = {
        "inputs": {
            "trace_csv": str(trace_csv.resolve()),
            "samples_csv": str(samples_csv.resolve()),
            "role_csv": str(role_csv.resolve()),
            "num_ids": int(args.num_ids),
            "seed": int(args.seed),
            "a_metric": str(args.a_metric),
            "early_start": int(args.early_start),
            "early_end": int(args.early_end),
        },
        "counts": {
            "n_sample_ids": int(len(sample_ids)),
            "n_rows_feature": int(len(feat_df)),
            "n_layers": int(feat_df["block_layer_idx"].nunique()),
            "n_fp_ids": int(sum(1 for sid in sample_ids if group_map.get(sid) == "fp_hall")),
            "n_tp_ids": int(sum(1 for sid in sample_ids if group_map.get(sid) == "tp_yes")),
        },
        "best_eval": best,
        "outputs": {
            "features_csv": str(feat_csv.resolve()),
            "eval_csv": str(eval_csv.resolve()),
            "dependency_csv": str(dep_csv.resolve()),
            "group_means_csv": str(mean_csv.resolve()),
            "sampled_ids_csv": str(sampled_ids_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", feat_csv)
    print("[saved]", eval_csv)
    print("[saved]", dep_csv)
    print("[saved]", mean_csv)
    print("[saved]", sampled_ids_csv)
    print("[saved]", summary_json)
    print("[best]", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
