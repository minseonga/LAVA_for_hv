#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

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


def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def load_role_sets(role_csv: Path) -> tuple[Dict[str, Set[int]], Dict[str, Set[int]]]:
    sup: Dict[str, Set[int]] = defaultdict(set)
    harm: Dict[str, Set[int]] = defaultdict(set)
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
    ap = argparse.ArgumentParser(description="Build layerwise RF proxy features (C,A,D,B) for all ids.")
    ap.add_argument("--trace_csv", type=str, required=True)
    ap.add_argument("--role_csv", type=str, required=True)
    ap.add_argument("--ids_csv", type=str, default="")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--summary_json", type=str, default="")
    ap.add_argument("--early_start", type=int, default=1)
    ap.add_argument("--early_end", type=int, default=4)
    ap.add_argument("--a_metric", type=str, default="yes_sim_objpatch_topk")
    ap.add_argument("--attn_idx_col", type=str, default="yes_attn_vis_topk_idx_json")
    ap.add_argument("--attn_w_col", type=str, default="yes_attn_vis_topk_weight_json")
    args = ap.parse_args()

    trace = pd.read_csv(args.trace_csv)
    trace["id"] = trace["id"].astype(str)

    ids_set: Set[str] | None = None
    if str(args.ids_csv).strip():
        ids_df = pd.read_csv(args.ids_csv)
        ids_set = set(ids_df[args.id_col].astype(str).tolist())
        trace = trace[trace["id"].isin(ids_set)].copy()

    sup_sets, harm_sets = load_role_sets(Path(args.role_csv))

    recs: List[Dict[str, Any]] = []
    for _, r in trace.iterrows():
        sid = str(r["id"])
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
        recs.append({
            "id": sid,
            "block_layer_idx": li,
            "C": float(c),
            "A": float(a),
            "D": float(d),
        })

    df = pd.DataFrame(recs)
    if df.empty:
        raise RuntimeError("No rows built")

    e0, e1 = min(args.early_start, args.early_end), max(args.early_start, args.early_end)
    early_ref = (
        df[(df["block_layer_idx"] >= e0) & (df["block_layer_idx"] <= e1)]
        .groupby("id", as_index=False)["C"]
        .mean()
        .rename(columns={"C": "C_early_mean"})
    )
    df = df.merge(early_ref, on="id", how="left")
    df["C_early_mean"] = df["C_early_mean"].fillna(0.0)
    df["B"] = np.maximum(0.0, df["C_early_mean"].values - df["C"].values)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[saved]", out_csv)

    if str(args.summary_json).strip():
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "inputs": {
                "trace_csv": str(Path(args.trace_csv).resolve()),
                "role_csv": str(Path(args.role_csv).resolve()),
                "ids_csv": str(Path(args.ids_csv).resolve()) if str(args.ids_csv).strip() else "",
                "a_metric": args.a_metric,
                "early_start": int(e0),
                "early_end": int(e1),
            },
            "counts": {
                "n_rows": int(len(df)),
                "n_ids": int(df["id"].nunique()),
                "n_layers": int(df["block_layer_idx"].nunique()),
            },
            "outputs": {
                "feature_csv": str(out_csv.resolve()),
                "summary_json": str(summary_path.resolve()),
            },
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[saved]", summary_path)


if __name__ == "__main__":
    main()
