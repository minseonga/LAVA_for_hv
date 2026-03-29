#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Set


TRUE_SET = {"1", "true", "t", "yes", "y"}


def as_bool(x: object) -> bool:
    return str("" if x is None else x).strip().lower() in TRUE_SET


def load_ids_from_per_sample(path: str) -> List[str]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out: List[str] = []
    for r in rows:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", "")).strip()
        if sid != "":
            out.append(sid)
    return out


def load_ok_by_id(path: str, eval_mode: str) -> Dict[str, bool]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out: Dict[str, bool] = {}
    mode = str(eval_mode).strip().lower()
    for r in rows:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", "")).strip()
        if sid == "":
            continue
        if mode == "strict" and str(r.get("is_success_strict", "")).strip() != "":
            ok = as_bool(r.get("is_success_strict", ""))
        elif mode == "heuristic" and str(r.get("is_success_heuristic", "")).strip() != "":
            ok = as_bool(r.get("is_success_heuristic", ""))
        else:
            ok = as_bool(r.get("is_success", ""))
        out[sid] = bool(ok)
    return out


def sample_uniform(ids: Sequence[str], n: int, rng: random.Random) -> List[str]:
    xs = list(dict.fromkeys([str(x) for x in ids if str(x).strip() != ""]))
    if n <= 0 or n >= len(xs):
        return xs
    return sorted(rng.sample(xs, n))


def sample_stratified_by_ok(ids: Sequence[str], ok_by_id: Dict[str, bool], n: int, rng: random.Random) -> List[str]:
    xs = [str(x) for x in ids if str(x).strip() != ""]
    pos = [sid for sid in xs if sid in ok_by_id and bool(ok_by_id[sid])]
    neg = [sid for sid in xs if sid in ok_by_id and (not bool(ok_by_id[sid]))]
    unk = [sid for sid in xs if sid not in ok_by_id]

    # If labels are missing, fallback to uniform.
    if len(pos) == 0 or len(neg) == 0:
        return sample_uniform(xs, n=n, rng=rng)

    if n <= 0 or n >= len(xs):
        return sorted(list(dict.fromkeys(xs)))

    # Preserve observed class ratio from labeled set.
    p = float(len(pos) / max(1, len(pos) + len(neg)))
    n_pos = int(round(float(n) * p))
    n_neg = int(n - n_pos)
    n_pos = max(0, min(n_pos, len(pos)))
    n_neg = max(0, min(n_neg, len(neg)))

    chosen: Set[str] = set()
    if n_pos > 0:
        chosen.update(rng.sample(pos, n_pos))
    if n_neg > 0:
        chosen.update(rng.sample(neg, n_neg))

    # Fill any deficit from remaining pool.
    deficit = int(n - len(chosen))
    if deficit > 0:
        pool = [sid for sid in xs if sid not in chosen]
        if deficit >= len(pool):
            chosen.update(pool)
        else:
            chosen.update(rng.sample(pool, deficit))

    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create subset id json for fast gate/selector validation.")
    ap.add_argument("--greedy_dir", type=str, required=True, help="Directory containing per_sample.csv")
    ap.add_argument("--expand_dir", type=str, default="", help="Optional expand dir to intersect ids")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--subset_size", type=int, default=1000)
    ap.add_argument("--sample_mode", type=str, default="stratified_ok", choices=["uniform", "stratified_ok"])
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    g_ps = os.path.join(os.path.abspath(args.greedy_dir), "per_sample.csv")
    if not os.path.isfile(g_ps):
        raise RuntimeError(f"missing greedy per_sample.csv: {g_ps}")
    ids = set(load_ids_from_per_sample(g_ps))
    if str(args.expand_dir).strip() != "":
        e_ps = os.path.join(os.path.abspath(args.expand_dir), "per_sample.csv")
        if not os.path.isfile(e_ps):
            raise RuntimeError(f"missing expand per_sample.csv: {e_ps}")
        ids = ids & set(load_ids_from_per_sample(e_ps))
    ids_sorted = sorted(ids)
    if len(ids_sorted) == 0:
        raise RuntimeError("no valid ids after filtering/intersection")

    rng = random.Random(int(args.seed))
    if str(args.sample_mode) == "stratified_ok":
        ok_by_id = load_ok_by_id(g_ps, eval_mode=str(args.eval_mode))
        chosen = sample_stratified_by_ok(ids_sorted, ok_by_id=ok_by_id, n=int(args.subset_size), rng=rng)
    else:
        chosen = sample_uniform(ids_sorted, n=int(args.subset_size), rng=rng)

    out_obj = [{"id": str(sid)} for sid in chosen]
    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print("[saved]", out_path)
    print("[meta] n_total_valid=", len(ids_sorted), "n_subset=", len(chosen), "mode=", args.sample_mode)


if __name__ == "__main__":
    main()

