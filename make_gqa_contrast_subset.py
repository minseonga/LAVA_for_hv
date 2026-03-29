#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


def norm_text(x: Any) -> str:
    s = str("" if x is None else x).strip().lower()
    s = re.sub(r"[^a-z0-9\\s]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


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
    ap = argparse.ArgumentParser(description="Build pair-aware GQA 1000 subset for Stage-2 contrast evaluation.")
    ap.add_argument(
        "--full_per_sample_csv",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/artrap_fragility_testdev_balanced_greedy_b1/per_sample.csv",
    )
    ap.add_argument("--questions_json", type=str, default="/home/kms/data/gqa/testdev_balanced_questions.json")
    ap.add_argument("--target_n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_local_pairs_per_group", type=int, default=20)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(int(args.seed))

    # Load full per-sample rows.
    rows: List[Dict[str, Any]] = []
    id_to_row: Dict[str, Dict[str, Any]] = {}
    with open(os.path.abspath(args.full_per_sample_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "").strip()
            if sid == "":
                continue
            rr = dict(r)
            rows.append(rr)
            id_to_row[sid] = rr
    if len(rows) == 0:
        raise RuntimeError("Empty full_per_sample_csv.")

    # Load GQA metadata.
    with open(os.path.abspath(args.questions_json), "r", encoding="utf-8") as f:
        qj = json.load(f)
    if not isinstance(qj, dict):
        raise RuntimeError("questions_json must be dict keyed by question id.")

    valid_ids: Set[str] = set(id_to_row.keys()) & set(str(k) for k in qj.keys())
    if len(valid_ids) == 0:
        raise RuntimeError("No overlapping IDs between per_sample and questions_json.")

    # Helper maps.
    answer_norm: Dict[str, str] = {}
    image_id_map: Dict[str, str] = {}
    is_success_map: Dict[str, bool] = {}
    local_group_map: Dict[str, str] = {}
    for sid in valid_ids:
        rr = id_to_row[sid]
        qq = qj[sid]
        answer_norm[sid] = norm_text(rr.get("answer"))
        image_id_map[sid] = str(rr.get("image_id") or qq.get("imageId") or "").strip()
        is_success_map[sid] = bool(parse_bool(rr.get("is_success")))
        local_group_map[sid] = str((qq.get("groups") or {}).get("local") or "").strip()

    # Pair containers.
    pair_equiv: Set[Tuple[str, str]] = set()
    pair_entailed: Set[Tuple[str, str]] = set()
    pair_local_flip: Set[Tuple[str, str]] = set()

    # equivalent (undirected).
    for sid in valid_ids:
        eqs = qj[sid].get("equivalent") or []
        for t in eqs:
            tt = str(t)
            if tt not in valid_ids or tt == sid:
                continue
            a, b = sorted([sid, tt])
            pair_equiv.add((a, b))

    # entailed (treated undirected for coverage).
    for sid in valid_ids:
        ents = qj[sid].get("entailed") or []
        for t in ents:
            tt = str(t)
            if tt not in valid_ids or tt == sid:
                continue
            a, b = sorted([sid, tt])
            pair_entailed.add((a, b))

    # local_flip pairs: same local group + same image + different gt answer.
    by_group_img: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for sid in valid_ids:
        lg = local_group_map.get(sid, "")
        im = image_id_map.get(sid, "")
        if lg == "" or im == "":
            continue
        by_group_img[(lg, im)].append(sid)

    for _, ids in by_group_img.items():
        m = len(ids)
        if m < 2:
            continue
        candidates: List[Tuple[str, str]] = []
        for i in range(m):
            for j in range(i + 1, m):
                a, b = ids[i], ids[j]
                if answer_norm.get(a, "") == answer_norm.get(b, ""):
                    continue
                x, y = sorted([a, b])
                candidates.append((x, y))
        if len(candidates) == 0:
            continue
        rng.shuffle(candidates)
        for p in candidates[: int(max(1, args.max_local_pairs_per_group))]:
            pair_local_flip.add(p)

    # Selection: prioritize pair coverage.
    target_n = int(max(1, args.target_n))
    selected: Set[str] = set()

    def try_add_pair(a: str, b: str) -> None:
        nonlocal selected
        if len(selected) >= target_n:
            return
        need = int(a not in selected) + int(b not in selected)
        if len(selected) + need <= target_n:
            selected.add(a)
            selected.add(b)

    def consume_pair_set(pset: Set[Tuple[str, str]]) -> None:
        plist = list(pset)
        rng.shuffle(plist)
        for a, b in plist:
            if len(selected) >= target_n:
                break
            try_add_pair(a, b)

    # Priority: equivalent -> local_flip -> entailed.
    consume_pair_set(pair_equiv)
    consume_pair_set(pair_local_flip)
    consume_pair_set(pair_entailed)

    # Fill remainder by high pair-degree nodes, then random.
    degree: Dict[str, int] = defaultdict(int)
    for a, b in (list(pair_equiv) + list(pair_local_flip) + list(pair_entailed)):
        degree[a] += 1
        degree[b] += 1

    candidates = sorted(valid_ids, key=lambda x: (degree.get(x, 0), int(is_success_map.get(x, False))), reverse=True)
    for sid in candidates:
        if len(selected) >= target_n:
            break
        selected.add(sid)

    if len(selected) < target_n:
        rest = [sid for sid in valid_ids if sid not in selected]
        rng.shuffle(rest)
        for sid in rest:
            if len(selected) >= target_n:
                break
            selected.add(sid)

    selected_ids = sorted(selected)

    # Build subset rows preserving original order.
    subset_rows: List[Dict[str, Any]] = []
    selected_set = set(selected_ids)
    for r in rows:
        sid = str(r.get("id") or "").strip()
        if sid in selected_set:
            subset_rows.append(r)

    # Pair stats inside subset.
    def count_pairs(pset: Set[Tuple[str, str]]) -> int:
        return sum(1 for a, b in pset if a in selected_set and b in selected_set)

    n_equiv_in = count_pairs(pair_equiv)
    n_local_in = count_pairs(pair_local_flip)
    n_entailed_in = count_pairs(pair_entailed)

    # stage-2 viability quick stats.
    # local_flip: one-correct-one-wrong pairs.
    n_local_onecw = 0
    for a, b in pair_local_flip:
        if a not in selected_set or b not in selected_set:
            continue
        if is_success_map.get(a, False) != is_success_map.get(b, False):
            n_local_onecw += 1

    out_subset = os.path.join(out_dir, "per_sample.csv")
    out_ids = os.path.join(out_dir, "subset_ids.txt")
    out_summary = os.path.join(out_dir, "summary.json")

    write_csv(out_subset, subset_rows)
    with open(out_ids, "w", encoding="utf-8") as f:
        for sid in selected_ids:
            f.write(f"{sid}\n")

    summary = {
        "inputs": {
            "full_per_sample_csv": os.path.abspath(args.full_per_sample_csv),
            "questions_json": os.path.abspath(args.questions_json),
            "target_n": int(target_n),
            "seed": int(args.seed),
            "max_local_pairs_per_group": int(args.max_local_pairs_per_group),
        },
        "counts": {
            "n_full_rows": int(len(rows)),
            "n_valid_ids": int(len(valid_ids)),
            "n_selected": int(len(selected_ids)),
            "n_subset_rows": int(len(subset_rows)),
            "n_success_subset": int(sum(1 for sid in selected_ids if is_success_map.get(sid, False))),
            "n_failure_subset": int(sum(1 for sid in selected_ids if not is_success_map.get(sid, False))),
            "n_equivalent_pairs_total": int(len(pair_equiv)),
            "n_local_flip_pairs_total": int(len(pair_local_flip)),
            "n_entailed_pairs_total": int(len(pair_entailed)),
            "n_equivalent_pairs_in_subset": int(n_equiv_in),
            "n_local_flip_pairs_in_subset": int(n_local_in),
            "n_entailed_pairs_in_subset": int(n_entailed_in),
            "n_local_flip_one_correct_one_wrong_in_subset": int(n_local_onecw),
        },
        "outputs": {
            "subset_per_sample_csv": out_subset,
            "subset_ids_txt": out_ids,
            "summary_json": out_summary,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", out_subset)
    print("[saved]", out_ids)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()

