#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


def read_gt_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_q_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == "":
                continue
            obj = json.loads(s)
            qid = str(obj.get("question_id", "")).strip()
            if qid == "":
                continue
            out[qid] = obj
    return out


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_answer(x: str) -> str:
    v = str(x or "").strip().lower()
    if v in {"yes", "no"}:
        return v
    return ""


def make_stratum_key(
    row: Dict[str, str],
    balance_category: bool,
    balance_answer: bool,
) -> Tuple[str, ...]:
    parts: List[str] = []
    if bool(balance_category):
        parts.append(str(row.get("category", "")).strip().lower() or "unknown_category")
    if bool(balance_answer):
        parts.append(str(row.get("answer", "")).strip().lower() or "unknown_answer")
    if len(parts) == 0:
        parts.append("all")
    return tuple(parts)


def allocate_equal_with_remainder(
    n_total: int,
    keys: Sequence[Tuple[str, ...]],
    capacities: Dict[Tuple[str, ...], int],
) -> Dict[Tuple[str, ...], int]:
    if n_total <= 0:
        return {k: 0 for k in keys}
    n_keys = len(keys)
    if n_keys <= 0:
        return {}

    alloc = {k: int(n_total // n_keys) for k in keys}
    rem = int(n_total - sum(alloc.values()))
    # Give remainder to larger strata first for stability.
    order = sorted(keys, key=lambda k: (int(capacities.get(k, 0)), str(k)), reverse=True)
    i = 0
    while rem > 0 and i < len(order):
        alloc[order[i]] += 1
        rem -= 1
        i += 1
        if i >= len(order) and rem > 0:
            i = 0
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strict POPE subset from full GT/Q files.")
    ap.add_argument("--full_gt_csv", type=str, required=True)
    ap.add_argument("--full_q_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance_category", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--balance_answer", type=str, default="true", choices=["true", "false"])
    args = ap.parse_args()

    n_total = int(args.n_total)
    if n_total <= 0:
        raise RuntimeError("--n_total must be > 0")

    balance_category = str(args.balance_category).strip().lower() == "true"
    balance_answer = str(args.balance_answer).strip().lower() == "true"

    rows = read_gt_csv(os.path.abspath(args.full_gt_csv))
    q_map = read_q_jsonl(os.path.abspath(args.full_q_jsonl))

    clean_rows: List[Dict[str, str]] = []
    for r in rows:
        sid = str(r.get("id", "")).strip()
        ans = normalize_answer(r.get("answer", ""))
        if sid == "" or ans == "":
            continue
        if sid not in q_map:
            continue
        rr = dict(r)
        rr["answer"] = ans
        clean_rows.append(rr)

    if len(clean_rows) < n_total:
        raise RuntimeError(f"Not enough valid rows: {len(clean_rows)} < {n_total}")

    by_stratum: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for r in clean_rows:
        k = make_stratum_key(r, balance_category=balance_category, balance_answer=balance_answer)
        by_stratum[k].append(r)

    keys = sorted(by_stratum.keys(), key=lambda x: str(x))
    capacities = {k: len(by_stratum[k]) for k in keys}
    alloc = allocate_equal_with_remainder(n_total=n_total, keys=keys, capacities=capacities)

    for k in keys:
        if int(alloc.get(k, 0)) > int(capacities.get(k, 0)):
            raise RuntimeError(
                f"Allocation exceeds capacity for stratum={k}: need {alloc[k]}, have {capacities[k]}"
            )

    rng = random.Random(int(args.seed))
    selected: List[Dict[str, str]] = []
    for k in keys:
        pool = list(by_stratum[k])
        rng.shuffle(pool)
        selected.extend(pool[: int(alloc.get(k, 0))])

    if len(selected) != n_total:
        raise RuntimeError(f"Internal error: selected={len(selected)} expected={n_total}")

    selected = sorted(selected, key=lambda r: int(float(r["id"])))
    selected_ids = [str(r["id"]).strip() for r in selected]

    q_rows: List[Dict[str, Any]] = []
    for sid in selected_ids:
        q = dict(q_map[sid])
        q["question_id"] = str(q.get("question_id", sid))
        q_rows.append(q)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_gt = os.path.join(out_dir, f"pope_strict_{n_total}_gt.csv")
    out_q = os.path.join(out_dir, f"pope_strict_{n_total}_q.jsonl")
    out_ids = os.path.join(out_dir, f"pope_strict_{n_total}_ids.csv")
    out_summary = os.path.join(out_dir, "summary.json")

    write_csv(out_gt, selected)
    write_jsonl(out_q, q_rows)
    write_csv(out_ids, [{"id": sid} for sid in selected_ids])

    by_cat: Dict[str, int] = defaultdict(int)
    by_ans: Dict[str, int] = defaultdict(int)
    by_str: Dict[str, int] = defaultdict(int)
    for r in selected:
        by_cat[str(r.get("category", "")).strip().lower()] += 1
        by_ans[str(r.get("answer", "")).strip().lower()] += 1
        sk = make_stratum_key(r, balance_category=balance_category, balance_answer=balance_answer)
        by_str[str(sk)] += 1

    summary = {
        "inputs": {
            "full_gt_csv": os.path.abspath(args.full_gt_csv),
            "full_q_jsonl": os.path.abspath(args.full_q_jsonl),
            "n_total": int(n_total),
            "seed": int(args.seed),
            "balance_category": bool(balance_category),
            "balance_answer": bool(balance_answer),
        },
        "counts": {
            "n_rows_full_valid": int(len(clean_rows)),
            "n_selected": int(len(selected)),
            "by_category": dict(sorted(by_cat.items())),
            "by_answer": dict(sorted(by_ans.items())),
            "by_stratum": dict(sorted(by_str.items())),
        },
        "outputs": {
            "subset_gt_csv": out_gt,
            "subset_q_jsonl": out_q,
            "subset_ids_csv": out_ids,
            "summary_json": out_summary,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_gt)
    print("[saved]", out_q)
    print("[saved]", out_ids)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()
