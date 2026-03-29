#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return rows

    # Normal jsonl path
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

    if rows:
        return rows

    # Fallback: some outputs are dumped as one giant line with literal "\n"
    # separators between JSON objects.
    for part in text.split('\\n'):
        part = part.strip()
        if not part:
            continue
        try:
            rows.append(json.loads(part))
        except Exception:
            continue
    return rows


def pick_text(row: dict, keys: List[str]) -> str:
    for k in keys:
        v = row.get(k, None)
        if v is not None and str(v).strip() != "":
            return str(v)
    return ""


def sampled_indices(n_total: int, n_subset: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(n_total, n_subset, replace=False)


def normalize_one_category(
    rows: List[dict],
    category: str,
    qid_start: int,
    text_keys: List[str],
    category_size: int,
    subset_size: int,
    sample_seed: int,
) -> List[dict]:
    out: List[dict] = []

    if subset_size > 0 and len(rows) == subset_size:
        # Reproduce POPEDataSet subset sampling used by VISTA/EAZY scripts.
        idx = sampled_indices(category_size, subset_size, sample_seed)
        qids = [qid_start + int(i) for i in idx]
        map_mode = "sampled_subset_indices"
    elif len(rows) == category_size:
        qids = [qid_start + i for i in range(len(rows))]
        map_mode = "full_sequential"
    else:
        # Fallback: sequential in observed row count.
        qids = [qid_start + i for i in range(len(rows))]
        map_mode = "fallback_sequential"

    for i, r in enumerate(rows):
        txt = pick_text(r, text_keys)
        out.append(
            {
                "question_id": str(qids[i]),
                "output": txt,
                "category": category,
                "row_index_in_category": i,
                "qid_map_mode": map_mode,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Normalize category-wise POPE predictions (adversarial/popular/random) into one jsonl with aligned question_id."
    )
    ap.add_argument("--adversarial_jsonl", type=str, required=True)
    ap.add_argument("--popular_jsonl", type=str, required=True)
    ap.add_argument("--random_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument(
        "--text_keys",
        type=str,
        default="output,text,ans,answer",
        help="Comma-separated fallback keys to read generated text from input jsonl rows.",
    )

    ap.add_argument("--adv_start", type=int, default=0)
    ap.add_argument("--pop_start", type=int, default=3000)
    ap.add_argument("--rnd_start", type=int, default=6000)

    ap.add_argument("--category_size", type=int, default=3000)
    ap.add_argument("--adv_subset_size", type=int, default=-1)
    ap.add_argument("--pop_subset_size", type=int, default=-1)
    ap.add_argument("--rnd_subset_size", type=int, default=-1)
    ap.add_argument("--sample_seed", type=int, default=1994)

    args = ap.parse_args()

    text_keys = [x.strip() for x in str(args.text_keys).split(",") if x.strip()]
    if not text_keys:
        text_keys = ["output", "text", "ans", "answer"]

    adv = read_jsonl(args.adversarial_jsonl)
    pop = read_jsonl(args.popular_jsonl)
    rnd = read_jsonl(args.random_jsonl)

    out_rows: List[dict] = []
    out_rows.extend(
        normalize_one_category(
            adv,
            "adversarial",
            args.adv_start,
            text_keys,
            args.category_size,
            args.adv_subset_size,
            args.sample_seed,
        )
    )
    out_rows.extend(
        normalize_one_category(
            pop,
            "popular",
            args.pop_start,
            text_keys,
            args.category_size,
            args.pop_subset_size,
            args.sample_seed,
        )
    )
    out_rows.extend(
        normalize_one_category(
            rnd,
            "random",
            args.rnd_start,
            text_keys,
            args.category_size,
            args.rnd_subset_size,
            args.sample_seed,
        )
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "inputs": {
            "adversarial_jsonl": os.path.abspath(args.adversarial_jsonl),
            "popular_jsonl": os.path.abspath(args.popular_jsonl),
            "random_jsonl": os.path.abspath(args.random_jsonl),
            "text_keys": text_keys,
            "category_size": args.category_size,
            "adv_subset_size": args.adv_subset_size,
            "pop_subset_size": args.pop_subset_size,
            "rnd_subset_size": args.rnd_subset_size,
            "sample_seed": args.sample_seed,
        },
        "counts": {
            "adversarial": len(adv),
            "popular": len(pop),
            "random": len(rnd),
            "total": len(out_rows),
        },
        "qid_ranges": {
            "adversarial": [args.adv_start, args.adv_start + max(0, len(adv) - 1)],
            "popular": [args.pop_start, args.pop_start + max(0, len(pop) - 1)],
            "random": [args.rnd_start, args.rnd_start + max(0, len(rnd) - 1)],
        },
        "output_jsonl": os.path.abspath(args.out_jsonl),
    }
    s_path = os.path.splitext(args.out_jsonl)[0] + ".summary.json"
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", args.out_jsonl)
    print("[saved]", s_path)
    print("[counts]", summary["counts"])


if __name__ == "__main__":
    main()
