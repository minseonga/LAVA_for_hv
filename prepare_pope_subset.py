#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare a POPE subset JSON compatible with analyze_artrap_pairwise_fragility.py"
    )
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=1000, help="Number of rows to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--hf_dataset",
        type=str,
        default="lmms-lab/POPE",
        help="Hugging Face dataset name",
    )
    ap.add_argument(
        "--hf_config",
        type=str,
        default=None,
        help="HF config name (e.g., default or Full). If omitted, default loader behavior is used.",
    )
    ap.add_argument(
        "--hf_split",
        type=str,
        default="test",
        help="HF split name (e.g., test, adversarial, popular, random)",
    )
    ap.add_argument(
        "--sampling",
        choices=["row", "balanced_category_row"],
        default="row",
        help="Sampling strategy across rows",
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "datasets package is required. Install with: pip install datasets"
        ) from e

    random.seed(int(args.seed))

    if args.hf_config is None:
        ds = load_dataset(args.hf_dataset, split=args.hf_split)
    else:
        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)

    # expected fields in lmms-lab/POPE: id, question, answer, image_source, category
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(ds):
        qid = str(r.get("id", r.get("question_id", i)))
        question = str(r.get("question", ""))
        answer = str(r.get("answer", ""))
        image_source = str(r.get("image_source", ""))
        category = str(r.get("category", ""))
        if image_source == "":
            continue
        rows.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                # analyze_artrap_pairwise_fragility.py uses image_root/{imageId}.jpg
                # POPE image_source is typically COCO_val2014_XXXXXXXXXXXX
                "imageId": image_source,
                "category": category,
            }
        )

    if len(rows) == 0:
        raise RuntimeError("No valid POPE rows loaded.")

    n = int(max(1, args.num_samples))
    if args.sampling == "row":
        if n > len(rows):
            raise ValueError(f"Requested {n} rows but only {len(rows)} available.")
        sampled = random.sample(rows, n)
    else:
        by_cat: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_cat.setdefault(str(r.get("category", "")), []).append(r)
        cats = sorted(by_cat.keys())
        if len(cats) == 0:
            raise RuntimeError("No category found for balanced sampling.")
        # round-robin sampling by category for a simple balanced subset.
        sampled = []
        pools = {c: random.sample(by_cat[c], len(by_cat[c])) for c in cats}
        ptr = {c: 0 for c in cats}
        while len(sampled) < n:
            progressed = False
            for c in cats:
                p = ptr[c]
                if p < len(pools[c]):
                    sampled.append(pools[c][p])
                    ptr[c] = p + 1
                    progressed = True
                    if len(sampled) >= n:
                        break
            if not progressed:
                break
        if len(sampled) < n:
            raise ValueError(
                f"Requested {n} rows but balanced round-robin produced only {len(sampled)}."
            )

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    uniq_images = len({str(r["imageId"]) for r in sampled})
    print("[saved]", out_json)
    print(
        f"[info] sampled_rows={len(sampled)} unique_images_in_subset={uniq_images} "
        f"hf_dataset={args.hf_dataset} config={args.hf_config} split={args.hf_split}"
    )


if __name__ == "__main__":
    main()

