#!/usr/bin/env python
"""
make_pope_style_discovery_set.py

Creates a POPE-style yes/no question dataset from COCO train2014 images.
COCO train2014 is DISJOINT from POPE/CHAIR/AMBER eval images (which use val2014).

Outputs JSONL files in format:
    {"image": "COCO_train2014_000000123456.jpg", "text": "Is there a dog in the image?", "label": "yes"}

Two sampling strategies:
  - random:     negatives drawn uniformly from all COCO categories not present in the image
  - adversarial: negatives drawn from categories that co-occur frequently with present categories,
                  but are NOT actually in the image

Usage:
    python scripts/make_pope_style_discovery_set.py \
        --ann_file /home/kms/data/images/mscoco/annotations/instances_train2014.json \
        --image_folder /home/kms/data/images/mscoco/images/train2014 \
        --out_dir /home/kms/LLaVA_calibration/experiments/pope_discovery \
        --n_images 500 \
        --n_questions_per_image 6 \
        --seed 42
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_file", type=str,
                    default="/home/kms/data/images/mscoco/annotations/instances_train2014.json")
    ap.add_argument("--image_folder", type=str,
                    default="/home/kms/data/images/mscoco/images/train2014")
    ap.add_argument("--out_dir", type=str,
                    default="/home/kms/LLaVA_calibration/experiments/pope_discovery")
    ap.add_argument("--n_images", type=int, default=500,
                    help="Number of images to sample from train2014")
    ap.add_argument("--n_questions_per_image", type=int, default=6,
                    help="Total yes/no questions per image (must be even: half yes, half no)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # ── 1. Load COCO train2014 annotations ──────────────────────────
    print(f"[DEBUG] Loading annotations from {args.ann_file}")
    with open(args.ann_file) as f:
        coco = json.load(f)

    # Build category id -> name mapping
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    all_categories = list(cat_id_to_name.values())
    all_cat_set = set(all_categories)
    print(f"[DEBUG] Total COCO categories: {len(all_categories)}")

    # Build image_id -> set of category names present
    img_to_cats = defaultdict(set)
    for ann in coco["annotations"]:
        img_to_cats[ann["image_id"]].add(cat_id_to_name[ann["category_id"]])

    # Build image_id -> filename
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # Filter to images that actually exist on disk
    valid_img_ids = []
    for img_id, fname in img_id_to_file.items():
        # Only include images that have at least 2 categories (so we can make yes/no pairs)
        if len(img_to_cats[img_id]) >= 2 and os.path.exists(os.path.join(args.image_folder, fname)):
            valid_img_ids.append(img_id)

    print(f"[DEBUG] Valid train2014 images with annotations: {len(valid_img_ids)}")

    # Sub-sample
    sampled_ids = random.sample(valid_img_ids, min(args.n_images, len(valid_img_ids)))
    print(f"[DEBUG] Sampled {len(sampled_ids)} images for discovery set")

    # ── 2. Build co-occurrence table (for adversarial negatives) ────
    print("[DEBUG] Building co-occurrence table...")
    cooccur = Counter()   # (catA, catB) -> count of images they both appear in
    for cats in img_to_cats.values():
        cat_list = sorted(cats)
        for i, a in enumerate(cat_list):
            for b in cat_list[i+1:]:
                cooccur[(a, b)] += 1
                cooccur[(b, a)] += 1

    def adversarial_negatives(present_cats, n):
        """Pick n categories that co-occur most often with present cats but are absent."""
        absent = all_cat_set - present_cats
        scores = Counter()
        for pc in present_cats:
            for ac in absent:
                scores[ac] += cooccur.get((pc, ac), 0)
        # Return top-n by co-occurrence score (break ties randomly)
        top = sorted(absent, key=lambda c: (-scores[c], random.random()))
        return top[:n]

    def random_negatives(present_cats, n):
        absent = list(all_cat_set - present_cats)
        random.shuffle(absent)
        return absent[:n]

    # ── 3. Generate questions ────────────────────────────────────────
    n_half = args.n_questions_per_image // 2  # half yes, half no

    random_rows = []
    adversarial_rows = []

    for img_id in sampled_ids:
        fname = img_id_to_file[img_id]
        present = img_to_cats[img_id]
        present_list = list(present)

        # Yes questions: pick from actually present categories
        yes_cats = random.sample(present_list, min(n_half, len(present_list)))

        # ── Random negatives ──
        no_cats_rand = random_negatives(present, n_half)
        for cat in yes_cats:
            random_rows.append({"image": fname,
                                 "text": f"Is there a {cat} in the image?",
                                 "label": "yes"})
        for cat in no_cats_rand:
            random_rows.append({"image": fname,
                                 "text": f"Is there a {cat} in the image?",
                                 "label": "no"})

        # ── Adversarial negatives ──
        no_cats_adv = adversarial_negatives(present, n_half)
        for cat in yes_cats:
            adversarial_rows.append({"image": fname,
                                      "text": f"Is there a {cat} in the image?",
                                      "label": "yes"})
        for cat in no_cats_adv:
            adversarial_rows.append({"image": fname,
                                      "text": f"Is there a {cat} in the image?",
                                      "label": "no"})

    # Shuffle
    random.shuffle(random_rows)
    random.shuffle(adversarial_rows)

    # ── 4. Save ──────────────────────────────────────────────────────
    for split_name, rows in [("random", random_rows), ("adversarial", adversarial_rows)]:
        out_path = os.path.join(args.out_dir, f"discovery_{split_name}.jsonl")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        yes_count = sum(1 for r in rows if r["label"] == "yes")
        no_count = sum(1 for r in rows if r["label"] == "no")
        print(f"[DEBUG] [{split_name}] {len(rows)} questions saved to {out_path}")
        print(f"          yes={yes_count}, no={no_count}, balance={yes_count/len(rows):.2f}")

    print("\n[DEBUG] Done. Discovery set ready.")

if __name__ == "__main__":
    main()
