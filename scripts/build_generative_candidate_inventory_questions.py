#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from extract_generative_semantic_pairwise_features import STOPWORDS, normalize_token, read_prediction_map, tokenize


NON_ENTITY_TERMS = {
    "action",
    "activity",
    "adventure",
    "addition",
    "additionally",
    "adding",
    "alongside",
    "also",
    "among",
    "another",
    "antique",
    "appearance",
    "away",
    "belonging",
    "captur",
    "camaraderie",
    "center",
    "closer",
    "close",
    "creating",
    "cleaning",
    "community",
    "covering",
    "distinctive",
    "drif",
    "due",
    "each",
    "edge",
    "enjoying",
    "experience",
    "excitement",
    "fashioned",
    "featur",
    "five",
    "focus",
    "four",
    "further",
    "giv",
    "ground",
    "gathering",
    "grooming",
    "held",
    "including",
    "includ",
    "indicating",
    "interior",
    "item",
    "landscape",
    "least",
    "left",
    "lively",
    "located",
    "location",
    "lot",
    "mak",
    "mess",
    "middle",
    "model",
    "more",
    "move",
    "multiple",
    "navigat",
    "nearby",
    "off",
    "observing",
    "old",
    "one",
    "open",
    "other",
    "outdoor",
    "parked",
    "parking",
    "parts",
    "place",
    "placed",
    "pile",
    "possibly",
    "presence",
    "present",
    "positioned",
    "portion",
    "posing",
    "public",
    "revealing",
    "right",
    "riding",
    "road",
    "scattered",
    "seen",
    "sense",
    "setting",
    "shared",
    "shaggy",
    "sid",
    "side",
    "significant",
    "sitting",
    "stand",
    "standing",
    "spread",
    "studio",
    "stored",
    "suggest",
    "such",
    "surface",
    "tak",
    "taking",
    "terrain",
    "thi",
    "three",
    "top",
    "toward",
    "two",
    "togetherness",
    "unity",
    "up",
    "out",
    "watching",
    "way",
    "where",
    "whom",
    "yard",
    "young",
}


DEFAULT_PROMPT_TEMPLATE = (
    "From the candidate terms below, select only the terms that are clearly visible "
    "objects or entities in the image. Use exact candidate terms only. Do not add "
    "new terms. If none are visible, answer none.\n"
    "Candidates: {candidates}"
)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def unique_ordered(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        item = str(item or "").strip().lower().replace("_", " ")
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def surface_content_tokens(text: str) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    for pos, surface in enumerate(tokenize(text)):
        norm = normalize_token(surface)
        if len(norm) <= 1 or norm in STOPWORDS or norm in NON_ENTITY_TERMS or norm.isdigit():
            continue
        out.append((norm, surface.lower(), pos))
    return out


def candidate_terms(base_caption: str, int_caption: str, max_candidates: int, candidate_mode: str) -> List[str]:
    base_tokens = surface_content_tokens(base_caption)
    int_units = {norm for norm, _, _ in surface_content_tokens(int_caption)}
    base_only: List[Tuple[str, str, int]] = [
        (norm, surface, pos)
        for norm, surface, pos in base_tokens
        if norm not in int_units and norm not in NON_ENTITY_TERMS and len(norm) > 2
    ]

    unigram_terms = [surface for _, surface, _ in sorted(base_only, key=lambda item: item[2])]
    base_only_by_pos = {pos: norm for norm, _, pos in base_only}
    phrase_terms: List[str] = []
    for (left, left_surface, left_pos), (right, right_surface, right_pos) in zip(base_tokens, base_tokens[1:]):
        if left_pos not in base_only_by_pos and right_pos not in base_only_by_pos:
            continue
        if right_pos - left_pos > 2:
            continue
        if left in NON_ENTITY_TERMS or right in NON_ENTITY_TERMS:
            continue
        if left == right or left in int_units and right in int_units:
            continue
        phrase_terms.append(f"{left_surface} {right_surface}")

    if candidate_mode == "phrase":
        raw_terms = phrase_terms
    elif candidate_mode == "both":
        raw_terms = phrase_terms + unigram_terms
    else:
        raw_terms = unigram_terms
    return unique_ordered(raw_terms)[: int(max_candidates)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build candidate-conditioned image inventory questions from baseline-only "
            "caption content. This avoids CHAIR/COCO vocab at inference and uses one "
            "image-conditioned verifier pass per sample."
        )
    )
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--max_candidates", type=int, default=16)
    ap.add_argument("--candidate_mode", choices=("token", "phrase", "both"), default="token")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    args = ap.parse_args()

    baseline = read_prediction_map(args.baseline_pred_jsonl)
    intervention = read_prediction_map(args.intervention_pred_jsonl)
    ids = sorted(
        set(baseline) & set(intervention),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    if int(args.limit) > 0:
        ids = ids[: int(args.limit)]

    rows: List[Dict[str, Any]] = []
    n_empty = 0
    total_candidates = 0
    for sid in ids:
        candidates = candidate_terms(
            baseline[sid]["text"],
            intervention[sid]["text"],
            max_candidates=int(args.max_candidates),
            candidate_mode=str(args.candidate_mode),
        )
        if not candidates:
            n_empty += 1
        total_candidates += len(candidates)
        cand_text = ", ".join(candidates) if candidates else "none"
        question = str(args.prompt_template).format(candidates=cand_text)
        rows.append(
            {
                "question_id": sid,
                "image_id": sid,
                "image": baseline[sid].get("image") or intervention[sid].get("image", ""),
                "question": question,
                "text": question,
                "label": "",
                "category": "candidate_inventory_probe",
                "candidate_terms_json": json.dumps(candidates, ensure_ascii=False),
                "candidate_terms": "|".join(candidates),
                "baseline_caption": baseline[sid]["text"],
                "intervention_caption": intervention[sid]["text"],
            }
        )

    write_jsonl(args.out_jsonl, rows)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if args.out_summary_json:
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "max_candidates": int(args.max_candidates),
                    "candidate_mode": str(args.candidate_mode),
                    "limit": int(args.limit),
                },
                "counts": {
                    "n_rows": len(rows),
                    "n_empty_candidate_rows": n_empty,
                    "mean_candidates": total_candidates / float(max(1, len(rows))),
                },
                "outputs": {"out_jsonl": os.path.abspath(args.out_jsonl)},
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
