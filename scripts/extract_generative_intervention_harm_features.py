#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extract_generative_claim_support_delta_features import (
    attach_replay_support_scores,
    load_feature_map,
    normalize_claim_phrase,
    safe_div,
)
from extract_vga_generative_mention_features import build_feature_payload
from frgavr_cleanroom.runtime import (
    CleanroomLlavaRuntime,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    safe_id,
    write_csv,
    write_json,
)


WORD_RE = re.compile(r"[A-Za-z0-9]+")


def mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0
    mu = mean(seq)
    var = sum((float(v) - mu) ** 2 for v in seq) / float(len(seq))
    return float(max(var, 0.0) ** 0.5)


def max_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(max(float(v) for v in values))


def min_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(min(float(v) for v in values))


def sum_vals(values: Iterable[float]) -> float:
    return float(sum(float(v) for v in values))


def tokenize_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(str(text or ""))]


def head_word(text: str) -> str:
    toks = tokenize_words(text)
    return str(toks[-1]) if toks else ""


def claim_type_for_mention(mention_row: Dict[str, Any]) -> str:
    kinds = {str(x).strip() for x in str(mention_row.get("kinds", "")).split("|") if str(x).strip()}
    if {"object_mention", "noun_phrase"} & kinds:
        return "object"
    if "relation_phrase" in kinds:
        return "relation"
    if "count_phrase" in kinds:
        return "count"
    return ""


def claim_key_for_mention(mention_row: Dict[str, Any]) -> str:
    claim_type = claim_type_for_mention(mention_row)
    if not claim_type:
        return ""
    phrase = normalize_claim_phrase(str(mention_row.get("text", "")))
    if not phrase:
        return ""
    return f"{claim_type}::{phrase}"


def claim_map_from_payload(
    payload: Dict[str, Any],
    *,
    runtime: CleanroomLlavaRuntime,
    image: Any,
    question: str,
) -> Dict[str, Dict[str, Any]]:
    mention_rows = [dict(row) for row in payload.get("mention_rows", [])]
    max_content_idx = int(payload.get("max_content_idx", 0))
    out: Dict[str, Dict[str, Any]] = {}
    support_cache: Dict[str, Dict[str, float]] = {}

    for row in mention_rows:
        key = claim_key_for_mention(row)
        if not key:
            continue
        claim_type, replay_text = key.split("::", 1)
        replay_text = str(replay_text).strip()
        item = {
            "key": key,
            "claim_type": claim_type,
            "text": replay_text,
            "replay_text": replay_text,
            "head": head_word(replay_text),
            "first_idx": int(row.get("first_idx", 0)),
            "last_idx": int(row.get("last_idx", 0)),
            "last_pos_frac": float(
                safe_div(float(int(row.get("last_idx", 0))), float(max(1, max_content_idx)))
            ),
            "n_occurrences": 1,
        }
        prev = out.get(key)
        if prev is None:
            out[key] = item
            continue
        prev["first_idx"] = int(min(int(prev["first_idx"]), int(item["first_idx"])))
        prev["last_idx"] = int(max(int(prev["last_idx"]), int(item["last_idx"])))
        prev["last_pos_frac"] = float(max(float(prev["last_pos_frac"]), float(item["last_pos_frac"])))
        prev["n_occurrences"] = int(prev.get("n_occurrences", 1)) + 1

    attach_replay_support_scores(
        out,
        runtime=runtime,
        image=image,
        question=question,
        support_cache=support_cache,
    )
    return out


def low_support_items(claims: Sequence[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    return [c for c in claims if float(c.get("support_score", 0.0)) <= float(threshold)]


def late_items(claims: Sequence[Dict[str, Any]], min_pos_frac: float) -> List[Dict[str, Any]]:
    return [c for c in claims if float(c.get("last_pos_frac", 0.0)) >= float(min_pos_frac)]


def repetition_density(claims: Sequence[Dict[str, Any]]) -> float:
    heads = [str(c.get("head", "")).strip() for c in claims if str(c.get("head", "")).strip()]
    if not heads:
        return 0.0
    uniq = len(set(heads))
    return float(safe_div(float(len(heads) - uniq), float(max(1, len(heads)))))


def support_stats(claims: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    vals = [float(c.get("support_score", 0.0)) for c in claims]
    return {
        "mean": float(mean(vals)),
        "std": float(std(vals)),
        "min": float(min_or_zero(vals)),
        "max": float(max_or_zero(vals)),
    }


def low_support_summary(claims: Sequence[Dict[str, Any]], *, threshold: float) -> Dict[str, float]:
    low = low_support_items(claims, threshold)
    lowness = [float(1.0 - float(c.get("support_score", 0.0))) for c in low]
    return {
        "count": int(len(low)),
        "rate": float(safe_div(float(len(low)), float(max(1, len(claims))))),
        "sum": float(sum_vals(lowness)),
        "mean": float(mean(lowness)),
    }


def claim_features(
    payload: Dict[str, Any],
    *,
    runtime: CleanroomLlavaRuntime,
    image: Any,
    question: str,
) -> Dict[str, Any]:
    claim_map = claim_map_from_payload(payload, runtime=runtime, image=image, question=question)
    claims = list(claim_map.values())
    object_claims = [c for c in claims if str(c.get("claim_type", "")) == "object"]
    relation_claims = [c for c in claims if str(c.get("claim_type", "")) == "relation"]
    count_claims = [c for c in claims if str(c.get("claim_type", "")) == "count"]

    support = support_stats(claims)
    object_support = support_stats(object_claims)
    relation_support = support_stats(relation_claims)
    count_support = support_stats(count_claims)

    low_all = low_support_summary(claims, threshold=0.5)
    very_low_all = low_support_summary(claims, threshold=0.25)
    low_obj = low_support_summary(object_claims, threshold=0.5)
    low_rel = low_support_summary(relation_claims, threshold=0.5)
    low_cnt = low_support_summary(count_claims, threshold=0.5)

    late_claims = late_items(claims, 0.5)
    late_low = low_support_summary(late_claims, threshold=0.5)
    very_late_claims = late_items(claims, 0.75)
    very_late_low = low_support_summary(very_late_claims, threshold=0.5)

    high_claims = [c for c in claims if float(c.get("support_score", 0.0)) >= 0.5]
    strong_claims = [c for c in claims if float(c.get("support_score", 0.0)) >= 0.75]
    last_high_pos = max((float(c.get("last_pos_frac", 0.0)) for c in high_claims), default=0.0)
    last_strong_pos = max((float(c.get("last_pos_frac", 0.0)) for c in strong_claims), default=0.0)

    low_repetition = repetition_density(low_support_items(claims, 0.5))
    low_obj_repetition = repetition_density(low_support_items(object_claims, 0.5))

    object_relation_gap = float(max(0.0, object_support["mean"] - relation_support["mean"])) if relation_claims else 0.0
    object_count_gap = float(max(0.0, object_support["mean"] - count_support["mean"])) if count_claims else 0.0
    inconsistency_score = float(object_relation_gap + object_count_gap + low_rel["rate"] + low_cnt["rate"])

    return {
        "pair_iharm_n_claims": int(len(claims)),
        "pair_iharm_n_object_claims": int(len(object_claims)),
        "pair_iharm_n_relation_claims": int(len(relation_claims)),
        "pair_iharm_n_count_claims": int(len(count_claims)),
        "pair_iharm_claim_support_mean": float(support["mean"]),
        "pair_iharm_claim_support_std": float(support["std"]),
        "pair_iharm_claim_support_min": float(support["min"]),
        "pair_iharm_claim_support_max": float(support["max"]),
        "pair_iharm_worst_claim_support": float(support["min"]),
        "pair_iharm_low_support_claim_count": int(low_all["count"]),
        "pair_iharm_low_support_claim_rate": float(low_all["rate"]),
        "pair_iharm_low_support_claim_sum": float(low_all["sum"]),
        "pair_iharm_low_support_claim_mean": float(low_all["mean"]),
        "pair_iharm_very_low_support_claim_count": int(very_low_all["count"]),
        "pair_iharm_very_low_support_claim_rate": float(very_low_all["rate"]),
        "pair_iharm_late_low_support_claim_count": int(late_low["count"]),
        "pair_iharm_late_low_support_claim_rate": float(late_low["rate"]),
        "pair_iharm_late_low_support_claim_sum": float(late_low["sum"]),
        "pair_iharm_very_late_low_support_claim_count": int(very_late_low["count"]),
        "pair_iharm_very_late_low_support_claim_rate": float(very_late_low["rate"]),
        "pair_iharm_last_high_support_claim_pos_frac": float(last_high_pos),
        "pair_iharm_tail_after_last_high_support_claim_frac": float(max(0.0, 1.0 - last_high_pos)),
        "pair_iharm_last_strong_support_claim_pos_frac": float(last_strong_pos),
        "pair_iharm_tail_after_last_strong_support_claim_frac": float(max(0.0, 1.0 - last_strong_pos)),
        "pair_iharm_object_support_mean": float(object_support["mean"]),
        "pair_iharm_relation_support_mean": float(relation_support["mean"]),
        "pair_iharm_count_support_mean": float(count_support["mean"]),
        "pair_iharm_low_support_object_claim_rate": float(low_obj["rate"]),
        "pair_iharm_low_support_relation_claim_rate": float(low_rel["rate"]),
        "pair_iharm_low_support_count_claim_rate": float(low_cnt["rate"]),
        "pair_iharm_object_relation_support_gap": float(object_relation_gap),
        "pair_iharm_object_count_support_gap": float(object_count_gap),
        "pair_iharm_count_relation_inconsistency_score": float(inconsistency_score),
        "pair_iharm_low_support_repetition_density": float(low_repetition),
        "pair_iharm_low_support_object_repetition_density": float(low_obj_repetition),
        "pair_iharm_low_support_claim_heads": " || ".join(
            str(c.get("text", "")) for c in low_support_items(claims, 0.5)[:8]
        ),
        "pair_iharm_claim_types": " || ".join(str(c.get("claim_type", "")) for c in claims[:12]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract intervention-only generative harm signature features.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--base_features_csv", type=str, default="")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pred_text_key", type=str, default="auto")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--max_mentions", type=int, default=12)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    intervention_map = load_prediction_text_map(args.intervention_pred_jsonl, text_key=args.pred_text_key)
    base_feature_map = load_feature_map(args.base_features_csv)

    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(args.model_base or None),
        conv_mode=args.conv_mode,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_base_feature = 0
    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        intervention_text = str(intervention_map.get(sample_id, "")).strip()
        image_path = os.path.join(args.image_folder, image_name)

        row: Dict[str, Any] = dict(base_feature_map.get(sample_id, {}))
        if not row:
            if base_feature_map:
                n_missing_base_feature += 1
            row = {"id": sample_id, "image": image_name, "question": question}
        row["pair_iharm_error"] = ""

        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not question:
                raise ValueError("Missing question text.")
            if not intervention_text:
                raise ValueError("Missing intervention prediction text.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = runtime.load_image(image_path)
            payload = build_feature_payload(
                runtime=runtime,
                image_path=image_path,
                question=question,
                candidate_text=intervention_text,
                sample_id=sample_id,
                image_name=image_name,
                image=image,
                max_mentions=int(args.max_mentions),
            )
            row.update(
                claim_features(
                    payload,
                    runtime=runtime,
                    image=image,
                    question=question,
                )
            )
        except Exception as exc:
            n_errors += 1
            row["pair_iharm_error"] = str(exc)

        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[iharm] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")

    if str(args.out_summary_json or "").strip():
        feature_keys = [k for k in rows[0].keys() if k.startswith("pair_iharm_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "base_features_csv": os.path.abspath(args.base_features_csv) if str(args.base_features_csv).strip() else "",
                    "model_path": args.model_path,
                    "model_base": args.model_base,
                    "conv_mode": args.conv_mode,
                    "device": args.device,
                },
                "counts": {
                    "n_questions": int(len(question_rows)),
                    "n_rows": int(len(rows)),
                    "n_iharm_features": int(len(feature_keys)),
                    "n_errors": int(n_errors),
                    "n_missing_base_feature_rows": int(n_missing_base_feature),
                },
                "settings": {
                    "iharm_mode": "intervention_only_claim_replay",
                },
                "outputs": {
                    "feature_csv": os.path.abspath(args.out_csv),
                },
            },
        )
        print(f"[saved] {os.path.abspath(args.out_summary_json)}")


if __name__ == "__main__":
    main()
