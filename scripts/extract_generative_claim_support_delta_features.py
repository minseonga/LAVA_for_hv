#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from extract_vga_generative_mention_features import (
    STOPWORDS,
    build_feature_payload,
    is_object_word,
)
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


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out


def tokenize_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(str(text or ""))]


def mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def percentile_scores(values: Sequence[float], *, higher_better: bool) -> List[float]:
    seq = [float(v) for v in values]
    n = len(seq)
    if n <= 1:
        return [1.0 for _ in seq]
    indexed = list(enumerate(seq))
    indexed.sort(key=lambda x: x[1], reverse=bool(higher_better))
    out = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_pos = (float(i) + float(j - 1)) / 2.0
        score = 1.0 - float(avg_pos / float(max(1, n - 1)))
        for k in range(i, j):
            out[indexed[k][0]] = float(score)
        i = j
    return out


def claim_head(text: str) -> str:
    toks = tokenize_words(text)
    object_toks = [tok for tok in toks if is_object_word(tok)]
    if object_toks:
        return str(object_toks[-1])
    content = [tok for tok in toks if tok not in STOPWORDS]
    if content:
        return str(content[-1])
    return ""


def claim_key_for_mention(mention_row: Dict[str, Any]) -> str:
    kinds = {str(x).strip() for x in str(mention_row.get("kinds", "")).split("|") if str(x).strip()}
    if not ({"object_mention", "noun_phrase"} & kinds):
        return ""
    head = claim_head(str(mention_row.get("text", "")))
    if not head:
        return ""
    return f"object_core::{head}"


def enrich_support_scores(mention_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in mention_rows]
    if not rows:
        return rows
    lp_vals = [float(maybe_float(row.get("lp_min")) or 0.0) for row in rows]
    gap_vals = [float(maybe_float(row.get("gap_min")) or 0.0) for row in rows]
    ent_vals = [float(maybe_float(row.get("ent_max")) or 0.0) for row in rows]
    tail_vals = [float(maybe_float(row.get("lp_tail_gap")) or 0.0) for row in rows]

    lp_pct = percentile_scores(lp_vals, higher_better=True)
    gap_pct = percentile_scores(gap_vals, higher_better=True)
    ent_pct = percentile_scores(ent_vals, higher_better=False)
    tail_pct = percentile_scores(tail_vals, higher_better=True)

    for idx, row in enumerate(rows):
        row["support_lp_pct"] = float(lp_pct[idx])
        row["support_gap_pct"] = float(gap_pct[idx])
        row["support_ent_pct"] = float(ent_pct[idx])
        row["support_tail_pct"] = float(tail_pct[idx])
        row["support_score"] = float(mean([lp_pct[idx], gap_pct[idx], ent_pct[idx], tail_pct[idx]]))
    return rows


def claim_map_from_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mention_rows = enrich_support_scores(payload.get("mention_rows", []))
    max_content_idx = int(payload.get("max_content_idx", 0))
    out: Dict[str, Dict[str, Any]] = {}
    for row in mention_rows:
        key = claim_key_for_mention(row)
        if not key:
            continue
        item = {
            "key": key,
            "text": str(row.get("text", "")),
            "support_score": float(row.get("support_score", 0.0)),
            "support_lp_pct": float(row.get("support_lp_pct", 0.0)),
            "support_gap_pct": float(row.get("support_gap_pct", 0.0)),
            "support_ent_pct": float(row.get("support_ent_pct", 0.0)),
            "support_tail_pct": float(row.get("support_tail_pct", 0.0)),
            "first_idx": int(row.get("first_idx", 0)),
            "last_idx": int(row.get("last_idx", 0)),
            "last_pos_frac": float(safe_div(float(int(row.get("last_idx", 0))), float(max(1, max_content_idx)))),
        }
        prev = out.get(key)
        if prev is None or float(item["support_score"]) > float(prev["support_score"]):
            out[key] = item
    return out


def sum_vals(items: Sequence[float]) -> float:
    return float(sum(float(x) for x in items))


def max_or_zero(items: Sequence[float]) -> float:
    if not items:
        return 0.0
    return float(max(float(x) for x in items))


def count_ge(items: Sequence[float], threshold: float) -> int:
    return int(sum(1 for x in items if float(x) >= float(threshold)))


def claim_delta_features(
    base_payload: Dict[str, Any],
    int_payload: Dict[str, Any],
) -> Dict[str, Any]:
    base_claims = claim_map_from_payload(base_payload)
    int_claims = claim_map_from_payload(int_payload)
    base_keys = set(base_claims.keys())
    int_keys = set(int_claims.keys())
    dropped = sorted(base_keys - int_keys)
    added = sorted(int_keys - base_keys)
    shared = sorted(base_keys & int_keys)

    dropped_supports = [float(base_claims[key]["support_score"]) for key in dropped]
    added_unsupported = [float(1.0 - float(int_claims[key]["support_score"])) for key in added]
    shared_weaken = [
        float(max(0.0, float(base_claims[key]["support_score"]) - float(int_claims[key]["support_score"])))
        for key in shared
    ]

    strong_dropped = [key for key in dropped if float(base_claims[key]["support_score"]) >= 0.5]
    strong_last_pos = max((float(base_claims[key]["last_pos_frac"]) for key in strong_dropped), default=0.0)
    any_last_pos = max((float(base_claims[key]["last_pos_frac"]) for key in dropped), default=0.0)

    return {
        "pair_claimdelta_n_base_object_claims": int(len(base_keys)),
        "pair_claimdelta_n_int_object_claims": int(len(int_keys)),
        "pair_claimdelta_n_shared_object_claims": int(len(shared)),
        "pair_claimdelta_n_dropped_object_claims": int(len(dropped)),
        "pair_claimdelta_n_added_object_claims": int(len(added)),
        "pair_claimdelta_supported_drop_sum": float(sum_vals(dropped_supports)),
        "pair_claimdelta_supported_drop_mean": float(mean(dropped_supports)),
        "pair_claimdelta_supported_drop_max": float(max_or_zero(dropped_supports)),
        "pair_claimdelta_supported_drop_rate": float(safe_div(sum_vals(dropped_supports), float(max(1, len(base_keys))))),
        "pair_claimdelta_supported_drop_count_ge_050": int(count_ge(dropped_supports, 0.5)),
        "pair_claimdelta_supported_drop_count_ge_075": int(count_ge(dropped_supports, 0.75)),
        "pair_claimdelta_supported_drop_frac_ge_050": float(
            safe_div(float(count_ge(dropped_supports, 0.5)), float(max(1, len(base_keys))))
        ),
        "pair_claimdelta_supported_drop_frac_ge_075": float(
            safe_div(float(count_ge(dropped_supports, 0.75)), float(max(1, len(base_keys))))
        ),
        "pair_claimdelta_unsupported_add_sum": float(sum_vals(added_unsupported)),
        "pair_claimdelta_unsupported_add_mean": float(mean(added_unsupported)),
        "pair_claimdelta_unsupported_add_max": float(max_or_zero(added_unsupported)),
        "pair_claimdelta_unsupported_add_rate": float(safe_div(sum_vals(added_unsupported), float(max(1, len(int_keys))))),
        "pair_claimdelta_unsupported_add_count_ge_050": int(count_ge(added_unsupported, 0.5)),
        "pair_claimdelta_shared_weaken_sum": float(sum_vals(shared_weaken)),
        "pair_claimdelta_shared_weaken_mean": float(mean(shared_weaken)),
        "pair_claimdelta_shared_weaken_max": float(max_or_zero(shared_weaken)),
        "pair_claimdelta_shared_weaken_count_ge_025": int(count_ge(shared_weaken, 0.25)),
        "pair_claimdelta_shared_weaken_count_ge_050": int(count_ge(shared_weaken, 0.5)),
        "pair_claimdelta_last_supported_drop_pos_frac": float(strong_last_pos),
        "pair_claimdelta_tail_after_last_supported_drop_frac": float(max(0.0, 1.0 - strong_last_pos)),
        "pair_claimdelta_last_any_drop_pos_frac": float(any_last_pos),
        "pair_claimdelta_tail_after_last_any_drop_frac": float(max(0.0, 1.0 - any_last_pos)),
        "pair_claimdelta_dropped_claim_heads": " || ".join(str(key.split("::", 1)[-1]) for key in dropped[:8]),
        "pair_claimdelta_added_claim_heads": " || ".join(str(key.split("::", 1)[-1]) for key in added[:8]),
    }


def load_feature_map(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not str(path or "").strip():
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = safe_id(row.get("id"))
            if sid:
                rows[sid] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract support-weighted claim delta features for generative fallback control.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--base_features_csv", type=str, default="")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_pred_text_key", type=str, default="auto")
    ap.add_argument("--intervention_pred_text_key", type=str, default="auto")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--max_mentions", type=int, default=12)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    baseline_map = load_prediction_text_map(args.baseline_pred_jsonl, text_key=args.baseline_pred_text_key)
    intervention_map = load_prediction_text_map(args.intervention_pred_jsonl, text_key=args.intervention_pred_text_key)
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
        baseline_text = str(baseline_map.get(sample_id, "")).strip()
        intervention_text = str(intervention_map.get(sample_id, "")).strip()
        image_path = os.path.join(args.image_folder, image_name)

        row: Dict[str, Any] = dict(base_feature_map.get(sample_id, {}))
        if not row and base_feature_map:
            n_missing_base_feature += 1
            row = {"id": sample_id, "image": image_name, "question": question}
        row["pair_claimdelta_error"] = ""

        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not question:
                raise ValueError("Missing question text.")
            if not baseline_text:
                raise ValueError("Missing baseline prediction text.")
            if not intervention_text:
                raise ValueError("Missing intervention prediction text.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            base_payload = build_feature_payload(
                runtime=runtime,
                image_path=image_path,
                question=question,
                candidate_text=baseline_text,
                sample_id=sample_id,
                image_name=image_name,
                max_mentions=int(args.max_mentions),
            )
            int_payload = build_feature_payload(
                runtime=runtime,
                image_path=image_path,
                question=question,
                candidate_text=intervention_text,
                sample_id=sample_id,
                image_name=image_name,
                max_mentions=int(args.max_mentions),
            )
            row.update(claim_delta_features(base_payload, int_payload))
        except Exception as exc:
            n_errors += 1
            row["pair_claimdelta_error"] = str(exc)

        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[claim-delta] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")

    if str(args.out_summary_json or "").strip():
        feature_keys = [k for k in rows[0].keys() if k.startswith("pair_claimdelta_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
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
                    "n_claimdelta_features": int(len(feature_keys)),
                    "n_errors": int(n_errors),
                    "n_missing_base_feature_rows": int(n_missing_base_feature),
                },
                "outputs": {
                    "feature_csv": os.path.abspath(args.out_csv),
                },
            },
        )
        print(f"[saved] {os.path.abspath(args.out_summary_json)}")


if __name__ == "__main__":
    main()
