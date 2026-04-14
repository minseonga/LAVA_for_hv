#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_caption_conditioned_object_extraction_proxy import normalize_object, split_object_list
from extract_generative_semantic_pairwise_features import (
    GENERIC_FILLERS,
    STOPWORDS,
    content_tokens,
    read_prediction_map,
    write_csv,
    write_json,
)


HIGH_FREQ_OBJECTS = {
    "person",
    "dining table",
    "car",
    "chair",
    "bottle",
    "cup",
    "sports ball",
    "cell phone",
}

GENERIC_OBJECTS = {
    "person",
    "people",
    "man",
    "woman",
    "child",
    "thing",
    "object",
    "item",
    "area",
}

GENERIC_PHRASES = (
    "there is",
    "there are",
    "appears to be",
    "can be seen",
    "visible",
    "various",
    "several",
    "many",
    "some",
    "a number of",
    "in the background",
    "in the foreground",
)

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']*")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        import csv

        return list(csv.DictReader(f))


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
                if int(limit) > 0 and len(rows) >= int(limit):
                    break
    return rows


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def norm_id(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def split_object_list_all(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
    text = re.sub(r"^\s*(?:objects?|object list|answer|output)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:and|or)\s+others?\b", "", text, flags=re.IGNORECASE)
    raw_parts = re.split(r"[,;\n]+", text)
    out: List[str] = []
    for part in raw_parts:
        obj = normalize_object(part)
        if obj:
            out.append(obj)
    return out


def object_family(obj: str) -> str:
    parts = str(obj or "").split()
    return parts[-1] if parts else ""


def object_stats(prefix: str, objects: Sequence[str], raw_objects: Sequence[str], out: Dict[str, Any]) -> None:
    unique = ordered_unique(objects)
    raw = [obj for obj in raw_objects if str(obj).strip()]
    counts = Counter(raw)
    n = len(unique)
    raw_n = len(raw)
    dup = max(0, raw_n - len(set(raw)))
    high_freq = sum(1 for obj in unique if obj in HIGH_FREQ_OBJECTS)
    generic = sum(1 for obj in unique if obj in GENERIC_OBJECTS or object_family(obj) in GENERIC_OBJECTS)
    compound = sum(1 for obj in unique if " " in obj)
    family_counts = Counter(object_family(obj) for obj in unique)
    repeated_families = sum(1 for _, c in family_counts.items() if c > 1)

    out.update(
        {
            f"{prefix}_object_count": n,
            f"{prefix}_raw_object_count": raw_n,
            f"{prefix}_duplicate_object_count": dup,
            f"{prefix}_duplicate_object_rate": safe_div(float(dup), float(raw_n)),
            f"{prefix}_high_freq_object_count": high_freq,
            f"{prefix}_high_freq_object_rate": safe_div(float(high_freq), float(n)),
            f"{prefix}_generic_object_count": generic,
            f"{prefix}_generic_object_rate": safe_div(float(generic), float(n)),
            f"{prefix}_compound_object_count": compound,
            f"{prefix}_compound_object_rate": safe_div(float(compound), float(n)),
            f"{prefix}_object_family_count": len([k for k in family_counts if k]),
            f"{prefix}_repeated_family_count": repeated_families,
            f"{prefix}_repeated_family_rate": safe_div(float(repeated_families), float(max(1, len(family_counts)))),
        }
    )


def text_tokens(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(str(text or ""))]


def bigram_repeat_rate(tokens: Sequence[str]) -> float:
    if len(tokens) < 4:
        return 0.0
    grams = list(zip(tokens, tokens[1:]))
    return safe_div(float(len(grams) - len(set(grams))), float(len(grams)))


def trigram_repeat_rate(tokens: Sequence[str]) -> float:
    if len(tokens) < 5:
        return 0.0
    grams = list(zip(tokens, tokens[1:], tokens[2:]))
    return safe_div(float(len(grams) - len(set(grams))), float(len(grams)))


def caption_stats(prefix: str, text: str, object_count: int, out: Dict[str, Any]) -> None:
    tokens = text_tokens(text)
    content = [tok for tok, _ in content_tokens(text)]
    generic_hits = sum(str(text or "").lower().count(phrase) for phrase in GENERIC_PHRASES)
    filler = sum(1 for tok in tokens if tok in GENERIC_FILLERS)
    stop = sum(1 for tok in tokens if tok in STOPWORDS)
    n = len(tokens)
    c = len(content)
    unique_content = len(set(content))
    out.update(
        {
            f"{prefix}_caption_word_count": n,
            f"{prefix}_caption_content_count": c,
            f"{prefix}_caption_unique_content_count": unique_content,
            f"{prefix}_caption_content_diversity": safe_div(float(unique_content), float(c)),
            f"{prefix}_caption_stopword_rate": safe_div(float(stop), float(n)),
            f"{prefix}_caption_filler_rate": safe_div(float(filler), float(n)),
            f"{prefix}_caption_generic_phrase_count": generic_hits,
            f"{prefix}_caption_generic_phrase_rate": safe_div(float(generic_hits), float(max(1, n))),
            f"{prefix}_caption_bigram_repeat_rate": bigram_repeat_rate(tokens),
            f"{prefix}_caption_trigram_repeat_rate": trigram_repeat_rate(tokens),
            f"{prefix}_object_density_per_word": safe_div(float(object_count), float(n)),
            f"{prefix}_object_density_per_content": safe_div(float(object_count), float(c)),
        }
    )


def set_join(values: Iterable[str]) -> str:
    return " | ".join(str(x) for x in values if str(x).strip())


def add_pairwise_features(out: Dict[str, Any]) -> None:
    def f(key: str) -> float:
        return safe_float(out.get(key))

    base_count = f("capobj_base_object_count")
    int_count = f("capobj_int_object_count")
    base_only = f("capobj_base_only_object_count")
    int_only = f("capobj_int_only_object_count")
    shared = f("capobj_shared_object_count")
    union = f("capobj_union_object_count")
    jaccard = f("capobj_object_jaccard")

    base_words = f("capcost_base_caption_word_count")
    int_words = f("capcost_int_caption_word_count")
    base_density = f("capcost_base_object_density_per_word")
    int_density = f("capcost_int_object_density_per_word")
    base_dup = f("capobj_base_duplicate_object_rate")
    int_dup = f("capobj_int_duplicate_object_rate")
    base_high = f("capobj_base_high_freq_object_rate")
    int_high = f("capobj_int_high_freq_object_rate")
    base_generic = f("capobj_base_generic_object_rate")
    int_generic = f("capobj_int_generic_object_rate")
    base_filler = f("capcost_base_caption_filler_rate")
    int_filler = f("capcost_int_caption_filler_rate")
    base_repeat = f("capcost_base_caption_bigram_repeat_rate")
    int_repeat = f("capcost_int_caption_bigram_repeat_rate")

    out.update(
        {
            # Keep v72/v73-compatible names for the primary benefit gate.
            "capobjyn_base_object_count": base_count,
            "capobjyn_int_object_count": int_count,
            "capobjyn_shared_object_count": shared,
            "capobjyn_union_object_count": union,
            "capobjyn_base_only_object_count": base_only,
            "capobjyn_int_only_object_count": int_only,
            "capobjyn_base_minus_int_object_count": base_count - int_count,
            "capobjyn_object_jaccard": jaccard,
            "capobjyn_base_only_x_jaccard_gap": base_only * (1.0 - jaccard),
            "capobjyn_int_only_x_jaccard_gap": int_only * (1.0 - jaccard),
            "capobjyn_base_only_rate": safe_div(base_only, base_count),
            "capobjyn_int_only_rate": safe_div(int_only, int_count),
            "capobjyn_shared_rate_over_base": safe_div(shared, base_count),
            "capobjyn_shared_rate_over_int": safe_div(shared, int_count),
            "capobjyn_base_to_int_count_ratio": safe_div(base_count, int_count),
            "capobjyn_int_to_base_count_ratio": safe_div(int_count, base_count),
            # Cost/shape features.
            "capcost_delta_caption_word_count_base_minus_int": base_words - int_words,
            "capcost_abs_delta_caption_word_count": abs(base_words - int_words),
            "capcost_base_to_int_caption_len_ratio": safe_div(base_words, int_words),
            "capcost_delta_object_density_base_minus_int": base_density - int_density,
            "capcost_abs_delta_object_density": abs(base_density - int_density),
            "capcost_base_object_overgeneration": max(0.0, base_count - int_count),
            "capcost_int_object_overgeneration": max(0.0, int_count - base_count),
            "capcost_base_only_overflow_gt2": max(0.0, base_only - 2.0),
            "capcost_base_only_overflow_gt3": max(0.0, base_only - 3.0),
            "capcost_base_only_overflow_gt4": max(0.0, base_only - 4.0),
            "capcost_base_only_moderate_score": base_only * (1.0 - jaccard) - max(0.0, base_only - 3.0),
            "capcost_base_oververbose_score": base_count * base_density,
            "capcost_base_divergence_oververbose_score": base_only * (1.0 - jaccard) * base_density,
            "capcost_delta_duplicate_rate_base_minus_int": base_dup - int_dup,
            "capcost_delta_high_freq_rate_base_minus_int": base_high - int_high,
            "capcost_delta_generic_object_rate_base_minus_int": base_generic - int_generic,
            "capcost_delta_filler_rate_base_minus_int": base_filler - int_filler,
            "capcost_delta_bigram_repeat_rate_base_minus_int": base_repeat - int_repeat,
            "capcost_base_noise_score": base_dup + base_generic + base_filler + base_repeat,
            "capcost_int_noise_score": int_dup + int_generic + int_filler + int_repeat,
            "capcost_delta_noise_score_base_minus_int": (base_dup + base_generic + base_filler + base_repeat)
            - (int_dup + int_generic + int_filler + int_repeat),
            "capcost_candidate_score_divergence_minus_noise": base_only * (1.0 - jaccard)
            - (base_dup + base_generic + base_filler + base_repeat),
        }
    )


def load_oracle_map(path: str, target_col: str) -> Dict[str, Dict[str, Any]]:
    if not str(path or "").strip() or not os.path.isfile(os.path.abspath(path)):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv_rows(path):
        sid = norm_id(row.get("id") or row.get("image_id") or row.get("question_id"))
        if sid:
            out[sid] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract caption/object-list divergence and over-generation cost features.")
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--baseline_object_pred_jsonl", required=True)
    ap.add_argument("--intervention_object_pred_jsonl", required=True)
    ap.add_argument("--oracle_rows_csv", default="")
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_pred_text_key", default="auto")
    ap.add_argument("--intervention_pred_text_key", default="auto")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    questions = read_jsonl(os.path.abspath(args.question_file), limit=int(args.limit))
    baseline = read_prediction_map(os.path.abspath(args.baseline_pred_jsonl), str(args.baseline_pred_text_key))
    intervention = read_prediction_map(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_text_key))
    base_obj_pred = read_prediction_map(os.path.abspath(args.baseline_object_pred_jsonl))
    int_obj_pred = read_prediction_map(os.path.abspath(args.intervention_object_pred_jsonl))
    oracle = load_oracle_map(str(args.oracle_rows_csv), str(args.target_col))

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    for q in questions:
        sid = norm_id(q.get("question_id") or q.get("image_id") or q.get("id"))
        row: Dict[str, Any] = {
            "id": sid,
            "image_id": sid,
            "image": q.get("image", ""),
            str(args.target_col): 0,
            "capcost_error": "",
        }
        try:
            if not sid:
                raise ValueError("Missing sample id.")
            b_text = str(baseline.get(sid, {}).get("text", ""))
            i_text = str(intervention.get(sid, {}).get("text", ""))
            b_obj_text = str(base_obj_pred.get(sid, {}).get("text", ""))
            i_obj_text = str(int_obj_pred.get(sid, {}).get("text", ""))

            b_objs = split_object_list(b_obj_text)
            i_objs = split_object_list(i_obj_text)
            b_raw = split_object_list_all(b_obj_text)
            i_raw = split_object_list_all(i_obj_text)
            b_set = set(b_objs)
            i_set = set(i_objs)
            shared = ordered_unique([obj for obj in b_objs if obj in i_set])
            base_only = ordered_unique([obj for obj in b_objs if obj not in i_set])
            int_only = ordered_unique([obj for obj in i_objs if obj not in b_set])
            union = b_set | i_set
            jaccard = safe_div(float(len(b_set & i_set)), float(len(union)))

            row.update(
                {
                    "base_caption": b_text,
                    "int_caption": i_text,
                    "capobj_base_object_names": set_join(b_objs),
                    "capobj_int_object_names": set_join(i_objs),
                    "capobj_base_only_object_names": set_join(base_only),
                    "capobj_int_only_object_names": set_join(int_only),
                    "capobj_shared_object_names": set_join(shared),
                    "capobj_base_object_count": len(b_objs),
                    "capobj_int_object_count": len(i_objs),
                    "capobj_shared_object_count": len(shared),
                    "capobj_union_object_count": len(union),
                    "capobj_base_only_object_count": len(base_only),
                    "capobj_int_only_object_count": len(int_only),
                    "capobj_object_jaccard": jaccard,
                }
            )
            object_stats("capobj_base", b_objs, b_raw, row)
            object_stats("capobj_int", i_objs, i_raw, row)
            object_stats("capobj_base_only", base_only, [obj for obj in b_raw if obj not in i_set], row)
            object_stats("capobj_int_only", int_only, [obj for obj in i_raw if obj not in b_set], row)
            caption_stats("capcost_base", b_text, len(b_objs), row)
            caption_stats("capcost_int", i_text, len(i_objs), row)
            add_pairwise_features(row)

            o = oracle.get(sid)
            if o:
                row[str(args.target_col)] = int(str(o.get(str(args.target_col), "")).strip().lower() in {"1", "true", "yes", "y"})
                for key in (
                    "failure_type",
                    "n_base_only_supported_unique",
                    "n_int_only_supported_unique",
                    "n_base_only_hallucinated_unique",
                    "n_int_only_hallucinated_unique",
                    "delta_recall_base_minus_int",
                    "delta_f1_unique_base_minus_int",
                    "delta_ci_unique_base_minus_int",
                ):
                    if key in o:
                        row[f"audit_{key}"] = o[key]
        except Exception as exc:
            n_errors += 1
            row["capcost_error"] = str(exc)
        rows.append(row)

    write_csv(args.out_csv, rows)
    print("[saved]", os.path.abspath(args.out_csv))
    if str(args.out_summary_json or "").strip():
        feature_keys = [key for key in rows[0] if key.startswith("capobj") or key.startswith("capcost")] if rows else []
        target_count = sum(int(row.get(str(args.target_col), 0) or 0) for row in rows)
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "baseline_object_pred_jsonl": os.path.abspath(args.baseline_object_pred_jsonl),
                    "intervention_object_pred_jsonl": os.path.abspath(args.intervention_object_pred_jsonl),
                    "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv)
                    if str(args.oracle_rows_csv or "").strip()
                    else "",
                    "target_col": str(args.target_col),
                    "limit": int(args.limit),
                },
                "counts": {
                    "n_rows": len(rows),
                    "n_errors": n_errors,
                    "n_target": target_count,
                    "target_rate": safe_div(float(target_count), float(len(rows))),
                    "n_features": len(feature_keys),
                },
                "feature_keys": feature_keys,
                "outputs": {"out_csv": os.path.abspath(args.out_csv)},
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
