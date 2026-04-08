#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


WORD_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "too",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "would",
    "you",
    "your",
    "yours",
}

ATTRIBUTE_WORDS = {
    "beige",
    "black",
    "blond",
    "blue",
    "brown",
    "clean",
    "closed",
    "colorful",
    "dark",
    "dirty",
    "empty",
    "full",
    "giant",
    "gold",
    "gray",
    "green",
    "grey",
    "large",
    "left",
    "little",
    "long",
    "metal",
    "middle",
    "open",
    "orange",
    "pink",
    "plastic",
    "purple",
    "red",
    "right",
    "round",
    "short",
    "silver",
    "small",
    "striped",
    "tall",
    "tiny",
    "top",
    "white",
    "wooden",
    "yellow",
}

RELATION_WORDS = {
    "above",
    "across",
    "against",
    "along",
    "around",
    "at",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "by",
    "carrying",
    "covering",
    "holding",
    "in",
    "inside",
    "near",
    "next",
    "on",
    "over",
    "riding",
    "standing",
    "under",
    "wearing",
    "with",
}

QUANTIFIER_WORDS = {
    "a",
    "an",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "many",
    "multiple",
    "several",
}

COCO_OBJECT_WORDS = {
    "airplane",
    "apple",
    "backpack",
    "banana",
    "baseball",
    "bat",
    "bear",
    "bed",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "bowl",
    "broccoli",
    "bus",
    "cake",
    "car",
    "carrot",
    "cat",
    "chair",
    "clock",
    "couch",
    "cow",
    "cup",
    "dog",
    "donut",
    "elephant",
    "fire",
    "fork",
    "frisbee",
    "giraffe",
    "glass",
    "glove",
    "hair",
    "handbag",
    "hat",
    "horse",
    "hydrant",
    "keyboard",
    "kite",
    "knife",
    "laptop",
    "light",
    "meter",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "oven",
    "parking",
    "person",
    "phone",
    "pizza",
    "plant",
    "plate",
    "racket",
    "refrigerator",
    "remote",
    "sandwich",
    "scissors",
    "seat",
    "sheep",
    "sign",
    "sink",
    "skateboard",
    "skis",
    "snowboard",
    "spoon",
    "sports",
    "stop",
    "stove",
    "suitcase",
    "surfboard",
    "table",
    "teddy",
    "tennis",
    "tie",
    "toaster",
    "toilet",
    "toothbrush",
    "traffic",
    "train",
    "truck",
    "tv",
    "umbrella",
    "vase",
    "wine",
    "zebra",
}


def parse_bool(value: object) -> bool:
    s = str(value if value is not None else "").strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def safe_id(value: object) -> str:
    return str(value if value is not None else "").strip()


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_question_rows(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    if int(limit) > 0:
        rows = rows[: int(limit)]
    return rows


def load_prediction_text_map(path: str, text_key: str = "auto") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl(path):
        sid = safe_id(row.get("question_id", row.get("id")))
        if not sid:
            continue
        if text_key == "auto":
            text = (
                str(row.get("text", "")).strip()
                or str(row.get("output", "")).strip()
                or str(row.get("answer", "")).strip()
                or str(row.get("caption", "")).strip()
            )
        else:
            text = str(row.get(text_key, "")).strip()
        out[sid] = text
    return out


def load_feature_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        if sid:
            out[sid] = row
    return out


def normalize_word(text: str) -> str:
    m = WORD_RE.search(str(text or "").lower())
    return m.group(0) if m else ""


def tokenize_words(text: str) -> List[str]:
    return [normalize_word(m.group(0)) for m in WORD_RE.finditer(str(text or ""))]


def is_attribute_word(word: str) -> bool:
    w = normalize_word(word)
    return bool(w and (w in ATTRIBUTE_WORDS or w in QUANTIFIER_WORDS or w.isdigit()))


def is_count_word(word: str) -> bool:
    w = normalize_word(word)
    return bool(w and (w in QUANTIFIER_WORDS or w.isdigit()))


def is_object_word(word: str) -> bool:
    w = normalize_word(word)
    return bool(w and (w in COCO_OBJECT_WORDS or (len(w) >= 4 and w not in STOPWORDS and w not in RELATION_WORDS)))


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def unique_set(values: Iterable[str]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return safe_div(float(len(a & b)), float(len(a | b)))


def overlap_frac(src: set[str], other: set[str]) -> float:
    return safe_div(float(len(src & other)), float(len(src)))


def repeated_token_rate(tokens: Sequence[str]) -> float:
    seq = [str(x).strip().lower() for x in tokens if str(x).strip()]
    if len(seq) <= 1:
        return 0.0
    unique_n = len(set(seq))
    return safe_div(float(len(seq) - unique_n), float(len(seq)))


def common_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    n = min(len(a), len(b))
    idx = 0
    while idx < n and a[idx] == b[idx]:
        idx += 1
    return idx


def count_set_features(prefix: str, base_set: set[str], int_set: set[str]) -> Dict[str, float]:
    added = int_set - base_set
    dropped = base_set - int_set
    inter = base_set & int_set
    return {
        f"{prefix}_base_count": int(len(base_set)),
        f"{prefix}_int_count": int(len(int_set)),
        f"{prefix}_shared_count": int(len(inter)),
        f"{prefix}_jaccard": float(jaccard(base_set, int_set)),
        f"{prefix}_overlap_base_frac": float(overlap_frac(base_set, int_set)),
        f"{prefix}_overlap_int_frac": float(overlap_frac(int_set, base_set)),
        f"{prefix}_add_count": int(len(added)),
        f"{prefix}_drop_count": int(len(dropped)),
        f"{prefix}_add_rate": float(safe_div(float(len(added)), float(max(1, len(base_set))))),
        f"{prefix}_drop_rate": float(safe_div(float(len(dropped)), float(max(1, len(base_set))))),
        f"{prefix}_symmetric_diff_count": int(len(added) + len(dropped)),
        f"{prefix}_symmetric_diff_rate": float(
            safe_div(float(len(added) + len(dropped)), float(max(1, len(base_set | int_set))))
        ),
    }


def build_pair_features(baseline_text: str, intervention_text: str) -> Dict[str, Any]:
    base_words = tokenize_words(baseline_text)
    int_words = tokenize_words(intervention_text)
    base_content = [w for w in base_words if w not in STOPWORDS]
    int_content = [w for w in int_words if w not in STOPWORDS]
    base_attrs = [w for w in base_words if is_attribute_word(w)]
    int_attrs = [w for w in int_words if is_attribute_word(w)]
    base_rel = [w for w in base_words if w in RELATION_WORDS]
    int_rel = [w for w in int_words if w in RELATION_WORDS]
    base_counts = [w for w in base_words if is_count_word(w)]
    int_counts = [w for w in int_words if is_count_word(w)]
    base_objects = [w for w in base_words if is_object_word(w)]
    int_objects = [w for w in int_words if is_object_word(w)]

    base_word_set = unique_set(base_words)
    int_word_set = unique_set(int_words)
    base_content_set = unique_set(base_content)
    int_content_set = unique_set(int_content)
    base_attr_set = unique_set(base_attrs)
    int_attr_set = unique_set(int_attrs)
    base_rel_set = unique_set(base_rel)
    int_rel_set = unique_set(int_rel)
    base_count_set = unique_set(base_counts)
    int_count_set = unique_set(int_counts)
    base_object_set = unique_set(base_objects)
    int_object_set = unique_set(int_objects)

    max_words = max(1, len(base_words))
    max_content = max(1, len(base_content))
    max_chars = max(1, len(str(baseline_text)))
    prefix_len_words = common_prefix_len(base_words, int_words)
    prefix_len_content = common_prefix_len(base_content, int_content)
    max_word_len = max(1, max(len(base_words), len(int_words)))
    max_content_len = max(1, max(len(base_content), len(int_content)))
    seq_ratio_words = difflib.SequenceMatcher(None, base_words, int_words).ratio()
    seq_ratio_content = difflib.SequenceMatcher(None, base_content, int_content).ratio()

    row: Dict[str, Any] = {
        "pair_base_n_words": int(len(base_words)),
        "pair_int_n_words": int(len(int_words)),
        "pair_word_len_delta_norm": float(safe_div(float(len(int_words) - len(base_words)), float(max_words))),
        "pair_word_len_ratio_int_to_base": float(safe_div(float(len(int_words)), float(max_words))),
        "pair_word_shorter_frac": float(safe_div(float(max(0, len(base_words) - len(int_words))), float(max_words))),
        "pair_word_longer_frac": float(safe_div(float(max(0, len(int_words) - len(base_words))), float(max_words))),
        "pair_base_n_chars": int(len(str(baseline_text))),
        "pair_int_n_chars": int(len(str(intervention_text))),
        "pair_char_len_delta_norm": float(
            safe_div(float(len(str(intervention_text)) - len(str(baseline_text))), float(max_chars))
        ),
        "pair_char_len_ratio_int_to_base": float(
            safe_div(float(len(str(intervention_text))), float(max_chars))
        ),
        "pair_base_n_content": int(len(base_content)),
        "pair_int_n_content": int(len(int_content)),
        "pair_content_len_delta_norm": float(
            safe_div(float(len(int_content) - len(base_content)), float(max_content))
        ),
        "pair_content_len_ratio_int_to_base": float(safe_div(float(len(int_content)), float(max_content))),
        "pair_content_shorter_frac": float(
            safe_div(float(max(0, len(base_content) - len(int_content))), float(max_content))
        ),
        "pair_content_longer_frac": float(
            safe_div(float(max(0, len(int_content) - len(base_content))), float(max_content))
        ),
        "pair_word_jaccard": float(jaccard(base_word_set, int_word_set)),
        "pair_content_jaccard": float(jaccard(base_content_set, int_content_set)),
        "pair_word_overlap_base_frac": float(overlap_frac(base_word_set, int_word_set)),
        "pair_word_overlap_int_frac": float(overlap_frac(int_word_set, base_word_set)),
        "pair_content_overlap_base_frac": float(overlap_frac(base_content_set, int_content_set)),
        "pair_content_overlap_int_frac": float(overlap_frac(int_content_set, base_content_set)),
        "pair_unique_content_add_count": int(len(int_content_set - base_content_set)),
        "pair_unique_content_drop_count": int(len(base_content_set - int_content_set)),
        "pair_unique_content_add_rate": float(
            safe_div(float(len(int_content_set - base_content_set)), float(max(1, len(base_content_set))))
        ),
        "pair_unique_content_drop_rate": float(
            safe_div(float(len(base_content_set - int_content_set)), float(max(1, len(base_content_set))))
        ),
        "pair_prefix_ratio_words": float(safe_div(float(prefix_len_words), float(max_word_len))),
        "pair_prefix_ratio_content": float(safe_div(float(prefix_len_content), float(max_content_len))),
        "pair_sequence_ratio_words": float(seq_ratio_words),
        "pair_sequence_ratio_content": float(seq_ratio_content),
        "pair_first_divergence_pos_frac_words": float(safe_div(float(prefix_len_words), float(max_word_len))),
        "pair_first_divergence_pos_frac_content": float(
            safe_div(float(prefix_len_content), float(max_content_len))
        ),
        "pair_base_repetition_rate": float(repeated_token_rate(base_content)),
        "pair_int_repetition_rate": float(repeated_token_rate(int_content)),
        "pair_repetition_delta": float(repeated_token_rate(int_content) - repeated_token_rate(base_content)),
        "pair_agreement_score": float(
            (
                jaccard(base_word_set, int_word_set)
                + jaccard(base_content_set, int_content_set)
                + seq_ratio_content
            ) / 3.0
        ),
        "pair_intervention_changed": int(base_words != int_words),
    }
    row.update(count_set_features("pair_object", base_object_set, int_object_set))
    row.update(count_set_features("pair_attr", base_attr_set, int_attr_set))
    row.update(count_set_features("pair_relation", base_rel_set, int_rel_set))
    row.update(count_set_features("pair_count", base_count_set, int_count_set))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract GT-free pairwise caption-drift features for generative fallback control.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--base_features_csv", type=str, default="")
    ap.add_argument("--baseline_pred_text_key", type=str, default="auto")
    ap.add_argument("--intervention_pred_text_key", type=str, default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=250)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    baseline_map = load_prediction_text_map(args.baseline_pred_jsonl, text_key=args.baseline_pred_text_key)
    intervention_map = load_prediction_text_map(args.intervention_pred_jsonl, text_key=args.intervention_pred_text_key)
    base_feature_map = load_feature_map(args.base_features_csv) if str(args.base_features_csv).strip() else {}

    rows: List[Dict[str, Any]] = []
    n_missing_base_pred = 0
    n_missing_intervention_pred = 0
    n_missing_base_feature = 0
    for idx, sample in enumerate(question_rows):
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        base_text = str(baseline_map.get(sid, "")).strip()
        int_text = str(intervention_map.get(sid, "")).strip()
        if not base_text:
            n_missing_base_pred += 1
        if not int_text:
            n_missing_intervention_pred += 1

        row: Dict[str, Any] = {
            "id": sid,
            "image": image_name,
            "question": question,
            "pair_baseline_text": base_text,
            "pair_intervention_text": int_text,
        }
        base_row = base_feature_map.get(sid)
        if base_row is not None:
            for key, value in base_row.items():
                row[key] = value
        elif base_feature_map:
            n_missing_base_feature += 1

        row.update(build_pair_features(base_text, int_text))
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[pairwise] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")

    if str(args.out_summary_json or "").strip():
        feature_keys = [k for k in rows[0].keys() if k.startswith("pair_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "base_features_csv": os.path.abspath(args.base_features_csv) if str(args.base_features_csv).strip() else "",
                    "baseline_pred_text_key": args.baseline_pred_text_key,
                    "intervention_pred_text_key": args.intervention_pred_text_key,
                },
                "counts": {
                    "n_questions": int(len(question_rows)),
                    "n_rows": int(len(rows)),
                    "n_pair_features": int(len(feature_keys)),
                    "n_missing_base_pred": int(n_missing_base_pred),
                    "n_missing_intervention_pred": int(n_missing_intervention_pred),
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
