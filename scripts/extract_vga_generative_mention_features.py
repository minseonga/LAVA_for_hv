#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import (
    CleanroomLlavaRuntime,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    safe_id,
    select_content_indices,
    write_csv,
    write_json,
)


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


def mean_or_zero(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def min_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(min(seq))


def max_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(max(seq))


def normalize_word(text: str) -> str:
    m = WORD_RE.search(str(text or "").lower())
    return m.group(0) if m else ""


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    idx = 0
    while idx < n and a[idx] == b[idx]:
        idx += 1
    return idx


def decode_token_spans(tokenizer: Any, cont_ids: torch.Tensor) -> Tuple[str, List[Tuple[int, int]]]:
    ids = [int(x) for x in cont_ids.tolist()]
    prev = ""
    spans: List[Tuple[int, int]] = []
    prefix_ids: List[int] = []
    for token_id in ids:
        prefix_ids.append(token_id)
        curr = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        start = len(prev) if curr.startswith(prev) else common_prefix_len(prev, curr)
        start = max(0, min(start, len(curr)))
        spans.append((start, len(curr)))
        prev = curr
    return prev, spans


def extract_word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0).lower()) for m in WORD_RE.finditer(str(text or ""))]


def is_attribute_word(word: str) -> bool:
    w = normalize_word(word)
    return bool(w and (w in ATTRIBUTE_WORDS or w in QUANTIFIER_WORDS or w.isdigit()))


def is_count_word(word: str) -> bool:
    w = normalize_word(word)
    return bool(w and (w in QUANTIFIER_WORDS or w.isdigit()))


def is_object_word(word: str) -> bool:
    w = normalize_word(word)
    return bool(w and (w in COCO_OBJECT_WORDS or (len(w) >= 4 and w not in STOPWORDS and w not in RELATION_WORDS)))


def add_mention(
    mention_map: Dict[Tuple[int, int], Dict[str, Any]],
    start: int,
    end: int,
    kind: str,
    text: str,
) -> None:
    key = (int(start), int(end))
    if end <= start:
        return
    entry = mention_map.get(key)
    if entry is None:
        mention_map[key] = {
            "start": int(start),
            "end": int(end),
            "text": str(text).strip(),
            "kinds": {str(kind)},
        }
        return
    entry["kinds"].add(str(kind))
    if len(str(text).strip()) > len(str(entry.get("text", "")).strip()):
        entry["text"] = str(text).strip()


def extract_mentions(
    text: str,
    max_mentions: int,
) -> Tuple[List[Dict[str, Any]], int]:
    words = extract_word_spans(text)
    mention_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    fallback_used = 0

    for idx, (start, end, word) in enumerate(words):
        if not is_object_word(word):
            continue
        left = idx
        while left > 0 and is_attribute_word(words[left - 1][2]):
            left -= 1
        span_start = words[left][0]
        span_end = end
        add_mention(mention_map, span_start, span_end, "object_mention", text[span_start:span_end])
        add_mention(mention_map, span_start, span_end, "noun_phrase", text[span_start:span_end])
        if left < idx:
            add_mention(mention_map, span_start, span_end, "attribute_phrase", text[span_start:span_end])
        if any(is_count_word(words[j][2]) for j in range(left, idx)):
            add_mention(mention_map, span_start, span_end, "count_phrase", text[span_start:span_end])

    for idx, (start, _, word) in enumerate(words):
        if word not in RELATION_WORDS:
            continue
        for jdx in range(idx + 1, min(len(words), idx + 5)):
            if not is_object_word(words[jdx][2]):
                continue
            left = jdx
            while left > idx + 1 and is_attribute_word(words[left - 1][2]):
                left -= 1
            span_end = words[jdx][1]
            add_mention(mention_map, start, span_end, "relation_phrase", text[start:span_end])
            break

    mentions = list(mention_map.values())
    mentions.sort(key=lambda x: (int(x["start"]), int(x["end"])))

    if not mentions:
        fallback_used = 1
        claim_words = [span for span in words if span[2] not in STOPWORDS]
        for span in claim_words[: max(1, int(max_mentions))]:
            add_mention(mention_map, span[0], span[1], "fallback_word", text[span[0]:span[1]])
        mentions = list(mention_map.values())
        mentions.sort(key=lambda x: (int(x["start"]), int(x["end"])))

    out: List[Dict[str, Any]] = []
    for item in mentions[: max(1, int(max_mentions))]:
        out.append(
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "text": str(item["text"]),
                "kinds": sorted(str(x) for x in item["kinds"]),
            }
        )
    return out, int(fallback_used)


def mention_token_indices(
    token_spans: Sequence[Tuple[int, int]],
    content_indices: Sequence[int],
    mention: Dict[str, Any],
) -> List[int]:
    content_set = {int(idx) for idx in content_indices}
    keep: List[int] = []
    start = int(mention["start"])
    end = int(mention["end"])
    for idx, (tok_start, tok_end) in enumerate(token_spans):
        if idx not in content_set or tok_end <= tok_start:
            continue
        if max(tok_start, start) < min(tok_end, end):
            keep.append(int(idx))
    return sorted(set(keep))


def unique_nonempty(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def compute_pack_values(pack: Any) -> Dict[str, List[float]]:
    logits = pack.logits.to(torch.float32)
    decision_positions = pack.decision_positions.long()
    target_ids = pack.labels_exp[pack.cont_label_positions.long()].long()

    token_logits = logits[decision_positions]
    log_probs = F.log_softmax(token_logits, dim=-1)
    probs = torch.softmax(token_logits, dim=-1)
    token_ent = -(probs * log_probs).sum(dim=-1)
    target_lp = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

    top2_vals, top2_idx = torch.topk(token_logits, k=2, dim=-1)
    top1_logit = top2_vals[:, 0]
    top2_logit = top2_vals[:, 1]
    top1_id = top2_idx[:, 0]
    target_logit = token_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    best_other_logit = torch.where(top1_id == target_ids, top2_logit, top1_logit)
    target_gap = target_logit - best_other_logit

    return {
        "lp": [float(x.item()) for x in target_lp],
        "gap": [float(x.item()) for x in target_gap],
        "ent": [float(x.item()) for x in token_ent],
        "n_tokens": int(target_ids.numel()),
    }


def pick(values: Sequence[float], idxs: Sequence[int]) -> List[float]:
    return [float(values[int(i)]) for i in idxs if 0 <= int(i) < len(values)]


def build_feature_payload(
    runtime: CleanroomLlavaRuntime,
    image_path: str,
    question: str,
    candidate_text: str,
    sample_id: str,
    image_name: str,
    *,
    image: Any = None,
    max_mentions: int,
) -> Dict[str, Any]:
    image_obj = image if image is not None else runtime.load_image(image_path)
    pack = runtime.teacher_force_candidate(
        image=image_obj,
        question=question,
        candidate_text=candidate_text,
        output_attentions=False,
    )
    content_indices = select_content_indices(runtime.tokenizer, pack.cont_ids)
    decoded_text, token_spans = decode_token_spans(runtime.tokenizer, pack.cont_ids)
    mentions, fallback_used = extract_mentions(decoded_text, max_mentions=max_mentions)
    values = compute_pack_values(pack)

    mention_rows: List[Dict[str, Any]] = []
    for mention in mentions:
        idxs = mention_token_indices(token_spans, content_indices, mention)
        if not idxs:
            continue
        lp_vals = pick(values["lp"], idxs)
        gap_vals = pick(values["gap"], idxs)
        ent_vals = pick(values["ent"], idxs)
        mention_rows.append(
            {
                "text": str(mention["text"]),
                "kinds": "|".join(str(x) for x in mention["kinds"]),
                "n_tokens": int(len(idxs)),
                "first_idx": int(min(idxs)),
                "last_idx": int(max(idxs)),
                "lp_min": min_or_zero(lp_vals),
                "gap_min": min_or_zero(gap_vals),
                "ent_max": max_or_zero(ent_vals),
                "lp_tail_gap": float(min_or_zero(lp_vals) - mean_or_zero(lp_vals)),
            }
        )

    if not mention_rows:
        raise RuntimeError("No mention spans aligned to content tokens.")

    weakest_lp = min(mention_rows, key=lambda x: float(x["lp_min"]))
    weakest_gap = min(mention_rows, key=lambda x: float(x["gap_min"]))
    weakest_ent = max(mention_rows, key=lambda x: float(x["ent_max"]))
    weakest_tail = min(mention_rows, key=lambda x: float(x["lp_tail_gap"]))

    object_mentions = sum(1 for row in mention_rows if "object_mention" in str(row["kinds"]).split("|"))
    noun_mentions = sum(1 for row in mention_rows if "noun_phrase" in str(row["kinds"]).split("|"))
    attr_mentions = sum(1 for row in mention_rows if "attribute_phrase" in str(row["kinds"]).split("|"))
    relation_mentions = sum(1 for row in mention_rows if "relation_phrase" in str(row["kinds"]).split("|"))
    count_mentions = sum(1 for row in mention_rows if "count_phrase" in str(row["kinds"]).split("|"))
    object_texts = unique_nonempty(row["text"] for row in mention_rows if "object_mention" in str(row["kinds"]).split("|"))
    mention_texts = unique_nonempty(row["text"] for row in mention_rows)

    ordered_content = sorted(int(x) for x in content_indices)
    n_content = len(ordered_content)
    head_n = max(1, n_content // 3) if n_content > 0 else 0
    tail_n = head_n
    head_slice = ordered_content[:head_n]
    tail_slice = ordered_content[-tail_n:] if tail_n > 0 else []
    lp_head = pick(values["lp"], head_slice)
    lp_tail = pick(values["lp"], tail_slice)
    gap_head = pick(values["gap"], head_slice)
    gap_tail = pick(values["gap"], tail_slice)
    ent_head = pick(values["ent"], head_slice)
    ent_tail = pick(values["ent"], tail_slice)

    midpoint = (n_content - 1) / 2.0 if n_content > 0 else 0.0
    first_half_object_mentions = sum(
        1
        for row in mention_rows
        if "object_mention" in str(row["kinds"]).split("|") and float(row["last_idx"]) <= midpoint
    )
    second_half_object_mentions = sum(
        1
        for row in mention_rows
        if "object_mention" in str(row["kinds"]).split("|") and float(row["first_idx"]) > midpoint
    )
    last_mention_idx = max(int(row["last_idx"]) for row in mention_rows)
    max_content_idx = max(ordered_content) if ordered_content else 0
    tail_tokens_after_last_mention = max(0, int(max_content_idx - last_mention_idx))
    last_mention_pos_frac = float(last_mention_idx / float(max(1, max_content_idx))) if max_content_idx > 0 else 0.0

    feature_row = {
        "id": sample_id,
        "image": image_name,
        "question": question,
        "probe_task_mode": "generative_mention",
        "probe_selector": "mention_span",
        "probe_selector_fallback": int(fallback_used),
        "probe_decoded_text": str(decoded_text),
        "probe_n_cont_tokens": int(values["n_tokens"]),
        "probe_n_content_tokens": int(len(content_indices)),
        "probe_n_mentions_total": int(len(mention_rows)),
        "probe_n_object_mentions": int(object_mentions),
        "probe_n_noun_phrases": int(noun_mentions),
        "probe_n_attribute_phrases": int(attr_mentions),
        "probe_n_relation_phrases": int(relation_mentions),
        "probe_n_count_phrases": int(count_mentions),
        "probe_object_diversity": int(len(object_texts)),
        "probe_mention_diversity": int(len(mention_texts)),
        "probe_first_half_object_mentions": int(first_half_object_mentions),
        "probe_second_half_object_mentions": int(second_half_object_mentions),
        "probe_tail_tokens_after_last_mention": int(tail_tokens_after_last_mention),
        "probe_last_mention_pos_frac": float(last_mention_pos_frac),
        "probe_lp_head_mean_real": float(mean_or_zero(lp_head)),
        "probe_lp_tail_mean_real": float(mean_or_zero(lp_tail)),
        "probe_lp_tail_minus_head_real": float(mean_or_zero(lp_tail) - mean_or_zero(lp_head)),
        "probe_gap_head_mean_real": float(mean_or_zero(gap_head)),
        "probe_gap_tail_mean_real": float(mean_or_zero(gap_tail)),
        "probe_gap_tail_minus_head_real": float(mean_or_zero(gap_tail) - mean_or_zero(gap_head)),
        "probe_entropy_head_mean_real": float(mean_or_zero(ent_head)),
        "probe_entropy_tail_mean_real": float(mean_or_zero(ent_tail)),
        "probe_entropy_tail_minus_head_real": float(mean_or_zero(ent_tail) - mean_or_zero(ent_head)),
        "probe_mention_lp_min_real": float(weakest_lp["lp_min"]),
        "probe_mention_target_gap_min_real": float(weakest_gap["gap_min"]),
        "probe_mention_entropy_max_real": float(weakest_ent["ent_max"]),
        "probe_mention_lp_tail_gap_real": float(weakest_tail["lp_tail_gap"]),
        "probe_weakest_lp_mention": str(weakest_lp["text"]),
        "probe_weakest_gap_mention": str(weakest_gap["text"]),
        "probe_weakest_entropy_mention": str(weakest_ent["text"]),
        "probe_weakest_tail_mention": str(weakest_tail["text"]),
        "probe_mention_texts": " || ".join(str(row["text"]) for row in mention_rows[:8]),
    }
    return {
        "row": feature_row,
        "decoded_text": str(decoded_text),
        "mentions": [dict(m) for m in mentions],
        "mention_rows": [dict(m) for m in mention_rows],
        "content_indices": [int(x) for x in ordered_content],
        "max_content_idx": int(max_content_idx),
        "last_mention_idx": int(last_mention_idx),
    }


def build_feature_row(
    runtime: CleanroomLlavaRuntime,
    image_path: str,
    question: str,
    candidate_text: str,
    sample_id: str,
    image_name: str,
    *,
    max_mentions: int,
) -> Dict[str, Any]:
    payload = build_feature_payload(
        runtime=runtime,
        image_path=image_path,
        question=question,
        candidate_text=candidate_text,
        sample_id=sample_id,
        image_name=image_name,
        max_mentions=max_mentions,
    )
    return dict(payload["row"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract generative mention-level brittleness features from baseline captions.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
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
    pred_map = load_prediction_text_map(args.baseline_pred_jsonl, text_key=args.pred_text_key)

    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(args.model_base or None),
        conv_mode=args.conv_mode,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        baseline_text = str(pred_map.get(sample_id, "")).strip()
        image_path = os.path.join(args.image_folder, image_name)

        row: Dict[str, Any] = {
            "id": sample_id,
            "image": image_name,
            "question": question,
            "score_error": "",
        }
        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not question:
                raise ValueError("Missing question text.")
            if not baseline_text:
                raise ValueError("Missing baseline prediction text.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            row.update(
                build_feature_row(
                    runtime=runtime,
                    image_path=image_path,
                    question=question,
                    candidate_text=baseline_text,
                    sample_id=sample_id,
                    image_name=image_name,
                    max_mentions=int(args.max_mentions),
                )
            )
        except Exception as exc:
            n_errors += 1
            row["score_error"] = str(exc)
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[mention-probe] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")

    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
                    "model_path": args.model_path,
                    "model_base": args.model_base,
                    "conv_mode": args.conv_mode,
                    "device": args.device,
                    "max_mentions": int(args.max_mentions),
                },
                "counts": {
                    "n_questions": int(len(question_rows)),
                    "n_rows": int(len(rows)),
                    "n_errors": int(n_errors),
                },
                "outputs": {
                    "feature_csv": os.path.abspath(args.out_csv),
                },
            },
        )
        print(f"[saved] {os.path.abspath(args.out_summary_json)}")


if __name__ == "__main__":
    main()
