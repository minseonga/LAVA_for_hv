#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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

ANSWER_ANCHORS = {
    "yes",
    "no",
    "not",
    "none",
}

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
    "no",
    "not",
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

HELPER_WORDS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "has",
    "have",
    "having",
    "is",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "was",
    "were",
    "will",
    "would",
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
    "cell",
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
    "hot",
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
    "potted",
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


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    idx = 0
    while idx < n and a[idx] == b[idx]:
        idx += 1
    return idx


def normalize_word(text: str) -> str:
    m = WORD_RE.search(str(text or "").lower())
    return m.group(0) if m else ""


def is_claim_like_word(word: str) -> bool:
    w = normalize_word(word)
    if not w:
        return False
    if w.isdigit():
        return True
    if w in COCO_OBJECT_WORDS or w in ATTRIBUTE_WORDS:
        return True
    if w in STOPWORDS or w in HELPER_WORDS:
        return False
    return len(w) >= 4


def extract_word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0).lower()) for m in WORD_RE.finditer(str(text or ""))]


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


def spans_to_token_indices(
    token_spans: Sequence[Tuple[int, int]],
    selected_spans: Sequence[Tuple[int, int, str]],
    content_indices: Sequence[int],
    fallback_limit: int,
) -> List[int]:
    content_set = {int(idx) for idx in content_indices}
    keep: List[int] = []
    for idx, (tok_start, tok_end) in enumerate(token_spans):
        if idx not in content_set or tok_end <= tok_start:
            continue
        for word_start, word_end, _ in selected_spans:
            if max(tok_start, word_start) < min(tok_end, word_end):
                keep.append(int(idx))
                break
    keep = sorted(set(keep))
    if keep:
        return keep
    fallback = [int(idx) for idx in content_indices[: max(1, int(fallback_limit))]]
    return fallback


def select_discriminative_spans(
    text: str,
    max_answer_words: int,
) -> Tuple[List[Tuple[int, int, str]], bool]:
    words = extract_word_spans(text)
    anchors = [span for span in words if span[2] in ANSWER_ANCHORS]
    if anchors:
        return anchors[: max(1, int(max_answer_words))], False
    digits = [span for span in words if span[2].isdigit()]
    if digits:
        return digits[:1], False
    keep: List[Tuple[int, int, str]] = []
    for span in words:
        if span[2] in STOPWORDS or span[2] in HELPER_WORDS:
            continue
        keep.append(span)
        if len(keep) >= max(1, int(max_answer_words)):
            break
    if keep:
        return keep, False
    return words[:1], True


def select_generative_spans(
    text: str,
    max_claim_words: int,
) -> Tuple[List[Tuple[int, int, str]], bool]:
    words = extract_word_spans(text)
    keep = [span for span in words if is_claim_like_word(span[2])]
    if keep:
        return keep[: max(1, int(max_claim_words))], False
    fallback: List[Tuple[int, int, str]] = []
    for span in words:
        if span[2] in STOPWORDS or span[2] in HELPER_WORDS:
            continue
        fallback.append(span)
        if len(fallback) >= max(1, int(max_claim_words)):
            break
    if fallback:
        return fallback, True
    return words[:1], True


def select_subset_indices(
    tokenizer: Any,
    cont_ids: torch.Tensor,
    task_mode: str,
    *,
    max_answer_words: int,
    max_claim_words: int,
) -> Tuple[List[int], Dict[str, Any]]:
    content_indices = select_content_indices(tokenizer, cont_ids)
    decoded_text, token_spans = decode_token_spans(tokenizer, cont_ids)
    if str(task_mode) == "discriminative":
        selected_spans, fallback_used = select_discriminative_spans(decoded_text, max_answer_words=max_answer_words)
        selector = "answer_critical"
        fallback_limit = max(1, int(max_answer_words))
    else:
        selected_spans, fallback_used = select_generative_spans(decoded_text, max_claim_words=max_claim_words)
        selector = "claim_like"
        fallback_limit = max(2, min(8, int(max_claim_words)))
    subset_indices = spans_to_token_indices(
        token_spans=token_spans,
        selected_spans=selected_spans,
        content_indices=content_indices,
        fallback_limit=fallback_limit,
    )
    return subset_indices, {
        "decoded_text": decoded_text,
        "selector": selector,
        "fallback_used": int(bool(fallback_used)),
        "selected_words": [str(span[2]) for span in selected_spans],
        "n_content_tokens": int(len(content_indices)),
    }


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


def build_feature_row(
    runtime: CleanroomLlavaRuntime,
    image_path: str,
    question: str,
    candidate_text: str,
    sample_id: str,
    image_name: str,
    *,
    task_mode: str,
    use_blur_control: bool,
    blur_radius: float,
    max_answer_words: int,
    max_claim_words: int,
) -> Dict[str, Any]:
    image = runtime.load_image(image_path)
    real_pack = runtime.teacher_force_candidate(
        image=image,
        question=question,
        candidate_text=candidate_text,
        output_attentions=False,
    )
    subset_indices, debug = select_subset_indices(
        runtime.tokenizer,
        real_pack.cont_ids,
        task_mode=task_mode,
        max_answer_words=max_answer_words,
        max_claim_words=max_claim_words,
    )
    real_vals = compute_pack_values(real_pack)
    real_lp = pick(real_vals["lp"], subset_indices)
    real_gap = pick(real_vals["gap"], subset_indices)
    real_ent = pick(real_vals["ent"], subset_indices)

    row: Dict[str, Any] = {
        "id": sample_id,
        "image": image_name,
        "question": question,
        "probe_task_mode": str(task_mode),
        "probe_selector": str(debug["selector"]),
        "probe_selector_fallback": int(debug["fallback_used"]),
        "probe_selected_words": "|".join(str(x) for x in debug["selected_words"]),
        "probe_decoded_text": str(debug["decoded_text"]),
        "probe_n_cont_tokens": int(real_vals["n_tokens"]),
        "probe_n_content_tokens": int(debug["n_content_tokens"]),
        "probe_n_subset_tokens": int(len(subset_indices)),
        "probe_subset_fraction": float(len(subset_indices) / float(max(1, debug["n_content_tokens"]))),
        "probe_lp_min_real": min_or_zero(real_lp),
        "probe_target_gap_min_real": min_or_zero(real_gap),
        "probe_entropy_max_real": max_or_zero(real_ent),
    }

    if bool(use_blur_control):
        blur_pack = runtime.teacher_force_candidate(
            image=runtime.make_blur_control(image, blur_radius=blur_radius),
            question=question,
            candidate_text=candidate_text,
            output_attentions=False,
        )
        blur_vals = compute_pack_values(blur_pack)
        blur_lp = pick(blur_vals["lp"], subset_indices)
        blur_gap = pick(blur_vals["gap"], subset_indices)
        blur_ent = pick(blur_vals["ent"], subset_indices)

        row.update(
            {
                "probe_lp_min_blur": min_or_zero(blur_lp),
                "probe_target_gap_min_blur": min_or_zero(blur_gap),
                "probe_entropy_max_blur": max_or_zero(blur_ent),
            }
        )
        row["probe_lp_min_real_minus_blur"] = float(row["probe_lp_min_real"] - row["probe_lp_min_blur"])
        row["probe_target_gap_min_real_minus_blur"] = float(
            row["probe_target_gap_min_real"] - row["probe_target_gap_min_blur"]
        )
        row["probe_entropy_max_real_minus_blur"] = float(
            row["probe_entropy_max_real"] - row["probe_entropy_max_blur"]
        )

    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract baseline-side subset claim features for quick VGA subset probes.")
    ap.add_argument("--task_mode", type=str, choices=["discriminative", "generative"], required=True)
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
    ap.add_argument("--use_blur_control", type=parse_bool, default=True)
    ap.add_argument("--blur_radius", type=float, default=8.0)
    ap.add_argument("--max_answer_words", type=int, default=2)
    ap.add_argument("--max_claim_words", type=int, default=24)
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
                    task_mode=str(args.task_mode),
                    use_blur_control=bool(args.use_blur_control),
                    blur_radius=float(args.blur_radius),
                    max_answer_words=int(args.max_answer_words),
                    max_claim_words=int(args.max_claim_words),
                )
            )
        except Exception as exc:
            n_errors += 1
            row["score_error"] = str(exc)
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[subset-probe:{args.task_mode}] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")

    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "task_mode": args.task_mode,
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
                    "model_path": args.model_path,
                    "model_base": args.model_base,
                    "conv_mode": args.conv_mode,
                    "device": args.device,
                    "use_blur_control": bool(args.use_blur_control),
                    "blur_radius": float(args.blur_radius),
                    "max_answer_words": int(args.max_answer_words),
                    "max_claim_words": int(args.max_claim_words),
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
