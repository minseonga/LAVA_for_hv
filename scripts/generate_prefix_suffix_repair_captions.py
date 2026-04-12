#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

try:
    import shortuuid  # type: ignore
except Exception:
    shortuuid = None


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "with",
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    import csv

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_id(value: object) -> str:
    return str(value or "").strip()


def maybe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def pick_text(row: Dict[str, Any], key: str) -> str:
    if key != "auto":
        return str(row.get(key, "")).strip()
    for cand in ("output", "text", "caption", "answer", "prediction"):
        text = str(row.get(cand, "")).strip()
        if text:
            return text
    return ""


def content_word_spans(text: str) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    for match in WORD_RE.finditer(str(text or "")):
        word = match.group(0).lower()
        if not word or word in STOPWORDS:
            continue
        out.append((int(match.start()), int(match.end()), word))
    return out


def text_novelty_stats(text: str) -> Dict[str, float]:
    spans = content_word_spans(text)
    words = [w for _, _, w in spans]
    if not words:
        return {
            "content_word_count": 0.0,
            "tail_new_word_rate": 0.0,
            "tail_repeat_rate": 0.0,
            "last_new_content_word_pos_frac": 0.0,
        }
    tail_n = max(1, len(words) // 3)
    head = words[: max(0, len(words) - tail_n)]
    tail = words[-tail_n:]
    seen_head = set(head)
    tail_new = sum(1 for word in tail if word not in seen_head)
    tail_repeat = sum(1 for word in tail if word in seen_head)
    seen = set()
    last_new_idx = 0
    for idx, word in enumerate(words):
        if word not in seen:
            last_new_idx = int(idx)
            seen.add(word)
    return {
        "content_word_count": float(len(words)),
        "tail_new_word_rate": float(tail_new / float(max(1, len(tail)))),
        "tail_repeat_rate": float(tail_repeat / float(max(1, len(tail)))),
        "last_new_content_word_pos_frac": float(last_new_idx / float(max(1, len(words) - 1))),
    }


def load_row_map(path: str, key: str = "question_id") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = safe_id(row.get(key, row.get("id")))
        if sid:
            out[sid] = row
    return out


def load_trace_map(path: str) -> Dict[str, Dict[str, str]]:
    p = str(path or "").strip()
    if not p:
        return {}
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(p):
        sid = safe_id(row.get("id", row.get("question_id")))
        if sid:
            out[sid] = row
    return out


def make_answer_id() -> str:
    if shortuuid is not None:
        return shortuuid.uuid()
    return uuid.uuid4().hex


def should_repair(
    trace: Dict[str, str],
    stats: Dict[str, float],
    repair_all: bool,
    last_new_pos_frac_max: float,
    tail_new_word_rate_max: float,
    tail_repeat_rate_min: float,
    min_content_words: int,
) -> Tuple[bool, str]:
    if repair_all:
        return True, "repair_all"
    n_words = int(stats.get("content_word_count", 0.0))
    if n_words < int(min_content_words):
        return False, "too_short"

    tail_new = maybe_float(trace.get("probe_tail_content_new_word_rate"))
    if tail_new is None:
        tail_new = stats.get("tail_new_word_rate", 0.0)
    tail_repeat = maybe_float(trace.get("probe_tail_content_repeat_rate"))
    if tail_repeat is None:
        tail_repeat = stats.get("tail_repeat_rate", 0.0)
    last_new = maybe_float(trace.get("probe_last_new_content_word_pos_frac"))
    if last_new is None:
        last_new = stats.get("last_new_content_word_pos_frac", 0.0)

    if float(last_new) <= float(last_new_pos_frac_max) and float(tail_new) <= float(tail_new_word_rate_max):
        return True, "early_last_new_and_low_tail_new"
    if float(tail_repeat) >= float(tail_repeat_rate_min) and float(tail_new) <= float(tail_new_word_rate_max):
        return True, "tail_repeat_and_low_tail_new"
    return False, "risk_gate_failed"


def choose_prefix(
    text: str,
    trace: Dict[str, str],
    min_prefix_content_words: int,
    prefix_margin_content_words: int,
    fallback_prefix_frac: float,
) -> Tuple[str, int, Dict[str, float]]:
    spans = content_word_spans(text)
    stats = text_novelty_stats(text)
    if not spans:
        return str(text or "").strip(), len(str(text or "")), stats

    frac = maybe_float(trace.get("probe_last_new_content_word_pos_frac"))
    if frac is None:
        frac = stats.get("last_new_content_word_pos_frac", fallback_prefix_frac)
    frac = min(1.0, max(0.0, float(frac)))
    idx = int(round(frac * float(max(0, len(spans) - 1))))
    idx = max(int(min_prefix_content_words) - 1, idx + int(prefix_margin_content_words))
    idx = min(max(0, idx), len(spans) - 1)
    char_end = int(spans[idx][1])

    # Prefer ending at the next whitespace/punctuation boundary. This avoids
    # passing a chopped word as the assistant prefix.
    while char_end < len(text) and text[char_end].isalnum():
        char_end += 1
    prefix = str(text[:char_end]).strip()
    if not prefix:
        prefix = str(text or "").strip()
        char_end = len(str(text or ""))
    return prefix, char_end, stats


def build_prompt(question: str, conv_mode: str, mm_use_im_start_end: bool, assistant_prefix: str) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
    from llava.conversation import conv_templates

    qs = str(question or "").strip()
    if mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prefix = str(assistant_prefix or "").strip()
    if prefix:
        prompt = prompt + " " + prefix
    return prompt


def clean_suffix(text: str, prefix: str) -> str:
    suffix = str(text or "").strip()
    prefix_norm = str(prefix or "").strip()
    if prefix_norm and suffix.lower().startswith(prefix_norm.lower()):
        suffix = suffix[len(prefix_norm) :].strip()
    # LLaVA can emit separator fragments when forced to continue from an
    # assistant prefix; trim the common ones before concatenating.
    for sep in ("</s>", "###", "ASSISTANT:", "Assistant:"):
        if sep in suffix:
            suffix = suffix.split(sep, 1)[0].strip()
    return suffix


def join_prefix_suffix(prefix: str, suffix: str) -> str:
    p = str(prefix or "").strip()
    s = str(suffix or "").strip()
    if not p:
        return s
    if not s:
        return p
    if p[-1] in {"-", "/", "("}:
        return (p + s).strip()
    if p[-1] in {".", ",", ";", ":"}:
        return f"{p} {s}".strip()
    return f"{p} {s}".strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate prefix-preserving suffix repair captions from intervention outputs."
    )
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--trace_features_csv", type=str, default="")
    ap.add_argument("--pred_text_key", type=str, default="auto")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse_if_exists", type=str, default="false")
    ap.add_argument("--repair_all", action="store_true")
    ap.add_argument("--last_new_pos_frac_max", type=float, default=0.65)
    ap.add_argument("--tail_new_word_rate_max", type=float, default=0.20)
    ap.add_argument("--tail_repeat_rate_min", type=float, default=0.80)
    ap.add_argument("--min_content_words_for_repair", type=int, default=6)
    ap.add_argument("--min_suffix_content_words", type=int, default=1)
    ap.add_argument("--min_prefix_content_words", type=int, default=4)
    ap.add_argument("--prefix_margin_content_words", type=int, default=1)
    ap.add_argument("--fallback_prefix_frac", type=float, default=0.60)
    args = ap.parse_args()

    if str(args.reuse_if_exists).strip().lower() in {"1", "true", "yes", "y"} and os.path.isfile(args.out_jsonl):
        print("[reuse]", os.path.abspath(args.out_jsonl))
        return

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    questions = read_jsonl(os.path.abspath(args.question_file))
    if int(args.limit) > 0:
        questions = questions[: int(args.limit)]
    pred_map = load_row_map(os.path.abspath(args.intervention_pred_jsonl))
    trace_map = load_trace_map(args.trace_features_csv)

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    model_base = args.model_base if str(args.model_base or "").strip() else None
    tokenizer, model, image_processor, _ = load_pretrained_model(
        os.path.expanduser(args.model_path),
        model_base,
        model_name,
        device=str(args.device),
    )
    device = torch.device(str(args.device))

    out_rows: List[Dict[str, Any]] = []
    n_repaired = 0
    n_missing_pred = 0
    for q in tqdm(questions):
        sid = safe_id(q.get("question_id", q.get("id")))
        pred = pred_map.get(sid, {})
        int_text = pick_text(pred, args.pred_text_key)
        if not int_text:
            n_missing_pred += 1
        trace = trace_map.get(sid, {})
        prefix, prefix_char_end, stats = choose_prefix(
            int_text,
            trace,
            min_prefix_content_words=int(args.min_prefix_content_words),
            prefix_margin_content_words=int(args.prefix_margin_content_words),
            fallback_prefix_frac=float(args.fallback_prefix_frac),
        )
        repair, repair_reason = should_repair(
            trace=trace,
            stats=stats,
            repair_all=bool(args.repair_all),
            last_new_pos_frac_max=float(args.last_new_pos_frac_max),
            tail_new_word_rate_max=float(args.tail_new_word_rate_max),
            tail_repeat_rate_min=float(args.tail_repeat_rate_min),
            min_content_words=int(args.min_content_words_for_repair),
        )

        suffix = ""
        final_text = int_text
        repair_attempted = bool(repair and int_text)
        repair_applied = False
        if repair and int_text:
            image = Image.open(os.path.join(args.image_folder, str(q["image"]))).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)
            prompt = build_prompt(
                question=str(q.get("text", "")),
                conv_mode=str(args.conv_mode),
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
                assistant_prefix=prefix,
            )
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=bool(float(args.temperature) > 0.0),
                    temperature=float(args.temperature),
                    top_p=args.top_p,
                    num_beams=int(args.num_beams),
                    max_new_tokens=int(args.max_new_tokens),
                    use_cache=True,
                )
            raw_suffix = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            suffix = clean_suffix(raw_suffix, prefix)
            if len(content_word_spans(suffix)) >= int(args.min_suffix_content_words):
                final_text = join_prefix_suffix(prefix, suffix)
                repair_applied = True
                n_repaired += 1
            else:
                final_text = int_text
                repair_reason = f"{repair_reason}:short_or_empty_suffix"

        image_id = q.get("image_id", pred.get("image_id", sid))
        out_rows.append(
            {
                "question_id": sid,
                "image_id": image_id,
                "image": q.get("image", pred.get("image", "")),
                "prompt": q.get("text", ""),
                "text": final_text,
                "output": final_text,
                "intervention_text": int_text,
                "repair_attempted": int(bool(repair_attempted)),
                "repair_applied": int(bool(repair_applied)),
                "repair_reason": repair_reason,
                "repair_prefix": prefix,
                "repair_suffix": suffix,
                "repair_prefix_char_end": int(prefix_char_end),
                "answer_id": make_answer_id(),
                "model_id": model_name,
                "metadata": {},
            }
        )

    write_jsonl(os.path.abspath(args.out_jsonl), out_rows)
    summary = {
        "inputs": {
            "question_file": os.path.abspath(args.question_file),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "trace_features_csv": os.path.abspath(args.trace_features_csv) if str(args.trace_features_csv).strip() else "",
            "model_path": str(args.model_path),
            "limit": int(args.limit),
        },
        "counts": {
            "n_rows": int(len(out_rows)),
            "n_repaired": int(n_repaired),
            "repair_rate": float(n_repaired / float(max(1, len(out_rows)))),
            "n_missing_pred": int(n_missing_pred),
        },
        "outputs": {
            "jsonl": os.path.abspath(args.out_jsonl),
        },
    }
    if str(args.out_summary_json).strip():
        write_json(os.path.abspath(args.out_summary_json), summary)
        print("[saved]", os.path.abspath(args.out_summary_json))
    print("[saved]", os.path.abspath(args.out_jsonl))


if __name__ == "__main__":
    main()
