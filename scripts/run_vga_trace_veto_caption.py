#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import uuid
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList, set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VGA_ROOT = os.environ.get("VGA_ROOT", os.path.join(REPO_ROOT, "VGA_origin"))
for path in (REPO_ROOT, os.path.join(VGA_ROOT, "eval"), VGA_ROOT):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from vcd_utils.greedy_sample import evolve_greedy_sampling  # noqa: E402

evolve_greedy_sampling()


TRACE_VETO_STOPWORDS = {
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
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def make_answer_id() -> str:
    return uuid.uuid4().hex


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def split_list(items: List[Dict[str, Any]], n_chunks: int) -> List[List[Dict[str, Any]]]:
    chunk_size = math.ceil(len(items) / max(1, int(n_chunks)))
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items: List[Dict[str, Any]], n_chunks: int, chunk_idx: int) -> List[Dict[str, Any]]:
    chunks = split_list(items, n_chunks)
    return chunks[int(chunk_idx)]


def is_content_like_token(token_text: str, min_chars: int) -> bool:
    text = str(token_text or "").strip().lower()
    if len(text) < int(min_chars):
        return False
    if text in TRACE_VETO_STOPWORDS:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if not all(ch.isalpha() or ch in {"'", "-"} for ch in text):
        return False
    if text in {"ing", "ed", "ly", "ion", "ions", "er", "ers", "est"}:
        return False
    return True


class TraceRiskVetoProcessor(LogitsProcessor):
    """Object-vocab-free online veto for unstable content-like top tokens."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        penalty: float,
        max_vetoes: int,
        min_step: int,
        min_token_chars: int,
        min_entropy: float,
        max_top1_gap: float,
        max_top1_logprob: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.penalty = float(penalty)
        self.max_vetoes = int(max_vetoes)
        self.min_step = int(min_step)
        self.min_token_chars = int(min_token_chars)
        self.min_entropy = float(min_entropy)
        self.max_top1_gap = float(max_top1_gap)
        self.max_top1_logprob = float(max_top1_logprob)
        self.n_steps = 0
        self.n_vetoes = 0
        self.events: List[Dict[str, Any]] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = int(self.n_steps)
        self.n_steps += 1
        if self.n_vetoes >= self.max_vetoes or step < self.min_step or self.penalty <= 0.0:
            return scores
        if scores.ndim != 2 or scores.shape[0] != 1:
            return scores

        row_scores = scores[0].to(torch.float32)
        top2_vals, top2_idx = torch.topk(row_scores, k=2, dim=-1)
        top1_id = int(top2_idx[0].item())
        top1_logit = float(top2_vals[0].item())
        top2_logit = float(top2_vals[1].item())
        top1_gap = float(top1_logit - top2_logit)
        log_probs = torch.log_softmax(row_scores, dim=-1)
        probs = torch.softmax(row_scores, dim=-1)
        top1_lp = float(log_probs[top1_id].item())
        entropy = float((-(probs * log_probs).sum()).item())
        token_text = self.tokenizer.decode([top1_id], skip_special_tokens=True)

        if (
            is_content_like_token(token_text, self.min_token_chars)
            and entropy >= self.min_entropy
            and top1_gap <= self.max_top1_gap
            and top1_lp <= self.max_top1_logprob
        ):
            scores[:, top1_id] = scores[:, top1_id] - float(self.penalty)
            self.n_vetoes += 1
            self.events.append(
                {
                    "step": int(step),
                    "token_id": int(top1_id),
                    "token_text": str(token_text),
                    "top1_logprob": float(top1_lp),
                    "top1_gap": float(top1_gap),
                    "entropy": float(entropy),
                    "penalty": float(self.penalty),
                    "veto_index": int(self.n_vetoes),
                }
            )
        return scores


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = sorted(set().union(*[set(row.keys()) for row in rows])) if rows else [
        "question_id",
        "image_id",
        "image",
        "step",
        "token_id",
        "token_text",
        "top1_logprob",
        "top1_gap",
        "entropy",
        "penalty",
        "veto_index",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def eval_model(args: argparse.Namespace) -> None:
    disable_torch_init()
    if os.path.exists(args.answers_file):
        raise FileExistsError(f"There is already a file at {args.answers_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    ans_file = open(args.answers_file, "w", encoding="utf-8")

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r", encoding="utf-8")]
    questions = get_chunk(questions, int(args.num_chunks), int(args.chunk_idx))
    if int(args.limit) > 0:
        questions = questions[: int(args.limit)]
    trace_veto_rows: List[Dict[str, Any]] = []

    for line in tqdm(questions):
        image_file = line["image"]
        qs = line["question"]
        obj = line.get("object")
        if isinstance(obj, str):
            obj = [obj]
        if not obj:
            obj = None

        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        _ = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        if obj is not None and len(obj[0]) > 0:
            object_id = [tokenizer(o, add_special_tokens=False, return_tensors="pt").input_ids[0] for o in obj]
        else:
            object_id = None

        with torch.inference_mode():
            outputs = model(
                input_ids[:, :-1],
                images=image_tensor.unsqueeze(0).half().cuda(),
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits
            vis_logits = F.softmax(logits[0, 35:611, :], dim=-1)

            if object_id is not None:
                grounding = []
                for obj_ids in object_id:
                    vl = vis_logits[:, obj_ids]
                    vl = vl[:, 0]
                    grounding.append(vl)
                grounding = torch.stack(grounding, dim=0)
                grounding = grounding.max(0).values
                grounding /= grounding.sum(0)
                vl_guidance = grounding
            else:
                top_k_scores, _ = torch.topk(vis_logits, 10, dim=-1)
                top_k_scores = top_k_scores.float()
                probabilities = -top_k_scores * torch.log(top_k_scores + 1e-8) / torch.log(torch.tensor(10))
                entropy = probabilities.sum(-1)
                vl_guidance = entropy / entropy.sum(0)
                vl_guidance = vl_guidance.to(vis_logits.dtype)

            trace_veto_proc = None
            if bool(args.enable_trace_veto):
                trace_veto_proc = TraceRiskVetoProcessor(
                    tokenizer,
                    penalty=float(args.trace_veto_penalty),
                    max_vetoes=int(args.trace_veto_max_per_caption),
                    min_step=int(args.trace_veto_min_step),
                    min_token_chars=int(args.trace_veto_min_token_chars),
                    min_entropy=float(args.trace_veto_min_entropy),
                    max_top1_gap=float(args.trace_veto_max_top1_gap),
                    max_top1_logprob=float(args.trace_veto_max_top1_logprob),
                )

            gen_kwargs = dict(
                images=image_tensor.unsqueeze(0).half().cuda(),
                past_key_values=outputs.past_key_values,
                vl_guidance=vl_guidance,
                vis_logits=vis_logits,
                cd_alpha=args.cd_alpha,
                add_layer=list(range(args.start_layer, args.end_layer + 1)),
                attn_coef=args.attn_coef,
                use_add=args.use_add,
                head_balancing=args.head_balancing,
                attn_norm=args.attn_norm,
                do_sample=True,
                sampling=args.sampling,
                num_beams=1,
                max_new_tokens=args.max_gen_len,
                use_cache=True,
            )
            if trace_veto_proc is not None:
                gen_kwargs["logits_processor"] = LogitsProcessorList([trace_veto_proc])

            output_ids = model.generate(input_ids[:, -1:], **gen_kwargs)
            if trace_veto_proc is not None:
                for event in trace_veto_proc.events:
                    trace_veto_rows.append(
                        {
                            "question_id": line["question_id"],
                            "image_id": line.get("image_id", ""),
                            "image": image_file,
                            **event,
                        }
                    )

        outputs_text = tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)[0]
        outputs_text = outputs_text.split("ASSISTANT:")[-1].strip()
        if outputs_text.endswith(stop_str):
            outputs_text = outputs_text[: -len(stop_str)]
        outputs_text = outputs_text.strip()
        ans_file.write(
            json.dumps(
                {
                    "question_id": line["question_id"],
                    "question": line["question"],
                    "output": outputs_text,
                    "label": line.get("label", ""),
                    "prompt": prompt,
                    "model_id": model_name,
                    "image": image_file,
                    "image_id": line["image_id"],
                    "trace_veto_count": int(len(trace_veto_proc.events)) if trace_veto_proc is not None else 0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.write(json.dumps(vars(args), ensure_ascii=False) + "\n")
    ans_file.close()

    if str(args.trace_veto_debug_dump or "").strip():
        write_csv(os.path.abspath(args.trace_veto_debug_dump), trace_veto_rows)
        print(f"[saved] trace veto debug rows: {os.path.abspath(args.trace_veto_debug_dump)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--use_add", type=parse_bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--attn_coef", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_layer", type=int, default=99)
    parser.add_argument("--end_layer", type=int, default=0)
    parser.add_argument("--head_balancing", type=str, default="attn", choices=["vattn", "battn", "simg", "simv", "simb", "simb-simg", "none"])
    parser.add_argument("--attn_norm", type=parse_bool, default=False)
    parser.add_argument("--sampling", type=parse_bool, default=False)
    parser.add_argument("--enable_trace_veto", type=parse_bool, default=False)
    parser.add_argument("--trace_veto_penalty", type=float, default=3.0)
    parser.add_argument("--trace_veto_max_per_caption", type=int, default=1)
    parser.add_argument("--trace_veto_min_step", type=int, default=4)
    parser.add_argument("--trace_veto_min_token_chars", type=int, default=3)
    parser.add_argument("--trace_veto_min_entropy", type=float, default=3.5)
    parser.add_argument("--trace_veto_max_top1_gap", type=float, default=0.25)
    parser.add_argument("--trace_veto_max_top1_logprob", type=float, default=-1.5)
    parser.add_argument("--trace_veto_debug_dump", type=str, default="")
    args = parser.parse_args()
    _ = (args.temperature, args.top_p, args.top_k)
    random.seed(args.seed)
    set_seed(args.seed)
    print(str(args), flush=True)
    eval_model(args)


if __name__ == "__main__":
    main()
