#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract VGA-style visual confidence / guidance-map features per sample.

Design goal:
- Keep preprocessing/prompting compatible with VGA eval path.
- Use same model (e.g., LLaVA-1.5) and object-conditioned grounding rule.
- Dump patch-level aggregate stats for offline feature screening.

Notes:
- This script only extracts VSC/G features from a single forward pass.
- It does NOT run the full VGA generation intervention loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import set_seed


# VGA-origin imports (runtime path injection)
def add_vga_paths(vga_root: str) -> None:
    vga_root = os.path.abspath(vga_root)
    sys.path.append(vga_root)
    sys.path.append(os.path.join(vga_root, "eval"))


def safe_list_object(obj_field: Any) -> List[str]:
    if obj_field is None:
        return []
    if isinstance(obj_field, str):
        return [obj_field] if obj_field.strip() else []
    if isinstance(obj_field, list):
        out: List[str] = []
        for x in obj_field:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    return []


def norm_entropy(prob: torch.Tensor, eps: float = 1e-12) -> float:
    # normalized Shannon entropy in [0,1]
    n = int(prob.numel())
    if n <= 1:
        return 0.0
    p = prob.clamp(min=eps)
    h = -(p * torch.log(p)).sum()
    return float((h / math.log(n)).item())


def effective_support_size(prob: torch.Tensor, eps: float = 1e-12) -> float:
    # 1 / sum p^2
    d = float(torch.sum(prob * prob).item())
    return float(1.0 / max(eps, d))


def build_prompt(question: str, conv_mode: str, model: Any) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
    from llava.conversation import conv_templates

    qs = str(question)
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_questions(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
            if limit > 0 and len(rows) >= int(limit):
                break
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract VGA-style VSC/G features for POPE-style jsonl.")
    ap.add_argument("--vga_root", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary", type=str, required=True)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--obj_topk", type=int, default=5)
    ap.add_argument("--entropy_topk", type=int, default=10)
    ap.add_argument("--img_start_idx", type=int, default=35, help="LLaVA-1.5 visual token start index")
    ap.add_argument("--img_end_idx", type=int, default=611, help="LLaVA-1.5 visual token end index (exclusive)")
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # record-only fields to align with VGA eval settings
    ap.add_argument("--use_add", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--cd_alpha", type=float, default=0.02)
    ap.add_argument("--attn_coef", type=float, default=0.2)
    ap.add_argument("--start_layer", type=int, default=2)
    ap.add_argument("--end_layer", type=int, default=15)
    ap.add_argument("--head_balancing", type=str, default="simg")
    ap.add_argument("--attn_norm", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--sampling", type=lambda x: str(x).lower() == "true", default=False)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)), exist_ok=True)

    random.seed(int(args.seed))
    set_seed(int(args.seed))

    add_vga_paths(args.vga_root)

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model_base = (None if str(args.model_base).strip() == "" else args.model_base)

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, model_name)
    tokenizer.padding_side = "right"

    rows = load_questions(args.question_file, limit=int(args.num_samples))

    out_rows: List[Dict[str, Any]] = []
    n_obj_mode = 0
    n_entropy_mode = 0
    n_skip = 0

    for line in tqdm(rows, desc="vga-vsc"):
        qid = str(line.get("question_id", line.get("id", "")))
        if qid == "":
            n_skip += 1
            continue

        image_file = str(line.get("image", "")).strip()
        if image_file == "":
            n_skip += 1
            continue

        question = str(line.get("question", line.get("text", ""))).strip()
        if question == "":
            n_skip += 1
            continue

        obj_list = safe_list_object(line.get("object", None))

        prompt = build_prompt(question, args.conv_mode, model)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        with torch.inference_mode():
            outputs = model(
                input_ids[:, :-1],
                images=image_tensor.unsqueeze(0).half().cuda(),
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits
            vis_logits = F.softmax(logits[0, int(args.img_start_idx):int(args.img_end_idx), :], dim=-1).float()

            object_id = []
            for o in obj_list:
                ids = tokenizer(o, add_special_tokens=False, return_tensors="pt").input_ids[0]
                if ids.numel() > 0:
                    object_id.append(ids)

            if len(object_id) > 0:
                grounding_list = []
                for ids in object_id:
                    # keep VGA-origin behavior: use first token of object phrase
                    vl = vis_logits[:, ids]
                    vl = vl[:, 0]
                    grounding_list.append(vl)
                grounding = torch.stack(grounding_list, dim=0).max(0).values
                mode = "object"
                n_obj_mode += 1
            else:
                k = max(2, min(int(args.entropy_topk), int(vis_logits.size(-1))))
                top_k_scores, _ = torch.topk(vis_logits, k, dim=-1)
                probs = -top_k_scores * torch.log(top_k_scores + 1e-8) / math.log(k)
                grounding = probs.sum(-1)
                mode = "entropy_fallback"
                n_entropy_mode += 1

            grounding = grounding.float().clamp(min=0.0)
            gsum = grounding.sum()
            if float(gsum.item()) <= 0.0:
                G = torch.ones_like(grounding) / float(max(1, grounding.numel()))
            else:
                G = grounding / gsum

            k_obj = max(1, min(int(args.obj_topk), int(grounding.numel())))
            obj_topk_mean = float(torch.topk(grounding, k_obj).values.mean().item())

            G_top1 = float(torch.topk(G, 1).values.sum().item())
            k5 = max(1, min(5, int(G.numel())))
            G_top5 = float(torch.topk(G, k5).values.sum().item())
            G_ent = norm_entropy(G)
            G_ess = effective_support_size(G)

            out_rows.append(
                {
                    "id": qid,
                    "question_id": qid,
                    "image": image_file,
                    "question": question,
                    "object_count": len(obj_list),
                    "mode": mode,
                    "obj_token_prob_max": float(grounding.max().item()),
                    "obj_token_prob_mean": float(grounding.mean().item()),
                    "obj_token_prob_lse": float(torch.logsumexp(grounding, dim=0).item()),
                    "obj_token_prob_topkmean": obj_topk_mean,
                    "entropy_score": G_ent,
                    "G_entropy": G_ent,
                    "G_top1_mass": G_top1,
                    "G_top5_mass": G_top5,
                    "G_effective_support_size": G_ess,
                }
            )

    write_csv(args.out_csv, out_rows)

    summary = {
        "inputs": {
            "vga_root": os.path.abspath(args.vga_root),
            "model_path": args.model_path,
            "model_base": model_base,
            "image_folder": os.path.abspath(args.image_folder),
            "question_file": os.path.abspath(args.question_file),
            "conv_mode": args.conv_mode,
            "obj_topk": int(args.obj_topk),
            "entropy_topk": int(args.entropy_topk),
            "img_start_idx": int(args.img_start_idx),
            "img_end_idx": int(args.img_end_idx),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "eval_alignment": {
                "use_add": bool(args.use_add),
                "cd_alpha": float(args.cd_alpha),
                "attn_coef": float(args.attn_coef),
                "start_layer": int(args.start_layer),
                "end_layer": int(args.end_layer),
                "head_balancing": str(args.head_balancing),
                "attn_norm": bool(args.attn_norm),
                "sampling": bool(args.sampling),
            },
        },
        "counts": {
            "n_input": int(len(rows)),
            "n_output": int(len(out_rows)),
            "n_obj_mode": int(n_obj_mode),
            "n_entropy_fallback": int(n_entropy_mode),
            "n_skipped": int(n_skip),
        },
        "outputs": {
            "out_csv": os.path.abspath(args.out_csv),
            "out_summary": os.path.abspath(args.out_summary),
        },
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", args.out_csv)
    print("[saved]", args.out_summary)


if __name__ == "__main__":
    main()
