#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def patch_legacy_transformers_bloom_masks() -> None:
    try:
        import transformers.models.bloom.modeling_bloom as bloom
    except Exception:
        return

    if not hasattr(bloom, "_expand_mask"):

        def _expand_mask(mask: torch.Tensor, tgt_length: Optional[int] = None) -> torch.BoolTensor:
            batch_size, src_length = mask.shape
            tgt_length = int(tgt_length) if tgt_length is not None else int(src_length)
            expanded_mask = ~mask[:, None, None, :].to(torch.bool)
            return expanded_mask.expand(batch_size, 1, tgt_length, src_length)

        bloom._expand_mask = _expand_mask  # type: ignore[attr-defined]

    if not hasattr(bloom, "_make_causal_mask"):

        def _make_causal_mask(
            input_ids_shape: Any,
            device: torch.device,
            past_key_values_length: int = 0,
        ) -> torch.BoolTensor:
            batch_size, tgt_length = int(input_ids_shape[0]), int(input_ids_shape[1])
            src_length = tgt_length + int(past_key_values_length)
            mask = torch.zeros((tgt_length, src_length), dtype=torch.bool, device=device)
            mask[:, int(past_key_values_length) :] = torch.triu(
                torch.ones((tgt_length, tgt_length), dtype=torch.bool, device=device),
                diagonal=1,
            )
            return mask[None, None, :, :].expand(batch_size, 1, tgt_length, src_length)

        bloom._make_causal_mask = _make_causal_mask  # type: ignore[attr-defined]


def import_vga_greedy_sample() -> Any:
    import transformers

    original_from_pretrained = transformers.AutoTokenizer.from_pretrained

    class _PlaceholderTokenizer:
        def convert_ids_to_tokens(self, token_ids: Any) -> List[str]:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.detach().cpu().reshape(-1).tolist()
            elif isinstance(token_ids, int):
                token_ids = [token_ids]
            return ["" for _ in token_ids]

    def patched_from_pretrained(pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(pretrained_model_name_or_path) == "path/to/llava-v1.5-7b":
            return _PlaceholderTokenizer()
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        import vcd_utils.greedy_sample as greedy_sample
    finally:
        transformers.AutoTokenizer.from_pretrained = original_from_pretrained
    return greedy_sample


def build_vss_guidance(vis_logits: torch.Tensor, *, mode: str, topk: int, eps: float = 1e-8) -> torch.Tensor:
    top_k_scores, _ = torch.topk(vis_logits, int(topk), dim=-1)
    p = top_k_scores.float().clamp_min(float(eps))
    denom = math.log(float(topk))
    if mode == "entropy":
        score = (-p * torch.log(p)).sum(dim=-1) / denom
    elif mode == "nll":
        score = (-torch.log(p)).sum(dim=-1) / denom
    else:
        raise ValueError(f"unsupported vss_mode={mode!r}")
    return (score / score.sum(dim=0).clamp_min(float(eps))).to(vis_logits.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VGA/LLaVA captioning with selectable VSS mode.")
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--max_gen_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use_add", type=parse_bool, default=True)
    ap.add_argument("--cd_alpha", type=float, default=0.02)
    ap.add_argument("--attn_coef", type=float, default=0.2)
    ap.add_argument("--start_layer", type=int, default=2)
    ap.add_argument("--end_layer", type=int, default=15)
    ap.add_argument(
        "--head_balancing",
        default="simg",
        choices=["vattn", "battn", "simg", "simv", "simb", "simb-simg", "none"],
    )
    ap.add_argument("--attn_norm", type=parse_bool, default=False)
    ap.add_argument("--sampling", type=parse_bool, default=False)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--vss_mode", default="entropy", choices=["entropy", "nll"])
    ap.add_argument("--vss_topk", type=int, default=10)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    vga_root = Path(args.vga_root)
    if not vga_root.is_absolute():
        vga_root = (repo_root / vga_root).resolve()
    sys.path.insert(0, str(vga_root))
    sys.path.insert(0, str(vga_root / "llava"))

    patch_legacy_transformers_bloom_masks()
    from transformers import set_seed

    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    greedy_sample = import_vga_greedy_sample()
    greedy_sample.evolve_greedy_sampling()

    set_seed(int(args.seed))
    disable_torch_init()

    if os.path.exists(args.answers_file):
        raise FileExistsError(f"answers file already exists: {args.answers_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    greedy_sample.tokenizer = tokenizer
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head

    questions = []
    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
            if int(args.limit) > 0 and len(questions) >= int(args.limit):
                break

    with open(args.answers_file, "w", encoding="utf-8") as ans_file:
        for line in tqdm(questions):
            image_file = line["image"]
            qs = line["question"]
            obj = line.get("object")
            if isinstance(obj, str):
                obj = [obj]
            if not obj:
                obj = None

            if model.config.mm_use_im_start_end:
                qs_for_model = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs_for_model = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_for_model)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

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
                        grounding.append(vl[:, 0])
                    grounding_t = torch.stack(grounding, dim=0).max(0).values
                    vl_guidance = (grounding_t / grounding_t.sum(0).clamp_min(1e-8)).to(vis_logits.dtype)
                else:
                    vl_guidance = build_vss_guidance(vis_logits, mode=args.vss_mode, topk=int(args.vss_topk))

                output_ids = model.generate(
                    input_ids[:, -1:],
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    past_key_values=outputs.past_key_values,
                    vl_guidance=vl_guidance,
                    vis_logits=vis_logits,
                    cd_alpha=float(args.cd_alpha),
                    add_layer=list(range(int(args.start_layer), int(args.end_layer) + 1)),
                    attn_coef=float(args.attn_coef),
                    use_add=bool(args.use_add),
                    head_balancing=args.head_balancing,
                    attn_norm=bool(args.attn_norm),
                    do_sample=True,
                    sampling=bool(args.sampling),
                    num_beams=1,
                    max_new_tokens=int(args.max_gen_len),
                    use_cache=True,
                )

            text = tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)[0]
            text = text.split("ASSISTANT:")[-1].strip()
            if text.endswith(stop_str):
                text = text[: -len(stop_str)]
            text = text.strip()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": line.get("question_id"),
                        "question": line.get("question"),
                        "output": text,
                        "label": line.get("label", ""),
                        "prompt": prompt,
                        "model_id": model_name,
                        "image": image_file,
                        "image_id": line.get("image_id", line.get("question_id")),
                        "vss_mode": args.vss_mode,
                        "head_balancing": args.head_balancing,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()

        ans_file.write(json.dumps(vars(args), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
