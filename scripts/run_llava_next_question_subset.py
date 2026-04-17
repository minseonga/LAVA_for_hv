#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from transformers import set_seed


TORCH_TYPE_MAP = {
    "fp16": "float16",
    "fp32": "float32",
    "bf16": "bfloat16",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def patch_transformers_compat() -> None:
    import transformers.cache_utils as cache_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils
    from transformers import CLIPVisionModel
    from transformers.generation.utils import GenerationMixin

    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))

    if not getattr(CLIPVisionModel, "_llava_next_safetensors_patch", False):
        original_from_pretrained = CLIPVisionModel.from_pretrained

        def from_pretrained_with_safetensors(cls: Any, pretrained_model_name_or_path: Any, *a: Any, **kw: Any) -> Any:
            kw.setdefault("use_safetensors", True)
            return original_from_pretrained(pretrained_model_name_or_path, *a, **kw)

        CLIPVisionModel.from_pretrained = classmethod(from_pretrained_with_safetensors)
        CLIPVisionModel._llava_next_safetensors_patch = True  # type: ignore[attr-defined]

    # LLaVA-Next vendor classes call super().generate(), relying on the
    # pre-4.50 behavior where PreTrainedModel inherited GenerationMixin.
    for name, value in GenerationMixin.__dict__.items():
        if name.startswith("__"):
            continue
        if not hasattr(modeling_utils.PreTrainedModel, name):
            setattr(modeling_utils.PreTrainedModel, name, value)

    if not hasattr(cache_utils.Cache, "seen_tokens"):
        cache_utils.Cache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]
    if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "seen_tokens"):
        cache_utils.DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())  # type: ignore[attr-defined]
    if not hasattr(cache_utils.Cache, "get_max_length") and hasattr(cache_utils.Cache, "get_max_cache_shape"):
        cache_utils.Cache.get_max_length = cache_utils.Cache.get_max_cache_shape  # type: ignore[attr-defined]
    if hasattr(cache_utils, "DynamicCache") and not getattr(cache_utils.DynamicCache, "_llava_next_none_legacy_patch", False):
        original_from_legacy_cache = cache_utils.DynamicCache.from_legacy_cache

        @classmethod
        def from_legacy_cache_with_none(cls: Any, past_key_values: Any = None) -> Any:
            if past_key_values is None:
                return cls()
            if isinstance(past_key_values, cache_utils.Cache):
                return past_key_values
            return original_from_legacy_cache(past_key_values)

        cache_utils.DynamicCache.from_legacy_cache = from_legacy_cache_with_none
        cache_utils.DynamicCache._llava_next_none_legacy_patch = True  # type: ignore[attr-defined]


def ensure_generation_config(model: Any, tokenizer: Any) -> None:
    from transformers import GenerationConfig

    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if getattr(model.generation_config, attr, None) is None and getattr(tokenizer, attr, None) is not None:
            setattr(model.generation_config, attr, getattr(tokenizer, attr))
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = getattr(model.generation_config, "eos_token_id", None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run vanilla LLaVA-Next on an arbitrary image-question JSONL subset.")
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--conv-mode", default="llava_llama_3")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--torch-type", default="bf16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--attn-type", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--do-sample", type=parse_bool, default=False)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--use-cache", type=parse_bool, default=True)
    args = ap.parse_args()

    if os.path.exists(args.answers_file):
        raise FileExistsError(f"answers file already exists: {args.answers_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    vga_root = Path(args.vga_root)
    if not vga_root.is_absolute():
        vga_root = (repo_root / vga_root).resolve()
    for path in (str(vga_root), str(vga_root / "eval")):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    patch_transformers_compat()

    from llava_next.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava_next.conversation import SeparatorStyle, conv_templates
    from llava_next.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
    from llava_next.model.builder import load_pretrained_model
    from llava_next.utils import disable_torch_init

    set_seed(int(args.seed))
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_base = os.path.expanduser(args.model_base) if str(args.model_base or "").strip() else None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map="cuda",
        attn_implementation=str(args.attn_type),
        torch_dtype=TORCH_TYPE_MAP[str(args.torch_type)],
    )
    ensure_generation_config(model, tokenizer)
    model.eval()

    rows = read_jsonl(args.question_file, limit=int(args.limit))
    with open(args.answers_file, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="llava-next", unit="sample"):
            image_name = str(row.get("image", "")).strip()
            question = str(row.get("question", row.get("text", ""))).strip()
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not image_name or not question or not qid:
                continue

            if bool(getattr(model.config, "mm_use_im_start_end", False)):
                model_question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                model_question = DEFAULT_IMAGE_TOKEN + "\n" + question

            conv = conv_templates[args.conv_mode].copy()
            if conv.sep_style == SeparatorStyle.LLAMA_3 and conv.tokenizer is None:
                conv.tokenizer = tokenizer
            conv.append_message(conv.roles[0], model_question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_name)).convert("RGB")
            image_sizes = [image.size]
            image_tensor = process_images([image], image_processor, model.config)[0]
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            gen_kwargs: Dict[str, Any] = {
                "images": image_tensor.unsqueeze(0).to(model.dtype).cuda(),
                "image_sizes": image_sizes,
                "do_sample": bool(args.do_sample),
                "num_beams": int(args.num_beams),
                "max_new_tokens": int(args.max_new_tokens),
                "use_cache": bool(args.use_cache),
                "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
                "stopping_criteria": [stopping_criteria],
                "pad_token_id": tokenizer.eos_token_id,
            }
            if bool(args.do_sample):
                gen_kwargs["temperature"] = float(args.temperature)
                gen_kwargs["top_p"] = float(args.top_p)

            with torch.inference_mode():
                output_ids = model.generate(input_ids, **gen_kwargs)

            if int(output_ids.shape[1]) > int(input_ids.shape[1]):
                gen_ids = output_ids[:, input_ids.shape[1] :]
            else:
                gen_ids = output_ids
            output_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            if output_text.endswith(stop_str):
                output_text = output_text[: -len(stop_str)].strip()

            f.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": question,
                        "output": output_text,
                        "text": output_text,
                        "caption": output_text,
                        "label": row.get("label", ""),
                        "prompt": prompt,
                        "model_id": model_name,
                        "image": image_name,
                        "image_id": image_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
