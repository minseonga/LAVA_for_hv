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
from transformers import set_seed

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

TORCH_TYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

OFFICIAL_TORCH_TYPE_ARG = {
    "fp16": "float16",
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


def normalize_model_base(value: Any) -> str | None:
    text = str(value or "").strip()
    return os.path.expanduser(text) if text else None


def to_cuda_images(images: Any, dtype: torch.dtype) -> Any:
    if isinstance(images, torch.Tensor):
        return images.to(device="cuda", dtype=dtype)
    if isinstance(images, list):
        return [x.to(device="cuda", dtype=dtype) for x in images]
    raise TypeError(f"Unsupported processed image type: {type(images)!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run official LLaVA-NeXT on an arbitrary image-question JSONL subset.")
    ap.add_argument("--llava-next-root", default=os.environ.get("LLAVA_NEXT_ROOT", "/home/kms/LLaVA-NeXT"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--conv-mode", default="llava_llama_3")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--torch-type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--attn-implementation", default="eager", choices=["none", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--do-sample", type=parse_bool, default=False)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--num-beams", type=int, default=1)
    args = ap.parse_args()

    if os.path.exists(args.answers_file):
        raise FileExistsError(f"answers file already exists: {args.answers_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)

    llava_next_root = Path(args.llava_next_root).expanduser().resolve()
    if not llava_next_root.exists():
        raise FileNotFoundError(f"official LLaVA-NeXT repo not found: {llava_next_root}")
    sys.path.insert(0, str(llava_next_root))

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    set_seed(int(args.seed))
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_base = normalize_model_base(args.model_base)
    model_name = str(args.model_name).strip() or get_model_name_from_path(model_path)

    load_kwargs: Dict[str, Any] = {
        "device_map": "cuda",
        "torch_dtype": OFFICIAL_TORCH_TYPE_ARG[str(args.torch_type)],
    }
    attn_implementation = str(args.attn_implementation)
    if attn_implementation == "none":
        # Official LLaVA-NeXT defaults to flash_attention_2 inside builder.py.
        # Use eager as the explicit no-flash-attn fallback.
        attn_implementation = "eager"
    load_kwargs["attn_implementation"] = attn_implementation

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        **load_kwargs,
    )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    use_official_mistral_path = "mistral" in str(args.conv_mode).lower() or "mistral" in str(model_name).lower()

    rows = read_jsonl(args.question_file, limit=int(args.limit))
    with open(args.answers_file, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="official-llava-next", unit="sample"):
            image_name = str(row.get("image", "")).strip()
            question = str(row.get("question", row.get("text", ""))).strip()
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not image_name or not question or not qid:
                continue

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
            if use_official_mistral_path:
                # Match the official LLaVA-NeXT Mistral VQA path. With
                # process_images + image_sizes, llava-v1.6-mistral-7b can
                # immediately emit EOS in this env, producing empty outputs.
                image_tensor = [
                    image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(
                        device="cuda",
                        dtype=TORCH_TYPE_MAP[str(args.torch_type)],
                    )
                ]
                image_sizes = []
            else:
                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = to_cuda_images(image_tensor, TORCH_TYPE_MAP[str(args.torch_type)])

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            gen_kwargs: Dict[str, Any] = {
                "images": image_tensor,
                "do_sample": bool(args.do_sample),
                "num_beams": int(args.num_beams),
                "max_new_tokens": int(args.max_new_tokens),
                "use_cache": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if image_sizes:
                gen_kwargs["image_sizes"] = image_sizes
            if not use_official_mistral_path:
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                gen_kwargs["modalities"] = ["image"] * int(input_ids.shape[0])
            if bool(args.do_sample):
                gen_kwargs["temperature"] = float(args.temperature)
                if args.top_p is not None:
                    gen_kwargs["top_p"] = float(args.top_p)

            with torch.inference_mode():
                try:
                    output_ids = model.generate(input_ids, **gen_kwargs)
                except ValueError as exc:
                    if "model_kwargs" not in str(exc) or "modalities" not in str(exc):
                        raise
                    gen_kwargs.pop("modalities", None)
                    output_ids = model.generate(input_ids, **gen_kwargs)

            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
