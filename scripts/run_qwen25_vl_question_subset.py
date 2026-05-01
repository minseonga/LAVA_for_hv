#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable

try:
    import numpy as np
except Exception:
    np = None

TORCH_DTYPE_CHOICES = ["bf16", "bfloat16", "float16", "float32", "fp16", "fp32"]


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


def setup_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_model(args: argparse.Namespace):
    import torch

    vga_root = os.path.abspath(os.path.expanduser(str(args.vga_root)))
    if vga_root not in sys.path:
        sys.path.insert(0, vga_root)

    from qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # type: ignore
    from qwen_vl_utils import process_vision_info  # type: ignore
    from transformers import AutoProcessor, AutoTokenizer

    torch_dtype = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    model_path = os.path.expanduser(str(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=str(args.attn_type),
        torch_dtype=torch_dtype[str(args.torch_type)],
        device_map=str(args.device_map),
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        trust_remote_code=True,
    )
    return model, processor, tokenizer, process_vision_info


def main() -> None:
    ap = argparse.ArgumentParser(description="Run vanilla Qwen2.5-VL on an arbitrary image-question JSONL subset.")
    ap.add_argument("--vga-root", type=str, default="")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--image-folder", type=str, required=True)
    ap.add_argument("--question-file", type=str, required=True)
    ap.add_argument("--answers-file", type=str, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--torch-type", type=str, default="bf16", choices=TORCH_DTYPE_CHOICES)
    ap.add_argument("--attn-type", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--device-map", type=str, default="cuda")
    ap.add_argument("--min-pixels", type=int, default=14 * 14 * 1280)
    ap.add_argument("--max-pixels", type=int, default=28 * 28 * 1280)
    ap.add_argument("--do-sample", type=parse_bool, default=False)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    if not str(args.vga_root).strip():
        raise SystemExit("--vga-root is required so the local VGA_origin/qwen2_5_vl model code is importable.")

    answers_file = os.path.abspath(os.path.expanduser(str(args.answers_file)))
    if os.path.exists(answers_file):
        raise FileExistsError(f"answers file already exists: {answers_file}")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    setup_seed(int(args.seed))
    model, processor, _tokenizer, process_vision_info = load_model(args)
    rows = read_jsonl(args.question_file, limit=int(args.limit))

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": 1,
        "do_sample": bool(args.do_sample),
        "use_cache": True,
    }
    if bool(args.do_sample):
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["top_p"] = float(args.top_p)

    with open(answers_file, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="qwen25-vl", unit="sample"):
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            question = str(row.get("question", row.get("text", ""))).strip()
            image_name = str(row.get("image", "")).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not qid or not question or not image_name:
                continue

            image_path = os.path.abspath(os.path.join(args.image_folder, image_name))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file://" + image_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            messages_batch = [messages]
            text = processor.apply_chat_template(
                messages_batch,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages_batch)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)

            import torch

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            f.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": question,
                        "output": output_text,
                        "text": output_text,
                        "caption": output_text,
                        "image": image_name,
                        "image_id": image_id,
                        "label": row.get("label", ""),
                        "prompt": text[0] if isinstance(text, list) else str(text),
                        "model_id": os.path.basename(str(args.model_path).rstrip("/")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
