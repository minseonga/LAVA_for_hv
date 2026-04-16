#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm


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
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def load_qwen35_vl_model(model_path: str, torch_dtype: str, device_map: str, attn_implementation: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype_value: Any
    if str(torch_dtype).lower() == "auto":
        dtype_value = "auto"
    else:
        dtype_value = getattr(torch, str(torch_dtype))

    kwargs: Dict[str, Any] = {
        "dtype": dtype_value,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if str(attn_implementation).strip():
        kwargs["attn_implementation"] = str(attn_implementation)

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs).eval()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Qwen3.5-VL through AutoModelForImageTextToText. "
            "Set MODEL_PATH to the exact Qwen3.5 vision-language checkpoint and "
            "use a transformers version that supports that checkpoint."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Qwen3.5-VL on an arbitrary image-question JSONL subset.")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--do-sample", type=parse_bool, default=False)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    if os.path.exists(args.answers_file):
        raise FileExistsError(f"answers file already exists: {args.answers_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)

    setup_seed(int(args.seed))
    model, processor = load_qwen35_vl_model(
        model_path=os.path.expanduser(args.model_path),
        torch_dtype=str(args.torch_dtype),
        device_map=str(args.device_map),
        attn_implementation=str(args.attn_implementation),
    )

    rows = read_jsonl(args.question_file, limit=int(args.limit))
    device = model_device(model)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
    }
    if bool(args.do_sample):
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["top_p"] = float(args.top_p)

    with open(args.answers_file, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="qwen35-vl", unit="sample"):
            image_name = str(row.get("image", "")).strip()
            question = str(row.get("question", row.get("text", ""))).strip()
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not image_name or not question or not qid:
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
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

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
                        "model_id": os.path.basename(str(args.model_path).rstrip("/")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
