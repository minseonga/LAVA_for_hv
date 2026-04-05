#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def prepare_model_image(image_processor: Any, raw_image: Image.Image) -> Any:
    image = None
    if hasattr(image_processor, "preprocess"):
        try:
            image = image_processor.preprocess(raw_image, return_tensors="pt")
        except TypeError:
            image = None

    if image is None:
        try:
            image = image_processor(raw_image, return_tensors="pt")
        except TypeError:
            image = image_processor(raw_image)

    if isinstance(image, dict) and "pixel_values" in image:
        pixel_values = image["pixel_values"]
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        elif isinstance(pixel_values, (list, tuple)) and pixel_values and isinstance(pixel_values[0], np.ndarray):
            pixel_values = torch.from_numpy(np.stack(pixel_values))
        elif not torch.is_tensor(pixel_values):
            pixel_values = torch.as_tensor(pixel_values)
        if torch.is_tensor(pixel_values) and pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        image["pixel_values"] = pixel_values

    return image


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PAI on an arbitrary question subset jsonl.")
    ap.add_argument("--pai_root", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--answers_file", type=str, required=True)
    ap.add_argument("--model", type=str, default="llava-1.5")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--use_attn", action="store_true")
    ap.add_argument("--use_cfg", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.1)
    ap.add_argument("--start_layer", type=int, default=2)
    ap.add_argument("--end_layer", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=927)
    args = ap.parse_args()

    pai_root = os.path.abspath(args.pai_root)
    os.chdir(pai_root)
    if pai_root not in sys.path:
        sys.path.insert(0, pai_root)

    from llava.utils import disable_torch_init  # type: ignore
    from attention import llama_modify  # type: ignore
    from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE  # type: ignore
    from transformers.generation.logits_process import LogitsProcessorList  # type: ignore
    import model_loader as pai_model_loader  # type: ignore

    setup_seeds(int(args.seed))
    disable_torch_init()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu_id))

    original_load_llava_model = pai_model_loader.load_llava_model

    def patched_load_llava_model(_: str):
        return original_load_llava_model(os.path.expanduser(str(args.model_path)))

    pai_model_loader.load_llava_model = patched_load_llava_model
    model_loader = pai_model_loader.ModelLoader(args.model)

    rows = read_jsonl(os.path.abspath(args.question_file))
    out_path = os.path.abspath(args.answers_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise SystemExit(f"Output already exists: {out_path}")

    template = INSTRUCTION_TEMPLATE[args.model]
    if args.model in {"llava-1.5", "shikra"}:
        template = SYSTEM_MESSAGE + template

    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="pai-subset", unit="sample"):
            image_name = str(row.get("image", "")).strip()
            query = str(row.get("question", row.get("text", ""))).strip()
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not image_name or not query or not qid:
                continue

            raw_image = Image.open(os.path.join(args.image_folder, image_name)).convert("RGB")
            image = prepare_model_image(model_loader.image_processor, raw_image)
            questions, kwargs = model_loader.prepare_inputs_for_model(template, [query], image)

            llama_modify(
                model_loader.llm_model,
                int(args.start_layer),
                int(args.end_layer),
                bool(args.use_attn),
                float(args.alpha),
                bool(args.use_cfg),
                model_loader.img_start_idx,
                model_loader.img_end_idx,
            )

            logits_processor = None
            if args.use_cfg:
                logits_processor = model_loader.init_cfg_processor(
                    questions,
                    float(args.gamma),
                    int(args.beam),
                    int(args.start_layer),
                    int(args.end_layer),
                )
            if logits_processor is not None:
                kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

            with torch.inference_mode():
                outputs = model_loader.llm_model.generate(
                    do_sample=bool(args.sample),
                    max_new_tokens=int(args.max_new_tokens),
                    use_cache=True,
                    num_beams=int(args.beam),
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    **kwargs,
                )

            generated_ids = getattr(outputs, "sequences", outputs)
            output_text = str(model_loader.decode(generated_ids)[0]).strip()
            f.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": query,
                        "output": output_text,
                        "caption": output_text,
                        "text": output_text,
                        "image": image_name,
                        "image_id": image_id,
                        "label": row.get("label", ""),
                        "model": args.model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
