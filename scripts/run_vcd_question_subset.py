#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VCD on an arbitrary question subset jsonl.")
    ap.add_argument("--vcd_root", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--answers_file", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--noise_step", type=int, default=500)
    ap.add_argument("--use_cd", action="store_true")
    ap.add_argument("--cd_alpha", type=float, default=1.0)
    ap.add_argument("--cd_beta", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vcd_root = os.path.abspath(args.vcd_root)
    os.chdir(vcd_root)
    experiments_root = os.path.join(vcd_root, "experiments")
    if vcd_root not in sys.path:
        sys.path.insert(0, vcd_root)
    if experiments_root not in sys.path:
        sys.path.insert(0, experiments_root)

    from transformers import set_seed  # type: ignore
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN  # type: ignore
    from llava.conversation import conv_templates, SeparatorStyle  # type: ignore
    from llava.model.builder import load_pretrained_model  # type: ignore
    from llava.utils import disable_torch_init  # type: ignore
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path  # type: ignore
    from vcd_utils.vcd_add_noise import add_diffusion_noise  # type: ignore
    from vcd_utils.vcd_sample import evolve_vcd_sampling  # type: ignore

    evolve_vcd_sampling()
    set_seed(int(args.seed))
    disable_torch_init()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu_id))

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        (args.model_base or None),
        model_name,
    )

    rows = read_jsonl(os.path.abspath(args.question_file))
    out_path = os.path.abspath(args.answers_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise SystemExit(f"Output already exists: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="vcd-subset", unit="sample"):
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_name = str(row.get("image", "")).strip()
            query = str(row.get("question", row.get("text", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not qid or not image_name or not query:
                continue

            cur_prompt = query
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + query

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            image = Image.open(os.path.join(args.image_folder, image_name)).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            if args.use_cd:
                image_tensor_cd = add_diffusion_noise(image_tensor, int(args.noise_step))
            else:
                image_tensor_cd = None

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            top_k = None if int(args.top_k) <= 0 else int(args.top_k)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                    cd_alpha=float(args.cd_alpha),
                    cd_beta=float(args.cd_beta),
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=top_k,
                    max_new_tokens=int(args.max_new_tokens),
                    use_cache=True,
                )

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            output_text = outputs.strip()

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
                        "model": model_name,
                        "prompt": cur_prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
