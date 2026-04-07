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


def _to_hw(size_like: Any) -> tuple[int, int]:
    if isinstance(size_like, dict):
        h = int(size_like.get("height", size_like.get("shortest_edge", 224)))
        w = int(size_like.get("width", size_like.get("shortest_edge", h)))
        return h, w
    if isinstance(size_like, (list, tuple)) and len(size_like) >= 2:
        return int(size_like[0]), int(size_like[1])
    size = int(size_like or 224)
    return size, size


def preprocess_image_no_numpy(image: Image.Image, image_processor: Any) -> torch.Tensor:
    image = image.convert("RGB")
    resize_h, resize_w = _to_hw(getattr(image_processor, "size", 224))
    crop_h, crop_w = _to_hw(getattr(image_processor, "crop_size", (resize_h, resize_w)))

    if bool(getattr(image_processor, "do_resize", True)):
        if isinstance(getattr(image_processor, "size", None), dict) and "shortest_edge" in getattr(image_processor, "size", {}):
            target = int(getattr(image_processor, "size", {}).get("shortest_edge", min(resize_h, resize_w)))
            w, h = image.size
            if min(w, h) > 0:
                if w <= h:
                    new_w = target
                    new_h = int(round(h * float(target) / float(w)))
                else:
                    new_h = target
                    new_w = int(round(w * float(target) / float(h)))
                image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        else:
            image = image.resize((resize_w, resize_h), resample=Image.BICUBIC)

    if bool(getattr(image_processor, "do_center_crop", True)):
        w, h = image.size
        left = max(0, int(round((w - crop_w) / 2.0)))
        top = max(0, int(round((h - crop_h) / 2.0)))
        image = image.crop((left, top, min(w, left + crop_w), min(h, top + crop_h)))
        if image.size != (crop_w, crop_h):
            canvas = Image.new("RGB", (crop_w, crop_h))
            canvas.paste(image, (0, 0))
            image = canvas

    byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = byte_tensor.view(image.size[1], image.size[0], 3).permute(2, 0, 1).to(torch.float32) / 255.0

    if bool(getattr(image_processor, "do_normalize", True)):
        mean = torch.tensor(getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073]), dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711]), dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


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
    ap.add_argument("--do_sample", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--noise_step", type=int, default=500)
    ap.add_argument("--use_cd", action="store_true")
    ap.add_argument("--cd_alpha", type=float, default=1.0)
    ap.add_argument("--cd_beta", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--question_suffix", type=str, default=" Please answer this question with one word.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    do_sample = str(args.do_sample).strip().lower() == "true"

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

            suffix = str(args.question_suffix or "")
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs + suffix)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            image = Image.open(os.path.join(args.image_folder, image_name)).convert("RGB")
            image_tensor = preprocess_image_no_numpy(image, image_processor)
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
                    do_sample=do_sample,
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
