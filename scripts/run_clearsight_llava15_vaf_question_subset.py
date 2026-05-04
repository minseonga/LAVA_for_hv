#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable

try:
    import numpy as np
except Exception:
    np = None


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
    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def image_id_from_filename(image_name: str) -> str:
    match = re.search(r"(\d+)(?=\.[^.]+$|$)", os.path.basename(str(image_name)))
    return str(int(match.group(1))) if match else ""


def normalize_model_base(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return os.path.expanduser(text) if text else None


def install_clearsight_vaf(model: Any, attn_adapter_cls: Any, start_layer: int, end_layer: int, enh_para: float, sup_para: float) -> List[int]:
    layers = list(getattr(getattr(model, "model", None), "layers", []))
    if not layers:
        raise RuntimeError("Could not resolve LLaVA-1.5 decoder layers for ClearSight VAF.")

    patched: List[int] = []
    for idx, layer in enumerate(layers):
        if not (int(start_layer) <= idx <= int(end_layer)):
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        adapter = attn_adapter_cls(attn.config, float(enh_para), float(sup_para))
        adapter.load_state_dict(attn.state_dict())
        ref_param = next(attn.parameters())
        adapter = adapter.to(device=ref_param.device, dtype=ref_param.dtype)
        layer.self_attn = adapter
        patched.append(idx)
    if not patched:
        raise RuntimeError(f"No LLaVA-1.5 attention layers were patched for ClearSight VAF: {start_layer}-{end_layer}.")
    return patched


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ClearSight/VAF on arbitrary LLaVA-1.5 image-question JSONL rows.")
    ap.add_argument("--clearsight-root", default=os.environ.get("CLEARSIGHT_ROOT", "ClearSight"))
    ap.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--use-visaug", type=parse_bool, default=True)
    ap.add_argument("--start-layer", type=int, default=9)
    ap.add_argument("--end-layer", type=int, default=14)
    ap.add_argument("--enh-para", type=float, default=1.15)
    ap.add_argument("--sup-para", type=float, default=0.95)
    ap.add_argument("--append-yesno", type=parse_bool, default=False)
    args = ap.parse_args()

    answers_file = os.path.abspath(os.path.expanduser(str(args.answers_file)))
    if os.path.exists(answers_file):
        raise FileExistsError(f"answers file already exists: {answers_file}")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    clearsight_root = Path(args.clearsight_root).expanduser()
    if not clearsight_root.is_absolute():
        clearsight_root = (repo_root / clearsight_root).resolve()
    if not clearsight_root.exists():
        raise FileNotFoundError(f"ClearSight repo not found: {clearsight_root}")

    for path in (clearsight_root / "LLaVA", clearsight_root / "visaug" / "inference"):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)

    from AttnAdapter import AttnAdapter  # type: ignore
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    setup_seed(int(args.seed))
    disable_torch_init()

    model_path = os.path.expanduser(str(args.model_path))
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        normalize_model_base(args.model_base),
        model_name,
    )
    model.eval()

    patched_layers: List[int] = []
    if bool(args.use_visaug):
        patched_layers = install_clearsight_vaf(
            model,
            AttnAdapter,
            start_layer=int(args.start_layer),
            end_layer=int(args.end_layer),
            enh_para=float(args.enh_para),
            sup_para=float(args.sup_para),
        )

    rows = read_jsonl(args.question_file, limit=int(args.limit))
    with open(answers_file, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="clearsight-llava15-vaf", unit="sample"):
            image_name = str(row.get("image", "")).strip()
            question = str(row.get("question", row.get("text", ""))).strip()
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_id = str(row.get("image_id", "")).strip() or image_id_from_filename(image_name)
            if not image_name or not question or not qid:
                continue

            model_question = question
            if bool(args.append_yesno):
                model_question = model_question.rstrip() + " Please just answer yes or no."
            cur_prompt = model_question
            if bool(getattr(model.config, "mm_use_im_start_end", False)):
                model_question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + model_question
            else:
                model_question = DEFAULT_IMAGE_TOKEN + "\n" + model_question

            conv = conv_templates[str(args.conv_mode)].copy()
            conv.append_message(conv.roles[0], model_question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).cuda()

            image = Image.open(os.path.join(str(args.image_folder), image_name)).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_token_len = int(input_ids.shape[1])
            decode_ids = output_ids[:, input_token_len:] if int(output_ids.shape[-1]) > input_token_len else output_ids
            output_text = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0].strip()
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
                        "image": image_name,
                        "image_id": image_id,
                        "label": row.get("label", ""),
                        "prompt": cur_prompt,
                        "model_id": model_name,
                        "method": "vaf",
                        "clearsight_layers": patched_layers,
                        "enh_para": float(args.enh_para),
                        "sup_para": float(args.sup_para),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
