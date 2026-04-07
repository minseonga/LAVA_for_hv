#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from itertools import chain
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


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "llava-next": "eval_configs/llava-next_eval.yaml",
    "llava-1.5_13b": "eval_configs/llava-1.5_13b_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "llava-1.5_13b": "USER: <ImageHere> <question> ASSISTANT:",
    "llava-next": "USER: <ImageHere> <question> ASSISTANT:",
}


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


def parse_layers(att_layer: str) -> List[int]:
    s = str(att_layer or "").strip()
    if not s:
        return [i for i in range(14, 25)]
    if s.isdigit():
        return [int(s)]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run EAZY one-pass intervention on an arbitrary question subset jsonl.")
    ap.add_argument("--eazy_root", type=str, required=True)
    ap.add_argument("--runtime_shim_root", type=str, default="", help="Optional runtime shim root prepared by prepare_eazy_origin_runtime.sh")
    ap.add_argument("--nltk_data_dir", type=str, default="", help="Optional NLTK data directory for EAZY runtime")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--answers_file", type=str, required=True)
    ap.add_argument("--model", type=str, default="llava-1.5")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--att_layer", type=str, default="14,15,16,17,18,19,20,21,22,23,24")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1994)
    args = ap.parse_args()

    eazy_root = os.path.abspath(args.eazy_root)
    os.chdir(eazy_root)
    runtime_shim_root = os.path.abspath(args.runtime_shim_root) if str(args.runtime_shim_root).strip() else ""
    nltk_data_dir = os.path.abspath(args.nltk_data_dir) if str(args.nltk_data_dir).strip() else ""
    if runtime_shim_root and os.path.isdir(runtime_shim_root) and runtime_shim_root not in sys.path:
        sys.path.insert(0, runtime_shim_root)
    if nltk_data_dir:
        prev = os.environ.get("NLTK_DATA", "").strip()
        os.environ["NLTK_DATA"] = nltk_data_dir if not prev else f"{nltk_data_dir}:{prev}"
    local_transformers_src = os.path.join(eazy_root, "transformers-4.29.2", "src")
    if os.path.isdir(local_transformers_src) and local_transformers_src not in sys.path:
        sys.path.insert(0, local_transformers_src)
    if eazy_root not in sys.path:
        sys.path.insert(0, eazy_root)

    from minigpt4.common.config import Config  # type: ignore
    from minigpt4.common.registry import registry  # type: ignore
    from minigpt4.models import load_preprocess  # type: ignore
    from utils.brute_force_zero_out import get_zero_out_list_from_object  # type: ignore
    from utils.chair_detector import CHAIR_detector  # type: ignore

    for module_name in [
        "minigpt4.datasets.builders",
        "minigpt4.models",
        "minigpt4.processors",
        "minigpt4.runners",
        "minigpt4.tasks",
    ]:
        importlib.import_module(module_name)

    cfg_ns = argparse.Namespace(
        model=args.model,
        gpu_id=args.gpu_id,
        cfg_path=MODEL_EVAL_CONFIG_PATH[args.model],
        options=None,
    )
    cfg = Config(cfg_ns)
    setup_seeds(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu_id))
        device = torch.device(f"cuda:{int(args.gpu_id)}")
    else:
        device = torch.device("cpu")

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    vis_processors, _ = load_preprocess(cfg.get_config().preprocess)
    chair_detector = CHAIR_detector()
    tokenizer = model.llama_tokenizer
    layer_idx = parse_layers(args.att_layer)

    out_path = os.path.abspath(args.answers_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise SystemExit(f"Output already exists: {out_path}")

    rows = read_jsonl(os.path.abspath(args.question_file))
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="eazy-subset", unit="sample"):
            image_name = str(row.get("image", "")).strip()
            query = str(row.get("question", row.get("text", ""))).strip()
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not image_name or not query or not qid:
                continue

            image_path = os.path.join(args.image_folder, image_name)
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            prompt = INSTRUCTION_TEMPLATE[args.model].replace("<question>", query)

            with torch.inference_mode():
                with torch.no_grad():
                    if args.model == "shikra":
                        initial_response, att, key_pos = model.generate(
                            {"image": image, "prompt": prompt},
                            use_nucleus_sampling=args.sample,
                            do_sample=False,
                            num_beams=args.beam,
                            max_new_tokens=args.max_new_tokens,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=None,
                        )
                    else:
                        initial_response, att, key_pos, output_ids, input_ids = model.generate(
                            {"image": image, "prompt": prompt},
                            use_nucleus_sampling=args.sample,
                            do_sample=False,
                            num_beams=args.beam,
                            max_new_tokens=args.max_new_tokens,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=None,
                        )

            _, object_list, _, _ = chair_detector.caption_to_words(initial_response[0])
            object_list = list(set(object_list))
            obj2zero_out = get_zero_out_list_from_object(
                object_list,
                initial_response,
                tokenizer,
                att,
                k=args.k,
                model_name=args.model,
                object_layer=layer_idx,
            )

            one_pass_zero_out_list = list(chain(*obj2zero_out.values()))
            with torch.inference_mode():
                with torch.no_grad():
                    if args.model == "shikra":
                        response_, _, _ = model.generate(
                            {"image": image, "prompt": prompt},
                            use_nucleus_sampling=args.sample,
                            do_sample=False,
                            num_beams=args.beam,
                            max_new_tokens=args.max_new_tokens,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=one_pass_zero_out_list,
                        )
                    else:
                        response_, _, _, _, _ = model.generate(
                            {"image": image, "prompt": prompt},
                            use_nucleus_sampling=args.sample,
                            do_sample=False,
                            num_beams=args.beam,
                            max_new_tokens=args.max_new_tokens,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=one_pass_zero_out_list,
                        )

            zero_out_list_final: List[int] = []
            for obj_name, zero_list in obj2zero_out.items():
                if obj_name not in response_[0]:
                    zero_out_list_final.extend(zero_list)
            zero_out_list_final = list(set(zero_out_list_final))

            with torch.inference_mode():
                with torch.no_grad():
                    if args.model == "shikra":
                        out, att, key_pos = model.generate(
                            {"image": image, "prompt": prompt},
                            use_nucleus_sampling=args.sample,
                            do_sample=False,
                            num_beams=args.beam,
                            max_new_tokens=args.max_new_tokens,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=zero_out_list_final,
                        )
                    else:
                        out, att, key_pos, output_ids, input_ids = model.generate(
                            {"image": image, "prompt": prompt},
                            use_nucleus_sampling=args.sample,
                            do_sample=False,
                            num_beams=args.beam,
                            max_new_tokens=args.max_new_tokens,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=zero_out_list_final,
                        )

            output_text = out[0] if isinstance(out, list) else str(out)
            f.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": query,
                        "output": output_text,
                        "caption": output_text,
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
