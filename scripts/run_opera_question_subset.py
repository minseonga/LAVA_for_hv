#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
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


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run OPERA on an arbitrary question subset jsonl.")
    ap.add_argument("--opera_root", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--answers_file", type=str, required=True)
    ap.add_argument("--model", type=str, default="llava-1.5")
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--scale_factor", type=float, default=50.0)
    ap.add_argument("--threshold", type=int, default=15)
    ap.add_argument("--num_attn_candidates", type=int, default=5)
    ap.add_argument("--penalty_weights", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    opera_root = os.path.abspath(args.opera_root)
    os.chdir(opera_root)
    local_transformers_src = os.path.join(opera_root, "transformers-4.29.2", "src")
    if os.path.isdir(local_transformers_src) and local_transformers_src not in sys.path:
        sys.path.insert(0, local_transformers_src)
    if opera_root not in sys.path:
        sys.path.insert(0, opera_root)

    from minigpt4.common.config import Config  # type: ignore
    from minigpt4.common.registry import registry  # type: ignore
    from minigpt4.models import load_preprocess  # type: ignore

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
    setup_seeds(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu_id))
        device = torch.device(f"cuda:{int(args.gpu_id)}")
    else:
        device = torch.device("cpu")

    model_config = cfg.model_cfg
    if str(args.model_path).strip():
        setattr(model_config, "merged_ckpt", os.path.expanduser(str(args.model_path)))
    setattr(model_config, "device_8bit", int(args.gpu_id))
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    vis_processors, _ = load_preprocess(cfg.get_config().preprocess)

    rows = read_jsonl(os.path.abspath(args.question_file))
    out_path = os.path.abspath(args.answers_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise SystemExit(f"Output already exists: {out_path}")

    template = INSTRUCTION_TEMPLATE[args.model]
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="opera-subset", unit="sample"):
            image_name = str(row.get("image", "")).strip()
            query = str(row.get("question", row.get("text", ""))).strip()
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not image_name or not query or not qid:
                continue

            raw_image = Image.open(os.path.join(args.image_folder, image_name)).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            prompt = template.replace("<question>", query)

            with torch.inference_mode():
                with torch.no_grad():
                    out = model.generate(
                        {"image": image, "prompt": [prompt]},
                        use_nucleus_sampling=bool(args.sample),
                        num_beams=int(args.beam),
                        max_new_tokens=int(args.max_new_tokens),
                        output_attentions=True,
                        opera_decoding=True,
                        scale_factor=float(args.scale_factor),
                        threshold=int(args.threshold),
                        num_attn_candidates=int(args.num_attn_candidates),
                        penalty_weights=float(args.penalty_weights),
                    )

            output_text = str(out[0] if isinstance(out, list) else out).strip()
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
