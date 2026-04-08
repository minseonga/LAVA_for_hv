#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "llava-next": "eval_configs/llava-next_eval.yaml",
}

POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "llava-next": "USER: <ImageHere> <question> ASSISTANT:",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run official-like EAZY POPE one-pass and dump raw outputs.")
    ap.add_argument("--eazy_root", type=str, required=True)
    ap.add_argument("--runtime_shim_root", type=str, default="")
    ap.add_argument("--nltk_data_dir", type=str, default="")
    ap.add_argument("--model", type=str, default="llava-1.5")
    ap.add_argument("--pope_type", type=str, required=True, choices=["random", "popular", "adversarial"])
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--save_jsonl", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def setup_imports(eazy_root: str, runtime_shim_root: str, nltk_data_dir: str) -> None:
    if runtime_shim_root:
        runtime_shim_root = os.path.abspath(runtime_shim_root)
        if os.path.isdir(runtime_shim_root) and runtime_shim_root not in sys.path:
            sys.path.insert(0, runtime_shim_root)
    local_transformers_src = os.path.join(eazy_root, "transformers-4.29.2", "src")
    if os.path.isdir(local_transformers_src) and local_transformers_src not in sys.path:
        sys.path.insert(0, local_transformers_src)
    if eazy_root not in sys.path:
        sys.path.insert(0, eazy_root)
    if nltk_data_dir:
        prev = os.environ.get("NLTK_DATA", "").strip()
        os.environ["NLTK_DATA"] = nltk_data_dir if not prev else f"{nltk_data_dir}:{prev}"


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main() -> None:
    args = parse_args()
    eazy_root = os.path.abspath(args.eazy_root)
    os.chdir(eazy_root)
    setup_imports(eazy_root, args.runtime_shim_root, args.nltk_data_dir)

    from minigpt4.common.config import Config  # type: ignore
    from minigpt4.common.registry import registry  # type: ignore
    from minigpt4.models import load_preprocess  # type: ignore
    from utils.brute_force_zero_out import get_zero_out_list_from_object  # type: ignore
    from utils.chair_detector import CHAIR_detector  # type: ignore
    from Projects.LVLM_hallucination.pope_loader import POPEDataSet  # type: ignore

    for module_name in [
        "minigpt4.datasets.builders",
        "minigpt4.models",
        "minigpt4.processors",
        "minigpt4.runners",
        "minigpt4.tasks",
    ]:
        __import__(module_name)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    cfg_ns = argparse.Namespace(
        model=args.model,
        pope_type=args.pope_type,
        cfg_path=MODEL_EVAL_CONFIG_PATH[args.model],
        pope_path=POPE_PATH[args.pope_type],
        gpu_id=args.gpu_id,
        options=None,
        data_path=args.data_path,
        zero_out_path=None,
        control_id_type="function",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sink_control=False,
        image_sink_control=False,
        start_layer=25,
        end_layer=32,
        opera=False,
        beam=args.beam,
        k=args.k,
        sample=bool(args.sample),
        scale_factor=50.0,
        threshold=15,
        num_attn_candidates=5,
        penalty_weights=1.0,
        zero_out_counter=0,
    )
    cfg = Config(cfg_ns)
    setup_seeds(int(args.seed))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu_id))
    print(
        "[gpu]",
        json.dumps(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_id_arg": int(args.gpu_id),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1,
            }
        ),
    )
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()
    vis_processors, _ = load_preprocess(cfg.get_config().preprocess)
    chair_detector = CHAIR_detector()
    tokenizer = model.llama_tokenizer

    pope_rel = POPE_PATH[args.pope_type]
    pope_abs = os.path.join(eazy_root, pope_rel)
    raw_rows = read_jsonl(pope_abs)
    dataset = POPEDataSet(
        pope_path=pope_abs,
        data_path=args.data_path,
        trans=vis_processors["eval"],
        zero_out_path=None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    template = INSTRUCTION_TEMPLATE[args.model]
    out_path = os.path.abspath(args.save_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise SystemExit(f"Output already exists: {out_path}")

    write_rows: List[Dict[str, Any]] = []
    row_offset = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for data in tqdm(loader, total=len(loader), desc=f"eazy-official:{args.pope_type}", unit="batch"):
            image = data["image"].to(device)
            queries = data["query"]
            labels = data["label"]
            qu = [template.replace("<question>", q) for q in queries]
            describ_qu = [template.replace("<question>", "Please describe this image in detail.") for _ in qu]

            with torch.inference_mode():
                with torch.no_grad():
                    if args.model == "shikra":
                        initial_response, att, key_pos = model.generate(
                            {"image": image, "prompt": describ_qu},
                            use_nucleus_sampling=bool(args.sample),
                            do_sample=False,
                            num_beams=int(args.beam),
                            max_new_tokens=512,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=None,
                        )
                    else:
                        initial_response, att, key_pos, output_ids, input_ids = model.generate(
                            {"image": image, "prompt": describ_qu},
                            use_nucleus_sampling=bool(args.sample),
                            do_sample=False,
                            num_beams=int(args.beam),
                            max_new_tokens=512,
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
                k=int(args.k),
                model_name=args.model,
            )
            one_pass_zero_out_list = []
            for zero_out_list in obj2zero_out.values():
                one_pass_zero_out_list.extend(zero_out_list)
            one_pass_zero_out_list = list(set(one_pass_zero_out_list))

            with torch.inference_mode():
                with torch.no_grad():
                    if args.model == "shikra":
                        response_, _, _ = model.generate(
                            {"image": image, "prompt": describ_qu},
                            use_nucleus_sampling=bool(args.sample),
                            do_sample=False,
                            num_beams=int(args.beam),
                            max_new_tokens=512,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=one_pass_zero_out_list,
                        )
                    else:
                        response_, _, _, _, _ = model.generate(
                            {"image": image, "prompt": describ_qu},
                            use_nucleus_sampling=bool(args.sample),
                            do_sample=False,
                            num_beams=int(args.beam),
                            max_new_tokens=512,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=one_pass_zero_out_list,
                        )

            zero_out_list_final: List[int] = []
            for obj_name, zero_out_list in obj2zero_out.items():
                if obj_name not in response_[0]:
                    zero_out_list_final.extend(zero_out_list)
            zero_out_list_final = list(set(zero_out_list_final))

            with torch.inference_mode():
                with torch.no_grad():
                    if args.model == "shikra":
                        out, att, key_pos = model.generate(
                            {"image": image, "prompt": qu},
                            use_nucleus_sampling=bool(args.sample),
                            do_sample=False,
                            num_beams=int(args.beam),
                            max_new_tokens=512,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=zero_out_list_final,
                        )
                    else:
                        out, att, key_pos, output_ids, input_ids = model.generate(
                            {"image": image, "prompt": qu},
                            use_nucleus_sampling=bool(args.sample),
                            do_sample=False,
                            num_beams=int(args.beam),
                            max_new_tokens=512,
                            output_attentions=True,
                            opera_decoding=False,
                            return_dict_in_generate=True,
                            zero_out_list=zero_out_list_final,
                        )

            batch_size = len(out)
            batch_refs = raw_rows[row_offset : row_offset + batch_size]
            for j, text in enumerate(out):
                ref = batch_refs[j] if j < len(batch_refs) else {}
                record = {
                    "question_id": str(row_offset + j),
                    "question": str(ref.get("text", "")),
                    "output": str(text),
                    "text": str(text),
                    "image": str(ref.get("image", "")),
                    "label": ref.get("label", ""),
                    "model": args.model,
                    "pope_type": args.pope_type,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                write_rows.append(record)
            f.flush()
            row_offset += batch_size

    summary_path = os.path.splitext(out_path)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "eazy_root": eazy_root,
                    "pope_type": args.pope_type,
                    "data_path": os.path.abspath(args.data_path),
                    "pope_path": pope_abs,
                    "batch_size": int(args.batch_size),
                    "num_workers": int(args.num_workers),
                    "beam": int(args.beam),
                    "k": int(args.k),
                    "sample": bool(args.sample),
                },
                "counts": {
                    "n_ref_rows": len(raw_rows),
                    "n_dump_rows": len(write_rows),
                },
                "outputs": {
                    "pred_jsonl": out_path,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[saved]", out_path)
    print("[saved]", summary_path)


if __name__ == "__main__":
    main()
