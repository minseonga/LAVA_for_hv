#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

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
    ap = argparse.ArgumentParser(description="Run VISTA on an arbitrary question subset jsonl.")
    ap.add_argument("--vista_root", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--answers_file", type=str, required=True)
    ap.add_argument("--model", type=str, default="llava-1.5")
    ap.add_argument("--vsv", action="store_true")
    ap.add_argument("--vsv_lambda", type=float, default=0.01)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--logits_aug", action="store_true")
    ap.add_argument("--logits_layers", type=str, default="25,30")
    ap.add_argument("--logits_alpha", type=float, default=0.3)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=None)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1994)
    args = ap.parse_args()

    vista_root = os.path.abspath(args.vista_root)
    os.chdir(vista_root)
    if vista_root not in sys.path:
        sys.path.insert(0, vista_root)

    import myutils  # type: ignore
    from llava.utils import disable_torch_init  # type: ignore
    from llm_layers import add_vsv_layers, remove_vsv_layers  # type: ignore
    from model_loader import ModelLoader  # type: ignore
    from steering_vector import add_logits_flag, obtain_vsv, remove_logits_flag  # type: ignore

    myutils.seed_everything(args.seed)
    disable_torch_init()
    template = myutils.prepare_template(args)
    model_loader = ModelLoader(args.model)

    out_path = os.path.abspath(args.answers_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        raise SystemExit(f"Output already exists: {out_path}")

    rows = read_jsonl(os.path.abspath(args.question_file))
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="vista-subset", unit="sample"):
            image_name = str(row.get("image", "")).strip()
            query = str(row.get("question", row.get("text", ""))).strip()
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not image_name or not query or not qid:
                continue

            raw_image = Image.open(os.path.join(args.image_folder, image_name)).convert("RGB")
            image = model_loader.image_processor(raw_image)

            with myutils.maybe_autocast(args.model, model_loader.vlm_model.device):
                questions, kwargs = model_loader.prepare_inputs_for_model(template, [query], image)

                if args.vsv:
                    neg_kwargs = model_loader.prepare_neg_prompt(args, questions, template=template)
                    pos_kwargs = model_loader.prepare_pos_prompt(args, kwargs)
                    visual_vector, _ = obtain_vsv(args, model_loader.llm_model, [[neg_kwargs, pos_kwargs]])
                    add_vsv_layers(model_loader.llm_model, visual_vector.unsqueeze(0).unsqueeze(1).cuda(), [args.vsv_lambda], args.layers)

                add_logits_flag(model_loader.llm_model, args)
                if args.do_sample:
                    kwargs["top_p"] = args.top_p
                    kwargs["top_k"] = args.top_k

                outputs = model_loader.llm_model.generate(
                    do_sample=args.do_sample,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    num_beams=args.num_beams,
                    output_attentions=False,
                    output_hidden_states=True if args.logits_aug else False,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    return_dict=True,
                    **kwargs,
                )
                remove_logits_flag(model_loader.llm_model)
                if args.vsv:
                    remove_vsv_layers(model_loader.llm_model)

            output_text = model_loader.decode(outputs)[0]
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
