#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def patch_legacy_transformers_bloom_masks() -> None:
    try:
        import transformers.models.bloom.modeling_bloom as bloom
    except Exception:
        return

    if not hasattr(bloom, "_expand_mask"):

        def _expand_mask(mask: torch.Tensor, tgt_length: Optional[int] = None) -> torch.BoolTensor:
            batch_size, src_length = mask.shape
            tgt_length = int(tgt_length) if tgt_length is not None else int(src_length)
            expanded_mask = ~mask[:, None, None, :].to(torch.bool)
            return expanded_mask.expand(batch_size, 1, tgt_length, src_length)

        bloom._expand_mask = _expand_mask  # type: ignore[attr-defined]

    if not hasattr(bloom, "_make_causal_mask"):

        def _make_causal_mask(
            input_ids_shape: Any,
            device: torch.device,
            past_key_values_length: int = 0,
        ) -> torch.BoolTensor:
            batch_size, tgt_length = int(input_ids_shape[0]), int(input_ids_shape[1])
            src_length = tgt_length + int(past_key_values_length)
            mask = torch.zeros((tgt_length, src_length), dtype=torch.bool, device=device)
            mask[:, int(past_key_values_length) :] = torch.triu(
                torch.ones((tgt_length, tgt_length), dtype=torch.bool, device=device),
                diagonal=1,
            )
            return mask[None, None, :, :].expand(batch_size, 1, tgt_length, src_length)

        bloom._make_causal_mask = _make_causal_mask  # type: ignore[attr-defined]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for VGA_origin/eval/object_hallucination_vqa_llava.py. "
            "It preserves the original runner while redirecting the hard-coded "
            "path/to/llava-v1.5-7b tokenizer placeholder to --model-path."
        )
    )
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--image-folder", type=str, default="")
    ap.add_argument("--question-file", type=str, default="tables/question.jsonl")
    ap.add_argument("--answers-file", type=str, default="answer.jsonl")
    ap.add_argument("--conv-mode", type=str, default="llava_v1")
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--chunk-idx", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_gen_len", type=int, default=512)
    ap.add_argument("--use_add", type=parse_bool, default=False)
    ap.add_argument("--cd_alpha", type=float, default=1)
    ap.add_argument("--attn_coef", type=float, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_layer", type=int, default=99)
    ap.add_argument("--end_layer", type=int, default=0)
    ap.add_argument(
        "--head_balancing",
        type=str,
        default="attn",
        choices=["vattn", "battn", "simg", "simv", "simb", "simb-simg", "none"],
    )
    ap.add_argument("--attn_norm", type=parse_bool, default=False)
    ap.add_argument("--sampling", type=parse_bool, default=False)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    vga_root = Path(args.vga_root)
    if not vga_root.is_absolute():
        vga_root = (repo_root / vga_root).resolve()

    for path in (str(vga_root), str(vga_root / "eval")):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    patch_legacy_transformers_bloom_masks()

    import transformers
    from transformers import set_seed

    original_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def patched_from_pretrained(pretrained_model_name_or_path: Any, *a: Any, **kw: Any) -> Any:
        if str(pretrained_model_name_or_path) == "path/to/llava-v1.5-7b":
            return original_from_pretrained(args.model_path, *a, **kw)
        return original_from_pretrained(pretrained_model_name_or_path, *a, **kw)

    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        origin_path = vga_root / "eval" / "object_hallucination_vqa_llava.py"
        spec = importlib.util.spec_from_file_location("vga_origin_object_hallucination_vqa_llava", origin_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load VGA origin runner from {origin_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        transformers.AutoTokenizer.from_pretrained = original_from_pretrained

    set_seed(int(args.seed))
    print(str(args), flush=True)
    module.eval_model(args)


if __name__ == "__main__":
    main()
