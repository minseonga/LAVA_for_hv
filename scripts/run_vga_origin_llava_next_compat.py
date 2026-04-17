#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def patch_transformers_compat() -> None:
    import transformers.modeling_utils as modeling_utils
    import transformers.pytorch_utils as pytorch_utils
    from transformers import CLIPVisionModel

    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))

    if not getattr(CLIPVisionModel, "_llava_next_safetensors_patch", False):
        original_from_pretrained = CLIPVisionModel.from_pretrained

        def from_pretrained_with_safetensors(cls: Any, pretrained_model_name_or_path: Any, *a: Any, **kw: Any) -> Any:
            kw.setdefault("use_safetensors", True)
            return original_from_pretrained(pretrained_model_name_or_path, *a, **kw)

        CLIPVisionModel.from_pretrained = classmethod(from_pretrained_with_safetensors)
        CLIPVisionModel._llava_next_safetensors_patch = True  # type: ignore[attr-defined]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for VGA_origin/eval/object_hallucination_vqa_llava-next.py. "
            "It patches small transformers API moves and redirects the original "
            "hard-coded tokenizer placeholder to --model-path."
        )
    )
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--image-folder", type=str, default="")
    ap.add_argument("--question-file", type=str, default="tables/question.jsonl")
    ap.add_argument("--answers-file", type=str, default="answer.jsonl")
    ap.add_argument("--conv-mode", type=str, default="llava_llama_3")
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--chunk-idx", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_gen_len", type=int, default=512)
    ap.add_argument("--use_add", type=parse_bool, default=False)
    ap.add_argument("--cd_alpha", type=float, default=1)
    ap.add_argument("--attn_coef", type=float, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--torch_type", type=str, default="bf16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--attn_type", type=str, default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--head_balancing", type=str, default="attn", choices=["attn", "simg", "simv", "simb", "none"])
    ap.add_argument("--start_layer", type=int, default=99)
    ap.add_argument("--end_layer", type=int, default=0)
    ap.add_argument("--attn_norm", type=parse_bool, default=False)
    ap.add_argument("--sampling", type=parse_bool, default=False)
    args = ap.parse_args()
    if not str(args.model_base or "").strip():
        args.model_base = None

    repo_root = Path(__file__).resolve().parents[1]
    vga_root = Path(args.vga_root)
    if not vga_root.is_absolute():
        vga_root = (repo_root / vga_root).resolve()
    for path in (str(vga_root), str(vga_root / "eval")):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    patch_transformers_compat()

    import transformers
    from transformers import set_seed

    original_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def patched_from_pretrained(pretrained_model_name_or_path: Any, *a: Any, **kw: Any) -> Any:
        if str(pretrained_model_name_or_path) == "/path/to/llama3-llava-next-8b":
            return original_from_pretrained(args.model_path, *a, **kw)
        return original_from_pretrained(pretrained_model_name_or_path, *a, **kw)

    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        origin_path = vga_root / "eval" / "object_hallucination_vqa_llava-next.py"
        spec = importlib.util.spec_from_file_location("vga_origin_object_hallucination_vqa_llava_next", origin_path)
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
