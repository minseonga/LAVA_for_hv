#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable

try:
    import numpy as np
except Exception:
    np = None

TORCH_DTYPE_CHOICES = ["bf16", "bfloat16", "float16", "float32", "fp16", "fp32"]


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
    import torch

    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_model(args: argparse.Namespace):
    import torch

    from qwen_vl_utils import process_vision_info  # type: ignore
    from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration  # type: ignore

    torch_dtype = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    model_path = os.path.expanduser(str(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch_dtype[str(args.torch_type)],
        device_map=str(args.device_map),
    ).eval()
    if not bool(args.do_sample) and getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        trust_remote_code=True,
    )
    return model, processor, tokenizer, process_vision_info


def import_qwen_attention_helpers():
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # type: ignore
            apply_multimodal_rotary_pos_emb,
            repeat_kv,
        )
        return apply_multimodal_rotary_pos_emb, repeat_kv
    except Exception:
        # Some local forks expose the same helpers under VGA_origin.
        from qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv  # type: ignore

        return apply_multimodal_rotary_pos_emb, repeat_kv


def _is_decoder_layer(module: Any) -> bool:
    attn = getattr(module, "self_attn", None)
    if attn is None:
        return False
    return all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))


def resolve_decoder_layers(model: Any) -> List[Any]:
    candidates = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("language_model.layers", getattr(getattr(model, "language_model", None), "layers", None)),
        (
            "model.language_model.layers",
            getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        ),
        ("transformer.layers", getattr(getattr(model, "transformer", None), "layers", None)),
    ]
    for _name, layers in candidates:
        if layers is None:
            continue
        layers_list = list(layers)
        if layers_list and all(_is_decoder_layer(layer) for layer in layers_list):
            return layers_list

    found: List[Any] = []
    seen: set[int] = set()
    for module in model.modules():
        if not _is_decoder_layer(module):
            continue
        ident = id(module)
        if ident in seen:
            continue
        seen.add(ident)
        found.append(module)
    if found:
        return found
    raise RuntimeError("Could not resolve Qwen2.5-VL decoder layers for visual attention patching.")


def make_visual_attention_forward(original_forward: Any):
    import torch
    import torch.nn as nn

    apply_multimodal_rotary_pos_emb, repeat_kv = import_qwen_attention_helpers()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ):
        img_idx = getattr(self, "_vaf_img_idx", None)
        if img_idx is None or not bool(getattr(self, "_vaf_enabled", False)):
            return original_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("Qwen2.5-VL VAF requires position_embeddings from the decoder layer.")
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling["mrope_section"],
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        img_start, img_end = int(img_idx[0]), int(img_idx[1])
        key_len = int(attn_weights.shape[-1])
        img_start = max(0, min(img_start, key_len))
        img_end = max(img_start, min(img_end, key_len))
        mode = str(getattr(self, "_vaf_mode", "vaf"))
        if mode == "pai_attn":
            alpha = float(getattr(self, "_pai_alpha", 0.2))
            attn_weights[:, :, -1:, img_start:img_end] = (
                attn_weights[:, :, -1:, img_start:img_end].abs() * alpha
                + attn_weights[:, :, -1:, img_start:img_end]
            )
        elif mode == "vaf":
            enh_para = float(getattr(self, "_vaf_enh_para", 1.15))
            sup_para = float(getattr(self, "_vaf_sup_para", 0.95))
            if img_end > img_start:
                if q_len > img_end:
                    query_slice = slice(img_end, None)
                else:
                    query_slice = slice(None)
                attn_weights[:, :, query_slice, img_start:img_end] = (
                    enh_para * attn_weights[:, :, query_slice, img_start:img_end]
                )
                if img_start > 0:
                    attn_weights[:, :, query_slice, :img_start] = (
                        sup_para * attn_weights[:, :, query_slice, :img_start]
                    )
        else:
            raise ValueError(f"Unsupported visual attention mode: {mode!r}")

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward


def install_visual_attention_patch(
    model: Any,
    start_layer: int,
    end_layer: int,
    *,
    mode: str,
    enh_para: float,
    sup_para: float,
    pai_alpha: float,
) -> List[int]:
    mode = str(mode)
    patched: List[int] = []
    for idx, layer in enumerate(resolve_decoder_layers(model)):
        if mode == "pai_attn":
            in_range = int(start_layer) <= idx < int(end_layer)
        else:
            in_range = int(start_layer) <= idx <= int(end_layer)
        if not in_range:
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        original_forward = attn.forward
        attn._vaf_original_forward = original_forward
        attn._vaf_enabled = True
        attn._vaf_mode = mode
        attn._vaf_enh_para = float(enh_para)
        attn._vaf_sup_para = float(sup_para)
        attn._pai_alpha = float(pai_alpha)
        attn._vaf_img_idx = None
        attn.forward = types.MethodType(make_visual_attention_forward(original_forward), attn)
        patched.append(idx)
    if not patched:
        raise RuntimeError(f"No Qwen2.5-VL decoder attention layers were patched for mode={mode!r}.")
    return patched


def set_vaf_image_span(model: Any, input_ids: Any) -> Tuple[int, int]:
    import torch

    image_token_id = int(model.config.image_token_id)
    image_token_indices = torch.where(input_ids[0] == image_token_id)[0]
    if int(image_token_indices.numel()) <= 0:
        raise ValueError("No image tokens found in Qwen2.5-VL input_ids.")
    start_pos = int(image_token_indices[0].item())
    end_pos = int(image_token_indices[-1].item()) + 1
    for layer in resolve_decoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and bool(getattr(attn, "_vaf_enabled", False)):
            attn._vaf_img_idx = (start_pos, end_pos)
    return start_pos, end_pos


def main() -> None:
    ap = argparse.ArgumentParser(description="Run visual-attention intervention for Qwen2.5-VL POPE inference.")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--image-folder", type=str, required=True)
    ap.add_argument("--question-file", type=str, required=True)
    ap.add_argument("--answers-file", type=str, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--torch-type", type=str, default="bf16", choices=TORCH_DTYPE_CHOICES)
    ap.add_argument("--device-map", type=str, default="cuda")
    ap.add_argument("--min-pixels", type=int, default=14 * 14 * 1280)
    ap.add_argument("--max-pixels", type=int, default=28 * 28 * 1280)
    ap.add_argument("--do-sample", type=parse_bool, default=False)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--start-layer", type=int, default=9)
    ap.add_argument("--end-layer", type=int, default=14)
    ap.add_argument("--mode", type=str, default="vaf", choices=["vaf", "pai_attn"])
    ap.add_argument("--enh-para", type=float, default=1.15)
    ap.add_argument("--sup-para", type=float, default=0.95)
    ap.add_argument("--pai-alpha", type=float, default=0.2)
    args = ap.parse_args()

    answers_file = os.path.abspath(os.path.expanduser(str(args.answers_file)))
    if os.path.exists(answers_file):
        raise FileExistsError(f"answers file already exists: {answers_file}")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    setup_seed(int(args.seed))
    model, processor, _tokenizer, process_vision_info = load_model(args)
    patched_layers = install_visual_attention_patch(
        model,
        start_layer=int(args.start_layer),
        end_layer=int(args.end_layer),
        mode=str(args.mode),
        enh_para=float(args.enh_para),
        sup_para=float(args.sup_para),
        pai_alpha=float(args.pai_alpha),
    )
    rows = read_jsonl(args.question_file, limit=int(args.limit))

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": 1,
        "do_sample": bool(args.do_sample),
        "use_cache": True,
    }
    if bool(args.do_sample):
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["top_p"] = float(args.top_p)

    with open(answers_file, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"qwen25-vl-{args.mode}", unit="sample"):
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            question = str(row.get("question", row.get("text", ""))).strip()
            image_name = str(row.get("image", "")).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not qid or not question or not image_name:
                continue

            image_path = os.path.abspath(os.path.join(args.image_folder, image_name))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file://" + image_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            messages_batch = [messages]
            text = processor.apply_chat_template(
                messages_batch,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages_batch)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(model.device)
            img_start, img_end = set_vaf_image_span(model, inputs.input_ids)

            import torch

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

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
                        "prompt": text[0] if isinstance(text, list) else str(text),
                        "model_id": os.path.basename(str(args.model_path).rstrip("/")),
                        "method": str(args.mode),
                        "vaf_layers": patched_layers,
                        "vaf_img_start": img_start,
                        "vaf_img_end": img_end,
                        "vaf_enh_para": float(args.enh_para),
                        "vaf_sup_para": float(args.sup_para),
                        "pai_alpha": float(args.pai_alpha),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
