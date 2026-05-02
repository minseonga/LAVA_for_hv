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

import torch

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable

try:
    import numpy as np
except Exception:
    np = None

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


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


def import_llama_attention_helpers():
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # type: ignore

    try:
        from transformers.models.llama.modeling_llama import repeat_kv  # type: ignore
    except Exception:
        def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            if int(n_rep) == 1:
                return hidden_states
            bsz, n_kv_heads, slen, head_dim = hidden_states.shape
            hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv_heads, int(n_rep), slen, head_dim)
            return hidden_states.reshape(bsz, n_kv_heads * int(n_rep), slen, head_dim)

    return apply_rotary_pos_emb, repeat_kv


def resolve_decoder_layers(model: Any) -> List[Any]:
    backbone = model.get_model() if hasattr(model, "get_model") else getattr(model, "model", None)
    layers = list(getattr(backbone, "layers", []))
    if not layers and hasattr(backbone, "model"):
        layers = list(getattr(backbone.model, "layers", []))
    if not layers:
        raise RuntimeError("Could not resolve LLaVA-NeXT decoder layers.")
    return layers


def make_llava_next_visual_attention_forward(original_forward: Any):
    import torch.nn as nn

    apply_rotary_pos_emb, repeat_kv = import_llama_attention_helpers()

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
        img_span = getattr(self, "_visual_attn_img_span", None)
        if img_span is None or not bool(getattr(self, "_visual_attn_enabled", False)):
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
        config = getattr(self, "config", None)
        num_heads_value = getattr(self, "num_heads", None)
        if num_heads_value is None:
            num_heads_value = getattr(config, "num_attention_heads", None)
        if num_heads_value is None:
            raise RuntimeError("Could not resolve LLaMA attention head count.")
        num_heads = int(num_heads_value)
        num_key_value_heads = int(getattr(self, "num_key_value_heads", num_heads))
        num_key_value_groups = int(getattr(self, "num_key_value_groups", num_heads // num_key_value_heads))
        head_dim = int(getattr(self, "head_dim", getattr(self, "hidden_size", hidden_states.shape[-1]) // num_heads))

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        cos = sin = None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            try:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            except TypeError:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        else:
            kv_seq_len = int(key_states.shape[-2])
            if past_key_value is not None:
                if hasattr(past_key_value, "get_usable_length"):
                    kv_seq_len += int(past_key_value.get_usable_length(kv_seq_len, getattr(self, "layer_idx", None)))
                elif isinstance(past_key_value, tuple) and past_key_value:
                    kv_seq_len += int(past_key_value[0].shape[-2])
            try:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            except TypeError:
                cos, sin = self.rotary_emb(value_states, position_ids)
            try:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            except TypeError:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        next_past_key_value = None
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs: Dict[str, Any] = {}
                if cos is not None and sin is not None:
                    cache_kwargs.update({"sin": sin, "cos": cos})
                if cache_position is not None:
                    cache_kwargs["cache_position"] = cache_position
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    getattr(self, "layer_idx", None),
                    cache_kwargs,
                )
                next_past_key_value = past_key_value if bool(use_cache) else None
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                next_past_key_value = (key_states, value_states) if bool(use_cache) else None
        elif bool(use_cache):
            next_past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            if int(attention_mask.dim()) == 4:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            else:
                causal_mask = attention_mask[:, None, None, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        img_start, img_end = int(img_span[0]), int(img_span[1])
        key_len = int(attn_weights.shape[-1])
        img_start = max(0, min(img_start, key_len))
        img_end = max(img_start, min(img_end, key_len))
        mode = str(getattr(self, "_visual_attn_mode", "vaf"))
        if img_end > img_start and mode == "pai_attn":
            alpha = float(getattr(self, "_pai_alpha", 0.2))
            attn_weights[:, :, -1:, img_start:img_end] = (
                attn_weights[:, :, -1:, img_start:img_end].abs() * alpha
                + attn_weights[:, :, -1:, img_start:img_end]
            )
        elif img_end > img_start and mode == "vaf":
            enh_para = float(getattr(self, "_vaf_enh_para", 1.15))
            sup_para = float(getattr(self, "_vaf_sup_para", 0.95))
            query_slice = slice(img_end, None) if int(q_len) > img_end else slice(None)
            attn_weights[:, :, query_slice, img_start:img_end] = (
                enh_para * attn_weights[:, :, query_slice, img_start:img_end]
            )
            if img_start > 0:
                attn_weights[:, :, query_slice, :img_start] = (
                    sup_para * attn_weights[:, :, query_slice, :img_start]
                )
        elif mode not in {"pai_attn", "vaf"}:
            raise ValueError(f"Unsupported visual attention mode: {mode!r}")

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        dropout_p = float(getattr(self, "attention_dropout", 0.0))
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        expected = (bsz, num_heads, q_len, head_dim)
        if attn_output.size() != expected:
            raise ValueError(f"`attn_output` should be of size {expected}, but is {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, next_past_key_value

    return forward


def install_visual_attention_patch(
    model: Any,
    *,
    start_layer: int,
    end_layer: int,
    mode: str,
    enh_para: float,
    sup_para: float,
    pai_alpha: float,
) -> List[int]:
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
        attn._visual_attn_original_forward = original_forward
        attn._visual_attn_enabled = True
        attn._visual_attn_mode = str(mode)
        attn._visual_attn_img_span = None
        attn._vaf_enh_para = float(enh_para)
        attn._vaf_sup_para = float(sup_para)
        attn._pai_alpha = float(pai_alpha)
        attn.forward = types.MethodType(make_llava_next_visual_attention_forward(original_forward), attn)
        patched.append(idx)
    if not patched:
        raise RuntimeError(f"No LLaVA-NeXT decoder attention layers were patched for mode={mode!r}.")
    return patched


def set_visual_image_span(runtime: Any, input_ids: torch.Tensor, image_tensor: Any, image_sizes: Any) -> Tuple[int, int]:
    from llava.constants import IGNORE_INDEX

    with torch.no_grad():
        _, _, _, labels_e = runtime._prepare_multimodal_expanded_sequence(
            full_ids=input_ids,
            images_tensor=image_tensor,
            image_sizes=image_sizes,
        )
    labels = labels_e[0]
    vision_positions = torch.where(labels == int(IGNORE_INDEX))[0]
    if int(vision_positions.numel()) <= 0:
        raise RuntimeError("No visual token span found in LLaVA-NeXT expanded sequence.")
    img_start = int(vision_positions.min().item())
    img_end = int(vision_positions.max().item()) + 1
    for layer in resolve_decoder_layers(runtime.model):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and bool(getattr(attn, "_visual_attn_enabled", False)):
            attn._visual_attn_img_span = (img_start, img_end)
    return img_start, img_end


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PAI-attn or VAF-style visual attention intervention for LLaVA-NeXT.")
    ap.add_argument("--llava-next-root", default=os.environ.get("LLAVA_NEXT_ROOT", "/home/kms/LLaVA-NeXT"))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--conv-mode", default="llava_llama_3")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--torch-type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--attn-implementation", default="eager", choices=["none", "flash_attention_2", "sdpa", "eager"])
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

    from frgavr_cleanroom.llava_next_runtime import OfficialLlavaNextRuntime

    runtime = OfficialLlavaNextRuntime(
        llava_next_root=str(args.llava_next_root),
        model_path=str(args.model_path),
        model_base=(None if not str(args.model_base or "").strip() else str(args.model_base)),
        conv_mode=str(args.conv_mode),
        device="cuda",
        torch_type=str(args.torch_type),
        attn_implementation=str(args.attn_implementation),
    )
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    patched_layers = install_visual_attention_patch(
        runtime.model,
        start_layer=int(args.start_layer),
        end_layer=int(args.end_layer),
        mode=str(args.mode),
        enh_para=float(args.enh_para),
        sup_para=float(args.sup_para),
        pai_alpha=float(args.pai_alpha),
    )

    rows = read_jsonl(args.question_file, limit=int(args.limit))
    with open(answers_file, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"llava-next-{args.mode}", unit="sample"):
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            question = str(row.get("question", row.get("text", ""))).strip()
            image_name = str(row.get("image", "")).strip()
            image_id = str(row.get("image_id", "")).strip()
            if not qid or not question or not image_name:
                continue

            prompt = runtime.prompt_text(question)
            input_ids = tokenizer_image_token(
                prompt,
                runtime.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(runtime.device)
            image = runtime.load_image(os.path.join(args.image_folder, image_name))
            image_tensor, image_sizes = runtime._process_image(image)
            img_start, img_end = set_visual_image_span(runtime, input_ids, image_tensor, image_sizes)

            with torch.inference_mode():
                output_ids = runtime.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    modalities=["image"] * int(input_ids.shape[0]),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(args.max_new_tokens),
                    use_cache=True,
                    pad_token_id=runtime.tokenizer.eos_token_id,
                )

            decode_ids = output_ids
            if int(output_ids.shape[-1]) > int(input_ids.shape[-1]):
                decode_ids = output_ids[:, int(input_ids.shape[-1]) :]
            output_text = runtime.tokenizer.batch_decode(
                decode_ids,
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
                        "label": row.get("label", ""),
                        "prompt": prompt,
                        "model_id": os.path.basename(str(args.model_path).rstrip("/")),
                        "image": image_name,
                        "image_id": image_id,
                        "method": str(args.mode),
                        "visual_attn_layers": patched_layers,
                        "visual_attn_img_start": img_start,
                        "visual_attn_img_end": img_end,
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
