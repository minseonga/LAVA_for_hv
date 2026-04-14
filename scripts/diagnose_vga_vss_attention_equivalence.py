#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    y = x.detach().float().reshape(-1)
    if y.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
    }


def safe_rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.detach().float()
    bf = b.detach().float()
    return float(torch.linalg.vector_norm(af - bf).item() / (torch.linalg.vector_norm(bf).item() + eps))


def safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    return float(torch.dot(af, bf).item() / (denom.item() + eps))


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    kk = min(int(k), int(a.numel()), int(b.numel()))
    if kk <= 0:
        return 0.0
    ai = set(torch.topk(a.detach().float().reshape(-1), kk).indices.cpu().tolist())
    bi = set(torch.topk(b.detach().float().reshape(-1), kk).indices.cpu().tolist())
    return float(len(ai & bi) / float(kk))


def vss_compare(vis_logits: torch.Tensor, k: int) -> Dict[str, Any]:
    top_k_scores, _ = torch.topk(vis_logits, int(k), dim=-1)
    p = top_k_scores.float().clamp_min(1e-12)
    denom = math.log(float(k))

    vss_code = (-p * torch.log(p)).sum(dim=-1) / denom
    # Literal form from the paper text as written: no probability weighting.
    # Dividing by K as an "average NLL" would not change sum-normalized guidance.
    vss_paper_unweighted = (-torch.log(p)).sum(dim=-1) / denom

    g_code = vss_code / vss_code.sum().clamp_min(1e-12)
    g_paper = vss_paper_unweighted / vss_paper_unweighted.sum().clamp_min(1e-12)
    delta = vss_code - vss_paper_unweighted

    return {
        "top_k_score": tensor_stats(p),
        "vss_code_entropy": tensor_stats(vss_code),
        "vss_paper_unweighted_nll": tensor_stats(vss_paper_unweighted),
        "vss_raw_abs_delta": tensor_stats(delta.abs()),
        "guidance_l1_delta": float(torch.sum(torch.abs(g_code - g_paper)).item()),
        "guidance_linf_delta": float(torch.max(torch.abs(g_code - g_paper)).item()),
        "guidance_cosine": safe_cosine(g_code, g_paper),
        "guidance_top10_overlap": topk_overlap(g_code, g_paper, 10),
        "guidance_top50_overlap": topk_overlap(g_code, g_paper, 50),
        "code_guidance_sum": float(g_code.sum().item()),
        "paper_guidance_sum": float(g_paper.sum().item()),
    }


class AttentionAddDebugger:
    def __init__(
        self,
        modeling_llama: Any,
        *,
        image_start: int,
        image_end: int,
        ideal_beta: float,
        max_rows: int,
    ) -> None:
        self.modeling_llama = modeling_llama
        self.image_start = int(image_start)
        self.image_end = int(image_end)
        self.ideal_beta = float(ideal_beta)
        self.max_rows = int(max_rows)
        self.rows: List[Dict[str, Any]] = []

    def hook(self, module: torch.nn.Module, args: Any, kwargs: Dict[str, Any]) -> None:
        if len(self.rows) >= self.max_rows:
            return
        if not parse_bool(kwargs.get("use_add", False)):
            return
        vl_guidance = kwargs.get("vl_guidance")
        if vl_guidance is None:
            return
        attn_coef_arg = kwargs.get("attn_coef")
        if attn_coef_arg is None:
            return

        try:
            row = self._compute(module, args, kwargs)
        except Exception as exc:  # keep diagnostics non-invasive
            row = {
                "error": repr(exc),
                "layer_idx": int(kwargs.get("layer_idx", -1) or -1),
            }
        self.rows.append(row)

    def _compute(self, module: torch.nn.Module, args: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            raise RuntimeError("missing hidden_states")
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        past_key_value = kwargs.get("past_key_value")
        use_cache = parse_bool(kwargs.get("use_cache", False))
        vl_guidance = kwargs["vl_guidance"]
        attn_coef_input = float(kwargs["attn_coef"])
        head_balancing = kwargs.get("head_balancing")
        attn_norm = parse_bool(kwargs.get("attn_norm", False))
        layer_idx = int(kwargs.get("layer_idx", -1) or -1)

        bsz, q_len, _ = hidden_states.size()
        if getattr(module, "pretraining_tp", 1) > 1:
            raise RuntimeError("pretraining_tp > 1 is not implemented in this diagnostic")

        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = module.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self.modeling_llama.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        _ = (key_states, value_states) if use_cache else None

        key_states = self.modeling_llama.repeat_kv(key_states, module.num_key_value_groups)
        value_states = self.modeling_llama.repeat_kv(value_states, module.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output_preadd = torch.matmul(attn_weights, value_states)

        support_fraction_t = (vl_guidance > 1e-8).sum().float() / float(vl_guidance.size(-1))
        support_fraction = float(support_fraction_t.item())
        attn_coef_scaled = attn_coef_input * support_fraction

        g = vl_guidance.unsqueeze(0).repeat(1, module.num_heads, 1)
        g = g.unsqueeze(2).to(value_states.dtype)
        gv = torch.matmul(g, value_states[:, :, self.image_start : self.image_end, :])
        vis_update = gv.expand(attn_output_preadd.size())

        if head_balancing == "simg":
            sim = F.cosine_similarity(vis_update, attn_output_preadd, dim=-1)
            headw = (1 + sim) / 2
            headw = headw / headw.sum(1)
            headw = headw * headw.size(1)
            headw = F.relu(2 - headw)
        elif head_balancing == "none":
            headw = torch.ones(vis_update.shape[:-1], dtype=value_states.dtype, device=value_states.device)
        else:
            raise RuntimeError(f"unsupported head_balancing for this diagnostic: {head_balancing}")

        coef = attn_coef_scaled * headw
        if attn_norm:
            coef = coef / (1 + coef)
            added_code = coef.unsqueeze(-1) * (vis_update - attn_output_preadd)
            effective_formula = coef.unsqueeze(-1) * (vis_update - attn_output_preadd)
        else:
            added_code = coef.unsqueeze(-1) * vis_update
            effective_formula = coef.unsqueeze(-1) * vis_update

        ideal_attncoef = self.ideal_beta * vis_update
        ideal_scaled = attn_coef_scaled * vis_update

        return {
            "layer_idx": layer_idx,
            "q_len": int(q_len),
            "kv_seq_len": int(kv_seq_len),
            "head_balancing": str(head_balancing),
            "attn_norm": int(bool(attn_norm)),
            "support_fraction": support_fraction,
            "attn_coef_input": attn_coef_input,
            "attn_coef_scaled": float(attn_coef_scaled),
            "ideal_beta": float(self.ideal_beta),
            "headw_mean": tensor_stats(headw)["mean"],
            "headw_std": tensor_stats(headw)["std"],
            "headw_min": tensor_stats(headw)["min"],
            "headw_max": tensor_stats(headw)["max"],
            "coef_mean": tensor_stats(coef)["mean"],
            "coef_std": tensor_stats(coef)["std"],
            "coef_min": tensor_stats(coef)["min"],
            "coef_max": tensor_stats(coef)["max"],
            "gv_norm": float(torch.linalg.vector_norm(vis_update.detach().float()).item()),
            "attn_output_preadd_norm": float(torch.linalg.vector_norm(attn_output_preadd.detach().float()).item()),
            "added_code_norm": float(torch.linalg.vector_norm(added_code.detach().float()).item()),
            "rel_err_raw_vis_update_vs_gv": safe_rel_err(vis_update, gv.expand(attn_output_preadd.size())),
            "rel_err_added_vs_beta_gv": safe_rel_err(added_code, ideal_attncoef),
            "rel_err_added_vs_scaled_attncoef_gv": safe_rel_err(added_code, ideal_scaled),
            "rel_err_added_vs_effective_formula": safe_rel_err(added_code, effective_formula),
        }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose VGA captioning VSS formula and attention-add scaling on real LLaVA/VGA forwards."
        )
    )
    ap.add_argument("--vga_root", default="VGA_origin")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--vss_k", type=int, default=10)
    ap.add_argument("--image_start", type=int, default=35)
    ap.add_argument("--image_end", type=int, default=611)
    ap.add_argument("--use_add", type=parse_bool, default=True)
    ap.add_argument("--cd_alpha", type=float, default=0.02)
    ap.add_argument("--attn_coef", type=float, default=0.2)
    ap.add_argument("--ideal_beta", type=float, default=None)
    ap.add_argument("--start_layer", type=int, default=2)
    ap.add_argument("--end_layer", type=int, default=15)
    ap.add_argument("--head_balancing", default="simg", choices=["simg", "none"])
    ap.add_argument("--attn_norm", type=parse_bool, default=False)
    ap.add_argument("--sampling", type=parse_bool, default=False)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--debug_max_rows", type=int, default=128)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    vga_root = (repo_root / args.vga_root).resolve()
    sys.path.insert(0, str(vga_root))
    sys.path.insert(0, str(vga_root / "llava"))

    from PIL import Image
    from transformers import set_seed

    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model import modeling_llama
    from llava.utils import disable_torch_init
    from vcd_utils.greedy_sample import evolve_greedy_sampling

    evolve_greedy_sampling()
    set_seed(int(args.seed))
    disable_torch_init()

    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        os.path.expanduser(args.model_path), args.model_base, model_name
    )
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head

    questions = []
    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
            if len(questions) >= int(args.limit):
                break

    debugger = AttentionAddDebugger(
        modeling_llama,
        image_start=int(args.image_start),
        image_end=int(args.image_end),
        ideal_beta=float(args.attn_coef if args.ideal_beta is None else args.ideal_beta),
        max_rows=int(args.debug_max_rows),
    )
    handles = []
    for module in model.modules():
        if module.__class__.__name__ == "LlamaAttention":
            handles.append(module.register_forward_pre_hook(debugger.hook, with_kwargs=True))

    samples: List[Dict[str, Any]] = []
    try:
        for idx, line in enumerate(questions):
            image_file = line["image"]
            qs = line["question"]
            if model.config.mm_use_im_start_end:
                qs_full = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs_full = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_full)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            with torch.inference_mode():
                outputs = model(
                    input_ids[:, :-1],
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits
                vis_logits = torch.softmax(logits[0, int(args.image_start) : int(args.image_end), :], dim=-1)
                vss = vss_compare(vis_logits, int(args.vss_k))

                # Object-agnostic guidance path used by caption questions without an object field.
                top_k_scores, _ = torch.topk(vis_logits, int(args.vss_k), dim=-1)
                p = top_k_scores.float().clamp_min(1e-12)
                entropy = (-p * torch.log(p)).sum(-1) / math.log(float(args.vss_k))
                vl_guidance = (entropy / entropy.sum().clamp_min(1e-12)).to(vis_logits.dtype)

                n_attn_before = len(debugger.rows)
                output_ids = model.generate(
                    input_ids[:, -1:],
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    past_key_values=outputs.past_key_values,
                    vl_guidance=vl_guidance,
                    vis_logits=vis_logits,
                    cd_alpha=float(args.cd_alpha),
                    add_layer=list(range(int(args.start_layer), int(args.end_layer) + 1)),
                    attn_coef=float(args.attn_coef),
                    use_add=bool(args.use_add),
                    head_balancing=str(args.head_balancing),
                    attn_norm=bool(args.attn_norm),
                    do_sample=True,
                    sampling=bool(args.sampling),
                    num_beams=1,
                    max_new_tokens=int(args.max_new_tokens),
                    use_cache=True,
                )
                n_attn_after = len(debugger.rows)
            decoded = tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)[0]
            samples.append(
                {
                    "row_index": idx,
                    "question_id": line.get("question_id", line.get("id", "")),
                    "image": image_file,
                    "image_id": line.get("image_id", ""),
                    "has_object_field": int("object" in line and bool(line.get("object"))),
                    "generated_preview": decoded[:300],
                    "vss": vss,
                    "n_attention_debug_rows": int(n_attn_after - n_attn_before),
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    attn_rows = debugger.rows
    summary: Dict[str, Any] = {
        "inputs": vars(args),
        "n_samples": len(samples),
        "n_attention_debug_rows": len(attn_rows),
        "samples": samples,
        "attention_debug_rows": attn_rows[: int(args.debug_max_rows)],
    }
    if attn_rows:
        for key in (
            "support_fraction",
            "attn_coef_scaled",
            "headw_mean",
            "headw_std",
            "coef_mean",
            "coef_std",
            "rel_err_raw_vis_update_vs_gv",
            "rel_err_added_vs_beta_gv",
            "rel_err_added_vs_scaled_attncoef_gv",
            "rel_err_added_vs_effective_formula",
        ):
            vals = torch.tensor([float(r[key]) for r in attn_rows if key in r], dtype=torch.float32)
            if vals.numel():
                summary[f"attention_{key}_stats"] = tensor_stats(vals)

    write_json(os.path.abspath(args.out_json), summary)
    print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
