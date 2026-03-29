#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import analyze_artrap_pairwise_fragility as pf


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def token_is_prefix(tok: str) -> bool:
    t = str(tok)
    # SentencePiece (LLaMA) word boundary: U+2581.
    # Keep Ġ as fallback for GPT/BPE tokenizers.
    # Exclude byte-fallback tokens like <0xAB>.
    if t.startswith("<0x") and t.endswith(">"):
        return False
    return t.startswith("\u2581") or t.startswith("Ġ")


def build_global_view(image, downsample_short: int):
    """
    Create a low-detail global view by heavy downsample->upsample.
    This avoids Q-only abyss while removing local patch detail.
    """
    w, h = image.size
    ds = int(max(8, downsample_short))
    if w <= h:
        nw = ds
        nh = int(max(1, round(h * ds / max(1, w))))
    else:
        nh = ds
        nw = int(max(1, round(w * ds / max(1, h))))
    try:
        from PIL import Image as PILImage
        bicubic = PILImage.Resampling.BICUBIC  # Pillow>=9
    except Exception:
        try:
            from PIL import Image as PILImage
            bicubic = PILImage.BICUBIC
        except Exception:
            bicubic = 3  # Fallback to bicubic enum value.
    small = image.resize((nw, nh), resample=bicubic)
    return small.resize((w, h), resample=bicubic)


@torch.no_grad()
def decode_dual_stream(
    *,
    model,
    tokenizer,
    input_ids_img: torch.Tensor,
    input_ids_ref: torch.Tensor,
    images_tensor: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    images_tensor_ref: Optional[torch.Tensor],
    image_sizes_ref: Optional[List[Tuple[int, int]]],
    ref_uses_image: bool,
    method: str,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    top_k_rank: int,
    alpha_rank: float,
    state_top_k: int,
    prefix_spike_z: float,
    beta_prefix: float,
    suffix_vpmi_floor: float,
    beta_suffix: float,
    apc_top_k: int,
    alpha_vpmi: float,
    vpmi_temp: float,
    apc_delta_logp: float,
    apc_clip: float,
) -> List[int]:
    # Warmup forward for each stream.
    out_img = model(
        input_ids=input_ids_img,
        images=images_tensor,
        image_sizes=image_sizes,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    out_ref_kwargs: Dict[str, Any] = {
        "input_ids": input_ids_ref,
        "use_cache": True,
        "output_attentions": False,
        "return_dict": True,
    }
    if bool(ref_uses_image):
        out_ref_kwargs["images"] = images_tensor_ref
        out_ref_kwargs["image_sizes"] = image_sizes_ref
    out_ref = model(**out_ref_kwargs)
    past_img = out_img.past_key_values
    past_ref = out_ref.past_key_values
    logits_img = out_img.logits[:, -1, :].float()
    logits_ref = out_ref.logits[:, -1, :].float()

    gen_ids: List[int] = []
    mm = str(method).strip().lower()
    use_a = mm in {"a_only", "a_plus_b"}
    use_b = mm in {"b_only", "a_plus_b"}
    use_apc = mm in {"apc_nudge"}
    use_lg = mm in {"local_global_contrast"}

    for _ in range(int(max_new_tokens)):
        adj = logits_img[0].clone()
        log_probs_img = F.log_softmax(logits_img[0], dim=-1)
        log_probs_ref = F.log_softmax(logits_ref[0], dim=-1)
        vpmi = (log_probs_img - log_probs_ref).clone()

        # Strategy A: rank-based reweighting inside top-k of image stream.
        if use_a:
            kk = int(min(max(1, top_k_rank), int(adj.numel())))
            top_ids = torch.topk(logits_img[0], k=kk, dim=-1).indices
            vp = vpmi[top_ids]
            order = torch.argsort(vp, descending=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(kk, device=order.device)
            denom = float(max(1, kk - 1))
            rank_pct = (float(kk - 1) - rank.float()) / denom
            # Centered bias to avoid global logit inflation.
            bias = float(alpha_rank) * (rank_pct - 0.5)
            adj[top_ids] = adj[top_ids] + bias

        # Strategy B: sub-word state machine over top-k candidates.
        if use_b:
            kk = int(min(max(1, state_top_k), int(adj.numel())))
            top_ids = torch.topk(adj, k=kk, dim=-1).indices
            top_v = vpmi[top_ids]
            mu = float(torch.mean(top_v).item())
            sd = float(torch.std(top_v, unbiased=False).item())
            thr = float(mu + float(prefix_spike_z) * sd)
            tok_strs = tokenizer.convert_ids_to_tokens(top_ids.tolist())
            for tid, tok in zip(top_ids.tolist(), tok_strs):
                v = float(vpmi[int(tid)].item())
                if token_is_prefix(tok):
                    if v > thr:
                        adj[int(tid)] -= float(beta_prefix) * float(v - thr)
                else:
                    if v >= float(suffix_vpmi_floor):
                        # Smooth boost for suffix tokens with sustained visual support.
                        boost = float(math.tanh((v - float(suffix_vpmi_floor)) / 2.0))
                        adj[int(tid)] += float(beta_suffix) * boost

        # Strategy C: continuous plausibility-constrained nudging (no hard threshold gate).
        if use_apc:
            kk = int(min(max(1, apc_top_k), int(adj.numel())))
            top_ids = torch.topk(logits_img[0], k=kk, dim=-1).indices
            top_lp = log_probs_img[top_ids]
            vp = vpmi[top_ids]

            lp_max = float(torch.max(top_lp).item())
            delta = float(max(1e-6, apc_delta_logp))
            temp = float(max(1e-6, vpmi_temp))

            plaus = torch.clamp(1.0 - ((lp_max - top_lp) / delta), min=0.0, max=1.0)
            nudge = torch.tanh(vp / temp)
            bias = float(alpha_vpmi) * plaus * nudge
            if float(apc_clip) > 0.0:
                bias = torch.clamp(bias, min=-float(apc_clip), max=float(apc_clip))
            adj[top_ids] = adj[top_ids] + bias

        # Strategy LG: local-vs-global contrast (replace Q-only reference).
        if use_lg:
            kk = int(min(max(1, apc_top_k), int(adj.numel())))
            top_ids = torch.topk(logits_img[0], k=kk, dim=-1).indices
            temp = float(max(1e-6, vpmi_temp))
            bias = float(alpha_vpmi) * torch.tanh(vpmi[top_ids] / temp)
            if float(apc_clip) > 0.0:
                bias = torch.clamp(bias, min=-float(apc_clip), max=float(apc_clip))
            adj[top_ids] = adj[top_ids] + bias

        next_id = int(torch.argmax(adj, dim=-1).item())
        if eos_token_id is not None and int(next_id) == int(eos_token_id):
            break
        gen_ids.append(int(next_id))

        step = torch.tensor([[int(next_id)]], dtype=torch.long, device=input_ids_img.device)
        out_img = model(
            input_ids=step,
            past_key_values=past_img,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        out_ref = model(
            input_ids=step,
            past_key_values=past_ref,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_img = out_img.past_key_values
        past_ref = out_ref.past_key_values
        logits_img = out_img.logits[:, -1, :].float()
        logits_ref = out_ref.logits[:, -1, :].float()

    return [int(x) for x in gen_ids]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-stream Beam1 ablation: baseline / A / B / A+B / APC.")
    ap.add_argument("--questions_json", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "baseline",
            "a_only",
            "b_only",
            "a_plus_b",
            "apc_nudge",
            "local_global_contrast",
        ],
    )
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--eval_match_mode", type=str, default="heuristic", choices=["strict", "heuristic"])

    # A params
    ap.add_argument("--top_k_rank", type=int, default=50)
    ap.add_argument("--alpha_rank", type=float, default=1.2)

    # B params
    ap.add_argument("--state_top_k", type=int, default=50)
    ap.add_argument("--prefix_spike_z", type=float, default=1.0)
    ap.add_argument("--beta_prefix", type=float, default=1.0)
    ap.add_argument("--suffix_vpmi_floor", type=float, default=-0.5)
    ap.add_argument("--beta_suffix", type=float, default=0.8)

    # APC params (threshold-free continuous nudge)
    ap.add_argument("--apc_top_k", type=int, default=50)
    ap.add_argument("--alpha_vpmi", type=float, default=0.8)
    ap.add_argument("--vpmi_temp", type=float, default=2.0)
    ap.add_argument("--apc_delta_logp", type=float, default=2.0)
    ap.add_argument("--apc_clip", type=float, default=1.0)
    ap.add_argument(
        "--global_downsample_short",
        type=int,
        default=32,
        help="Short side for local-global reference image (downsample then upsample).",
    )

    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    qpath = os.path.abspath(args.questions_json)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = pf.read_questions(qpath)
    if int(args.num_samples) > 0:
        rows = rows[: int(args.num_samples)]

    # Lazy imports from LLaVA runtime.
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from PIL import Image

    # Populate globals used by helper functions imported from pairwise analyzer.
    pf.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
    pf.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
    pf.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
    pf.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
    pf.conv_templates = conv_templates

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_4bit=False,
        load_8bit=False,
        use_flash_attn=bool(args.use_flash_attn),
        device_map="auto",
    )
    model.eval()

    if str(args.attn_impl) != "auto":
        try:
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = str(args.attn_impl)
        except Exception:
            pass
        try:
            mm = model.get_model()
            if hasattr(mm.config, "attn_implementation"):
                mm.config.attn_implementation = str(args.attn_impl)
        except Exception:
            pass

    conv_mode = pf.resolve_conv_mode(model_name, args.conv_mode)
    device = model.get_model().embed_tokens.weight.device
    eos_id = getattr(tokenizer, "eos_token_id", None)

    per_sample: List[Dict[str, Any]] = []
    t0 = time.time()

    pbar = tqdm(rows, total=len(rows), desc=f"dualstream-{args.method}", dynamic_ncols=True)
    for i, r in enumerate(pbar):
        qid = str(r.get("id", ""))
        question = str(r.get("question", ""))
        answer = str(r.get("answer", ""))
        image_id = str(r.get("imageId", ""))
        image_path = os.path.join(args.image_root, f"{image_id}.jpg")
        if not os.path.isfile(image_path):
            per_sample.append({"id": qid, "error": f"missing_image:{image_path}"})
            continue

        ts = time.time()
        try:
            img_prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            q_prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=False,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            input_ids_img = tokenizer_image_token(
                img_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            input_ids_q = tokenizer(q_prompt, return_tensors="pt").input_ids.to(device)

            image = Image.open(image_path).convert("RGB")
            image_sizes = [image.size]
            images_tensor = process_images([image], image_processor, model.config).to(
                device=model.device,
                dtype=torch.float16,
            )
            mm = str(args.method).strip().lower()
            if mm == "local_global_contrast":
                global_image = build_global_view(image, int(args.global_downsample_short))
                image_sizes_ref = [global_image.size]
                images_tensor_ref = process_images([global_image], image_processor, model.config).to(
                    device=model.device,
                    dtype=torch.float16,
                )
                input_ids_ref = input_ids_img
                ref_uses_image = True
            else:
                input_ids_ref = input_ids_q
                images_tensor_ref = None
                image_sizes_ref = None
                ref_uses_image = False

            gen_ids = decode_dual_stream(
                model=model,
                tokenizer=tokenizer,
                input_ids_img=input_ids_img,
                input_ids_ref=input_ids_ref,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
                images_tensor_ref=images_tensor_ref,
                image_sizes_ref=image_sizes_ref,
                ref_uses_image=bool(ref_uses_image),
                method=str(args.method),
                max_new_tokens=int(args.max_new_tokens),
                eos_token_id=eos_id,
                top_k_rank=int(args.top_k_rank),
                alpha_rank=float(args.alpha_rank),
                state_top_k=int(args.state_top_k),
                prefix_spike_z=float(args.prefix_spike_z),
                beta_prefix=float(args.beta_prefix),
                suffix_vpmi_floor=float(args.suffix_vpmi_floor),
                beta_suffix=float(args.beta_suffix),
                apc_top_k=int(args.apc_top_k),
                alpha_vpmi=float(args.alpha_vpmi),
                vpmi_temp=float(args.vpmi_temp),
                apc_delta_logp=float(args.apc_delta_logp),
                apc_clip=float(args.apc_clip),
            )
        except Exception as e:
            per_sample.append({"id": qid, "error": f"infer:{e}"})
            continue

        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        short_answer = pf.extract_core_answer_text(question=question, text=text, max_words=6)
        pred_answer = pf.first_clause(short_answer if short_answer else text)
        ok_strict = bool(pf.norm_text(pred_answer) == pf.norm_text(answer))
        ok_heur = bool(
            pf.is_success_heuristic(
                question=question,
                answer=answer,
                champ_text=text,
                champ_short=short_answer,
            )
        )
        ok = bool(ok_heur if str(args.eval_match_mode) == "heuristic" else ok_strict)

        per_sample.append(
            {
                "id": qid,
                "image_id": image_id,
                "question": question,
                "answer": answer,
                "method": str(args.method),
                "pred_text": text,
                "pred_answer_eval": pred_answer,
                "is_success": bool(ok),
                "is_success_strict": bool(ok_strict),
                "is_success_heuristic": bool(ok_heur),
                "elapsed_sec_sample": float(time.time() - ts),
                "n_gen_tokens": int(len(gen_ids)),
                "error": None,
            }
        )

        if hasattr(pbar, "set_postfix"):
            done = int(i + 1)
            elapsed = float(time.time() - t0)
            avg = float(elapsed / max(1, done))
            pbar.set_postfix(avg_s=f"{avg:.2f}", ok=int(sum(1 for x in per_sample if x.get('error') is None and bool(x.get('is_success')))))

    valid = [r for r in per_sample if r.get("error") is None]
    n_valid = int(len(valid))
    acc = (
        None
        if n_valid == 0
        else float(sum(1 for r in valid if bool(r.get("is_success", False))) / n_valid)
    )
    acc_s = (
        None
        if n_valid == 0
        else float(sum(1 for r in valid if bool(r.get("is_success_strict", False))) / n_valid)
    )
    acc_h = (
        None
        if n_valid == 0
        else float(sum(1 for r in valid if bool(r.get("is_success_heuristic", False))) / n_valid)
    )
    mean_latency = (
        None
        if n_valid == 0
        else float(
            sum(float(safe_float(r.get("elapsed_sec_sample")) or 0.0) for r in valid) / n_valid
        )
    )
    mean_tokens = (
        None
        if n_valid == 0
        else float(
            sum(float(safe_float(r.get("n_gen_tokens")) or 0.0) for r in valid) / n_valid
        )
    )

    summary = {
        "inputs": {
            "questions_json": qpath,
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "method": str(args.method),
            "max_new_tokens": int(args.max_new_tokens),
            "eval_match_mode": str(args.eval_match_mode),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "top_k_rank": int(args.top_k_rank),
            "alpha_rank": float(args.alpha_rank),
            "state_top_k": int(args.state_top_k),
            "prefix_spike_z": float(args.prefix_spike_z),
            "beta_prefix": float(args.beta_prefix),
            "suffix_vpmi_floor": float(args.suffix_vpmi_floor),
            "beta_suffix": float(args.beta_suffix),
            "apc_top_k": int(args.apc_top_k),
            "alpha_vpmi": float(args.alpha_vpmi),
            "vpmi_temp": float(args.vpmi_temp),
            "apc_delta_logp": float(args.apc_delta_logp),
            "apc_clip": float(args.apc_clip),
            "global_downsample_short": int(args.global_downsample_short),
        },
        "counts": {
            "n_total": int(len(rows)),
            "n_valid": int(n_valid),
            "n_error": int(len(rows) - n_valid),
            "accuracy": acc,
            "accuracy_strict": acc_s,
            "accuracy_heuristic": acc_h,
            "mean_latency_sec_per_sample": mean_latency,
            "mean_generated_tokens": mean_tokens,
        },
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_sample.csv"), per_sample)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
