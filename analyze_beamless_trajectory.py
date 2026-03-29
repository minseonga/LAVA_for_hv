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


TRUE_SET = {"1", "true", "t", "yes", "y"}


def as_bool(x: Any) -> bool:
    return str("" if x is None else x).strip().lower() in TRUE_SET


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


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


def token_is_prefix(tok: str) -> bool:
    t = str(tok)
    if t.startswith("<0x") and t.endswith(">"):
        return False
    return t.startswith("\u2581") or t.startswith("Ġ")


def trim_generated(seq_ids: List[int], prompt_ids: List[int], max_new_tokens: int, eos_id: Optional[int]) -> List[int]:
    if len(seq_ids) >= len(prompt_ids) and len(prompt_ids) > 0 and seq_ids[: len(prompt_ids)] == prompt_ids:
        out = [int(x) for x in seq_ids[len(prompt_ids):]]
    else:
        out = [int(x) for x in seq_ids]
    if len(out) > int(max_new_tokens):
        out = out[: int(max_new_tokens)]
    if eos_id is not None and int(eos_id) in out:
        out = out[: int(out.index(int(eos_id)))]
    return [int(x) for x in out]


@torch.no_grad()
def greedy_generate(
    *,
    model,
    input_ids_img: torch.Tensor,
    images_tensor: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    max_new_tokens: int,
    eos_id: Optional[int],
) -> List[int]:
    out = model.generate(
        input_ids_img,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=int(max_new_tokens),
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=False,
    )
    seq = [int(x) for x in out.sequences[0].tolist()]
    prompt = [int(x) for x in input_ids_img[0].tolist()]
    return trim_generated(seq, prompt, int(max_new_tokens), eos_id)


@torch.no_grad()
def teacher_forced_step_logits(
    *,
    model,
    prefix_ids: torch.Tensor,
    cont_ids: List[int],
    images_tensor: Optional[torch.Tensor],
    image_sizes: Optional[List[Tuple[int, int]]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Returns: step_logits[T,V], selected_logp[T]
    if len(cont_ids) == 0:
        return None, None
    device = prefix_ids.device
    cont = torch.tensor([cont_ids], dtype=torch.long, device=device)
    inp = torch.cat([prefix_ids, cont], dim=1)
    try:
        out = model(
            input_ids=inp,
            images=images_tensor,
            image_sizes=image_sizes,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    except Exception:
        return None, None
    p = int(prefix_ids.size(1))
    t = int(cont.size(1))
    if p <= 0:
        return None, None
    step_logits = out.logits[:, p - 1: p - 1 + t, :].float()[0]  # [T,V]
    step_logp = F.log_softmax(step_logits, dim=-1)
    selected = step_logp.gather(dim=-1, index=cont[0].unsqueeze(-1)).squeeze(-1)  # [T]
    return step_logits, selected


def trajectory_features(
    *,
    tokenizer,
    gen_ids: List[int],
    logits_img: torch.Tensor,
    logits_q: torch.Tensor,
    sel_logp_img: torch.Tensor,
    sel_logp_q: torch.Tensor,
    rank_top_k: int,
    k_prefix: int,
    k_suffix: int,
) -> Dict[str, Any]:
    # logits_*: [T,V], sel_logp_*: [T]
    t = int(len(gen_ids))
    if t == 0:
        return {
            "vpmi_toks": [],
            "rank_pct_toks": [],
            "margin_toks": [],
            "vpmi_mean": None,
            "prefix_mean": None,
            "suffix_min": None,
            "collapse_gap": None,
            "rank_mean": None,
            "rank_min": None,
            "margin_min": None,
            "energy_mean": None,
        }
    lp_img = F.log_softmax(logits_img, dim=-1)
    lp_q = F.log_softmax(logits_q, dim=-1)
    vpmi_t = (sel_logp_img - sel_logp_q).detach().cpu().tolist()

    rank_pct_toks: List[float] = []
    margin_toks: List[float] = []
    kk = int(max(1, rank_top_k))
    for i in range(t):
        li = lp_img[i]
        lq = lp_q[i]
        k = int(min(kk, int(li.numel())))
        topv, topi = torch.topk(li, k=k, dim=-1)
        vp = li[topi] - lq[topi]
        order = torch.argsort(vp, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(k, device=order.device)
        tid = int(gen_ids[i])
        pos = (topi == tid).nonzero(as_tuple=False)
        if int(pos.numel()) == 0:
            rank_pct_toks.append(0.0)
        else:
            r = int(ranks[int(pos[0].item())].item())
            rank_pct_toks.append(float((k - 1 - r) / max(1, k - 1)))

        m2 = torch.topk(logits_img[i], k=2, dim=-1).values
        margin_toks.append(float((m2[0] - m2[1]).item()))

    kp = int(max(1, min(k_prefix, t)))
    ks = int(max(1, min(k_suffix, t)))
    pref = [float(x) for x in vpmi_t[:kp]]
    suf = [float(x) for x in vpmi_t[t - ks:]]
    vpmi_mean = float(sum(vpmi_t) / len(vpmi_t))
    prefix_mean = float(sum(pref) / len(pref))
    suffix_min = float(min(suf))
    collapse_gap = float(suffix_min - prefix_mean)
    rank_mean = float(sum(rank_pct_toks) / len(rank_pct_toks))
    rank_min = float(min(rank_pct_toks))
    margin_min = float(min(margin_toks))
    energy_mean = float(-sum(float(x) for x in sel_logp_img.detach().cpu().tolist()) / t)

    return {
        "vpmi_toks": [float(x) for x in vpmi_t],
        "rank_pct_toks": [float(x) for x in rank_pct_toks],
        "margin_toks": [float(x) for x in margin_toks],
        "vpmi_mean": vpmi_mean,
        "prefix_mean": prefix_mean,
        "suffix_min": suffix_min,
        "collapse_gap": collapse_gap,
        "rank_mean": rank_mean,
        "rank_min": rank_min,
        "margin_min": margin_min,
        "energy_mean": energy_mean,
    }


def pick_cut(vpmi_toks: List[float]) -> int:
    n = int(len(vpmi_toks))
    if n <= 2:
        return int(max(1, n - 1))
    s = int(n // 2)
    j = int(min(range(s, n), key=lambda i: float(vpmi_toks[i])))
    return int(max(1, min(j, n - 1)))


@torch.no_grad()
def branch_first_tokens(
    *,
    model,
    input_ids_img: torch.Tensor,
    input_ids_q: torch.Tensor,
    prefix_gen: List[int],
    images_tensor: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    branch_top_k: int,
    branch_budget: int,
    lambda_rank: float,
    lambda_vpmi: float,
) -> List[int]:
    device = input_ids_img.device
    pimg = torch.tensor([list(input_ids_img[0].tolist()) + [int(x) for x in prefix_gen]], dtype=torch.long, device=device)
    pq = torch.tensor([list(input_ids_q[0].tolist()) + [int(x) for x in prefix_gen]], dtype=torch.long, device=device)
    try:
        out_i = model(
            input_ids=pimg,
            images=images_tensor,
            image_sizes=image_sizes,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        out_q = model(
            input_ids=pq,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    except Exception:
        return []
    li = out_i.logits[:, -1, :].float()[0]
    lq = out_q.logits[:, -1, :].float()[0]
    lpi = F.log_softmax(li, dim=-1)
    lpq = F.log_softmax(lq, dim=-1)
    vp = lpi - lpq

    k = int(min(max(1, branch_top_k), int(li.numel())))
    topi = torch.topk(lpi, k=k, dim=-1).indices
    vals_vp = vp[topi]
    order = torch.argsort(vals_vp, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(k, device=order.device)
    rank_pct = (float(k - 1) - ranks.float()) / float(max(1, k - 1))
    score = lpi[topi] + float(lambda_rank) * rank_pct + float(lambda_vpmi) * vals_vp
    bb = int(min(max(1, branch_budget), k))
    best_local = torch.topk(score, k=bb, dim=-1).indices
    return [int(topi[int(j)].item()) for j in best_local.tolist()]


@torch.no_grad()
def complete_from_prefix(
    *,
    model,
    input_ids_img: torch.Tensor,
    prefix_gen: List[int],
    first_token: int,
    images_tensor: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    max_new_tokens_total: int,
    eos_id: Optional[int],
) -> List[int]:
    prompt_ids = [int(x) for x in input_ids_img[0].tolist()]
    base = [int(x) for x in (prompt_ids + prefix_gen + [int(first_token)])]
    rem = int(max_new_tokens_total) - int(len(prefix_gen)) - 1
    if rem < 0:
        rem = 0
    inp = torch.tensor([base], dtype=torch.long, device=input_ids_img.device)
    out = model.generate(
        inp,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=int(rem),
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=False,
    )
    seq = [int(x) for x in out.sequences[0].tolist()]
    if len(seq) >= len(prompt_ids) and seq[: len(prompt_ids)] == prompt_ids:
        gen = [int(x) for x in seq[len(prompt_ids):]]
    else:
        gen = [int(x) for x in seq]
    if len(gen) > int(max_new_tokens_total):
        gen = gen[: int(max_new_tokens_total)]
    if eos_id is not None and int(eos_id) in gen:
        gen = gen[: int(gen.index(int(eos_id)))]
    return [int(x) for x in gen]


def main() -> None:
    ap = argparse.ArgumentParser(description="Beamless trajectory method: Phase1->Phase3->Phase4")
    ap.add_argument("--questions_json", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--eval_match_mode", type=str, default="heuristic", choices=["strict", "heuristic"])
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # rank/self features
    ap.add_argument("--rank_top_k", type=int, default=50)
    ap.add_argument("--k_prefix", type=int, default=2)
    ap.add_argument("--k_suffix", type=int, default=2)

    # risk gate
    ap.add_argument("--tau_collapse_gap", type=float, default=-0.8)
    ap.add_argument("--tau_rank_min", type=float, default=0.05)
    ap.add_argument("--tau_energy", type=float, default=7.0)
    ap.add_argument("--gate_mode", type=str, default="or", choices=["or", "and"])

    # tail micro regen
    ap.add_argument("--branch_top_k", type=int, default=8)
    ap.add_argument("--branch_budget", type=int, default=3)
    ap.add_argument("--branch_lambda_rank", type=float, default=0.8)
    ap.add_argument("--branch_lambda_vpmi", type=float, default=0.8)
    ap.add_argument("--select_w_suffix", type=float, default=0.5)
    ap.add_argument("--select_w_rank", type=float, default=0.2)
    ap.add_argument("--switch_margin", type=float, default=0.05, help="required score gap to switch")
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    qpath = os.path.abspath(args.questions_json)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = pf.read_questions(qpath)
    if int(args.num_samples) > 0:
        rows = rows[: int(args.num_samples)]

    if torch is None:
        raise RuntimeError("PyTorch required.")

    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.utils import disable_torch_init
    from PIL import Image

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
    pbar = tqdm(rows, total=len(rows), desc="beamless-trajectory", dynamic_ncols=True)
    for i, r in enumerate(pbar):
        sid = str(r.get("id", ""))
        q = str(r.get("question", ""))
        a = str(r.get("answer", ""))
        image_id = str(r.get("imageId", ""))
        image_path = os.path.join(args.image_root, f"{image_id}.jpg")
        if not os.path.isfile(image_path):
            per_sample.append({"id": sid, "error": f"missing_image:{image_path}"})
            continue

        ts = time.time()
        try:
            img_prompt = pf.build_prompt(
                question=q,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            q_prompt = pf.build_prompt(
                question=q,
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
        except Exception as e:
            per_sample.append({"id": sid, "error": f"build_input:{e}"})
            continue

        # Phase1: greedy V+Q
        try:
            greedy_ids = greedy_generate(
                model=model,
                input_ids_img=input_ids_img,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
                max_new_tokens=int(args.max_new_tokens),
                eos_id=eos_id,
            )
        except Exception as e:
            per_sample.append({"id": sid, "error": f"greedy:{e}"})
            continue
        if len(greedy_ids) == 0:
            per_sample.append({"id": sid, "error": "empty_greedy"})
            continue

        # Phase3: Q-only (and image) teacher-forced once on trajectory
        logits_img, sel_img = teacher_forced_step_logits(
            model=model,
            prefix_ids=input_ids_img,
            cont_ids=greedy_ids,
            images_tensor=images_tensor,
            image_sizes=image_sizes,
        )
        logits_q, sel_q = teacher_forced_step_logits(
            model=model,
            prefix_ids=input_ids_q,
            cont_ids=greedy_ids,
            images_tensor=None,
            image_sizes=None,
        )
        if logits_img is None or sel_img is None or logits_q is None or sel_q is None:
            per_sample.append({"id": sid, "error": "score_greedy_failed"})
            continue
        ft = trajectory_features(
            tokenizer=tokenizer,
            gen_ids=greedy_ids,
            logits_img=logits_img,
            logits_q=logits_q,
            sel_logp_img=sel_img,
            sel_logp_q=sel_q,
            rank_top_k=int(args.rank_top_k),
            k_prefix=int(args.k_prefix),
            k_suffix=int(args.k_suffix),
        )

        cond_collapse = bool(safe_float(ft.get("collapse_gap")) is not None and float(ft["collapse_gap"]) <= float(args.tau_collapse_gap))
        cond_rank = bool(safe_float(ft.get("rank_min")) is not None and float(ft["rank_min"]) <= float(args.tau_rank_min))
        cond_energy = bool(safe_float(ft.get("energy_mean")) is not None and float(ft["energy_mean"]) >= float(args.tau_energy))
        risk = bool(
            (cond_collapse or cond_rank or cond_energy)
            if str(args.gate_mode) == "or"
            else (cond_collapse and cond_rank and cond_energy)
        )

        final_ids = [int(x) for x in greedy_ids]
        switched = False
        best_score = float(
            (safe_float(ft.get("vpmi_mean")) or -1e9)
            + float(args.select_w_suffix) * float(safe_float(ft.get("suffix_min")) or -1e9)
            + float(args.select_w_rank) * float(safe_float(ft.get("rank_mean")) or 0.0)
        )
        if risk:
            cut = pick_cut([float(x) for x in ft.get("vpmi_toks", [])])
            prefix_gen = [int(x) for x in greedy_ids[:cut]]
            first_tokens = branch_first_tokens(
                model=model,
                input_ids_img=input_ids_img,
                input_ids_q=input_ids_q,
                prefix_gen=prefix_gen,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
                branch_top_k=int(args.branch_top_k),
                branch_budget=int(args.branch_budget),
                lambda_rank=float(args.branch_lambda_rank),
                lambda_vpmi=float(args.branch_lambda_vpmi),
            )
            cand_ids: List[List[int]] = [[int(x) for x in greedy_ids]]
            seen = {tuple(int(x) for x in greedy_ids)}
            for tid in first_tokens:
                try:
                    c = complete_from_prefix(
                        model=model,
                        input_ids_img=input_ids_img,
                        prefix_gen=prefix_gen,
                        first_token=int(tid),
                        images_tensor=images_tensor,
                        image_sizes=image_sizes,
                        max_new_tokens_total=int(args.max_new_tokens),
                        eos_id=eos_id,
                    )
                except Exception:
                    continue
                k = tuple(int(x) for x in c)
                if len(k) == 0 or k in seen:
                    continue
                seen.add(k)
                cand_ids.append([int(x) for x in c])

            for cids in cand_ids:
                li, si = teacher_forced_step_logits(
                    model=model,
                    prefix_ids=input_ids_img,
                    cont_ids=cids,
                    images_tensor=images_tensor,
                    image_sizes=image_sizes,
                )
                lq, sq = teacher_forced_step_logits(
                    model=model,
                    prefix_ids=input_ids_q,
                    cont_ids=cids,
                    images_tensor=None,
                    image_sizes=None,
                )
                if li is None or si is None or lq is None or sq is None:
                    continue
                ff = trajectory_features(
                    tokenizer=tokenizer,
                    gen_ids=cids,
                    logits_img=li,
                    logits_q=lq,
                    sel_logp_img=si,
                    sel_logp_q=sq,
                    rank_top_k=int(args.rank_top_k),
                    k_prefix=int(args.k_prefix),
                    k_suffix=int(args.k_suffix),
                )
                sc = float(
                    (safe_float(ff.get("vpmi_mean")) or -1e9)
                    + float(args.select_w_suffix) * float(safe_float(ff.get("suffix_min")) or -1e9)
                    + float(args.select_w_rank) * float(safe_float(ff.get("rank_mean")) or 0.0)
                )
                if sc > float(best_score + float(args.switch_margin)):
                    best_score = float(sc)
                    final_ids = [int(x) for x in cids]

            switched = bool(tuple(final_ids) != tuple(greedy_ids))

        # Evaluate
        greedy_text = tokenizer.decode(greedy_ids, skip_special_tokens=True).strip()
        final_text = tokenizer.decode(final_ids, skip_special_tokens=True).strip()
        greedy_short = pf.extract_core_answer_text(question=q, text=greedy_text, max_words=6)
        final_short = pf.extract_core_answer_text(question=q, text=final_text, max_words=6)
        pred_g = pf.first_clause(greedy_short if greedy_short else greedy_text)
        pred_f = pf.first_clause(final_short if final_short else final_text)
        ok_g_strict = bool(pf.norm_text(pred_g) == pf.norm_text(a))
        ok_f_strict = bool(pf.norm_text(pred_f) == pf.norm_text(a))
        ok_g_heur = bool(pf.is_success_heuristic(question=q, answer=a, champ_text=greedy_text, champ_short=greedy_short))
        ok_f_heur = bool(pf.is_success_heuristic(question=q, answer=a, champ_text=final_text, champ_short=final_short))
        ok_g = bool(ok_g_heur if str(args.eval_match_mode) == "heuristic" else ok_g_strict)
        ok_f = bool(ok_f_heur if str(args.eval_match_mode) == "heuristic" else ok_f_strict)

        per_sample.append(
            {
                "id": sid,
                "image_id": image_id,
                "question": q,
                "answer": a,
                "risk": bool(risk),
                "risk_collapse": bool(cond_collapse),
                "risk_rank": bool(cond_rank),
                "risk_energy": bool(cond_energy),
                "switched": bool(switched),
                "greedy_pred": pred_g,
                "final_pred": pred_f,
                "greedy_is_success": bool(ok_g),
                "is_success": bool(ok_f),
                "gain": bool((not ok_g) and ok_f),
                "harm": bool(ok_g and (not ok_f)),
                "greedy_text": greedy_text,
                "final_text": final_text,
                "greedy_len": int(len(greedy_ids)),
                "final_len": int(len(final_ids)),
                "traj_vpmi_mean": safe_float(ft.get("vpmi_mean")),
                "traj_prefix_mean": safe_float(ft.get("prefix_mean")),
                "traj_suffix_min": safe_float(ft.get("suffix_min")),
                "traj_collapse_gap": safe_float(ft.get("collapse_gap")),
                "traj_rank_mean": safe_float(ft.get("rank_mean")),
                "traj_rank_min": safe_float(ft.get("rank_min")),
                "traj_margin_min": safe_float(ft.get("margin_min")),
                "traj_energy_mean": safe_float(ft.get("energy_mean")),
                "elapsed_sec_sample": float(time.time() - ts),
                "error": None,
            }
        )

        if hasattr(pbar, "set_postfix"):
            done = int(i + 1)
            elapsed = float(time.time() - t0)
            avg = float(elapsed / max(1, done))
            risk_n = int(sum(1 for x in per_sample if bool(x.get("risk", False))))
            sw = int(sum(1 for x in per_sample if bool(x.get("switched", False))))
            ok = int(sum(1 for x in per_sample if x.get("error") is None and bool(x.get("is_success", False))))
            pbar.set_postfix(avg_s=f"{avg:.2f}", risk=risk_n, switched=sw, ok=ok)

    valid = [r for r in per_sample if r.get("error") is None]
    n = int(len(valid))
    base_acc = None if n == 0 else float(sum(1 for r in valid if bool(r.get("greedy_is_success", False))) / n)
    final_acc = None if n == 0 else float(sum(1 for r in valid if bool(r.get("is_success", False))) / n)
    gain = int(sum(1 for r in valid if bool(r.get("gain", False))))
    harm = int(sum(1 for r in valid if bool(r.get("harm", False))))
    risk_n = int(sum(1 for r in valid if bool(r.get("risk", False))))
    sw = int(sum(1 for r in valid if bool(r.get("switched", False))))
    mean_latency = None if n == 0 else float(sum(float(safe_float(r.get("elapsed_sec_sample")) or 0.0) for r in valid) / n)

    summary = {
        "inputs": {
            "questions_json": qpath,
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "eval_match_mode": str(args.eval_match_mode),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "max_new_tokens": int(args.max_new_tokens),
            "rank_top_k": int(args.rank_top_k),
            "k_prefix": int(args.k_prefix),
            "k_suffix": int(args.k_suffix),
            "tau_collapse_gap": float(args.tau_collapse_gap),
            "tau_rank_min": float(args.tau_rank_min),
            "tau_energy": float(args.tau_energy),
            "gate_mode": str(args.gate_mode),
            "branch_top_k": int(args.branch_top_k),
            "branch_budget": int(args.branch_budget),
            "branch_lambda_rank": float(args.branch_lambda_rank),
            "branch_lambda_vpmi": float(args.branch_lambda_vpmi),
            "select_w_suffix": float(args.select_w_suffix),
            "select_w_rank": float(args.select_w_rank),
            "switch_margin": float(args.switch_margin),
        },
        "counts": {
            "n_total": int(len(rows)),
            "n_valid": int(n),
            "n_error": int(len(rows) - n),
            "base_accuracy": base_acc,
            "final_accuracy": final_acc,
            "delta_accuracy": (None if base_acc is None or final_acc is None else float(final_acc - base_acc)),
            "risk_count": int(risk_n),
            "risk_rate": (None if n == 0 else float(risk_n / n)),
            "switched": int(sw),
            "switch_rate": (None if n == 0 else float(sw / n)),
            "gain": int(gain),
            "harm": int(harm),
            "net": int(gain - harm),
            "precision_gain": (None if sw == 0 else float(gain / sw)),
            "mean_latency_sec_per_sample": mean_latency,
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

