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


def percentile_rank_desc_map(cands: List[Dict[str, Any]], key: str) -> Dict[int, Optional[float]]:
    vals: List[Tuple[int, float]] = []
    for c in cands:
        idx = int(c["cand_idx"])
        v = safe_float(c.get(key))
        if v is None:
            continue
        vals.append((idx, float(v)))
    vals = sorted(vals, key=lambda x: float(x[1]), reverse=True)
    out: Dict[int, Optional[float]] = {int(c["cand_idx"]): None for c in cands}
    n = len(vals)
    if n == 0:
        return out
    if n == 1:
        out[int(vals[0][0])] = 1.0
        return out
    for r, (idx, _) in enumerate(vals):
        out[int(idx)] = float((n - 1 - r) / (n - 1))
    return out


def score_candidates(
    *,
    model,
    tokenizer,
    question: str,
    answer: str,
    cand_rows: List[pf.GeneratedCandidate],
    input_ids_img: torch.Tensor,
    input_ids_q: torch.Tensor,
    images_tensor: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    phrase_id_map: List[Tuple[str, List[int]]],
    answer_span_max_tokens: int,
    pad_id: int,
    k_prefix: int,
    k_suffix: int,
) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    core_ids_for_sq: List[List[int]] = []
    core_img_for_sq: List[List[float]] = []
    core_idxs_for_sq: List[List[int]] = []

    for ci, gen_c in enumerate(cand_rows):
        cand_ids = [int(x) for x in gen_c.token_ids]
        token_logps_img = gen_c.token_logps_img
        if token_logps_img is None or len(token_logps_img) != len(cand_ids):
            token_logps_img = pf.sequence_token_logps(
                model=model,
                prefix_ids=input_ids_img,
                cont_ids=cand_ids,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
            )
        if token_logps_img is None or len(token_logps_img) == 0:
            continue

        text = tokenizer.decode(cand_ids, skip_special_tokens=True).strip()
        short_answer = pf.extract_core_answer_text(question=question, text=text, max_words=6)

        core_start_aligned, core_match_ids = pf.locate_core_span_from_text(
            tokenizer=tokenizer,
            cand_ids=[int(x) for x in cand_ids],
            core_text=short_answer,
        )
        if core_start_aligned is not None and len(core_match_ids) > 0:
            core_idxs = list(range(int(core_start_aligned), int(core_start_aligned) + int(len(core_match_ids))))
            format_len = int(max(0, int(core_start_aligned)))
        else:
            format_len_phrase, _ = pf.detect_format_len(tokenizer, cand_ids, phrase_id_map)
            format_len_prefix = pf.infer_prefix_format_len_from_tokens(tokenizer, cand_ids, question)
            format_len = int(max(int(format_len_phrase), int(format_len_prefix)))
            core_start = int(min(format_len, max(0, len(cand_ids) - 1)))
            span_len = int(answer_span_max_tokens) if int(answer_span_max_tokens) > 0 else 1
            span_len = max(1, span_len)
            core_idxs = list(range(core_start, min(len(cand_ids), core_start + span_len)))

        if int(answer_span_max_tokens) > 0 and len(core_idxs) > int(answer_span_max_tokens):
            core_idxs = core_idxs[: int(answer_span_max_tokens)]
        if len(core_idxs) == 0 and len(cand_ids) > 0:
            core_idxs = [0]

        core_ids = [int(cand_ids[j]) for j in core_idxs if 0 <= int(j) < len(cand_ids)]
        if len(core_ids) == 0 and len(cand_ids) > 0:
            core_ids = [int(cand_ids[0])]
            core_idxs = [0]
        core_img = [float(token_logps_img[j]) for j in core_idxs if 0 <= int(j) < len(token_logps_img)]

        cands.append(
            {
                "cand_idx": int(ci),
                "token_ids": [int(x) for x in cand_ids],
                "text": text,
                "short_answer": short_answer,
                "s_full": pf.mean_selected(token_logps_img, list(range(len(token_logps_img)))),
                "s_format_img": (
                    pf.mean_selected(token_logps_img, list(range(min(int(format_len), len(token_logps_img)))))
                    if int(format_len) > 0
                    else None
                ),
                "s_core_img": pf.mean_selected(token_logps_img, core_idxs),
                "core_img_toks": [float(x) for x in core_img],
                "core_idxs": [int(x) for x in core_idxs],
                "core_ids": [int(x) for x in core_ids],
            }
        )
        core_ids_for_sq.append([int(x) for x in core_ids])
        core_img_for_sq.append([float(x) for x in core_img])
        core_idxs_for_sq.append([int(x) for x in core_idxs])

    if len(cands) == 0:
        return []

    token_logps_q_rows = pf.sequence_token_logps_batched(
        model=model,
        prefix_ids=input_ids_q,
        cont_ids_list=core_ids_for_sq,
        pad_token_id=pad_id,
        images_tensor=None,
        image_sizes=None,
    )
    token_margin_rows = pf.sequence_token_top2_margin_batched(
        model=model,
        prefix_ids=input_ids_img,
        cont_ids_list=[[int(x) for x in c["token_ids"]] for c in cands],
        pad_token_id=pad_id,
        images_tensor=images_tensor,
        image_sizes=image_sizes,
    )

    for ci, c in enumerate(cands):
        q_toks = token_logps_q_rows[ci] if ci < len(token_logps_q_rows) else None
        c["s_ans_q"] = pf.mean_selected(q_toks, list(range(len(q_toks or []))))
        img_toks = core_img_for_sq[ci] if ci < len(core_img_for_sq) else []
        q2 = (q_toks or [])
        m = int(min(len(img_toks), len(q2)))
        if m > 0:
            vp = [float(img_toks[k]) - float(q2[k]) for k in range(m)]
            c["core_vpmi_toks"] = [float(x) for x in vp]
            c["vpmi_core_mean"] = float(sum(vp) / len(vp))
            pref = vp[: max(1, min(int(k_prefix), len(vp)))]
            suf = vp[max(0, len(vp) - max(1, min(int(k_suffix), len(vp)))):]
            c["vpmi_prefix_max_k"] = None if len(pref) == 0 else float(max(pref))
            c["vpmi_prefix_mean_k"] = None if len(pref) == 0 else float(sum(pref) / len(pref))
            c["vpmi_suffix_min_k"] = None if len(suf) == 0 else float(min(suf))
        else:
            c["core_vpmi_toks"] = []
            c["vpmi_core_mean"] = None
            c["vpmi_prefix_max_k"] = None
            c["vpmi_prefix_mean_k"] = None
            c["vpmi_suffix_min_k"] = None

        mm = token_margin_rows[ci] if ci < len(token_margin_rows) else None
        if mm is not None:
            idxs = core_idxs_for_sq[ci] if ci < len(core_idxs_for_sq) else []
            margins = [float(mm[k]) for k in idxs if 0 <= int(k) < len(mm)]
            c["margin_core_img_min"] = None if len(margins) == 0 else float(min(margins))
        else:
            c["margin_core_img_min"] = None

        pred_ans = pf.first_clause(c["short_answer"] if c["short_answer"] else c["text"])
        ok_strict = bool(pf.norm_text(pred_ans) == pf.norm_text(answer))
        ok_heur = bool(
            pf.is_success_heuristic(
                question=question,
                answer=answer,
                champ_text=c["text"],
                champ_short=c["short_answer"],
            )
        )
        c["is_correct_strict"] = bool(ok_strict)
        c["is_correct_heuristic"] = bool(ok_heur)

    return cands


def select_with_features(
    *,
    base: Dict[str, Any],
    pool: List[Dict[str, Any]],
    tau_rank_gap: float,
    tau_suffix_gap: float,
    tau_prefix_max: float,
    max_sfull_drop: float,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if len(pool) == 0:
        return base, None
    rank_v = percentile_rank_desc_map(pool, "vpmi_core_mean")
    rank_s = percentile_rank_desc_map(pool, "vpmi_suffix_min_k")

    base_idx = int(base["cand_idx"])
    base_rank_v = rank_v.get(base_idx)
    base_suffix = safe_float(base.get("vpmi_suffix_min_k"))
    base_sfull = safe_float(base.get("s_full"))

    elig: List[Dict[str, Any]] = []
    for c in pool:
        if int(c["cand_idx"]) == base_idx:
            continue
        rv = rank_v.get(int(c["cand_idx"]))
        if rv is None or base_rank_v is None:
            continue
        d_rank = float(rv - base_rank_v)
        c_suffix = safe_float(c.get("vpmi_suffix_min_k"))
        if c_suffix is None or base_suffix is None:
            continue
        d_suffix = float(c_suffix - base_suffix)
        pref_max = safe_float(c.get("vpmi_prefix_max_k"))
        if pref_max is None:
            continue
        c_sfull = safe_float(c.get("s_full"))
        if c_sfull is None or base_sfull is None:
            continue
        if d_rank < float(tau_rank_gap):
            continue
        if d_suffix < float(tau_suffix_gap):
            continue
        if pref_max > float(tau_prefix_max):
            continue
        if c_sfull < float(base_sfull - float(max_sfull_drop)):
            continue
        cc = dict(c)
        cc["_d_rank"] = float(d_rank)
        cc["_d_suffix"] = float(d_suffix)
        cc["_score"] = float(d_rank + 0.3 * d_suffix)
        elig.append(cc)

    if len(elig) == 0:
        return base, None
    best = max(elig, key=lambda x: float(x["_score"]))
    # final safeguard: challenger should not be much worse on vpmi mean.
    bvp = safe_float(base.get("vpmi_core_mean"))
    cvp = safe_float(best.get("vpmi_core_mean"))
    if bvp is not None and cvp is not None and cvp < bvp:
        return base, None
    return best, best


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy + gated micro-branch + feature selector")
    ap.add_argument("--questions_json", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--answer_span_max_tokens", type=int, default=4)
    ap.add_argument("--eval_match_mode", type=str, default="heuristic", choices=["strict", "heuristic"])
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # Gate: run micro-branch only when greedy looks fragile.
    ap.add_argument("--gate_tau_margin", type=float, default=1.0, help="expand when greedy margin_core_img_min <= tau")
    ap.add_argument("--gate_tau_suffix", type=float, default=-0.3, help="expand when greedy vpmi_suffix_min_k <= tau")
    ap.add_argument("--gate_mode", type=str, default="and", choices=["and", "or"])

    # Micro-branch budget
    ap.add_argument("--micro_width", type=int, default=3)
    ap.add_argument("--micro_budget", type=int, default=3)

    # Selector thresholds
    ap.add_argument("--tau_rank_gap", type=float, default=0.2)
    ap.add_argument("--tau_suffix_gap", type=float, default=0.0)
    ap.add_argument("--tau_prefix_max", type=float, default=2.0)
    ap.add_argument("--max_sfull_drop", type=float, default=0.2)
    ap.add_argument("--k_prefix", type=int, default=2)
    ap.add_argument("--k_suffix", type=int, default=2)
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

    # Fill globals expected by helper functions in pf.
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
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None or int(pad_id) < 0:
        pad_id = eos_id if eos_id is not None else 0
    phrase_id_map = pf.build_phrase_id_map(tokenizer, pf.FORMAT_PHRASES)

    per_sample: List[Dict[str, Any]] = []
    t0 = time.time()
    pbar = tqdm(rows, total=len(rows), desc="adaptive-microbranch", dynamic_ncols=True)
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

        # 1) greedy(B1)
        try:
            greedy_rows = pf.generate_candidates(
                model=model,
                tokenizer=tokenizer,
                input_ids_img=input_ids_img,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
                question=q,
                phrase_id_map=phrase_id_map,
                candidate_gen_mode="beam",
                num_beams=1,
                num_beam_groups=1,
                diversity_penalty=0.0,
                num_return_sequences=1,
                anchor_depth=2,
                anchor_width=2,
                anchor_prefix_cap=2,
                anchor_completion_budget=2,
                num_extra_samples=0,
                extra_sample_temperature=1.0,
                extra_sample_top_p=0.9,
                extra_sample_top_k=0,
                max_new_tokens=int(args.max_new_tokens),
                eos_token_id=eos_id,
            )
        except Exception as e:
            per_sample.append({"id": sid, "error": f"greedy_generate:{e}"})
            continue

        greedy_pool = score_candidates(
            model=model,
            tokenizer=tokenizer,
            question=q,
            answer=a,
            cand_rows=greedy_rows,
            input_ids_img=input_ids_img,
            input_ids_q=input_ids_q,
            images_tensor=images_tensor,
            image_sizes=image_sizes,
            phrase_id_map=phrase_id_map,
            answer_span_max_tokens=int(args.answer_span_max_tokens),
            pad_id=int(pad_id),
            k_prefix=int(args.k_prefix),
            k_suffix=int(args.k_suffix),
        )
        if len(greedy_pool) == 0:
            per_sample.append({"id": sid, "error": "no_greedy_scored"})
            continue
        base = max(greedy_pool, key=lambda x: float(safe_float(x.get("s_full")) or -1e9))

        g_margin = safe_float(base.get("margin_core_img_min"))
        g_suffix = safe_float(base.get("vpmi_suffix_min_k"))
        cond_margin = bool(g_margin is not None and float(g_margin) <= float(args.gate_tau_margin))
        cond_suffix = bool(g_suffix is not None and float(g_suffix) <= float(args.gate_tau_suffix))
        gate = bool((cond_margin and cond_suffix) if str(args.gate_mode) == "and" else (cond_margin or cond_suffix))

        pool = list(greedy_pool)
        challenger = None
        if gate:
            try:
                micro_rows = pf.generate_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids_img=input_ids_img,
                    images_tensor=images_tensor,
                    image_sizes=image_sizes,
                    question=q,
                    phrase_id_map=phrase_id_map,
                    candidate_gen_mode="core_anchor",
                    num_beams=1,
                    num_beam_groups=1,
                    diversity_penalty=0.0,
                    num_return_sequences=1,
                    anchor_depth=2,
                    anchor_width=int(max(1, args.micro_width)),
                    anchor_prefix_cap=int(max(1, args.micro_budget)),
                    anchor_completion_budget=int(max(1, args.micro_budget)),
                    num_extra_samples=0,
                    extra_sample_temperature=1.0,
                    extra_sample_top_p=0.9,
                    extra_sample_top_k=0,
                    max_new_tokens=int(args.max_new_tokens),
                    eos_token_id=eos_id,
                )
                micro_pool = score_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    question=q,
                    answer=a,
                    cand_rows=micro_rows,
                    input_ids_img=input_ids_img,
                    input_ids_q=input_ids_q,
                    images_tensor=images_tensor,
                    image_sizes=image_sizes,
                    phrase_id_map=phrase_id_map,
                    answer_span_max_tokens=int(args.answer_span_max_tokens),
                    pad_id=int(pad_id),
                    k_prefix=int(args.k_prefix),
                    k_suffix=int(args.k_suffix),
                )
                if len(micro_pool) > 0:
                    # De-dup by token sequence.
                    seen = set()
                    merged = []
                    for c in (pool + micro_pool):
                        key = tuple(int(x) for x in c.get("token_ids", []))
                        if len(key) == 0 or key in seen:
                            continue
                        seen.add(key)
                        merged.append(c)
                    pool = merged
            except Exception:
                pass

        final, challenger = select_with_features(
            base=base,
            pool=pool,
            tau_rank_gap=float(args.tau_rank_gap),
            tau_suffix_gap=float(args.tau_suffix_gap),
            tau_prefix_max=float(args.tau_prefix_max),
            max_sfull_drop=float(args.max_sfull_drop),
        )

        base_ok = bool(
            base["is_correct_heuristic"] if str(args.eval_match_mode) == "heuristic" else base["is_correct_strict"]
        )
        final_ok = bool(
            final["is_correct_heuristic"] if str(args.eval_match_mode) == "heuristic" else final["is_correct_strict"]
        )

        per_sample.append(
            {
                "id": sid,
                "image_id": image_id,
                "question": q,
                "answer": a,
                "gate": bool(gate),
                "gate_cond_margin": bool(cond_margin),
                "gate_cond_suffix": bool(cond_suffix),
                "n_pool": int(len(pool)),
                "base_cand_idx": int(base["cand_idx"]),
                "final_cand_idx": int(final["cand_idx"]),
                "switched": bool(int(final["cand_idx"]) != int(base["cand_idx"])),
                "base_text": base.get("text"),
                "final_text": final.get("text"),
                "base_short_answer": base.get("short_answer"),
                "final_short_answer": final.get("short_answer"),
                "base_is_success": bool(base_ok),
                "is_success": bool(final_ok),
                "gain": bool((not base_ok) and final_ok),
                "harm": bool(base_ok and (not final_ok)),
                "base_margin_core_img_min": g_margin,
                "base_vpmi_suffix_min_k": g_suffix,
                "base_vpmi_core_mean": base.get("vpmi_core_mean"),
                "final_vpmi_core_mean": final.get("vpmi_core_mean"),
                "challenger_d_rank": (None if challenger is None else challenger.get("_d_rank")),
                "challenger_d_suffix": (None if challenger is None else challenger.get("_d_suffix")),
                "elapsed_sec_sample": float(time.time() - ts),
                "error": None,
            }
        )

        if hasattr(pbar, "set_postfix"):
            done = int(i + 1)
            elapsed = float(time.time() - t0)
            avg_s = float(elapsed / max(1, done))
            expanded = int(sum(1 for x in per_sample if bool(x.get("gate", False))))
            switched = int(sum(1 for x in per_sample if bool(x.get("switched", False))))
            ok = int(sum(1 for x in per_sample if x.get("error") is None and bool(x.get("is_success", False))))
            pbar.set_postfix(avg_s=f"{avg_s:.2f}", expanded=expanded, switched=switched, ok=ok)

    valid = [r for r in per_sample if r.get("error") is None]
    n = int(len(valid))
    base_acc = (None if n == 0 else float(sum(1 for r in valid if bool(r.get("base_is_success", False))) / n))
    final_acc = (None if n == 0 else float(sum(1 for r in valid if bool(r.get("is_success", False))) / n))
    gain = int(sum(1 for r in valid if bool(r.get("gain", False))))
    harm = int(sum(1 for r in valid if bool(r.get("harm", False))))
    switched = int(sum(1 for r in valid if bool(r.get("switched", False))))
    expanded = int(sum(1 for r in valid if bool(r.get("gate", False))))
    mean_latency = (
        None
        if n == 0
        else float(sum(float(safe_float(r.get("elapsed_sec_sample")) or 0.0) for r in valid) / n)
    )

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
            "answer_span_max_tokens": int(args.answer_span_max_tokens),
            "gate_mode": str(args.gate_mode),
            "gate_tau_margin": float(args.gate_tau_margin),
            "gate_tau_suffix": float(args.gate_tau_suffix),
            "micro_width": int(args.micro_width),
            "micro_budget": int(args.micro_budget),
            "tau_rank_gap": float(args.tau_rank_gap),
            "tau_suffix_gap": float(args.tau_suffix_gap),
            "tau_prefix_max": float(args.tau_prefix_max),
            "max_sfull_drop": float(args.max_sfull_drop),
            "k_prefix": int(args.k_prefix),
            "k_suffix": int(args.k_suffix),
            "attn_impl": str(args.attn_impl),
            "use_flash_attn": bool(args.use_flash_attn),
        },
        "counts": {
            "n_total": int(len(rows)),
            "n_valid": int(n),
            "n_error": int(len(rows) - n),
            "base_accuracy": base_acc,
            "final_accuracy": final_acc,
            "delta_accuracy": (None if base_acc is None or final_acc is None else float(final_acc - base_acc)),
            "expanded": int(expanded),
            "expand_rate": (None if n == 0 else float(expanded / n)),
            "switched": int(switched),
            "switch_rate": (None if n == 0 else float(switched / n)),
            "gain": int(gain),
            "harm": int(harm),
            "net": int(gain - harm),
            "precision_gain": (None if switched == 0 else float(gain / switched)),
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

