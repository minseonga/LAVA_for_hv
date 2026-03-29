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


def mean_or_none(xs: List[float]) -> Optional[float]:
    vals = [float(v) for v in xs if safe_float(v) is not None]
    if len(vals) == 0:
        return None
    return float(sum(vals) / len(vals))


def quantile(vals: List[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if safe_float(v) is not None)
    if len(xs) == 0:
        return None
    if len(xs) == 1:
        return float(xs[0])
    p = min(1.0, max(0.0, float(q))) * (len(xs) - 1)
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    if lo == hi:
        return float(xs[lo])
    w = p - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def interpolate_curve(vals: List[float], n_bins: int) -> List[float]:
    m = len(vals)
    if m <= 0:
        return [0.0 for _ in range(n_bins)]
    if m == 1:
        return [float(vals[0]) for _ in range(n_bins)]
    out: List[float] = []
    for b in range(n_bins):
        pos = float(b / max(1, n_bins - 1))
        t = pos * float(m - 1)
        lo = int(math.floor(t))
        hi = int(math.ceil(t))
        if lo == hi:
            out.append(float(vals[lo]))
        else:
            w = float(t - lo)
            out.append(float((1.0 - w) * vals[lo] + w * vals[hi]))
    return out


def svg_line_chart(
    out_path: str,
    y_success: List[Optional[float]],
    y_failure: List[Optional[float]],
    title: str,
    subtitle: str,
    y_label: str,
) -> None:
    w, h = 1080, 640
    ml, mr, mt, mb = 86, 40, 92, 72
    pw = w - ml - mr
    ph = h - mt - mb

    vals = [float(v) for v in (y_success + y_failure) if v is not None and safe_float(v) is not None]
    if len(vals) == 0:
        vals = [-1.0, 1.0]
    y_min = float(min(vals))
    y_max = float(max(vals))
    if y_max <= y_min + 1e-12:
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    n = int(max(1, len(y_success)))

    def x_of(i: int) -> float:
        if n <= 1:
            return float(ml + pw / 2.0)
        return float(ml + (float(i) / float(n - 1)) * pw)

    def y_of(v: float) -> float:
        return float(mt + (1.0 - (float(v) - y_min) / (y_max - y_min)) * ph)

    def poly(ys: List[Optional[float]]) -> str:
        pts: List[str] = []
        for i, v in enumerate(ys):
            if v is None:
                continue
            pts.append(f"{x_of(i):.2f},{y_of(float(v)):.2f}")
        return " ".join(pts)

    p_s = poly(y_success)
    p_f = poly(y_failure)

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{ml}" y="38" font-size="24" font-family="Arial" fill="#111">{title}</text>')
    lines.append(f'<text x="{ml}" y="63" font-size="15" font-family="Arial" fill="#444">{subtitle}</text>')

    for k in range(6):
        yy = mt + ph * (k / 5.0)
        yv = y_max - (y_max - y_min) * (k / 5.0)
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml + pw}" y2="{yy:.2f}" stroke="#e9e9e9" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">{yv:.3f}</text>')

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#222" stroke-width="1.5"/>')
    lines.append(f'<text x="{ml + pw / 2:.2f}" y="{h - 22}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">normalized generation step (prefix → suffix)</text>')
    lines.append(f'<text x="22" y="{mt + ph / 2:.2f}" transform="rotate(-90 22 {mt + ph / 2:.2f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">{y_label}</text>')

    for k in range(6):
        i = int(round((n - 1) * (k / 5.0)))
        xx = x_of(i)
        lines.append(f'<line x1="{xx:.2f}" y1="{mt + ph}" x2="{xx:.2f}" y2="{mt + ph + 6}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{mt + ph + 24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#666">{k/5.0:.1f}</text>')

    if p_s:
        lines.append(f'<polyline points="{p_s}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    if p_f:
        lines.append(f'<polyline points="{p_f}" fill="none" stroke="#d62728" stroke-width="3"/>')

    lx, ly = ml + 20, mt + 20
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 36}" y2="{ly}" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 44}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#1f77b4">correct</text>')
    lines.append(f'<line x1="{lx + 130}" y1="{ly}" x2="{lx + 166}" y2="{ly}" stroke="#d62728" stroke-width="3"/>')
    lines.append(f'<text x="{lx + 174}" y="{ly + 5}" font-size="13" font-family="Arial" fill="#d62728">fail</text>')

    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@torch.no_grad()
def greedy_observe_one_sample(
    *,
    model,
    tokenizer,
    input_ids_img: torch.Tensor,
    input_ids_q: torch.Tensor,
    images_tensor: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    eos_id: Optional[int],
    max_new_tokens: int,
    rank_top_k: int,
    save_topk_json: bool,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    # Warm start: first next-token logits from full prompts.
    out_img = model(
        input_ids=input_ids_img,
        images=images_tensor,
        image_sizes=image_sizes,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    out_q = model(
        input_ids=input_ids_q,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past_img = out_img.past_key_values
    past_q = out_q.past_key_values
    logits_img = out_img.logits[:, -1, :].float()
    logits_q = out_q.logits[:, -1, :].float()

    gen_ids: List[int] = []
    token_logs: List[Dict[str, Any]] = []

    for step_idx in range(int(max_new_tokens)):
        li = logits_img[0]
        lq = logits_q[0]
        next_id = int(torch.argmax(li, dim=-1).item())
        if eos_id is not None and int(next_id) == int(eos_id):
            break

        logp_img = F.log_softmax(li, dim=-1)
        logp_q = F.log_softmax(lq, dim=-1)
        sel_logit_img = float(li[next_id].item())
        sel_logit_q = float(lq[next_id].item())
        sel_logp_img = float(logp_img[next_id].item())
        sel_logp_q = float(logp_q[next_id].item())
        vpmi_logit = float(sel_logit_img - sel_logit_q)
        vpmi_logp = float(sel_logp_img - sel_logp_q)

        k = int(min(max(1, rank_top_k), int(li.numel())))
        top_li_vals, top_li_ids = torch.topk(li, k=k, dim=-1)
        top_vpmi = (li[top_li_ids] - lq[top_li_ids]).detach()
        order = torch.argsort(top_vpmi, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(k, device=order.device)
        pos = (top_li_ids == next_id).nonzero(as_tuple=False)
        if int(pos.numel()) == 0:
            vpmi_rank_topk = int(k + 1)
            vpmi_rank_pct = 0.0
        else:
            rp0 = int(pos[0].item())
            vpmi_rank_topk = int(ranks[rp0].item() + 1)  # 1=best
            vpmi_rank_pct = float((k - vpmi_rank_topk) / max(1, k - 1))

        top2_vals = torch.topk(logp_img, k=2, dim=-1).values
        margin_top1_top2_logp = float((top2_vals[0] - top2_vals[1]).item())

        tok_str = str(tokenizer.convert_ids_to_tokens([next_id])[0])
        rec: Dict[str, Any] = {
            "step_idx": int(step_idx),
            "token_id": int(next_id),
            "token_str": tok_str,
            "logit_vq": sel_logit_img,
            "logit_q": sel_logit_q,
            "vpmi_logit": vpmi_logit,
            "logp_vq": sel_logp_img,
            "logp_q": sel_logp_q,
            "vpmi_logp": vpmi_logp,
            "vpmi_rank_topk": int(vpmi_rank_topk),
            "vpmi_rank_pct_topk": float(vpmi_rank_pct),
            "top1_top2_margin_logp_vq": margin_top1_top2_logp,
        }
        if bool(save_topk_json):
            rec["topk_ids_json"] = json.dumps([int(x) for x in top_li_ids.tolist()], ensure_ascii=False)
            rec["topk_tokens_json"] = json.dumps([str(tokenizer.convert_ids_to_tokens([int(x)])[0]) for x in top_li_ids.tolist()], ensure_ascii=False)
            rec["topk_vpmi_logit_json"] = json.dumps([float(x) for x in top_vpmi.tolist()], ensure_ascii=False)
        token_logs.append(rec)
        gen_ids.append(int(next_id))

        step = torch.tensor([[int(next_id)]], dtype=torch.long, device=input_ids_img.device)
        out_img = model(
            input_ids=step,
            past_key_values=past_img,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        out_q = model(
            input_ids=step,
            past_key_values=past_q,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_img = out_img.past_key_values
        past_q = out_q.past_key_values
        logits_img = out_img.logits[:, -1, :].float()
        logits_q = out_q.logits[:, -1, :].float()

    return [int(x) for x in gen_ids], token_logs


def main() -> None:
    ap = argparse.ArgumentParser(description="Pure observational profiling for greedy V-PMI trajectories.")
    ap.add_argument("--questions_json", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--rank_top_k", type=int, default=50)
    ap.add_argument("--curve_bins", type=int, default=20)
    ap.add_argument("--save_topk_json", action="store_true")
    ap.add_argument("--eval_match_mode", type=str, default="heuristic", choices=["strict", "heuristic"])

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
    per_token: List[Dict[str, Any]] = []
    by_label_vpmi: Dict[str, List[List[float]]] = {"success": [], "failure": []}
    by_label_rank: Dict[str, List[List[float]]] = {"success": [], "failure": []}
    t0 = time.time()

    pbar = tqdm(rows, total=len(rows), desc="greedy-observe", dynamic_ncols=True)
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

            gen_ids, tok_logs = greedy_observe_one_sample(
                model=model,
                tokenizer=tokenizer,
                input_ids_img=input_ids_img,
                input_ids_q=input_ids_q,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
                eos_id=eos_id,
                max_new_tokens=int(args.max_new_tokens),
                rank_top_k=int(args.rank_top_k),
                save_topk_json=bool(args.save_topk_json),
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
        label = "success" if ok else "failure"

        # Attach sample metadata to per-token rows.
        vpmi_seq: List[float] = []
        rank_seq: List[float] = []
        for tr in tok_logs:
            tr2 = dict(tr)
            tr2["id"] = qid
            tr2["image_id"] = image_id
            tr2["question"] = question
            tr2["answer"] = answer
            tr2["is_success"] = bool(ok)
            tr2["label"] = label
            per_token.append(tr2)
            vp = safe_float(tr.get("vpmi_logit"))
            rk = safe_float(tr.get("vpmi_rank_pct_topk"))
            if vp is not None:
                vpmi_seq.append(float(vp))
            if rk is not None:
                rank_seq.append(float(rk))

        by_label_vpmi[label].append([float(x) for x in vpmi_seq])
        by_label_rank[label].append([float(x) for x in rank_seq])

        k40 = int(max(1, math.ceil(0.4 * max(1, len(vpmi_seq)))))
        pref = vpmi_seq[:k40] if len(vpmi_seq) > 0 else []
        suff = vpmi_seq[len(vpmi_seq) - k40 :] if len(vpmi_seq) > 0 else []
        vpmi_prefix_mean = mean_or_none(pref)
        vpmi_suffix_min = (None if len(suff) == 0 else float(min(suff)))
        vpmi_collapse_gap = (
            None
            if vpmi_prefix_mean is None or vpmi_suffix_min is None
            else float(vpmi_suffix_min - vpmi_prefix_mean)
        )

        per_sample.append(
            {
                "id": qid,
                "image_id": image_id,
                "question": question,
                "answer": answer,
                "pred_text": text,
                "pred_answer_eval": pred_answer,
                "is_success": bool(ok),
                "is_success_strict": bool(ok_strict),
                "is_success_heuristic": bool(ok_heur),
                "label": label,
                "n_gen_tokens": int(len(gen_ids)),
                "vpmi_logit_mean": mean_or_none(vpmi_seq),
                "vpmi_logit_min": (None if len(vpmi_seq) == 0 else float(min(vpmi_seq))),
                "vpmi_logit_max": (None if len(vpmi_seq) == 0 else float(max(vpmi_seq))),
                "vpmi_prefix_mean_k40": vpmi_prefix_mean,
                "vpmi_suffix_min_k40": vpmi_suffix_min,
                "vpmi_collapse_gap_k40": vpmi_collapse_gap,
                "rank_pct_mean": mean_or_none(rank_seq),
                "rank_pct_min": (None if len(rank_seq) == 0 else float(min(rank_seq))),
                "elapsed_sec_sample": float(time.time() - ts),
                "error": None,
            }
        )

        if hasattr(pbar, "set_postfix"):
            done = int(i + 1)
            elapsed = float(time.time() - t0)
            avg = float(elapsed / max(1, done))
            ok_cnt = int(sum(1 for x in per_sample if x.get("error") is None and bool(x.get("is_success", False))))
            pbar.set_postfix(avg_s=f"{avg:.2f}", ok=ok_cnt)

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

    # Aggregate correct/fail trajectories.
    bins = int(max(5, args.curve_bins))
    vpmi_curve_rows: List[Dict[str, Any]] = []
    rank_curve_rows: List[Dict[str, Any]] = []
    s_v = [interpolate_curve(v, bins) for v in by_label_vpmi["success"] if len(v) > 0]
    f_v = [interpolate_curve(v, bins) for v in by_label_vpmi["failure"] if len(v) > 0]
    s_r = [interpolate_curve(v, bins) for v in by_label_rank["success"] if len(v) > 0]
    f_r = [interpolate_curve(v, bins) for v in by_label_rank["failure"] if len(v) > 0]

    vpmi_s_means: List[Optional[float]] = []
    vpmi_f_means: List[Optional[float]] = []
    rank_s_means: List[Optional[float]] = []
    rank_f_means: List[Optional[float]] = []
    for b in range(bins):
        sv = [float(x[b]) for x in s_v]
        fv = [float(x[b]) for x in f_v]
        sr = [float(x[b]) for x in s_r]
        fr = [float(x[b]) for x in f_r]
        msv = mean_or_none(sv)
        mfv = mean_or_none(fv)
        msr = mean_or_none(sr)
        mfr = mean_or_none(fr)
        vpmi_s_means.append(msv)
        vpmi_f_means.append(mfv)
        rank_s_means.append(msr)
        rank_f_means.append(mfr)

        vpmi_curve_rows.append(
            {
                "bin_idx": int(b),
                "pos_norm": float(b / max(1, bins - 1)),
                "vpmi_logit_mean_success": msv,
                "vpmi_logit_mean_failure": mfv,
                "vpmi_logit_diff_success_minus_failure": (
                    None if msv is None or mfv is None else float(msv - mfv)
                ),
                "n_success": int(len(s_v)),
                "n_failure": int(len(f_v)),
            }
        )
        rank_curve_rows.append(
            {
                "bin_idx": int(b),
                "pos_norm": float(b / max(1, bins - 1)),
                "rank_pct_mean_success": msr,
                "rank_pct_mean_failure": mfr,
                "rank_pct_diff_success_minus_failure": (
                    None if msr is None or mfr is None else float(msr - mfr)
                ),
                "n_success": int(len(s_r)),
                "n_failure": int(len(f_r)),
            }
        )

    write_csv(os.path.join(out_dir, "per_sample.csv"), per_sample)
    write_csv(os.path.join(out_dir, "per_token.csv"), per_token)
    write_csv(os.path.join(out_dir, "curve_vpmi_logit_correct_vs_fail.csv"), vpmi_curve_rows)
    write_csv(os.path.join(out_dir, "curve_rankpct_correct_vs_fail.csv"), rank_curve_rows)

    svg_line_chart(
        out_path=os.path.join(out_dir, "curve_vpmi_logit_correct_vs_fail.svg"),
        y_success=vpmi_s_means,
        y_failure=vpmi_f_means,
        title="Greedy V-PMI Trajectory (Pure Observational)",
        subtitle="selected token per step | V-PMI = logit(V+Q)-logit(Q)",
        y_label="selected-token V-PMI (logit diff)",
    )
    svg_line_chart(
        out_path=os.path.join(out_dir, "curve_rankpct_correct_vs_fail.svg"),
        y_success=rank_s_means,
        y_failure=rank_f_means,
        title="Greedy V-PMI Rank Trajectory (Pure Observational)",
        subtitle=f"selected token V-PMI rank inside top-{int(args.rank_top_k)} by image logits",
        y_label="selected-token V-PMI rank percentile",
    )

    summary = {
        "inputs": {
            "questions_json": qpath,
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "max_new_tokens": int(args.max_new_tokens),
            "rank_top_k": int(args.rank_top_k),
            "curve_bins": int(bins),
            "eval_match_mode": str(args.eval_match_mode),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "attn_impl": str(args.attn_impl),
            "use_flash_attn": bool(args.use_flash_attn),
            "save_topk_json": bool(args.save_topk_json),
        },
        "counts": {
            "n_total": int(len(rows)),
            "n_valid": int(n_valid),
            "n_error": int(len(rows) - n_valid),
            "n_success": int(sum(1 for r in valid if bool(r.get("is_success", False)))),
            "n_failure": int(sum(1 for r in valid if not bool(r.get("is_success", False)))),
            "accuracy": acc,
            "accuracy_strict": acc_s,
            "accuracy_heuristic": acc_h,
            "mean_latency_sec_per_sample": mean_or_none([float(r.get("elapsed_sec_sample", 0.0) or 0.0) for r in valid]),
            "mean_generated_tokens": mean_or_none([float(r.get("n_gen_tokens", 0.0) or 0.0) for r in valid]),
            "total_token_rows": int(len(per_token)),
        },
        "distribution": {
            "vpmi_logit_mean_success": mean_or_none([float(r.get("vpmi_logit_mean")) for r in valid if bool(r.get("is_success", False)) and safe_float(r.get("vpmi_logit_mean")) is not None]),
            "vpmi_logit_mean_failure": mean_or_none([float(r.get("vpmi_logit_mean")) for r in valid if not bool(r.get("is_success", False)) and safe_float(r.get("vpmi_logit_mean")) is not None]),
            "vpmi_collapse_gap_k40_success_median": quantile([float(r.get("vpmi_collapse_gap_k40")) for r in valid if bool(r.get("is_success", False)) and safe_float(r.get("vpmi_collapse_gap_k40")) is not None], 0.5),
            "vpmi_collapse_gap_k40_failure_median": quantile([float(r.get("vpmi_collapse_gap_k40")) for r in valid if not bool(r.get("is_success", False)) and safe_float(r.get("vpmi_collapse_gap_k40")) is not None], 0.5),
        },
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample.csv"),
            "per_token_csv": os.path.join(out_dir, "per_token.csv"),
            "curve_vpmi_csv": os.path.join(out_dir, "curve_vpmi_logit_correct_vs_fail.csv"),
            "curve_rank_csv": os.path.join(out_dir, "curve_rankpct_correct_vs_fail.csv"),
            "curve_vpmi_svg": os.path.join(out_dir, "curve_vpmi_logit_correct_vs_fail.svg"),
            "curve_rank_svg": os.path.join(out_dir, "curve_rankpct_correct_vs_fail.svg"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample.csv"))
    print("[saved]", os.path.join(out_dir, "per_token.csv"))
    print("[saved]", os.path.join(out_dir, "curve_vpmi_logit_correct_vs_fail.csv"))
    print("[saved]", os.path.join(out_dir, "curve_rankpct_correct_vs_fail.csv"))
    print("[saved]", os.path.join(out_dir, "curve_vpmi_logit_correct_vs_fail.svg"))
    print("[saved]", os.path.join(out_dir, "curve_rankpct_correct_vs_fail.svg"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
