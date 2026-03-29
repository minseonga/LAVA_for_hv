#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import analyze_artrap_pairwise_fragility as pf
from analyze_pope_visual_disconnect import (
    choose_cont_ids,
    extract_pope_object,
    find_cont_label_positions,
    locate_phrase_start,
    normalize_yesno,
    parse_bool,
    resolve_image_path,
    safe_float,
    select_object_patch_indices,
)
from eval_pope_objpatch_mask_rerun import (
    build_valid_patch_index_from_rect,
    class_logit_from_ids,
    class_token_ids,
    compute_content_rect_in_model_view,
    make_model_view_image,
    mask_model_view_patches,
    read_csv,
    select_target_ids,
    stable_hash_int,
    write_csv,
)


def parse_groups(s: str) -> List[str]:
    out = []
    for p in str(s or "").split(","):
        x = p.strip()
        if x != "":
            out.append(x)
    return out


def choose_candidate_patches(
    removed: Sequence[int],
    sim_valid_map: Dict[int, float],
    topn: int,
    mode: str,
    hybrid_topm: int,
    seed: int,
) -> List[int]:
    rem = sorted(set(int(x) for x in removed))
    if len(rem) == 0 or int(topn) <= 0:
        return []
    k = int(min(int(topn), len(rem)))
    mm = str(mode).strip().lower()
    if mm == "sim_top":
        rr = sorted(rem, key=lambda p: float(sim_valid_map.get(int(p), -1e9)), reverse=True)
        return rr[:k]
    if mm == "random":
        rng = random.Random(int(seed))
        return rng.sample(rem, k)
    # hybrid
    m = int(min(max(0, int(hybrid_topm)), k))
    rr = sorted(rem, key=lambda p: float(sim_valid_map.get(int(p), -1e9)), reverse=True)
    top_part = rr[:m]
    rest = [x for x in rem if x not in set(top_part)]
    need = int(k - len(top_part))
    if need <= 0:
        return top_part
    if len(rest) <= need:
        rnd_part = rest
    else:
        rng = random.Random(int(seed) + 911)
        rnd_part = rng.sample(rest, need)
    return top_part + rnd_part


def first_step_logits_batch(
    model,
    image_processor,
    model_config,
    prompt_ids: torch.Tensor,
    masked_model_views: Sequence[Image.Image],
    device: torch.device,
    batch_size: int,
) -> List[torch.Tensor]:
    from llava.mm_utils import process_images

    outs: List[torch.Tensor] = []
    bsz = int(max(1, batch_size))
    for i in range(0, len(masked_model_views), bsz):
        chunk = list(masked_model_views[i : i + bsz])
        if len(chunk) == 0:
            continue
        images_tensor = process_images(chunk, image_processor, model_config).to(
            device=model.device,
            dtype=torch.float16,
        )
        image_sizes = [im.size for im in chunk]
        with torch.no_grad():
            out = model(
                input_ids=prompt_ids.repeat(len(chunk), 1),
                images=images_tensor,
                image_sizes=image_sizes,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        logits = out.logits[:, -1, :].float().cpu()  # [B, V]
        for j in range(int(logits.size(0))):
            outs.append(logits[j])
    return outs


def role_label_from_delta(group: str, delta_yes_margin: float, delta_gt_margin: float, eps: float) -> str:
    g = str(group).strip().lower()
    th = float(max(0.0, eps))
    if g == "fp_hall":
        if delta_yes_margin > th:
            return "harmful"
        if delta_yes_margin < -th:
            return "supportive"
        return "neutral"
    if g == "tp_yes":
        if delta_gt_margin > th:
            return "supportive"
        if delta_gt_margin < -th:
            return "harmful"
        return "neutral"
    if delta_gt_margin > th:
        return "supportive"
    if delta_gt_margin < -th:
        return "harmful"
    return "neutral"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast single-patch addback role labeling on POPE (keep-k base).")
    ap.add_argument("--samples_csv", type=str, required=True)
    ap.add_argument("--per_layer_trace_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--target_layer", type=int, default=17)
    ap.add_argument("--target_groups", type=str, default="fp_hall,tp_yes")
    ap.add_argument("--top_n_per_group", type=int, default=100)
    ap.add_argument("--sort_metric", type=str, default="yes_sim_objpatch_max")
    ap.add_argument("--sort_desc", type=parse_bool, default=True)
    ap.add_argument("--keep_k", type=int, default=5)
    ap.add_argument("--candidate_topn", type=int, default=32)
    ap.add_argument("--candidate_mode", type=str, default="hybrid", choices=["sim_top", "random", "hybrid"])
    ap.add_argument("--hybrid_topm", type=int, default=16)
    ap.add_argument("--candidate_pool", type=str, default="valid", choices=["valid", "objpool"])
    ap.add_argument("--object_patch_topk", type=int, default=64)
    ap.add_argument("--exclude_padding_patches", type=parse_bool, default=True)
    ap.add_argument("--mask_mode", type=str, default="black", choices=["black", "gray", "mean"])
    ap.add_argument("--batch_candidates", type=int, default=8)
    ap.add_argument("--role_eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows_trace = read_csv(os.path.abspath(args.per_layer_trace_csv))
    rows_in = read_csv(os.path.abspath(args.samples_csv))
    if len(rows_trace) == 0:
        raise RuntimeError("No rows in per_layer_trace_csv.")
    if len(rows_in) == 0:
        raise RuntimeError("No rows in samples_csv.")

    groups = parse_groups(args.target_groups)
    valid_groups = {"all", "fp_hall", "tp_yes", "tn_no", "fn_miss"}
    groups = [g for g in groups if g in valid_groups]
    if len(groups) == 0:
        raise RuntimeError("No valid target groups.")

    target_ids_ordered: List[Tuple[str, str]] = []  # (group, id)
    for g in groups:
        ids = select_target_ids(
            rows_trace=rows_trace,
            target_layer=int(args.target_layer),
            group=str(g),
            sort_metric=str(args.sort_metric),
            top_n=int(args.top_n_per_group),
            desc=bool(args.sort_desc),
        )
        for sid in ids:
            target_ids_ordered.append((str(g), str(sid)))
    if len(target_ids_ordered) == 0:
        raise RuntimeError("No target ids selected.")

    samples_map: Dict[str, Dict[str, Any]] = {}
    for r in rows_in:
        sid = str(r.get("id") or "").strip()
        if sid != "":
            samples_map[sid] = r

    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IGNORE_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

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
        use_flash_attn=False,
        device_map="auto",
    )
    model.eval()
    conv_mode = pf.resolve_conv_mode(model_name, args.conv_mode)
    device = model.get_model().embed_tokens.weight.device
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    yes_ids = class_token_ids(tokenizer, "yes")
    no_ids = class_token_ids(tokenizer, "no")

    per_patch_rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    skipped = 0

    try:
        from tqdm.auto import tqdm

        iterable = tqdm(target_ids_ordered, total=len(target_ids_ordered), desc="pope-role-fast", dynamic_ncols=True)
    except Exception:
        iterable = target_ids_ordered

    for group, sid in iterable:
        base_row = samples_map.get(str(sid))
        if base_row is None:
            skipped += 1
            continue
        question = str(base_row.get("question") or "")
        answer = str(base_row.get("answer") or "")
        image_id = str(base_row.get("image_id") or base_row.get("imageId") or "")
        base_pred_text = str(
            base_row.get("pred_answer_eval")
            or base_row.get("pred_text")
            or base_row.get("champ_text")
            or ""
        ).strip()
        if question == "" or answer == "" or image_id == "" or base_pred_text == "":
            skipped += 1
            continue

        gt = normalize_yesno(answer)
        base_pred = normalize_yesno(base_pred_text)
        if gt not in {"yes", "no"} or base_pred not in {"yes", "no"}:
            skipped += 1
            continue

        image_path = resolve_image_path(args.image_root, image_id)
        if image_path is None:
            skipped += 1
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            model_view = make_model_view_image(img, image_processor=image_processor, model_config=model.config)

            prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            cont_ids = choose_cont_ids(tokenizer, base_pred_text)
            tlen = int(len(cont_ids))
            if tlen <= 0:
                skipped += 1
                continue
            yesno_idx = locate_phrase_start(tokenizer, cont_ids, base_pred)
            if yesno_idx is None:
                yesno_idx = 0
            yesno_idx = int(max(0, min(tlen - 1, int(yesno_idx))))

            cont_t = torch.tensor([cont_ids], dtype=torch.long, device=device)
            full_ids = torch.cat([prompt_ids, cont_t], dim=1)

            from llava.mm_utils import process_images

            images_tensor = process_images([img], image_processor, model.config).to(
                device=model.device,
                dtype=torch.float16,
            )
            image_sizes = [img.size]
            with torch.no_grad():
                base_attn = torch.ones_like(full_ids, dtype=torch.long, device=device)
                _, pos_ids_e, attn_mask_e, _, mm_embeds_e, labels_e = model.prepare_inputs_labels_for_multimodal(
                    full_ids, None, base_attn, None, full_ids, images_tensor, image_sizes
                )
                labels_exp = labels_e[0]
                cont_label_pos = find_cont_label_positions(labels_exp, cont_ids, int(IGNORE_INDEX))
                if cont_label_pos is None or int(cont_label_pos.numel()) != tlen:
                    skipped += 1
                    continue
                dec_pos = cont_label_pos - 1
                if int(dec_pos.min().item()) < 0:
                    skipped += 1
                    continue
                vision_pos = torch.where(labels_exp == int(IGNORE_INDEX))[0]
                if int(vision_pos.numel()) <= 0:
                    skipped += 1
                    continue

                out = model(
                    inputs_embeds=mm_embeds_e,
                    attention_mask=attn_mask_e,
                    position_ids=pos_ids_e,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs_all = out.hidden_states

            hidx = int(args.target_layer) + 1
            if not (0 <= hidx < len(hs_all)):
                skipped += 1
                continue

            vis = mm_embeds_e[0, vision_pos, :].float().cpu()
            p = int(vis.size(0))
            g = int(round(math.sqrt(float(p))))
            if g * g != p:
                skipped += 1
                continue
            grid_w = g
            grid_h = g

            valid_patch_idx = torch.arange(int(p), dtype=torch.long)
            if bool(args.exclude_padding_patches):
                content_rect = compute_content_rect_in_model_view(
                    orig_w=int(img.size[0]),
                    orig_h=int(img.size[1]),
                    model_w=int(model_view.size[0]),
                    model_h=int(model_view.size[1]),
                    image_aspect_ratio=image_aspect_ratio,
                )
                valid_patch_idx = build_valid_patch_index_from_rect(
                    grid_w=grid_w,
                    grid_h=grid_h,
                    model_w=int(model_view.size[0]),
                    model_h=int(model_view.size[1]),
                    rect_xyxy=content_rect,
                )
            valid_mask = torch.zeros(int(p), dtype=torch.bool)
            valid_mask[valid_patch_idx] = True

            obj_phrase = extract_pope_object(question)
            obj_emb = None
            if obj_phrase != "":
                ids_obj = choose_cont_ids(tokenizer, obj_phrase)
                if len(ids_obj) > 0:
                    ids_obj_t = torch.tensor(ids_obj, dtype=torch.long, device=device)
                    with torch.no_grad():
                        obj_emb = model.get_model().embed_tokens(ids_obj_t).float().mean(dim=0).cpu()
            obj_patch_idx, _ = select_object_patch_indices(
                vis=vis,
                obj_text_emb=obj_emb,
                object_patch_topk=int(args.object_patch_topk),
            )
            if int(obj_patch_idx.numel()) <= 0:
                obj_patch_idx = torch.arange(int(vis.size(0)), dtype=torch.long)
            obj_patch_idx = obj_patch_idx[valid_mask[obj_patch_idx]]
            if int(obj_patch_idx.numel()) <= 0:
                obj_patch_idx = valid_patch_idx

            yes_dec_pos = int(dec_pos[yesno_idx].item())
            hs = hs_all[hidx][0]
            if yes_dec_pos < 0 or yes_dec_pos >= int(hs.size(0)):
                skipped += 1
                continue
            h = hs[yes_dec_pos, :].float().cpu()
            h_n = F.normalize(h, dim=-1)

            vis_obj = vis[obj_patch_idx, :]
            vis_obj_n = F.normalize(vis_obj, dim=-1)
            sim_obj = torch.matmul(vis_obj_n, h_n)
            keep_k = int(min(max(1, int(args.keep_k)), int(sim_obj.numel())))
            keep_vals, keep_idx_local = torch.topk(sim_obj, k=keep_k, dim=-1)
            keep_idx_global = obj_patch_idx[keep_idx_local]
            keep_set = set(int(x) for x in keep_idx_global.tolist())

            if str(args.candidate_pool) == "objpool":
                cand_pool = [int(x) for x in obj_patch_idx.tolist()]
            else:
                cand_pool = [int(x) for x in valid_patch_idx.tolist()]
            removed = [int(x) for x in cand_pool if int(x) not in keep_set]
            if len(removed) <= 0:
                skipped += 1
                continue

            vis_valid = vis[valid_patch_idx, :]
            vis_valid_n = F.normalize(vis_valid, dim=-1)
            sim_valid = torch.matmul(vis_valid_n, h_n)
            valid_list = [int(x) for x in valid_patch_idx.tolist()]
            sim_valid_map = {int(valid_list[i]): float(sim_valid[i].item()) for i in range(len(valid_list))}

            cand_seed = int(args.seed) + 10007 * int(args.target_layer) + stable_hash_int(str(sid))
            cand_list = choose_candidate_patches(
                removed=removed,
                sim_valid_map=sim_valid_map,
                topn=int(args.candidate_topn),
                mode=str(args.candidate_mode),
                hybrid_topm=int(args.hybrid_topm),
                seed=cand_seed,
            )
            if len(cand_list) <= 0:
                skipped += 1
                continue

            keep_base_sorted = sorted(set(int(x) for x in keep_set))
            keep_sets: List[List[int]] = [keep_base_sorted]
            for c in cand_list:
                keep_sets.append(sorted(set(keep_base_sorted + [int(c)])))

            masked_imgs: List[Image.Image] = []
            for ks in keep_sets:
                ks_set = set(int(x) for x in ks)
                mask_idx = [int(x) for x in valid_patch_idx.tolist() if int(x) not in ks_set]
                mi = mask_model_view_patches(
                    img_model_view=model_view,
                    patch_indices=mask_idx,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    mode=str(args.mask_mode),
                )
                masked_imgs.append(mi)

            logits_list = first_step_logits_batch(
                model=model,
                image_processor=image_processor,
                model_config=model.config,
                prompt_ids=prompt_ids,
                masked_model_views=masked_imgs,
                device=device,
                batch_size=int(args.batch_candidates),
            )
            if len(logits_list) != len(masked_imgs):
                skipped += 1
                continue

            logits_keep = logits_list[0]
            yes_keep = class_logit_from_ids(logits_keep, yes_ids)
            no_keep = class_logit_from_ids(logits_keep, no_ids)
            if yes_keep is None or no_keep is None:
                skipped += 1
                continue
            margin_keep = float(yes_keep - no_keep)
            if gt == "yes":
                gt_margin_keep = float(margin_keep)
            else:
                gt_margin_keep = float(no_keep - yes_keep)

            sample_patch_rows = []
            for j, c in enumerate(cand_list):
                logits_c = logits_list[j + 1]
                yes_c = class_logit_from_ids(logits_c, yes_ids)
                no_c = class_logit_from_ids(logits_c, no_ids)
                if yes_c is None or no_c is None:
                    continue
                margin_c = float(yes_c - no_c)
                if gt == "yes":
                    gt_margin_c = float(margin_c)
                else:
                    gt_margin_c = float(no_c - yes_c)
                d_yes = float(margin_c - margin_keep)
                d_gt = float(gt_margin_c - gt_margin_keep)
                role = role_label_from_delta(group=str(group), delta_yes_margin=d_yes, delta_gt_margin=d_gt, eps=float(args.role_eps))
                row = {
                    "group": str(group),
                    "id": sid,
                    "image_id": image_id,
                    "question": question,
                    "object_phrase": obj_phrase,
                    "answer_gt": gt,
                    "base_pred": base_pred,
                    "target_layer": int(args.target_layer),
                    "keep_k": int(keep_k),
                    "candidate_topn": int(len(cand_list)),
                    "candidate_mode": str(args.candidate_mode),
                    "candidate_pool": str(args.candidate_pool),
                    "candidate_rank": int(j),
                    "candidate_patch_idx": int(c),
                    "candidate_patch_sim_valid": float(sim_valid_map.get(int(c), float("nan"))),
                    "keep_base_margin_yes_minus_no": float(margin_keep),
                    "candidate_margin_yes_minus_no": float(margin_c),
                    "delta_yes_minus_no": float(d_yes),
                    "keep_base_gt_margin": float(gt_margin_keep),
                    "candidate_gt_margin": float(gt_margin_c),
                    "delta_gt_margin": float(d_gt),
                    "role_label": str(role),
                }
                sample_patch_rows.append(row)
                per_patch_rows.append(row)

            if len(sample_patch_rows) == 0:
                skipped += 1
                continue
            n_all = int(len(sample_patch_rows))
            n_h = int(sum(1 for r in sample_patch_rows if str(r["role_label"]) == "harmful"))
            n_s = int(sum(1 for r in sample_patch_rows if str(r["role_label"]) == "supportive"))
            n_n = int(sum(1 for r in sample_patch_rows if str(r["role_label"]) == "neutral"))
            del_yes = [float(r["delta_yes_minus_no"]) for r in sample_patch_rows]
            del_gt = [float(r["delta_gt_margin"]) for r in sample_patch_rows]

            per_sample_rows.append(
                {
                    "group": str(group),
                    "id": sid,
                    "image_id": image_id,
                    "answer_gt": gt,
                    "base_pred": base_pred,
                    "n_candidates": int(n_all),
                    "n_harmful": int(n_h),
                    "n_supportive": int(n_s),
                    "n_neutral": int(n_n),
                    "harmful_ratio": float(n_h / n_all),
                    "supportive_ratio": float(n_s / n_all),
                    "mean_delta_yes_minus_no": float(sum(del_yes) / len(del_yes)),
                    "mean_delta_gt_margin": float(sum(del_gt) / len(del_gt)),
                    "max_delta_yes_minus_no": float(max(del_yes)),
                    "min_delta_yes_minus_no": float(min(del_yes)),
                    "max_delta_gt_margin": float(max(del_gt)),
                    "min_delta_gt_margin": float(min(del_gt)),
                }
            )

        except Exception:
            skipped += 1
            continue

    group_rows: List[Dict[str, Any]] = []
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_patch_rows:
        by_group[str(r["group"])].append(r)
    for g, rr in by_group.items():
        n = int(len(rr))
        if n <= 0:
            continue
        n_h = int(sum(1 for r in rr if str(r["role_label"]) == "harmful"))
        n_s = int(sum(1 for r in rr if str(r["role_label"]) == "supportive"))
        n_n = int(sum(1 for r in rr if str(r["role_label"]) == "neutral"))
        d_yes = [float(r["delta_yes_minus_no"]) for r in rr]
        d_gt = [float(r["delta_gt_margin"]) for r in rr]
        group_rows.append(
            {
                "group": g,
                "n_patches": int(n),
                "n_samples": int(len(set(str(r["id"]) for r in rr))),
                "n_harmful": int(n_h),
                "n_supportive": int(n_s),
                "n_neutral": int(n_n),
                "harmful_ratio": float(n_h / n),
                "supportive_ratio": float(n_s / n),
                "mean_delta_yes_minus_no": float(sum(d_yes) / len(d_yes)),
                "mean_delta_gt_margin": float(sum(d_gt) / len(d_gt)),
            }
        )

    out_patch = os.path.join(out_dir, "per_patch_role_effect.csv")
    out_sample = os.path.join(out_dir, "per_sample_role_summary.csv")
    out_group = os.path.join(out_dir, "group_role_summary.csv")
    out_summary = os.path.join(out_dir, "summary.json")
    write_csv(out_patch, per_patch_rows)
    write_csv(out_sample, per_sample_rows)
    write_csv(out_group, group_rows)

    summary = {
        "inputs": {
            "samples_csv": os.path.abspath(args.samples_csv),
            "per_layer_trace_csv": os.path.abspath(args.per_layer_trace_csv),
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "target_layer": int(args.target_layer),
            "target_groups": groups,
            "top_n_per_group": int(args.top_n_per_group),
            "sort_metric": str(args.sort_metric),
            "sort_desc": bool(args.sort_desc),
            "keep_k": int(args.keep_k),
            "candidate_topn": int(args.candidate_topn),
            "candidate_mode": str(args.candidate_mode),
            "hybrid_topm": int(args.hybrid_topm),
            "candidate_pool": str(args.candidate_pool),
            "object_patch_topk": int(args.object_patch_topk),
            "exclude_padding_patches": bool(args.exclude_padding_patches),
            "mask_mode": str(args.mask_mode),
            "batch_candidates": int(args.batch_candidates),
            "role_eps": float(args.role_eps),
            "seed": int(args.seed),
        },
        "counts": {
            "n_target_rows": int(len(target_ids_ordered)),
            "n_patch_rows": int(len(per_patch_rows)),
            "n_sample_rows": int(len(per_sample_rows)),
            "n_skipped": int(skipped),
        },
        "outputs": {
            "per_patch_csv": out_patch,
            "per_sample_csv": out_sample,
            "group_csv": out_group,
            "summary_json": out_summary,
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_patch)
    print("[saved]", out_sample)
    print("[saved]", out_group)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()

