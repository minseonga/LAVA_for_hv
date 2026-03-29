#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

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


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def make_model_view_image(img: Image.Image, image_processor, model_config) -> Image.Image:
    from llava.mm_utils import expand2square

    image_aspect_ratio = getattr(model_config, "image_aspect_ratio", None)
    if image_aspect_ratio == "pad":
        proc_in = expand2square(img, tuple(int(x * 255) for x in image_processor.image_mean))
        px = image_processor.preprocess(proc_in, return_tensors="pt")["pixel_values"][0]
    elif image_aspect_ratio == "anyres":
        px = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
    else:
        px = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]

    mean = torch.tensor(image_processor.image_mean, dtype=px.dtype).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=px.dtype).view(3, 1, 1)
    rgb = (px * std + mean).clamp(0.0, 1.0)
    arr = (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr)


def compute_content_rect_in_model_view(
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
    image_aspect_ratio: Optional[str],
) -> Tuple[float, float, float, float]:
    if str(image_aspect_ratio) != "pad":
        return 0.0, 0.0, float(model_w), float(model_h)
    if orig_w <= 0 or orig_h <= 0 or model_w <= 0 or model_h <= 0:
        return 0.0, 0.0, float(model_w), float(model_h)

    scale = min(float(model_w) / float(orig_w), float(model_h) / float(orig_h))
    content_w = float(orig_w) * scale
    content_h = float(orig_h) * scale
    x0 = max(0.0, 0.5 * (float(model_w) - content_w))
    y0 = max(0.0, 0.5 * (float(model_h) - content_h))
    x1 = min(float(model_w), x0 + content_w)
    y1 = min(float(model_h), y0 + content_h)
    return x0, y0, x1, y1


def build_valid_patch_index_from_rect(
    grid_w: int,
    grid_h: int,
    model_w: int,
    model_h: int,
    rect_xyxy: Tuple[float, float, float, float],
) -> torch.Tensor:
    x0, y0, x1, y1 = rect_xyxy
    idx: List[int] = []
    for r in range(int(grid_h)):
        cy = (float(r) + 0.5) * float(model_h) / float(grid_h)
        for c in range(int(grid_w)):
            cx = (float(c) + 0.5) * float(model_w) / float(grid_w)
            if (cx >= x0) and (cx < x1) and (cy >= y0) and (cy < y1):
                idx.append(int(r * int(grid_w) + c))
    if len(idx) == 0:
        return torch.arange(int(grid_w) * int(grid_h), dtype=torch.long)
    return torch.tensor(idx, dtype=torch.long)


def patch_index_to_box(idx: int, grid_w: int, grid_h: int, w: int, h: int) -> Tuple[int, int, int, int]:
    r = int(idx) // int(grid_w)
    c = int(idx) % int(grid_w)
    x0 = int(round(c * float(w) / float(grid_w)))
    x1 = int(round((c + 1) * float(w) / float(grid_w)))
    y0 = int(round(r * float(h) / float(grid_h)))
    y1 = int(round((r + 1) * float(h) / float(grid_h)))
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    return x0, y0, x1, y1


def mask_model_view_patches(
    img_model_view: Image.Image,
    patch_indices: Sequence[int],
    grid_w: int,
    grid_h: int,
    mode: str = "black",
) -> Image.Image:
    out = img_model_view.copy().convert("RGB")
    dr = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    mm = str(mode).strip().lower()
    for pidx in patch_indices:
        x0, y0, x1, y1 = patch_index_to_box(int(pidx), grid_w, grid_h, w, h)
        if mm == "black":
            dr.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 255))
        elif mm == "gray":
            dr.rectangle([x0, y0, x1, y1], fill=(127, 127, 127, 255))
        elif mm == "mean":
            dr.rectangle([x0, y0, x1, y1], fill=(128, 116, 104, 255))
        else:
            raise RuntimeError(f"Unknown mask_mode: {mode}")
    return out


def extract_new_ids(output_ids: torch.Tensor, prompt_ids: torch.Tensor, eos_token_id: Optional[int]) -> List[int]:
    seq = [int(x) for x in output_ids[0].tolist()]
    pref = [int(x) for x in prompt_ids[0].tolist()]
    if len(seq) >= len(pref) and seq[: len(pref)] == pref:
        gen = seq[len(pref) :]
    else:
        gen = seq
    if eos_token_id is not None and int(eos_token_id) in gen:
        gen = gen[: gen.index(int(eos_token_id))]
    return [int(x) for x in gen]


def class_token_ids(tokenizer, label: str) -> List[int]:
    forms = [str(label), str(label).capitalize(), " " + str(label), " " + str(label).capitalize()]
    out: List[int] = []
    seen = set()
    for s in forms:
        ids = choose_cont_ids(tokenizer, s)
        if len(ids) <= 0:
            continue
        tid = int(ids[0])
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def class_logit_from_ids(logits_vec: torch.Tensor, ids: Sequence[int]) -> Optional[float]:
    if logits_vec is None or logits_vec.ndim != 1:
        return None
    vals: List[float] = []
    vsize = int(logits_vec.numel())
    for tid in ids:
        t = int(tid)
        if 0 <= t < vsize:
            vals.append(float(logits_vec[t].item()))
    if len(vals) == 0:
        return None
    return float(max(vals))


def select_target_ids(
    rows_trace: List[Dict[str, Any]],
    target_layer: int,
    group: str,
    sort_metric: str,
    top_n: int,
    desc: bool,
) -> List[str]:
    rows_l = []
    for r in rows_trace:
        li = safe_float(r.get("block_layer_idx"))
        if li is None or int(li) != int(target_layer):
            continue
        if group == "all":
            pass
        elif group == "fp_hall" and not parse_bool(r.get("is_fp_hallucination")):
            continue
        if group == "tp_yes" and not parse_bool(r.get("is_tp_yes")):
            continue
        if group == "tn_no" and not parse_bool(r.get("is_tn_no")):
            continue
        if group == "fn_miss" and not parse_bool(r.get("is_fn_miss")):
            continue
        rows_l.append(r)
    if sort_metric.strip() != "":
        rows_l = [r for r in rows_l if safe_float(r.get(sort_metric)) is not None]
        rows_l = sorted(
            rows_l,
            key=lambda x: float(safe_float(x.get(sort_metric)) or -1e9),
            reverse=bool(desc),
        )
    if int(top_n) > 0:
        rows_l = rows_l[: int(top_n)]
    ids: List[str] = []
    seen = set()
    for r in rows_l:
        sid = str(r.get("id") or "").strip()
        if sid == "" or sid in seen:
            continue
        ids.append(sid)
        seen.add(sid)
    return ids


def stable_hash_int(x: str) -> int:
    s = str(x or "")
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mask top-k object patches (from layer yes_sim_objpatch_max) and rerun POPE.")
    ap.add_argument("--samples_csv", type=str, required=True, help="POPE per_sample.csv (baseline predictions).")
    ap.add_argument("--per_layer_trace_csv", type=str, required=True, help="per_layer_yes_trace.csv from analyze_pope_visual_disconnect.")
    ap.add_argument("--image_root", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--target_layer", type=int, default=17)
    ap.add_argument("--target_group", type=str, default="fp_hall", choices=["all", "fp_hall", "tp_yes", "tn_no", "fn_miss"])
    ap.add_argument("--sort_metric", type=str, default="yes_sim_objpatch_max")
    ap.add_argument("--sort_desc", type=parse_bool, default=True)
    ap.add_argument("--top_n_samples", type=int, default=0, help="0 means all selected ids.")
    ap.add_argument("--mask_topk_patches", type=int, default=5)
    ap.add_argument("--object_patch_topk", type=int, default=64)
    ap.add_argument("--exclude_padding_patches", type=parse_bool, default=True)
    ap.add_argument("--mask_mode", type=str, default="black", choices=["black", "gray", "mean"])
    ap.add_argument(
        "--mask_strategy",
        type=str,
        default="mask_selected",
        choices=["mask_selected", "keep_selected", "random_selected", "keep_random", "keep_selected_addback"],
        help=(
            "mask_selected: mask chosen top-k patches; "
            "keep_selected: mask all valid patches except chosen top-k; "
            "random_selected: mask random k patches (control); "
            "keep_random: keep random k patches and mask the rest; "
            "keep_selected_addback: keep selected k + addback patches from removed context."
        ),
    )
    ap.add_argument(
        "--random_pool",
        type=str,
        default="valid",
        choices=["valid", "objpool"],
        help="Pool for random_selected: valid (all valid patches) or objpool (object-candidate pool).",
    )
    ap.add_argument("--addback_k", type=int, default=0, help="For keep_selected_addback: number of removed-context patches to add back.")
    ap.add_argument(
        "--addback_mode",
        type=str,
        default="harmful",
        choices=["harmful", "random"],
        help="For keep_selected_addback: addback patches by harmful rank (high sim) or random.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
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

    target_ids = select_target_ids(
        rows_trace=rows_trace,
        target_layer=int(args.target_layer),
        group=str(args.target_group),
        sort_metric=str(args.sort_metric),
        top_n=int(args.top_n_samples),
        desc=bool(args.sort_desc),
    )
    if len(target_ids) == 0:
        raise RuntimeError("No target ids selected.")
    target_set = set(target_ids)

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
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
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
    eos_id = getattr(tokenizer, "eos_token_id", None)
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    yes_ids = class_token_ids(tokenizer, "yes")
    no_ids = class_token_ids(tokenizer, "no")

    per_rows: List[Dict[str, Any]] = []
    skipped = 0

    try:
        from tqdm.auto import tqdm

        iterable = tqdm(target_ids, total=len(target_ids), desc="pope-objpatch-mask-rerun", dynamic_ncols=True)
    except Exception:
        iterable = target_ids

    for sid in iterable:
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
        is_fp_hall = bool(gt == "no" and base_pred == "yes")
        is_tp_yes = bool(gt == "yes" and base_pred == "yes")
        is_tn_no = bool(gt == "no" and base_pred == "no")
        is_fn_miss = bool(gt == "yes" and base_pred == "no")
        if gt not in {"yes", "no"}:
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

            vis_obj = vis[obj_patch_idx, :]
            vis_obj_n = F.normalize(vis_obj, dim=-1)
            h_n = F.normalize(h, dim=-1)
            sim_obj = torch.matmul(vis_obj_n, h_n)
            kk = int(min(max(1, int(args.mask_topk_patches)), int(sim_obj.numel())))
            top_vals, top_idx_local = torch.topk(sim_obj, k=kk, dim=-1)
            top_idx_global = obj_patch_idx[top_idx_local]

            # For random/keep/addback controls on the valid patch set.
            vis_valid = vis[valid_patch_idx, :]
            vis_valid_n = F.normalize(vis_valid, dim=-1)
            sim_valid = torch.matmul(vis_valid_n, h_n)  # [Nv]
            valid_list = [int(x) for x in valid_patch_idx.tolist()]
            sim_valid_map = {int(valid_list[i]): float(sim_valid[i].item()) for i in range(len(valid_list))}

            mstrategy = str(args.mask_strategy)
            if mstrategy == "keep_selected":
                keep_set = set(int(x) for x in top_idx_global.tolist())
                mask_idx = [int(x) for x in valid_patch_idx.tolist() if int(x) not in keep_set]
                keep_idx = sorted(keep_set)
            elif mstrategy == "random_selected":
                if str(args.random_pool) == "objpool":
                    pool = [int(x) for x in obj_patch_idx.tolist()]
                else:
                    pool = [int(x) for x in valid_patch_idx.tolist()]
                pool = sorted(set(pool))
                if len(pool) <= 0:
                    pool = [int(x) for x in valid_patch_idx.tolist()]
                kk_r = int(min(int(kk), len(pool)))
                rng = random.Random(int(args.seed) + 10007 * int(args.target_layer) + stable_hash_int(str(sid)))
                if kk_r > 0:
                    mask_idx = rng.sample(pool, kk_r)
                else:
                    mask_idx = []
                keep_idx = []
            elif mstrategy == "keep_random":
                if str(args.random_pool) == "objpool":
                    pool = [int(x) for x in obj_patch_idx.tolist()]
                else:
                    pool = [int(x) for x in valid_patch_idx.tolist()]
                pool = sorted(set(pool))
                if len(pool) <= 0:
                    pool = [int(x) for x in valid_patch_idx.tolist()]
                kk_r = int(min(int(kk), len(pool)))
                rng = random.Random(7919 + int(args.seed) + 10007 * int(args.target_layer) + stable_hash_int(str(sid)))
                keep_idx = rng.sample(pool, kk_r) if kk_r > 0 else []
                keep_set = set(int(x) for x in keep_idx)
                mask_idx = [int(x) for x in valid_patch_idx.tolist() if int(x) not in keep_set]
            elif mstrategy == "keep_selected_addback":
                keep_base = [int(x) for x in top_idx_global.tolist()]
                keep_set = set(keep_base)
                removed = [int(x) for x in valid_patch_idx.tolist() if int(x) not in keep_set]
                addk = int(max(0, args.addback_k))
                addk = int(min(addk, len(removed)))
                addback: List[int] = []
                if addk > 0:
                    if str(args.addback_mode) == "random":
                        rng = random.Random(1543 + int(args.seed) + 10007 * int(args.target_layer) + stable_hash_int(str(sid)))
                        addback = rng.sample(removed, addk)
                    else:
                        # harmful: add removed patches with highest decision-similarity.
                        removed_sorted = sorted(removed, key=lambda pidx: sim_valid_map.get(int(pidx), -1e9), reverse=True)
                        addback = removed_sorted[:addk]
                keep_idx = sorted(set(keep_base + addback))
                keep_set2 = set(keep_idx)
                mask_idx = [int(x) for x in valid_patch_idx.tolist() if int(x) not in keep_set2]
            else:
                mask_idx = [int(x) for x in top_idx_global.tolist()]
                keep_idx = []

            masked_model_view = mask_model_view_patches(
                img_model_view=model_view,
                patch_indices=mask_idx,
                grid_w=grid_w,
                grid_h=grid_h,
                mode=str(args.mask_mode),
            )

            masked_tensor = process_images([masked_model_view], image_processor, model.config).to(
                device=model.device,
                dtype=torch.float16,
            )
            masked_sizes = [masked_model_view.size]

            with torch.no_grad():
                do_sample = bool(float(args.temperature) > 0.0)
                out_ids = model.generate(
                    prompt_ids,
                    images=masked_tensor,
                    image_sizes=masked_sizes,
                    do_sample=do_sample,
                    temperature=(float(max(1e-5, args.temperature)) if do_sample else 0.0),
                    top_p=float(min(1.0, max(0.0, args.top_p))),
                    num_beams=int(max(1, args.num_beams)),
                    max_new_tokens=int(max(1, args.max_new_tokens)),
                    use_cache=True,
                )

                # Counterfactual drop probes at first decoding decision.
                out_base_step = model(
                    input_ids=prompt_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                out_mask_step = model(
                    input_ids=prompt_ids,
                    images=masked_tensor,
                    image_sizes=masked_sizes,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits_base = out_base_step.logits[0, -1, :].float().cpu()
                logits_mask = out_mask_step.logits[0, -1, :].float().cpu()

                base_yes_logit = class_logit_from_ids(logits_base, yes_ids)
                base_no_logit = class_logit_from_ids(logits_base, no_ids)
                mask_yes_logit = class_logit_from_ids(logits_mask, yes_ids)
                mask_no_logit = class_logit_from_ids(logits_mask, no_ids)

                base_margin_yes_minus_no = (
                    None
                    if base_yes_logit is None or base_no_logit is None
                    else float(base_yes_logit - base_no_logit)
                )
                mask_margin_yes_minus_no = (
                    None
                    if mask_yes_logit is None or mask_no_logit is None
                    else float(mask_yes_logit - mask_no_logit)
                )
                drop_margin_yes_minus_no = (
                    None
                    if base_margin_yes_minus_no is None or mask_margin_yes_minus_no is None
                    else float(base_margin_yes_minus_no - mask_margin_yes_minus_no)
                )

                if gt == "yes":
                    base_gt_logit = base_yes_logit
                    mask_gt_logit = mask_yes_logit
                else:
                    base_gt_logit = base_no_logit
                    mask_gt_logit = mask_no_logit
                drop_gt_logit = (
                    None
                    if base_gt_logit is None or mask_gt_logit is None
                    else float(base_gt_logit - mask_gt_logit)
                )

            gen_ids = extract_new_ids(out_ids, prompt_ids, eos_id)
            masked_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            masked_pred = normalize_yesno(masked_text)

            base_ok = (base_pred == gt) if base_pred in {"yes", "no"} else False
            masked_ok = (masked_pred == gt) if masked_pred in {"yes", "no"} else False

            per_rows.append(
                {
                    "id": sid,
                    "image_id": image_id,
                    "question": question,
                    "answer_gt": gt,
                    "base_pred": base_pred,
                    "base_pred_text": base_pred_text,
                    "is_fp_hallucination": bool(is_fp_hall),
                    "is_tp_yes": bool(is_tp_yes),
                    "is_tn_no": bool(is_tn_no),
                    "is_fn_miss": bool(is_fn_miss),
                    "masked_pred": masked_pred,
                    "masked_text": masked_text,
                    "base_ok": bool(base_ok),
                    "masked_ok": bool(masked_ok),
                    "changed_pred": bool((base_pred in {"yes", "no"}) and (masked_pred in {"yes", "no"}) and (base_pred != masked_pred)),
                    "repair": bool((not base_ok) and masked_ok),
                    "harm": bool(base_ok and (not masked_ok)),
                    "target_layer": int(args.target_layer),
                    "topk_mask_patches": int(kk),
                    "mask_strategy": str(args.mask_strategy),
                    "random_pool": str(args.random_pool),
                    "addback_k": int(args.addback_k),
                    "addback_mode": str(args.addback_mode),
                    "n_masked_patches": int(len(mask_idx)),
                    "n_kept_patches": int(len(keep_idx)),
                    "masked_patch_idx_global_json": json.dumps([int(x) for x in mask_idx]),
                    "kept_patch_idx_global_json": json.dumps([int(x) for x in keep_idx]),
                    "reference_top_patch_idx_global_json": json.dumps([int(x) for x in top_idx_global.tolist()]),
                    "masked_patch_sim_json": json.dumps([float(x) for x in top_vals.tolist()]),
                    "grid_w": int(grid_w),
                    "grid_h": int(grid_h),
                    "n_valid_patches": int(valid_patch_idx.numel()),
                    "base_yes_logit": base_yes_logit,
                    "base_no_logit": base_no_logit,
                    "masked_yes_logit": mask_yes_logit,
                    "masked_no_logit": mask_no_logit,
                    "base_margin_yes_minus_no": base_margin_yes_minus_no,
                    "masked_margin_yes_minus_no": mask_margin_yes_minus_no,
                    "drop_margin_yes_minus_no": drop_margin_yes_minus_no,
                    "base_gt_logit": base_gt_logit,
                    "masked_gt_logit": mask_gt_logit,
                    "drop_gt_logit": drop_gt_logit,
                }
            )
        except Exception as e:
            skipped += 1
            per_rows.append(
                {
                    "id": sid,
                    "image_id": image_id,
                    "question": question,
                    "answer_gt": gt,
                    "base_pred": base_pred,
                    "base_pred_text": base_pred_text,
                    "masked_pred": None,
                    "masked_text": None,
                    "base_ok": (None if base_pred not in {"yes", "no"} else bool(base_pred == gt)),
                    "masked_ok": None,
                    "changed_pred": None,
                    "repair": None,
                    "harm": None,
                    "error": str(e),
                }
            )

    valid = [r for r in per_rows if r.get("masked_ok") is not None]
    if len(valid) == 0:
        raise RuntimeError("No valid rows produced.")

    n = len(valid)
    base_acc = float(sum(1 for r in valid if bool(r.get("base_ok"))) / max(1, n))
    masked_acc = float(sum(1 for r in valid if bool(r.get("masked_ok"))) / max(1, n))
    repair = int(sum(1 for r in valid if bool(r.get("repair"))))
    harm = int(sum(1 for r in valid if bool(r.get("harm"))))
    changed = int(sum(1 for r in valid if bool(r.get("changed_pred"))))

    yes_to_no = int(
        sum(1 for r in valid if str(r.get("base_pred")) == "yes" and str(r.get("masked_pred")) == "no")
    )
    no_to_yes = int(
        sum(1 for r in valid if str(r.get("base_pred")) == "no" and str(r.get("masked_pred")) == "yes")
    )
    dm_vals = [safe_float(r.get("drop_margin_yes_minus_no")) for r in valid]
    dm_vals = [float(v) for v in dm_vals if v is not None]
    dgt_vals = [safe_float(r.get("drop_gt_logit")) for r in valid]
    dgt_vals = [float(v) for v in dgt_vals if v is not None]
    fp_dm = [safe_float(r.get("drop_margin_yes_minus_no")) for r in valid if bool(r.get("is_fp_hallucination"))]
    fp_dm = [float(v) for v in fp_dm if v is not None]
    tp_dm = [safe_float(r.get("drop_margin_yes_minus_no")) for r in valid if bool(r.get("is_tp_yes"))]
    tp_dm = [float(v) for v in tp_dm if v is not None]

    summary = {
        "inputs": {
            "samples_csv": os.path.abspath(args.samples_csv),
            "per_layer_trace_csv": os.path.abspath(args.per_layer_trace_csv),
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "target_layer": int(args.target_layer),
            "target_group": str(args.target_group),
            "sort_metric": str(args.sort_metric),
            "sort_desc": bool(args.sort_desc),
            "top_n_samples": int(args.top_n_samples),
            "mask_topk_patches": int(args.mask_topk_patches),
            "object_patch_topk": int(args.object_patch_topk),
            "exclude_padding_patches": bool(args.exclude_padding_patches),
            "mask_mode": str(args.mask_mode),
            "mask_strategy": str(args.mask_strategy),
            "random_pool": str(args.random_pool),
            "addback_k": int(args.addback_k),
            "addback_mode": str(args.addback_mode),
            "max_new_tokens": int(args.max_new_tokens),
            "num_beams": int(args.num_beams),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "seed": int(args.seed),
        },
        "counts": {
            "n_target_ids": int(len(target_ids)),
            "n_output_rows": int(len(per_rows)),
            "n_valid": int(len(valid)),
            "n_skipped_or_error": int(skipped),
            "n_changed_pred": int(changed),
            "repair": int(repair),
            "harm": int(harm),
            "yes_to_no": int(yes_to_no),
            "no_to_yes": int(no_to_yes),
        },
        "metrics": {
            "base_acc": float(base_acc),
            "masked_acc": float(masked_acc),
            "delta_acc": float(masked_acc - base_acc),
            "mean_drop_margin_yes_minus_no": (None if len(dm_vals) == 0 else float(sum(dm_vals) / len(dm_vals))),
            "mean_drop_gt_logit": (None if len(dgt_vals) == 0 else float(sum(dgt_vals) / len(dgt_vals))),
            "mean_drop_margin_fp_hall": (None if len(fp_dm) == 0 else float(sum(fp_dm) / len(fp_dm))),
            "mean_drop_margin_tp_yes": (None if len(tp_dm) == 0 else float(sum(tp_dm) / len(tp_dm))),
        },
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample_mask_rerun.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_sample_mask_rerun.csv"), per_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample_mask_rerun.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
