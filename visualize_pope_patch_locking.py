#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

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
    write_csv,
)


def parse_layers(s: str) -> List[int]:
    out: List[int] = []
    for t in str(s).split(","):
        tt = t.strip()
        if tt == "":
            continue
        out.append(int(tt))
    if len(out) == 0:
        raise RuntimeError("No valid layers parsed.")
    return out


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


def draw_patch_boxes(
    img: Image.Image,
    global_top: Sequence[int],
    obj_top: Sequence[int],
    grid_w: int,
    grid_h: int,
) -> Image.Image:
    out = img.copy().convert("RGB")
    dr = ImageDraw.Draw(out, "RGBA")
    w, h = out.size

    # global top-k: red
    for rank, pidx in enumerate(global_top, start=1):
        x0, y0, x1, y1 = patch_index_to_box(int(pidx), grid_w, grid_h, w, h)
        dr.rectangle([x0, y0, x1, y1], outline=(220, 20, 20, 255), width=2)
        dr.rectangle([x0, y0, x0 + 14, y0 + 12], fill=(220, 20, 20, 180))
        dr.text((x0 + 2, y0), str(rank), fill=(255, 255, 255, 255))

    # obj top-k: cyan
    for rank, pidx in enumerate(obj_top, start=1):
        x0, y0, x1, y1 = patch_index_to_box(int(pidx), grid_w, grid_h, w, h)
        dr.rectangle([x0, y0, x1, y1], outline=(20, 200, 220, 255), width=2)
        dr.rectangle([x1 - 16, y1 - 12, x1, y1], fill=(20, 200, 220, 180))
        dr.text((x1 - 14, y1 - 12), str(rank), fill=(0, 0, 0, 255))
    return out


def _wrap_text_to_width(dr: ImageDraw.ImageDraw, text: str, max_w: int) -> List[str]:
    t = str(text or "").strip()
    if t == "":
        return []
    words = t.split()
    if len(words) == 0:
        return [t]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = f"{cur} {w}"
        try:
            tw = float(dr.textlength(cand))
        except Exception:
            tw = float(len(cand) * 7)
        if tw <= float(max_w):
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def add_question_panel(
    img: Image.Image,
    header: str,
    question: str,
    max_lines: int = 3,
) -> Image.Image:
    src = img.convert("RGB")
    w, h = src.size
    panel_h = 76
    out = Image.new("RGB", (w, h + panel_h), (246, 246, 246))
    out.paste(src, (0, panel_h))
    dr = ImageDraw.Draw(out, "RGBA")
    dr.rectangle([0, 0, w, panel_h - 1], outline=(200, 200, 200, 255), width=1)
    dr.text((8, 6), str(header), fill=(30, 30, 30, 255))
    q_lines = _wrap_text_to_width(dr, f"Q: {question}", max_w=max(20, w - 16))
    if int(max_lines) > 0:
        q_lines = q_lines[: int(max_lines)]
    y = 24
    for ln in q_lines:
        dr.text((8, y), ln, fill=(25, 25, 25, 255))
        y += 14
        if y >= panel_h - 2:
            break
    return out


def make_model_view_image(img: Image.Image, image_processor, model_config) -> Image.Image:
    # Reconstruct the image after CLIP preprocessor for alignment with patch grid.
    from llava.mm_utils import expand2square

    image_aspect_ratio = getattr(model_config, "image_aspect_ratio", None)
    if image_aspect_ratio == "pad":
        proc_in = expand2square(img, tuple(int(x * 255) for x in image_processor.image_mean))
        px = image_processor.preprocess(proc_in, return_tensors="pt")["pixel_values"][0]
    elif image_aspect_ratio == "anyres":
        # For anyres, accurate single-grid backprojection is ambiguous.
        # Fall back to standard preprocess view for debugging.
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
    # Only pad mode keeps aspect ratio with centered padding.
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


def select_group_ids(
    rows_trace: List[Dict[str, Any]],
    select_layer: int,
    select_metric: str,
    n_fp: int,
    n_tp: int,
    selection_mode: str,
    hall_high: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    rows_layer = [r for r in rows_trace if int(float(r.get("block_layer_idx", -1))) == int(select_layer)]
    fp = [r for r in rows_layer if parse_bool(r.get("is_fp_hallucination"))]
    tp = [r for r in rows_layer if parse_bool(r.get("is_tp_yes"))]

    def _sort_by_metric(group_rows: List[Dict[str, Any]], reverse: bool) -> List[Dict[str, Any]]:
        rows2 = [r for r in group_rows if safe_float(r.get(select_metric)) is not None]
        return sorted(rows2, key=lambda x: float(safe_float(x.get(select_metric)) or -1e9), reverse=bool(reverse))

    def _take(group_rows: List[Dict[str, Any]], n_take: int) -> List[Dict[str, Any]]:
        if selection_mode == "random":
            rng = random.Random(seed + len(group_rows) * 17 + n_take)
            rows2 = group_rows[:]
            rng.shuffle(rows2)
            return rows2[: int(min(n_take, len(rows2)))]
        rows2 = _sort_by_metric(group_rows, reverse=True)
        return rows2[: int(min(n_take, len(rows2)))]

    if selection_mode == "contrast":
        # Make visual distinction explicit:
        # If metric is higher in hallucination, pick FP-high vs TP-low.
        # Else pick FP-low vs TP-high.
        if bool(hall_high):
            sel_fp = _sort_by_metric(fp, reverse=True)[: int(min(n_fp, len(fp)))]
            sel_tp = _sort_by_metric(tp, reverse=False)[: int(min(n_tp, len(tp)))]
        else:
            sel_fp = _sort_by_metric(fp, reverse=False)[: int(min(n_fp, len(fp)))]
            sel_tp = _sort_by_metric(tp, reverse=True)[: int(min(n_tp, len(tp)))]
    else:
        sel_fp = _take(fp, int(n_fp))
        sel_tp = _take(tp, int(n_tp))
    out: List[Dict[str, Any]] = []
    for r in sel_fp:
        x = dict(r)
        x["group"] = "fp_hall"
        out.append(x)
    for r in sel_tp:
        x = dict(r)
        x["group"] = "tp_yes"
        out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize POPE patch-locking (FP vs TP) with patch overlays.")
    ap.add_argument("--samples_csv", type=str, required=True)
    ap.add_argument("--per_layer_trace_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--layers", type=str, default="17,18,23")
    ap.add_argument("--select_layer", type=int, default=17)
    ap.add_argument("--select_metric", type=str, default="yes_sim_objpatch_max")
    ap.add_argument("--n_fp", type=int, default=20)
    ap.add_argument("--n_tp", type=int, default=20)
    ap.add_argument("--selection_mode", type=str, default="top", choices=["top", "random", "contrast"])
    ap.add_argument("--hall_high", type=parse_bool, default=True, help="True if select_metric is higher in hallucination.")
    ap.add_argument("--topk_patch", type=int, default=5)
    ap.add_argument("--object_patch_topk", type=int, default=64)
    ap.add_argument("--exclude_padding_patches", type=parse_bool, default=True)
    ap.add_argument("--annotate_question", type=parse_bool, default=True)
    ap.add_argument("--question_max_lines", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    layers = parse_layers(args.layers)

    rows_trace: List[Dict[str, Any]] = []
    with open(os.path.abspath(args.per_layer_trace_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows_trace.append(dict(r))

    selected = select_group_ids(
        rows_trace=rows_trace,
        select_layer=int(args.select_layer),
        select_metric=str(args.select_metric),
        n_fp=int(args.n_fp),
        n_tp=int(args.n_tp),
        selection_mode=str(args.selection_mode),
        hall_high=bool(args.hall_high),
        seed=int(args.seed),
    )
    if len(selected) == 0:
        raise RuntimeError("No selected samples from per_layer_trace.")

    # id -> sample row for stable fields from original samples csv.
    samples_map: Dict[str, Dict[str, Any]] = {}
    with open(os.path.abspath(args.samples_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid != "":
                samples_map[sid] = dict(r)

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
    import analyze_artrap_pairwise_fragility as pf

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

    manifests: List[Dict[str, Any]] = []
    warnings = 0
    for sidx, srow in enumerate(selected):
        sid = str(srow.get("id") or "")
        group = str(srow.get("group") or "unknown")

        base = samples_map.get(sid, srow)
        question = str(base.get("question") or srow.get("question") or "")
        image_id = str(base.get("image_id") or base.get("imageId") or srow.get("image_id") or "")
        pred_answer_eval = str(base.get("pred_answer_eval") or "").strip()
        pred_text = str(base.get("pred_text") or base.get("champ_text") or srow.get("pred_text") or pred_answer_eval).strip()
        answer = str(base.get("answer") or srow.get("answer_gt") or "")
        pred = normalize_yesno(pred_answer_eval if pred_answer_eval != "" else pred_text)
        gt = normalize_yesno(answer)

        image_path = resolve_image_path(args.image_root, image_id)
        if image_path is None or question == "" or pred_text == "":
            continue

        cont_ids = choose_cont_ids(tokenizer, pred_text)
        tlen = int(len(cont_ids))
        if tlen <= 0:
            continue
        yesno_idx = locate_phrase_start(tokenizer, cont_ids, pred)
        if yesno_idx is None:
            yesno_idx = 0
        yesno_idx = int(max(0, min(tlen - 1, int(yesno_idx))))

        obj_phrase = extract_pope_object(question)

        try:
            img = Image.open(image_path).convert("RGB")
            model_view = make_model_view_image(img, image_processor=image_processor, model_config=model.config)

            img_prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            prompt_ids = tokenizer_image_token(
                img_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
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
                    continue
                dec_pos = cont_label_pos - 1
                if int(dec_pos.min().item()) < 0:
                    continue
                vision_pos = torch.where(labels_exp == int(IGNORE_INDEX))[0]
                if int(vision_pos.numel()) <= 0:
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

            vis = mm_embeds_e[0, vision_pos, :].float().cpu()  # [P, D]
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

            p = int(vis.size(0))
            g = int(round(math.sqrt(float(p))))
            if g * g != p:
                if warnings < 5:
                    print(f"[warn] skip non-square patch grid: id={sid} p={p}")
                    warnings += 1
                continue
            grid_w = g
            grid_h = g
            yes_dec_pos = int(dec_pos[yesno_idx].item())

            valid_patch_idx = torch.arange(int(p), dtype=torch.long)
            content_rect = (0.0, 0.0, float(model_view.size[0]), float(model_view.size[1]))
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

            sample_dir = os.path.join(out_dir, group, f"id_{sid}")
            os.makedirs(sample_dir, exist_ok=True)

            for layer in layers:
                hidx = int(layer) + 1
                if hidx < 0 or hidx >= len(hs_all):
                    continue
                hs = hs_all[hidx][0]
                if yes_dec_pos < 0 or yes_dec_pos >= int(hs.size(0)):
                    continue
                h = hs[yes_dec_pos, :].float().cpu()

                vis_n = F.normalize(vis, dim=-1)
                h_n = F.normalize(h, dim=-1)
                sim_global_all = torch.matmul(vis_n, h_n)  # [P]
                sim_global = sim_global_all[valid_patch_idx]
                kg = int(min(max(1, int(args.topk_patch)), int(sim_global.numel())))
                topg_vals, topg_idx_local = torch.topk(sim_global, k=kg, dim=-1)
                topg_idx = valid_patch_idx[topg_idx_local]

                obj_patch_idx_valid = obj_patch_idx[valid_mask[obj_patch_idx]]
                if int(obj_patch_idx_valid.numel()) <= 0:
                    obj_patch_idx_valid = valid_patch_idx
                vis_obj = vis[obj_patch_idx_valid, :]
                vis_obj_n = F.normalize(vis_obj, dim=-1)
                sim_obj = torch.matmul(vis_obj_n, h_n)  # [Kobj]
                ko = int(min(max(1, int(args.topk_patch)), int(sim_obj.numel())))
                topo_vals, topo_idx_local = torch.topk(sim_obj, k=ko, dim=-1)
                topo_idx_global = obj_patch_idx_valid[topo_idx_local]

                over_model = draw_patch_boxes(
                    img=model_view,
                    global_top=[int(x) for x in topg_idx.tolist()],
                    obj_top=[int(x) for x in topo_idx_global.tolist()],
                    grid_w=grid_w,
                    grid_h=grid_h,
                )
                if bool(args.annotate_question):
                    over_model = add_question_panel(
                        img=over_model,
                        header=f"id={sid} group={group} layer={int(layer)} gt={gt} pred={pred}",
                        question=question,
                        max_lines=int(args.question_max_lines),
                    )
                over_model.save(os.path.join(sample_dir, f"layer_{layer:02d}_model_overlay.png"))

                over_orig = draw_patch_boxes(
                    img=img,
                    global_top=[int(x) for x in topg_idx.tolist()],
                    obj_top=[int(x) for x in topo_idx_global.tolist()],
                    grid_w=grid_w,
                    grid_h=grid_h,
                )
                if bool(args.annotate_question):
                    over_orig = add_question_panel(
                        img=over_orig,
                        header=f"id={sid} group={group} layer={int(layer)} gt={gt} pred={pred}",
                        question=question,
                        max_lines=int(args.question_max_lines),
                    )
                over_orig.save(os.path.join(sample_dir, f"layer_{layer:02d}_orig_overlay_approx.png"))

                manifests.append(
                    {
                        "id": sid,
                        "group": group,
                        "image_id": image_id,
                        "question": question,
                        "answer_gt": gt,
                        "answer_pred": pred,
                        "pred_text": pred_text,
                        "object_phrase": obj_phrase,
                        "select_layer": int(args.select_layer),
                        "select_metric": str(args.select_metric),
                        "select_metric_value": safe_float(srow.get(args.select_metric)),
                        "layer": int(layer),
                        "yesno_idx": int(yesno_idx),
                        "yesno_token_str": str(tokenizer.convert_ids_to_tokens(int(cont_ids[yesno_idx]))),
                        "n_visual_tokens": int(p),
                        "n_valid_patches": int(valid_patch_idx.numel()),
                        "grid_w": int(grid_w),
                        "grid_h": int(grid_h),
                        "content_rect_model_xyxy": json.dumps([float(content_rect[0]), float(content_rect[1]), float(content_rect[2]), float(content_rect[3])]),
                        "top_global_idx": json.dumps([int(x) for x in topg_idx.tolist()]),
                        "top_global_sim": json.dumps([float(x) for x in topg_vals.tolist()]),
                        "top_obj_idx_global": json.dumps([int(x) for x in topo_idx_global.tolist()]),
                        "top_obj_sim": json.dumps([float(x) for x in topo_vals.tolist()]),
                        "model_overlay_path": os.path.join(sample_dir, f"layer_{layer:02d}_model_overlay.png"),
                        "orig_overlay_approx_path": os.path.join(sample_dir, f"layer_{layer:02d}_orig_overlay_approx.png"),
                    }
                )
        except Exception as e:
            if warnings < 10:
                print(f"[warn] id={sid} failed: {type(e).__name__}: {e}")
                warnings += 1
            continue

    if len(manifests) == 0:
        raise RuntimeError("No overlays were generated.")

    write_csv(os.path.join(out_dir, "overlay_manifest.csv"), manifests)

    summary = {
        "inputs": {
            "samples_csv": os.path.abspath(args.samples_csv),
            "per_layer_trace_csv": os.path.abspath(args.per_layer_trace_csv),
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "layers": layers,
            "select_layer": int(args.select_layer),
            "select_metric": str(args.select_metric),
            "n_fp": int(args.n_fp),
            "n_tp": int(args.n_tp),
            "selection_mode": str(args.selection_mode),
            "hall_high": bool(args.hall_high),
            "topk_patch": int(args.topk_patch),
            "object_patch_topk": int(args.object_patch_topk),
            "exclude_padding_patches": bool(args.exclude_padding_patches),
            "annotate_question": bool(args.annotate_question),
            "question_max_lines": int(args.question_max_lines),
            "seed": int(args.seed),
        },
        "counts": {
            "n_selected": int(len(selected)),
            "n_overlays": int(len(manifests)),
            "n_unique_ids": int(len(set(str(r.get("id")) for r in manifests))),
        },
        "outputs": {
            "overlay_manifest_csv": os.path.join(out_dir, "overlay_manifest.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "overlay_manifest.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
