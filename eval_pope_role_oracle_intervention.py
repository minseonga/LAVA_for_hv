#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image

import analyze_artrap_pairwise_fragility as pf
from analyze_pope_visual_disconnect import choose_cont_ids, normalize_yesno, parse_bool, resolve_image_path
from eval_pope_objpatch_mask_rerun import (
    build_valid_patch_index_from_rect,
    class_logit_from_ids,
    class_token_ids,
    compute_content_rect_in_model_view,
    make_model_view_image,
    mask_model_view_patches,
    read_csv,
    write_csv,
)


def parse_groups(s: str) -> List[str]:
    out = []
    for p in str(s or "").split(","):
        x = p.strip()
        if x:
            out.append(x)
    return out


def normalize_group(g: str) -> str:
    x = str(g or "").strip().lower()
    if x in {"fp", "fp_hall"}:
        return "fp_hall"
    if x in {"tp", "tp_yes"}:
        return "tp_yes"
    if x in {"tn", "tn_no"}:
        return "tn_no"
    if x in {"fn", "fn_miss"}:
        return "fn_miss"
    return x


def extract_new_ids(output_ids: torch.Tensor, prompt_ids: torch.Tensor, eos_token_id: int | None) -> List[int]:
    seq = [int(x) for x in output_ids[0].tolist()]
    pref = [int(x) for x in prompt_ids[0].tolist()]
    if len(seq) >= len(pref) and seq[: len(pref)] == pref:
        gen = seq[len(pref) :]
    else:
        gen = seq
    if eos_token_id is not None and int(eos_token_id) in gen:
        gen = gen[: gen.index(int(eos_token_id))]
    return gen


def pick_role_patches(
    rows_for_id: Sequence[Dict[str, Any]],
    group: str,
    assertive_topk: int,
    supportive_topk: int,
) -> Tuple[List[int], List[int]]:
    g = normalize_group(group)
    assertive: List[Tuple[float, int]] = []
    supportive: List[Tuple[float, int]] = []
    for r in rows_for_id:
        role = str(r.get("role_label") or "").strip().lower()
        try:
            pidx = int(r.get("candidate_patch_idx"))
        except Exception:
            continue
        if g == "fp_hall":
            if role == "harmful":
                val = float(r.get("delta_yes_minus_no") or 0.0)
                assertive.append((val, pidx))
            elif role == "supportive":
                val = float(r.get("delta_yes_minus_no") or 0.0)
                supportive.append((val, pidx))
        elif g == "tp_yes":
            if role == "supportive":
                val = float(r.get("delta_gt_margin") or 0.0)
                supportive.append((val, pidx))
            elif role == "harmful":
                val = float(r.get("delta_gt_margin") or 0.0)
                assertive.append((val, pidx))
    assertive = sorted(assertive, key=lambda x: float(x[0]), reverse=True)
    supportive = sorted(supportive, key=lambda x: float(x[0]), reverse=True)
    ap = []
    sp = []
    seen = set()
    for _, p in assertive:
        if p in seen:
            continue
        ap.append(int(p))
        seen.add(int(p))
        if len(ap) >= int(max(0, assertive_topk)):
            break
    seen.clear()
    for _, p in supportive:
        if p in seen:
            continue
        sp.append(int(p))
        seen.add(int(p))
        if len(sp) >= int(max(0, supportive_topk)):
            break
    return ap, sp


def confusion(rows: Sequence[Dict[str, Any]], pred_key: str) -> Dict[str, int]:
    tp = fp = tn = fn = n = 0
    for r in rows:
        gt = normalize_yesno(r.get("answer_gt"))
        pr = normalize_yesno(r.get(pred_key))
        if gt not in {"yes", "no"} or pr not in {"yes", "no"}:
            continue
        n += 1
        if gt == "yes" and pr == "yes":
            tp += 1
        elif gt == "no" and pr == "yes":
            fp += 1
        elif gt == "no" and pr == "no":
            tn += 1
        elif gt == "yes" and pr == "no":
            fn += 1
    return {"n": int(n), "tp_yes": int(tp), "fp_no_to_yes": int(fp), "tn_no": int(tn), "fn_yes_to_no": int(fn)}


def acc_from_conf(c: Dict[str, int]) -> float:
    n = int(c.get("n", 0))
    if n <= 0:
        return 0.0
    return float((int(c.get("tp_yes", 0)) + int(c.get("tn_no", 0))) / float(n))


def main() -> None:
    ap = argparse.ArgumentParser(description="GT-aware oracle role-conditioned patch intervention on POPE.")
    ap.add_argument("--samples_csv", type=str, required=True)
    ap.add_argument("--role_csv", type=str, required=True, help="per_patch_role_effect.csv")
    ap.add_argument("--image_root", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--target_groups", type=str, default="fp_hall,tp_yes")
    ap.add_argument("--top_n_per_group", type=int, default=0, help="0 means all ids in each group.")
    ap.add_argument("--oracle_arm", type=str, default="rebalance", choices=["assertive_mask", "supportive_keep", "rebalance"])
    ap.add_argument(
        "--tp_rebalance_policy",
        type=str,
        default="assertive_mask",
        choices=["assertive_mask", "supportive_keep", "none"],
        help="In rebalance mode, policy used for TP_Yes rows.",
    )
    ap.add_argument(
        "--protect_supportive",
        type=parse_bool,
        default=True,
        help="When masking assertive patches, exclude supportive patches from mask.",
    )
    ap.add_argument("--assertive_topk", type=int, default=5)
    ap.add_argument("--supportive_topk", type=int, default=5)
    ap.add_argument("--exclude_padding_patches", type=parse_bool, default=True)
    ap.add_argument("--mask_mode", type=str, default="black", choices=["black", "gray", "mean"])
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows_in = read_csv(os.path.abspath(args.samples_csv))
    role_rows = read_csv(os.path.abspath(args.role_csv))
    if not rows_in:
        raise RuntimeError("No rows in samples_csv.")
    if not role_rows:
        raise RuntimeError("No rows in role_csv.")

    groups = [normalize_group(x) for x in parse_groups(args.target_groups)]
    valid_groups = {"fp_hall", "tp_yes", "tn_no", "fn_miss"}
    groups = [g for g in groups if g in valid_groups]
    if not groups:
        raise RuntimeError("No valid target_groups.")

    by_group: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in role_rows:
        g = normalize_group(r.get("group"))
        sid = str(r.get("id") or "").strip()
        if g not in valid_groups or sid == "":
            continue
        by_group[g][sid].append(r)

    selected: List[Tuple[str, str]] = []
    for g in groups:
        ids = list(by_group[g].keys())
        if int(args.top_n_per_group) > 0:
            scored: List[Tuple[float, str]] = []
            for sid in ids:
                arr = by_group[g][sid]
                if g == "fp_hall":
                    best = max([float(x.get("delta_yes_minus_no") or -1e9) for x in arr if str(x.get("role_label") or "").lower() == "harmful"] or [-1e9])
                elif g == "tp_yes":
                    best = max([float(x.get("delta_gt_margin") or -1e9) for x in arr if str(x.get("role_label") or "").lower() == "supportive"] or [-1e9])
                else:
                    best = max([abs(float(x.get("delta_gt_margin") or 0.0)) for x in arr] or [0.0])
                scored.append((float(best), sid))
            scored.sort(key=lambda x: float(x[0]), reverse=True)
            ids = [sid for _, sid in scored[: int(args.top_n_per_group)]]
        selected.extend([(g, sid) for sid in ids])

    samples_map: Dict[str, Dict[str, Any]] = {}
    for r in rows_in:
        sid = str(r.get("id") or "").strip()
        if sid:
            samples_map[sid] = r

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
    eos_id = getattr(tokenizer, "eos_token_id", None)
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", None)
    yes_ids = class_token_ids(tokenizer, "yes")
    no_ids = class_token_ids(tokenizer, "no")

    per_rows: List[Dict[str, Any]] = []
    skipped = 0
    changed = gain = harm = 0

    try:
        from tqdm.auto import tqdm

        iterable = tqdm(selected, total=len(selected), desc="pope-role-oracle", dynamic_ncols=True)
    except Exception:
        iterable = selected

    for g, sid in iterable:
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

        role_arr = by_group.get(g, {}).get(str(sid), [])
        assertive_idx, supportive_idx = pick_role_patches(
            rows_for_id=role_arr,
            group=g,
            assertive_topk=int(args.assertive_topk),
            supportive_topk=int(args.supportive_topk),
        )

        try:
            img = Image.open(image_path).convert("RGB")
            model_view = make_model_view_image(img, image_processor=image_processor, model_config=model.config)

            gsize = int(getattr(model.config, "image_grid_pinpoints", None) and 24 or 24)
            grid_w = grid_h = gsize

            # Use content-aware valid patch set for keep strategy.
            valid_patch_idx = torch.arange(int(grid_w * grid_h), dtype=torch.long)
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
            valid_set = set(int(x) for x in valid_patch_idx.tolist())

            mode = str(args.oracle_arm).strip().lower()
            mask_idx: List[int] = []
            keep_idx: List[int] = []
            intervention = "none"

            if mode == "assertive_mask":
                if assertive_idx:
                    mset = set(int(x) for x in assertive_idx if int(x) in valid_set)
                    if bool(args.protect_supportive):
                        mset = mset - set(int(x) for x in supportive_idx if int(x) in valid_set)
                    mask_idx = sorted(mset)
                    intervention = "assertive_mask"
            elif mode == "supportive_keep":
                if g == "tp_yes" and supportive_idx:
                    keep_idx = [int(x) for x in supportive_idx if int(x) in valid_set]
                    kset = set(keep_idx)
                    mask_idx = [int(x) for x in valid_set if int(x) not in kset]
                    intervention = "supportive_keep"
            else:  # rebalance
                if g == "fp_hall" and assertive_idx:
                    mset = set(int(x) for x in assertive_idx if int(x) in valid_set)
                    if bool(args.protect_supportive):
                        mset = mset - set(int(x) for x in supportive_idx if int(x) in valid_set)
                    mask_idx = sorted(mset)
                    intervention = "assertive_mask"
                elif g == "tp_yes":
                    pol = str(args.tp_rebalance_policy).strip().lower()
                    if pol == "supportive_keep" and supportive_idx:
                        keep_idx = [int(x) for x in supportive_idx if int(x) in valid_set]
                        kset = set(keep_idx)
                        mask_idx = [int(x) for x in valid_set if int(x) not in kset]
                        intervention = "supportive_keep"
                    elif pol == "assertive_mask" and assertive_idx:
                        mset = set(int(x) for x in assertive_idx if int(x) in valid_set)
                        if bool(args.protect_supportive):
                            mset = mset - set(int(x) for x in supportive_idx if int(x) in valid_set)
                        mask_idx = sorted(mset)
                        intervention = "assertive_mask"
                    else:
                        intervention = "none"

            masked_model_view = mask_model_view_patches(
                img_model_view=model_view,
                patch_indices=mask_idx,
                grid_w=grid_w,
                grid_h=grid_h,
                mode=str(args.mask_mode),
            )

            prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
                model.device
            )
            images_tensor = process_images([masked_model_view], image_processor, model.config).to(
                device=model.device,
                dtype=torch.float16,
            )
            image_sizes = [masked_model_view.size]

            with torch.no_grad():
                do_sample = bool(float(args.temperature) > 0.0)
                out_ids = model.generate(
                    prompt_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=do_sample,
                    temperature=(float(max(1e-5, args.temperature)) if do_sample else 0.0),
                    top_p=float(min(1.0, max(0.0, args.top_p))),
                    num_beams=int(max(1, args.num_beams)),
                    max_new_tokens=int(max(1, args.max_new_tokens)),
                    use_cache=True,
                )
                txt = tokenizer.decode(extract_new_ids(out_ids, prompt_ids, eos_id), skip_special_tokens=True).strip()

                out = model(
                    input_ids=prompt_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                logits1 = out.logits[0, -1, :].float().cpu()

            pred_new = normalize_yesno(txt)
            yes_logit = class_logit_from_ids(logits1, yes_ids)
            no_logit = class_logit_from_ids(logits1, no_ids)
            margin = float((yes_logit if yes_logit is not None else 0.0) - (no_logit if no_logit is not None else 0.0))

            base_ok = bool(base_pred == gt)
            new_ok = bool(pred_new == gt)
            changed_pred = bool(pred_new in {"yes", "no"} and pred_new != base_pred)
            if changed_pred:
                changed += 1
            if (not base_ok) and new_ok:
                gain += 1
            if base_ok and (not new_ok):
                harm += 1

            per_rows.append(
                {
                    "group": g,
                    "id": sid,
                    "image_id": image_id,
                    "question": question,
                    "answer_gt": gt,
                    "base_pred": base_pred,
                    "pred_new": pred_new,
                    "pred_text_new": txt,
                    "intervention": intervention,
                    "assertive_topk_idx": json.dumps(assertive_idx, ensure_ascii=False),
                    "supportive_topk_idx": json.dumps(supportive_idx, ensure_ascii=False),
                    "mask_idx": json.dumps(sorted(set(mask_idx)), ensure_ascii=False),
                    "keep_idx": json.dumps(sorted(set(keep_idx)), ensure_ascii=False),
                    "n_mask": int(len(set(mask_idx))),
                    "n_keep": int(len(set(keep_idx))),
                    "yes_minus_no_logit": margin,
                    "changed_pred": changed_pred,
                    "gain": bool((not base_ok) and new_ok),
                    "harm": bool(base_ok and (not new_ok)),
                }
            )
        except Exception as e:
            skipped += 1
            per_rows.append({"group": g, "id": sid, "error": str(e)})
            continue

    per_csv = os.path.join(out_dir, "per_sample_oracle_intervention.csv")
    write_csv(per_csv, per_rows)

    valid_rows = [r for r in per_rows if normalize_yesno(r.get("pred_new")) in {"yes", "no"}]
    conf_base = confusion(valid_rows, "base_pred")
    conf_new = confusion(valid_rows, "pred_new")
    summary = {
        "inputs": {
            "samples_csv": os.path.abspath(args.samples_csv),
            "role_csv": os.path.abspath(args.role_csv),
            "image_root": os.path.abspath(args.image_root),
            "model_path": args.model_path,
            "conv_mode": args.conv_mode,
            "target_groups": groups,
            "top_n_per_group": int(args.top_n_per_group),
            "oracle_arm": args.oracle_arm,
            "tp_rebalance_policy": args.tp_rebalance_policy,
            "protect_supportive": bool(args.protect_supportive),
            "assertive_topk": int(args.assertive_topk),
            "supportive_topk": int(args.supportive_topk),
            "exclude_padding_patches": bool(args.exclude_padding_patches),
            "mask_mode": args.mask_mode,
        },
        "counts": {
            "n_selected": int(len(selected)),
            "n_output_rows": int(len(per_rows)),
            "n_valid": int(len(valid_rows)),
            "n_skipped_or_error": int(skipped),
            "n_changed_pred": int(changed),
            "gain": int(gain),
            "harm": int(harm),
            "net_gain": int(gain - harm),
        },
        "metrics": {
            "base_acc": float(acc_from_conf(conf_base)),
            "new_acc": float(acc_from_conf(conf_new)),
            "delta_acc": float(acc_from_conf(conf_new) - acc_from_conf(conf_base)),
        },
        "confusion_base": conf_base,
        "confusion_new": conf_new,
        "outputs": {
            "per_sample_csv": per_csv,
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] {per_csv}")
    print(f"[saved] {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
