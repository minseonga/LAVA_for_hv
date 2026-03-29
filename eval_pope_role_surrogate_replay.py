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
from analyze_pope_visual_disconnect import normalize_yesno, parse_bool, resolve_image_path
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


def quantile(vals: Sequence[float], q: float) -> float:
    arr = sorted(float(v) for v in vals if v is not None)
    if len(arr) == 0:
        return 0.0
    i = int(round((len(arr) - 1) * float(q)))
    i = max(0, min(len(arr) - 1, i))
    return float(arr[i])


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


def pick_top_ratio(rows: Sequence[Dict[str, Any]], key: str, ratio: float) -> List[int]:
    rr = [(float(r.get(key) or 0.0), int(r.get("candidate_patch_idx"))) for r in rows]
    rr = sorted(rr, key=lambda x: float(x[0]), reverse=True)
    if len(rr) == 0 or float(ratio) <= 0.0:
        return []
    k = int(round(float(len(rr)) * float(ratio)))
    k = max(1, min(len(rr), k))
    out = []
    seen = set()
    for _, p in rr:
        if p in seen:
            continue
        out.append(int(p))
        seen.add(int(p))
        if len(out) >= k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Surrogate-based role replay on POPE.")
    ap.add_argument("--samples_csv", type=str, required=True)
    ap.add_argument("--scored_csv", type=str, required=True, help="merged_role_feature_scored.csv from analyze_pope_role_surrogate_v2.py")
    ap.add_argument("--image_root", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--target_groups", type=str, default="fp_hall,tp_yes")
    ap.add_argument("--use_split", type=str, default="val", choices=["all", "train", "val"])
    ap.add_argument("--operator", type=str, default="patch_only", choices=["patch_only", "harmful_head_aware", "bipolar_head_aware"])
    ap.add_argument("--assertive_ratio", type=float, default=0.2)
    ap.add_argument("--supportive_ratio", type=float, default=0.2)
    ap.add_argument("--harmful_gate_quantile", type=float, default=0.5)
    ap.add_argument("--faithful_gate_quantile", type=float, default=0.5)
    ap.add_argument("--protect_supportive", type=parse_bool, default=True)
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
    scored_rows = read_csv(os.path.abspath(args.scored_csv))
    if not rows_in:
        raise RuntimeError("No rows in samples_csv.")
    if not scored_rows:
        raise RuntimeError("No rows in scored_csv.")

    groups = [normalize_group(x) for x in parse_groups(args.target_groups)]
    valid_groups = {"fp_hall", "tp_yes", "tn_no", "fn_miss"}
    groups = [g for g in groups if g in valid_groups]
    if not groups:
        raise RuntimeError("No valid target_groups.")

    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    kept_rows = []
    for r in scored_rows:
        sid = str(r.get("id") or "").strip()
        if sid == "":
            continue
        g = normalize_group(r.get("group"))
        if g not in groups:
            continue
        sp = str(r.get("split") or "").strip().lower()
        if str(args.use_split) != "all" and sp != str(args.use_split):
            continue
        rr = dict(r)
        rr["group"] = g
        by_id[sid].append(rr)
        kept_rows.append(rr)
    if not by_id:
        raise RuntimeError("No rows after group/split filter.")

    harm_vals = []
    faith_vals = []
    for sid, arr in by_id.items():
        hv = sum(float(x.get("harmful_head_attn") or 0.0) for x in arr) / float(len(arr))
        fv = sum(float(x.get("faithful_head_attn") or 0.0) for x in arr) / float(len(arr))
        harm_vals.append(hv)
        faith_vals.append(fv)
    harm_th = quantile(harm_vals, float(args.harmful_gate_quantile))
    faith_th = quantile(faith_vals, float(args.faithful_gate_quantile))

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

    ids = sorted(by_id.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    try:
        from tqdm.auto import tqdm

        iterable = tqdm(ids, total=len(ids), desc="pope-surrogate-replay", dynamic_ncols=True)
    except Exception:
        iterable = ids

    for sid in iterable:
        cand = by_id.get(sid, [])
        if len(cand) == 0:
            continue
        g = normalize_group(cand[0].get("group"))
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
            # 24x24 for llava-v1.5 vision token grid
            grid_w = 24
            grid_h = 24
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

            assertive = pick_top_ratio(cand, "S_assertive", float(args.assertive_ratio))
            supportive = pick_top_ratio(cand, "S_supportive", float(args.supportive_ratio))

            sample_harm = sum(float(x.get("harmful_head_attn") or 0.0) for x in cand) / float(len(cand))
            sample_faith = sum(float(x.get("faithful_head_attn") or 0.0) for x in cand) / float(len(cand))

            operator = str(args.operator).strip().lower()
            apply_harmful = True
            protect_support = bool(args.protect_supportive)
            if operator == "harmful_head_aware":
                apply_harmful = bool(sample_harm >= harm_th)
                protect_support = False
            elif operator == "bipolar_head_aware":
                apply_harmful = bool(sample_harm >= harm_th)
                protect_support = bool(sample_faith >= faith_th)

            mask_set = set()
            if apply_harmful:
                mask_set = set(int(x) for x in assertive if int(x) in valid_set)
            if protect_support:
                mask_set = mask_set - set(int(x) for x in supportive if int(x) in valid_set)
            mask_idx = sorted(mask_set)

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
                    "operator": operator,
                    "assertive_ratio": float(args.assertive_ratio),
                    "supportive_ratio": float(args.supportive_ratio),
                    "n_candidates": int(len(cand)),
                    "n_assertive_selected": int(len(assertive)),
                    "n_supportive_selected": int(len(supportive)),
                    "sample_harmful_head_attn": float(sample_harm),
                    "sample_faithful_head_attn": float(sample_faith),
                    "harmful_gate_threshold": float(harm_th),
                    "faithful_gate_threshold": float(faith_th),
                    "apply_harmful": bool(apply_harmful),
                    "protect_supportive": bool(protect_support),
                    "mask_idx": json.dumps(mask_idx, ensure_ascii=False),
                    "n_mask": int(len(mask_idx)),
                    "yes_minus_no_logit": margin,
                    "changed_pred": changed_pred,
                    "gain": bool((not base_ok) and new_ok),
                    "harm": bool(base_ok and (not new_ok)),
                }
            )
        except Exception as e:
            skipped += 1
            per_rows.append({"group": g, "id": sid, "error": str(e)})

    per_csv = os.path.join(out_dir, "per_sample_surrogate_replay.csv")
    write_csv(per_csv, per_rows)

    valid_rows = [r for r in per_rows if normalize_yesno(r.get("pred_new")) in {"yes", "no"}]
    conf_base = confusion(valid_rows, "base_pred")
    conf_new = confusion(valid_rows, "pred_new")
    summary = {
        "inputs": {
            "samples_csv": os.path.abspath(args.samples_csv),
            "scored_csv": os.path.abspath(args.scored_csv),
            "image_root": os.path.abspath(args.image_root),
            "model_path": args.model_path,
            "conv_mode": args.conv_mode,
            "target_groups": groups,
            "use_split": args.use_split,
            "operator": args.operator,
            "assertive_ratio": float(args.assertive_ratio),
            "supportive_ratio": float(args.supportive_ratio),
            "harmful_gate_quantile": float(args.harmful_gate_quantile),
            "faithful_gate_quantile": float(args.faithful_gate_quantile),
            "protect_supportive": bool(args.protect_supportive),
        },
        "counts": {
            "n_ids": int(len(ids)),
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
    print("[saved]", per_csv)
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()

