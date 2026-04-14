#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_vga_lost_object_guidance import (
    build_vss_guidance,
    choose_oracle_samples,
    compute_token_visual_row,
    object_tokens,
    parse_bool,
    patch_legacy_transformers_bloom_masks,
    read_csv_rows,
    read_jsonl_map,
    safe_float,
    safe_id,
    split_bar_items,
    mode_for_oracle_row,
    sum_norm,
    token_norm_from_id,
    token_piece_word,
    topk_indices_and_values,
)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                cols.append(key)
                seen.add(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_guidance_selected_samples(path: str, *, limit_samples: int) -> List[str]:
    rows = read_csv_rows(path)
    scored: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = safe_id(row)
        if not sid:
            continue
        vanilla_rank = safe_float(row.get("vanilla_target_rank"), 10**9)
        vanilla_lp = safe_float(row.get("vanilla_target_logprob"), -10**9)
        cosine = safe_float(row.get("before_guidance_token_dist_cosine"), 1.0)
        top10 = safe_float(row.get("before_guidance_token_dist_top10_overlap"), 1.0)
        visual_max = safe_float(row.get("before_visual_prob_max"), 0.0)
        high_conf = vanilla_rank <= 2 and vanilla_lp >= -1.5
        low_align = top10 <= 0.0 and cosine < 0.2
        evidence = high_conf and low_align and visual_max >= 0.05
        strong = high_conf and low_align
        cur = scored.setdefault(
            sid,
            {
                "id": sid,
                "mode": row.get("mode", ""),
                "n": 0,
                "strong": 0,
                "evidence": 0,
                "min_cosine": 1.0,
                "max_visual": 0.0,
            },
        )
        cur["n"] += 1
        cur["strong"] += int(strong)
        cur["evidence"] += int(evidence)
        cur["min_cosine"] = min(cur["min_cosine"], cosine)
        cur["max_visual"] = max(cur["max_visual"], visual_max)

    ordered = sorted(
        scored.values(),
        key=lambda x: (
            x["evidence"] / max(1, x["n"]),
            x["strong"] / max(1, x["n"]),
            x["evidence"],
            x["strong"],
            x["n"],
            -x["min_cosine"],
        ),
        reverse=True,
    )
    return [str(x["id"]) for x in ordered[: int(limit_samples)]]


def read_sample_manifest(path: str) -> List[Dict[str, str]]:
    rows = read_csv_rows(path)
    out: List[Dict[str, str]] = []
    for row in rows:
        sid = safe_id(row)
        if sid:
            out.append({"id": sid, "group": row.get("group", ""), "manifest_mode": row.get("mode", "")})
    return out


def candidate_token_ids(tokenizer: Any, lost_objects: Sequence[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for obj in lost_objects:
        for word in object_tokens(obj):
            surfaces = [word, " " + word, word.capitalize(), " " + word.capitalize()]
            for surface in surfaces:
                ids = tokenizer(surface, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
                for token_id in ids:
                    # LLaMA tokenization can emit a standalone whitespace marker before
                    # the semantic piece for " word"; that token is not an object cue.
                    if token_piece_word(tokenizer, int(token_id)):
                        out.setdefault(int(token_id), word)
                        break
    return out


def logp_rank(logp: torch.Tensor, token_id: int) -> int:
    return int((logp > logp[int(token_id)]).sum().item() + 1)


def entropy_from_logp(logp: torch.Tensor) -> float:
    probs = torch.exp(logp)
    return float((-(probs.clamp_min(1e-12)) * logp).sum().item())


def topk_summary(tokenizer: Any, logp: torch.Tensor, *, topk: int) -> Dict[str, Any]:
    vals, ids = torch.topk(logp, int(topk))
    id_list = [int(x) for x in ids.detach().cpu().tolist()]
    texts = tokenizer.convert_ids_to_tokens(id_list)
    return {
        "top1_token_id": id_list[0],
        "top1_token_text": texts[0],
        "top1_logprob": float(vals[0].item()),
        "top1_gap": float((vals[0] - vals[1]).item()) if len(id_list) > 1 else 0.0,
        f"top{topk}_token_ids": "|".join(str(x) for x in id_list),
        f"top{topk}_token_texts": "|".join(texts),
        f"top{topk}_logprobs": "|".join(f"{float(v):.6g}" for v in vals.detach().cpu().tolist()),
    }


def best_candidate_stats(
    tokenizer: Any,
    logp: torch.Tensor,
    candidate_ids: Dict[int, str],
    *,
    topk_ids: Sequence[int],
    prefix: str,
) -> Dict[str, Any]:
    if not candidate_ids:
        return {
            f"{prefix}_lost_best_token_id": "",
            f"{prefix}_lost_best_word": "",
            f"{prefix}_lost_best_logprob": "",
            f"{prefix}_lost_best_rank": "",
            f"{prefix}_lost_in_topk": 0,
        }
    best_id = max(candidate_ids, key=lambda tid: float(logp[int(tid)].item()))
    topk_set = set(int(x) for x in topk_ids)
    return {
        f"{prefix}_lost_best_token_id": int(best_id),
        f"{prefix}_lost_best_token_text": tokenizer.convert_ids_to_tokens([int(best_id)])[0],
        f"{prefix}_lost_best_word": candidate_ids[int(best_id)],
        f"{prefix}_lost_best_logprob": float(logp[int(best_id)].item()),
        f"{prefix}_lost_best_rank": logp_rank(logp, int(best_id)),
        f"{prefix}_lost_in_topk": int(any(int(x) in topk_set for x in candidate_ids)),
    }


def normalized_words_from_ids(tokenizer: Any, token_ids: Sequence[int]) -> set:
    text = tokenizer.decode([int(x) for x in token_ids], skip_special_tokens=True)
    return set(object_tokens(text))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Free-run vanilla and VGA/PVG side by side to diagnose lost-object trajectory drift."
    )
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--oracle-rows-csv", required=True)
    ap.add_argument("--guidance-rows-csv", default="")
    ap.add_argument("--sample-manifest-csv", default="")
    ap.add_argument("--target-col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--sample-id", action="append", default=[])
    ap.add_argument("--limit-samples", type=int, default=10)
    ap.add_argument("--samples-per-mode", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--vss-mode", default="entropy", choices=["entropy", "nll"])
    ap.add_argument("--vss-topk", type=int, default=10)
    ap.add_argument("--image-start", type=int, default=35)
    ap.add_argument("--image-end", type=int, default=611)
    ap.add_argument("--use-add", type=parse_bool, default=True)
    ap.add_argument("--cd-alpha", type=float, default=0.02)
    ap.add_argument("--attn-coef", type=float, default=0.2)
    ap.add_argument("--start-layer", type=int, default=2)
    ap.add_argument("--end-layer", type=int, default=15)
    ap.add_argument("--head-balancing", default="simg", choices=["simg", "none"])
    ap.add_argument("--attn-norm", type=parse_bool, default=False)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-steps-csv", required=True)
    ap.add_argument("--out-samples-json", required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    vga_root = Path(args.vga_root)
    if not vga_root.is_absolute():
        vga_root = (repo_root / vga_root).resolve()
    sys.path.insert(0, str(vga_root))
    sys.path.insert(0, str(vga_root / "llava"))

    from transformers import set_seed

    patch_legacy_transformers_bloom_masks()
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    set_seed(int(args.seed))
    disable_torch_init()

    oracle_rows = read_csv_rows(args.oracle_rows_csv)
    manifest_by_id: Dict[str, Dict[str, str]] = {}
    if args.sample_id:
        sample_ids = [str(x) for x in args.sample_id]
    elif args.sample_manifest_csv:
        manifest = read_sample_manifest(args.sample_manifest_csv)
        sample_ids = [row["id"] for row in manifest[: int(args.limit_samples)] if row.get("id")]
        manifest_by_id = {row["id"]: row for row in manifest}
    elif args.guidance_rows_csv:
        sample_ids = read_guidance_selected_samples(args.guidance_rows_csv, limit_samples=int(args.limit_samples))
    else:
        chosen = choose_oracle_samples(
            oracle_rows,
            target_col=args.target_col,
            sample_ids=[],
            samples_per_mode=int(args.samples_per_mode),
            limit_samples=int(args.limit_samples),
        )
        sample_ids = [str(x["id"]) for x in chosen]

    oracle_by_id = {safe_id(row): row for row in oracle_rows if safe_id(row)}
    questions = read_jsonl_map(args.question_file)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head
    eos_id = model.generation_config.eos_token_id
    pad_id = model.generation_config.pad_token_id if model.generation_config.pad_token_id is not None else eos_id

    step_rows: List[Dict[str, Any]] = []
    sample_summaries: List[Dict[str, Any]] = []

    for sid in sample_ids:
        sid = str(int(float(sid))) if str(sid).replace(".", "", 1).isdigit() else str(sid)
        oracle = oracle_by_id.get(sid)
        q = questions.get(sid)
        if oracle is None or q is None:
            sample_summaries.append({"id": sid, "error": "missing oracle or question row"})
            continue

        lost_objects = split_bar_items(oracle.get("base_only_supported_unique", ""))
        mode = mode_for_oracle_row(oracle)
        sample_group = manifest_by_id.get(sid, {}).get("group", "target_recoverable" if int(safe_float(oracle.get(args.target_col))) == 1 else "unlabeled")
        lost_words = set()
        for obj in lost_objects:
            lost_words.update(object_tokens(obj))

        if model.config.mm_use_im_start_end:
            qs_for_model = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + str(q["question"])
        else:
            qs_for_model = DEFAULT_IMAGE_TOKEN + "\n" + str(q["question"])
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs_for_model)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        image_file = q["image"]
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        prompt_attention_mask = torch.ones_like(input_ids[:, :-1], dtype=torch.long, device=input_ids.device)
        step_attention_mask = torch.ones((1, 1), dtype=torch.long, device=input_ids.device)
        candidate_ids = candidate_token_ids(tokenizer, lost_objects)

        vanilla_generated: List[int] = []
        pvg_generated: List[int] = []
        first_divergence_step: Optional[int] = None
        first_vanilla_lost_step: Optional[int] = None
        first_pvg_lost_step: Optional[int] = None

        try:
            with torch.inference_mode():
                image_batch = image_tensor.unsqueeze(0).half().cuda()
                prompt_outputs = model(
                    input_ids[:, :-1],
                    attention_mask=prompt_attention_mask,
                    images=image_batch,
                    use_cache=True,
                    return_dict=True,
                )
                vis_logits = F.softmax(
                    prompt_outputs.logits[0, int(args.image_start) : int(args.image_end), :],
                    dim=-1,
                )
                vl_guidance = build_vss_guidance(vis_logits, mode=args.vss_mode, topk=int(args.vss_topk))

                vanilla_past = prompt_outputs.past_key_values
                pvg_past = prompt_outputs.past_key_values
                vanilla_input = input_ids[:, -1:]
                pvg_input = input_ids[:, -1:]
                vanilla_finished = False
                pvg_finished = False

                for step in range(int(args.max_new_tokens)):
                    prefix_equal_before = first_divergence_step is None
                    row: Dict[str, Any] = {
                        "id": sid,
                        "image": image_file,
                        "mode": mode,
                        "sample_group": sample_group,
                        "step": int(step),
                        "prefix_equal_before_step": int(prefix_equal_before),
                        "lost_objects": "|".join(lost_objects),
                        "lost_words": "|".join(sorted(lost_words)),
                        "base_caption": oracle.get("base_caption", ""),
                        "int_caption": oracle.get("int_caption", ""),
                    }

                    if not vanilla_finished:
                        vanilla_outputs = model(
                            vanilla_input,
                            attention_mask=step_attention_mask,
                            images=image_batch,
                            past_key_values=vanilla_past,
                            use_cache=True,
                            return_dict=True,
                            use_add=False,
                        )
                        v_logp = torch.log_softmax(vanilla_outputs.logits[:, -1, :].float(), dim=-1)[0]
                        v_top_vals, v_top_ids_t = torch.topk(v_logp, int(args.topk))
                        v_top_ids = [int(x) for x in v_top_ids_t.detach().cpu().tolist()]
                        v_next = int(v_top_ids[0])
                        row.update({"vanilla_" + k: v for k, v in topk_summary(tokenizer, v_logp, topk=int(args.topk)).items()})
                        row["vanilla_entropy"] = entropy_from_logp(v_logp)
                        row.update(best_candidate_stats(tokenizer, v_logp, candidate_ids, topk_ids=v_top_ids, prefix="vanilla"))
                    else:
                        v_logp = None
                        v_next = int(pad_id) if pad_id is not None else 0
                        row["vanilla_finished"] = 1

                    if not pvg_finished:
                        pvg_outputs = model(
                            pvg_input,
                            attention_mask=step_attention_mask,
                            images=image_batch,
                            past_key_values=pvg_past,
                            use_cache=True,
                            return_dict=True,
                            vl_guidance=vl_guidance,
                            add_layer=list(range(int(args.start_layer), int(args.end_layer) + 1)),
                            attn_coef=float(args.attn_coef),
                            use_add=bool(args.use_add),
                            head_balancing=str(args.head_balancing),
                            attn_norm=bool(args.attn_norm),
                        )
                        p_logp = torch.log_softmax(pvg_outputs.logits[:, -1, :].float(), dim=-1)[0]
                        p_top_vals, p_top_ids_t = torch.topk(p_logp, int(args.topk))
                        p_top_ids = [int(x) for x in p_top_ids_t.detach().cpu().tolist()]
                        p_next = int(p_top_ids[0])
                        row.update({"pvg_" + k: v for k, v in topk_summary(tokenizer, p_logp, topk=int(args.topk)).items()})
                        row["pvg_entropy"] = entropy_from_logp(p_logp)
                        row.update(best_candidate_stats(tokenizer, p_logp, candidate_ids, topk_ids=p_top_ids, prefix="pvg"))
                    else:
                        p_logp = None
                        p_next = int(pad_id) if pad_id is not None else 0
                        row["pvg_finished"] = 1

                    if v_logp is not None and p_logp is not None:
                        v_best_id = row.get("vanilla_lost_best_token_id", "")
                        if v_best_id != "":
                            v_best_id_int = int(v_best_id)
                            row["pvg_logprob_for_vanilla_best_lost"] = float(p_logp[v_best_id_int].item())
                            row["pvg_rank_for_vanilla_best_lost"] = logp_rank(p_logp, v_best_id_int)
                            row["vanilla_minus_pvg_for_vanilla_best_lost"] = float(
                                v_logp[v_best_id_int].item() - p_logp[v_best_id_int].item()
                            )
                            align = compute_token_visual_row(
                                vl_guidance=vl_guidance,
                                vis_logits=vis_logits,
                                token_id=v_best_id_int,
                                topk=10,
                                prefix="pvg_guidance_vs_vanilla_best_lost",
                            )
                            for key in [
                                "pvg_guidance_vs_vanilla_best_lost_guidance_token_dist_cosine",
                                "pvg_guidance_vs_vanilla_best_lost_guidance_token_dist_top10_overlap",
                                "pvg_guidance_vs_vanilla_best_lost_guidance_token_dist_l1",
                            ]:
                                row[key] = align.get(key, "")

                    row["next_token_same"] = int(v_next == p_next)
                    row["vanilla_next_token_id"] = v_next
                    row["pvg_next_token_id"] = p_next
                    row["vanilla_next_token_text"] = tokenizer.convert_ids_to_tokens([v_next])[0]
                    row["pvg_next_token_text"] = tokenizer.convert_ids_to_tokens([p_next])[0]
                    row["vanilla_next_token_norm"] = token_norm_from_id(tokenizer, v_next)
                    row["pvg_next_token_norm"] = token_norm_from_id(tokenizer, p_next)

                    if first_divergence_step is None and v_next != p_next:
                        first_divergence_step = int(step)
                        row["is_first_divergence_step"] = 1
                    else:
                        row["is_first_divergence_step"] = 0

                    step_rows.append(row)

                    if not vanilla_finished:
                        vanilla_generated.append(v_next)
                        vanilla_past = vanilla_outputs.past_key_values
                        vanilla_input = torch.tensor([[v_next]], dtype=torch.long, device=input_ids.device)
                        vanilla_words = normalized_words_from_ids(tokenizer, vanilla_generated)
                        if first_vanilla_lost_step is None and (vanilla_words & lost_words):
                            first_vanilla_lost_step = int(step)
                        v_text = tokenizer.decode(vanilla_generated, skip_special_tokens=True)
                        vanilla_finished = (eos_id is not None and v_next == int(eos_id)) or (stop_str and stop_str in v_text)

                    if not pvg_finished:
                        pvg_generated.append(p_next)
                        pvg_past = pvg_outputs.past_key_values
                        pvg_input = torch.tensor([[p_next]], dtype=torch.long, device=input_ids.device)
                        pvg_words = normalized_words_from_ids(tokenizer, pvg_generated)
                        if first_pvg_lost_step is None and (pvg_words & lost_words):
                            first_pvg_lost_step = int(step)
                        p_token_text = tokenizer.convert_ids_to_tokens([p_next])[0]
                        if float(args.cd_alpha) > 0 and p_token_text.startswith("▁"):
                            token_visual = sum_norm(vis_logits[:, p_next].float().clamp_min(0.0))
                            vl_guidance = (1.0 + float(args.cd_alpha)) * vl_guidance.float() - float(args.cd_alpha) * token_visual
                            vl_guidance = F.relu(vl_guidance)
                            vl_guidance = sum_norm(vl_guidance).to(vis_logits.dtype)
                        p_text = tokenizer.decode(pvg_generated, skip_special_tokens=True)
                        pvg_finished = (eos_id is not None and p_next == int(eos_id)) or (stop_str and stop_str in p_text)

                    if vanilla_finished and pvg_finished:
                        break

            vanilla_text = tokenizer.decode(vanilla_generated, skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
            pvg_text = tokenizer.decode(pvg_generated, skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
            if stop_str and vanilla_text.endswith(stop_str):
                vanilla_text = vanilla_text[: -len(stop_str)].strip()
            if stop_str and pvg_text.endswith(stop_str):
                pvg_text = pvg_text[: -len(stop_str)].strip()
            sample_summaries.append(
                {
                    "id": sid,
                    "image": image_file,
                    "mode": mode,
                    "sample_group": sample_group,
                    "lost_objects": lost_objects,
                    "lost_words": sorted(lost_words),
                    "first_divergence_step": first_divergence_step,
                    "first_vanilla_lost_step": first_vanilla_lost_step,
                    "first_pvg_lost_step": first_pvg_lost_step,
                    "vanilla_generated_text": vanilla_text,
                    "pvg_generated_text": pvg_text,
                }
            )
            print(
                f"[sample] id={sid} divergence={first_divergence_step} "
                f"vanilla_lost={first_vanilla_lost_step} pvg_lost={first_pvg_lost_step}",
                flush=True,
            )
        except Exception as exc:
            sample_summaries.append({"id": sid, "image": image_file, "error": repr(exc)})
            print(f"[error] id={sid} {exc!r}", flush=True)

    write_csv(args.out_steps_csv, step_rows)
    write_json(
        args.out_samples_json,
        {
            "inputs": vars(args),
            "counts": {
                "n_selected_samples": len(sample_ids),
                "n_step_rows": len(step_rows),
                "n_sample_summaries": len(sample_summaries),
            },
            "sample_ids": sample_ids,
            "samples": sample_summaries,
            "outputs": {
                "steps_csv": os.path.abspath(args.out_steps_csv),
                "samples_json": os.path.abspath(args.out_samples_json),
            },
        },
    )
    print("[saved]", os.path.abspath(args.out_steps_csv))
    print("[saved]", os.path.abspath(args.out_samples_json))


if __name__ == "__main__":
    main()
