#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_vga_intervention_process_features import (  # noqa: E402
    compute_token_visual_row,
    entropy_from_logp,
    guidance_stats,
    kl_from_logp,
    logp_rank,
    read_jsonl_rows,
    read_prediction_map,
    safe_float,
    safe_id,
    summarize_sample,
    sum_norm,
    tokenize_caption,
    topk_overlap_from_logp,
    topk_summary,
    write_csv,
    write_json,
)
from run_vga_origin_llava_next_compat import (  # noqa: E402
    ensure_generation_config,
    materialize_vendor_question_file,
    parse_bool,
    patch_llava_next_multimodal_signature,
    patch_transformers_compat,
)


TPN_MAP = {
    "fp16": "float16",
    "fp32": "float32",
    "bf16": "bfloat16",
}


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_label_map(path: str) -> Dict[str, Dict[str, str]]:
    if not str(path or "").strip():
        return {}
    keep = {
        "baseline_label",
        "intervention_label",
        "baseline_text",
        "intervention_text",
        "baseline_correct",
        "intervention_correct",
        "harm",
        "help",
        "neutral",
        "category",
        "answer",
        "label",
    }
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row)
        if not sid:
            continue
        out[sid] = {key: row.get(key, "") for key in keep if key in row}
    return out


def object_ids_for_row(tokenizer: Any, row: Dict[str, Any]) -> List[torch.Tensor]:
    obj = row.get("object")
    if obj is None:
        return []
    if isinstance(obj, str):
        objects: Sequence[Any] = [obj]
    elif isinstance(obj, Sequence):
        objects = obj
    else:
        objects = [obj]
    ids: List[torch.Tensor] = []
    for item in objects:
        text = str(item).strip()
        if not text:
            continue
        token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if token_ids.numel() > 0:
            ids.append(token_ids)
    return ids


def entropy_guidance(vis_logits: torch.Tensor, *, topk: int) -> torch.Tensor:
    k = min(int(topk), int(vis_logits.shape[-1]))
    top_k_scores, _ = torch.topk(vis_logits, k, dim=-1)
    top_k_scores = top_k_scores.float().clamp_min(1e-12)
    denom = torch.log(torch.tensor(float(k), device=top_k_scores.device)).clamp_min(1e-12)
    entropy = (-top_k_scores * torch.log(top_k_scores) / denom).sum(-1)
    return sum_norm(entropy).to(vis_logits.dtype)


def object_guidance(vis_logits: torch.Tensor, object_ids: Sequence[torch.Tensor]) -> torch.Tensor:
    guidance_rows: List[torch.Tensor] = []
    for token_ids in object_ids:
        token_ids = token_ids.to(vis_logits.device)
        vl = vis_logits[:, token_ids]
        vl = vl[:, 0]
        guidance_rows.append(vl)
    if not guidance_rows:
        return entropy_guidance(vis_logits, topk=10)
    guidance = torch.stack(guidance_rows, dim=0).max(0).values
    return sum_norm(guidance).to(vis_logits.dtype)


def boundary_token(token_text: str) -> bool:
    return str(token_text).startswith(("▁", "Ġ"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract VGA LLaVA-NeXT intervention-process trace features by replaying "
            "the generated intervention answer and comparing local no-add vs add logits."
        )
    )
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--intervention-pred-jsonl", required=True)
    ap.add_argument("--pred-text-key", default="auto")
    ap.add_argument("--label-rows-csv", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-id", action="append", default=[])
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--conv-mode", default="llava_llama_3")
    ap.add_argument("--vss-topk", type=int, default=10)
    ap.add_argument("--use-add", type=parse_bool, default=True)
    ap.add_argument("--cd-alpha", type=float, default=0.02)
    ap.add_argument("--attn-coef", type=float, default=0.2)
    ap.add_argument("--start-layer", type=int, default=2)
    ap.add_argument("--end-layer", type=int, default=15)
    ap.add_argument("--head-balancing", default="simg", choices=["attn", "simg", "simv", "simb", "none"])
    ap.add_argument("--attn-norm", type=parse_bool, default=False)
    ap.add_argument("--torch-type", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--attn-type", default="sdpa", choices=["eager", "sdpa"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-steps-csv", required=True)
    ap.add_argument("--out-features-csv", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    vga_root = Path(args.vga_root)
    if not vga_root.is_absolute():
        vga_root = (repo_root / vga_root).resolve()
    for path in (str(vga_root), str(vga_root / "eval")):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    patch_transformers_compat()

    import transformers
    from transformers import set_seed

    original_from_pretrained = transformers.AutoTokenizer.from_pretrained
    tokenizer_placeholders = {
        "/path/to/llama3-llava-next-8b",
        "/path/to/Meta-Llama-3-8B-Instruct",
        "/path/to/Meta-Llama-3-8B",
    }

    def patched_from_pretrained(pretrained_model_name_or_path: Any, *a: Any, **kw: Any) -> Any:
        if str(pretrained_model_name_or_path) in tokenizer_placeholders:
            return original_from_pretrained(args.model_path, *a, **kw)
        return original_from_pretrained(pretrained_model_name_or_path, *a, **kw)

    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        from vcd_utils.greedy_sample_next import evolve_greedy_sampling

        evolve_greedy_sampling()
    finally:
        transformers.AutoTokenizer.from_pretrained = original_from_pretrained

    from llava_next.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava_next.conversation import SeparatorStyle, conv_templates
    from llava_next.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava_next.model.builder import load_pretrained_model
    from llava_next.utils import disable_torch_init

    patch_llava_next_multimodal_signature()
    set_seed(int(args.seed))
    disable_torch_init()

    pred_map = read_prediction_map(args.intervention_pred_jsonl, text_key=args.pred_text_key)
    label_map = read_label_map(args.label_rows_csv)

    # Reuse the vendor schema materializer so field defaults match actual VGA runs.
    schema_question_file = materialize_vendor_question_file(
        args.question_file,
        os.path.join(os.path.dirname(os.path.abspath(args.out_features_csv)), "schema_probe.jsonl"),
        limit=int(args.limit),
    )
    question_rows = read_jsonl_rows(schema_question_file)
    if args.sample_id:
        sample_ids = {
            str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else str(x)
            for x in args.sample_id
        }
        question_rows = [row for row in question_rows if safe_id(row) in sample_ids]

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    torch_dtype = TPN_MAP[str(args.torch_type)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            args.model_base,
            model_name,
            device_map="cuda",
            attn_implementation=str(args.attn_type),
            torch_dtype=torch_dtype,
        )
    model.eval()
    model.model.lm_head = model.lm_head
    ensure_generation_config(model, tokenizer)
    eos_id = model.generation_config.eos_token_id

    step_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    n_errors = 0

    for idx, q in enumerate(question_rows):
        sid = safe_id(q)
        image_file = str(q.get("image", "")).strip()
        question = str(q.get("question", q.get("text", ""))).strip()
        caption = pred_map.get(sid, "")
        sample_step_rows: List[Dict[str, Any]] = []

        try:
            if not sid:
                raise ValueError("missing sample id")
            if not image_file:
                raise ValueError("missing image")
            if not question:
                raise ValueError("missing question")
            if not caption:
                raise ValueError("missing intervention caption")

            if model.config.mm_use_im_start_end:
                qs_for_model = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                qs_for_model = DEFAULT_IMAGE_TOKEN + "\n" + question
            conv = copy.deepcopy(conv_templates[str(args.conv_mode)])
            conv.append_message(conv.roles[0], qs_for_model)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + "\n\n"
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_sizes = [image.size]
            image_tensor = process_images([image], image_processor, model.config)[0]
            image_batch = image_tensor.unsqueeze(0).to(model.dtype).cuda()
            target_tokens = tokenize_caption(tokenizer, caption, max_new_tokens=int(args.max_new_tokens))
            object_ids = object_ids_for_row(tokenizer, q)

            with torch.inference_mode():
                for layer_idx in range(int(args.start_layer), int(args.end_layer) + 1):
                    if 0 <= layer_idx < len(model.model.layers):
                        model.model.layers[layer_idx].self_attn.vis_pooled = None

                prompt_outputs = model(
                    input_ids[:, :-1],
                    images=image_batch,
                    image_sizes=image_sizes,
                    use_cache=True,
                    return_dict=True,
                )
                logits = prompt_outputs.logits
                seq_length = int(logits.shape[1])
                user_len = int(input_ids[:, :-1].size(-1) - 45 - 1)
                img_idx = (45, seq_length - user_len, seq_length)
                vis_logits = F.softmax(logits[0, img_idx[0] : img_idx[1], :], dim=-1)
                vl_guidance = object_guidance(vis_logits, object_ids) if object_ids else entropy_guidance(
                    vis_logits,
                    topk=int(args.vss_topk),
                )

                pvg_past = prompt_outputs.past_key_values
                pvg_input = input_ids[:, -1:]

                for step, actual_id_raw in enumerate(target_tokens):
                    step_attention_mask = torch.ones((1, seq_length + step + 1), dtype=torch.bool, device=input_ids.device)
                    common_kwargs = {
                        "attention_mask": step_attention_mask,
                        "images": image_batch,
                        "image_sizes": image_sizes,
                        "past_key_values": pvg_past,
                        "use_cache": True,
                        "return_dict": True,
                        "img_idx": img_idx,
                    }
                    noadd_outputs = model(
                        pvg_input,
                        **common_kwargs,
                        use_add=False,
                    )
                    add_outputs = model(
                        pvg_input,
                        **common_kwargs,
                        vl_guidance=vl_guidance,
                        add_layer=list(range(int(args.start_layer), int(args.end_layer) + 1)),
                        attn_coef=float(args.attn_coef),
                        use_add=bool(args.use_add),
                        head_balancing=str(args.head_balancing),
                        attn_norm=bool(args.attn_norm),
                    )
                    no_logp = torch.log_softmax(noadd_outputs.logits[:, -1, :].float(), dim=-1)[0]
                    ad_logp = torch.log_softmax(add_outputs.logits[:, -1, :].float(), dim=-1)[0]
                    no_top = int(torch.argmax(no_logp).item())
                    ad_top = int(torch.argmax(ad_logp).item())
                    actual_id = int(actual_id_raw)
                    actual_token_text = tokenizer.convert_ids_to_tokens([actual_id])[0]

                    row: Dict[str, Any] = {
                        "id": sid,
                        "image": image_file,
                        "step": int(step),
                        "actual_token_id": actual_id,
                        "actual_token_text": actual_token_text,
                        "top1_changed": int(no_top != ad_top),
                        "actual_noadd_logprob": float(no_logp[actual_id].item()),
                        "actual_add_logprob": float(ad_logp[actual_id].item()),
                        "actual_add_minus_noadd_logprob": float(ad_logp[actual_id].item() - no_logp[actual_id].item()),
                        "actual_noadd_rank": logp_rank(no_logp, actual_id),
                        "actual_add_rank": logp_rank(ad_logp, actual_id),
                        "noadd_top1_logprob_drop_by_add": float(no_logp[no_top].item() - ad_logp[no_top].item()),
                        "noadd_top1_add_rank": logp_rank(ad_logp, no_top),
                        "add_top1_logprob_boost_over_noadd": float(ad_logp[ad_top].item() - no_logp[ad_top].item()),
                        "add_top1_noadd_rank": logp_rank(no_logp, ad_top),
                        "noadd_entropy": entropy_from_logp(no_logp),
                        "add_entropy": entropy_from_logp(ad_logp),
                        "entropy_delta_add_minus_noadd": entropy_from_logp(ad_logp) - entropy_from_logp(no_logp),
                        "kl_add_to_noadd": kl_from_logp(ad_logp, no_logp),
                        "kl_noadd_to_add": kl_from_logp(no_logp, ad_logp),
                        "top10_overlap_noadd_add": topk_overlap_from_logp(no_logp, ad_logp, 10),
                        "top50_overlap_noadd_add": topk_overlap_from_logp(no_logp, ad_logp, 50),
                    }
                    row.update(topk_summary(tokenizer, no_logp, topk=int(args.topk), prefix="noadd"))
                    row.update(topk_summary(tokenizer, ad_logp, topk=int(args.topk), prefix="add"))
                    if eos_id is not None:
                        row["eos_noadd_logprob"] = float(no_logp[int(eos_id)].item())
                        row["eos_add_logprob"] = float(ad_logp[int(eos_id)].item())
                        row["eos_add_minus_noadd_logprob"] = float(ad_logp[int(eos_id)].item() - no_logp[int(eos_id)].item())
                    row.update(guidance_stats(vl_guidance))
                    row.update(
                        compute_token_visual_row(
                            vl_guidance=vl_guidance,
                            vis_logits=vis_logits,
                            token_id=actual_id,
                            topk=10,
                            prefix="actual",
                        )
                    )
                    row.update(
                        compute_token_visual_row(
                            vl_guidance=vl_guidance,
                            vis_logits=vis_logits,
                            token_id=no_top,
                            topk=10,
                            prefix="noadd_top1",
                        )
                    )
                    row.update(
                        compute_token_visual_row(
                            vl_guidance=vl_guidance,
                            vis_logits=vis_logits,
                            token_id=ad_top,
                            topk=10,
                            prefix="add_top1",
                        )
                    )

                    if float(args.cd_alpha) > 0 and boundary_token(actual_token_text):
                        token_visual = sum_norm(vis_logits[:, actual_id].float().clamp_min(0.0))
                        updated = (1.0 + float(args.cd_alpha)) * vl_guidance.float() - float(args.cd_alpha) * token_visual
                        updated = F.relu(updated)
                        updated = sum_norm(updated).to(vis_logits.dtype)
                        row["guidance_update_l1"] = float(torch.sum(torch.abs(updated.float() - vl_guidance.float())).item())
                        row["guidance_update_cosine"] = float(
                            F.cosine_similarity(updated.float().reshape(1, -1), vl_guidance.float().reshape(1, -1)).item()
                        )
                        vl_guidance = updated
                    else:
                        row["guidance_update_l1"] = 0.0
                        row["guidance_update_cosine"] = 1.0

                    sample_step_rows.append(row)
                    step_rows.append(row)
                    pvg_past = add_outputs.past_key_values
                    pvg_input = torch.tensor([[actual_id]], dtype=torch.long, device=input_ids.device)

                    if stop_str and stop_str in tokenizer.decode(target_tokens[: step + 1], skip_special_tokens=True):
                        break

            feature = summarize_sample(sample_step_rows, sid=sid, image=image_file, caption=caption)
            feature.update(label_map.get(sid, {}))
            feature_rows.append(feature)
        except Exception as exc:
            n_errors += 1
            row = {"id": sid, "image": image_file, "caption": caption, "error": repr(exc)}
            row.update(label_map.get(sid, {}))
            feature_rows.append(row)
            print(f"[error] id={sid} {exc!r}", flush=True)

        if (idx + 1) % 25 == 0:
            print(f"[process-next-trace] {idx + 1}/{len(question_rows)}", flush=True)

    write_csv(args.out_steps_csv, step_rows)
    write_csv(args.out_features_csv, feature_rows)
    write_json(
        args.out_summary_json,
        {
            "inputs": vars(args),
            "counts": {
                "n_questions": len(question_rows),
                "n_step_rows": len(step_rows),
                "n_feature_rows": len(feature_rows),
                "n_errors": n_errors,
            },
            "outputs": {
                "steps_csv": os.path.abspath(args.out_steps_csv),
                "features_csv": os.path.abspath(args.out_features_csv),
                "summary_json": os.path.abspath(args.out_summary_json),
            },
        },
    )
    print("[saved]", os.path.abspath(args.out_features_csv), flush=True)
    print("[saved]", os.path.abspath(args.out_summary_json), flush=True)


if __name__ == "__main__":
    main()
