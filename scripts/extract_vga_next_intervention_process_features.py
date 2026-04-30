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


def sample_id(row: Dict[str, Any]) -> str:
    """POPE rows must be joined on question id, not COCO image id."""
    for key in ("question_id", "id", "qid", "image_id"):
        raw = str(row.get(key, "")).strip()
        if raw:
            try:
                return str(int(float(raw)))
            except Exception:
                return raw
    return safe_id(row)


def read_prediction_map_by_sample_id(path: str, *, text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl_rows(path):
        sid = sample_id(row)
        if not sid:
            continue
        if text_key and text_key != "auto":
            text = str(row.get(text_key, "")).strip()
        else:
            text = ""
            for key in ("text", "answer", "caption", "prediction", "output"):
                value = str(row.get(key, "")).strip()
                if value:
                    text = value
                    break
        out[sid] = text
    return out


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
        sid = sample_id(row)
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


def parse_yes_no(text: object) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("yes"):
        return "yes"
    return ""


def yes_no_token_sets(tokenizer: Any) -> Dict[str, List[int]]:
    variants = {
        "yes": ["yes", "Yes", " yes", " Yes", "\nyes", "\nYes"],
        "no": ["no", "No", " no", " No", "\nno", "\nNo"],
    }
    out: Dict[str, List[int]] = {}
    for label, texts in variants.items():
        ids: List[int] = []
        for text in texts:
            token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
            if token_ids:
                ids.append(int(token_ids[0]))
        out[label] = sorted(set(ids))
    return out


def logsumexp_token_set(logp: torch.Tensor, token_ids: Sequence[int]) -> float:
    ids = [int(x) for x in token_ids if 0 <= int(x) < int(logp.numel())]
    if not ids:
        return float("nan")
    idx = torch.tensor(ids, dtype=torch.long, device=logp.device)
    return float(torch.logsumexp(logp[idx].float(), dim=0).item())


def finite_values(values: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        x = safe_float(value, None)
        if x is not None:
            out.append(float(x))
    return out


def finite_mean(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return float(sum(vals) / float(len(vals))) if vals else 0.0


def finite_min(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return float(min(vals)) if vals else 0.0


def finite_max(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return float(max(vals)) if vals else 0.0


def finite_std(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / float(len(vals))
    return float((sum((v - mu) ** 2 for v in vals) / float(len(vals))) ** 0.5)


def label_force_features(
    *,
    no_logp: torch.Tensor,
    ad_logp: torch.Tensor,
    token_sets: Dict[str, List[int]],
    candidate_label: str,
) -> Dict[str, float]:
    cand = str(candidate_label or "").strip().lower()
    if cand not in {"yes", "no"}:
        return {}
    alt = "no" if cand == "yes" else "yes"
    no_cand = logsumexp_token_set(no_logp, token_sets.get(cand, []))
    no_alt = logsumexp_token_set(no_logp, token_sets.get(alt, []))
    ad_cand = logsumexp_token_set(ad_logp, token_sets.get(cand, []))
    ad_alt = logsumexp_token_set(ad_logp, token_sets.get(alt, []))
    no_margin = no_cand - no_alt
    ad_margin = ad_cand - ad_alt
    return {
        "proc_label_noadd_candidate_lp": no_cand,
        "proc_label_noadd_alt_lp": no_alt,
        "proc_label_add_candidate_lp": ad_cand,
        "proc_label_add_alt_lp": ad_alt,
        "proc_label_noadd_candidate_minus_alt": no_margin,
        "proc_label_add_candidate_minus_alt": ad_margin,
        "proc_label_candidate_lp_boost": ad_cand - no_cand,
        "proc_label_alt_lp_boost": ad_alt - no_alt,
        "proc_label_margin_boost": ad_margin - no_margin,
        "proc_label_add_kl_times_margin_boost": (ad_margin - no_margin) * kl_from_logp(ad_logp, no_logp),
    }


def parse_layer_indices(value: str) -> List[int]:
    out: List[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def layer_label_margins(
    *,
    outputs: Any,
    model: Any,
    token_sets: Dict[str, List[int]],
    candidate_label: str,
    layers: Sequence[int],
    prefix: str,
) -> Dict[str, float]:
    cand = str(candidate_label or "").strip().lower()
    if cand not in {"yes", "no"}:
        return {}
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        return {}
    alt = "no" if cand == "yes" else "yes"
    last_idx = len(hidden_states) - 1
    margins: List[float] = []
    out: Dict[str, float] = {}
    for layer in layers:
        idx = int(layer)
        if idx < 0:
            idx = last_idx + idx + 1
        if idx < 0 or idx > last_idx:
            continue
        h = hidden_states[idx][:, -1, :]
        if idx != last_idx and hasattr(model, "model") and hasattr(model.model, "norm"):
            h = model.model.norm(h)
        logits = model.lm_head(h).float()[0]
        logp = torch.log_softmax(logits, dim=-1)
        cand_lp = logsumexp_token_set(logp, token_sets.get(cand, []))
        alt_lp = logsumexp_token_set(logp, token_sets.get(alt, []))
        margin = cand_lp - alt_lp
        margins.append(float(margin))
        out[f"{prefix}_layer_l{int(layer)}_candidate_minus_alt"] = float(margin)
        out[f"{prefix}_layer_l{int(layer)}_candidate_supported"] = float(margin > 0.0)

    if margins:
        out[f"{prefix}_layer_candidate_margin_mean"] = finite_mean(margins)
        out[f"{prefix}_layer_candidate_margin_min"] = finite_min(margins)
        out[f"{prefix}_layer_candidate_margin_max"] = finite_max(margins)
        out[f"{prefix}_layer_candidate_margin_std"] = finite_std(margins)
        out[f"{prefix}_layer_candidate_supported_rate"] = finite_mean([float(v > 0.0) for v in margins])
        out[f"{prefix}_layer_candidate_margin_slope"] = float(margins[-1] - margins[0]) if len(margins) >= 2 else 0.0
        out[f"{prefix}_layer_candidate_sign_flip_count"] = float(
            sum(int((margins[i - 1] > 0.0) != (margins[i] > 0.0)) for i in range(1, len(margins)))
        )
    return out


def attention_step_stats(attentions: Any, img_idx: Sequence[int], *, prefix: str, topk: int = 10) -> Dict[str, float]:
    if not attentions:
        return {}
    start, end = int(img_idx[0]), int(img_idx[1])
    masses: List[float] = []
    top_fracs: List[float] = []
    for attn in attentions:
        if attn is None:
            continue
        # Expected shape: [batch, heads, q_len, kv_len].
        if not torch.is_tensor(attn) or attn.ndim < 4:
            continue
        vec = attn[0, :, -1, :].detach().float()
        if vec.numel() == 0 or vec.shape[-1] <= start:
            continue
        ee = min(end, int(vec.shape[-1]))
        if ee <= start:
            continue
        visual_mass_per_head = vec[:, start:ee].sum(-1)
        masses.extend(float(x) for x in visual_mass_per_head.detach().cpu().tolist())
        kk = min(int(topk), int(vec.shape[-1]))
        if kk > 0:
            top_idx = torch.topk(vec, kk, dim=-1).indices
            is_visual = ((top_idx >= start) & (top_idx < ee)).float().mean(-1)
            top_fracs.extend(float(x) for x in is_visual.detach().cpu().tolist())
    if not masses:
        return {}
    return {
        f"{prefix}_vision_mass_mean": finite_mean(masses),
        f"{prefix}_vision_mass_min": finite_min(masses),
        f"{prefix}_vision_mass_max": finite_max(masses),
        f"{prefix}_top{int(topk)}_visual_frac_mean": finite_mean(top_fracs),
        f"{prefix}_top{int(topk)}_visual_frac_min": finite_min(top_fracs),
    }


def add_process_summary_extras(feature: Dict[str, Any], step_rows: Sequence[Dict[str, Any]]) -> None:
    add_ranks = finite_values([row.get("actual_add_rank") for row in step_rows])
    noadd_ranks = finite_values([row.get("actual_noadd_rank") for row in step_rows])
    feature["proc_actual_add_top1_match_rate"] = finite_mean([float(v == 1.0) for v in add_ranks])
    feature["proc_actual_noadd_top1_match_rate"] = finite_mean([float(v == 1.0) for v in noadd_ranks])
    feature["proc_actual_add_rank_mean"] = finite_mean(add_ranks)
    feature["proc_actual_noadd_rank_mean"] = finite_mean(noadd_ranks)
    feature["proc_actual_rank_improve_mean"] = finite_mean([n - a for n, a in zip(noadd_ranks, add_ranks)])

    for kind in ("add_attn", "noadd_attn"):
        mass = [row.get(f"{kind}_vision_mass_mean") for row in step_rows]
        top_frac = [row.get(f"{kind}_top10_visual_frac_mean") for row in step_rows]
        feature[f"proc_{kind}_vision_mass_mean"] = finite_mean(mass)
        feature[f"proc_{kind}_vision_mass_min"] = finite_min(mass)
        feature[f"proc_{kind}_vision_mass_max"] = finite_max(mass)
        feature[f"proc_{kind}_top10_visual_frac_mean"] = finite_mean(top_frac)
        feature[f"proc_{kind}_top10_visual_frac_min"] = finite_min(top_frac)
    add_mass = finite_values([row.get("add_attn_vision_mass_mean") for row in step_rows])
    noadd_mass = finite_values([row.get("noadd_attn_vision_mass_mean") for row in step_rows])
    feature["proc_attn_add_minus_noadd_vision_mass_mean"] = finite_mean([a - n for a, n in zip(add_mass, noadd_mass)])


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


def first_token_id(value: Any) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, set)):
        for item in value:
            token_id = first_token_id(item)
            if token_id is not None:
                return token_id
        return None
    try:
        return int(value)
    except Exception:
        return None


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
    ap.add_argument("--collect-attention-features", type=parse_bool, default=False)
    ap.add_argument("--collect-layer-features", type=parse_bool, default=True)
    ap.add_argument("--trace-layers", default="8,16,24,32")
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

    pred_map = read_prediction_map_by_sample_id(args.intervention_pred_jsonl, text_key=args.pred_text_key)
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
    eos_id = first_token_id(model.generation_config.eos_token_id)
    yn_token_sets = yes_no_token_sets(tokenizer)
    trace_layers = parse_layer_indices(args.trace_layers)
    collect_attention = bool(args.collect_attention_features)
    collect_layer = bool(args.collect_layer_features)

    step_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    n_errors = 0

    for idx, q in enumerate(question_rows):
        sid = sample_id(q)
        image_file = str(q.get("image", "")).strip()
        question = str(q.get("question", q.get("text", ""))).strip()
        caption = pred_map.get(sid, "")
        sample_step_rows: List[Dict[str, Any]] = []
        sample_label_trace: Dict[str, float] = {}

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
            if getattr(conv, "tokenizer", None) is None:
                conv.tokenizer = tokenizer
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
                        "output_attentions": collect_attention,
                        "output_hidden_states": collect_layer,
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
                    if collect_attention:
                        row.update(attention_step_stats(noadd_outputs.attentions, img_idx, prefix="noadd_attn", topk=10))
                        row.update(attention_step_stats(add_outputs.attentions, img_idx, prefix="add_attn", topk=10))
                    if step == 0:
                        label_row = label_map.get(sid, {})
                        candidate_label = str(label_row.get("intervention_label", "")).strip().lower()
                        if candidate_label not in {"yes", "no"}:
                            candidate_label = parse_yes_no(caption)
                        sample_label_trace = label_force_features(
                            no_logp=no_logp,
                            ad_logp=ad_logp,
                            token_sets=yn_token_sets,
                            candidate_label=candidate_label,
                        )
                        if collect_layer:
                            sample_label_trace.update(
                                layer_label_margins(
                                    outputs=noadd_outputs,
                                    model=model,
                                    token_sets=yn_token_sets,
                                    candidate_label=candidate_label,
                                    layers=trace_layers,
                                    prefix="proc_label_noadd",
                                )
                            )
                            sample_label_trace.update(
                                layer_label_margins(
                                    outputs=add_outputs,
                                    model=model,
                                    token_sets=yn_token_sets,
                                    candidate_label=candidate_label,
                                    layers=trace_layers,
                                    prefix="proc_label_add",
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
            add_process_summary_extras(feature, sample_step_rows)
            feature.update(sample_label_trace)
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
