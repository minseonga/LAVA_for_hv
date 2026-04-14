#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']*")
STOP = {"a", "an", "and", "of", "or", "the"}
LIGHT_ALIASES = {
    "people": "person",
    "persons": "person",
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "boy": "person",
    "boys": "person",
    "girl": "person",
    "girls": "person",
    "child": "person",
    "children": "person",
    "player": "person",
    "players": "person",
    "fridge": "refrigerator",
    "luggage": "suitcase",
    "bag": "handbag",
    "bags": "handbag",
    "bike": "bicycle",
    "bikes": "bicycle",
    "doughnut": "donut",
    "doughnuts": "donut",
    "cellphone": "phone",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def patch_legacy_transformers_bloom_masks() -> None:
    """Unblock VGA/LLaVA imports on newer Transformers versions."""
    try:
        import transformers.models.bloom.modeling_bloom as bloom
    except Exception:
        return

    if not hasattr(bloom, "_expand_mask"):

        def _expand_mask(mask: torch.Tensor, tgt_length: Optional[int] = None) -> torch.BoolTensor:
            batch_size, src_length = mask.shape
            tgt_length = int(tgt_length) if tgt_length is not None else int(src_length)
            expanded_mask = ~mask[:, None, None, :].to(torch.bool)
            return expanded_mask.expand(batch_size, 1, tgt_length, src_length)

        bloom._expand_mask = _expand_mask  # type: ignore[attr-defined]

    if not hasattr(bloom, "_make_causal_mask"):

        def _make_causal_mask(
            input_ids_shape: Any,
            device: torch.device,
            past_key_values_length: int = 0,
        ) -> torch.BoolTensor:
            batch_size, tgt_length = int(input_ids_shape[0]), int(input_ids_shape[1])
            src_length = tgt_length + int(past_key_values_length)
            mask = torch.zeros((tgt_length, src_length), dtype=torch.bool, device=device)
            mask[:, int(past_key_values_length) :] = torch.triu(
                torch.ones((tgt_length, tgt_length), dtype=torch.bool, device=device),
                diagonal=1,
            )
            return mask[None, None, :, :].expand(batch_size, 1, tgt_length, src_length)

        bloom._make_causal_mask = _make_causal_mask  # type: ignore[attr-defined]


def safe_id(row: Dict[str, Any]) -> str:
    for key in ("id", "image_id", "question_id", "qid"):
        raw = str(row.get(key, "")).strip()
        if raw:
            try:
                return str(int(float(raw)))
            except Exception:
                return raw
    image = str(row.get("image", "")).strip()
    if image:
        m = re.search(r"(\d+)", os.path.basename(image))
        if m:
            return str(int(m.group(1)))
    return ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def normalize_token(token: str) -> str:
    tok = token.lower().strip().strip("'").replace("▁", "").replace("Ġ", "")
    tok = LIGHT_ALIASES.get(tok, tok)
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 5 and tok.endswith("ves"):
        tok = tok[:-3] + "f"
    elif len(tok) > 4 and tok.endswith(("ches", "shes", "xes", "zes")):
        tok = tok[:-2]
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us")):
        tok = tok[:-1]
    return LIGHT_ALIASES.get(tok, tok)


def object_tokens(text: str) -> List[str]:
    out: List[str] = []
    for raw in TOKEN_RE.findall(str(text).lower()):
        tok = normalize_token(raw)
        if tok and tok not in STOP:
            out.append(tok)
    return out


def split_bar_items(text: str) -> List[str]:
    return [x.strip() for x in str(text or "").split("|") if x.strip()]


def mode_for_oracle_row(row: Dict[str, str]) -> str:
    if safe_float(row.get("n_int_only_hallucinated_unique")) > 0:
        return "rewrite_substitution_with_int_hall"
    if safe_float(row.get("n_base_only_supported_unique")) >= 2 or safe_float(row.get("delta_recall_base_minus_int")) >= 0.5:
        return "hall_free_broad_coverage_collapse"
    return "hall_free_local_clean_omission"


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_map(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = safe_id(obj)
            if sid:
                out[sid] = obj
    return out


def choose_oracle_samples(
    rows: Sequence[Dict[str, str]],
    *,
    target_col: str,
    sample_ids: Sequence[str],
    samples_per_mode: int,
    limit_samples: int,
) -> List[Dict[str, Any]]:
    by_id = {safe_id(row): row for row in rows if safe_id(row)}
    chosen: List[Dict[str, Any]] = []
    if sample_ids:
        for sid in sample_ids:
            row = by_id.get(str(int(float(sid))) if str(sid).replace(".", "", 1).isdigit() else str(sid))
            if row:
                chosen.append({"id": safe_id(row), "mode": mode_for_oracle_row(row), "row": row})
        return chosen[:limit_samples] if limit_samples > 0 else chosen

    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(safe_float(row.get(target_col))) != 1:
            continue
        if not split_bar_items(row.get("base_only_supported_unique", "")):
            continue
        buckets[mode_for_oracle_row(row)].append(row)

    modes = [
        "hall_free_local_clean_omission",
        "hall_free_broad_coverage_collapse",
        "rewrite_substitution_with_int_hall",
    ]
    for mode in modes:
        for row in buckets.get(mode, [])[: max(0, int(samples_per_mode))]:
            chosen.append({"id": safe_id(row), "mode": mode, "row": row})
    if limit_samples > 0:
        chosen = chosen[: int(limit_samples)]
    return chosen


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


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    y = x.detach().float().reshape(-1)
    if y.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
    }


def safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    return float(torch.dot(af, bf).item() / (denom.item() + eps))


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    kk = min(int(k), int(a.numel()), int(b.numel()))
    if kk <= 0:
        return 0.0
    ai = set(torch.topk(a.detach().float().reshape(-1), kk).indices.cpu().tolist())
    bi = set(torch.topk(b.detach().float().reshape(-1), kk).indices.cpu().tolist())
    return float(len(ai & bi) / float(kk))


def topk_indices_and_values(x: torch.Tensor, k: int) -> Tuple[str, str]:
    kk = min(int(k), int(x.numel()))
    if kk <= 0:
        return "", ""
    vals, idx = torch.topk(x.detach().float().reshape(-1), kk)
    return "|".join(str(int(i)) for i in idx.cpu().tolist()), "|".join(f"{float(v):.6g}" for v in vals.cpu().tolist())


def sum_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=False).clamp_min(eps)


def build_vss_guidance(vis_logits: torch.Tensor, *, mode: str, topk: int, eps: float = 1e-8) -> torch.Tensor:
    top_k_scores, _ = torch.topk(vis_logits, int(topk), dim=-1)
    p = top_k_scores.float().clamp_min(float(eps))
    denom = math.log(float(topk))
    if mode == "entropy":
        score = (-p * torch.log(p)).sum(dim=-1) / denom
    elif mode == "nll":
        score = (-torch.log(p)).sum(dim=-1) / denom
    else:
        raise ValueError(f"unsupported vss_mode={mode!r}")
    return (score / score.sum(dim=0).clamp_min(float(eps))).to(vis_logits.dtype)


def token_boundary(token_text: str) -> bool:
    return bool(token_text) and (token_text.startswith("▁") or token_text.startswith("Ġ") or token_text.startswith(" "))


def token_norm_from_id(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    toks = object_tokens(text)
    if toks:
        return toks[-1]
    raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    toks = object_tokens(raw)
    return toks[-1] if toks else normalize_token(raw)


def token_piece_word(tokenizer: Any, token_id: int) -> str:
    raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    cleaned = raw.replace("▁", "").replace("Ġ", "")
    parts = TOKEN_RE.findall(cleaned.lower())
    if parts:
        return "".join(parts)
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    parts = TOKEN_RE.findall(decoded.lower())
    return "".join(parts)


def build_lost_step_matches(
    tokenizer: Any,
    token_ids: Sequence[int],
    lost_token_by_object: Dict[str, set],
) -> Dict[int, Dict[str, str]]:
    """Map replay token steps to lost objects, allowing subword-split words."""
    lost_by_token: Dict[str, set] = defaultdict(set)
    for obj, toks in lost_token_by_object.items():
        for tok in toks:
            lost_by_token[normalize_token(tok)].add(obj)

    out: Dict[int, Dict[str, str]] = {}
    word_parts: List[str] = []
    word_steps: List[int] = []

    def flush() -> None:
        if not word_steps:
            return
        norm_word = normalize_token("".join(word_parts))
        objects = lost_by_token.get(norm_word, set())
        if objects:
            for st in word_steps:
                out.setdefault(st, {"objects": set(), "words": set()})
                out[st]["objects"].update(objects)
                out[st]["words"].add(norm_word)
        word_parts.clear()
        word_steps.clear()

    for step, token_id in enumerate(token_ids):
        raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        piece = token_piece_word(tokenizer, int(token_id))
        starts_word = raw.startswith("▁") or raw.startswith("Ġ")
        if starts_word and word_steps:
            flush()
        if not piece:
            flush()
            continue
        word_parts.append(piece)
        word_steps.append(int(step))
    flush()

    return {
        step: {
            "objects": "|".join(sorted(value["objects"])),
            "words": "|".join(sorted(value["words"])),
        }
        for step, value in out.items()
    }


def compute_token_visual_row(
    *,
    vl_guidance: torch.Tensor,
    vis_logits: torch.Tensor,
    token_id: int,
    topk: int,
    prefix: str,
) -> Dict[str, Any]:
    raw = vis_logits[:, int(token_id)].float().clamp_min(0.0)
    if float(raw.sum().item()) <= 0:
        dist = torch.ones_like(raw) / float(raw.numel())
    else:
        dist = raw / raw.sum().clamp_min(1e-12)
    gi, gv = topk_indices_and_values(vl_guidance, topk)
    ti, tv = topk_indices_and_values(dist, topk)
    return {
        f"{prefix}_visual_prob_mean": tensor_stats(raw)["mean"],
        f"{prefix}_visual_prob_max": tensor_stats(raw)["max"],
        f"{prefix}_visual_dist_entropy": float((-(dist.clamp_min(1e-12)) * torch.log(dist.clamp_min(1e-12))).sum().item()),
        f"{prefix}_guidance_dot_token_dist": float(torch.dot(vl_guidance.float().reshape(-1), dist.reshape(-1)).item()),
        f"{prefix}_guidance_token_dist_cosine": safe_cosine(vl_guidance, dist),
        f"{prefix}_guidance_token_dist_l1": float(torch.sum(torch.abs(vl_guidance.float() - dist)).item()),
        f"{prefix}_guidance_token_dist_top10_overlap": topk_overlap(vl_guidance, dist, 10),
        f"{prefix}_guidance_token_dist_top50_overlap": topk_overlap(vl_guidance, dist, 50),
        f"{prefix}_guidance_top{topk}_idx": gi,
        f"{prefix}_guidance_top{topk}_val": gv,
        f"{prefix}_token_visual_top{topk}_idx": ti,
        f"{prefix}_token_visual_top{topk}_val": tv,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Teacher-force baseline captions for target lost-object samples and dump "
            "lost-object token confidence plus VGA/PVG guidance alignment diagnostics."
        )
    )
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--oracle-rows-csv", required=True)
    ap.add_argument("--target-col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--sample-id", action="append", default=[])
    ap.add_argument("--samples-per-mode", type=int, default=5)
    ap.add_argument("--limit-samples", type=int, default=15)
    ap.add_argument("--max-caption-tokens", type=int, default=220)
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
    ap.add_argument("--log-vanilla-contrast", type=parse_bool, default=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--topk-patches", type=int, default=10)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
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
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    set_seed(int(args.seed))
    disable_torch_init()

    oracle_rows = read_csv_rows(args.oracle_rows_csv)
    chosen = choose_oracle_samples(
        oracle_rows,
        target_col=args.target_col,
        sample_ids=args.sample_id,
        samples_per_mode=int(args.samples_per_mode),
        limit_samples=int(args.limit_samples),
    )
    questions = read_jsonl_map(args.question_file)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head

    rows: List[Dict[str, Any]] = []
    sample_summaries: List[Dict[str, Any]] = []
    for item in chosen:
        sid = str(item["id"])
        oracle = item["row"]
        q = questions.get(sid)
        if q is None:
            sample_summaries.append({"id": sid, "mode": item["mode"], "error": "missing question row"})
            continue

        lost_objects = split_bar_items(oracle.get("base_only_supported_unique", ""))
        lost_token_by_object = {obj: set(object_tokens(obj)) for obj in lost_objects}
        lost_tokens = set().union(*lost_token_by_object.values()) if lost_token_by_object else set()
        if not lost_tokens:
            sample_summaries.append({"id": sid, "mode": item["mode"], "error": "no lost object tokens"})
            continue

        qs = str(q["question"])
        if model.config.mm_use_im_start_end:
            qs_for_model = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs_for_model = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs_for_model)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_file = q["image"]
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        prompt_attention_mask = torch.ones_like(input_ids[:, :-1], dtype=torch.long, device=input_ids.device)
        step_attention_mask = torch.ones((1, 1), dtype=torch.long, device=input_ids.device)
        target_ids = tokenizer(str(oracle.get("base_caption", "")), add_special_tokens=False, return_tensors="pt").input_ids[0]
        target_ids = target_ids[: int(args.max_caption_tokens)].cuda()
        lost_step_matches = build_lost_step_matches(
            tokenizer,
            [int(x) for x in target_ids.detach().cpu().tolist()],
            lost_token_by_object,
        )

        sample_match_count = 0
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
                prompt_guidance_top_idx, prompt_guidance_top_val = topk_indices_and_values(
                    vl_guidance, int(args.topk_patches)
                )
                past_key_values = prompt_outputs.past_key_values
                vanilla_past_key_values = prompt_outputs.past_key_values
                current_input = input_ids[:, -1:]

                for step, target_id_t in enumerate(target_ids):
                    target_id = int(target_id_t.item())
                    vanilla_logp = None
                    vanilla_probs = None
                    vanilla_top_vals = None
                    vanilla_top_ids = None
                    vanilla_rank = None
                    vanilla_entropy = None
                    if bool(args.log_vanilla_contrast):
                        vanilla_outputs = model(
                            current_input,
                            attention_mask=step_attention_mask,
                            images=image_batch,
                            past_key_values=vanilla_past_key_values,
                            use_cache=True,
                            return_dict=True,
                            use_add=False,
                        )
                        vanilla_logits = vanilla_outputs.logits[:, -1, :].float()
                        vanilla_logp = torch.log_softmax(vanilla_logits, dim=-1)[0]
                        vanilla_probs = torch.softmax(vanilla_logits, dim=-1)[0]
                        vanilla_top_vals, vanilla_top_ids = torch.topk(vanilla_logp, 5)
                        vanilla_rank = int((vanilla_logp > vanilla_logp[target_id]).sum().item() + 1)
                        vanilla_entropy = float((-(vanilla_probs.clamp_min(1e-12)) * torch.log(vanilla_probs.clamp_min(1e-12))).sum().item())
                        vanilla_past_key_values = vanilla_outputs.past_key_values

                    outputs = model(
                        current_input,
                        attention_mask=step_attention_mask,
                        images=image_batch,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                        vl_guidance=vl_guidance,
                        add_layer=list(range(int(args.start_layer), int(args.end_layer) + 1)),
                        attn_coef=float(args.attn_coef),
                        use_add=bool(args.use_add),
                        head_balancing=str(args.head_balancing),
                        attn_norm=bool(args.attn_norm),
                    )
                    logits = outputs.logits[:, -1, :].float()
                    logp = torch.log_softmax(logits, dim=-1)[0]
                    probs = torch.softmax(logits, dim=-1)[0]
                    top_vals, top_ids = torch.topk(logp, 5)
                    rank = int((logp > logp[target_id]).sum().item() + 1)
                    entropy = float((-(probs.clamp_min(1e-12)) * torch.log(probs.clamp_min(1e-12))).sum().item())
                    token_text = tokenizer.convert_ids_to_tokens([target_id])[0]
                    token_decoded = tokenizer.decode([target_id], skip_special_tokens=True)
                    token_norm = token_norm_from_id(tokenizer, target_id)
                    is_lost_token = step in lost_step_matches

                    if is_lost_token:
                        matched_objects = split_bar_items(lost_step_matches[step]["objects"])
                        row: Dict[str, Any] = {
                            "id": sid,
                            "image": image_file,
                            "mode": item["mode"],
                            "step": int(step),
                            "target_token_id": target_id,
                            "target_token_text": token_text,
                            "target_token_decoded": token_decoded,
                            "target_token_norm": token_norm,
                            "matched_word_norm": lost_step_matches[step]["words"],
                            "matched_lost_objects": "|".join(matched_objects),
                            "all_lost_objects": "|".join(lost_objects),
                            "base_only_supported_unique": oracle.get("base_only_supported_unique", ""),
                            "int_only_hallucinated_unique": oracle.get("int_only_hallucinated_unique", ""),
                            "base_caption": oracle.get("base_caption", ""),
                            "int_caption": oracle.get("int_caption", ""),
                            "delta_recall_base_minus_int": safe_float(oracle.get("delta_recall_base_minus_int")),
                            "delta_f1_unique_base_minus_int": safe_float(oracle.get("delta_f1_unique_base_minus_int")),
                            "delta_ci_unique_base_minus_int": safe_float(oracle.get("delta_ci_unique_base_minus_int")),
                            "target_logprob": float(logp[target_id].item()),
                            "target_prob": float(probs[target_id].item()),
                            "target_rank": rank,
                            "target_margin_vs_top1": float(logp[target_id].item() - top_vals[0].item()),
                            "top1_logprob": float(top_vals[0].item()),
                            "top1_token_id": int(top_ids[0].item()),
                            "top1_token_text": tokenizer.convert_ids_to_tokens([int(top_ids[0].item())])[0],
                            "top1_gap": float((top_vals[0] - top_vals[1]).item()) if top_vals.numel() > 1 else 0.0,
                            "entropy": entropy,
                            "prompt_guidance_top_idx": prompt_guidance_top_idx,
                            "prompt_guidance_top_val": prompt_guidance_top_val,
                        }
                        if vanilla_logp is not None:
                            row.update(
                                {
                                    "vanilla_target_logprob": float(vanilla_logp[target_id].item()),
                                    "vanilla_target_prob": float(vanilla_probs[target_id].item()),
                                    "vanilla_target_rank": int(vanilla_rank),
                                    "vanilla_target_margin_vs_top1": float(vanilla_logp[target_id].item() - vanilla_top_vals[0].item()),
                                    "vanilla_top1_logprob": float(vanilla_top_vals[0].item()),
                                    "vanilla_top1_token_id": int(vanilla_top_ids[0].item()),
                                    "vanilla_top1_token_text": tokenizer.convert_ids_to_tokens([int(vanilla_top_ids[0].item())])[0],
                                    "vanilla_top1_gap": float((vanilla_top_vals[0] - vanilla_top_vals[1]).item()) if vanilla_top_vals.numel() > 1 else 0.0,
                                    "vanilla_entropy": vanilla_entropy,
                                    "pvg_minus_vanilla_target_logprob": float(logp[target_id].item() - vanilla_logp[target_id].item()),
                                    "vanilla_minus_pvg_target_logprob": float(vanilla_logp[target_id].item() - logp[target_id].item()),
                                    "pvg_minus_vanilla_target_rank": int(rank - int(vanilla_rank)),
                                    "vanilla_top1_equals_pvg_top1": int(int(vanilla_top_ids[0].item()) == int(top_ids[0].item())),
                                    "vanilla_top1_is_target": int(int(vanilla_top_ids[0].item()) == int(target_id)),
                                    "pvg_top1_is_target": int(int(top_ids[0].item()) == int(target_id)),
                                }
                            )
                        row.update(
                            compute_token_visual_row(
                                vl_guidance=vl_guidance,
                                vis_logits=vis_logits,
                                token_id=target_id,
                                topk=int(args.topk_patches),
                                prefix="before",
                            )
                        )
                        rows.append(row)
                        sample_match_count += 1

                    if float(args.cd_alpha) > 0 and token_boundary(token_text):
                        token_visual = sum_norm(vis_logits[:, target_id].float().clamp_min(0.0))
                        vl_guidance = (1.0 + float(args.cd_alpha)) * vl_guidance.float() - float(args.cd_alpha) * token_visual
                        vl_guidance = F.relu(vl_guidance)
                        vl_guidance = sum_norm(vl_guidance).to(vis_logits.dtype)

                    past_key_values = outputs.past_key_values
                    current_input = target_id_t.reshape(1, 1)

            sample_summaries.append(
                {
                    "id": sid,
                    "mode": item["mode"],
                    "image": image_file,
                    "lost_objects": lost_objects,
                    "lost_tokens": sorted(lost_tokens),
                    "n_target_tokens_replayed": int(len(target_ids)),
                    "n_lost_token_matches": int(sample_match_count),
                }
            )
            print(f"[sample] id={sid} mode={item['mode']} lost_token_matches={sample_match_count}", flush=True)
        except Exception as exc:
            sample_summaries.append({"id": sid, "mode": item["mode"], "image": image_file, "error": repr(exc)})
            print(f"[error] id={sid} {exc!r}", flush=True)

    write_csv(args.out_csv, rows)
    summary = {
        "inputs": {
            **vars(args),
            "diagnostic_note": (
                "This replays the baseline caption under VGA/PVG guidance updates. "
                "It approximates the lost-object time point under a baseline prefix; "
                "it is not the exact free-run VGA prefix after divergence."
            ),
        },
        "counts": {
            "n_selected_samples": len(chosen),
            "n_sample_summaries": len(sample_summaries),
            "n_lost_token_rows": len(rows),
        },
        "samples": sample_summaries,
        "outputs": {
            "lost_object_guidance_csv": os.path.abspath(args.out_csv),
            "summary_json": os.path.abspath(args.out_json),
        },
    }
    write_json(args.out_json, summary)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
