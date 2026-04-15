#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def patch_legacy_transformers_bloom_masks() -> None:
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


def import_vga_greedy_sample() -> Any:
    import transformers

    original_from_pretrained = transformers.AutoTokenizer.from_pretrained

    class _PlaceholderTokenizer:
        def convert_ids_to_tokens(self, token_ids: Any) -> List[str]:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.detach().cpu().reshape(-1).tolist()
            elif isinstance(token_ids, int):
                token_ids = [token_ids]
            return ["" for _ in token_ids]

    def patched_from_pretrained(pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(pretrained_model_name_or_path) == "path/to/llava-v1.5-7b":
            return _PlaceholderTokenizer()
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        import vcd_utils.greedy_sample as greedy_sample
    finally:
        transformers.AutoTokenizer.from_pretrained = original_from_pretrained
    return greedy_sample


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("question_id") or row.get("image_id") or row.get("id") or "").strip()
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def pass_thresholds(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if safe_float(row.get("risk_object_count"), 0.0) < float(args.risk_min_object_count):
        return False
    if not str(row.get("risk_top_object", "")).strip():
        return False
    if safe_float(row.get("risk_top_yes_prob"), 1.0) > float(args.risk_max_yes_prob):
        return False
    if safe_float(row.get("risk_top_lp_margin"), 0.0) > float(args.risk_max_lp_margin):
        return False
    if safe_float(row.get("risk_second_minus_top_yes_prob"), 0.0) < float(args.risk_min_second_gap):
        return False
    return True


def token_variants(obj: str) -> List[str]:
    base = str(obj or "").strip()
    if not base:
        return []
    variants = [base, " " + base, base.capitalize(), " " + base.capitalize()]
    if not base.endswith("s"):
        variants += [base + "s", " " + base + "s"]
    out: List[str] = []
    seen: Set[str] = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def content_token_ids(tokenizer: Any, text: str) -> List[int]:
    toks = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    out: List[int] = []
    for tok in toks:
        tid = int(tok)
        if tid < 0 or tid in special:
            continue
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            decoded = ""
        # Some LLaMA tokenizers emit an explicit leading-space token for strings
        # like " car". Suppressing that token changes the whole decoding path.
        if not str(decoded).strip():
            continue
        out.append(tid)
    return out


def suppression_token_ids(tokenizer: Any, obj: str, mode: str) -> List[int]:
    ids: Set[int] = set()
    for variant in token_variants(obj):
        toks = content_token_ids(tokenizer, variant)
        if not toks:
            continue
        if mode == "single_token":
            if len(toks) == 1:
                ids.add(toks[0])
        elif mode == "first_token":
            ids.add(toks[0])
        elif mode == "all_tokens":
            ids.update(toks)
        else:
            raise ValueError(f"unsupported suppress_mode={mode!r}")
    return sorted(ids)


class TokenSuppressionProcessor(LogitsProcessor):
    def __init__(self, token_ids: Sequence[int], bias: float) -> None:
        self.token_ids = [int(t) for t in token_ids]
        self.bias = float(bias)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.token_ids:
            scores[:, self.token_ids] = scores[:, self.token_ids] + self.bias
        return scores


def build_vss_guidance(vis_logits: torch.Tensor, *, topk: int, eps: float = 1e-8) -> torch.Tensor:
    top_k_scores, _ = torch.topk(vis_logits, int(topk), dim=-1)
    p = top_k_scores.float().clamp_min(float(eps))
    score = (-p * torch.log(p)).sum(dim=-1) / math.log(float(topk))
    return (score / score.sum(dim=0).clamp_min(float(eps))).to(vis_logits.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VGA/LLaVA captioning with object-token logit suppression for selected samples.")
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--risk-features-csv", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--out-summary-json", default="")
    ap.add_argument("--conv-mode", default="llava_v1")
    ap.add_argument("--max_gen_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use_add", type=parse_bool, default=True)
    ap.add_argument("--cd_alpha", type=float, default=0.02)
    ap.add_argument("--attn_coef", type=float, default=0.2)
    ap.add_argument("--start_layer", type=int, default=2)
    ap.add_argument("--end_layer", type=int, default=15)
    ap.add_argument("--head_balancing", default="simg", choices=["vattn", "battn", "simg", "simv", "simb", "simb-simg", "none"])
    ap.add_argument("--attn_norm", type=parse_bool, default=False)
    ap.add_argument("--sampling", type=parse_bool, default=False)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--vss_topk", type=int, default=10)
    ap.add_argument("--risk_max_yes_prob", type=float, default=0.40)
    ap.add_argument("--risk_max_lp_margin", type=float, default=999.0)
    ap.add_argument("--risk_min_second_gap", type=float, default=0.0)
    ap.add_argument("--risk_min_object_count", type=int, default=1)
    ap.add_argument("--suppress_mode", choices=["single_token", "first_token", "all_tokens"], default="first_token")
    ap.add_argument("--suppress_bias", type=float, default=-100.0)
    ap.add_argument("--skip_without_suppress_ids", type=parse_bool, default=True)
    args = ap.parse_args()

    if os.path.exists(args.answers_file):
        raise FileExistsError(f"answers file already exists: {args.answers_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)

    questions_all = read_jsonl(args.question_file, limit=int(args.limit))
    risk_by_id = {safe_id(row): row for row in read_csv_rows(args.risk_features_csv)}
    selected: List[Dict[str, Any]] = []
    for q in questions_all:
        sid = safe_id(q)
        risk = risk_by_id.get(sid)
        if risk is None or not pass_thresholds(risk, args):
            continue
        item = dict(q)
        item["_risk"] = risk
        selected.append(item)

    repo_root = Path(__file__).resolve().parents[1]
    vga_root = Path(args.vga_root)
    if not vga_root.is_absolute():
        vga_root = (repo_root / vga_root).resolve()
    sys.path.insert(0, str(vga_root))
    sys.path.insert(0, str(vga_root / "llava"))

    patch_legacy_transformers_bloom_masks()
    from transformers import set_seed

    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    greedy_sample = import_vga_greedy_sample()
    greedy_sample.evolve_greedy_sampling()

    set_seed(int(args.seed))
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    greedy_sample.tokenizer = tokenizer
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head

    n_written = 0
    n_skipped_no_token_ids = 0
    selected_meta: List[Dict[str, Any]] = []
    with open(args.answers_file, "w", encoding="utf-8") as ans_file:
        for line in tqdm(selected):
            risk = dict(line["_risk"])
            risk_obj = str(risk.get("risk_top_object", "")).strip()
            suppress_ids = suppression_token_ids(tokenizer, risk_obj, str(args.suppress_mode))
            if not suppress_ids and bool(args.skip_without_suppress_ids):
                n_skipped_no_token_ids += 1
                continue

            image_file = line["image"]
            qs = line["question"]
            if model.config.mm_use_im_start_end:
                qs_for_model = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs_for_model = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_for_model)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            _ = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                outputs = model(
                    input_ids[:, :-1],
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits
                vis_logits = F.softmax(logits[0, 35:611, :], dim=-1)
                vl_guidance = build_vss_guidance(vis_logits, topk=int(args.vss_topk))
                output_ids = model.generate(
                    input_ids[:, -1:],
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    past_key_values=outputs.past_key_values,
                    vl_guidance=vl_guidance,
                    vis_logits=vis_logits,
                    cd_alpha=float(args.cd_alpha),
                    add_layer=list(range(int(args.start_layer), int(args.end_layer) + 1)),
                    attn_coef=float(args.attn_coef),
                    use_add=bool(args.use_add),
                    head_balancing=args.head_balancing,
                    attn_norm=bool(args.attn_norm),
                    logits_processor=LogitsProcessorList([TokenSuppressionProcessor(suppress_ids, float(args.suppress_bias))]),
                    do_sample=True,
                    sampling=bool(args.sampling),
                    num_beams=1,
                    max_new_tokens=int(args.max_gen_len),
                    use_cache=True,
                )

            text = tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)[0]
            text = text.split("ASSISTANT:")[-1].strip()
            if text.endswith(stop_str):
                text = text[: -len(stop_str)]
            text = text.strip()
            row = {
                "question_id": line.get("question_id"),
                "question": line.get("question"),
                "output": text,
                "label": line.get("label", ""),
                "prompt": prompt,
                "model_id": model_name,
                "image": image_file,
                "image_id": line.get("image_id", line.get("question_id")),
                "negative_objects": risk_obj,
                "suppressed_object": risk_obj,
                "suppressed_token_ids": "|".join(str(x) for x in suppress_ids),
                "suppressed_token_texts": "|".join(tokenizer.decode([x], skip_special_tokens=True) for x in suppress_ids),
                "suppress_mode": str(args.suppress_mode),
                "suppress_bias": float(args.suppress_bias),
                "risk_top_yes_prob": risk.get("risk_top_yes_prob", ""),
                "risk_top_lp_margin": risk.get("risk_top_lp_margin", ""),
            }
            ans_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            ans_file.flush()
            n_written += 1
            selected_meta.append(
                {
                    k: row[k]
                    for k in (
                        "question_id",
                        "suppressed_object",
                        "suppressed_token_ids",
                        "suppressed_token_texts",
                        "risk_top_yes_prob",
                    )
                }
            )

    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": vars(args),
                "counts": {
                    "n_questions": len(questions_all),
                    "n_selected_by_threshold": len(selected),
                    "n_written": n_written,
                    "n_skipped_no_token_ids": n_skipped_no_token_ids,
                },
                "selected": selected_meta[:200],
                "outputs": {"answers_file": os.path.abspath(args.answers_file)},
            },
        )


if __name__ == "__main__":
    main()
