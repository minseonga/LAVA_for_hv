#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_vga_lost_object_guidance import (
    build_vss_guidance,
    compute_token_visual_row,
    parse_bool,
    patch_legacy_transformers_bloom_masks,
    safe_float,
    safe_id,
    sum_norm,
    token_norm_from_id,
)


DEFAULT_TARGET_COL = "oracle_recall_gain_f1_nondecrease_ci_unique_noworse"


def read_jsonl_rows(path: str, *, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def prediction_text(row: Dict[str, Any], text_key: str = "auto") -> str:
    if text_key and text_key != "auto":
        return str(row.get(text_key, "")).strip()
    for key in ("text", "answer", "caption", "prediction", "output"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def read_prediction_map(path: str, *, text_key: str = "auto") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl_rows(path):
        sid = safe_id(row)
        if sid:
            out[sid] = prediction_text(row, text_key=text_key)
    return out


def entropy_from_logp(logp: torch.Tensor) -> float:
    probs = torch.exp(logp.float())
    return float((-(probs.clamp_min(1e-12)) * logp.float()).sum().item())


def logp_rank(logp: torch.Tensor, token_id: int) -> int:
    return int((logp.float() > logp[int(token_id)].float()).sum().item() + 1)


def topk_summary(tokenizer: Any, logp: torch.Tensor, *, topk: int, prefix: str) -> Dict[str, Any]:
    vals, ids = torch.topk(logp.float(), int(topk))
    id_list = [int(x) for x in ids.detach().cpu().tolist()]
    toks = tokenizer.convert_ids_to_tokens(id_list)
    out = {
        f"{prefix}_top1_token_id": id_list[0],
        f"{prefix}_top1_token_text": toks[0],
        f"{prefix}_top1_logprob": float(vals[0].item()),
        f"{prefix}_top1_gap": float((vals[0] - vals[1]).item()) if len(id_list) > 1 else 0.0,
        f"{prefix}_top{topk}_ids": "|".join(str(x) for x in id_list),
        f"{prefix}_top{topk}_tokens": "|".join(toks),
    }
    return out


def topk_overlap_from_logp(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    kk = min(int(k), int(a.numel()), int(b.numel()))
    if kk <= 0:
        return 0.0
    ai = set(torch.topk(a.float(), kk).indices.detach().cpu().tolist())
    bi = set(torch.topk(b.float(), kk).indices.detach().cpu().tolist())
    return float(len(ai & bi) / float(kk))


def kl_from_logp(p_logp: torch.Tensor, q_logp: torch.Tensor) -> float:
    p = torch.exp(p_logp.float())
    return float((p * (p_logp.float() - q_logp.float())).sum().item())


def guidance_stats(guidance: torch.Tensor) -> Dict[str, float]:
    g = guidance.detach().float().reshape(-1).clamp_min(0.0)
    total = g.sum().clamp_min(1e-12)
    p = g / total
    vals, _ = torch.sort(p, descending=True)
    return {
        "guidance_entropy": float((-(p.clamp_min(1e-12)) * torch.log(p.clamp_min(1e-12))).sum().item()),
        "guidance_max": float(vals[0].item()) if vals.numel() else 0.0,
        "guidance_top5_mass": float(vals[:5].sum().item()) if vals.numel() else 0.0,
        "guidance_top10_mass": float(vals[:10].sum().item()) if vals.numel() else 0.0,
        "guidance_active_frac": float((g > 1e-8).sum().item() / float(max(1, g.numel()))),
    }


def tokenize_caption(tokenizer: Any, caption: str, *, max_new_tokens: int) -> List[int]:
    ids = tokenizer(str(caption), add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    if int(max_new_tokens) > 0:
        ids = ids[: int(max_new_tokens)]
    return [int(x) for x in ids]


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / float(len(vals))) if vals else 0.0


def max_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(max(vals)) if vals else 0.0


def min_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(min(vals)) if vals else 0.0


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def summarize_sample(step_rows: Sequence[Dict[str, Any]], *, sid: str, image: str, caption: str) -> Dict[str, Any]:
    def vals(key: str) -> List[float]:
        out: List[float] = []
        for row in step_rows:
            value = safe_float(row.get(key), None)
            if value is not None:
                out.append(float(value))
        return out

    top1_changed = [safe_float(row.get("top1_changed"), 0.0) for row in step_rows]
    noadd_top1_drop = vals("noadd_top1_logprob_drop_by_add")
    add_top1_boost = vals("add_top1_logprob_boost_over_noadd")
    actual_boost = vals("actual_add_minus_noadd_logprob")
    entropy_delta = vals("entropy_delta_add_minus_noadd")
    kl_add_noadd = vals("kl_add_to_noadd")
    kl_noadd_add = vals("kl_noadd_to_add")
    top10_overlap = vals("top10_overlap_noadd_add")
    eos_boost = vals("eos_add_minus_noadd_logprob")
    guidance_entropy = vals("guidance_entropy")
    guidance_max = vals("guidance_max")
    actual_cos = vals("actual_guidance_token_dist_cosine")
    actual_top10 = vals("actual_guidance_token_dist_top10_overlap")
    actual_visual_max = vals("actual_visual_prob_max")
    update_l1 = vals("guidance_update_l1")

    first_changed = next((int(row["step"]) for row in step_rows if int(safe_float(row.get("top1_changed"), 0.0)) == 1), "")
    first_big_drop = next((int(row["step"]) for row in step_rows if safe_float(row.get("noadd_top1_logprob_drop_by_add"), 0.0) >= 1.0), "")
    early = [row for row in step_rows if int(row.get("step", 0)) < 15]
    tail_start = max(0, len(step_rows) - 20)
    tail = [row for row in step_rows if int(row.get("step", 0)) >= tail_start]

    def step_mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
        return mean(safe_float(row.get(key), None) for row in rows)

    return {
        "id": sid,
        "image": image,
        "caption": caption,
        "n_steps": int(len(step_rows)),
        "proc_top1_change_rate": mean(top1_changed),
        "proc_first_top1_changed_step": first_changed,
        "proc_noadd_top1_drop_mean": mean(noadd_top1_drop),
        "proc_noadd_top1_drop_max": max_or_zero(noadd_top1_drop),
        "proc_noadd_top1_drop_ge_0p5_count": int(sum(v >= 0.5 for v in noadd_top1_drop)),
        "proc_noadd_top1_drop_ge_1p0_count": int(sum(v >= 1.0 for v in noadd_top1_drop)),
        "proc_noadd_top1_drop_ge_2p0_count": int(sum(v >= 2.0 for v in noadd_top1_drop)),
        "proc_first_noadd_top1_drop_ge_1p0_step": first_big_drop,
        "proc_add_top1_boost_mean": mean(add_top1_boost),
        "proc_add_top1_boost_max": max_or_zero(add_top1_boost),
        "proc_actual_token_boost_mean": mean(actual_boost),
        "proc_actual_token_boost_min": min_or_zero(actual_boost),
        "proc_actual_token_suppressed_count": int(sum(v <= -0.5 for v in actual_boost)),
        "proc_entropy_delta_mean": mean(entropy_delta),
        "proc_entropy_delta_max": max_or_zero(entropy_delta),
        "proc_kl_add_to_noadd_mean": mean(kl_add_noadd),
        "proc_kl_noadd_to_add_mean": mean(kl_noadd_add),
        "proc_top10_overlap_mean": mean(top10_overlap),
        "proc_top10_overlap_min": min_or_zero(top10_overlap),
        "proc_eos_boost_mean": mean(eos_boost),
        "proc_eos_boost_max": max_or_zero(eos_boost),
        "proc_guidance_entropy_mean": mean(guidance_entropy),
        "proc_guidance_max_mean": mean(guidance_max),
        "proc_actual_guidance_cosine_mean": mean(actual_cos),
        "proc_actual_guidance_cosine_min": min_or_zero(actual_cos),
        "proc_actual_guidance_top10_overlap_mean": mean(actual_top10),
        "proc_actual_visual_prob_max_mean": mean(actual_visual_max),
        "proc_actual_visual_prob_max_max": max_or_zero(actual_visual_max),
        "proc_guidance_update_l1_mean": mean(update_l1),
        "proc_guidance_update_l1_max": max_or_zero(update_l1),
        "proc_early_top1_change_rate": step_mean(early, "top1_changed"),
        "proc_early_noadd_top1_drop_mean": step_mean(early, "noadd_top1_logprob_drop_by_add"),
        "proc_early_kl_add_to_noadd_mean": step_mean(early, "kl_add_to_noadd"),
        "proc_tail_top1_change_rate": step_mean(tail, "top1_changed"),
        "proc_tail_noadd_top1_drop_mean": step_mean(tail, "noadd_top1_logprob_drop_by_add"),
        "proc_tail_kl_add_to_noadd_mean": step_mean(tail, "kl_add_to_noadd"),
    }


def build_labels(oracle_rows_csv: str, target_col: str) -> Dict[str, Dict[str, Any]]:
    if not str(oracle_rows_csv or "").strip():
        return {}
    labels: Dict[str, Dict[str, Any]] = {}
    for row in read_csv_rows(oracle_rows_csv):
        sid = safe_id(row)
        if not sid:
            continue
        delta_f1 = safe_float(row.get("delta_f1_unique_base_minus_int"), 0.0)
        delta_recall = safe_float(row.get("delta_recall_base_minus_int"), 0.0)
        delta_ci = safe_float(row.get("delta_ci_unique_base_minus_int"), 0.0)
        delta_chair_s = safe_float(row.get("delta_chair_s_base_minus_int"), 0.0)
        gain = flag(row.get(target_col))
        harm = delta_f1 < -1e-12 or delta_ci > 1e-12 or delta_chair_s > 1e-12
        labels[sid] = {
            "fallback_gain": int(gain),
            "fallback_harm": int(harm),
            "fallback_same": int(not gain and not harm),
            "fallback_utility": delta_f1 + 0.25 * delta_recall - max(0.0, delta_ci) - 0.2 * max(0.0, delta_chair_s),
            "delta_f1": delta_f1,
            "delta_recall": delta_recall,
            "delta_ci_unique": delta_ci,
            "delta_chair_s": delta_chair_s,
        }
    return labels


def auc_high(values: Sequence[Optional[float]], labels: Sequence[bool]) -> Optional[Tuple[float, str, float, int, int]]:
    pairs = [(float(v), bool(label)) for v, label in zip(values, labels) if v is not None]
    pos = [value for value, label in pairs if label]
    neg = [value for value, label in pairs if not label]
    if not pos or not neg:
        return None
    score = 0.0
    total = 0
    for pval in pos:
        for nval in neg:
            if pval > nval:
                score += 1.0
            elif pval == nval:
                score += 0.5
            total += 1
    raw = score / float(total)
    return max(raw, 1.0 - raw), ("high" if raw >= 0.5 else "low"), raw, len(pos), len(neg)


def add_label_metrics(feature_rows: Sequence[Dict[str, Any]], labels: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not labels or not feature_rows:
        return []
    metric_rows: List[Dict[str, Any]] = []
    numeric_cols = [
        key
        for key in feature_rows[0]
        if key.startswith("proc_") and all(safe_float(row.get(key), None) is not None for row in feature_rows)
    ]
    for col in numeric_cols:
        values = [safe_float(row.get(col), None) for row in feature_rows]
        for label_col in ("fallback_gain", "fallback_harm"):
            labs = [bool(labels.get(str(row["id"]), {}).get(label_col, 0)) for row in feature_rows]
            res = auc_high(values, labs)
            if res is None:
                continue
            auc, direction, raw, n_pos, n_neg = res
            pos_values = [v for v, lab in zip(values, labs) if lab and v is not None]
            neg_values = [v for v, lab in zip(values, labs) if (not lab) and v is not None]
            metric_rows.append(
                {
                    "comparison": f"{label_col}_vs_rest",
                    "feature": col,
                    "direction": direction,
                    "auc": auc,
                    "auc_high": raw,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "pos_mean": mean(pos_values),
                    "neg_mean": mean(neg_values),
                }
            )
    return sorted(metric_rows, key=lambda row: float(row["auc"]), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Replay VGA/PVG intervention captions and compare local no-add vs add logits "
            "under the intervention prefix. Features are intervention-process-only; "
            "oracle rows are optional and used only for offline correlation metrics."
        )
    )
    ap.add_argument("--vga-root", default="VGA_origin")
    ap.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--intervention-pred-jsonl", required=True)
    ap.add_argument("--pred-text-key", default="auto")
    ap.add_argument("--oracle-rows-csv", default="")
    ap.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    ap.add_argument("--sample-id", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
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
    ap.add_argument("--out-features-csv", required=True)
    ap.add_argument("--out-metrics-csv", default="")
    ap.add_argument("--out-summary-json", required=True)
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

    question_rows = read_jsonl_rows(args.question_file, limit=int(args.limit))
    if args.sample_id:
        sample_ids = {
            str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else str(x)
            for x in args.sample_id
        }
        question_rows = [row for row in question_rows if safe_id(row) in sample_ids]
    pred_map = read_prediction_map(args.intervention_pred_jsonl, text_key=args.pred_text_key)
    labels = build_labels(args.oracle_rows_csv, args.target_col)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head
    eos_id = model.generation_config.eos_token_id

    step_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    n_errors = 0

    for idx, q in enumerate(question_rows):
        sid = safe_id(q)
        image_file = str(q.get("image", "")).strip()
        question = str(q.get("text", q.get("question", ""))).strip()
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
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_for_model)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            prompt_attention_mask = torch.ones_like(input_ids[:, :-1], dtype=torch.long, device=input_ids.device)
            step_attention_mask = torch.ones((1, 1), dtype=torch.long, device=input_ids.device)
            target_tokens = tokenize_caption(tokenizer, caption, max_new_tokens=int(args.max_new_tokens))
            if eos_id is not None and (not target_tokens or target_tokens[-1] != int(eos_id)):
                # Do not force EOS, but keep EOS pressure as a diagnostic at all steps.
                pass

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

                pvg_past = prompt_outputs.past_key_values
                pvg_input = input_ids[:, -1:]

                for step, actual_id in enumerate(target_tokens):
                    noadd_outputs = model(
                        pvg_input,
                        attention_mask=step_attention_mask,
                        images=image_batch,
                        past_key_values=pvg_past,
                        use_cache=True,
                        return_dict=True,
                        use_add=False,
                    )
                    add_outputs = model(
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
                    no_logp = torch.log_softmax(noadd_outputs.logits[:, -1, :].float(), dim=-1)[0]
                    ad_logp = torch.log_softmax(add_outputs.logits[:, -1, :].float(), dim=-1)[0]
                    no_top = int(torch.argmax(no_logp).item())
                    ad_top = int(torch.argmax(ad_logp).item())
                    actual_id = int(actual_id)

                    row: Dict[str, Any] = {
                        "id": sid,
                        "image": image_file,
                        "step": int(step),
                        "actual_token_id": actual_id,
                        "actual_token_text": tokenizer.convert_ids_to_tokens([actual_id])[0],
                        "actual_token_norm": token_norm_from_id(tokenizer, actual_id),
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

                    token_text = tokenizer.convert_ids_to_tokens([actual_id])[0]
                    if float(args.cd_alpha) > 0 and token_text.startswith("▁"):
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
            if sid in labels:
                feature.update(labels[sid])
            feature_rows.append(feature)
        except Exception as exc:
            n_errors += 1
            feature_rows.append({"id": sid, "image": image_file, "caption": caption, "error": repr(exc)})
            print(f"[error] id={sid} {exc!r}", flush=True)

        if (idx + 1) % 25 == 0:
            print(f"[process] {idx + 1}/{len(question_rows)}", flush=True)

    metric_rows = add_label_metrics([row for row in feature_rows if not row.get("error")], labels)

    write_csv(args.out_steps_csv, step_rows)
    write_csv(args.out_features_csv, feature_rows)
    if args.out_metrics_csv:
        write_csv(args.out_metrics_csv, metric_rows)
    write_json(
        args.out_summary_json,
        {
            "inputs": vars(args),
            "counts": {
                "n_questions": len(question_rows),
                "n_step_rows": len(step_rows),
                "n_feature_rows": len(feature_rows),
                "n_errors": n_errors,
                "n_metrics": len(metric_rows),
            },
            "top_metrics": metric_rows[:30],
            "outputs": {
                "steps_csv": os.path.abspath(args.out_steps_csv),
                "features_csv": os.path.abspath(args.out_features_csv),
                "metrics_csv": os.path.abspath(args.out_metrics_csv) if args.out_metrics_csv else "",
                "summary_json": os.path.abspath(args.out_summary_json),
            },
        },
    )
    print("[saved]", os.path.abspath(args.out_steps_csv))
    print("[saved]", os.path.abspath(args.out_features_csv))
    if args.out_metrics_csv:
        print("[saved]", os.path.abspath(args.out_metrics_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
