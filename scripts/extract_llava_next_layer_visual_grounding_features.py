#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


LABEL_KEEP = {
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
    "gt_label",
    "answer",
    "label",
}

SUMMARY_FEATURES = [
    "vis_obj_logit_max",
    "vis_obj_logit_top5_mean",
    "vis_obj_logit_mean",
    "vis_obj_logit_std",
    "vis_obj_contra_logit_max",
    "vis_obj_contra_logit_top5_mean",
    "vis_obj_support_logit_max",
    "vis_obj_support_logit_top5_mean",
    "hid_first_decision_to_vision_mean_cos",
    "hid_first_decision_to_vision_max_cos",
    "hid_first_decision_vision_minus_prompt",
    "hid_mean_decision_to_vision_mean_cos",
    "hid_mean_decision_to_vision_max_cos",
    "hid_mean_decision_vision_minus_prompt",
    "hid_answer_to_vision_mean_cos",
    "hid_answer_to_vision_max_cos",
    "hid_answer_vision_minus_prompt",
    "hid_object_answer_to_vision_mean_cos",
    "hid_object_answer_to_vision_max_cos",
    "hid_object_answer_vision_minus_prompt",
]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def maybe_float(value: Any) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def read_jsonl_rows(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_text_map(path: str, text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl_rows(path):
        sid = safe_id(row.get("question_id", row.get("id")))
        if not sid:
            continue
        if text_key == "auto":
            for key in ("output", "text", "answer", "caption"):
                value = str(row.get(key, "")).strip()
                if value:
                    out[sid] = value
                    break
        else:
            out[sid] = str(row.get(text_key, "")).strip()
    return out


def read_label_rows(path: str) -> Dict[str, Dict[str, str]]:
    if not str(path or "").strip():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id", row.get("question_id", row.get("qid"))))
        if not sid:
            continue
        out[sid] = {key: str(row.get(key, "")) for key in LABEL_KEEP if key in row}
    return out


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
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
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(max(0.0, sum((x - mu) ** 2 for x in values) / float(len(values)))))


def binary_auroc(xs: Sequence[float], ys: Sequence[int]) -> float:
    pos = [float(x) for x, y in zip(xs, ys) if int(y) == 1]
    neg = [float(x) for x, y in zip(xs, ys) if int(y) == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / float(len(pos) * len(neg)))


def normalize_text(text: Any) -> str:
    chars: List[str] = []
    for ch in str(text or "").strip().lower():
        chars.append(ch if ch.isalnum() else " ")
    return " ".join("".join(chars).split())


def object_terms_from_sample(sample: Mapping[str, Any]) -> List[str]:
    raw = sample.get("object", sample.get("objects", ""))
    if isinstance(raw, str):
        vals: List[Any] = [raw]
    elif isinstance(raw, Sequence):
        vals = list(raw)
    else:
        vals = []
    out: List[str] = []
    seen = set()
    for value in vals:
        term = str(value or "").strip()
        norm = normalize_text(term)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(term)
    return out


def final_label_from_text(text: str) -> str:
    s = str(text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(part.strip().lower() for part in first.split())
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    return ""


def object_token_id_set(tokenizer: Any, object_terms: Sequence[str]) -> List[int]:
    ids: List[int] = []
    seen = set()
    variants: List[str] = []
    for term in object_terms:
        text = str(term or "").strip()
        if not text:
            continue
        variants.extend([text, text.lower(), text.title(), " " + text, " " + text.lower(), " " + text.title()])
    for text in variants:
        try:
            token_ids = tokenizer(str(text), add_special_tokens=False).input_ids
        except Exception:
            token_ids = []
        for token_id in token_ids:
            tid = int(token_id)
            if tid in seen:
                continue
            try:
                decoded = tokenizer.decode([tid], skip_special_tokens=True)
            except Exception:
                decoded = ""
            if not normalize_text(decoded):
                continue
            seen.add(tid)
            ids.append(tid)
    return sorted(ids)


def mean_hidden(hidden: Any, indices: Sequence[int]) -> Optional[Any]:
    import torch

    valid = [int(i) for i in indices if 0 <= int(i) < int(hidden.shape[0])]
    if not valid:
        return None
    idx = torch.tensor(valid, dtype=torch.long, device=hidden.device)
    return hidden.index_select(0, idx).mean(dim=0)


def cosine_or_zero(left: Optional[Any], right: Optional[Any]) -> float:
    if left is None or right is None:
        return 0.0
    import torch
    import torch.nn.functional as F

    return float(F.cosine_similarity(left.to(torch.float32), right.to(torch.float32), dim=0).item())


def max_cosine_or_zero(query: Optional[Any], candidates: Optional[Any], *, top_k: int = 5) -> Dict[str, float]:
    if query is None or candidates is None or int(candidates.shape[0]) <= 0:
        return {"max": 0.0, "topk_mean": 0.0}
    import torch
    import torch.nn.functional as F

    q = F.normalize(query.to(torch.float32), dim=0)
    c = F.normalize(candidates.to(torch.float32), dim=1)
    sims = torch.matmul(c, q)
    k = min(max(1, int(top_k)), int(sims.numel()))
    top = torch.topk(sims, k=k).values
    return {"max": float(top[0].item()), "topk_mean": float(top.mean().item())}


def summarize_values(values: Any, top_k: int) -> Dict[str, float]:
    import torch

    if values is None or int(values.numel()) <= 0:
        return {"max": 0.0, "topk_mean": 0.0, "mean": 0.0, "std": 0.0}
    vals = values.to(torch.float32).flatten()
    k = min(max(1, int(top_k)), int(vals.numel()))
    top = torch.topk(vals, k=k).values
    return {
        "max": float(vals.max().item()),
        "topk_mean": float(top.mean().item()),
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
    }


def visual_object_logits(
    hidden: Any,
    *,
    lm_head: Any,
    token_ids: Sequence[int],
    top_k: int,
) -> Dict[str, float]:
    import torch

    valid = [int(i) for i in token_ids if 0 <= int(i) < int(lm_head.weight.shape[0])]
    if not valid or int(hidden.shape[0]) <= 0:
        return {"max": 0.0, "topk_mean": 0.0, "mean": 0.0, "std": 0.0}
    idx = torch.tensor(valid, dtype=torch.long, device=hidden.device)
    weight = lm_head.weight.index_select(0, idx).to(device=hidden.device, dtype=torch.float32)
    logits = hidden.to(torch.float32).matmul(weight.t())
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias.index_select(0, idx).to(device=hidden.device, dtype=torch.float32)
    token_scores = torch.logsumexp(logits, dim=-1)
    return summarize_values(token_scores, top_k)


def layer_grounding_trajectory(
    runtime: Any,
    *,
    image: Any,
    question: str,
    candidate_text: str,
    sample: Mapping[str, Any],
    apply_final_norm: bool,
    top_k_visual: int,
    top_k_cos: int,
) -> List[Dict[str, Any]]:
    import torch
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from frgavr_cleanroom.runtime import select_content_indices
    from run_discriminative_meta_strong_online import object_token_indices

    prompt = runtime.prompt_text(question)
    prompt_ids = tokenizer_image_token(prompt, runtime.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
        runtime.device
    )
    cont_ids = runtime.tokenizer(str(candidate_text or ""), add_special_tokens=False, return_tensors="pt").input_ids[0].to(
        runtime.device
    )
    if int(cont_ids.numel()) <= 0:
        raise ValueError("Candidate text tokenization is empty.")
    full_ids = torch.cat([prompt_ids[0], cont_ids], dim=0).unsqueeze(0)
    images_tensor, image_sizes = runtime._process_image(image)
    object_terms = object_terms_from_sample(sample)
    object_token_ids = object_token_id_set(runtime.tokenizer, object_terms)
    candidate_label = final_label_from_text(str(candidate_text or ""))

    with torch.no_grad():
        pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = runtime._prepare_multimodal_expanded_sequence(
            full_ids=full_ids,
            images_tensor=images_tensor,
            image_sizes=image_sizes,
        )
        backbone = runtime.model.get_model() if hasattr(runtime.model, "get_model") else getattr(runtime.model, "model", None)
        if backbone is None:
            raise RuntimeError("Could not resolve language-model backbone.")
        forward_kwargs: Dict[str, Any] = {
            "inputs_embeds": mm_embeds_e,
            "attention_mask": attn_mask_e,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if pos_ids_e is not None:
            forward_kwargs["position_ids"] = pos_ids_e
        if runtime.teacher_force_forward_mode in {"model", "full", "legacy"}:
            outputs = runtime.model(**forward_kwargs)
        else:
            outputs = backbone(**forward_kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Forward did not return hidden_states.")

        labels_exp = labels_e[0]
        text_positions_t = torch.where(labels_exp != int(IGNORE_INDEX))[0]
        if int(text_positions_t.numel()) < int(cont_ids.numel()):
            raise RuntimeError("Expanded sequence is shorter than continuation token count.")
        cont_label_positions = text_positions_t[-int(cont_ids.numel()):]
        decision_positions_t = cont_label_positions - 1
        if int(decision_positions_t.min().item()) < 0:
            raise RuntimeError("Invalid decision positions after expansion.")
        vision_positions_t = torch.where(labels_exp == int(IGNORE_INDEX))[0]
        if int(vision_positions_t.numel()) <= 0:
            raise RuntimeError("No visual token span found in expanded sequence.")

        cont_positions = [int(x) for x in cont_label_positions.tolist()]
        decision_positions = [int(x) for x in decision_positions_t.tolist()]
        vision_positions = [int(x) for x in vision_positions_t.tolist()]
        text_positions = [int(x) for x in text_positions_t.tolist()]
        cont_set = set(cont_positions)
        prompt_positions = [int(x) for x in text_positions if int(x) not in cont_set]
        content_indices = select_content_indices(runtime.tokenizer, cont_ids.detach().cpu())
        content_positions = [
            cont_positions[int(i)]
            for i in content_indices
            if 0 <= int(i) < len(cont_positions)
        ] or list(cont_positions)
        content_decisions = [
            decision_positions[int(i)]
            for i in content_indices
            if 0 <= int(i) < len(decision_positions)
        ] or list(decision_positions)
        ans_object_indices = object_token_indices(runtime.tokenizer, cont_ids.detach().cpu(), object_terms)
        ans_object_positions = [
            cont_positions[int(i)]
            for i in ans_object_indices
            if 0 <= int(i) < len(cont_positions)
        ]

        norm = None
        if bool(apply_final_norm):
            norm = getattr(backbone, "norm", None)
            if norm is None and hasattr(backbone, "model"):
                norm = getattr(backbone.model, "norm", None)

        v_idx = torch.tensor(vision_positions, dtype=torch.long, device=runtime.device)
        rows: List[Dict[str, Any]] = []
        n_hidden = int(len(hidden_states))
        for idx, hidden in enumerate(hidden_states):
            h = hidden[0]
            if norm is not None:
                h = norm(h)
            h = h.to(torch.float32)
            vision_h = h.index_select(0, v_idx)
            prompt_mean = mean_hidden(h, prompt_positions)
            vision_mean = mean_hidden(h, vision_positions)
            first_decision = mean_hidden(h, decision_positions[:1])
            mean_decision = mean_hidden(h, content_decisions)
            mean_answer = mean_hidden(h, content_positions)
            mean_answer_object = mean_hidden(h, ans_object_positions)

            obj = visual_object_logits(vision_h, lm_head=runtime.model.lm_head, token_ids=object_token_ids, top_k=top_k_visual)
            sign = -1.0 if candidate_label == "yes" else 1.0 if candidate_label == "no" else 0.0
            support_sign = -sign
            first_max = max_cosine_or_zero(first_decision, vision_h, top_k=top_k_cos)
            mean_dec_max = max_cosine_or_zero(mean_decision, vision_h, top_k=top_k_cos)
            answer_max = max_cosine_or_zero(mean_answer, vision_h, top_k=top_k_cos)
            object_max = max_cosine_or_zero(mean_answer_object, vision_h, top_k=top_k_cos)
            first_vision = cosine_or_zero(first_decision, vision_mean)
            first_prompt = cosine_or_zero(first_decision, prompt_mean)
            mean_dec_vision = cosine_or_zero(mean_decision, vision_mean)
            mean_dec_prompt = cosine_or_zero(mean_decision, prompt_mean)
            answer_vision = cosine_or_zero(mean_answer, vision_mean)
            answer_prompt = cosine_or_zero(mean_answer, prompt_mean)
            object_vision = cosine_or_zero(mean_answer_object, vision_mean)
            object_prompt = cosine_or_zero(mean_answer_object, prompt_mean)

            rows.append(
                {
                    "layer_index": int(idx),
                    "layer_frac": float(idx / max(1, n_hidden - 1)),
                    "is_final_layer": int(idx == n_hidden - 1),
                    "candidate_label": candidate_label,
                    "n_object_terms": int(len(object_terms)),
                    "n_object_token_ids": int(len(object_token_ids)),
                    "n_answer_object_tokens": int(len(ans_object_positions)),
                    "n_vision_tokens": int(len(vision_positions)),
                    "vis_obj_logit_max": obj["max"],
                    "vis_obj_logit_top5_mean": obj["topk_mean"],
                    "vis_obj_logit_mean": obj["mean"],
                    "vis_obj_logit_std": obj["std"],
                    "vis_obj_contra_logit_max": float(sign * obj["max"]),
                    "vis_obj_contra_logit_top5_mean": float(sign * obj["topk_mean"]),
                    "vis_obj_support_logit_max": float(support_sign * obj["max"]),
                    "vis_obj_support_logit_top5_mean": float(support_sign * obj["topk_mean"]),
                    "hid_first_decision_to_vision_mean_cos": first_vision,
                    "hid_first_decision_to_vision_max_cos": first_max["max"],
                    "hid_first_decision_to_vision_top5_mean_cos": first_max["topk_mean"],
                    "hid_first_decision_to_prompt_cos": first_prompt,
                    "hid_first_decision_vision_minus_prompt": float(first_vision - first_prompt),
                    "hid_mean_decision_to_vision_mean_cos": mean_dec_vision,
                    "hid_mean_decision_to_vision_max_cos": mean_dec_max["max"],
                    "hid_mean_decision_to_vision_top5_mean_cos": mean_dec_max["topk_mean"],
                    "hid_mean_decision_to_prompt_cos": mean_dec_prompt,
                    "hid_mean_decision_vision_minus_prompt": float(mean_dec_vision - mean_dec_prompt),
                    "hid_answer_to_vision_mean_cos": answer_vision,
                    "hid_answer_to_vision_max_cos": answer_max["max"],
                    "hid_answer_to_vision_top5_mean_cos": answer_max["topk_mean"],
                    "hid_answer_to_prompt_cos": answer_prompt,
                    "hid_answer_vision_minus_prompt": float(answer_vision - answer_prompt),
                    "hid_object_answer_to_vision_mean_cos": object_vision,
                    "hid_object_answer_to_vision_max_cos": object_max["max"],
                    "hid_object_answer_to_vision_top5_mean_cos": object_max["topk_mean"],
                    "hid_object_answer_to_prompt_cos": object_prompt,
                    "hid_object_answer_vision_minus_prompt": float(object_vision - object_prompt),
                }
            )
    return rows


def summarize_layers(long_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_layer: Dict[int, Dict[str, Any]] = {}
    for row in long_rows:
        layer_f = maybe_float(row.get("layer_index"))
        harm_f = maybe_float(row.get("harm"))
        help_f = maybe_float(row.get("help"))
        if layer_f is None or harm_f is None or help_f is None:
            continue
        layer = int(layer_f)
        harm = int(harm_f)
        help_ = int(help_f)
        if harm not in {0, 1} or help_ not in {0, 1} or (harm == 0 and help_ == 0):
            continue
        item = by_layer.setdefault(
            layer,
            {
                "ys": [],
                **{f"{name}_values": [] for name in SUMMARY_FEATURES},
                **{f"{name}_harm": [] for name in SUMMARY_FEATURES},
                **{f"{name}_help": [] for name in SUMMARY_FEATURES},
            },
        )
        item["ys"].append(harm)
        for name in SUMMARY_FEATURES:
            value = maybe_float(row.get(name))
            if value is None:
                continue
            item[f"{name}_values"].append(value)
            if harm == 1:
                item[f"{name}_harm"].append(value)
            if help_ == 1:
                item[f"{name}_help"].append(value)
    out: List[Dict[str, Any]] = []
    for layer in sorted(by_layer):
        item = by_layer[layer]
        ys = [int(x) for x in item["ys"]]
        if not ys:
            continue
        summary: Dict[str, Any] = {
            "layer_index": layer,
            "n": len(ys),
            "n_harm": sum(ys),
            "n_help": len(ys) - sum(ys),
        }
        for name in SUMMARY_FEATURES:
            values = [float(x) for x in item[f"{name}_values"]]
            if len(values) != len(ys):
                continue
            auc_high = binary_auroc(values, ys)
            auc_low = binary_auroc([-x for x in values], ys)
            summary[f"harm_{name}_mean"] = mean(item[f"{name}_harm"])
            summary[f"harm_{name}_std"] = std(item[f"{name}_harm"])
            summary[f"help_{name}_mean"] = mean(item[f"{name}_help"])
            summary[f"help_{name}_std"] = std(item[f"{name}_help"])
            summary[f"{name}_auroc"] = max(auc_high, auc_low)
            summary[f"{name}_direction"] = "high" if auc_high >= auc_low else "low"
            summary[f"{name}_raw_auroc_high"] = auc_high
        out.append(summary)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract LLaVA-NeXT layer-wise visual grounding and hidden-similarity replay features."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--label_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default="/home/kms/models/llama3-llava-next-8b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_llama_3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llava_next_root", default="/home/kms/LLaVA-NeXT")
    ap.add_argument("--llava_next_torch_type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--llava_next_attn_implementation", default="sdpa", choices=["none", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--apply_final_norm", type=parse_bool, default=True)
    ap.add_argument("--only_label_rows", type=parse_bool, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k_visual", type=int, default=5)
    ap.add_argument("--top_k_cos", type=int, default=5)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    long_csv = os.path.join(out_dir, "layer_visual_grounding_long.csv")
    summary_csv = os.path.join(out_dir, "layer_visual_grounding_summary.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json, flush=True)
        return

    labels = read_label_rows(os.path.abspath(args.label_rows_csv))
    label_ids = set(labels)
    questions = read_jsonl_rows(os.path.abspath(args.question_file), limit=0)
    if bool(args.only_label_rows):
        questions = [row for row in questions if safe_id(row.get("question_id", row.get("id"))) in label_ids]
    if int(args.limit) > 0:
        questions = questions[: int(args.limit)]
    pred_map = read_jsonl_text_map(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_key))

    from frgavr_cleanroom.llava_next_runtime import OfficialLlavaNextRuntime

    runtime = OfficialLlavaNextRuntime(
        llava_next_root=str(args.llava_next_root),
        model_path=str(args.model_path),
        model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
        torch_type=str(args.llava_next_torch_type),
        attn_implementation=str(args.llava_next_attn_implementation),
    )

    long_rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_intervention = 0
    timings: List[float] = []
    for idx, sample in enumerate(questions):
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        label_meta = labels.get(sid, {})
        candidate_text = str(pred_map.get(sid, "") or label_meta.get("intervention_text", "")).strip()
        meta = {
            "id": sid,
            "image": image_name,
            "question": question,
            "intervention_text": candidate_text,
            **label_meta,
        }
        try:
            if not sid:
                raise ValueError("missing sample id")
            if not image_name:
                raise ValueError("missing image")
            if not question:
                raise ValueError("missing question")
            if not candidate_text:
                n_missing_intervention += 1
                raise ValueError("missing intervention prediction")
            image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            image = runtime.load_image(image_path)
            t0 = time.perf_counter()
            rows = layer_grounding_trajectory(
                runtime,
                image=image,
                question=question,
                candidate_text=candidate_text,
                sample=sample,
                apply_final_norm=bool(args.apply_final_norm),
                top_k_visual=int(args.top_k_visual),
                top_k_cos=int(args.top_k_cos),
            )
            timings.append(time.perf_counter() - t0)
            for row in rows:
                long_rows.append({**meta, **row, "score_error": ""})
        except Exception as exc:
            n_errors += 1
            long_rows.append({**meta, "layer_index": "", "score_error": str(exc), "score_error_traceback": traceback.format_exc()})
            print(f"[error] id={sid} {exc!r}", flush=True)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[layer-ground] {idx + 1}/{len(questions)}", flush=True)

    summary_rows = summarize_layers(long_rows)
    write_csv(long_csv, long_rows)
    write_csv(summary_csv, summary_rows)
    write_json(
        summary_json,
        {
            "mode": "llava_next_layer_visual_grounding_replay",
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "intervention_pred_key": str(args.intervention_pred_key),
                "label_rows_csv": os.path.abspath(args.label_rows_csv),
                "model_path": str(args.model_path),
                "conv_mode": str(args.conv_mode),
                "llava_next_root": str(args.llava_next_root),
                "llava_next_attn_implementation": str(args.llava_next_attn_implementation),
                "apply_final_norm": bool(args.apply_final_norm),
                "top_k_visual": int(args.top_k_visual),
                "top_k_cos": int(args.top_k_cos),
                "only_label_rows": bool(args.only_label_rows),
            },
            "counts": {
                "n_rows": int(len(questions)),
                "n_errors": int(n_errors),
                "n_missing_intervention": int(n_missing_intervention),
                "n_long_rows": int(len(long_rows)),
                "n_summary_layers": int(len(summary_rows)),
            },
            "timing": {
                "feature_total_sec": float(sum(timings)),
                "feature_mean_ms": float(1000.0 * sum(timings) / max(1, len(timings))),
            },
            "outputs": {
                "long_csv": long_csv,
                "summary_csv": summary_csv,
            },
        },
    )
    print("[saved]", long_csv)
    print("[saved]", summary_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
