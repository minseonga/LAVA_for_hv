#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from extract_decision_margin_layer_trajectories import (  # noqa: E402
    binary_auroc,
    label_margin_from_logits,
    mean,
    parse_bool,
    read_jsonl_rows,
    read_jsonl_text_map,
    read_label_rows,
    safe_id,
    std,
    write_csv,
    write_json,
)


OBJECT_SUMMARY_FEATURES = [
    ("obj_target_gap_min", "obj_target_gap_min"),
    ("obj_target_gap_mean", "obj_target_gap_mean"),
    ("obj_first_target_gap", "obj_first_target_gap"),
    ("obj_target_lp_min", "obj_target_lp_min"),
    ("obj_target_lp_mean", "obj_target_lp_mean"),
    ("obj_entropy_mean", "obj_entropy_mean"),
    ("obj_top1_match_rate", "obj_top1_match_rate"),
    ("decision_candidate_margin", "decision_candidate_minus_alt"),
]


OBJECT_ALIASES = {
    "airplane": ["plane", "aircraft", "aeroplane"],
    "baseball bat": ["bat"],
    "baseball glove": ["glove"],
    "cell phone": ["phone", "mobile phone", "cellphone", "smartphone"],
    "couch": ["sofa"],
    "dining table": ["table"],
    "fire hydrant": ["hydrant"],
    "hair drier": ["hair dryer", "dryer"],
    "motorcycle": ["motorbike", "bike"],
    "potted plant": ["plant"],
    "remote": ["remote control"],
    "sports ball": ["ball"],
    "stop sign": ["sign"],
    "tennis racket": ["racket", "racquet"],
    "traffic light": ["light"],
    "tv": ["television", "tv monitor", "monitor", "screen"],
    "wine glass": ["glass"],
}


def extract_object_phrase(sample: Mapping[str, Any], question: str) -> str:
    obj = sample.get("object", "")
    if isinstance(obj, list) and obj:
        return str(obj[0]).strip()
    if isinstance(obj, str) and obj.strip():
        return obj.strip()
    q = str(question or "").strip().lower().rstrip("?")
    m = re.match(r"^is there (?:a|an)\s+(.+?)\s+in the image$", q)
    if m:
        return m.group(1).strip()
    m = re.match(r"^is there\s+(.+?)\s+in the image$", q)
    if m:
        return m.group(1).strip()
    return ""


def unique_variants(text: str) -> List[str]:
    raw = str(text or "").strip()
    forms = [raw]
    if raw and not raw.endswith("s"):
        forms.append(raw + "s")
    if raw.endswith("y") and len(raw) > 1:
        forms.append(raw[:-1] + "ies")
    if raw.endswith("s") and len(raw) > 1:
        forms.append(raw[:-1])
    variants: List[str] = []
    for form in forms:
        for cased in (form, form.lower(), form.upper(), form.capitalize(), form.title()):
            for item in (cased, " " + cased):
                if item and item not in variants:
                    variants.append(item)
    return variants


def object_phrase_candidates(object_phrase: str) -> List[str]:
    phrase = str(object_phrase or "").strip()
    candidates: List[str] = []
    for item in [phrase, phrase.lower()]:
        if item and item not in candidates:
            candidates.append(item)
    for alias in OBJECT_ALIASES.get(phrase.lower(), []):
        if alias not in candidates:
            candidates.append(alias)
    return candidates


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return None


def token_ids_for(tokenizer: Any, text: str) -> List[int]:
    ids = tokenizer(str(text), add_special_tokens=False, return_tensors="pt").input_ids[0].detach().cpu().tolist()
    return [int(x) for x in ids]


def find_object_token_indices(tokenizer: Any, cont_ids: Sequence[int], object_phrase: str) -> tuple[List[int], str]:
    cont = [int(x) for x in cont_ids]
    phrase = str(object_phrase or "").strip()
    if not phrase:
        return [], "empty_object"

    tried: List[List[int]] = []
    for phrase_candidate in object_phrase_candidates(phrase):
        for variant in unique_variants(phrase_candidate):
            ids = token_ids_for(tokenizer, variant)
            if not ids or ids in tried:
                continue
            tried.append(ids)
            start = find_subsequence(cont, ids)
            if start is not None:
                mode = "phrase_exact" if phrase_candidate == phrase else "phrase_alias"
                return list(range(start, start + len(ids))), mode

    matched: List[int] = []
    for word in [w for cand in object_phrase_candidates(phrase) for w in re.split(r"\s+", cand) if w]:
        found_word = False
        for variant in unique_variants(word):
            ids = token_ids_for(tokenizer, variant)
            if not ids:
                continue
            start = find_subsequence(cont, ids)
            if start is not None:
                matched.extend(range(start, start + len(ids)))
                found_word = True
                break
        if not found_word:
            return sorted(set(matched)), "partial_words"
    return sorted(set(matched)), "word_exact" if matched else "not_found"


def object_layer_trajectory(
    runtime: Any,
    *,
    image: Any,
    question: str,
    candidate_text: str,
    object_phrase: str,
    apply_final_norm: bool,
) -> List[Dict[str, Any]]:
    import torch
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from run_discriminative_meta_strong_online import final_label_from_text, yesno_token_id_sets

    prompt = runtime.prompt_text(question)
    prompt_ids = tokenizer_image_token(prompt, runtime.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
        runtime.device
    )
    cont_ids = runtime.tokenizer(str(candidate_text or ""), add_special_tokens=False, return_tensors="pt").input_ids[0].to(
        runtime.device
    )
    if int(cont_ids.numel()) <= 0:
        raise ValueError("Candidate text tokenization is empty.")

    object_indices, object_match_mode = find_object_token_indices(
        runtime.tokenizer,
        cont_ids.detach().cpu().tolist(),
        object_phrase,
    )
    if not object_indices:
        raise ValueError(f"Could not locate object tokens in candidate text: object={object_phrase!r}")

    full_ids = torch.cat([prompt_ids[0], cont_ids], dim=0).unsqueeze(0)
    images_tensor, image_sizes = runtime._process_image(image)
    token_ids = yesno_token_id_sets(runtime.tokenizer)
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
        target_model = runtime.model if runtime.teacher_force_forward_mode in {"model", "full", "legacy"} else backbone
        try:
            outputs = target_model(**forward_kwargs)
        except TypeError as exc:
            if "position_ids" not in str(exc):
                raise
            forward_kwargs.pop("position_ids", None)
            outputs = target_model(**forward_kwargs)

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Forward did not return hidden_states.")
        if runtime.teacher_force_forward_mode in {"model", "full", "legacy"}:
            final_logits = getattr(outputs, "logits", None)
            if final_logits is None:
                raise RuntimeError("Model forward did not return logits.")
        else:
            final_logits = runtime.model.lm_head(outputs[0])
        final_logits = final_logits.float()[0]

        labels_exp = labels_e[0]
        text_positions = torch.where(labels_exp != int(IGNORE_INDEX))[0]
        if int(text_positions.numel()) < int(cont_ids.numel()):
            raise RuntimeError("Expanded sequence is shorter than continuation token count.")
        cont_label_positions = text_positions[-int(cont_ids.numel()):]
        decision_positions = cont_label_positions - 1
        if int(decision_positions.min().item()) < 0:
            raise RuntimeError("Invalid decision positions after expansion.")

        target_ids = labels_exp[cont_label_positions].long()
        object_pick = torch.tensor(object_indices, dtype=torch.long, device=target_ids.device)
        object_decision_positions = decision_positions.index_select(0, object_pick)
        object_target_ids = target_ids.index_select(0, object_pick)
        first_decision_pos = int(decision_positions[0].item())

        norm = None
        if bool(apply_final_norm):
            norm = getattr(backbone, "norm", None)
            if norm is None and hasattr(backbone, "model"):
                norm = getattr(backbone.model, "norm", None)

        decoded_tokens = [runtime.tokenizer.decode([int(cont_ids[i].item())]) for i in object_indices]
        rows: List[Dict[str, Any]] = []
        n_hidden = int(len(hidden_states))
        for idx, hidden in enumerate(hidden_states):
            is_final = idx == n_hidden - 1
            if is_final:
                decision_logits = final_logits[first_decision_pos]
                token_logits = final_logits[object_decision_positions]
            else:
                h_decision = hidden[:, first_decision_pos, :]
                h_object = hidden[:, object_decision_positions, :]
                if norm is not None:
                    h_decision = norm(h_decision)
                    h_object = norm(h_object)
                decision_logits = runtime.model.lm_head(h_decision).float()[0]
                token_logits = runtime.model.lm_head(h_object).float()[0]

            decision_vals = label_margin_from_logits(decision_logits, token_ids=token_ids, candidate_label=candidate_label)
            log_probs = torch.log_softmax(token_logits.float(), dim=-1)
            probs = torch.softmax(token_logits.float(), dim=-1)
            token_ent = -(probs * log_probs).sum(dim=-1)
            top2_vals, top2_idx = torch.topk(token_logits.float(), k=2, dim=-1)
            top1_logit = top2_vals[:, 0]
            top2_logit = top2_vals[:, 1]
            top1_id = top2_idx[:, 0]
            target_logit = token_logits.gather(1, object_target_ids.unsqueeze(-1)).squeeze(-1).float()
            best_other_logit = torch.where(top1_id == object_target_ids, top2_logit, top1_logit).float()
            target_gap = target_logit - best_other_logit
            target_lp = log_probs.gather(1, object_target_ids.unsqueeze(-1)).squeeze(-1)
            top1_match = (top1_id == object_target_ids).float()

            rows.append(
                {
                    "layer_index": int(idx),
                    "layer_frac": float(idx / max(1, n_hidden - 1)),
                    "is_final_layer": int(is_final),
                    "candidate_label": candidate_label,
                    "object_phrase": object_phrase,
                    "object_match_mode": object_match_mode,
                    "object_token_indices": " ".join(str(i) for i in object_indices),
                    "object_token_text": "|".join(decoded_tokens),
                    "object_n_tokens": int(len(object_indices)),
                    "decision_candidate_minus_alt": float(decision_vals["candidate_minus_alt"]),
                    "decision_candidate_prob_binary": float(decision_vals["candidate_prob_binary"]),
                    "decision_margin_abs": float(decision_vals["margin_abs"]),
                    "obj_target_gap_min": float(target_gap.min().item()),
                    "obj_target_gap_mean": float(target_gap.mean().item()),
                    "obj_first_target_gap": float(target_gap[0].item()),
                    "obj_last_target_gap": float(target_gap[-1].item()),
                    "obj_target_lp_min": float(target_lp.min().item()),
                    "obj_target_lp_mean": float(target_lp.mean().item()),
                    "obj_first_target_lp": float(target_lp[0].item()),
                    "obj_entropy_mean": float(token_ent.mean().item()),
                    "obj_top1_match_rate": float(top1_match.mean().item()),
                    "obj_target_logit_mean": float(target_logit.mean().item()),
                    "obj_best_other_logit_mean": float(best_other_logit.mean().item()),
                }
            )
    return rows


def summarize_layers(long_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_layer: Dict[int, Dict[str, Any]] = {}
    for row in long_rows:
        try:
            layer = int(row["layer_index"])
            harm = int(float(row.get("harm", 0) or 0))
            help_ = int(float(row.get("help", 0) or 0))
        except Exception:
            continue
        if harm not in {0, 1} or help_ not in {0, 1} or (harm == 0 and help_ == 0):
            continue
        item = by_layer.setdefault(
            layer,
            {
                "ys": [],
                **{f"{name}_values": [] for name, _ in OBJECT_SUMMARY_FEATURES},
                **{f"{name}_harm": [] for name, _ in OBJECT_SUMMARY_FEATURES},
                **{f"{name}_help": [] for name, _ in OBJECT_SUMMARY_FEATURES},
            },
        )
        item["ys"].append(harm)
        for name, key in OBJECT_SUMMARY_FEATURES:
            try:
                value = float(row[key])
            except Exception:
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
        summary: Dict[str, Any] = {
            "layer_index": layer,
            "n": len(ys),
            "n_harm": sum(ys),
            "n_help": len(ys) - sum(ys),
        }
        for name, _ in OBJECT_SUMMARY_FEATURES:
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
    ap = argparse.ArgumentParser(description="Extract layer-wise object-token replay support under teacher forcing.")
    ap.add_argument("--runtime_backend", choices=["llava15_cleanroom", "llava_next_official"], required=True)
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_pred_jsonl", default="")
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--label_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llava_next_root", default="/home/kms/LLaVA-NeXT")
    ap.add_argument("--llava_next_torch_type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--llava_next_attn_implementation", default="sdpa", choices=["none", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--apply_final_norm", type=parse_bool, default=True)
    ap.add_argument("--only_label_rows", type=parse_bool, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    long_csv = os.path.join(out_dir, "object_token_layer_trajectory_long.csv")
    summary_csv = os.path.join(out_dir, "object_token_layer_trajectory_summary.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json)
        return

    labels = read_label_rows(os.path.abspath(args.label_rows_csv))
    label_ids = set(labels)
    questions = read_jsonl_rows(os.path.abspath(args.question_file), limit=0)
    if bool(args.only_label_rows):
        questions = [row for row in questions if safe_id(row.get("question_id", row.get("id"))) in label_ids]
    if int(args.limit) > 0:
        questions = questions[: int(args.limit)]
    pred_map = (
        read_jsonl_text_map(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_key))
        if str(args.intervention_pred_jsonl or "").strip()
        else {}
    )

    if str(args.runtime_backend) == "llava_next_official":
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
    else:
        from frgavr_cleanroom.runtime import CleanroomLlavaRuntime

        runtime = CleanroomLlavaRuntime(
            model_path=str(args.model_path),
            model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
            conv_mode=str(args.conv_mode),
            device=str(args.device),
        )

    long_rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_intervention = 0
    n_missing_object = 0
    timings: List[float] = []
    for idx, sample in enumerate(questions):
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        object_phrase = extract_object_phrase(sample, question)
        label_meta = labels.get(sid, {})
        candidate_text = str(pred_map.get(sid, "") or label_meta.get("intervention_text", "")).strip()
        meta = {
            "id": sid,
            "image": image_name,
            "question": question,
            "object_phrase": object_phrase,
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
            if not object_phrase:
                n_missing_object += 1
                raise ValueError("missing object phrase")
            if not candidate_text:
                n_missing_intervention += 1
                raise ValueError("missing intervention prediction")
            image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            t0 = time.perf_counter()
            image = runtime.load_image(image_path)
            traj = object_layer_trajectory(
                runtime,
                image=image,
                question=question,
                candidate_text=candidate_text,
                object_phrase=object_phrase,
                apply_final_norm=bool(args.apply_final_norm),
            )
            timings.append(time.perf_counter() - t0)
            for row in traj:
                long_rows.append({**meta, **row, "score_error": ""})
        except Exception as exc:
            n_errors += 1
            long_rows.append(
                {
                    **meta,
                    "layer_index": "",
                    "score_error": str(exc),
                    "score_error_traceback": traceback.format_exc(),
                }
            )
            print(f"[error] id={sid} {exc!r}", flush=True)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[object-layer] {idx + 1}/{len(questions)}", flush=True)

    summary_rows = summarize_layers(long_rows)
    write_csv(long_csv, long_rows)
    write_csv(summary_csv, summary_rows)
    summary = {
        "mode": "object_token_layer_trajectory",
        "inputs": {
            "runtime_backend": str(args.runtime_backend),
            "question_file": os.path.abspath(args.question_file),
            "image_folder": os.path.abspath(args.image_folder),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl)
            if str(args.intervention_pred_jsonl or "").strip()
            else "",
            "label_rows_csv": os.path.abspath(args.label_rows_csv),
            "model_path": str(args.model_path),
            "conv_mode": str(args.conv_mode),
            "apply_final_norm": bool(args.apply_final_norm),
            "only_label_rows": bool(args.only_label_rows),
        },
        "counts": {
            "n_questions": int(len(questions)),
            "n_long_rows": int(len(long_rows)),
            "n_errors": int(n_errors),
            "n_missing_intervention": int(n_missing_intervention),
            "n_missing_object": int(n_missing_object),
            "n_summary_layers": int(len(summary_rows)),
        },
        "timing": {
            "mean_sec": float(mean(timings)) if timings else 0.0,
            "total_sec": float(sum(timings)),
        },
        "outputs": {
            "long_csv": long_csv,
            "summary_csv": summary_csv,
            "summary_json": summary_json,
        },
    }
    write_json(summary_json, summary)
    print("[saved]", long_csv)
    print("[saved]", summary_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
