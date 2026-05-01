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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_layers(value: str) -> List[int]:
    out: List[int] = []
    for raw in str(value or "").split(","):
        s = raw.strip().lower()
        if not s:
            continue
        if s in {"final", "last"}:
            out.append(-1)
        else:
            out.append(int(s))
    return out or [8, -1]


def layer_tag(layer: int, n_hidden: int) -> str:
    idx = resolve_hidden_index(layer, n_hidden)
    return "final" if idx == n_hidden - 1 else f"l{idx}"


def resolve_hidden_index(layer: int, n_hidden: int) -> int:
    if int(layer) < 0:
        return max(0, int(n_hidden) + int(layer))
    return min(max(0, int(layer)), max(0, int(n_hidden) - 1))


def label_margin_from_logits(
    logits: torch.Tensor,
    *,
    token_ids: Mapping[str, Sequence[int]],
    candidate_label: str,
) -> Dict[str, float]:
    import torch
    from run_discriminative_meta_strong_online import logsumexp_ids

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    yes_lp_t = logsumexp_ids(log_probs, token_ids.get("yes", []))
    no_lp_t = logsumexp_ids(log_probs, token_ids.get("no", []))
    yes_lp = float(yes_lp_t.item())
    no_lp = float(no_lp_t.item())
    yes_minus_no = float((yes_lp_t - no_lp_t).item())
    yes_prob = float(torch.sigmoid(yes_lp_t - no_lp_t).item())
    no_prob = float(1.0 - yes_prob)
    if candidate_label == "yes":
        cand_lp = yes_lp
        alt_lp = no_lp
        cand_margin = yes_minus_no
        cand_prob = yes_prob
    elif candidate_label == "no":
        cand_lp = no_lp
        alt_lp = yes_lp
        cand_margin = -yes_minus_no
        cand_prob = no_prob
    else:
        cand_lp = -100.0
        alt_lp = -100.0
        cand_margin = 0.0
        cand_prob = 0.0
    return {
        "yes_lp": yes_lp,
        "no_lp": no_lp,
        "yes_minus_no": yes_minus_no,
        "yes_prob_binary": yes_prob,
        "no_prob_binary": no_prob,
        "candidate_label_lp": float(cand_lp),
        "alt_label_lp": float(alt_lp),
        "candidate_minus_alt": float(cand_margin),
        "candidate_prob_binary": float(cand_prob),
        "margin_abs": float(abs(yes_minus_no)),
    }


def logit_lens_features(
    runtime: Any,
    *,
    image: Image.Image,
    question: str,
    candidate_text: str,
    layers: Sequence[int],
    mode_prefix: str,
    apply_final_norm: bool,
) -> Dict[str, Any]:
    import torch
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from run_discriminative_meta_strong_online import final_label_from_text, yesno_token_id_sets

    prompt = runtime.prompt_text(question)
    prompt_ids = tokenizer_image_token(
        prompt,
        runtime.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(runtime.device)
    cont_ids = runtime.tokenizer(
        str(candidate_text or ""),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].to(runtime.device)
    if int(cont_ids.numel()) <= 0:
        raise ValueError("Candidate text tokenization is empty.")

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
            raise RuntimeError("Could not resolve official LLaVA-NeXT language-model backbone.")
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
            raise RuntimeError("LLaVA-NeXT forward did not return hidden_states.")

        labels_exp = labels_e[0]
        text_positions = torch.where(labels_exp != int(IGNORE_INDEX))[0]
        if int(text_positions.numel()) < int(cont_ids.numel()):
            raise RuntimeError("Expanded sequence is shorter than continuation token count.")
        cont_label_positions = text_positions[-int(cont_ids.numel()):]
        decision_positions = cont_label_positions - 1
        if int(decision_positions.min().item()) < 0:
            raise RuntimeError("Invalid decision positions after expansion.")
        first_decision_pos = int(decision_positions[0].item())

        out: Dict[str, Any] = {
            f"{mode_prefix}_candidate_label": candidate_label,
            f"{mode_prefix}_n_hidden_states": int(len(hidden_states)),
        }
        layer_margins: Dict[str, float] = {}
        norm = None
        if bool(apply_final_norm):
            norm = getattr(backbone, "norm", None)
            if norm is None and hasattr(backbone, "model"):
                norm = getattr(backbone.model, "norm", None)
        for layer in layers:
            idx = resolve_hidden_index(int(layer), len(hidden_states))
            tag = layer_tag(int(layer), len(hidden_states))
            h = hidden_states[idx][:, first_decision_pos, :]
            if norm is not None and idx != len(hidden_states) - 1:
                h = norm(h)
            logits = runtime.model.lm_head(h).float()[0]
            vals = label_margin_from_logits(logits, token_ids=token_ids, candidate_label=candidate_label)
            for key, value in vals.items():
                out[f"{mode_prefix}_{tag}_{key}"] = float(value)
            layer_margins[tag] = float(vals["candidate_minus_alt"])

        tags = [layer_tag(int(layer), len(hidden_states)) for layer in layers]
        if len(tags) >= 2:
            first = tags[0]
            last = tags[-1]
            out[f"{mode_prefix}_flip_{last}_minus_{first}_candidate_minus_alt"] = float(
                layer_margins[last] - layer_margins[first]
            )
            out[f"{mode_prefix}_abs_flip_{last}_minus_{first}_candidate_minus_alt"] = float(
                abs(layer_margins[last] - layer_margins[first])
            )
        if len(tags) >= 3:
            diffs = [layer_margins[tags[i + 1]] - layer_margins[tags[i]] for i in range(len(tags) - 1)]
            out[f"{mode_prefix}_trajectory_slope_mean"] = float(sum(diffs) / float(len(diffs)))
            out[f"{mode_prefix}_trajectory_abs_step_mean"] = float(sum(abs(x) for x in diffs) / float(len(diffs)))
            signs = [1 if layer_margins[tag] >= 0.0 else -1 for tag in tags]
            out[f"{mode_prefix}_trajectory_sign_flip_count"] = int(
                sum(1 for i in range(len(signs) - 1) if signs[i + 1] != signs[i])
            )
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract constrained LLaVA-NeXT logit-lens teacher-forcing replay features.")
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--label_rows_csv", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llava_next_root", default="/home/kms/LLaVA-NeXT")
    ap.add_argument("--model_path", default="/home/kms/models/llama3-llava-next-8b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_llama_3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llava_next_torch_type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--llava_next_attn_implementation", default="sdpa", choices=["none", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--layers", default="8,final")
    ap.add_argument("--modes", default="orig,black", help="Comma-separated subset of orig,black,gray,blur,noise.")
    ap.add_argument(
        "--apply_final_norm",
        type=parse_bool,
        default=True,
        help="Apply the LLaMA final RMSNorm before the LM head for intermediate-layer logit lens.",
    )
    ap.add_argument("--blur_radius", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rows_csv = os.path.join(out_dir, "logit_lens_replay_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json, flush=True)
        return

    from extract_llava_next_vision_ablation_replay_features import (
        ablate_image,
        load_prediction_text_map,
        read_jsonl_rows,
        read_label_rows,
        safe_id,
    )
    from frgavr_cleanroom.llava_next_runtime import OfficialLlavaNextRuntime

    layers = parse_layers(str(args.layers))
    modes = [x.strip().lower() for x in str(args.modes).split(",") if x.strip()]
    if not modes:
        modes = ["orig"]
    for mode in modes:
        if mode not in {"orig", "black", "gray", "blur", "noise"}:
            raise ValueError(f"unsupported mode: {mode}")

    questions = read_jsonl_rows(os.path.abspath(args.question_file), limit=int(args.limit))
    intervention_map = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_key))
    label_map = read_label_rows(str(args.label_rows_csv))
    runtime = OfficialLlavaNextRuntime(
        llava_next_root=str(args.llava_next_root),
        model_path=str(args.model_path),
        model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
        torch_type=str(args.llava_next_torch_type),
        attn_implementation=str(args.llava_next_attn_implementation),
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_intervention = 0
    timings: List[float] = []
    for idx, sample in enumerate(questions):
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        candidate_text = str(intervention_map.get(sid, "")).strip()
        row: Dict[str, Any] = {
            "id": sid,
            "image": image_name,
            "question": question,
            "intervention_text": candidate_text,
            "score_error": "",
        }
        row.update(label_map.get(sid, {}))
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
            t0 = time.perf_counter()
            image = runtime.load_image(image_path)
            for mode in modes:
                replay_image = image if mode == "orig" else ablate_image(
                    image,
                    mode,
                    blur_radius=float(args.blur_radius),
                    seed=int(args.seed),
                    sample_id_value=sid,
                )
                row.update(
                    logit_lens_features(
                        runtime,
                        image=replay_image,
                        question=question,
                        candidate_text=candidate_text,
                        layers=layers,
                        mode_prefix=f"lens_{mode}",
                        apply_final_norm=bool(args.apply_final_norm),
                    )
                )
            dt = time.perf_counter() - t0
            row["feature_ms"] = float(dt * 1000.0)
            timings.append(float(dt))
        except Exception as exc:
            n_errors += 1
            row["score_error"] = str(exc)
            row["score_error_traceback"] = traceback.format_exc()
            print(f"[error] id={sid} {exc!r}", flush=True)
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[logit-lens] {idx + 1}/{len(questions)}", flush=True)

    write_csv(rows_csv, rows)
    write_json(
        summary_json,
        {
            "mode": "llava_next_logit_lens_replay",
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "intervention_pred_key": str(args.intervention_pred_key),
                "label_rows_csv": os.path.abspath(args.label_rows_csv) if str(args.label_rows_csv).strip() else "",
                "model_path": str(args.model_path),
                "conv_mode": str(args.conv_mode),
                "llava_next_root": str(args.llava_next_root),
                "llava_next_torch_type": str(args.llava_next_torch_type),
                "llava_next_attn_implementation": str(args.llava_next_attn_implementation),
                "layers": layers,
                "modes": modes,
                "apply_final_norm": bool(args.apply_final_norm),
            },
            "counts": {
                "n_rows": int(len(questions)),
                "n_errors": int(n_errors),
                "n_missing_intervention": int(n_missing_intervention),
                "n_with_labels": int(sum(1 for row in rows if str(row.get("harm", "")).strip() != "")),
            },
            "timing": {
                "feature_total_sec": float(sum(timings)),
                "feature_mean_ms": float(1000.0 * sum(timings) / max(1, len(timings))),
            },
            "outputs": {
                "rows_csv": rows_csv,
            },
        },
    )
    print("[saved]", rows_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
