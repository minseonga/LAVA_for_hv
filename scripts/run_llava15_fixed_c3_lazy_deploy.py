#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frgavr_cleanroom.runtime import (  # noqa: E402
    CleanroomLlavaRuntime,
    load_label_map,
    load_prediction_text_map,
    load_question_rows,
    safe_id,
    select_content_indices,
    write_jsonl,
)
from pnp_deploy.discriminative_fixed_c import (  # noqa: E402
    FixedCTransitionController,
    parse_yes_no,
    score_or_blank,
)
from pnp_deploy.discriminative_meta import write_csv, write_json  # noqa: E402
from scripts.run_discriminative_meta_strong_online import (  # noqa: E402
    cheap_features_from_pack,
    object_token_indices,
    object_terms_from_sample,
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


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


def prepare_vga_origin(vga_root: str, model_path: str) -> None:
    root = Path(vga_root)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    for path in (str(root), str(root / "eval")):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    patch_legacy_transformers_bloom_masks()

    import transformers

    original_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def patched_from_pretrained(pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        if str(pretrained_model_name_or_path) == "path/to/llava-v1.5-7b":
            return original_from_pretrained(model_path, *args, **kwargs)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        from vcd_utils.greedy_sample import evolve_greedy_sampling
    finally:
        transformers.AutoTokenizer.from_pretrained = original_from_pretrained

    evolve_greedy_sampling()


def read_jsonl_text_map(path: str, text_key: str) -> Dict[str, str]:
    if not str(path or "").strip():
        return {}
    return load_prediction_text_map(os.path.abspath(path), text_key=str(text_key))


def image_name_from_row(row: Mapping[str, Any]) -> str:
    image = str(row.get("image", "")).strip()
    if image:
        return image
    image_id = str(row.get("image_id", "")).strip()
    if image_id:
        return image_id if image_id.endswith(".jpg") else f"{image_id}.jpg"
    return ""


def question_from_row(row: Mapping[str, Any]) -> str:
    return str(row.get("text", row.get("question", ""))).strip()


def stop_string(conv_mode: str) -> str:
    from llava.conversation import SeparatorStyle, conv_templates

    conv = conv_templates[conv_mode].copy()
    return conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


def clean_decoded_text(text: str, conv_mode: str) -> str:
    out = str(text or "").split("ASSISTANT:")[-1].strip()
    stop = stop_string(conv_mode)
    if stop and out.endswith(stop):
        out = out[: -len(stop)]
    return out.strip()


def vga_visual_guidance(
    runtime: CleanroomLlavaRuntime,
    *,
    image_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    object_terms: Sequence[str],
) -> tuple[Any, Any, Any]:
    with torch.inference_mode():
        outputs = runtime.model(
            input_ids[:, :-1],
            images=image_tensor.unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
    logits = outputs.logits
    vis_logits = F.softmax(logits[0, 35:611, :], dim=-1)

    if object_terms:
        object_ids = [
            runtime.tokenizer(str(obj), add_special_tokens=False, return_tensors="pt").input_ids[0]
            for obj in object_terms
            if str(obj or "").strip()
        ]
    else:
        object_ids = []

    if object_ids:
        grounding = []
        for obj in object_ids:
            ids = obj.to(vis_logits.device)
            vl = vis_logits[:, ids]
            vl = vl[:, 0]
            grounding.append(vl)
        guide = torch.stack(grounding, dim=0).max(0).values
        guide = guide / guide.sum(0)
    else:
        top_k_scores, _ = torch.topk(vis_logits, 10, dim=-1)
        top_k_scores = top_k_scores.float()
        probabilities = -top_k_scores * torch.log(top_k_scores + 1e-8) / torch.log(torch.tensor(10.0))
        entropy = probabilities.sum(-1)
        guide = entropy / entropy.sum(0)
        guide = guide.to(vis_logits.dtype)

    return outputs.past_key_values, vis_logits, guide


def generate_vga_like(
    runtime: CleanroomLlavaRuntime,
    *,
    image: Any,
    question: str,
    object_terms: Sequence[str],
    max_new_tokens: int,
    use_add: bool,
    cd_alpha: float,
    attn_coef: float,
    start_layer: int,
    end_layer: int,
    head_balancing: str,
    attn_norm: bool,
    sampling: bool,
) -> str:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token

    prompt = runtime.prompt_text(question)
    input_ids = tokenizer_image_token(
        prompt,
        runtime.tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(runtime.device)
    image_tensor, _ = runtime._process_image(image)
    past_key_values, vis_logits, vl_guidance = vga_visual_guidance(
        runtime,
        image_tensor=image_tensor,
        input_ids=input_ids,
        object_terms=object_terms,
    )
    with torch.inference_mode():
        output_ids = runtime.model.generate(
            input_ids[:, -1:],
            images=image_tensor.unsqueeze(0),
            past_key_values=past_key_values,
            vl_guidance=vl_guidance,
            vis_logits=vis_logits,
            cd_alpha=float(cd_alpha),
            add_layer=list(range(int(start_layer), int(end_layer) + 1)),
            attn_coef=float(attn_coef),
            use_add=bool(use_add),
            head_balancing=str(head_balancing),
            attn_norm=bool(attn_norm),
            do_sample=True,
            sampling=bool(sampling),
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
        )
    decoded = runtime.tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)[0]
    return clean_decoded_text(decoded, runtime.conv_mode)


def feature_row_for_method_answer(
    runtime: CleanroomLlavaRuntime,
    *,
    image: Any,
    sample_id: str,
    image_name: str,
    question: str,
    method_text: str,
    object_terms: Sequence[str],
    lp_tail_quantile: float,
    lp_tail_eps: float,
    lp_len_corr_alpha: float,
) -> Dict[str, Any]:
    pack = runtime.teacher_force_candidate(
        image=image,
        question=question,
        candidate_text=method_text,
        output_attentions=False,
        output_hidden_states=False,
    )
    content_indices = select_content_indices(runtime.tokenizer, pack.cont_ids)
    object_indices = object_token_indices(runtime.tokenizer, pack.cont_ids, object_terms)
    return cheap_features_from_pack(
        runtime=runtime,
        pack=pack,
        sample_id=sample_id,
        image_name=image_name,
        question=question,
        content_indices=content_indices,
        object_terms=object_terms,
        object_indices=object_indices,
        lp_tail_quantile=float(lp_tail_quantile),
        lp_tail_eps=float(lp_tail_eps),
        lp_len_corr_alpha=float(lp_len_corr_alpha),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Lazy deployment runner for LLaVA-1.5 fixed-C3 transition-split RAPIC. "
            "It generates the method answer, replays it for both directional scores, "
            "and only generates the baseline answer when at least one score reaches tau."
        )
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--fixed_json", required=True)
    ap.add_argument("--target", default="vga_llava15")
    ap.add_argument("--dataset", default="mscoco")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vga_root", default="/home/kms/LLaVA_calibration/VGA_origin")
    ap.add_argument("--method", default="vga", choices=["vga", "baseline"])
    ap.add_argument(
        "--deployment_order",
        default="score_first_lazy",
        choices=["score_first_lazy", "baseline_first_replay_on_changed"],
        help=(
            "score_first_lazy replays the method answer before optional baseline generation. "
            "baseline_first_replay_on_changed generates baseline first and skips replay when "
            "the parsed baseline/method answers are unchanged."
        ),
    )
    ap.add_argument("--method_pred_jsonl", default="", help="Optional cached method predictions for parity/debug.")
    ap.add_argument("--baseline_pred_jsonl", default="", help="Optional cached baseline predictions for parity/debug.")
    ap.add_argument("--pred_key", default="auto")
    ap.add_argument("--gt_csv", default="")
    ap.add_argument("--gt_id_col", default="id")
    ap.add_argument("--gt_label_col", default="answer")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--baseline_max_new_tokens", type=int, default=8)
    ap.add_argument("--vga_use_add", type=parse_bool, default=True)
    ap.add_argument("--vga_cd_alpha", type=float, default=0.02)
    ap.add_argument("--vga_attn_coef", type=float, default=0.2)
    ap.add_argument("--vga_start_layer", type=int, default=2)
    ap.add_argument("--vga_end_layer", type=int, default=15)
    ap.add_argument("--vga_head_balancing", default="simg")
    ap.add_argument("--vga_attn_norm", type=parse_bool, default=False)
    ap.add_argument("--vga_sampling", type=parse_bool, default=False)
    ap.add_argument("--lp_tail_quantile", type=float, default=0.10)
    ap.add_argument("--lp_tail_eps", type=float, default=1e-6)
    ap.add_argument("--lp_len_corr_alpha", type=float, default=0.35)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json, flush=True)
        return

    prepare_vga_origin(str(args.vga_root), str(args.model_path))
    controller = FixedCTransitionController.from_fixed_json(
        os.path.abspath(args.fixed_json),
        target=str(args.target),
        dataset=str(args.dataset),
    )
    questions = load_question_rows(os.path.abspath(args.question_file), limit=int(args.limit))
    method_map = read_jsonl_text_map(str(args.method_pred_jsonl), str(args.pred_key))
    baseline_map = read_jsonl_text_map(str(args.baseline_pred_jsonl), str(args.pred_key))
    gt_map = (
        load_label_map(os.path.abspath(args.gt_csv), id_col=str(args.gt_id_col), label_col=str(args.gt_label_col))
        if str(args.gt_csv or "").strip()
        else {}
    )

    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )
    try:
        runtime.model.model.lm_head = runtime.model.lm_head
    except Exception:
        pass

    feature_rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    final_preds: List[Dict[str, Any]] = []
    n_errors = 0
    n_method_generated = 0
    n_method_cached = 0
    n_baseline_generated = 0
    n_baseline_cached = 0
    n_baseline_skipped = 0
    n_replay_score_computed = 0
    n_replay_score_skipped = 0
    n_answer_changed = 0
    n_answer_unchanged = 0
    n_parse_failure = 0
    t_total = time.perf_counter()

    for idx, sample in enumerate(questions):
        sample_t0 = time.perf_counter()
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = image_name_from_row(sample)
        question = question_from_row(sample)
        object_terms = object_terms_from_sample(sample)
        gt_label = str(gt_map.get(sid, "")).strip().lower()
        row: Dict[str, Any] = {
            "id": sid,
            "question_id": sid,
            "image": image_name,
            "question": question,
            "object_terms": "; ".join(object_terms),
            "gt_label": gt_label,
            "score_error": "",
            "method_cached": 0,
            "method_generated_live": 0,
            "baseline_triggered": 0,
            "baseline_cached": 0,
            "baseline_generated_live": 0,
            "baseline_skipped": 0,
            "replay_score_computed": 0,
            "replay_score_skipped": 0,
            "answer_changed": "",
            "elapsed_image_load_sec": 0.0,
            "elapsed_method_sec": 0.0,
            "elapsed_replay_score_sec": 0.0,
            "elapsed_baseline_sec": 0.0,
            "elapsed_decision_sec": 0.0,
            "elapsed_total_sec": 0.0,
        }
        try:
            if not sid:
                raise ValueError("missing sample id")
            if not image_name:
                raise ValueError("missing image")
            if not question:
                raise ValueError("missing question")
            image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            t_stage = time.perf_counter()
            image = runtime.load_image(image_path)
            row["elapsed_image_load_sec"] = time.perf_counter() - t_stage

            method_text = str(method_map.get(sid, "")).strip()
            if method_text:
                n_method_cached += 1
                row["method_cached"] = 1
            elif str(args.method) == "baseline":
                t_stage = time.perf_counter()
                method_text = generate_vga_like(
                    runtime,
                    image=image,
                    question=question,
                    object_terms=object_terms,
                    max_new_tokens=int(args.max_new_tokens),
                    use_add=False,
                    cd_alpha=0.0,
                    attn_coef=0.0,
                    start_layer=int(args.vga_start_layer),
                    end_layer=int(args.vga_end_layer),
                    head_balancing=str(args.vga_head_balancing),
                    attn_norm=bool(args.vga_attn_norm),
                    sampling=bool(args.vga_sampling),
                )
                row["elapsed_method_sec"] = time.perf_counter() - t_stage
                row["method_generated_live"] = 1
                n_method_generated += 1
            else:
                t_stage = time.perf_counter()
                method_text = generate_vga_like(
                    runtime,
                    image=image,
                    question=question,
                    object_terms=object_terms,
                    max_new_tokens=int(args.max_new_tokens),
                    use_add=bool(args.vga_use_add),
                    cd_alpha=float(args.vga_cd_alpha),
                    attn_coef=float(args.vga_attn_coef),
                    start_layer=int(args.vga_start_layer),
                    end_layer=int(args.vga_end_layer),
                    head_balancing=str(args.vga_head_balancing),
                    attn_norm=bool(args.vga_attn_norm),
                    sampling=bool(args.vga_sampling),
                )
                row["elapsed_method_sec"] = time.perf_counter() - t_stage
                row["method_generated_live"] = 1
                n_method_generated += 1

            row["method_text"] = method_text
            row["intervention_text"] = method_text

            baseline_text = ""
            baseline_generated = False

            if str(args.deployment_order) == "baseline_first_replay_on_changed":
                cached_baseline = str(baseline_map.get(sid, "")).strip()
                if cached_baseline:
                    baseline_text = cached_baseline
                    baseline_generated = True
                    n_baseline_cached += 1
                    row["baseline_cached"] = 1
                else:
                    t_stage = time.perf_counter()
                    baseline_text = generate_vga_like(
                        runtime,
                        image=image,
                        question=question,
                        object_terms=object_terms,
                        max_new_tokens=int(args.baseline_max_new_tokens),
                        use_add=False,
                        cd_alpha=0.0,
                        attn_coef=0.0,
                        start_layer=int(args.vga_start_layer),
                        end_layer=int(args.vga_end_layer),
                        head_balancing=str(args.vga_head_balancing),
                        attn_norm=bool(args.vga_attn_norm),
                        sampling=bool(args.vga_sampling),
                    )
                    row["elapsed_baseline_sec"] = time.perf_counter() - t_stage
                    row["baseline_generated_live"] = 1
                    baseline_generated = True
                    n_baseline_generated += 1

                row["baseline_triggered"] = int(baseline_generated)
                row["baseline_text"] = baseline_text
                method_label = parse_yes_no(method_text)
                baseline_label = parse_yes_no(baseline_text)
                row["method_label"] = method_label
                row["intervention_label"] = method_label
                row["baseline_label"] = baseline_label
                row["baseline_generated"] = int(baseline_generated)

                if method_label not in {"yes", "no"} or baseline_label not in {"yes", "no"}:
                    n_parse_failure += 1
                    n_replay_score_skipped += 1
                    row["replay_score_skipped"] = 1
                    row["actual_direction"] = ""
                    row["route"] = "method"
                    row["final_source"] = "method_parse_failure_replay_skipped"
                    row["final_text"] = method_text
                    row["selected_score"] = ""
                    row["selected_tau"] = ""
                    row["decision_reason"] = "unparseable_yesno_replay_skipped"
                elif method_label == baseline_label:
                    n_answer_unchanged += 1
                    n_replay_score_skipped += 1
                    row["answer_changed"] = 0
                    row["replay_score_skipped"] = 1
                    row["actual_direction"] = "unchanged"
                    row["route"] = "method"
                    row["final_source"] = "method_unchanged_replay_skipped"
                    row["final_text"] = method_text
                    row["selected_score"] = ""
                    row["selected_tau"] = ""
                    row["decision_reason"] = "baseline_and_method_same_label_replay_skipped"
                else:
                    n_answer_changed += 1
                    row["answer_changed"] = 1
                    t_stage = time.perf_counter()
                    feat = feature_row_for_method_answer(
                        runtime,
                        image=image,
                        sample_id=sid,
                        image_name=image_name,
                        question=question,
                        method_text=method_text,
                        object_terms=object_terms,
                        lp_tail_quantile=float(args.lp_tail_quantile),
                        lp_tail_eps=float(args.lp_tail_eps),
                        lp_len_corr_alpha=float(args.lp_len_corr_alpha),
                    )
                    row.update(feat)
                    row["method_text"] = method_text
                    row["intervention_text"] = method_text
                    row["baseline_text"] = baseline_text
                    scores = controller.score(row)
                    n_replay_score_computed += 1
                    row["replay_score_computed"] = 1
                    row["yes_to_no_score"] = score_or_blank(scores.yes_to_no_score)
                    row["no_to_yes_score"] = score_or_blank(scores.no_to_yes_score)
                    row["yes_to_no_tau"] = scores.yes_to_no_tau
                    row["no_to_yes_tau"] = scores.no_to_yes_tau
                    row["elapsed_replay_score_sec"] = time.perf_counter() - t_stage

                    t_stage = time.perf_counter()
                    decision = controller.decide(
                        method_text=method_text,
                        baseline_text=baseline_text,
                        scores=scores,
                        baseline_generated=baseline_generated,
                    )
                    row["elapsed_decision_sec"] = time.perf_counter() - t_stage
                    row["method_label"] = decision.method_label
                    row["intervention_label"] = decision.method_label
                    row["baseline_label"] = decision.baseline_label
                    row["actual_direction"] = decision.actual_direction
                    row["route"] = decision.route
                    row["final_source"] = decision.final_source
                    row["final_text"] = decision.final_text
                    row["selected_score"] = score_or_blank(decision.selected_score)
                    row["selected_tau"] = score_or_blank(decision.selected_tau)
                    row["baseline_generated"] = int(decision.baseline_generated)
                    row["decision_reason"] = decision.reason
            else:
                t_stage = time.perf_counter()
                feat = feature_row_for_method_answer(
                    runtime,
                    image=image,
                    sample_id=sid,
                    image_name=image_name,
                    question=question,
                    method_text=method_text,
                    object_terms=object_terms,
                    lp_tail_quantile=float(args.lp_tail_quantile),
                    lp_tail_eps=float(args.lp_tail_eps),
                    lp_len_corr_alpha=float(args.lp_len_corr_alpha),
                )
                row.update(feat)
                row["method_text"] = method_text
                row["intervention_text"] = method_text
                scores = controller.score(row)
                n_replay_score_computed += 1
                row["replay_score_computed"] = 1
                row["yes_to_no_score"] = score_or_blank(scores.yes_to_no_score)
                row["no_to_yes_score"] = score_or_blank(scores.no_to_yes_score)
                row["yes_to_no_tau"] = scores.yes_to_no_tau
                row["no_to_yes_tau"] = scores.no_to_yes_tau
                row["baseline_triggered"] = int(scores.may_need_baseline)
                row["elapsed_replay_score_sec"] = time.perf_counter() - t_stage

                if scores.may_need_baseline:
                    cached_baseline = str(baseline_map.get(sid, "")).strip()
                    if cached_baseline:
                        baseline_text = cached_baseline
                        baseline_generated = True
                        n_baseline_cached += 1
                        row["baseline_cached"] = 1
                    else:
                        t_stage = time.perf_counter()
                        baseline_text = generate_vga_like(
                            runtime,
                            image=image,
                            question=question,
                            object_terms=object_terms,
                            max_new_tokens=int(args.baseline_max_new_tokens),
                            use_add=False,
                            cd_alpha=0.0,
                            attn_coef=0.0,
                            start_layer=int(args.vga_start_layer),
                            end_layer=int(args.vga_end_layer),
                            head_balancing=str(args.vga_head_balancing),
                            attn_norm=bool(args.vga_attn_norm),
                            sampling=bool(args.vga_sampling),
                        )
                        row["elapsed_baseline_sec"] = time.perf_counter() - t_stage
                        row["baseline_generated_live"] = 1
                        baseline_generated = True
                        n_baseline_generated += 1
                else:
                    row["baseline_skipped"] = 1
                    n_baseline_skipped += 1

                t_stage = time.perf_counter()
                decision = controller.decide(
                    method_text=method_text,
                    baseline_text=baseline_text,
                    scores=scores,
                    baseline_generated=baseline_generated,
                )
                row["elapsed_decision_sec"] = time.perf_counter() - t_stage
                row["baseline_text"] = baseline_text
                row["method_label"] = decision.method_label
                row["intervention_label"] = decision.method_label
                row["baseline_label"] = decision.baseline_label
                row["actual_direction"] = decision.actual_direction
                row["route"] = decision.route
                row["final_source"] = decision.final_source
                row["final_text"] = decision.final_text
                row["selected_score"] = score_or_blank(decision.selected_score)
                row["selected_tau"] = score_or_blank(decision.selected_tau)
                row["baseline_generated"] = int(decision.baseline_generated)
                row["decision_reason"] = decision.reason

            if gt_label in {"yes", "no"}:
                method_label = str(row.get("method_label", "")).strip()
                baseline_label = str(row.get("baseline_label", "")).strip()
                row["method_correct"] = int(method_label == gt_label) if method_label else ""
                row["intervention_correct"] = row["method_correct"]
                row["baseline_correct"] = int(baseline_label == gt_label) if baseline_label else ""
                final_label = baseline_label if str(row.get("route", "")) == "baseline" else method_label
                row["final_label"] = final_label
                row["final_correct"] = int(final_label == gt_label) if final_label else ""
            else:
                row["method_correct"] = ""
                row["intervention_correct"] = ""
                row["baseline_correct"] = ""
                row["final_label"] = ""
                row["final_correct"] = ""
        except Exception as exc:
            n_errors += 1
            row["score_error"] = str(exc)
            row["score_error_traceback"] = traceback.format_exc()
            row["route"] = "method"
            row["final_source"] = "error_method"
            row["final_text"] = str(row.get("method_text", ""))

        row["elapsed_total_sec"] = time.perf_counter() - sample_t0
        feature_rows.append(row)
        route_rows.append(
            {
                "id": row.get("id", ""),
                "route": row.get("route", "method"),
                "final_source": row.get("final_source", ""),
                "actual_direction": row.get("actual_direction", ""),
                "yes_to_no_score": row.get("yes_to_no_score", ""),
                "yes_to_no_tau": row.get("yes_to_no_tau", ""),
                "no_to_yes_score": row.get("no_to_yes_score", ""),
                "no_to_yes_tau": row.get("no_to_yes_tau", ""),
                "selected_score": row.get("selected_score", ""),
                "selected_tau": row.get("selected_tau", ""),
                "baseline_generated": row.get("baseline_generated", 0),
                "baseline_triggered": row.get("baseline_triggered", 0),
                "baseline_cached": row.get("baseline_cached", 0),
                "baseline_generated_live": row.get("baseline_generated_live", 0),
                "baseline_skipped": row.get("baseline_skipped", 0),
                "replay_score_computed": row.get("replay_score_computed", 0),
                "replay_score_skipped": row.get("replay_score_skipped", 0),
                "answer_changed": row.get("answer_changed", ""),
                "elapsed_method_sec": row.get("elapsed_method_sec", 0.0),
                "elapsed_replay_score_sec": row.get("elapsed_replay_score_sec", 0.0),
                "elapsed_baseline_sec": row.get("elapsed_baseline_sec", 0.0),
                "elapsed_total_sec": row.get("elapsed_total_sec", 0.0),
                "decision_reason": row.get("decision_reason", ""),
                "baseline_label": row.get("baseline_label", ""),
                "intervention_label": row.get("intervention_label", ""),
                "final_label": row.get("final_label", ""),
                "baseline_correct": row.get("baseline_correct", ""),
                "intervention_correct": row.get("intervention_correct", ""),
                "final_correct": row.get("final_correct", ""),
                "score_error": row.get("score_error", ""),
            }
        )
        final_preds.append(
            {
                "question_id": row.get("id", ""),
                "id": row.get("id", ""),
                "image": row.get("image", ""),
                "text": row.get("final_text", ""),
                "route": row.get("route", "method"),
                "source": row.get("final_source", ""),
            }
        )
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[fixed-c3-lazy] {idx + 1}/{len(questions)}", flush=True)

    feature_csv = os.path.join(out_dir, "online_feature_rows.csv")
    route_csv = os.path.join(out_dir, "pcp_route_rows_lazy.csv")
    final_jsonl = os.path.join(out_dir, "pred_fixed_c3_lazy.jsonl")
    write_csv(feature_csv, feature_rows)
    write_csv(route_csv, route_rows)
    write_jsonl(final_jsonl, final_preds)

    eval_rows = [r for r in feature_rows if str(r.get("final_correct", "")).strip() in {"0", "1"}]
    n_eval = len(eval_rows)
    final_acc = None
    method_acc = None
    baseline_acc_on_generated = None
    if n_eval:
        final_acc = sum(int(r["final_correct"]) for r in eval_rows) / float(n_eval)
        method_acc = sum(int(r["intervention_correct"]) for r in eval_rows if str(r.get("intervention_correct", "")).strip() in {"0", "1"}) / float(n_eval)
        baseline_rows = [r for r in eval_rows if str(r.get("baseline_correct", "")).strip() in {"0", "1"}]
        if baseline_rows:
            baseline_acc_on_generated = sum(int(r["baseline_correct"]) for r in baseline_rows) / float(len(baseline_rows))

    def flag_count(key: str) -> int:
        return sum(1 for r in feature_rows if str(r.get(key, "")).strip().lower() in {"1", "1.0", "true"})

    def finite_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if out != out or out in {float("inf"), float("-inf")}:
            return None
        return out

    def mean_or_none(values: Sequence[Any]) -> Optional[float]:
        nums = [v for v in (finite_float(x) for x in values) if v is not None]
        return None if not nums else float(sum(nums) / float(len(nums)))

    total_sec = time.perf_counter() - t_total
    n_rows = len(questions)
    denom_rows = float(max(1, n_rows))
    completed_rows = [r for r in feature_rows if not str(r.get("score_error", "")).strip()]
    n_completed = len(completed_rows)
    baseline_triggered = flag_count("baseline_triggered")
    baseline_generated_live = flag_count("baseline_generated_live")
    baseline_cached = flag_count("baseline_cached")
    baseline_skipped = flag_count("baseline_skipped")
    replay_score_computed = flag_count("replay_score_computed")
    replay_score_skipped = flag_count("replay_score_skipped")
    answer_changed = flag_count("answer_changed")
    route_baseline = sum(1 for r in route_rows if str(r.get("route")) == "baseline")
    baseline_trigger_rate = baseline_triggered / denom_rows
    baseline_skip_rate = baseline_skipped / denom_rows
    replay_score_compute_rate = replay_score_computed / denom_rows
    replay_score_skip_rate = replay_score_skipped / denom_rows
    answer_changed_rate = answer_changed / denom_rows
    route_baseline_rate = route_baseline / denom_rows
    mean_total_sec = mean_or_none(r.get("elapsed_total_sec") for r in completed_rows)
    mean_image_load_sec = mean_or_none(r.get("elapsed_image_load_sec") for r in completed_rows)
    mean_method_generated_sec = mean_or_none(
        r.get("elapsed_method_sec") for r in completed_rows if str(r.get("method_generated_live")) in {"1", "1.0"}
    )
    mean_replay_score_sec = mean_or_none(
        r.get("elapsed_replay_score_sec")
        for r in completed_rows
        if str(r.get("replay_score_computed")) in {"1", "1.0"}
    )
    mean_baseline_generated_sec = mean_or_none(
        r.get("elapsed_baseline_sec")
        for r in completed_rows
        if str(r.get("baseline_generated_live")) in {"1", "1.0"}
    )
    estimated_always_baseline_mean_sec = None
    estimated_always_baseline_total_sec = None
    estimated_method_only_mean_sec = None
    estimated_score_only_no_baseline_mean_sec = None
    estimated_always_replay_mean_sec = None
    estimated_always_replay_total_sec = None
    estimated_speedup_vs_always_replay = None
    estimated_replay_skip_savings_pct = None
    estimated_speedup_vs_always_baseline = None
    estimated_latency_savings_pct = None
    estimated_lazy_over_method_only_sec = None
    if mean_image_load_sec is not None and mean_method_generated_sec is not None:
        estimated_method_only_mean_sec = mean_image_load_sec + mean_method_generated_sec
        if mean_total_sec is not None:
            estimated_lazy_over_method_only_sec = mean_total_sec - estimated_method_only_mean_sec
    if mean_total_sec is not None and mean_replay_score_sec is not None:
        estimated_always_replay_mean_sec = mean_total_sec + replay_score_skip_rate * mean_replay_score_sec
        estimated_always_replay_total_sec = total_sec + replay_score_skipped * mean_replay_score_sec
        if mean_total_sec > 0:
            estimated_speedup_vs_always_replay = estimated_always_replay_mean_sec / mean_total_sec
        if estimated_always_replay_mean_sec and estimated_always_replay_mean_sec > 0:
            estimated_replay_skip_savings_pct = 100.0 * (1.0 - mean_total_sec / estimated_always_replay_mean_sec)
    if mean_total_sec is not None and mean_baseline_generated_sec is not None:
        estimated_score_only_no_baseline_mean_sec = mean_total_sec - baseline_trigger_rate * mean_baseline_generated_sec
        estimated_always_baseline_mean_sec = mean_total_sec + baseline_skip_rate * mean_baseline_generated_sec
        estimated_always_baseline_total_sec = total_sec + baseline_skipped * mean_baseline_generated_sec
        if mean_total_sec > 0:
            estimated_speedup_vs_always_baseline = estimated_always_baseline_mean_sec / mean_total_sec
        if estimated_always_baseline_mean_sec and estimated_always_baseline_mean_sec > 0:
            estimated_latency_savings_pct = 100.0 * (1.0 - mean_total_sec / estimated_always_baseline_mean_sec)

    write_json(
        summary_json,
        {
            "mode": "llava15_fixed_c3_lazy_deploy",
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "fixed_json": os.path.abspath(args.fixed_json),
                "target": str(args.target),
                "dataset": str(args.dataset),
                "model_path": str(args.model_path),
                "conv_mode": str(args.conv_mode),
                "method": str(args.method),
                "deployment_order": str(args.deployment_order),
                "method_pred_jsonl": os.path.abspath(args.method_pred_jsonl) if str(args.method_pred_jsonl).strip() else "",
                "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl) if str(args.baseline_pred_jsonl).strip() else "",
                "gt_csv": os.path.abspath(args.gt_csv) if str(args.gt_csv).strip() else "",
            },
            "counts": {
                "n_rows": n_rows,
                "n_completed": n_completed,
                "n_errors": n_errors,
                "n_method_generated": n_method_generated,
                "n_method_cached": n_method_cached,
                "n_baseline_generated": n_baseline_generated,
                "n_baseline_cached": n_baseline_cached,
                "n_baseline_skipped": n_baseline_skipped,
                "n_baseline_triggered": baseline_triggered,
                "n_baseline_generated_live": baseline_generated_live,
                "n_replay_score_computed": replay_score_computed,
                "n_replay_score_skipped": replay_score_skipped,
                "n_answer_changed": n_answer_changed,
                "n_answer_unchanged": n_answer_unchanged,
                "n_parse_failure": n_parse_failure,
                "n_route_baseline": route_baseline,
            },
            "evaluation": {
                "n_eval": n_eval,
                "method_acc": method_acc,
                "final_acc": final_acc,
                "delta_vs_method": None if final_acc is None or method_acc is None else final_acc - method_acc,
                "baseline_acc_on_generated_or_cached": baseline_acc_on_generated,
            },
            "timing": {
                "total_sec": total_sec,
                "mean_total_sec_per_sample": mean_total_sec,
                "mean_image_load_sec": mean_image_load_sec,
                "mean_method_generated_sec": mean_method_generated_sec,
                "mean_replay_score_sec": mean_replay_score_sec,
                "mean_baseline_generated_sec": mean_baseline_generated_sec,
                "baseline_trigger_rate": baseline_trigger_rate,
                "baseline_skip_rate": baseline_skip_rate,
                "replay_score_compute_rate": replay_score_compute_rate,
                "replay_score_skip_rate": replay_score_skip_rate,
                "answer_changed_rate": answer_changed_rate,
                "route_baseline_rate": route_baseline_rate,
                "estimated_method_only_mean_sec_per_sample": estimated_method_only_mean_sec,
                "estimated_score_only_no_baseline_mean_sec_per_sample": estimated_score_only_no_baseline_mean_sec,
                "estimated_always_replay_mean_sec_per_sample": estimated_always_replay_mean_sec,
                "estimated_always_replay_total_sec": estimated_always_replay_total_sec,
                "estimated_speedup_vs_always_replay": estimated_speedup_vs_always_replay,
                "estimated_replay_skip_savings_pct": estimated_replay_skip_savings_pct,
                "estimated_always_baseline_mean_sec_per_sample": estimated_always_baseline_mean_sec,
                "estimated_always_baseline_total_sec": estimated_always_baseline_total_sec,
                "estimated_speedup_vs_always_baseline": estimated_speedup_vs_always_baseline,
                "estimated_latency_savings_pct": estimated_latency_savings_pct,
                "estimated_lazy_over_method_only_sec_per_sample": estimated_lazy_over_method_only_sec,
                "note": (
                    "Lazy latency includes method generation and replay-score computation for every sample, "
                    "then baseline generation only when either directional score reaches tau. "
                    "Always-baseline estimates add one observed live baseline-generation cost for each skipped baseline. "
                    "If baseline predictions are cached, run without --baseline_pred_jsonl for live latency estimates."
                ),
            },
            "outputs": {
                "online_feature_rows_csv": feature_csv,
                "route_rows_csv": route_csv,
                "final_predictions_jsonl": final_jsonl,
            },
        },
    )
    print("[saved]", feature_csv, flush=True)
    print("[saved]", route_csv, flush=True)
    print("[saved]", final_jsonl, flush=True)
    print("[saved]", summary_json, flush=True)


if __name__ == "__main__":
    main()
