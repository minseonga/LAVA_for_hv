#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import (  # noqa: E402
    CleanroomLlavaRuntime,
    ForwardPack,
    load_headset,
    load_label_map,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    parse_yes_no,
    safe_id,
    select_content_indices,
    stage_a_score_from_pack,
    write_jsonl,
)
from pnp_deploy.discriminative_meta import (  # noqa: E402
    MetaStrongController,
    maybe_int,
    read_json,
    write_csv,
    write_json,
)


def mean_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0
    mu = mean_or_zero(seq)
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(max(var, 0.0) ** 0.5)


def min_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    return float(min(seq)) if seq else 0.0


def max_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    return float(max(seq)) if seq else 0.0


def quantile_or_zero(values: Sequence[float], q: float) -> float:
    seq = sorted(float(v) for v in values)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    qq = min(max(float(q), 0.0), 1.0)
    pos = qq * float(len(seq) - 1)
    lo = int(torch.floor(torch.tensor(pos)).item())
    hi = int(torch.ceil(torch.tensor(pos)).item())
    if lo == hi:
        return float(seq[lo])
    w = pos - float(lo)
    return float((1.0 - w) * seq[lo] + w * seq[hi])


def summarize(values: Sequence[float], prefix: str) -> Dict[str, float]:
    seq = [float(v) for v in values]
    return {
        f"{prefix}_mean": mean_or_zero(seq),
        f"{prefix}_std": std_or_zero(seq),
        f"{prefix}_min": min_or_zero(seq),
        f"{prefix}_max": max_or_zero(seq),
    }


def cheap_features_from_pack(
    runtime: CleanroomLlavaRuntime,
    pack: ForwardPack,
    *,
    sample_id: str,
    image_name: str,
    question: str,
    content_indices: Sequence[int],
    lp_tail_quantile: float,
    lp_tail_eps: float,
    lp_len_corr_alpha: float,
) -> Dict[str, Any]:
    logits = pack.logits.to(torch.float32)
    decision_positions = pack.decision_positions.long()
    target_ids = pack.labels_exp[pack.cont_label_positions.long()].long()

    token_logits = logits[decision_positions]
    log_probs = F.log_softmax(token_logits, dim=-1)
    probs = torch.softmax(token_logits, dim=-1)
    token_ent = -(probs * log_probs).sum(dim=-1)
    target_lp = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

    top2_vals, top2_idx = torch.topk(token_logits, k=2, dim=-1)
    top1_logit = top2_vals[:, 0]
    top2_logit = top2_vals[:, 1]
    top1_id = top2_idx[:, 0]
    top1_margin = top1_logit - top2_logit
    target_logit = token_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    best_other_logit = torch.where(top1_id == target_ids, top2_logit, top1_logit)
    target_gap = target_logit - best_other_logit
    target_is_argmax = (top1_id == target_ids).to(torch.float32)

    cont_idx = list(range(int(target_ids.numel())))
    pick = list(content_indices) if content_indices else cont_idx

    def pick_values(tensor: torch.Tensor, indices: Sequence[int]) -> List[float]:
        return [float(tensor[int(i)].item()) for i in indices]

    lp_all = pick_values(target_lp, cont_idx)
    lp_content = pick_values(target_lp, pick)
    ent_all = pick_values(token_ent, cont_idx)
    ent_content = pick_values(token_ent, pick)
    margin_all = pick_values(top1_margin, cont_idx)
    margin_content = pick_values(top1_margin, pick)
    gap_all = pick_values(target_gap, cont_idx)
    gap_content = pick_values(target_gap, pick)
    argmax_all = pick_values(target_is_argmax, cont_idx)
    argmax_content = pick_values(target_is_argmax, pick)

    lp_content_mean = mean_or_zero(lp_content)
    lp_content_std = std_or_zero(lp_content)
    lp_content_min = min_or_zero(lp_content)
    lp_content_tail_gap = float(lp_content_min - lp_content_mean)
    lp_content_tail_z = float(lp_content_tail_gap / float(max(lp_content_std, float(lp_tail_eps))))
    lp_content_q10 = quantile_or_zero(lp_content, float(lp_tail_quantile))
    lp_content_min_len_corr = float(lp_content_min + float(lp_len_corr_alpha) * torch.log(torch.tensor(max(1, len(pick)), dtype=torch.float32)).item())

    row: Dict[str, Any] = {
        "id": sample_id,
        "image": image_name,
        "question": question,
        "n_cont_tokens": int(len(cont_idx)),
        "n_content_tokens": int(len(pick)),
        "cheap_content_fraction": float(len(pick) / max(1, len(cont_idx))),
        "cheap_conflict_lp_minus_entropy": float(mean_or_zero(lp_content) - mean_or_zero(ent_content)),
        "cheap_conflict_gap_minus_entropy": float(mean_or_zero(gap_content) - mean_or_zero(ent_content)),
        "cheap_lp_content_tail_gap": lp_content_tail_gap,
        "cheap_lp_content_tail_z": lp_content_tail_z,
        "cheap_lp_content_q10": lp_content_q10,
        "cheap_lp_content_min_len_corr": lp_content_min_len_corr,
    }
    row.update(summarize(lp_all, "cheap_lp_all"))
    row.update(summarize(lp_content, "cheap_lp_content"))
    row.update(summarize(ent_all, "cheap_entropy_all"))
    row.update(summarize(ent_content, "cheap_entropy_content"))
    row.update(summarize(margin_all, "cheap_margin_all"))
    row.update(summarize(margin_content, "cheap_margin_content"))
    row.update(summarize(gap_all, "cheap_target_gap_all"))
    row.update(summarize(gap_content, "cheap_target_gap_content"))
    row.update(summarize(argmax_all, "cheap_target_argmax_all"))
    row.update(summarize(argmax_content, "cheap_target_argmax_content"))
    return row


def final_label_from_text(text: str) -> str:
    return parse_yes_no(text) if str(text or "").strip() else ""


def evaluate_actual_routes(rows: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    route_rows: List[Dict[str, Any]] = []
    n = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    final_correct_total = 0
    total_harm = 0
    selected_count = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0
    expert_counts: Dict[str, int] = {}

    for row in rows:
        baseline_correct = maybe_int(row.get("baseline_correct"))
        intervention_correct = maybe_int(row.get("intervention_correct"))
        final_correct = maybe_int(row.get("final_correct"))
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        route = str(row.get("route", "method"))
        expert = str(row.get("expert", "none"))
        expert_counts[expert] = int(expert_counts.get(expert, 0)) + 1
        route_rows.append(
            {
                "id": str(row.get("id", "")).strip(),
                "expert": expert,
                "route": route,
                "b_score": row.get("b_score"),
                "c_score": row.get("c_score"),
                "f_score": row.get("f_score"),
                "score": row.get("meta_score"),
                "tau": row.get("meta_tau"),
                "harm": harm,
                "help": help_,
                "baseline_correct": baseline_correct,
                "intervention_correct": intervention_correct,
                "final_correct": final_correct,
            }
        )
        if baseline_correct is None or intervention_correct is None or final_correct is None:
            continue
        n += 1
        baseline_correct_total += int(baseline_correct)
        intervention_correct_total += int(intervention_correct)
        final_correct_total += int(final_correct)
        total_harm += harm
        if route == "baseline":
            selected_count += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int((harm == 0) and (help_ == 0))

    precision = float(selected_harm / max(1, selected_count))
    recall = float(selected_harm / max(1, total_harm))
    f1 = 0.0 if precision + recall == 0.0 else float(2.0 * precision * recall / (precision + recall))
    return route_rows, {
        "mode": "online_actual_routes",
        "n_eval": int(n),
        "baseline_rate": float(selected_count / max(1, n)),
        "method_rate": float(1.0 - selected_count / max(1, n)),
        "final_acc": float(final_correct_total / max(1, n)),
        "baseline_acc": float(baseline_correct_total / max(1, n)),
        "intervention_acc": float(intervention_correct_total / max(1, n)),
        "delta_vs_intervention": float((final_correct_total - intervention_correct_total) / max(1, n)),
        "selected_count": int(selected_count),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "selected_harm_precision": precision,
        "selected_harm_recall": recall,
        "selected_harm_f1": f1,
        "expert_counts": expert_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Sample-wise online feature extraction and routing for the discriminative meta_strong controller. "
            "The script consumes an intervention answer, computes stage_a + cheap C features in one model load, "
            "and routes to baseline or method with the calibrated meta_strong policy. "
            "For cached parity, Stage-A and cheap-C use the same prompt/forward conventions "
            "as run_frgavr_cleanroom.py and extract_c_stage_cheap_online_features.py."
        )
    )
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--policy_bundle_json", type=str, required=True)
    ap.add_argument("--headset_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, default="")
    ap.add_argument("--gt_csv", type=str, default="")
    ap.add_argument("--gt_id_col", type=str, default="id")
    ap.add_argument("--gt_label_col", type=str, default="answer")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--intervention_pred_key", type=str, default="auto")
    ap.add_argument("--baseline_pred_key", type=str, default="auto")
    ap.add_argument("--generate_baseline_on_fallback", type=parse_bool, default=False)
    ap.add_argument("--baseline_max_new_tokens", type=int, default=8)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lambda_a", type=float, default=0.5)
    ap.add_argument("--late_start", type=int, default=-1)
    ap.add_argument("--late_end", type=int, default=-1)
    ap.add_argument("--lp_tail_quantile", type=float, default=0.10)
    ap.add_argument("--lp_tail_eps", type=float, default=1e-6)
    ap.add_argument("--lp_len_corr_alpha", type=float, default=0.35)
    ap.add_argument(
        "--feature_order",
        type=str,
        default="cheap_first",
        choices=["cheap_first", "stage_first"],
        help=(
            "Forward order for online feature extraction. The cached 87.82 meta_strong "
            "artifact was built from a standalone cheap-C extractor, so cheap_first keeps "
            "the C-feature replay isolated from the attention-returning Stage-A replay."
        ),
    )
    ap.add_argument(
        "--stage_a_prefilter_c_score_min",
        type=float,
        default=None,
        help=(
            "Optional cascade speedup. When set, cheap-C is always computed first; "
            "rows with preliminary c_score below this threshold skip the expensive "
            "attention Stage-A replay and are forced to route=method. For the cached "
            "meta_strong VGA policy, 2.0 preserved held-out accuracy while reducing "
            "Stage-A replays from 9000 to about 380."
        ),
    )
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    feature_rows_csv = os.path.join(out_dir, "online_feature_rows.csv")
    route_rows_csv = os.path.join(out_dir, "meta_route_rows.csv")
    final_pred_jsonl = os.path.join(out_dir, "pred_meta_strong_online.jsonl")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json, flush=True)
        return

    questions = load_question_rows(os.path.abspath(args.question_file), limit=int(args.limit))
    intervention_map = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), text_key=str(args.intervention_pred_key))
    baseline_map = (
        load_prediction_text_map(os.path.abspath(args.baseline_pred_jsonl), text_key=str(args.baseline_pred_key))
        if str(args.baseline_pred_jsonl or "").strip()
        else {}
    )
    gt_map = (
        load_label_map(os.path.abspath(args.gt_csv), id_col=str(args.gt_id_col), label_col=str(args.gt_label_col))
        if str(args.gt_csv or "").strip()
        else {}
    )
    headset = load_headset(os.path.abspath(args.headset_json), late_start=int(args.late_start), late_end=int(args.late_end))
    controller = MetaStrongController.from_bundle(read_json(os.path.abspath(args.policy_bundle_json)))
    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )

    feature_rows: List[Dict[str, Any]] = []
    routing_inputs: List[Dict[str, Any]] = []
    final_preds: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_intervention = 0
    n_missing_baseline = 0
    n_generated_baseline = 0
    n_stage_a_prefilter_skipped = 0
    n_stage_a_computed = 0
    feature_secs: List[float] = []
    baseline_secs: List[float] = []
    stage_a_prefilter = args.stage_a_prefilter_c_score_min

    for idx, sample in enumerate(questions):
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        # Cached meta eval used two existing extractors with slightly different
        # question-field precedence. Keep both conventions for exact parity.
        stage_question = str(sample.get("question", sample.get("text", ""))).strip()
        cheap_question = str(sample.get("text", sample.get("question", ""))).strip()
        intervention_text = str(intervention_map.get(sid, "")).strip()
        baseline_text = str(baseline_map.get(sid, "")).strip()
        gt_label = str(gt_map.get(sid, "")).strip().lower()
        row: Dict[str, Any] = {
            "id": sid,
            "image": image_name,
            "question": stage_question,
            "stage_question": stage_question,
            "cheap_question": cheap_question,
            "intervention_text": intervention_text,
            "baseline_text": baseline_text,
            "gt_label": gt_label,
            "score_error": "",
        }
        try:
            if not sid:
                raise ValueError("Missing sample id.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not stage_question:
                raise ValueError("Missing question.")
            if not cheap_question:
                raise ValueError("Missing cheap question.")
            if not intervention_text:
                n_missing_intervention += 1
                raise ValueError("Missing intervention prediction.")
            image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            t0 = time.perf_counter()
            image = runtime.load_image(image_path)
            def compute_cheap() -> Dict[str, Any]:
                cheap_pack = runtime.teacher_force_candidate(
                    image=image,
                    question=cheap_question,
                    candidate_text=intervention_text,
                    output_attentions=False,
                )
                cheap_content_indices = select_content_indices(runtime.tokenizer, cheap_pack.cont_ids)
                return cheap_features_from_pack(
                    runtime=runtime,
                    pack=cheap_pack,
                    sample_id=sid,
                    image_name=image_name,
                    question=cheap_question,
                    content_indices=cheap_content_indices,
                    lp_tail_quantile=float(args.lp_tail_quantile),
                    lp_tail_eps=float(args.lp_tail_eps),
                    lp_len_corr_alpha=float(args.lp_len_corr_alpha),
                )

            def compute_stage_a() -> Dict[str, float]:
                stage_pack = runtime.teacher_force_candidate(
                    image=image,
                    question=stage_question,
                    candidate_text=intervention_text,
                    output_attentions=True,
                )
                stage_content_indices = select_content_indices(runtime.tokenizer, stage_pack.cont_ids)
                return stage_a_score_from_pack(
                    pack=stage_pack,
                    headset=headset,
                    beta=float(args.beta),
                    lambda_a=float(args.lambda_a),
                    content_indices=stage_content_indices,
                )

            force_method_by_prefilter = False
            if stage_a_prefilter is not None:
                cheap = compute_cheap()
                row.update(cheap)
                prelim_scores = controller.score_components(row)
                prelim_c_score = prelim_scores.get("c_score")
                row["stage_a_prefilter_c_score"] = prelim_c_score
                if prelim_c_score is None or float(prelim_c_score) < float(stage_a_prefilter):
                    stage_a = {}
                    force_method_by_prefilter = True
                    n_stage_a_prefilter_skipped += 1
                else:
                    stage_a = compute_stage_a()
                    n_stage_a_computed += 1
            elif str(args.feature_order) == "cheap_first":
                cheap = compute_cheap()
                stage_a = compute_stage_a()
                n_stage_a_computed += 1
            else:
                stage_a = compute_stage_a()
                cheap = compute_cheap()
                n_stage_a_computed += 1
            feature_dt = time.perf_counter() - t0
            feature_secs.append(float(feature_dt))

            row.update(cheap)
            row["stage_question"] = stage_question
            row["cheap_question"] = cheap_question
            row.update(stage_a)
            row["stage_a_prefilter_skipped"] = int(force_method_by_prefilter)
            row["feature_ms"] = float(feature_dt * 1000.0)
        except Exception as exc:
            n_errors += 1
            row["score_error"] = str(exc)
            row["score_error_traceback"] = traceback.format_exc()

        intervention_label = final_label_from_text(intervention_text)
        baseline_label = final_label_from_text(baseline_text)
        row["intervention_label"] = intervention_label
        row["baseline_label"] = baseline_label
        row["intervention_correct"] = (
            None if gt_label not in {"yes", "no"} or intervention_label == "" else int(intervention_label == gt_label)
        )
        row["baseline_correct"] = (
            None if gt_label not in {"yes", "no"} or baseline_label == "" else int(baseline_label == gt_label)
        )
        if row["baseline_correct"] is not None and row["intervention_correct"] is not None:
            row["harm"] = int(int(row["baseline_correct"]) == 1 and int(row["intervention_correct"]) == 0)
            row["help"] = int(int(row["baseline_correct"]) == 0 and int(row["intervention_correct"]) == 1)
        else:
            row["harm"] = 0
            row["help"] = 0

        if int(maybe_int(row.get("stage_a_prefilter_skipped")) or 0) == 1:
            scores = controller.score_components(row)
            row["expert"] = "cheap_prefilter"
            row["route"] = "method"
            row["b_score"] = scores.get("b_score")
            row["c_score"] = scores.get("c_score")
            row["f_score"] = scores.get("f_score")
            row["meta_score"] = scores.get("c_score")
            row["meta_tau"] = stage_a_prefilter
        else:
            decision = controller.decide(row)
            row["expert"] = decision.expert
            row["route"] = decision.route
            row["b_score"] = decision.b_score
            row["c_score"] = decision.c_score
            row["f_score"] = decision.f_score
            row["meta_score"] = decision.score
            row["meta_tau"] = decision.tau

        final_text = intervention_text
        final_source = "method"
        if str(row.get("route")) == "baseline":
            if baseline_text:
                final_text = baseline_text
                final_source = "baseline_cached"
            elif bool(args.generate_baseline_on_fallback):
                try:
                    t0 = time.perf_counter()
                    image = runtime.load_image(os.path.join(os.path.abspath(args.image_folder), image_name))
                    final_text = runtime.generate_baseline(
                        image=image,
                        question=stage_question,
                        max_new_tokens=int(args.baseline_max_new_tokens),
                    )
                    baseline_secs.append(float(time.perf_counter() - t0))
                    n_generated_baseline += 1
                    final_source = "baseline_live"
                except Exception as exc:
                    n_missing_baseline += 1
                    row["baseline_generation_error"] = str(exc)
                    final_text = intervention_text
                    final_source = "method_missing_baseline"
            else:
                n_missing_baseline += 1
                final_source = "method_missing_baseline"
        row["final_source"] = final_source
        row["final_text"] = final_text
        final_label = final_label_from_text(final_text)
        row["final_label"] = final_label
        row["final_correct"] = None if gt_label not in {"yes", "no"} or final_label == "" else int(final_label == gt_label)

        feature_rows.append(row)
        routing_inputs.append(row)
        final_preds.append(
            {
                "question_id": sid,
                "id": sid,
                "image": image_name,
                "text": final_text,
                "route": row.get("route"),
                "expert": row.get("expert"),
                "source": final_source,
            }
        )
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[meta-online] {idx + 1}/{len(questions)}", flush=True)

    route_rows, evaluation = evaluate_actual_routes(routing_inputs)
    # Preserve final_source/final_text fields in the route CSV for deployment audits.
    details_by_id = {str(row.get("id", "")): row for row in feature_rows}
    enriched_route_rows: List[Dict[str, Any]] = []
    for route_row in route_rows:
        detail = details_by_id.get(str(route_row.get("id", "")), {})
        merged = dict(route_row)
        for key in ("final_source", "final_text", "final_label", "score_error"):
            merged[key] = detail.get(key, "")
        enriched_route_rows.append(merged)

    n_final_eval = 0
    n_final_correct = 0
    n_intervention_eval = 0
    n_intervention_correct = 0
    for row in feature_rows:
        final_correct = maybe_int(row.get("final_correct"))
        intervention_correct = maybe_int(row.get("intervention_correct"))
        if final_correct is not None:
            n_final_eval += 1
            n_final_correct += int(final_correct)
        if intervention_correct is not None:
            n_intervention_eval += 1
            n_intervention_correct += int(intervention_correct)

    write_csv(feature_rows_csv, feature_rows)
    write_csv(route_rows_csv, enriched_route_rows)
    write_jsonl(final_pred_jsonl, final_preds)
    write_json(
        summary_json,
        {
            "mode": "online_meta_strong",
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "intervention_pred_key": str(args.intervention_pred_key),
                "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl) if str(args.baseline_pred_jsonl or "").strip() else "",
                "baseline_pred_key": str(args.baseline_pred_key),
                "gt_csv": os.path.abspath(args.gt_csv) if str(args.gt_csv or "").strip() else "",
                "policy_bundle_json": os.path.abspath(args.policy_bundle_json),
                "headset_json": os.path.abspath(args.headset_json),
                "model_path": str(args.model_path),
                "model_base": str(args.model_base),
                "conv_mode": str(args.conv_mode),
                "device": str(args.device),
                "beta": float(args.beta),
                "lambda_a": float(args.lambda_a),
                "late_start": int(args.late_start),
                "late_end": int(args.late_end),
                "feature_order": str(args.feature_order),
                "stage_a_prefilter_c_score_min": stage_a_prefilter,
                "generate_baseline_on_fallback": bool(args.generate_baseline_on_fallback),
            },
            "counts": {
                "n_rows": int(len(questions)),
                "n_errors": int(n_errors),
                "n_missing_intervention": int(n_missing_intervention),
                "n_missing_baseline_for_selected": int(n_missing_baseline),
                "n_generated_baseline": int(n_generated_baseline),
                "n_stage_a_prefilter_skipped": int(n_stage_a_prefilter_skipped),
                "n_stage_a_computed": int(n_stage_a_computed),
            },
            "evaluation_from_cached_labels": evaluation,
            "evaluation_from_final_text": {
                "n_eval": int(n_final_eval),
                "intervention_acc": None if n_intervention_eval == 0 else float(n_intervention_correct / float(n_intervention_eval)),
                "final_acc": None if n_final_eval == 0 else float(n_final_correct / float(n_final_eval)),
                "delta_vs_intervention": (
                    None
                    if n_final_eval == 0 or n_intervention_eval == 0
                    else float(n_final_correct / float(n_final_eval) - n_intervention_correct / float(n_intervention_eval))
                ),
            },
            "timing": {
                "feature_total_sec": float(sum(feature_secs)),
                "feature_mean_ms": float(1000.0 * mean_or_zero(feature_secs)),
                "baseline_total_sec": float(sum(baseline_secs)),
                "baseline_mean_ms": float(1000.0 * mean_or_zero(baseline_secs)),
            },
            "outputs": {
                "online_feature_rows_csv": feature_rows_csv,
                "meta_route_rows_csv": route_rows_csv,
                "final_predictions_jsonl": final_pred_jsonl,
            },
            "adapter_contract": {
                "method_output": "Any intervention method, including VGA or PAI, only needs to provide the candidate answer text before routing.",
                "fallback_output": "For lowest latency, cache or share the baseline answer; otherwise generate it only when route == baseline.",
                "vendor_patch_policy": "Do not patch VGA/PAI vendor code in the deploy repo. Expose generation/logit hooks in a thin adapter and keep method-specific changes documented there.",
            },
        },
    )
    print("[saved]", feature_rows_csv, flush=True)
    print("[saved]", route_rows_csv, flush=True)
    print("[saved]", final_pred_jsonl, flush=True)
    print("[saved]", summary_json, flush=True)


if __name__ == "__main__":
    main()
