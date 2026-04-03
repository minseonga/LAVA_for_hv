#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import (
    load_label_map,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    parse_yes_no,
    safe_id,
    write_csv,
    write_json,
)
from pnp_controller.adapters.vga_online import VGAOnlineAdapter, VGAOnlineConfig


def maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: object) -> Optional[int]:
    f = maybe_float(value)
    if f is None:
        return None
    return int(round(f))


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
    var = sum((float(v) - mu) ** 2 for v in seq) / float(len(seq))
    return float(math.sqrt(max(0.0, var)))


def min_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    return float(min(seq)) if seq else 0.0


def max_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    return float(max(seq)) if seq else 0.0


def summarize(values: Sequence[float], prefix: str) -> Dict[str, float]:
    seq = [float(v) for v in values]
    return {
        f"{prefix}_mean": mean_or_zero(seq),
        f"{prefix}_std": std_or_zero(seq),
        f"{prefix}_min": min_or_zero(seq),
        f"{prefix}_max": max_or_zero(seq),
    }


def load_policy(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_policy(policy: Dict[str, Any], row: Dict[str, Any]) -> Tuple[Optional[float], int]:
    value_a = maybe_float(row.get(str(policy.get("feature_a", "")).strip()))
    score = None
    if str(policy.get("policy_type", "")) == "single":
        if value_a is not None:
            score = -float(value_a) if str(policy.get("direction_a", "")).strip() == "low" else float(value_a)
    elif str(policy.get("policy_type", "")) == "pair_sum_z":
        feature_b = str(policy.get("feature_b", "")).strip()
        value_b = maybe_float(row.get(feature_b))
        if value_a is not None and value_b is not None:
            oa = -float(value_a) if str(policy.get("direction_a", "")).strip() == "low" else float(value_a)
            ob = -float(value_b) if str(policy.get("direction_b", "")).strip() == "low" else float(value_b)
            mu_a = float(policy.get("mu_a", 0.0) or 0.0)
            mu_b = float(policy.get("mu_b", 0.0) or 0.0)
            sd_a = float(max(float(policy.get("sd_a", 1.0) or 1.0), 1e-6))
            sd_b = float(max(float(policy.get("sd_b", 1.0) or 1.0), 1e-6))
            score = float((oa - mu_a) / sd_a + (ob - mu_b) / sd_b)
    rescue = int(score is not None and float(score) >= float(policy["tau"]))
    return score, rescue


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def build_cheap_feature_row_from_prepared(
    adapter: VGAOnlineAdapter,
    prepared: Dict[str, Any],
    candidate_text: str,
) -> Dict[str, Any]:
    if str(candidate_text or "").strip() == "":
        raise ValueError("Missing candidate text for cheap feature extraction.")

    prompt_ids = prepared["input_ids"]
    cont_ids = adapter.tokenizer(
        str(candidate_text),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0].to(adapter.device)
    if int(cont_ids.numel()) <= 0:
        raise ValueError("Candidate tokenization is empty.")

    full_ids = torch.cat([prompt_ids[0], cont_ids], dim=0).unsqueeze(0)
    base_mask = torch.ones_like(full_ids, dtype=torch.long, device=adapter.device)
    images_tensor = prepared["image_tensor"].unsqueeze(0)
    image_sizes = [prepared["image_size"]]

    with torch.no_grad():
        _, pos_ids_e, attn_mask_e, _, mm_embeds_e, labels_e = adapter.model.prepare_inputs_labels_for_multimodal(
            full_ids,
            None,
            base_mask,
            None,
            full_ids,
            images_tensor,
            image_sizes,
        )
        if mm_embeds_e is None or labels_e is None:
            raise RuntimeError("Failed to build multimodal teacher-forced sequence.")
        out = adapter.model(
            inputs_embeds=mm_embeds_e,
            attention_mask=attn_mask_e,
            position_ids=pos_ids_e,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    labels_exp = labels_e[0]
    text_positions = torch.where(labels_exp != int(adapter.IGNORE_INDEX))[0]
    if int(text_positions.numel()) < int(cont_ids.numel()):
        raise RuntimeError("Expanded sequence shorter than continuation token count.")
    cont_label_positions = text_positions[-int(cont_ids.numel()):]
    decision_positions = cont_label_positions - 1
    target_ids = labels_exp[cont_label_positions].long()

    token_logits = out.logits[0][decision_positions].to(torch.float32)
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

    cont_token_ids = [int(x) for x in cont_ids.tolist()]
    cont_idx = list(range(int(len(cont_token_ids))))
    pick = adapter._select_content_token_indices(cont_token_ids)
    if not pick:
        pick = cont_idx

    def pick_values(t: torch.Tensor, idxs: Sequence[int]) -> List[float]:
        return [float(t[int(i)].item()) for i in idxs]

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

    row: Dict[str, Any] = {
        "id": prepared["sample_id"],
        "image": prepared["image_file"],
        "question": prepared["question"],
        "n_cont_tokens": int(len(cont_idx)),
        "n_content_tokens": int(len(pick)),
        "cheap_content_fraction": float(len(pick) / max(1, len(cont_idx))),
        "cheap_conflict_lp_minus_entropy": float(mean_or_zero(lp_content) - mean_or_zero(ent_content)),
        "cheap_conflict_gap_minus_entropy": float(mean_or_zero(gap_content) - mean_or_zero(ent_content)),
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


def compute_binary_metrics(rows: Sequence[Dict[str, Any]], label_key: str) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    n = 0
    for row in rows:
        pred = maybe_int(row.get("proxy_rescue"))
        label = maybe_int(row.get(label_key))
        if pred is None or label is None:
            continue
        n += 1
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1
    precision = None if (tp + fp) == 0 else float(tp / float(tp + fp))
    recall = None if (tp + fn) == 0 else float(tp / float(tp + fn))
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {
        "n": int(n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="True full sample-wise VGA + cheap proxy + conditional baseline replay.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--vga_root", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--headset_json", type=str, default="")
    ap.add_argument("--max_gen_len", type=int, default=8)
    ap.add_argument("--use_add", type=parse_bool, default=True)
    ap.add_argument("--attn_coef", type=float, default=0.2)
    ap.add_argument("--head_balancing", type=str, default="simg")
    ap.add_argument("--sampling", type=parse_bool, default=False)
    ap.add_argument("--cd_alpha", type=float, default=0.02)
    ap.add_argument("--start_layer", type=int, default=2)
    ap.add_argument("--end_layer", type=int, default=15)
    ap.add_argument("--attn_norm", type=parse_bool, default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--reference_vga_pred_jsonl", type=str, default="")
    ap.add_argument("--reference_baseline_pred_jsonl", type=str, default="")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    decisions_csv = os.path.join(out_dir, "decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(decisions_csv) and os.path.isfile(summary_json):
        print(f"[reuse] {summary_json}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    gt_map = load_label_map(args.gt_csv)
    ref_vga_map = (
        load_prediction_text_map(args.reference_vga_pred_jsonl, text_key="auto")
        if str(args.reference_vga_pred_jsonl or "").strip()
        else {}
    )
    ref_base_map = (
        load_prediction_text_map(args.reference_baseline_pred_jsonl, text_key="auto")
        if str(args.reference_baseline_pred_jsonl or "").strip()
        else {}
    )
    policy = load_policy(args.policy_json)

    adapter = VGAOnlineAdapter(
        VGAOnlineConfig(
            vga_root=args.vga_root,
            model_path=args.model_path,
            image_folder=args.image_folder,
            conv_mode=args.conv_mode,
                model_base=(args.model_base or None),
                device=args.device,
                headset_json=str(args.headset_json or ""),
                sampling=bool(args.sampling),
                max_gen_len=int(args.max_gen_len),
            cd_alpha=float(args.cd_alpha),
            attn_coef=float(args.attn_coef),
            start_layer=int(args.start_layer),
            end_layer=int(args.end_layer),
            head_balancing=str(args.head_balancing),
            attn_norm=bool(args.attn_norm),
            seed=int(args.seed),
            prefer_local_llava=False,
            proxy_trace_enabled=False,
        )
    )

    rows: List[Dict[str, Any]] = []
    vga_secs: List[float] = []
    cheap_secs: List[float] = []
    baseline_secs: List[float] = []
    n_errors = 0
    intervention_correct_sum = 0
    final_correct_sum = 0
    baseline_live_correct_sum = 0
    baseline_ref_correct_sum = 0
    n_eval = 0
    n_baseline_live = 0
    n_baseline_ref = 0
    vga_ref_text_match = 0
    vga_ref_label_match = 0
    vga_ref_total = 0
    base_ref_text_match = 0
    base_ref_label_match = 0
    base_ref_total = 0

    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        gt_label = str(gt_map.get(sample_id, "")).strip().lower()
        row: Dict[str, Any] = {
            "id": sample_id,
            "image": str(sample.get("image", "")).strip(),
            "question": str(sample.get("text", sample.get("question", ""))).strip(),
            "gt_label": gt_label,
            "feature_error": "",
            "proxy_rescue": 0,
        }
        try:
            vga_t0 = time.perf_counter()
            ctx = adapter._prepare_runtime_context(sample)
            pred = adapter._generate_from_prepared(
                sample=sample,
                prepared=ctx["prepared"],
                gen_kwargs=ctx["method_gen_kwargs"],
                use_add=bool(args.use_add),
                capture_proxy=False,
            )
            vga_dt = time.perf_counter() - vga_t0
            vga_secs.append(float(vga_dt))

            intervention_text = str(pred.get("output", "")).strip()
            intervention_label = parse_yes_no(intervention_text) if intervention_text else ""
            intervention_correct = None if gt_label not in {"yes", "no"} or intervention_label == "" else int(intervention_label == gt_label)
            row.update(
                {
                    "intervention_text_live": intervention_text,
                    "intervention_label_live": intervention_label,
                    "intervention_correct_live": intervention_correct,
                    "vga_live_ms": float(vga_dt * 1000.0),
                }
            )

            ref_vga_text = str(ref_vga_map.get(sample_id, "")).strip()
            ref_vga_label = parse_yes_no(ref_vga_text) if ref_vga_text else ""
            row["reference_vga_text"] = ref_vga_text
            row["reference_vga_label"] = ref_vga_label
            row["reference_vga_text_match"] = None if not ref_vga_text else int(normalize_text(ref_vga_text) == normalize_text(intervention_text))
            row["reference_vga_label_match"] = None if not ref_vga_label or not intervention_label else int(ref_vga_label == intervention_label)
            if ref_vga_text:
                vga_ref_total += 1
                vga_ref_text_match += int(row["reference_vga_text_match"] or 0)
            if ref_vga_label and intervention_label:
                vga_ref_label_match += int(row["reference_vga_label_match"] or 0)

            cheap_t0 = time.perf_counter()
            feature_row = build_cheap_feature_row_from_prepared(
                adapter=adapter,
                prepared=ctx["prepared"],
                candidate_text=intervention_text,
            )
            cheap_dt = time.perf_counter() - cheap_t0
            cheap_secs.append(float(cheap_dt))
            row.update(feature_row)
            row["cheap_feature_ms"] = float(cheap_dt * 1000.0)

            score, rescue = apply_policy(policy, row)
            row["proxy_score"] = score
            row["proxy_rescue"] = int(rescue)

            ref_base_text = str(ref_base_map.get(sample_id, "")).strip()
            ref_base_label = parse_yes_no(ref_base_text) if ref_base_text else ""
            ref_base_correct = None if gt_label not in {"yes", "no"} or ref_base_label == "" else int(ref_base_label == gt_label)
            if ref_base_correct is not None:
                baseline_ref_correct_sum += int(ref_base_correct)
                n_baseline_ref += 1
            row["reference_baseline_text"] = ref_base_text
            row["reference_baseline_label"] = ref_base_label
            row["reference_baseline_correct"] = ref_base_correct
            row["actual_rescue_eval"] = (
                None if ref_base_correct is None or intervention_correct is None
                else int(int(ref_base_correct) == 1 and int(intervention_correct) == 0)
            )

            final_text = intervention_text
            final_label = intervention_label
            final_correct = intervention_correct
            if int(rescue) == 1:
                base_t0 = time.perf_counter()
                base_pred = adapter._generate_from_prepared(
                    sample=sample,
                    prepared=ctx["prepared"],
                    gen_kwargs=ctx["base_gen_kwargs"],
                    use_add=False,
                    capture_proxy=False,
                )
                base_dt = time.perf_counter() - base_t0
                baseline_secs.append(float(base_dt))
                baseline_text_live = str(base_pred.get("output", "")).strip()
                baseline_label_live = parse_yes_no(baseline_text_live) if baseline_text_live else ""
                baseline_correct_live = (
                    None if gt_label not in {"yes", "no"} or baseline_label_live == ""
                    else int(baseline_label_live == gt_label)
                )
                if baseline_correct_live is not None:
                    baseline_live_correct_sum += int(baseline_correct_live)
                    n_baseline_live += 1

                row["baseline_text_live"] = baseline_text_live
                row["baseline_label_live"] = baseline_label_live
                row["baseline_correct_live"] = baseline_correct_live
                row["baseline_live_ms"] = float(base_dt * 1000.0)

                if ref_base_text:
                    base_ref_total += 1
                    row["reference_baseline_text_match_live"] = int(normalize_text(ref_base_text) == normalize_text(baseline_text_live))
                    base_ref_text_match += int(row["reference_baseline_text_match_live"])
                    if ref_base_label and baseline_label_live:
                        row["reference_baseline_label_match_live"] = int(ref_base_label == baseline_label_live)
                        base_ref_label_match += int(row["reference_baseline_label_match_live"])
                    else:
                        row["reference_baseline_label_match_live"] = None
                else:
                    row["reference_baseline_text_match_live"] = None
                    row["reference_baseline_label_match_live"] = None

                final_text = baseline_text_live
                final_label = baseline_label_live
                final_correct = baseline_correct_live
            else:
                row["baseline_text_live"] = ""
                row["baseline_label_live"] = ""
                row["baseline_correct_live"] = None
                row["baseline_live_ms"] = ""
                row["reference_baseline_text_match_live"] = None
                row["reference_baseline_label_match_live"] = None

            row["final_text_live"] = final_text
            row["final_label_live"] = final_label
            row["final_correct_live"] = final_correct

            if intervention_correct is not None:
                intervention_correct_sum += int(intervention_correct)
                n_eval += 1
            if final_correct is not None:
                final_correct_sum += int(final_correct)
        except Exception as exc:
            n_errors += 1
            row["feature_error"] = str(exc)

        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[true-full] {idx + 1}/{len(question_rows)}")

    rescue_count = sum(int(maybe_int(row.get("proxy_rescue")) or 0) for row in rows)
    rescue_rate = None if not rows else float(rescue_count / float(len(rows)))
    intervention_acc = None if n_eval == 0 else float(intervention_correct_sum / float(n_eval))
    final_acc = None if n_eval == 0 else float(final_correct_sum / float(n_eval))
    actual_metrics = compute_binary_metrics(rows, "actual_rescue_eval")

    live_gain_count = 0
    live_gain_total = 0
    for row in rows:
        if int(maybe_int(row.get("proxy_rescue")) or 0) != 1:
            continue
        bc = maybe_int(row.get("baseline_correct_live"))
        ic = maybe_int(row.get("intervention_correct_live"))
        if bc is None or ic is None:
            continue
        live_gain_total += 1
        if int(bc) == 1 and int(ic) == 0:
            live_gain_count += 1

    vga_total_sec = float(sum(vga_secs))
    cheap_total_sec = float(sum(cheap_secs))
    baseline_total_sec = float(sum(baseline_secs))
    total_sec = float(vga_total_sec + cheap_total_sec + baseline_total_sec)
    ratio_vs_vga = None if vga_total_sec <= 0 else float(total_sec / vga_total_sec)

    write_csv(decisions_csv, rows)
    write_json(
        summary_json,
        {
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "policy_json": os.path.abspath(args.policy_json),
                "gt_csv": os.path.abspath(args.gt_csv),
                "vga_root": os.path.abspath(args.vga_root),
                "reference_vga_pred_jsonl": (
                    os.path.abspath(args.reference_vga_pred_jsonl)
                    if str(args.reference_vga_pred_jsonl or "").strip()
                    else ""
                ),
                "reference_baseline_pred_jsonl": (
                    os.path.abspath(args.reference_baseline_pred_jsonl)
                    if str(args.reference_baseline_pred_jsonl or "").strip()
                    else ""
                ),
                "model_path": args.model_path,
                "model_base": args.model_base,
                "conv_mode": args.conv_mode,
                "device": args.device,
                "headset_json": (
                    os.path.abspath(args.headset_json)
                    if str(args.headset_json or "").strip()
                    else ""
                ),
                "max_gen_len": int(args.max_gen_len),
                "use_add": bool(args.use_add),
                "attn_coef": float(args.attn_coef),
                "head_balancing": str(args.head_balancing),
                "sampling": bool(args.sampling),
                "cd_alpha": float(args.cd_alpha),
                "start_layer": int(args.start_layer),
                "end_layer": int(args.end_layer),
                "attn_norm": bool(args.attn_norm),
            },
            "policy": policy,
            "evaluation": {
                "n_eval": int(n_eval),
                "n_errors": int(n_errors),
                "rescue_count": int(rescue_count),
                "rescue_rate": rescue_rate,
                "intervention_acc_live": intervention_acc,
                "final_acc_live": final_acc,
                "delta_vs_intervention_live": (
                    None if intervention_acc is None or final_acc is None else float(final_acc - intervention_acc)
                ),
                "reference_baseline_acc_eval": (
                    None if n_baseline_ref == 0 else float(baseline_ref_correct_sum / float(n_baseline_ref))
                ),
                "baseline_acc_live_rescued_only": (
                    None if n_baseline_live == 0 else float(baseline_live_correct_sum / float(n_baseline_live))
                ),
                "actual_precision_eval": actual_metrics["precision"],
                "actual_recall_eval": actual_metrics["recall"],
                "actual_f1_eval": actual_metrics["f1"],
                "actual_tp_eval": actual_metrics["tp"],
                "actual_fp_eval": actual_metrics["fp"],
                "actual_tn_eval": actual_metrics["tn"],
                "actual_fn_eval": actual_metrics["fn"],
                "rescued_live_gain_rate": (
                    None if live_gain_total == 0 else float(live_gain_count / float(live_gain_total))
                ),
                "reference_vga_text_match_rate": (
                    None if vga_ref_total == 0 else float(vga_ref_text_match / float(vga_ref_total))
                ),
                "reference_vga_label_match_rate": (
                    None if vga_ref_total == 0 else float(vga_ref_label_match / float(vga_ref_total))
                ),
                "reference_baseline_text_match_rate_rescued": (
                    None if base_ref_total == 0 else float(base_ref_text_match / float(base_ref_total))
                ),
                "reference_baseline_label_match_rate_rescued": (
                    None if base_ref_total == 0 else float(base_ref_label_match / float(base_ref_total))
                ),
            },
            "timing": {
                "vga_total_sec": vga_total_sec,
                "vga_mean_ms": float(1000.0 * mean_or_zero(vga_secs)),
                "cheap_feature_total_sec": cheap_total_sec,
                "cheap_feature_mean_ms": float(1000.0 * mean_or_zero(cheap_secs)),
                "baseline_live_total_sec": baseline_total_sec,
                "baseline_live_mean_ms_rescued": float(1000.0 * mean_or_zero(baseline_secs)),
                "full_total_sec": total_sec,
                "full_mean_ms_per_sample": (None if not rows else float(1000.0 * total_sec / float(len(rows)))),
                "ratio_vs_vga": ratio_vs_vga,
                "pass_proxy_structural": (None if rescue_rate is None else float(2.0 + rescue_rate)),
            },
            "outputs": {
                "decision_rows_csv": os.path.abspath(decisions_csv),
            },
        },
    )
    print(f"[saved] {decisions_csv}")
    print(f"[saved] {summary_json}")


if __name__ == "__main__":
    main()
