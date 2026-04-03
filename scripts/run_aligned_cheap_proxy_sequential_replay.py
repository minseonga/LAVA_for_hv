#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from extract_c_stage_cheap_online_features import build_feature_row
from frgavr_cleanroom.runtime import (
    CleanroomLlavaRuntime,
    load_label_map,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    parse_yes_no,
    safe_id,
    write_csv,
    write_json,
)


def maybe_float(value: object) -> Optional[float]:
    s = str(value or "").strip()
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


def eval_binary_flags(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    n = 0
    for row in rows:
        rescue = int(row.get("proxy_rescue", 0))
        label = maybe_int(row.get(key))
        if label is None:
            continue
        n += 1
        if rescue == 1 and int(label) == 1:
            tp += 1
        elif rescue == 1 and int(label) == 0:
            fp += 1
        elif rescue == 0 and int(label) == 0:
            tn += 1
        elif rescue == 0 and int(label) == 1:
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
    ap = argparse.ArgumentParser(description="Sequential replay for aligned cheap proxy on canonical VGA outputs.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--intervention_pred_key", type=str, default="auto")
    ap.add_argument("--baseline_eval_pred_jsonl", type=str, default="")
    ap.add_argument("--baseline_eval_pred_key", type=str, default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--vga_total_wall_sec", type=float, default=-1.0)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    decisions_csv = os.path.join(out_dir, "decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(decisions_csv) and os.path.isfile(summary_json):
        print(f"[reuse] {summary_json}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    intervention_map = load_prediction_text_map(args.intervention_pred_jsonl, text_key=args.intervention_pred_key)
    baseline_eval_map = (
        load_prediction_text_map(args.baseline_eval_pred_jsonl, text_key=args.baseline_eval_pred_key)
        if str(args.baseline_eval_pred_jsonl or "").strip()
        else {}
    )
    gt_map = load_label_map(args.gt_csv)
    policy = load_policy(args.policy_json)

    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(args.model_base or None),
        conv_mode=args.conv_mode,
        device=args.device,
    )

    decision_rows: List[Dict[str, Any]] = []
    n_errors = 0
    cheap_secs: List[float] = []
    baseline_secs: List[float] = []
    intervention_correct_sum = 0
    final_correct_sum = 0
    baseline_eval_correct_sum = 0
    n_eval = 0
    n_baseline_eval = 0

    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        image_path = os.path.join(args.image_folder, image_name)
        intervention_text = str(intervention_map.get(sample_id, "")).strip()
        gt_label = str(gt_map.get(sample_id, "")).strip().lower()
        intervention_label = parse_yes_no(intervention_text) if intervention_text else ""
        intervention_correct = None if gt_label not in {"yes", "no"} or intervention_label == "" else int(intervention_label == gt_label)

        row: Dict[str, Any] = {
            "id": sample_id,
            "image": image_name,
            "question": question,
            "gt_label": gt_label,
            "intervention_text": intervention_text,
            "intervention_label": intervention_label,
            "intervention_correct": intervention_correct,
            "feature_error": "",
            "baseline_text_live": "",
            "baseline_label_live": "",
            "baseline_correct_live": "",
            "proxy_score": "",
            "proxy_rescue": 0,
            "cheap_feature_ms": "",
            "baseline_live_ms": "",
        }
        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not question:
                raise ValueError("Missing question text.")
            if not intervention_text:
                raise ValueError("Missing intervention prediction text.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            cheap_t0 = time.perf_counter()
            feature_row = build_feature_row(
                runtime=runtime,
                image_path=image_path,
                question=question,
                candidate_text=intervention_text,
                sample_id=sample_id,
                image_name=image_name,
            )
            cheap_dt = time.perf_counter() - cheap_t0
            cheap_secs.append(float(cheap_dt))
            row["cheap_feature_ms"] = float(cheap_dt * 1000.0)
            row.update(feature_row)

            score, rescue = apply_policy(policy, row)
            row["proxy_score"] = score
            row["proxy_rescue"] = int(rescue)

            baseline_eval_text = str(baseline_eval_map.get(sample_id, "")).strip()
            baseline_eval_label = parse_yes_no(baseline_eval_text) if baseline_eval_text else ""
            baseline_eval_correct = None
            if gt_label in {"yes", "no"} and baseline_eval_label:
                baseline_eval_correct = int(baseline_eval_label == gt_label)
                baseline_eval_correct_sum += int(baseline_eval_correct)
                n_baseline_eval += 1
            row["baseline_correct_eval"] = baseline_eval_correct
            row["actual_rescue_eval"] = (
                None if baseline_eval_correct is None or intervention_correct is None
                else int(int(baseline_eval_correct) == 1 and int(intervention_correct) == 0)
            )

            final_text = intervention_text
            final_label = intervention_label
            final_correct = intervention_correct
            if int(rescue) == 1:
                image = runtime.load_image(image_path)
                baseline_t0 = time.perf_counter()
                baseline_live_text = runtime.generate_baseline(
                    image=image,
                    question=question,
                    max_new_tokens=int(args.max_new_tokens),
                )
                baseline_dt = time.perf_counter() - baseline_t0
                baseline_secs.append(float(baseline_dt))
                baseline_live_label = parse_yes_no(baseline_live_text)
                baseline_live_correct = None if gt_label not in {"yes", "no"} else int(baseline_live_label == gt_label)

                row["baseline_text_live"] = baseline_live_text
                row["baseline_label_live"] = baseline_live_label
                row["baseline_correct_live"] = baseline_live_correct
                row["baseline_live_ms"] = float(baseline_dt * 1000.0)

                final_text = baseline_live_text
                final_label = baseline_live_label
                final_correct = baseline_live_correct

            row["final_text"] = final_text
            row["final_label"] = final_label
            row["final_correct"] = final_correct

            if intervention_correct is not None:
                intervention_correct_sum += int(intervention_correct)
                n_eval += 1
            if final_correct is not None:
                final_correct_sum += int(final_correct)
        except Exception as exc:
            n_errors += 1
            row["feature_error"] = str(exc)

        decision_rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[replay] {idx + 1}/{len(question_rows)}")

    rescue_flags = [int(maybe_int(row.get("proxy_rescue")) or 0) for row in decision_rows]
    rescue_count = int(sum(rescue_flags))
    rescue_rate = None if not decision_rows else float(rescue_count / float(len(decision_rows)))
    intervention_acc = None if n_eval == 0 else float(intervention_correct_sum / float(n_eval))
    final_acc = None if n_eval == 0 else float(final_correct_sum / float(n_eval))

    actual_metrics = eval_binary_flags(decision_rows, "actual_rescue_eval")
    live_rescue_gain_count = 0
    live_rescue_eval_count = 0
    for row in decision_rows:
        if int(maybe_int(row.get("proxy_rescue")) or 0) != 1:
            continue
        bc = maybe_int(row.get("baseline_correct_live"))
        ic = maybe_int(row.get("intervention_correct"))
        if bc is None or ic is None:
            continue
        live_rescue_eval_count += 1
        if int(bc) == 1 and int(ic) == 0:
            live_rescue_gain_count += 1

    cheap_total_sec = float(sum(cheap_secs))
    baseline_total_sec = float(sum(baseline_secs))
    post_vga_total_sec = float(cheap_total_sec + baseline_total_sec)
    vga_total_wall_sec = None if float(args.vga_total_wall_sec) <= 0 else float(args.vga_total_wall_sec)
    estimated_total_sec = None if vga_total_wall_sec is None else float(vga_total_wall_sec + post_vga_total_sec)
    estimated_ratio_vs_vga = None if vga_total_wall_sec is None or vga_total_wall_sec <= 0 else float(estimated_total_sec / vga_total_wall_sec)

    write_csv(decisions_csv, decision_rows)
    write_json(
        summary_json,
        {
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "policy_json": os.path.abspath(args.policy_json),
                "gt_csv": os.path.abspath(args.gt_csv),
                "baseline_eval_pred_jsonl": (
                    os.path.abspath(args.baseline_eval_pred_jsonl)
                    if str(args.baseline_eval_pred_jsonl or "").strip()
                    else ""
                ),
                "model_path": args.model_path,
                "model_base": args.model_base,
                "conv_mode": args.conv_mode,
                "device": args.device,
                "max_new_tokens": int(args.max_new_tokens),
                "vga_total_wall_sec": vga_total_wall_sec,
            },
            "policy": policy,
            "evaluation": {
                "n_eval": int(n_eval),
                "n_errors": int(n_errors),
                "rescue_count": int(rescue_count),
                "rescue_rate": rescue_rate,
                "baseline_acc_eval": (None if n_baseline_eval == 0 else float(baseline_eval_correct_sum / float(n_baseline_eval))),
                "intervention_acc": intervention_acc,
                "final_acc": final_acc,
                "delta_vs_intervention": (
                    None if intervention_acc is None or final_acc is None else float(final_acc - intervention_acc)
                ),
                "actual_precision_eval": actual_metrics["precision"],
                "actual_recall_eval": actual_metrics["recall"],
                "actual_f1_eval": actual_metrics["f1"],
                "actual_tp_eval": actual_metrics["tp"],
                "actual_fp_eval": actual_metrics["fp"],
                "actual_tn_eval": actual_metrics["tn"],
                "actual_fn_eval": actual_metrics["fn"],
                "rescued_live_gain_rate": (
                    None if live_rescue_eval_count == 0 else float(live_rescue_gain_count / float(live_rescue_eval_count))
                ),
            },
            "timing": {
                "cheap_feature_total_sec": cheap_total_sec,
                "cheap_feature_mean_ms": float(1000.0 * mean_or_zero(cheap_secs)),
                "baseline_live_total_sec": baseline_total_sec,
                "baseline_live_mean_ms_rescued": float(1000.0 * mean_or_zero(baseline_secs)),
                "post_vga_total_sec": post_vga_total_sec,
                "post_vga_mean_ms_per_sample": (
                    None if not decision_rows else float(1000.0 * post_vga_total_sec / float(len(decision_rows)))
                ),
                "pass_proxy_structural": (
                    None if rescue_rate is None else float(2.0 + rescue_rate)
                ),
                "estimated_total_sec_including_vga": estimated_total_sec,
                "estimated_ratio_vs_vga": estimated_ratio_vs_vga,
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
