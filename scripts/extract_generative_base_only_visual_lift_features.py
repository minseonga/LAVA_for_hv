#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import (  # noqa: E402
    CleanroomLlavaRuntime,
    load_question_rows,
    parse_bool,
    safe_id,
    select_content_indices,
    write_csv,
    write_json,
)
from analyze_generative_base_only_confidence_proxy import (  # noqa: E402
    average_precision,
    auc_high,
    feature_metrics,
    flag,
    read_csv_rows,
)
from extract_generative_semantic_pairwise_features import (  # noqa: E402
    normalize_token,
    read_prediction_map,
    sorted_preview,
    unit_summary,
)
from extract_vga_generative_mention_features import (  # noqa: E402
    compute_pack_values,
    decode_token_spans,
    extract_word_spans,
)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / float(len(values))) if values else 0.0


def min0(values: Sequence[float]) -> float:
    return float(min(values)) if values else 0.0


def max0(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    idx = min(len(xs) - 1, max(0, int(round(float(q) * (len(xs) - 1)))))
    return float(xs[idx])


def pick(values: Sequence[float], idxs: Sequence[int]) -> List[float]:
    return [float(values[int(idx)]) for idx in idxs if 0 <= int(idx) < len(values)]


def make_ablation(runtime: CleanroomLlavaRuntime, image: Any, mode: str, blur_radius: float) -> Any:
    mode = str(mode or "blur").strip().lower()
    if mode == "blur":
        return runtime.make_blur_control(image, blur_radius=float(blur_radius))
    if mode == "gray":
        # Keep the same resolution and image channel structure while removing
        # most spatial/semantic content.
        from PIL import Image

        px = image.resize((1, 1)).getpixel((0, 0))
        if not isinstance(px, tuple):
            px = (int(px), int(px), int(px))
        return Image.new("RGB", image.size, tuple(int(x) for x in px[:3]))
    if mode == "black":
        from PIL import Image

        return Image.new("RGB", image.size, (0, 0, 0))
    raise ValueError(f"Unsupported ablation_mode={mode!r}")


def word_trace_from_packs(runtime: CleanroomLlavaRuntime, real_pack: Any, control_pack: Any) -> List[Dict[str, Any]]:
    content_indices = select_content_indices(runtime.tokenizer, real_pack.cont_ids)
    content_set = {int(idx) for idx in content_indices}
    decoded_text, token_spans = decode_token_spans(runtime.tokenizer, real_pack.cont_ids)
    real_vals = compute_pack_values(real_pack)
    ctrl_vals = compute_pack_values(control_pack)
    rows: List[Dict[str, Any]] = []
    word_pos = 0
    for start, end, raw_word in extract_word_spans(decoded_text):
        word = normalize_token(raw_word)
        if not word:
            continue
        idxs: List[int] = []
        for idx, (tok_start, tok_end) in enumerate(token_spans):
            if idx not in content_set or tok_end <= tok_start:
                continue
            if max(int(tok_start), int(start)) < min(int(tok_end), int(end)):
                idxs.append(int(idx))
        if not idxs:
            word_pos += 1
            continue
        real_lp = pick(real_vals["lp"], idxs)
        real_gap = pick(real_vals["gap"], idxs)
        real_ent = pick(real_vals["ent"], idxs)
        ctrl_lp = pick(ctrl_vals["lp"], idxs)
        ctrl_gap = pick(ctrl_vals["gap"], idxs)
        ctrl_ent = pick(ctrl_vals["ent"], idxs)
        lp_lift_vals = [a - b for a, b in zip(real_lp, ctrl_lp)]
        gap_lift_vals = [a - b for a, b in zip(real_gap, ctrl_gap)]
        ent_delta_vals = [a - b for a, b in zip(real_ent, ctrl_ent)]
        rows.append(
            {
                "word": word,
                "pos": int(word_pos),
                "n_tokens": int(len(idxs)),
                "real_lp_min": min0(real_lp),
                "real_lp_mean": mean(real_lp),
                "real_gap_min": min0(real_gap),
                "real_gap_mean": mean(real_gap),
                "real_ent_max": max0(real_ent),
                "real_ent_mean": mean(real_ent),
                "ctrl_lp_min": min0(ctrl_lp),
                "ctrl_lp_mean": mean(ctrl_lp),
                "ctrl_gap_min": min0(ctrl_gap),
                "ctrl_gap_mean": mean(ctrl_gap),
                "ctrl_ent_max": max0(ctrl_ent),
                "ctrl_ent_mean": mean(ctrl_ent),
                "lp_lift_min": min0(lp_lift_vals),
                "lp_lift_mean": mean(lp_lift_vals),
                "lp_lift_max": max0(lp_lift_vals),
                "gap_lift_min": min0(gap_lift_vals),
                "gap_lift_mean": mean(gap_lift_vals),
                "gap_lift_max": max0(gap_lift_vals),
                "ent_delta_mean": mean(ent_delta_vals),
            }
        )
        word_pos += 1
    return rows


def add_lift_stats(row: Dict[str, Any], prefix: str, traces: Sequence[Dict[str, Any]]) -> None:
    real_lp = [float(t["real_lp_min"]) for t in traces]
    real_gap = [float(t["real_gap_min"]) for t in traces]
    real_ent = [float(t["real_ent_max"]) for t in traces]
    lp_lift = [float(t["lp_lift_mean"]) for t in traces]
    gap_lift = [float(t["gap_lift_mean"]) for t in traces]
    ent_delta = [float(t["ent_delta_mean"]) for t in traces]
    support = [
        max(0.0, float(t["lp_lift_mean"]))
        + max(0.0, float(t["gap_lift_mean"]))
        + max(0.0, float(t["real_gap_min"]))
        for t in traces
    ]
    row[f"{prefix}_trace_count"] = int(len(traces))
    row[f"{prefix}_real_lp_mean"] = mean(real_lp)
    row[f"{prefix}_real_lp_min"] = min0(real_lp)
    row[f"{prefix}_real_gap_mean"] = mean(real_gap)
    row[f"{prefix}_real_gap_min"] = min0(real_gap)
    row[f"{prefix}_real_ent_mean"] = mean(real_ent)
    row[f"{prefix}_real_ent_max"] = max0(real_ent)
    row[f"{prefix}_lp_lift_mean"] = mean(lp_lift)
    row[f"{prefix}_lp_lift_min"] = min0(lp_lift)
    row[f"{prefix}_lp_lift_max"] = max0(lp_lift)
    row[f"{prefix}_lp_lift_q90"] = quantile(lp_lift, 0.90)
    row[f"{prefix}_lp_lift_sum_pos"] = sum(max(0.0, v) for v in lp_lift)
    row[f"{prefix}_lp_lift_count_gt_000"] = sum(1 for v in lp_lift if v > 0.0)
    row[f"{prefix}_lp_lift_count_gt_025"] = sum(1 for v in lp_lift if v > 0.25)
    row[f"{prefix}_lp_lift_count_gt_050"] = sum(1 for v in lp_lift if v > 0.50)
    row[f"{prefix}_gap_lift_mean"] = mean(gap_lift)
    row[f"{prefix}_gap_lift_min"] = min0(gap_lift)
    row[f"{prefix}_gap_lift_max"] = max0(gap_lift)
    row[f"{prefix}_gap_lift_sum_pos"] = sum(max(0.0, v) for v in gap_lift)
    row[f"{prefix}_ent_delta_mean"] = mean(ent_delta)
    row[f"{prefix}_support_score_sum"] = sum(support)
    row[f"{prefix}_support_score_mean"] = mean(support)
    row[f"{prefix}_vis_conf_count_gap000_ent250_lift000"] = sum(
        1
        for t in traces
        if float(t["real_gap_min"]) >= 0.0
        and float(t["real_ent_max"]) <= 2.5
        and float(t["lp_lift_mean"]) > 0.0
    )
    row[f"{prefix}_vis_conf_count_gap000_ent250_lift025"] = sum(
        1
        for t in traces
        if float(t["real_gap_min"]) >= 0.0
        and float(t["real_ent_max"]) <= 2.5
        and float(t["lp_lift_mean"]) > 0.25
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract no-CHAIR/GT baseline-only visual-lift features. "
            "For baseline-only caption units, compares teacher-forced logprob "
            "under the real image vs an ablated image."
        )
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--oracle_rows_csv", default="")
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_feature_metrics_csv", default="")
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pred_text_key", default="auto")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--ablation_mode", default="blur", choices=["blur", "gray", "black"])
    ap.add_argument("--blur_radius", type=float, default=8.0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--preview_limit", type=int, default=120)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    baseline = read_prediction_map(args.baseline_pred_jsonl, text_key=str(args.pred_text_key))
    intervention = read_prediction_map(args.intervention_pred_jsonl, text_key=str(args.pred_text_key))
    oracle = {}
    if args.oracle_rows_csv and os.path.isfile(args.oracle_rows_csv):
        oracle = {safe_id(row.get("image_id") or row.get("id") or row.get("question_id")): row for row in read_csv_rows(args.oracle_rows_csv)}

    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=(str(args.model_base).strip() or None),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id") or sample.get("image_id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text") or sample.get("question") or "").strip()
        image_path = os.path.join(args.image_folder, image_name)
        base_text = baseline.get(sample_id, {}).get("text", "")
        int_text = intervention.get(sample_id, {}).get("text", "")
        row: Dict[str, Any] = {
            "id": sample_id,
            "image": image_name,
            "score_error": "",
        }
        if oracle:
            row[str(args.target_col)] = flag(oracle.get(sample_id, {}).get(args.target_col))
        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not image_name or not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not question:
                raise ValueError("Missing question text.")
            if not base_text or not int_text:
                raise ValueError("Missing baseline or intervention text.")

            base_units = set(unit_summary(base_text)["unique_token_units"])
            int_units = set(unit_summary(int_text)["unique_token_units"])
            base_only_units = base_units - int_units
            shared_units = base_units & int_units
            image = runtime.load_image(image_path)
            control = make_ablation(runtime, image, mode=str(args.ablation_mode), blur_radius=float(args.blur_radius))
            real_pack = runtime.teacher_force_candidate(
                image=image,
                question=question,
                candidate_text=base_text,
                output_attentions=False,
            )
            ctrl_pack = runtime.teacher_force_candidate(
                image=control,
                question=question,
                candidate_text=base_text,
                output_attentions=False,
            )
            trace = word_trace_from_packs(runtime, real_pack, ctrl_pack)
            bo_trace = [t for t in trace if str(t["word"]) in base_only_units]
            shared_trace = [t for t in trace if str(t["word"]) in shared_units]
            bo_words = {str(t["word"]) for t in bo_trace}
            shared_words = {str(t["word"]) for t in shared_trace}

            row.update(
                {
                    "base_only_unit_count": len(base_only_units),
                    "shared_unit_count": len(shared_units),
                    "base_unit_count": len(base_units),
                    "int_unit_count": len(int_units),
                    "base_only_trace_coverage": len(bo_words) / float(max(1, len(base_only_units))),
                    "shared_trace_coverage": len(shared_words) / float(max(1, len(shared_units))),
                    "base_only_units": sorted_preview(base_only_units, int(args.preview_limit)),
                    "base_only_trace_words": sorted_preview(bo_words, int(args.preview_limit)),
                    "word_trace_json": json.dumps(trace, ensure_ascii=False, separators=(",", ":")),
                }
            )
            add_lift_stats(row, "bo_vlift", bo_trace)
            add_lift_stats(row, "shared_vlift", shared_trace)
            row["bo_minus_shared_lp_lift_mean"] = float(row["bo_vlift_lp_lift_mean"] - row["shared_vlift_lp_lift_mean"])
            row["bo_minus_shared_gap_lift_mean"] = float(row["bo_vlift_gap_lift_mean"] - row["shared_vlift_gap_lift_mean"])
            row["bo_lift_score_x_bo_count"] = float(row["bo_vlift_support_score_sum"] * len(base_only_units))
            row["bo_lift_conf_count_x_bo_count"] = float(
                row["bo_vlift_vis_conf_count_gap000_ent250_lift000"] * len(base_only_units)
            )
        except Exception as exc:
            n_errors += 1
            row["score_error"] = str(exc)
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[visual-lift] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {os.path.abspath(args.out_csv)}")

    metrics: List[Dict[str, Any]] = []
    if args.out_feature_metrics_csv and oracle:
        metrics = feature_metrics(rows, str(args.target_col))
        write_csv(args.out_feature_metrics_csv, metrics)
        print(f"[saved] {os.path.abspath(args.out_feature_metrics_csv)}")
        for metric in metrics[:10]:
            print(
                "[metric]",
                metric["feature"],
                "dir=",
                metric["direction"],
                "auc=",
                metric["auroc"],
                "ap=",
                metric["ap"],
            )

    write_json(
        args.out_summary_json,
        {
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv) if args.oracle_rows_csv else "",
                "target_col": args.target_col,
                "model_path": args.model_path,
                "model_base": args.model_base,
                "conv_mode": args.conv_mode,
                "device": args.device,
                "limit": int(args.limit),
                "ablation_mode": args.ablation_mode,
                "blur_radius": float(args.blur_radius),
            },
            "counts": {
                "n_rows": len(rows),
                "n_errors": int(n_errors),
                "n_feature_metrics": len(metrics),
            },
            "top_feature_metrics": metrics[:30],
            "outputs": {
                "features_csv": os.path.abspath(args.out_csv),
                "feature_metrics_csv": os.path.abspath(args.out_feature_metrics_csv) if args.out_feature_metrics_csv else "",
                "summary_json": os.path.abspath(args.out_summary_json),
            },
        },
    )
    print(f"[saved] {os.path.abspath(args.out_summary_json)}")


if __name__ == "__main__":
    main()
