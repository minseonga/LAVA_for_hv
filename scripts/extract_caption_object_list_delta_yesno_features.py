#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Sequence

from analyze_caption_conditioned_object_extraction_proxy import split_object_list
from extract_chair_object_delta_yesno_features import add_prefix_stats, score_objects
from extract_generative_semantic_pairwise_features import read_prediction_map, write_csv, write_json
from frgavr_cleanroom.runtime import CleanroomLlavaRuntime, load_question_rows, parse_bool, safe_id


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def get_float(row: Dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0
    return value if math.isfinite(value) else 0.0


def ordered_unique(values: Sequence[str], *, max_items: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if int(max_items) > 0 and len(out) >= int(max_items):
            break
    return out


def add_router_features(row: Dict[str, Any]) -> None:
    base_only_count = get_float(row, "capobjyn_base_only_count")
    int_only_count = get_float(row, "capobjyn_int_only_count")
    base_all_count = get_float(row, "capobjyn_base_all_count")
    int_all_count = get_float(row, "capobjyn_int_all_count")

    base_only_yes = get_float(row, "capobjyn_base_only_yes_prob_sum")
    base_only_risk = get_float(row, "capobjyn_base_only_no_risk_sum")
    int_only_yes = get_float(row, "capobjyn_int_only_yes_prob_sum")
    int_only_risk = get_float(row, "capobjyn_int_only_no_risk_sum")
    base_all_yes_mean = get_float(row, "capobjyn_base_all_yes_prob_mean")
    base_all_risk_mean = get_float(row, "capobjyn_base_all_no_risk_mean")
    base_all_risk_sum = get_float(row, "capobjyn_base_all_no_risk_sum")
    int_all_yes_mean = get_float(row, "capobjyn_int_all_yes_prob_mean")
    base_only_yes_min = get_float(row, "capobjyn_base_only_yes_prob_min")
    base_only_margin_mean = get_float(row, "capobjyn_base_only_lp_margin_mean")
    int_only_margin_mean = get_float(row, "capobjyn_int_only_lp_margin_mean")

    verified_base_only = get_float(row, "capobjyn_base_only_yes_prob_gt_050_count")
    verified_int_only = get_float(row, "capobjyn_int_only_yes_prob_gt_050_count")
    risky_base_all = get_float(row, "capobjyn_base_all_yes_prob_lt_040_count")
    risky_base_only = get_float(row, "capobjyn_base_only_yes_prob_lt_040_count")

    rollback_gain = float(base_only_yes + int_only_risk)
    rollback_cost = float(base_only_risk + int_only_yes)
    rollback_cost_with_base_risk = float(rollback_cost + base_all_risk_mean)
    clean_base_factor = max(0.0, 1.0 - base_all_risk_mean)
    low_int_supported_factor = max(0.0, 1.0 - int_only_yes)

    row.update(
        {
            "capobjyn_verified_base_only_count": verified_base_only,
            "capobjyn_verified_int_only_count": verified_int_only,
            "capobjyn_verified_base_minus_int_count": float(verified_base_only - verified_int_only),
            "capobjyn_base_only_verified_rate": safe_ratio(verified_base_only, base_only_count),
            "capobjyn_int_only_verified_rate": safe_ratio(verified_int_only, int_only_count),
            "capobjyn_base_all_risky_count_lt040": risky_base_all,
            "capobjyn_base_all_risky_rate_lt040": safe_ratio(risky_base_all, base_all_count),
            "capobjyn_base_only_risky_count_lt040": risky_base_only,
            "capobjyn_base_only_risky_rate_lt040": safe_ratio(risky_base_only, base_only_count),
            "capobjyn_base_only_support_sum_x_count": float(base_only_yes * base_only_count),
            "capobjyn_base_only_support_min_x_count": float(base_only_yes_min * base_only_count),
            "capobjyn_base_only_margin_minus_int_only_margin": float(base_only_margin_mean - int_only_margin_mean),
            "capobjyn_base_all_support_minus_int_all_support": float(base_all_yes_mean - int_all_yes_mean),
            "capobjyn_rollback_gain": rollback_gain,
            "capobjyn_rollback_cost": rollback_cost,
            "capobjyn_rollback_cost_with_base_risk": rollback_cost_with_base_risk,
            "capobjyn_rollback_gain_minus_cost": float(rollback_gain - rollback_cost),
            "capobjyn_rollback_gain_minus_cost_with_base_risk": float(rollback_gain - rollback_cost_with_base_risk),
            "capobjyn_rollback_gain_cost_ratio_eps010": float(rollback_gain / (rollback_cost + 0.10)),
            "capobjyn_rollback_gain_cost_ratio_eps100": float(rollback_gain / (rollback_cost + 1.00)),
            "capobjyn_rollback_gain_cost_base_risk_ratio_eps010": float(
                rollback_gain / (rollback_cost_with_base_risk + 0.10)
            ),
            "capobjyn_rollback_gain_x_clean_base": float(rollback_gain * clean_base_factor),
            "capobjyn_base_only_support_x_clean_base": float(base_only_yes * clean_base_factor),
            "capobjyn_base_only_support_x_clean_base_x_low_int_support": float(
                base_only_yes * clean_base_factor * low_int_supported_factor
            ),
            "capobjyn_verified_base_only_x_clean_base": float(verified_base_only * clean_base_factor),
            "capobjyn_verified_base_only_x_low_base_risk": float(verified_base_only * max(0.0, 1.0 - base_all_risk_sum)),
            "capobjyn_has_base_only_no_int_only": float(1.0 if base_only_count > 0.0 and int_only_count <= 0.0 else 0.0),
            "capobjyn_has_verified_base_only_no_verified_int_only": float(
                1.0 if verified_base_only > 0.0 and verified_int_only <= 0.0 else 0.0
            ),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Image-grounded yes/no verifier over caption-extracted baseline/intervention object-list deltas."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--baseline_object_pred_jsonl", required=True)
    ap.add_argument("--intervention_object_pred_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_objects", type=int, default=12)
    ap.add_argument("--question_template", default="Is there a {object} in the image? Answer yes or no.")
    ap.add_argument("--yes_text", default="Yes")
    ap.add_argument("--no_text", default="No")
    ap.add_argument("--score_mode", choices=["yesno", "yes_only"], default="yesno")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    questions = load_question_rows(os.path.abspath(args.question_file), limit=int(args.limit))
    baseline_objects = read_prediction_map(os.path.abspath(args.baseline_object_pred_jsonl))
    intervention_objects = read_prediction_map(os.path.abspath(args.intervention_object_pred_jsonl))
    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=(str(args.model_base) or None),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_object_probes = 0
    for idx, sample in enumerate(questions):
        sid = safe_id(sample.get("question_id", sample.get("id", sample.get("image_id"))))
        try:
            sid = str(int(sid))
        except Exception:
            pass
        image_name = str(sample.get("image", "")).strip()
        row: Dict[str, Any] = {
            "id": sid,
            "image_id": sid,
            "image": image_name,
            "capobjyn_error": "",
        }
        try:
            if not sid or sid not in baseline_objects or sid not in intervention_objects:
                raise ValueError("Missing baseline/intervention object prediction row.")
            image_path = os.path.join(str(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            base_objs = ordered_unique(
                split_object_list(str(baseline_objects[sid].get("text", ""))),
                max_items=int(args.max_objects),
            )
            int_objs = ordered_unique(
                split_object_list(str(intervention_objects[sid].get("text", ""))),
                max_items=int(args.max_objects),
            )
            base_set = set(base_objs)
            int_set = set(int_objs)
            shared_objs = ordered_unique([obj for obj in base_objs if obj in int_set], max_items=int(args.max_objects))
            base_only_objs = ordered_unique([obj for obj in base_objs if obj not in int_set], max_items=int(args.max_objects))
            int_only_objs = ordered_unique([obj for obj in int_objs if obj not in base_set], max_items=int(args.max_objects))

            image = runtime.load_image(image_path)
            cache: Dict[str, Dict[str, float]] = {}
            base_only_scored = score_objects(
                base_only_objs,
                runtime=runtime,
                image=image,
                question_template=str(args.question_template),
                yes_text=str(args.yes_text),
                no_text=str(args.no_text),
                score_mode=str(args.score_mode),
                cache=cache,
            )
            int_only_scored = score_objects(
                int_only_objs,
                runtime=runtime,
                image=image,
                question_template=str(args.question_template),
                yes_text=str(args.yes_text),
                no_text=str(args.no_text),
                score_mode=str(args.score_mode),
                cache=cache,
            )
            base_all_scored = score_objects(
                base_objs,
                runtime=runtime,
                image=image,
                question_template=str(args.question_template),
                yes_text=str(args.yes_text),
                no_text=str(args.no_text),
                score_mode=str(args.score_mode),
                cache=cache,
            )
            int_all_scored = score_objects(
                int_objs,
                runtime=runtime,
                image=image,
                question_template=str(args.question_template),
                yes_text=str(args.yes_text),
                no_text=str(args.no_text),
                score_mode=str(args.score_mode),
                cache=cache,
            )
            shared_scored = score_objects(
                shared_objs,
                runtime=runtime,
                image=image,
                question_template=str(args.question_template),
                yes_text=str(args.yes_text),
                no_text=str(args.no_text),
                score_mode=str(args.score_mode),
                cache=cache,
            )
            n_object_probes += len(set(base_only_objs + int_only_objs + base_objs + int_objs + shared_objs))

            union = base_set | int_set
            row.update(
                {
                    "capobjyn_base_object_count": int(len(base_objs)),
                    "capobjyn_int_object_count": int(len(int_objs)),
                    "capobjyn_shared_object_count": int(len(shared_objs)),
                    "capobjyn_union_object_count": int(len(union)),
                    "capobjyn_base_only_object_count": int(len(base_only_objs)),
                    "capobjyn_int_only_object_count": int(len(int_only_objs)),
                    "capobjyn_base_minus_int_object_count": int(len(base_objs) - len(int_objs)),
                    "capobjyn_object_jaccard": safe_ratio(float(len(base_set & int_set)), float(len(union))),
                    "capobjyn_base_only_x_jaccard_gap": float(
                        len(base_only_objs) * (1.0 - safe_ratio(float(len(base_set & int_set)), float(len(union))))
                    ),
                    "capobjyn_base_object_names": " | ".join(base_objs),
                    "capobjyn_int_object_names": " | ".join(int_objs),
                    "capobjyn_base_only_object_names": " | ".join(base_only_objs),
                    "capobjyn_int_only_object_names": " | ".join(int_only_objs),
                    "capobjyn_shared_object_names": " | ".join(shared_objs),
                }
            )
            add_prefix_stats(row, "capobjyn_base_only", base_only_scored)
            add_prefix_stats(row, "capobjyn_int_only", int_only_scored)
            add_prefix_stats(row, "capobjyn_base_all", base_all_scored)
            add_prefix_stats(row, "capobjyn_int_all", int_all_scored)
            add_prefix_stats(row, "capobjyn_shared", shared_scored)
            row["capobjyn_base_only_details_json"] = json.dumps(base_only_scored, ensure_ascii=False, sort_keys=True)
            row["capobjyn_int_only_details_json"] = json.dumps(int_only_scored, ensure_ascii=False, sort_keys=True)
            add_router_features(row)
        except Exception as exc:
            n_errors += 1
            row["capobjyn_error"] = str(exc)
            add_prefix_stats(row, "capobjyn_base_only", [])
            add_prefix_stats(row, "capobjyn_int_only", [])
            add_prefix_stats(row, "capobjyn_base_all", [])
            add_prefix_stats(row, "capobjyn_int_all", [])
            add_prefix_stats(row, "capobjyn_shared", [])
            add_router_features(row)
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[caption-object-yesno] {idx + 1}/{len(questions)} unique_object_probes~={n_object_probes}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")
    if str(args.out_summary_json or "").strip():
        feature_keys = [key for key in rows[0].keys() if key.startswith("capobjyn_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "baseline_object_pred_jsonl": os.path.abspath(args.baseline_object_pred_jsonl),
                    "intervention_object_pred_jsonl": os.path.abspath(args.intervention_object_pred_jsonl),
                    "model_path": str(args.model_path),
                    "model_base": str(args.model_base),
                    "conv_mode": str(args.conv_mode),
                    "device": str(args.device),
                    "question_template": str(args.question_template),
                    "score_mode": str(args.score_mode),
                    "limit": int(args.limit),
                    "max_objects": int(args.max_objects),
                },
                "counts": {
                    "n_rows": int(len(rows)),
                    "n_errors": int(n_errors),
                    "n_unique_object_probes_est": int(n_object_probes),
                    "n_forward_passes_est": int(
                        n_object_probes * (1 if str(args.score_mode) == "yes_only" else 2)
                    ),
                    "n_features": int(len(feature_keys)),
                },
                "feature_keys": feature_keys,
                "outputs": {"out_csv": os.path.abspath(args.out_csv)},
            },
        )


if __name__ == "__main__":
    main()
