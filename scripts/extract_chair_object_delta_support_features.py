#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence

from extract_generative_claim_support_delta_features import replay_claim_metrics, zero_replay_metrics
from frgavr_cleanroom.runtime import CleanroomLlavaRuntime, load_question_rows, parse_bool, safe_id, write_csv, write_json


def canonical_object(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[-1]
    return str(value if value is not None else "").strip().lower()


def object_set(values: Iterable[Any]) -> set[str]:
    return {obj for obj in (canonical_object(value) for value in values) if obj}


def load_chair_sentence_map(path: str) -> Dict[str, Dict[str, Any]]:
    payload = json.load(open(path, "r", encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("sentences", []):
        sid = safe_id(row.get("image_id"))
        if sid:
            try:
                sid = str(int(sid))
            except Exception:
                pass
            out[sid] = dict(row)
    return out


def sigmoid(value: float) -> float:
    x = max(-50.0, min(50.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-x)))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(x) for x in values) / float(len(values)))


def sum_vals(values: Sequence[float]) -> float:
    return float(sum(float(x) for x in values))


def min_or_zero(values: Sequence[float]) -> float:
    return float(min(values)) if values else 0.0


def max_or_zero(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def score_objects(
    objects: Sequence[str],
    *,
    runtime: CleanroomLlavaRuntime,
    image: Any,
    question: str,
    claim_template: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in objects:
        claim_text = str(claim_template).replace("{object}", str(obj))
        try:
            metrics = replay_claim_metrics(runtime, image=image, question=question, claim_text=claim_text)
        except Exception:
            metrics = zero_replay_metrics()
        gap_mean = float(metrics.get("replay_gap_mean", 0.0))
        gap_min = float(metrics.get("replay_gap_min", 0.0))
        argmax = float(metrics.get("replay_argmax_mean", 0.0))
        support_prob = mean([sigmoid(gap_mean), sigmoid(gap_min), argmax])
        rows.append(
            {
                "object": str(obj),
                "claim_text": claim_text,
                "support_prob": float(support_prob),
                "unsupported_prob": float(1.0 - support_prob),
                **{f"support_{key}": value for key, value in metrics.items()},
            }
        )
    return rows


def add_prefix_stats(out: Dict[str, Any], prefix: str, items: Sequence[Dict[str, Any]]) -> None:
    probs = [float(item.get("support_prob", 0.0)) for item in items]
    risks = [float(item.get("unsupported_prob", 0.0)) for item in items]
    gap_means = [float(item.get("support_replay_gap_mean", 0.0)) for item in items]
    gap_mins = [float(item.get("support_replay_gap_min", 0.0)) for item in items]
    lp_means = [float(item.get("support_replay_lp_mean", 0.0)) for item in items]
    argmax = [float(item.get("support_replay_argmax_mean", 0.0)) for item in items]
    out.update(
        {
            f"{prefix}_count": int(len(items)),
            f"{prefix}_support_prob_sum": sum_vals(probs),
            f"{prefix}_support_prob_mean": mean(probs),
            f"{prefix}_support_prob_max": max_or_zero(probs),
            f"{prefix}_support_prob_min": min_or_zero(probs),
            f"{prefix}_unsupported_prob_sum": sum_vals(risks),
            f"{prefix}_unsupported_prob_mean": mean(risks),
            f"{prefix}_unsupported_prob_max": max_or_zero(risks),
            f"{prefix}_gap_mean_sum": sum_vals(gap_means),
            f"{prefix}_gap_mean_mean": mean(gap_means),
            f"{prefix}_gap_mean_max": max_or_zero(gap_means),
            f"{prefix}_gap_min_min": min_or_zero(gap_mins),
            f"{prefix}_lp_mean_mean": mean(lp_means),
            f"{prefix}_argmax_mean": mean(argmax),
            f"{prefix}_support_prob_gt_050_count": int(sum(1 for x in probs if x > 0.50)),
            f"{prefix}_support_prob_gt_060_count": int(sum(1 for x in probs if x > 0.60)),
            f"{prefix}_support_prob_gt_070_count": int(sum(1 for x in probs if x > 0.70)),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract CHAIR-vocabulary generated-object delta support features without using GT labels."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--baseline_chair_json", required=True)
    ap.add_argument("--intervention_chair_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--claim_template", default="{object}")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    base_map = load_chair_sentence_map(args.baseline_chair_json)
    int_map = load_chair_sentence_map(args.intervention_chair_json)
    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(args.model_base or None),
        conv_mode=args.conv_mode,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    for idx, sample in enumerate(question_rows):
        sid = safe_id(sample.get("question_id", sample.get("id", sample.get("image_id"))))
        try:
            sid = str(int(sid))
        except Exception:
            pass
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        row: Dict[str, Any] = {
            "id": sid,
            "image_id": sid,
            "image": image_name,
            "question": question,
            "pair_chairobj_error": "",
        }
        try:
            if not sid:
                raise ValueError("Missing sample id.")
            if sid not in base_map or sid not in int_map:
                raise ValueError("Missing CHAIR sentence row.")
            image_path = os.path.join(args.image_folder, image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = runtime.load_image(image_path)

            base_objs = sorted(object_set(base_map[sid].get("mscoco_generated_words", [])))
            int_objs = sorted(object_set(int_map[sid].get("mscoco_generated_words", [])))
            shared_objs = sorted(set(base_objs) & set(int_objs))
            base_only_objs = sorted(set(base_objs) - set(int_objs))
            int_only_objs = sorted(set(int_objs) - set(base_objs))

            base_only_scored = score_objects(
                base_only_objs,
                runtime=runtime,
                image=image,
                question=question,
                claim_template=str(args.claim_template),
            )
            int_only_scored = score_objects(
                int_only_objs,
                runtime=runtime,
                image=image,
                question=question,
                claim_template=str(args.claim_template),
            )

            row.update(
                {
                    "pair_chairobj_base_object_count": int(len(base_objs)),
                    "pair_chairobj_int_object_count": int(len(int_objs)),
                    "pair_chairobj_shared_object_count": int(len(shared_objs)),
                    "pair_chairobj_base_only_object_count": int(len(base_only_objs)),
                    "pair_chairobj_int_only_object_count": int(len(int_only_objs)),
                    "pair_chairobj_base_only_object_names": " | ".join(base_only_objs),
                    "pair_chairobj_int_only_object_names": " | ".join(int_only_objs),
                    "pair_chairobj_shared_object_names": " | ".join(shared_objs),
                    "pair_chairobj_object_drop_rate": float(len(base_only_objs) / max(1, len(base_objs))),
                    "pair_chairobj_object_add_rate": float(len(int_only_objs) / max(1, len(int_objs))),
                }
            )
            add_prefix_stats(row, "pair_chairobj_base_only", base_only_scored)
            add_prefix_stats(row, "pair_chairobj_int_only", int_only_scored)
            gain = float(row["pair_chairobj_base_only_support_prob_sum"])
            cost = float(row["pair_chairobj_base_only_unsupported_prob_sum"])
            int_supported_loss = float(row["pair_chairobj_int_only_support_prob_sum"])
            int_unsupported_removed = float(row["pair_chairobj_int_only_unsupported_prob_sum"])
            net_gain = float(gain + int_unsupported_removed)
            net_cost = float(cost + int_supported_loss)
            row.update(
                {
                    "pair_chairobj_rollback_gain_minus_cost": float(gain - cost),
                    "pair_chairobj_rollback_gain_cost_ratio_eps_010": float(gain / (cost + 0.10)),
                    "pair_chairobj_rollback_gain_cost_ratio_eps_100": float(gain / (cost + 1.00)),
                    "pair_chairobj_rollback_gain_x_low_cost": float(gain * max(0.0, 1.0 - cost)),
                    "pair_chairobj_rollback_net_gain": net_gain,
                    "pair_chairobj_rollback_net_cost": net_cost,
                    "pair_chairobj_rollback_net_gain_minus_cost": float(net_gain - net_cost),
                    "pair_chairobj_rollback_net_gain_cost_ratio_eps_010": float(net_gain / (net_cost + 0.10)),
                    "pair_chairobj_rollback_net_gain_cost_ratio_eps_100": float(net_gain / (net_cost + 1.00)),
                    "pair_chairobj_rollback_net_gain_x_low_cost": float(net_gain * max(0.0, 1.0 - net_cost)),
                }
            )
        except Exception as exc:
            n_errors += 1
            row["pair_chairobj_error"] = str(exc)
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[chair-object-delta] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")
    if str(args.out_summary_json or "").strip():
        feature_keys = [key for key in rows[0].keys() if key.startswith("pair_chairobj_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
                    "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
                    "model_path": str(args.model_path),
                    "model_base": str(args.model_base),
                    "conv_mode": str(args.conv_mode),
                    "device": str(args.device),
                    "claim_template": str(args.claim_template),
                    "limit": int(args.limit),
                },
                "counts": {
                    "n_rows": int(len(rows)),
                    "n_errors": int(n_errors),
                    "n_features": int(len(feature_keys)),
                },
                "feature_keys": feature_keys,
                "outputs": {"out_csv": os.path.abspath(args.out_csv)},
            },
        )


if __name__ == "__main__":
    main()
