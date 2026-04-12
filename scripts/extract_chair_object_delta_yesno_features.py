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


def yesno_metrics(
    *,
    runtime: CleanroomLlavaRuntime,
    image: Any,
    object_name: str,
    question_template: str,
    yes_text: str,
    no_text: str,
    score_mode: str,
    cache: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    question = str(question_template).replace("{object}", str(object_name))
    cache_key = f"{score_mode}\t{question}\t{yes_text}\t{no_text}"
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    try:
        yes = replay_claim_metrics(runtime, image=image, question=question, claim_text=str(yes_text))
    except Exception:
        yes = zero_replay_metrics()

    if str(score_mode) == "yes_only":
        no = zero_replay_metrics()
    else:
        try:
            no = replay_claim_metrics(runtime, image=image, question=question, claim_text=str(no_text))
        except Exception:
            no = zero_replay_metrics()

    yes_lp = float(yes.get("replay_lp_mean", 0.0))
    no_lp = float(no.get("replay_lp_mean", 0.0))
    yes_gap = float(yes.get("replay_gap_mean", 0.0))
    no_gap = float(no.get("replay_gap_mean", 0.0))
    yes_argmax = float(yes.get("replay_argmax_mean", 0.0))
    no_argmax = float(no.get("replay_argmax_mean", 0.0))
    margin = float(yes_lp - no_lp)
    gap_margin = float(yes_gap - no_gap)
    argmax_margin = float(yes_argmax - no_argmax)
    prob = sigmoid(margin)
    out = {
        "yesno_yes_lp": yes_lp,
        "yesno_no_lp": no_lp,
        "yesno_lp_margin": margin,
        "yesno_yes_gap": yes_gap,
        "yesno_no_gap": no_gap,
        "yesno_gap_margin": gap_margin,
        "yesno_yes_argmax": yes_argmax,
        "yesno_no_argmax": no_argmax,
        "yesno_argmax_margin": argmax_margin,
        "yesno_prob": float(prob),
        "yesno_risk": float(1.0 - prob),
        "yesno_error": float(max(float(yes.get("replay_error", 0.0)), float(no.get("replay_error", 0.0)))),
    }
    cache[cache_key] = dict(out)
    return out


def score_objects(
    objects: Sequence[str],
    *,
    runtime: CleanroomLlavaRuntime,
    image: Any,
    question_template: str,
    yes_text: str,
    no_text: str,
    score_mode: str,
    cache: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for obj in objects:
        metrics = yesno_metrics(
            runtime=runtime,
            image=image,
            object_name=str(obj),
            question_template=question_template,
            yes_text=yes_text,
            no_text=no_text,
            score_mode=score_mode,
            cache=cache,
        )
        rows.append({"object": str(obj), **metrics})
    return rows


def add_prefix_stats(out: Dict[str, Any], prefix: str, items: Sequence[Dict[str, Any]]) -> None:
    yes_lps = [float(item.get("yesno_yes_lp", 0.0)) for item in items]
    no_lps = [float(item.get("yesno_no_lp", 0.0)) for item in items]
    probs = [float(item.get("yesno_prob", 0.0)) for item in items]
    risks = [float(item.get("yesno_risk", 0.0)) for item in items]
    lp_margins = [float(item.get("yesno_lp_margin", 0.0)) for item in items]
    gap_margins = [float(item.get("yesno_gap_margin", 0.0)) for item in items]
    argmax_margins = [float(item.get("yesno_argmax_margin", 0.0)) for item in items]
    out.update(
        {
            f"{prefix}_count": int(len(items)),
            f"{prefix}_yes_lp_sum": sum_vals(yes_lps),
            f"{prefix}_yes_lp_mean": mean(yes_lps),
            f"{prefix}_yes_lp_min": min_or_zero(yes_lps),
            f"{prefix}_yes_lp_max": max_or_zero(yes_lps),
            f"{prefix}_no_lp_sum": sum_vals(no_lps),
            f"{prefix}_no_lp_mean": mean(no_lps),
            f"{prefix}_no_lp_min": min_or_zero(no_lps),
            f"{prefix}_no_lp_max": max_or_zero(no_lps),
            f"{prefix}_yes_prob_sum": sum_vals(probs),
            f"{prefix}_yes_prob_mean": mean(probs),
            f"{prefix}_yes_prob_min": min_or_zero(probs),
            f"{prefix}_yes_prob_max": max_or_zero(probs),
            f"{prefix}_no_risk_sum": sum_vals(risks),
            f"{prefix}_no_risk_mean": mean(risks),
            f"{prefix}_no_risk_max": max_or_zero(risks),
            f"{prefix}_lp_margin_sum": sum_vals(lp_margins),
            f"{prefix}_lp_margin_mean": mean(lp_margins),
            f"{prefix}_lp_margin_min": min_or_zero(lp_margins),
            f"{prefix}_lp_margin_max": max_or_zero(lp_margins),
            f"{prefix}_gap_margin_mean": mean(gap_margins),
            f"{prefix}_argmax_margin_mean": mean(argmax_margins),
            f"{prefix}_yes_prob_gt_050_count": int(sum(1 for x in probs if x > 0.50)),
            f"{prefix}_yes_prob_gt_060_count": int(sum(1 for x in probs if x > 0.60)),
            f"{prefix}_yes_prob_gt_070_count": int(sum(1 for x in probs if x > 0.70)),
            f"{prefix}_yes_prob_lt_040_count": int(sum(1 for x in probs if x < 0.40)),
            f"{prefix}_yes_prob_lt_030_count": int(sum(1 for x in probs if x < 0.30)),
        }
    )
    n = max(1, int(len(items)))
    out[f"{prefix}_yes_precision_gt_050"] = float(out[f"{prefix}_yes_prob_gt_050_count"] / n)
    out[f"{prefix}_yes_precision_gt_060"] = float(out[f"{prefix}_yes_prob_gt_060_count"] / n)
    out[f"{prefix}_yes_precision_gt_070"] = float(out[f"{prefix}_yes_prob_gt_070_count"] / n)


def add_competition_features(out: Dict[str, Any]) -> None:
    def get_float(key: str) -> float:
        try:
            return float(out.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    base_yes_sum = get_float("pair_chairyn_base_only_yes_prob_sum")
    int_yes_sum = get_float("pair_chairyn_int_only_yes_prob_sum")
    base_yes_mean = get_float("pair_chairyn_base_only_yes_prob_mean")
    int_yes_mean = get_float("pair_chairyn_int_only_yes_prob_mean")
    base_yes_lp_sum = get_float("pair_chairyn_base_only_yes_lp_sum")
    int_yes_lp_sum = get_float("pair_chairyn_int_only_yes_lp_sum")
    base_yes_lp_mean = get_float("pair_chairyn_base_only_yes_lp_mean")
    int_yes_lp_mean = get_float("pair_chairyn_int_only_yes_lp_mean")
    base_yes_min = get_float("pair_chairyn_base_only_yes_prob_min")
    int_yes_max = get_float("pair_chairyn_int_only_yes_prob_max")
    base_lp_mean = get_float("pair_chairyn_base_only_lp_margin_mean")
    int_lp_mean = get_float("pair_chairyn_int_only_lp_margin_mean")
    base_lp_min = get_float("pair_chairyn_base_only_lp_margin_min")
    int_lp_max = get_float("pair_chairyn_int_only_lp_margin_max")
    int_count = get_float("pair_chairyn_int_only_object_count")
    base_no_risk = get_float("pair_chairyn_base_only_no_risk_sum")
    int_no_risk = get_float("pair_chairyn_int_only_no_risk_sum")
    gain_cost_ratio = get_float("pair_chairyn_rollback_gain_cost_ratio_eps_010")
    net_gain_cost_ratio = get_float("pair_chairyn_rollback_net_gain_cost_ratio_eps_010")
    yes_mean_margin = float(base_yes_mean - int_yes_mean)
    yes_lp_mean_margin = float(base_yes_lp_mean - int_yes_lp_mean)
    lp_mean_margin = float(base_lp_mean - int_lp_mean)
    no_competition = float(1.0 if int_count <= 0.0 else 0.0)

    out.update(
        {
            "pair_chairyn_comp_yes_sum_margin": float(base_yes_sum - int_yes_sum),
            "pair_chairyn_comp_yes_mean_margin": yes_mean_margin,
            "pair_chairyn_comp_yes_lp_sum_margin": float(base_yes_lp_sum - int_yes_lp_sum),
            "pair_chairyn_comp_yes_lp_mean_margin": yes_lp_mean_margin,
            "pair_chairyn_comp_yes_min_vs_int_max_margin": float(base_yes_min - int_yes_max),
            "pair_chairyn_comp_lp_mean_margin": lp_mean_margin,
            "pair_chairyn_comp_lp_min_vs_int_max_margin": float(base_lp_min - int_lp_max),
            "pair_chairyn_comp_base_advantage_yes_sum": float(max(0.0, base_yes_sum - int_yes_sum)),
            "pair_chairyn_comp_base_advantage_yes_mean": float(max(0.0, yes_mean_margin)),
            "pair_chairyn_comp_base_advantage_yes_lp_sum": float(max(0.0, base_yes_lp_sum - int_yes_lp_sum)),
            "pair_chairyn_comp_base_advantage_yes_lp_mean": float(max(0.0, yes_lp_mean_margin)),
            "pair_chairyn_comp_base_advantage_lp_mean": float(max(0.0, lp_mean_margin)),
            "pair_chairyn_comp_int_competition_count": float(int_count),
            "pair_chairyn_comp_base_only_no_competition": no_competition,
            "pair_chairyn_comp_base_yes_sum_x_no_competition": float(base_yes_sum * no_competition),
            "pair_chairyn_comp_base_yes_sum_x_comp_margin": float(base_yes_sum * max(0.0, yes_mean_margin)),
            "pair_chairyn_comp_gain_cost_ratio_no_competition": float(gain_cost_ratio * no_competition),
            "pair_chairyn_comp_gain_cost_ratio_x_comp_margin": float(gain_cost_ratio * max(0.0, yes_mean_margin)),
            "pair_chairyn_comp_net_gain_cost_ratio_x_comp_margin": float(
                net_gain_cost_ratio * max(0.0, yes_mean_margin)
            ),
            "pair_chairyn_comp_safe_support_score": float(
                base_yes_sum * max(0.0, 1.0 - base_no_risk) * max(0.0, 1.0 - int_yes_sum)
            ),
            "pair_chairyn_comp_safe_net_score": float(
                (base_yes_sum + int_no_risk) * max(0.0, 1.0 - base_no_risk) * max(0.0, 1.0 - int_yes_sum)
            ),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract CHAIR-vocabulary object-delta yes/no visual support features without using GT labels."
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
    ap.add_argument("--question_template", default="Is there a {object} in the image? Answer yes or no.")
    ap.add_argument("--yes_text", default="Yes")
    ap.add_argument("--no_text", default="No")
    ap.add_argument(
        "--score_mode",
        choices=["yesno", "yes_only"],
        default="yesno",
        help="yes_only skips the No replay pass and exposes one-sided yes-likelihood competition features.",
    )
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
    n_object_probes = 0
    for idx, sample in enumerate(question_rows):
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
            "pair_chairyn_error": "",
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
            n_object_probes += int(len(base_only_scored) + len(int_only_scored))

            row.update(
                {
                    "pair_chairyn_base_object_count": int(len(base_objs)),
                    "pair_chairyn_int_object_count": int(len(int_objs)),
                    "pair_chairyn_shared_object_count": int(len(shared_objs)),
                    "pair_chairyn_base_only_object_count": int(len(base_only_objs)),
                    "pair_chairyn_int_only_object_count": int(len(int_only_objs)),
                    "pair_chairyn_base_only_object_names": " | ".join(base_only_objs),
                    "pair_chairyn_int_only_object_names": " | ".join(int_only_objs),
                    "pair_chairyn_shared_object_names": " | ".join(shared_objs),
                    "pair_chairyn_object_drop_rate": float(len(base_only_objs) / max(1, len(base_objs))),
                    "pair_chairyn_object_add_rate": float(len(int_only_objs) / max(1, len(int_objs))),
                }
            )
            add_prefix_stats(row, "pair_chairyn_base_only", base_only_scored)
            add_prefix_stats(row, "pair_chairyn_int_only", int_only_scored)

            gain = float(row["pair_chairyn_base_only_yes_prob_sum"])
            cost = float(row["pair_chairyn_base_only_no_risk_sum"])
            int_supported_loss = float(row["pair_chairyn_int_only_yes_prob_sum"])
            int_unsupported_removed = float(row["pair_chairyn_int_only_no_risk_sum"])
            net_gain = float(gain + int_unsupported_removed)
            net_cost = float(cost + int_supported_loss)
            row.update(
                {
                    "pair_chairyn_rollback_gain_minus_cost": float(gain - cost),
                    "pair_chairyn_rollback_gain_cost_ratio_eps_010": float(gain / (cost + 0.10)),
                    "pair_chairyn_rollback_gain_cost_ratio_eps_100": float(gain / (cost + 1.00)),
                    "pair_chairyn_rollback_gain_x_low_cost": float(gain * max(0.0, 1.0 - cost)),
                    "pair_chairyn_rollback_net_gain": net_gain,
                    "pair_chairyn_rollback_net_cost": net_cost,
                    "pair_chairyn_rollback_net_gain_minus_cost": float(net_gain - net_cost),
                    "pair_chairyn_rollback_net_gain_cost_ratio_eps_010": float(net_gain / (net_cost + 0.10)),
                    "pair_chairyn_rollback_net_gain_cost_ratio_eps_100": float(net_gain / (net_cost + 1.00)),
                    "pair_chairyn_rollback_net_gain_x_low_cost": float(net_gain * max(0.0, 1.0 - net_cost)),
                }
            )
            add_competition_features(row)
        except Exception as exc:
            n_errors += 1
            row["pair_chairyn_error"] = str(exc)
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[chair-object-yesno] {idx + 1}/{len(question_rows)} object_probes={n_object_probes}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")
    if str(args.out_summary_json or "").strip():
        feature_keys = [key for key in rows[0].keys() if key.startswith("pair_chairyn_")] if rows else []
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
                    "question_template": str(args.question_template),
                    "yes_text": str(args.yes_text),
                    "no_text": str(args.no_text),
                    "score_mode": str(args.score_mode),
                    "limit": int(args.limit),
                },
                "counts": {
                    "n_rows": int(len(rows)),
                    "n_errors": int(n_errors),
                    "n_object_probes": int(n_object_probes),
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
