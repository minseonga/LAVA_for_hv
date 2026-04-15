#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

from analyze_caption_conditioned_object_extraction_proxy import (
    object_matches,
    split_object_list,
    split_pipe_objects,
)
from extract_chair_object_delta_yesno_features import add_prefix_stats, score_objects
from extract_generative_semantic_pairwise_features import read_prediction_map, write_csv, write_json
from extract_intervention_object_inventory_yesno_features import extract_mentioned_objects, parse_object_vocab
from frgavr_cleanroom.runtime import CleanroomLlavaRuntime, load_question_rows, parse_bool, safe_id


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def norm_id(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


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


def normalize_objects_to_vocab(objects: Sequence[str], vocab: Sequence[str], *, max_items: int) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for obj in objects:
        for item in extract_mentioned_objects(str(obj), vocab):
            if item in seen:
                continue
            seen.add(item)
            normalized.append(item)
            if int(max_items) > 0 and len(normalized) >= int(max_items):
                return normalized
    return normalized


def oracle_topk_hit(candidates: Sequence[str], gold: Sequence[str], k: int) -> int:
    top = list(candidates)[: max(0, int(k))]
    return int(any(object_matches(g, c) for g in gold for c in top))


def add_risk_features(row: Dict[str, Any], scored: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(scored, key=lambda item: safe_float(item.get("yesno_prob"), 1.0))
    probs = [safe_float(item.get("yesno_prob"), 0.0) for item in ranked]
    risks = [safe_float(item.get("yesno_risk"), 0.0) for item in ranked]
    margins = [safe_float(item.get("yesno_lp_margin"), 0.0) for item in ranked]

    top = ranked[0] if ranked else {}
    second = ranked[1] if len(ranked) > 1 else {}
    top_prob = safe_float(top.get("yesno_prob"), 1.0) if top else 1.0
    second_prob = safe_float(second.get("yesno_prob"), 1.0) if second else 1.0
    top_margin = safe_float(top.get("yesno_lp_margin"), 0.0) if top else 0.0
    second_margin = safe_float(second.get("yesno_lp_margin"), 0.0) if second else 0.0

    row.update(
        {
            "risk_object_count": int(len(ranked)),
            "risk_top_object": str(top.get("object", "")) if top else "",
            "risk_top_yes_prob": float(top_prob),
            "risk_top_no_risk": float(1.0 - top_prob),
            "risk_top_lp_margin": float(top_margin),
            "risk_top_yes_lp": safe_float(top.get("yesno_yes_lp"), 0.0) if top else 0.0,
            "risk_top_no_lp": safe_float(top.get("yesno_no_lp"), 0.0) if top else 0.0,
            "risk_second_object": str(second.get("object", "")) if second else "",
            "risk_second_yes_prob": float(second_prob),
            "risk_second_no_risk": float(1.0 - second_prob),
            "risk_second_lp_margin": float(second_margin),
            "risk_second_minus_top_yes_prob": float(second_prob - top_prob),
            "risk_top_minus_second_lp_margin": float(top_margin - second_margin),
            "risk_min_yes_prob": float(min(probs)) if probs else 1.0,
            "risk_mean_yes_prob": float(sum(probs) / len(probs)) if probs else 1.0,
            "risk_max_no_risk": float(max(risks)) if risks else 0.0,
            "risk_mean_no_risk": float(sum(risks) / len(risks)) if risks else 0.0,
            "risk_min_lp_margin": float(min(margins)) if margins else 0.0,
            "risk_mean_lp_margin": float(sum(margins) / len(margins)) if margins else 0.0,
            "risk_yes_prob_lt_030_count": int(sum(1 for x in probs if x < 0.30)),
            "risk_yes_prob_lt_040_count": int(sum(1 for x in probs if x < 0.40)),
            "risk_yes_prob_lt_050_count": int(sum(1 for x in probs if x < 0.50)),
            "risk_lp_margin_lt_000_count": int(sum(1 for x in margins if x < 0.0)),
            "risk_ranked_objects": " | ".join(str(item.get("object", "")) for item in ranked),
        }
    )
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score intervention-caption objects with image-grounded yes/no and select top hallucination-risk object."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_object_pred_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--oracle_rows_csv", default="")
    ap.add_argument("--oracle_object_col", default="int_hallucinated_unique")
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--object_vocab", default="coco80")
    ap.add_argument("--filter_to_vocab", type=parse_bool, default=False)
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
    int_objects_pred = read_prediction_map(os.path.abspath(args.intervention_object_pred_jsonl))
    object_vocab = parse_object_vocab(str(args.object_vocab))
    oracle_by_id: Dict[str, Dict[str, str]] = {}
    if str(args.oracle_rows_csv or "").strip() and os.path.isfile(os.path.abspath(args.oracle_rows_csv)):
        oracle_by_id = {norm_id(row.get("id") or row.get("image_id") or row.get("question_id")): row for row in read_csv_rows(args.oracle_rows_csv)}

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
        sid = norm_id(sample.get("question_id", sample.get("id", sample.get("image_id"))))
        image_name = str(sample.get("image", "")).strip()
        row: Dict[str, Any] = {
            "id": sid,
            "image_id": sid,
            "image": image_name,
            "risk_error": "",
        }
        try:
            if not sid or sid not in int_objects_pred:
                raise ValueError("Missing intervention object prediction row.")
            image_path = os.path.join(str(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            raw_objects = ordered_unique(
                split_object_list(str(int_objects_pred[sid].get("text", ""))),
                max_items=int(args.max_objects),
            )
            objects = (
                normalize_objects_to_vocab(raw_objects, object_vocab, max_items=int(args.max_objects))
                if bool(args.filter_to_vocab)
                else list(raw_objects)
            )
            image = runtime.load_image(image_path)
            scored = score_objects(
                objects,
                runtime=runtime,
                image=image,
                question_template=str(args.question_template),
                yes_text=str(args.yes_text),
                no_text=str(args.no_text),
                score_mode=str(args.score_mode),
                cache={},
            )
            n_object_probes += len(objects)
            row.update(
                {
                    "int_raw_object_names": " | ".join(raw_objects),
                    "int_raw_object_count": int(len(raw_objects)),
                    "int_object_names": " | ".join(objects),
                    "int_object_count": int(len(objects)),
                    "risk_filter_to_vocab": int(bool(args.filter_to_vocab)),
                }
            )
            add_prefix_stats(row, "risk_all", scored)
            ranked = add_risk_features(row, scored)
            row["risk_details_json"] = json.dumps(ranked, ensure_ascii=False, sort_keys=True)

            oracle = oracle_by_id.get(sid, {})
            gold = split_pipe_objects(oracle.get(str(args.oracle_object_col), "")) if oracle else []
            ranked_names = [str(item.get("object", "")) for item in ranked]
            row.update(
                {
                    "oracle_object_col": str(args.oracle_object_col),
                    "oracle_hallucinated_objects": " | ".join(gold),
                    "oracle_hallucinated_object_count": int(len(gold)),
                    "oracle_any_hallucinated_object": int(bool(gold)),
                    "oracle_risk_top1_hit": oracle_topk_hit(ranked_names, gold, 1),
                    "oracle_risk_top2_hit": oracle_topk_hit(ranked_names, gold, 2),
                    "oracle_risk_top3_hit": oracle_topk_hit(ranked_names, gold, 3),
                    str(args.target_col): int(str(oracle.get(str(args.target_col), "")).strip().lower() in {"1", "true", "yes"}),
                    "audit_int_ci_unique": safe_float(oracle.get("int_ci_unique"), 0.0) if oracle else 0.0,
                    "audit_int_chair_s": safe_float(oracle.get("int_chair_s"), 0.0) if oracle else 0.0,
                }
            )
        except Exception as exc:
            n_errors += 1
            row["risk_error"] = str(exc)
            row["int_object_names"] = ""
            row["int_object_count"] = 0
            add_prefix_stats(row, "risk_all", [])
            add_risk_features(row, [])
            row["risk_details_json"] = "[]"
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[intervention-object-risk] {idx + 1}/{len(questions)} object_probes={n_object_probes}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")
    if str(args.out_summary_json or "").strip():
        selected_like = [row for row in rows if safe_float(row.get("risk_top_yes_prob"), 1.0) < 0.5]
        top1_hits = sum(int(row.get("oracle_risk_top1_hit", 0) or 0) for row in rows)
        any_gold = sum(int(row.get("oracle_any_hallucinated_object", 0) or 0) for row in rows)
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "intervention_object_pred_jsonl": os.path.abspath(args.intervention_object_pred_jsonl),
                    "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv) if args.oracle_rows_csv else "",
                    "oracle_object_col": str(args.oracle_object_col),
                    "model_path": str(args.model_path),
                    "conv_mode": str(args.conv_mode),
                    "question_template": str(args.question_template),
                    "score_mode": str(args.score_mode),
                    "limit": int(args.limit),
                    "max_objects": int(args.max_objects),
                    "object_vocab": str(args.object_vocab),
                    "filter_to_vocab": bool(args.filter_to_vocab),
                },
                "counts": {
                    "n_rows": int(len(rows)),
                    "n_errors": int(n_errors),
                    "n_object_probes": int(n_object_probes),
                    "n_forward_passes_est": int(n_object_probes * (1 if str(args.score_mode) == "yes_only" else 2)),
                    "n_oracle_any_hallucinated_object": int(any_gold),
                    "oracle_top1_hit_rate_over_all": float(top1_hits / max(1, len(rows))),
                    "oracle_top1_hit_rate_over_gold": float(top1_hits / max(1, any_gold)),
                    "n_risk_top_yes_prob_lt_050": int(len(selected_like)),
                },
                "outputs": {"out_csv": os.path.abspath(args.out_csv)},
            },
        )
        print(f"[saved] {args.out_summary_json}")


if __name__ == "__main__":
    main()
