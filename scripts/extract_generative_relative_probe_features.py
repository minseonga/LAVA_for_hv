#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from extract_vga_generative_mention_features import build_feature_row
from frgavr_cleanroom.runtime import (
    CleanroomLlavaRuntime,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    safe_id,
)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_feature_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        if sid:
            out[sid] = row
    return out


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


PROBE_NUMERIC_KEYS = [
    "probe_entropy_head_mean_real",
    "probe_entropy_tail_mean_real",
    "probe_entropy_tail_minus_head_real",
    "probe_gap_head_mean_real",
    "probe_gap_tail_mean_real",
    "probe_gap_tail_minus_head_real",
    "probe_lp_head_mean_real",
    "probe_lp_tail_mean_real",
    "probe_lp_tail_minus_head_real",
    "probe_lp_content_mean_real",
    "probe_lp_content_std_real",
    "probe_lp_content_min_real",
    "probe_target_gap_content_mean_real",
    "probe_target_gap_content_std_real",
    "probe_target_gap_content_min_real",
    "probe_entropy_content_mean_real",
    "probe_entropy_content_std_real",
    "probe_entropy_content_max_real",
    "probe_mention_entropy_max_real",
    "probe_mention_lp_min_real",
    "probe_mention_lp_tail_gap_real",
    "probe_mention_target_gap_min_real",
    "probe_n_cont_tokens",
    "probe_n_content_tokens",
    "probe_n_mentions_total",
    "probe_n_object_mentions",
    "probe_n_attribute_phrases",
    "probe_n_relation_phrases",
    "probe_n_count_phrases",
    "probe_object_diversity",
    "probe_mention_diversity",
    "probe_first_half_object_mentions",
    "probe_second_half_object_mentions",
    "probe_tail_tokens_after_last_mention",
    "probe_last_mention_pos_frac",
]


def add_relative_probe_features(row: Dict[str, Any], intervention_probe: Dict[str, Any]) -> None:
    for key in PROBE_NUMERIC_KEYS:
        base_v = maybe_float(row.get(key))
        int_v = maybe_float(intervention_probe.get(key))
        if int_v is None:
            continue
        suffix = str(key).replace("probe_", "")
        row[f"pair_intprobe_{suffix}"] = float(int_v)
        if base_v is None:
            continue
        delta = float(int_v - base_v)
        row[f"pair_delta_probe_{suffix}"] = delta
        row[f"pair_abs_delta_probe_{suffix}"] = abs(delta)
        row[f"pair_ratio_probe_{suffix}"] = float(safe_div(float(int_v), float(base_v))) if abs(float(base_v)) > 1e-12 else 0.0

    # Directional collapse features aligned with discriminative intuition.
    base_ent_tail = maybe_float(row.get("probe_entropy_tail_mean_real"))
    int_ent_tail = maybe_float(intervention_probe.get("probe_entropy_tail_mean_real"))
    if base_ent_tail is not None and int_ent_tail is not None:
        row["pair_probe_bad_shift_entropy_tail"] = float(int_ent_tail - base_ent_tail)

    base_ent_max = maybe_float(row.get("probe_mention_entropy_max_real"))
    int_ent_max = maybe_float(intervention_probe.get("probe_mention_entropy_max_real"))
    if base_ent_max is not None and int_ent_max is not None:
        row["pair_probe_bad_shift_entropy_max"] = float(int_ent_max - base_ent_max)

    base_gap_tail = maybe_float(row.get("probe_gap_tail_mean_real"))
    int_gap_tail = maybe_float(intervention_probe.get("probe_gap_tail_mean_real"))
    if base_gap_tail is not None and int_gap_tail is not None:
        row["pair_probe_bad_shift_gap_tail"] = float(base_gap_tail - int_gap_tail)

    base_gap_min = maybe_float(row.get("probe_mention_target_gap_min_real"))
    int_gap_min = maybe_float(intervention_probe.get("probe_mention_target_gap_min_real"))
    if base_gap_min is not None and int_gap_min is not None:
        row["pair_probe_bad_shift_gap_min"] = float(base_gap_min - int_gap_min)

    base_lp_tail = maybe_float(row.get("probe_lp_tail_mean_real"))
    int_lp_tail = maybe_float(intervention_probe.get("probe_lp_tail_mean_real"))
    if base_lp_tail is not None and int_lp_tail is not None:
        row["pair_probe_bad_shift_lp_tail"] = float(base_lp_tail - int_lp_tail)

    base_lp_min = maybe_float(row.get("probe_mention_lp_min_real"))
    int_lp_min = maybe_float(intervention_probe.get("probe_mention_lp_min_real"))
    if base_lp_min is not None and int_lp_min is not None:
        row["pair_probe_bad_shift_lp_min"] = float(base_lp_min - int_lp_min)

    base_content_lp_min = maybe_float(row.get("probe_lp_content_min_real"))
    int_content_lp_min = maybe_float(intervention_probe.get("probe_lp_content_min_real"))
    if base_content_lp_min is not None and int_content_lp_min is not None:
        row["pair_probe_bad_shift_lp_content_min"] = float(base_content_lp_min - int_content_lp_min)

    base_content_gap_min = maybe_float(row.get("probe_target_gap_content_min_real"))
    int_content_gap_min = maybe_float(intervention_probe.get("probe_target_gap_content_min_real"))
    if base_content_gap_min is not None and int_content_gap_min is not None:
        row["pair_probe_bad_shift_target_gap_content_min"] = float(base_content_gap_min - int_content_gap_min)

    base_content_lp_std = maybe_float(row.get("probe_lp_content_std_real"))
    int_content_lp_std = maybe_float(intervention_probe.get("probe_lp_content_std_real"))
    if base_content_lp_std is not None and int_content_lp_std is not None:
        row["pair_probe_bad_shift_lp_content_std"] = float(int_content_lp_std - base_content_lp_std)

    base_tokens = maybe_float(row.get("probe_n_cont_tokens"))
    int_tokens = maybe_float(intervention_probe.get("probe_n_cont_tokens"))
    if base_tokens is not None and int_tokens is not None:
        row["pair_probe_token_drop_frac"] = float(safe_div(float(max(0.0, base_tokens - int_tokens)), float(max(1.0, base_tokens))))

    base_mentions = maybe_float(row.get("probe_n_mentions_total"))
    int_mentions = maybe_float(intervention_probe.get("probe_n_mentions_total"))
    if base_mentions is not None and int_mentions is not None:
        row["pair_probe_mention_drop_frac"] = float(
            safe_div(float(max(0.0, base_mentions - int_mentions)), float(max(1.0, base_mentions)))
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract intervention-relative probe-collapse features for generative fallback control.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--base_features_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pred_text_key", type=str, default="auto")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--max_mentions", type=int, default=12)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    base_feature_map = load_feature_map(args.base_features_csv)
    intervention_map = load_prediction_text_map(args.intervention_pred_jsonl, text_key=args.pred_text_key)

    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(args.model_base or None),
        conv_mode=args.conv_mode,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_base_features = 0
    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        intervention_text = str(intervention_map.get(sample_id, "")).strip()
        image_path = os.path.join(args.image_folder, image_name)

        row: Dict[str, Any] = dict(base_feature_map.get(sample_id, {}))
        if not row:
            n_missing_base_features += 1
            row = {"id": sample_id, "image": image_name, "question": question}
        row["pair_relative_probe_error"] = ""

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
            intervention_probe = build_feature_row(
                runtime=runtime,
                image_path=image_path,
                question=question,
                candidate_text=intervention_text,
                sample_id=sample_id,
                image_name=image_name,
                max_mentions=int(args.max_mentions),
            )
            add_relative_probe_features(row, intervention_probe)
        except Exception as exc:
            n_errors += 1
            row["pair_relative_probe_error"] = str(exc)

        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[relative-probe] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")

    if str(args.out_summary_json or "").strip():
        feature_keys = [k for k in rows[0].keys() if k.startswith("pair_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "base_features_csv": os.path.abspath(args.base_features_csv),
                    "model_path": args.model_path,
                    "model_base": args.model_base,
                    "conv_mode": args.conv_mode,
                    "device": args.device,
                },
                "counts": {
                    "n_questions": int(len(question_rows)),
                    "n_rows": int(len(rows)),
                    "n_pair_features": int(len(feature_keys)),
                    "n_errors": int(n_errors),
                    "n_missing_base_feature_rows": int(n_missing_base_features),
                },
                "outputs": {
                    "feature_csv": os.path.abspath(args.out_csv),
                },
            },
        )
        print(f"[saved] {os.path.abspath(args.out_summary_json)}")


if __name__ == "__main__":
    main()
