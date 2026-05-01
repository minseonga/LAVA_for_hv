#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from extract_decision_margin_layer_trajectories import (  # noqa: E402
    binary_auroc,
    layer_trajectory,
    mean,
    parse_bool,
    read_jsonl_rows,
    read_jsonl_text_map,
    read_label_rows,
    safe_id,
    std,
    write_csv,
    write_json,
)
from extract_llava_next_vision_ablation_replay_features import ablate_image, parse_modes  # noqa: E402


BASE_FEATURES = [
    "candidate_minus_alt",
    "candidate_prob_binary",
    "margin_abs",
    "yes_minus_no",
    "yes_prob_binary",
    "no_prob_binary",
    "candidate_label_lp",
    "alt_label_lp",
    "c_target_gap_content_min",
    "c_entropy_content_mean",
    "c_first_target_gap",
]


def maybe_float(value: Any) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def numeric_feature_keys(row: Mapping[str, Any]) -> List[str]:
    blocked = {
        "id",
        "image",
        "question",
        "intervention_text",
        "baseline_text",
        "gt_label",
        "answer",
        "label",
        "baseline_label",
        "intervention_label",
        "candidate_label",
        "ablation_mode",
        "score_error",
        "score_error_traceback",
    }
    out: List[str] = []
    for key, value in row.items():
        if key in blocked:
            continue
        if key.startswith(("orig__", "blind__", "delta_", "abs_delta__", "rel_delta_")) and maybe_float(value) is not None:
            out.append(key)
    return out


def add_layer_contrast(
    *,
    meta: Mapping[str, Any],
    mode: str,
    orig_rows: Sequence[Mapping[str, Any]],
    blind_rows: Sequence[Mapping[str, Any]],
    rel_eps: float,
) -> List[Dict[str, Any]]:
    by_orig = {int(row["layer_index"]): row for row in orig_rows}
    by_blind = {int(row["layer_index"]): row for row in blind_rows}
    rows: List[Dict[str, Any]] = []
    for layer in sorted(set(by_orig) & set(by_blind)):
        orig = by_orig[layer]
        blind = by_blind[layer]
        out: Dict[str, Any] = {
            **meta,
            "ablation_mode": str(mode),
            "layer_index": int(layer),
            "layer_frac": orig.get("layer_frac", ""),
            "is_final_layer": orig.get("is_final_layer", ""),
            "candidate_label": orig.get("candidate_label", ""),
        }
        for key in BASE_FEATURES:
            o = maybe_float(orig.get(key))
            b = maybe_float(blind.get(key))
            if o is None or b is None:
                continue
            out[f"orig__{key}"] = float(o)
            out[f"blind__{key}"] = float(b)
            out[f"delta_orig_minus_blind__{key}"] = float(o - b)
            out[f"delta_blind_minus_orig__{key}"] = float(b - o)
            out[f"abs_delta__{key}"] = float(abs(o - b))
            out[f"rel_delta_orig_minus_blind__{key}"] = float((o - b) / max(float(rel_eps), abs(o)))
        rows.append(out)
    return rows


def summarize_layers(long_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    feature_keys = sorted({key for row in long_rows for key in numeric_feature_keys(row)})
    grouped: Dict[tuple[str, int], Dict[str, Any]] = {}
    for row in long_rows:
        layer_f = maybe_float(row.get("layer_index"))
        harm_f = maybe_float(row.get("harm"))
        help_f = maybe_float(row.get("help"))
        if layer_f is None or harm_f is None or help_f is None:
            continue
        harm = int(harm_f)
        help_ = int(help_f)
        if harm not in {0, 1} or help_ not in {0, 1} or (harm == 0 and help_ == 0):
            continue
        key = (str(row.get("ablation_mode", "")), int(layer_f))
        item = grouped.setdefault(
            key,
            {
                "ys": [],
                **{f"{feature}_values": [] for feature in feature_keys},
                **{f"{feature}_harm": [] for feature in feature_keys},
                **{f"{feature}_help": [] for feature in feature_keys},
            },
        )
        item["ys"].append(harm)
        for feature in feature_keys:
            value = maybe_float(row.get(feature))
            if value is None:
                continue
            item[f"{feature}_values"].append(value)
            if harm == 1:
                item[f"{feature}_harm"].append(value)
            if help_ == 1:
                item[f"{feature}_help"].append(value)

    out: List[Dict[str, Any]] = []
    for (mode, layer) in sorted(grouped, key=lambda x: (x[0][0], x[0][1])):
        item = grouped[(mode, layer)]
        ys = [int(x) for x in item["ys"]]
        if not ys:
            continue
        summary: Dict[str, Any] = {
            "ablation_mode": mode,
            "layer_index": layer,
            "n": len(ys),
            "n_harm": sum(ys),
            "n_help": len(ys) - sum(ys),
        }
        for feature in feature_keys:
            values = [float(x) for x in item[f"{feature}_values"]]
            if len(values) != len(ys):
                continue
            auc_high = binary_auroc(values, ys)
            auc_low = binary_auroc([-x for x in values], ys)
            summary[f"harm_{feature}_mean"] = mean(item[f"{feature}_harm"])
            summary[f"harm_{feature}_std"] = std(item[f"{feature}_harm"])
            summary[f"help_{feature}_mean"] = mean(item[f"{feature}_help"])
            summary[f"help_{feature}_std"] = std(item[f"{feature}_help"])
            summary[f"{feature}_auroc"] = max(auc_high, auc_low)
            summary[f"{feature}_direction"] = "high" if auc_high >= auc_low else "low"
            summary[f"{feature}_raw_auroc_high"] = auc_high
        out.append(summary)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract LLaVA-NeXT layer-wise original-vs-ablated replay decision features."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--label_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default="/home/kms/models/llama3-llava-next-8b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_llama_3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llava_next_root", default="/home/kms/LLaVA-NeXT")
    ap.add_argument("--llava_next_torch_type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--llava_next_attn_implementation", default="sdpa", choices=["none", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--ablation_modes", default="black", help="Comma-separated: black,gray,blur,noise")
    ap.add_argument("--blur_radius", type=float, default=32.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--apply_final_norm", type=parse_bool, default=True)
    ap.add_argument("--only_label_rows", type=parse_bool, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rel_eps", type=float, default=1e-6)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    long_csv = os.path.join(out_dir, "layer_vision_ablation_long.csv")
    summary_csv = os.path.join(out_dir, "layer_vision_ablation_summary.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json, flush=True)
        return

    modes = parse_modes(str(args.ablation_modes))
    labels = read_label_rows(os.path.abspath(args.label_rows_csv))
    label_ids = set(labels)
    questions = read_jsonl_rows(os.path.abspath(args.question_file), limit=0)
    if bool(args.only_label_rows):
        questions = [row for row in questions if safe_id(row.get("question_id", row.get("id"))) in label_ids]
    if int(args.limit) > 0:
        questions = questions[: int(args.limit)]
    pred_map = read_jsonl_text_map(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_key))

    from frgavr_cleanroom.llava_next_runtime import OfficialLlavaNextRuntime

    runtime = OfficialLlavaNextRuntime(
        llava_next_root=str(args.llava_next_root),
        model_path=str(args.model_path),
        model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
        torch_type=str(args.llava_next_torch_type),
        attn_implementation=str(args.llava_next_attn_implementation),
    )

    long_rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_intervention = 0
    timings: List[float] = []
    for idx, sample in enumerate(questions):
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        label_meta = labels.get(sid, {})
        candidate_text = str(pred_map.get(sid, "") or label_meta.get("intervention_text", "")).strip()
        meta = {
            "id": sid,
            "image": image_name,
            "question": question,
            "intervention_text": candidate_text,
            **label_meta,
        }
        try:
            if not sid:
                raise ValueError("missing sample id")
            if not image_name:
                raise ValueError("missing image")
            if not question:
                raise ValueError("missing question")
            if not candidate_text:
                n_missing_intervention += 1
                raise ValueError("missing intervention prediction")
            image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            image = runtime.load_image(image_path)
            t0 = time.perf_counter()
            orig = layer_trajectory(
                runtime,
                image=image,
                question=question,
                candidate_text=candidate_text,
                apply_final_norm=bool(args.apply_final_norm),
            )
            for mode in modes:
                blind_image = ablate_image(
                    image,
                    str(mode),
                    blur_radius=float(args.blur_radius),
                    seed=int(args.seed),
                    sample_id_value=sid,
                )
                blind = layer_trajectory(
                    runtime,
                    image=blind_image,
                    question=question,
                    candidate_text=candidate_text,
                    apply_final_norm=bool(args.apply_final_norm),
                )
                long_rows.extend(
                    {
                        **row,
                        "score_error": "",
                    }
                    for row in add_layer_contrast(
                        meta=meta,
                        mode=str(mode),
                        orig_rows=orig,
                        blind_rows=blind,
                        rel_eps=float(args.rel_eps),
                    )
                )
            timings.append(time.perf_counter() - t0)
        except Exception as exc:
            n_errors += 1
            long_rows.append({**meta, "layer_index": "", "score_error": str(exc), "score_error_traceback": traceback.format_exc()})
            print(f"[error] id={sid} {exc!r}", flush=True)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[layer-abl] {idx + 1}/{len(questions)}", flush=True)

    summary_rows = summarize_layers(long_rows)
    write_csv(long_csv, long_rows)
    write_csv(summary_csv, summary_rows)
    write_json(
        summary_json,
        {
            "mode": "llava_next_layer_vision_ablation_replay",
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "intervention_pred_key": str(args.intervention_pred_key),
                "label_rows_csv": os.path.abspath(args.label_rows_csv),
                "model_path": str(args.model_path),
                "conv_mode": str(args.conv_mode),
                "llava_next_root": str(args.llava_next_root),
                "llava_next_attn_implementation": str(args.llava_next_attn_implementation),
                "ablation_modes": modes,
                "apply_final_norm": bool(args.apply_final_norm),
                "only_label_rows": bool(args.only_label_rows),
            },
            "counts": {
                "n_rows": int(len(questions)),
                "n_errors": int(n_errors),
                "n_missing_intervention": int(n_missing_intervention),
                "n_long_rows": int(len(long_rows)),
                "n_summary_rows": int(len(summary_rows)),
            },
            "timing": {
                "feature_total_sec": float(sum(timings)),
                "feature_mean_ms": float(1000.0 * sum(timings) / max(1, len(timings))),
            },
            "outputs": {
                "long_csv": long_csv,
                "summary_csv": summary_csv,
            },
        },
    )
    print("[saved]", long_csv)
    print("[saved]", summary_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
