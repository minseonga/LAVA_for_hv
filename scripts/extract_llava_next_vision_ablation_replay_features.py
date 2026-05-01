#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


LABEL_KEEP = {
    "baseline_label",
    "intervention_label",
    "baseline_text",
    "intervention_text",
    "baseline_correct",
    "intervention_correct",
    "final_correct",
    "harm",
    "help",
    "neutral",
    "category",
    "gt_label",
    "answer",
    "label",
}


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_id(value: Any) -> str:
    return str(value or "").strip()


def read_jsonl_rows(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def prediction_text(row: Mapping[str, Any], key: str) -> str:
    if key and key != "auto":
        return str(row.get(key, "")).strip()
    for name in ("text", "output", "answer", "caption", "prediction"):
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def load_prediction_text_map(path: str, text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl_rows(path):
        sid = sample_id(row)
        if sid:
            out[sid] = prediction_text(row, text_key)
    return out


def sample_id(row: Mapping[str, Any]) -> str:
    for key in ("question_id", "id", "qid", "image_id"):
        raw = str(row.get(key, "")).strip()
        if raw:
            try:
                return str(int(float(raw)))
            except Exception:
                return raw
    return ""


def read_label_rows(path: str) -> Dict[str, Dict[str, str]]:
    if not str(path or "").strip():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = sample_id(row)
        if not sid:
            continue
        out[sid] = {key: str(row.get(key, "")) for key in LABEL_KEEP if key in row}
    return out


def parse_modes(value: str) -> List[str]:
    modes: List[str] = []
    for raw in str(value or "").split(","):
        mode = raw.strip().lower()
        if not mode:
            continue
        if mode not in {"black", "gray", "noise", "blur"}:
            raise ValueError(f"unsupported ablation mode: {mode}")
        modes.append(mode)
    return modes or ["black"]


def ablate_image(image: Image.Image, mode: str, *, blur_radius: float, seed: int, sample_id_value: str) -> Image.Image:
    from PIL import Image, ImageFilter

    mode = str(mode).strip().lower()
    if mode == "black":
        return Image.new("RGB", image.size, (0, 0, 0))
    if mode == "gray":
        return Image.new("RGB", image.size, (127, 127, 127))
    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    if mode == "noise":
        # Deterministic noise by sample id. Use PIL bytes to avoid requiring numpy.
        rng = random.Random(f"{int(seed)}:{sample_id_value}")
        width, height = image.size
        data = bytes(rng.randrange(0, 256) for _ in range(width * height * 3))
        return Image.frombytes("RGB", image.size, data)
    raise ValueError(f"unsupported ablation mode: {mode}")


def is_number(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x)


def numeric_keys(row: Mapping[str, Any]) -> List[str]:
    blocked = {"id", "question_id", "image", "question"}
    return [key for key, value in row.items() if key not in blocked and is_number(value)]


def add_contrast_features(out: Dict[str, Any], mode: str, orig: Mapping[str, Any], blind: Mapping[str, Any]) -> None:
    common = sorted(set(numeric_keys(orig)) & set(numeric_keys(blind)))
    for key in common:
        o = float(orig[key])
        b = float(blind[key])
        out[f"abl_{mode}_orig__{key}"] = o
        out[f"abl_{mode}_blind__{key}"] = b
        out[f"abl_{mode}_delta_orig_minus_blind__{key}"] = o - b
        out[f"abl_{mode}_delta_blind_minus_orig__{key}"] = b - o
        out[f"abl_{mode}_abs_delta__{key}"] = abs(o - b)
        out[f"abl_{mode}_rel_delta_orig_minus_blind__{key}"] = (o - b) / max(1e-6, abs(o))


def feature_pack(
    runtime: Any,
    *,
    image: Image.Image,
    sample: Mapping[str, Any],
    sid: str,
    image_name: str,
    question: str,
    candidate_text: str,
    output_hidden_states: bool,
    lp_tail_quantile: float,
    lp_tail_eps: float,
    lp_len_corr_alpha: float,
) -> Dict[str, Any]:
    from frgavr_cleanroom.runtime import select_content_indices
    from run_discriminative_meta_strong_online import (
        cheap_features_from_pack,
        object_terms_from_sample,
        object_token_indices,
    )

    object_terms = object_terms_from_sample(sample)
    pack = runtime.teacher_force_candidate(
        image=image,
        question=question,
        candidate_text=candidate_text,
        output_attentions=False,
        output_hidden_states=bool(output_hidden_states),
    )
    content_indices = select_content_indices(runtime.tokenizer, pack.cont_ids)
    object_indices = object_token_indices(runtime.tokenizer, pack.cont_ids, object_terms)
    return cheap_features_from_pack(
        runtime=runtime,
        pack=pack,
        sample_id=sid,
        image_name=image_name,
        question=question,
        content_indices=content_indices,
        object_terms=object_terms,
        object_indices=object_indices,
        lp_tail_quantile=float(lp_tail_quantile),
        lp_tail_eps=float(lp_tail_eps),
        lp_len_corr_alpha=float(lp_len_corr_alpha),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract LLaVA-NeXT intervention-answer replay features under original vs vision-ablated images."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--label_rows_csv", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llava_next_root", default="/home/kms/LLaVA-NeXT")
    ap.add_argument("--model_path", default="/home/kms/models/llama3-llava-next-8b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_llama_3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llava_next_torch_type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--llava_next_attn_implementation", default="sdpa", choices=["none", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--ablation_modes", default="black", help="Comma-separated: black,gray,blur,noise")
    ap.add_argument("--blur_radius", type=float, default=32.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output_hidden_states", type=parse_bool, default=True)
    ap.add_argument("--lp_tail_quantile", type=float, default=0.10)
    ap.add_argument("--lp_tail_eps", type=float, default=1e-6)
    ap.add_argument("--lp_len_corr_alpha", type=float, default=0.35)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rows_csv = os.path.join(out_dir, "vision_ablation_replay_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json, flush=True)
        return

    modes = parse_modes(str(args.ablation_modes))
    from frgavr_cleanroom.llava_next_runtime import OfficialLlavaNextRuntime

    questions = read_jsonl_rows(os.path.abspath(args.question_file), limit=int(args.limit))
    intervention_map = load_prediction_text_map(
        os.path.abspath(args.intervention_pred_jsonl),
        text_key=str(args.intervention_pred_key),
    )
    label_map = read_label_rows(str(args.label_rows_csv))

    runtime = OfficialLlavaNextRuntime(
        llava_next_root=str(args.llava_next_root),
        model_path=str(args.model_path),
        model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
        torch_type=str(args.llava_next_torch_type),
        attn_implementation=str(args.llava_next_attn_implementation),
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_intervention = 0
    timings: List[float] = []
    for idx, sample in enumerate(questions):
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        candidate_text = str(intervention_map.get(sid, "")).strip()
        row: Dict[str, Any] = {
            "id": sid,
            "image": image_name,
            "question": question,
            "intervention_text": candidate_text,
            "score_error": "",
        }
        row.update(label_map.get(sid, {}))
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

            t0 = time.perf_counter()
            image = runtime.load_image(image_path)
            orig = feature_pack(
                runtime,
                image=image,
                sample=sample,
                sid=sid,
                image_name=image_name,
                question=question,
                candidate_text=candidate_text,
                output_hidden_states=bool(args.output_hidden_states),
                lp_tail_quantile=float(args.lp_tail_quantile),
                lp_tail_eps=float(args.lp_tail_eps),
                lp_len_corr_alpha=float(args.lp_len_corr_alpha),
            )
            for mode in modes:
                blind_image = ablate_image(
                    image,
                    mode,
                    blur_radius=float(args.blur_radius),
                    seed=int(args.seed),
                    sample_id_value=sid,
                )
                blind = feature_pack(
                    runtime,
                    image=blind_image,
                    sample=sample,
                    sid=sid,
                    image_name=image_name,
                    question=question,
                    candidate_text=candidate_text,
                    output_hidden_states=bool(args.output_hidden_states),
                    lp_tail_quantile=float(args.lp_tail_quantile),
                    lp_tail_eps=float(args.lp_tail_eps),
                    lp_len_corr_alpha=float(args.lp_len_corr_alpha),
                )
                add_contrast_features(row, mode, orig, blind)
            dt = time.perf_counter() - t0
            row["feature_ms"] = float(dt * 1000.0)
            timings.append(float(dt))
        except Exception as exc:
            n_errors += 1
            row["score_error"] = str(exc)
            row["score_error_traceback"] = traceback.format_exc()
            print(f"[error] id={sid} {exc!r}", flush=True)

        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[vision-ablation] {idx + 1}/{len(questions)}", flush=True)

    write_csv(rows_csv, rows)
    write_json(
        summary_json,
        {
            "mode": "llava_next_vision_ablation_replay",
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "intervention_pred_key": str(args.intervention_pred_key),
                "label_rows_csv": os.path.abspath(args.label_rows_csv) if str(args.label_rows_csv).strip() else "",
                "model_path": str(args.model_path),
                "conv_mode": str(args.conv_mode),
                "llava_next_root": str(args.llava_next_root),
                "llava_next_torch_type": str(args.llava_next_torch_type),
                "llava_next_attn_implementation": str(args.llava_next_attn_implementation),
                "ablation_modes": modes,
                "output_hidden_states": bool(args.output_hidden_states),
            },
            "counts": {
                "n_rows": int(len(questions)),
                "n_errors": int(n_errors),
                "n_missing_intervention": int(n_missing_intervention),
                "n_with_labels": int(sum(1 for row in rows if str(row.get("harm", "")).strip() != "")),
            },
            "timing": {
                "feature_total_sec": float(sum(timings)),
                "feature_mean_ms": float(1000.0 * sum(timings) / max(1, len(timings))),
            },
            "outputs": {
                "rows_csv": rows_csv,
            },
        },
    )
    print("[saved]", rows_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
