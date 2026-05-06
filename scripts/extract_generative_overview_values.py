#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from extract_generative_semantic_pairwise_features import read_prediction_map
from extract_intervention_object_replay_risk_features import content_token_ids, object_spans, ordered_unique
from frgavr_cleanroom.runtime import CleanroomLlavaRuntime, load_question_rows, write_json
from run_vga_caption_with_token_suppression import suppression_token_ids


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


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_id(row: dict[str, Any]) -> str:
    return norm_id(row.get("question_id") or row.get("id") or row.get("image_id"))


def image_digits(value: object) -> str:
    return "".join(ch for ch in Path(str(value or "")).name if ch.isdigit())


def matches_sample(row: dict[str, Any], *, question_id: str, image: str) -> bool:
    qid = norm_id(question_id)
    if qid and row_id(row) == qid:
        return True
    target_image = Path(str(image or "")).name
    if target_image and Path(str(row.get("image") or "")).name == target_image:
        return True
    return bool(target_image and image_digits(row.get("image") or row.get("image_id")) == image_digits(target_image))


def find_sample(rows: Sequence[dict[str, Any]], *, question_id: str, image: str) -> dict[str, Any]:
    for row in rows:
        if matches_sample(row, question_id=question_id, image=image):
            return row
    raise KeyError(f"sample not found: question_id={question_id!r} image={image!r}")


def load_risk_row(path: str, *, question_id: str, image: str) -> dict[str, str]:
    for row in read_csv_rows(path):
        if matches_sample(row, question_id=question_id, image=image):
            return row
    raise KeyError(f"risk row not found: question_id={question_id!r} image={image!r}")


def parse_risk_details(row: dict[str, str]) -> list[dict[str, Any]]:
    raw = str(row.get("risk_details_json") or "").strip()
    if raw:
        try:
            details = json.loads(raw)
            if isinstance(details, list):
                return [dict(item) for item in details if isinstance(item, dict)]
        except Exception:
            pass
    details: list[dict[str, Any]] = []
    top_obj = str(row.get("risk_top_object", "")).strip()
    if top_obj:
        details.append({"object": top_obj, "yesno_prob": safe_float(row.get("risk_top_yes_prob"), 1.0)})
    second_obj = str(row.get("risk_second_object", "")).strip()
    if second_obj:
        details.append({"object": second_obj, "yesno_prob": safe_float(row.get("risk_second_yes_prob"), 1.0)})
    return details


def choose_objects(details: Sequence[dict[str, Any]], selected_object: str, *, max_objects: int) -> list[str]:
    by_obj: dict[str, float] = {}
    for item in details:
        obj = str(item.get("object", "")).strip()
        if obj:
            by_obj[obj] = safe_float(item.get("yesno_prob"), item.get("risk_support_score", 0.0))
    selected = str(selected_object or "").strip()
    high = sorted(
        [obj for obj in by_obj if obj.lower() != selected.lower()],
        key=lambda obj: by_obj.get(obj, 0.0),
        reverse=True,
    )
    out = high[: max(0, int(max_objects) - 1)]
    if selected:
        out.append(selected)
    return ordered_unique(out, max_items=int(max_objects))


def minmax(values: Sequence[float]) -> list[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-8:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def token_metrics_for_rel_idx(pack: Any, rel_idx: int) -> dict[str, float]:
    target_id = int(pack.cont_ids[int(rel_idx)].item())
    decision_pos = int(pack.decision_positions[int(rel_idx)].item())
    vec = pack.logits[decision_pos].to(torch.float32)
    log_probs = F.log_softmax(vec, dim=-1)
    probs = torch.softmax(vec, dim=-1)
    target_logit = float(vec[target_id].item())
    target_lp = float(log_probs[target_id].item())
    target_prob = float(probs[target_id].item())
    top2_vals, top2_idx = torch.topk(vec, k=2, dim=-1)
    top1_id = int(top2_idx[0].item())
    best_other = float(top2_vals[1].item() if top1_id == target_id else top2_vals[0].item())
    return {
        "target_id": float(target_id),
        "decision_pos": float(decision_pos),
        "target_logit": target_logit,
        "target_lp": target_lp,
        "target_prob": target_prob,
        "target_gap": float(target_logit - best_other),
    }


def mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else float(default)


def object_replay_metrics(tokenizer: Any, pack: Any, objects: Sequence[str]) -> list[dict[str, Any]]:
    cont_ids = [int(x) for x in pack.cont_ids.tolist()]
    out: list[dict[str, Any]] = []
    for obj in objects:
        spans = object_spans(tokenizer, cont_ids, str(obj), [str(obj)])
        metrics: list[dict[str, float]] = []
        for start, end, _term in spans:
            for rel_idx in range(int(start), int(end)):
                metrics.append(token_metrics_for_rel_idx(pack, rel_idx))
        out.append(
            {
                "object": str(obj),
                "n_spans": int(len(spans)),
                "n_tokens": int(len(metrics)),
                "target_logit_mean": mean([m["target_logit"] for m in metrics]),
                "target_logit_max": max([m["target_logit"] for m in metrics], default=0.0),
                "target_gap_mean": mean([m["target_gap"] for m in metrics]),
                "target_prob_mean": mean([m["target_prob"] for m in metrics]),
            }
        )
    return out


def first_object_decision_position(tokenizer: Any, pack: Any, obj: str) -> int:
    spans = object_spans(tokenizer, [int(x) for x in pack.cont_ids.tolist()], str(obj), [str(obj)])
    if not spans:
        toks = content_token_ids(tokenizer, str(obj))
        if toks:
            target_id = int(toks[0])
            positions = torch.where(pack.cont_ids.long() == target_id)[0]
            if int(positions.numel()) > 0:
                return int(pack.decision_positions[int(positions[0].item())].item())
        raise ValueError(f"selected object span not found in method caption: {obj!r}")
    start, _end, _term = spans[0]
    return int(pack.decision_positions[int(start)].item())


def suppression_logit_rows(
    tokenizer: Any,
    pack: Any,
    selected_object: str,
    *,
    suppress_mode: str,
    suppress_bias: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    decision_pos = first_object_decision_position(tokenizer, pack, selected_object)
    vec = pack.logits[decision_pos].to(torch.float32)
    ids = suppression_token_ids(tokenizer, str(selected_object), str(suppress_mode))
    after_vec = vec.clone()
    if ids:
        after_vec[[int(x) for x in ids]] += float(suppress_bias)
    rows: list[dict[str, Any]] = []
    for token_id in ids:
        tid = int(token_id)
        before = float(vec[tid].item())
        after = float(after_vec[tid].item())
        before_rank = int(torch.sum(vec > before).item()) + 1
        after_rank = int(torch.sum(after_vec > after).item()) + 1
        rows.append(
            {
                "token_id": tid,
                "token_text": tokenizer.decode([tid], skip_special_tokens=True),
                "before_logit": before,
                "after_logit": after,
                "before_rank": before_rank,
                "after_rank": after_rank,
                "suppress_bias": float(suppress_bias),
            }
        )
    rows = sorted(rows, key=lambda row: safe_float(row.get("before_logit")), reverse=True)
    return rows[: max(1, int(max_tokens))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract real single-sample values for the generative overview bar panels.")
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--method_pred_jsonl", required=True)
    ap.add_argument("--repaired_pred_jsonl", default="")
    ap.add_argument("--risk_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--question_id", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--objects", default="")
    ap.add_argument("--selected_object", default="")
    ap.add_argument("--max_objects", type=int, default=4)
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suppress_mode", choices=["single_token", "first_token", "all_tokens"], default="first_token")
    ap.add_argument("--suppress_bias", type=float, default=-1.0)
    ap.add_argument("--max_suppression_tokens", type=int, default=3)
    args = ap.parse_args()

    questions = load_question_rows(os.path.abspath(args.question_file), limit=0)
    sample = find_sample(questions, question_id=str(args.question_id), image=str(args.image))
    sid = row_id(sample)
    image_name = str(sample.get("image", args.image)).strip()
    question = str(sample.get("question", sample.get("text", ""))).strip()

    method_map = read_prediction_map(os.path.abspath(args.method_pred_jsonl), text_key="auto")
    if sid not in method_map:
        raise KeyError(f"method caption missing for id={sid} in {args.method_pred_jsonl}")
    method_caption = str(method_map[sid].get("text", "")).strip()

    repaired_caption = ""
    if str(args.repaired_pred_jsonl or "").strip():
        repaired_map = read_prediction_map(os.path.abspath(args.repaired_pred_jsonl), text_key="auto")
        repaired_caption = str(repaired_map.get(sid, {}).get("text", "")).strip()

    risk_row = load_risk_row(os.path.abspath(args.risk_csv), question_id=sid, image=image_name)
    risk_details = parse_risk_details(risk_row)
    selected_object = str(args.selected_object or risk_row.get("risk_top_object") or "").strip()
    objects = [x.strip() for x in str(args.objects or "").split(",") if x.strip()]
    if not objects:
        objects = choose_objects(risk_details, selected_object, max_objects=int(args.max_objects))
    support_by_obj = {
        str(item.get("object", "")).strip().lower(): safe_float(item.get("yesno_prob"), item.get("risk_support_score", 0.0))
        for item in risk_details
        if str(item.get("object", "")).strip()
    }
    support_probs = [support_by_obj.get(obj.lower(), 0.0) for obj in objects]

    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=(str(args.model_base) or None),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )
    image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
    image = runtime.load_image(image_path)
    pack = runtime.teacher_force_candidate(
        image=image,
        question=question,
        candidate_text=method_caption,
        output_attentions=False,
    )

    method_object_rows = object_replay_metrics(runtime.tokenizer, pack, objects)
    method_raw_logits = [safe_float(row.get("target_logit_mean"), 0.0) for row in method_object_rows]
    method_logits_plot = minmax(method_raw_logits)
    suppression_rows = suppression_logit_rows(
        runtime.tokenizer,
        pack,
        selected_object,
        suppress_mode=str(args.suppress_mode),
        suppress_bias=float(args.suppress_bias),
        max_tokens=int(args.max_suppression_tokens),
    )
    before_raw = [safe_float(row.get("before_logit"), 0.0) for row in suppression_rows]
    after_raw = [safe_float(row.get("after_logit"), 0.0) for row in suppression_rows]
    both_norm = minmax([*before_raw, *after_raw])
    before_plot = both_norm[: len(before_raw)]
    after_plot = both_norm[len(before_raw) :]
    token_labels = [str(row.get("token_text", "")).strip() or str(row.get("token_id")) for row in suppression_rows]

    out = {
        "sample": {
            "id": sid,
            "image": image_name,
            "image_abs_path": image_path,
            "question": question,
            "method_caption": method_caption,
            "repaired_caption": repaired_caption,
        },
        "risk": {
            "risk_csv": os.path.abspath(args.risk_csv),
            "selected_object": selected_object,
            "risk_top_yes_prob": safe_float(risk_row.get("risk_top_yes_prob"), 1.0),
            "risk_details": risk_details,
        },
        "plot": {
            "objects": objects,
            "method_logits": method_logits_plot,
            "support_probs": support_probs,
            "selected_object": selected_object,
            "token_labels": token_labels,
            "before": before_plot,
            "after": after_plot,
        },
        "raw_values": {
            "method_object_token_logits": method_object_rows,
            "method_logits_metric": "minmax-normalized mean target logits from teacher-forced method-caption replay",
            "support_metric": "yes probability from object yes/no support probe",
            "suppression_token_logits": suppression_rows,
            "suppression_metric": "same-step token logits before and after adding suppress_bias",
        },
    }
    write_json(args.out_json, out)
    print("[saved]", args.out_json)
    print("[sample]", sid, image_name)
    print("[selected]", selected_object, "p_yes=", out["risk"]["risk_top_yes_prob"])
    print("[objects]", ", ".join(objects))
    print("[support_probs]", ", ".join(f"{x:.6f}" for x in support_probs))
    print("[method_logits_plot]", ", ".join(f"{x:.6f}" for x in method_logits_plot))
    print("[tokens]", ", ".join(token_labels))
    print("[before]", ", ".join(f"{x:.6f}" for x in before_plot))
    print("[after]", ", ".join(f"{x:.6f}" for x in after_plot))


if __name__ == "__main__":
    main()
