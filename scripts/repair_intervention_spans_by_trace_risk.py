#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import (  # noqa: E402
    CleanroomLlavaRuntime,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    safe_id,
    write_csv,
    write_json,
    write_jsonl,
)
from scripts.extract_vga_generative_mention_features import build_feature_payload  # noqa: E402


def clean_caption_text(text: str) -> str:
    out = str(text or "")
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,;:])", r"\2", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.!?])$", r"\1", out)
    out = out.strip()
    return out


def remove_spans(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    out = str(text or "")
    for start, end in sorted({(int(a), int(b)) for a, b in spans if int(b) > int(a)}, reverse=True):
        start = max(0, min(start, len(out)))
        end = max(start, min(end, len(out)))
        out = out[:start] + " " + out[end:]
    return clean_caption_text(out)


def mention_risk_score(row: Dict[str, Any]) -> float:
    ent = float(row.get("ent_max", 0.0))
    gap = float(row.get("gap_min", 0.0))
    lp = float(row.get("lp_min", 0.0))
    tail_gap = float(row.get("lp_tail_gap", 0.0))
    return float(ent + max(0.0, -gap) + 0.5 * max(0.0, -lp) + max(0.0, -tail_gap))


def is_candidate_kind(row: Dict[str, Any], kind_filter: str) -> bool:
    kinds = {part.strip() for part in str(row.get("kinds", "")).split("|") if part.strip()}
    if kind_filter == "any":
        return True
    if kind_filter == "object":
        return "object_mention" in kinds
    if kind_filter == "noun":
        return bool({"object_mention", "noun_phrase"} & kinds)
    return str(kind_filter) in kinds


def select_risky_mentions(
    mention_rows: Sequence[Dict[str, Any]],
    *,
    kind_filter: str,
    min_risk_score: float,
    min_ent: float,
    max_gap: float,
    max_lp: float,
    max_lp_tail_gap: float,
    max_repairs: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in mention_rows:
        if not is_candidate_kind(row, kind_filter):
            continue
        try:
            risk = mention_risk_score(row)
            ent = float(row.get("ent_max", 0.0))
            gap = float(row.get("gap_min", 0.0))
            lp = float(row.get("lp_min", 0.0))
            tail_gap = float(row.get("lp_tail_gap", 0.0))
        except Exception:
            continue
        if risk < float(min_risk_score):
            continue
        if ent < float(min_ent):
            continue
        if gap > float(max_gap):
            continue
        if lp > float(max_lp):
            continue
        if tail_gap > float(max_lp_tail_gap):
            continue
        out = dict(row)
        out["risk_score"] = float(risk)
        candidates.append(out)

    candidates.sort(
        key=lambda r: (
            float(r.get("risk_score", 0.0)),
            float(r.get("ent_max", 0.0)),
            -float(r.get("gap_min", 0.0)),
            -float(r.get("lp_min", 0.0)),
        ),
        reverse=True,
    )
    return candidates[: max(0, int(max_repairs))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply intervention-only local span deletion for high-risk generated mention spans."
    )
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_rows_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--pred_text_key", type=str, default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_mentions", type=int, default=128)
    ap.add_argument("--kind_filter", type=str, default="object", choices=["object", "noun", "any", "object_mention", "noun_phrase"])
    ap.add_argument("--min_risk_score", type=float, default=4.0)
    ap.add_argument("--min_ent", type=float, default=2.5)
    ap.add_argument("--max_gap", type=float, default=0.5)
    ap.add_argument("--max_lp", type=float, default=-0.5)
    ap.add_argument("--max_lp_tail_gap", type=float, default=0.25)
    ap.add_argument("--max_repairs_per_caption", type=int, default=1)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_jsonl):
        print(f"[reuse] {os.path.abspath(args.out_jsonl)}")
        return

    question_rows = load_question_rows(os.path.abspath(args.question_file), limit=int(args.limit))
    pred_map = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), text_key=str(args.pred_text_key))

    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=str(args.model_base or "") or None,
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )

    out_preds: List[Dict[str, Any]] = []
    repair_rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_repaired = 0
    n_spans_removed = 0

    for sample in tqdm(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        candidate_text = str(pred_map.get(sample_id, "")).strip()
        image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
        final_text = candidate_text
        selected: List[Dict[str, Any]] = []
        error = ""
        decoded_text = candidate_text
        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not question:
                raise ValueError("Missing question text.")
            if not candidate_text:
                raise ValueError("Missing intervention prediction text.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            payload = build_feature_payload(
                runtime=runtime,
                image_path=image_path,
                question=question,
                candidate_text=candidate_text,
                sample_id=sample_id,
                image_name=image_name,
                max_mentions=int(args.max_mentions),
            )
            decoded_text = str(payload.get("decoded_text", candidate_text))
            mention_rows = [dict(x) for x in payload.get("mention_rows", [])]
            selected = select_risky_mentions(
                mention_rows,
                kind_filter=str(args.kind_filter),
                min_risk_score=float(args.min_risk_score),
                min_ent=float(args.min_ent),
                max_gap=float(args.max_gap),
                max_lp=float(args.max_lp),
                max_lp_tail_gap=float(args.max_lp_tail_gap),
                max_repairs=int(args.max_repairs_per_caption),
            )
            spans = [(int(row["start"]), int(row["end"])) for row in selected if "start" in row and "end" in row]
            if spans:
                final_text = remove_spans(decoded_text, spans)
                n_repaired += 1
                n_spans_removed += len(spans)
            else:
                final_text = decoded_text
        except Exception as exc:
            n_errors += 1
            error = str(exc)

        out_row = {
            "question_id": sample_id,
            "id": sample_id,
            "image_id": sample.get("image_id", sample_id),
            "image": image_name,
            "output": final_text,
            "text": final_text,
            "span_repair_applied": int(bool(selected)),
            "span_repair_n": int(len(selected)),
            "span_repair_error": error,
        }
        out_preds.append(out_row)
        repair_rows.append(
            {
                "id": sample_id,
                "image": image_name,
                "span_repair_applied": int(bool(selected)),
                "span_repair_n": int(len(selected)),
                "selected_texts": " || ".join(str(row.get("text", "")) for row in selected),
                "selected_scores": " || ".join(f"{float(row.get('risk_score', 0.0)):.4f}" for row in selected),
                "selected_ent_max": " || ".join(str(row.get("ent_max", "")) for row in selected),
                "selected_gap_min": " || ".join(str(row.get("gap_min", "")) for row in selected),
                "selected_lp_min": " || ".join(str(row.get("lp_min", "")) for row in selected),
                "intervention_text": candidate_text,
                "decoded_text": decoded_text,
                "repaired_text": final_text,
                "error": error,
            }
        )

    write_jsonl(os.path.abspath(args.out_jsonl), out_preds)
    write_csv(os.path.abspath(args.out_rows_csv), repair_rows)
    write_json(
        os.path.abspath(args.out_summary_json),
        {
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "limit": int(args.limit),
                "kind_filter": str(args.kind_filter),
                "min_risk_score": float(args.min_risk_score),
                "min_ent": float(args.min_ent),
                "max_gap": float(args.max_gap),
                "max_lp": float(args.max_lp),
                "max_lp_tail_gap": float(args.max_lp_tail_gap),
                "max_repairs_per_caption": int(args.max_repairs_per_caption),
            },
            "counts": {
                "n_rows": int(len(out_preds)),
                "n_errors": int(n_errors),
                "n_repaired": int(n_repaired),
                "repair_rate": float(n_repaired / float(max(1, len(out_preds)))),
                "n_spans_removed": int(n_spans_removed),
            },
            "outputs": {
                "pred_jsonl": os.path.abspath(args.out_jsonl),
                "rows_csv": os.path.abspath(args.out_rows_csv),
                "summary_json": os.path.abspath(args.out_summary_json),
            },
        },
    )
    print("[saved]", os.path.abspath(args.out_jsonl))
    print("[saved]", os.path.abspath(args.out_rows_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
