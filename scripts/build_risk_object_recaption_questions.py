#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List


DEFAULT_TEMPLATE = """Describe the image in a detailed, accurate caption.

Avoid mentioning this uncertain object unless it is clearly visible: {objects}.
Include all salient visible objects you are confident about.
Do not make the caption overly short."""


def read_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("question_id") or row.get("image_id") or row.get("id") or "").strip()
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


def pass_thresholds(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if safe_float(row.get("risk_object_count"), 0.0) < float(args.min_object_count):
        return False
    if not str(row.get("risk_top_object", "")).strip():
        return False
    if safe_float(row.get("risk_top_yes_prob"), 1.0) > float(args.max_yes_prob):
        return False
    if safe_float(row.get("risk_top_lp_margin"), 0.0) > float(args.max_lp_margin):
        return False
    if safe_float(row.get("risk_second_minus_top_yes_prob"), 0.0) < float(args.min_second_gap):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build top-risk-object recaption prompts from deployable risk features.")
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--risk_features_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_selected_ids_json", default="")
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_yes_prob", type=float, default=0.40)
    ap.add_argument("--max_lp_margin", type=float, default=999.0)
    ap.add_argument("--min_second_gap", type=float, default=0.0)
    ap.add_argument("--min_object_count", type=int, default=1)
    args = ap.parse_args()

    questions = read_jsonl(args.question_file, limit=int(args.limit))
    risk_by_id = {safe_id(row): row for row in read_csv_rows(args.risk_features_csv)}

    out_rows: List[Dict[str, Any]] = []
    selected_ids: List[str] = []
    n_missing_risk = 0
    n_rejected = 0
    for idx, q in enumerate(questions):
        sid = safe_id(q) or str(idx)
        risk = risk_by_id.get(sid)
        if risk is None:
            n_missing_risk += 1
            continue
        if not pass_thresholds(risk, args):
            n_rejected += 1
            continue

        obj = str(risk.get("risk_top_object", "")).strip()
        prompt = str(args.template).replace("{objects}", obj)
        selected_ids.append(sid)
        out_rows.append(
            {
                "question_id": sid,
                "image_id": sid,
                "image": q.get("image", ""),
                "question": prompt,
                "text": prompt,
                "label": "",
                "split": q.get("split", ""),
                "category": "deployable_risk_object_recaption",
                "negative_objects": obj,
                "negative_object_count": 1,
                "risk_top_yes_prob": risk.get("risk_top_yes_prob", ""),
                "risk_top_lp_margin": risk.get("risk_top_lp_margin", ""),
                "risk_second_minus_top_yes_prob": risk.get("risk_second_minus_top_yes_prob", ""),
            }
        )

    write_jsonl(args.out_jsonl, out_rows)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if str(args.out_selected_ids_json or "").strip():
        write_json(args.out_selected_ids_json, {"selected_ids": selected_ids})
        print("[saved]", os.path.abspath(args.out_selected_ids_json))
    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "risk_features_csv": os.path.abspath(args.risk_features_csv),
                    "limit": int(args.limit),
                    "max_yes_prob": float(args.max_yes_prob),
                    "max_lp_margin": float(args.max_lp_margin),
                    "min_second_gap": float(args.min_second_gap),
                    "min_object_count": int(args.min_object_count),
                },
                "counts": {
                    "n_questions": len(questions),
                    "n_output_rows": len(out_rows),
                    "n_missing_risk": n_missing_risk,
                    "n_rejected": n_rejected,
                },
                "outputs": {
                    "out_jsonl": os.path.abspath(args.out_jsonl),
                    "out_selected_ids_json": os.path.abspath(args.out_selected_ids_json) if args.out_selected_ids_json else "",
                },
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
