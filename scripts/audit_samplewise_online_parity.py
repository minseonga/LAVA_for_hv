#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def read_json(path: str) -> Any:
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_id(row: Mapping[str, Any]) -> str:
    raw = str(row.get("question_id") or row.get("image_id") or row.get("id") or "").strip()
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def pick_text(row: Mapping[str, Any], key: str) -> str:
    if key != "auto":
        return str(row.get(key, "")).strip()
    for cand in ("output", "text", "caption", "answer", "prediction"):
        value = str(row.get(cand, "")).strip()
        if value:
            return value
    return ""


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def maybe_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def jsonl_map(path: str) -> Dict[str, Dict[str, Any]]:
    return {safe_id(row): row for row in read_jsonl(path) if safe_id(row)}


def csv_map(path: str) -> Dict[str, Dict[str, str]]:
    return {safe_id(row): row for row in read_csv_rows(path) if safe_id(row)}


def compare_predictions(
    reference_jsonl: str,
    candidate_jsonl: str,
    *,
    reference_text_key: str,
    candidate_text_key: str,
    max_examples: int,
) -> Dict[str, Any]:
    ref = jsonl_map(reference_jsonl)
    cand = jsonl_map(candidate_jsonl)
    ids = sorted(set(ref) | set(cand), key=lambda x: (len(x), x))
    n_both = 0
    exact = 0
    normalized = 0
    missing_ref: List[str] = []
    missing_candidate: List[str] = []
    examples: List[Dict[str, Any]] = []
    for sid in ids:
        if sid not in ref:
            missing_ref.append(sid)
            continue
        if sid not in cand:
            missing_candidate.append(sid)
            continue
        n_both += 1
        r_text = pick_text(ref[sid], reference_text_key)
        c_text = pick_text(cand[sid], candidate_text_key)
        is_exact = r_text == c_text
        is_norm = normalize_text(r_text) == normalize_text(c_text)
        exact += int(is_exact)
        normalized += int(is_norm)
        if not is_norm and len(examples) < int(max_examples):
            examples.append(
                {
                    "id": sid,
                    "reference_text": r_text,
                    "candidate_text": c_text,
                }
            )
    return {
        "reference_path": os.path.abspath(reference_jsonl),
        "candidate_path": os.path.abspath(candidate_jsonl),
        "n_reference": len(ref),
        "n_candidate": len(cand),
        "n_both": n_both,
        "n_missing_reference": len(missing_ref),
        "n_missing_candidate": len(missing_candidate),
        "exact_text_match": exact,
        "normalized_text_match": normalized,
        "exact_text_match_rate": None if n_both == 0 else exact / float(n_both),
        "normalized_text_match_rate": None if n_both == 0 else normalized / float(n_both),
        "missing_reference_examples": missing_ref[:max_examples],
        "missing_candidate_examples": missing_candidate[:max_examples],
        "mismatch_examples": examples,
    }


def compare_csv_fields(
    reference_csv: str,
    candidate_csv: str,
    *,
    fields: Sequence[str],
    float_tol: float,
    max_examples: int,
) -> Dict[str, Any]:
    ref = csv_map(reference_csv)
    cand = csv_map(candidate_csv)
    ids = sorted(set(ref) | set(cand), key=lambda x: (len(x), x))
    field_stats: Dict[str, Dict[str, Any]] = {
        field: {"n": 0, "match": 0, "mismatch": 0, "max_abs_diff": 0.0}
        for field in fields
    }
    examples: List[Dict[str, Any]] = []
    missing_ref: List[str] = []
    missing_candidate: List[str] = []
    for sid in ids:
        if sid not in ref:
            missing_ref.append(sid)
            continue
        if sid not in cand:
            missing_candidate.append(sid)
            continue
        row_mismatches: Dict[str, Dict[str, Any]] = {}
        for field in fields:
            r = ref[sid].get(field, "")
            c = cand[sid].get(field, "")
            stat = field_stats[field]
            stat["n"] += 1
            r_f = maybe_float(r)
            c_f = maybe_float(c)
            if r_f is not None and c_f is not None:
                diff = abs(float(r_f) - float(c_f))
                stat["max_abs_diff"] = max(float(stat["max_abs_diff"]), diff)
                same = diff <= float(float_tol)
            else:
                same = str(r) == str(c)
            if same:
                stat["match"] += 1
            else:
                stat["mismatch"] += 1
                row_mismatches[field] = {"reference": r, "candidate": c}
        if row_mismatches and len(examples) < int(max_examples):
            examples.append({"id": sid, "mismatches": row_mismatches})
    for stat in field_stats.values():
        n = int(stat["n"])
        stat["match_rate"] = None if n == 0 else float(stat["match"]) / float(n)
    return {
        "reference_path": os.path.abspath(reference_csv),
        "candidate_path": os.path.abspath(candidate_csv),
        "n_reference": len(ref),
        "n_candidate": len(cand),
        "n_missing_reference": len(missing_ref),
        "n_missing_candidate": len(missing_candidate),
        "fields": field_stats,
        "missing_reference_examples": missing_ref[:max_examples],
        "missing_candidate_examples": missing_candidate[:max_examples],
        "mismatch_examples": examples,
    }


def chair_entries(path: str) -> Dict[str, Dict[str, Any]]:
    obj = read_json(path)
    rows = obj.get("entries", [])
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('split', '')}::{row.get('method', '')}"
        out[key] = dict(row)
    return out


def compare_chair_summary(reference_json: str, candidate_json: str) -> Dict[str, Any]:
    ref = chair_entries(reference_json)
    cand = chair_entries(candidate_json)
    keys = sorted(set(ref) | set(cand))
    metrics = ["CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len"]
    rows: List[Dict[str, Any]] = []
    for key in keys:
        r = ref.get(key, {})
        c = cand.get(key, {})
        row: Dict[str, Any] = {"entry": key}
        for metric in metrics:
            r_v = maybe_float(r.get(metric))
            c_v = maybe_float(c.get(metric))
            row[f"reference_{metric}"] = r_v
            row[f"candidate_{metric}"] = c_v
            row[f"delta_{metric}"] = None if r_v is None or c_v is None else float(c_v - r_v)
        rows.append(row)
    return {
        "reference_path": os.path.abspath(reference_json),
        "candidate_path": os.path.abspath(candidate_json),
        "entries": rows,
    }


def parse_fields(text: str) -> List[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit cached-vs-online sample-wise parity for discriminative/generative pipelines.")
    ap.add_argument("--reference_pred_jsonl", type=str, default="")
    ap.add_argument("--candidate_pred_jsonl", type=str, default="")
    ap.add_argument("--reference_text_key", type=str, default="auto")
    ap.add_argument("--candidate_text_key", type=str, default="auto")
    ap.add_argument("--reference_csv", type=str, default="")
    ap.add_argument("--candidate_csv", type=str, default="")
    ap.add_argument("--csv_fields", type=str, default="route,expert,b_score,c_score,f_score,final_correct")
    ap.add_argument("--reference_chair_summary_json", type=str, default="")
    ap.add_argument("--candidate_chair_summary_json", type=str, default="")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--float_tol", type=float, default=1e-6)
    ap.add_argument("--max_examples", type=int, default=20)
    args = ap.parse_args()

    out: Dict[str, Any] = {"checks": {}}
    if args.reference_pred_jsonl and args.candidate_pred_jsonl:
        out["checks"]["predictions"] = compare_predictions(
            args.reference_pred_jsonl,
            args.candidate_pred_jsonl,
            reference_text_key=str(args.reference_text_key),
            candidate_text_key=str(args.candidate_text_key),
            max_examples=int(args.max_examples),
        )
    if args.reference_csv and args.candidate_csv:
        out["checks"]["csv_fields"] = compare_csv_fields(
            args.reference_csv,
            args.candidate_csv,
            fields=parse_fields(args.csv_fields),
            float_tol=float(args.float_tol),
            max_examples=int(args.max_examples),
        )
    if args.reference_chair_summary_json and args.candidate_chair_summary_json:
        out["checks"]["chair_summary"] = compare_chair_summary(
            args.reference_chair_summary_json,
            args.candidate_chair_summary_json,
        )

    write_json(args.out_json, out)
    print("[saved]", os.path.abspath(args.out_json))
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True)[:8000])


if __name__ == "__main__":
    main()

