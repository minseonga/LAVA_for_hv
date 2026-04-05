#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def safe_id(value: object) -> str:
    return str(value or "").strip()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_prediction_text_map(path: str, text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = safe_id(row.get("question_id", row.get("id")))
        if not sample_id:
            continue
        if text_key == "auto":
            text = str(row.get("text", "")).strip()
            if not text:
                text = str(row.get("output", "")).strip()
            if not text:
                text = str(row.get("answer", "")).strip()
            if not text:
                text = str(row.get("caption", "")).strip()
        else:
            text = str(row.get(text_key, "")).strip()
        out[sample_id] = text
    return out


def normalize_term(value: object) -> str:
    if isinstance(value, (list, tuple)):
        parts = [normalize_term(x) for x in value]
        parts = [x for x in parts if x]
        return parts[-1] if parts else ""
    s = str(value if value is not None else "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_term_set(values: Iterable[object]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        term = normalize_term(value)
        if term:
            out.add(term)
    return out


def parse_image_id(value: object) -> str:
    s = safe_id(value)
    if not s:
        return ""
    digits = re.findall(r"(\d+)", s)
    if not digits:
        return ""
    return str(int(digits[-1]))


def word_count(row: Dict[str, Any]) -> int:
    words = row.get("words")
    if isinstance(words, list):
        return int(len(words))
    caption = str(row.get("caption", "")).strip()
    if not caption:
        return 0
    return int(len(re.findall(r"[A-Za-z0-9]+", caption)))


def load_feature_map(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        if sid:
            out[sid] = row
    return out


def load_claim_map(path: str) -> Dict[str, Dict[str, Any]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    rows = obj.get("sentences", [])
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image_id = safe_id(row.get("image_id"))
        if not image_id:
            continue
        gt_terms = normalize_term_set(row.get("mscoco_gt_words", []))
        generated_terms = normalize_term_set(row.get("mscoco_generated_words", []))
        hallucinated_terms = normalize_term_set(row.get("mscoco_hallucinated_words", []))
        supported_terms = gt_terms.intersection(generated_terms)
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        out[image_id] = {
            "caption": str(row.get("caption", "")).strip(),
            "gt_terms": gt_terms,
            "generated_terms": generated_terms,
            "hallucinated_terms": hallucinated_terms,
            "supported_terms": supported_terms,
            "chair_i": maybe_float(metrics.get("CHAIRi")),
            "chair_s": maybe_float(metrics.get("CHAIRs")),
            "recall": maybe_float(metrics.get("Recall")),
            "word_count": int(word_count(row)),
        }
    return out


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def sorted_join(values: Iterable[str]) -> str:
    seq = sorted({str(v).strip() for v in values if str(v).strip()})
    return " || ".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build generative method tables from claim-aware utility deltas.")
    ap.add_argument("--baseline_features_csv", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--method_name", type=str, required=True)
    ap.add_argument("--benchmark_name", type=str, default="pope_discovery_caption")
    ap.add_argument("--split_name", type=str, default="claim_aware_probe")
    ap.add_argument("--supported_weight", type=float, default=1.0)
    ap.add_argument("--hall_weight", type=float, default=1.0)
    ap.add_argument("--length_weight", type=float, default=0.25)
    ap.add_argument("--epsilon", type=float, default=1e-12)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, required=True)
    ap.add_argument("--baseline_pred_text_key", type=str, default="auto")
    ap.add_argument("--intervention_pred_text_key", type=str, default="auto")
    args = ap.parse_args()

    feature_map = load_feature_map(os.path.abspath(args.baseline_features_csv))
    baseline_map = load_prediction_text_map(os.path.abspath(args.baseline_pred_jsonl), args.baseline_pred_text_key)
    intervention_map = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), args.intervention_pred_text_key)
    claim_base = load_claim_map(os.path.abspath(args.baseline_chair_json))
    claim_int = load_claim_map(os.path.abspath(args.intervention_chair_json))

    rows: List[Dict[str, Any]] = []
    missing_claim = 0
    missing_image_id = 0
    for sid, feat in feature_map.items():
        image_id = (
            safe_id(feat.get("image_id"))
            or parse_image_id(feat.get("image"))
            or parse_image_id(feat.get("id"))
        )
        if not image_id:
            missing_image_id += 1
            continue
        base = claim_base.get(image_id)
        intr = claim_int.get(image_id)
        if base is None or intr is None:
            missing_claim += 1
            continue

        gt_terms = set(base["gt_terms"]) or set(intr["gt_terms"])
        gt_count = max(1, len(gt_terms))

        base_supported = set(base["supported_terms"])
        int_supported = set(intr["supported_terms"])
        base_hall = set(base["hallucinated_terms"])
        int_hall = set(intr["hallucinated_terms"])

        supported_gain = int_supported - base_supported
        supported_drop = base_supported - int_supported
        wrong_added = int_hall - base_hall
        wrong_removed = base_hall - int_hall

        base_supported_recall = safe_div(float(len(base_supported)), float(gt_count))
        int_supported_recall = safe_div(float(len(int_supported)), float(gt_count))
        base_hall_rate = safe_div(float(len(base_hall)), float(max(1, len(base["generated_terms"]))))
        int_hall_rate = safe_div(float(len(int_hall)), float(max(1, len(intr["generated_terms"]))))
        length_collapse = safe_div(
            float(max(0, int(base["word_count"]) - int(intr["word_count"]))),
            float(max(1, int(base["word_count"]))),
        )

        base_claim_utility = (
            float(args.supported_weight) * base_supported_recall
            - float(args.hall_weight) * base_hall_rate
        )
        int_claim_utility = (
            float(args.supported_weight) * int_supported_recall
            - float(args.hall_weight) * int_hall_rate
            - float(args.length_weight) * length_collapse
        )
        utility_delta = float(int_claim_utility - base_claim_utility)

        help_ = int(utility_delta > float(args.epsilon))
        harm = int(utility_delta < (-float(args.epsilon)))
        neutral = int((help_ == 0) and (harm == 0))

        row: Dict[str, Any] = {
            "id": sid,
            "method": args.method_name,
            "benchmark": args.benchmark_name,
            "split": args.split_name,
            "image": safe_id(feat.get("image")),
            "image_id": image_id,
            "question": safe_id(feat.get("question")),
            "baseline_text": str(baseline_map.get(sid, "")).strip(),
            "intervention_text": str(intervention_map.get(sid, "")).strip(),
            "baseline_claim_utility": base_claim_utility,
            "intervention_claim_utility": int_claim_utility,
            "claim_utility_delta": utility_delta,
            "base_supported_recall": base_supported_recall,
            "int_supported_recall": int_supported_recall,
            "delta_supported_recall": float(int_supported_recall - base_supported_recall),
            "base_hall_rate": base_hall_rate,
            "int_hall_rate": int_hall_rate,
            "delta_hall_rate": float(int_hall_rate - base_hall_rate),
            "length_collapse_penalty": length_collapse,
            "n_gt_terms": int(len(gt_terms)),
            "n_base_supported": int(len(base_supported)),
            "n_int_supported": int(len(int_supported)),
            "n_base_hall": int(len(base_hall)),
            "n_int_hall": int(len(int_hall)),
            "n_supported_gained": int(len(supported_gain)),
            "n_supported_dropped": int(len(supported_drop)),
            "n_wrong_added": int(len(wrong_added)),
            "n_wrong_removed": int(len(wrong_removed)),
            "supported_gained_terms": sorted_join(supported_gain),
            "supported_dropped_terms": sorted_join(supported_drop),
            "wrong_added_terms": sorted_join(wrong_added),
            "wrong_removed_terms": sorted_join(wrong_removed),
            "base_supported_terms": sorted_join(base_supported),
            "int_supported_terms": sorted_join(int_supported),
            "base_hall_terms": sorted_join(base_hall),
            "int_hall_terms": sorted_join(int_hall),
            "help": help_,
            "harm": harm,
            "neutral": neutral,
            "claim_supported_gained": int(len(supported_gain) > 0),
            "claim_supported_dropped": int(len(supported_drop) > 0),
            "claim_wrong_added": int(len(wrong_added) > 0),
            "claim_wrong_removed": int(len(wrong_removed) > 0),
            "label_delta": "help" if help_ else "harm" if harm else "neutral",
            "oracle_route": "baseline" if harm else "method",
        }
        for key, value in feat.items():
            if key in row:
                continue
            row[key] = value
        rows.append(row)

    rows.sort(key=lambda r: int(str(r["id"])))
    write_csv(args.out_csv, rows)

    n = len(rows)
    summary = {
        "inputs": {
            "baseline_features_csv": os.path.abspath(args.baseline_features_csv),
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
            "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
            "method_name": args.method_name,
            "supported_weight": float(args.supported_weight),
            "hall_weight": float(args.hall_weight),
            "length_weight": float(args.length_weight),
        },
        "counts": {
            "n_rows": int(n),
            "missing_claim": int(missing_claim),
            "missing_image_id": int(missing_image_id),
            "help_rate": safe_div(float(sum(int(r["help"]) for r in rows)), float(max(1, n))),
            "harm_rate": safe_div(float(sum(int(r["harm"]) for r in rows)), float(max(1, n))),
            "neutral_rate": safe_div(float(sum(int(r["neutral"]) for r in rows)), float(max(1, n))),
            "mean_base_claim_utility": safe_div(float(sum(float(r["baseline_claim_utility"]) for r in rows)), float(max(1, n))),
            "mean_intervention_claim_utility": safe_div(float(sum(float(r["intervention_claim_utility"]) for r in rows)), float(max(1, n))),
            "mean_claim_utility_delta": safe_div(float(sum(float(r["claim_utility_delta"]) for r in rows)), float(max(1, n))),
            "mean_delta_supported_recall": safe_div(float(sum(float(r["delta_supported_recall"]) for r in rows)), float(max(1, n))),
            "mean_delta_hall_rate": safe_div(float(sum(float(r["delta_hall_rate"]) for r in rows)), float(max(1, n))),
            "mean_length_collapse_penalty": safe_div(float(sum(float(r["length_collapse_penalty"]) for r in rows)), float(max(1, n))),
            "claim_wrong_added_rate": safe_div(float(sum(int(r["claim_wrong_added"]) for r in rows)), float(max(1, n))),
            "claim_wrong_removed_rate": safe_div(float(sum(int(r["claim_wrong_removed"]) for r in rows)), float(max(1, n))),
            "claim_supported_gained_rate": safe_div(float(sum(int(r["claim_supported_gained"]) for r in rows)), float(max(1, n))),
            "claim_supported_dropped_rate": safe_div(float(sum(int(r["claim_supported_dropped"]) for r in rows)), float(max(1, n))),
        },
        "outputs": {
            "table_csv": os.path.abspath(args.out_csv),
        },
    }
    write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
