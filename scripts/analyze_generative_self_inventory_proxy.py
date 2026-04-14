#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from extract_generative_semantic_pairwise_features import (
    content_tokens,
    read_prediction_map,
    semantic_units,
    unit_summary,
    write_csv,
    write_json,
)


LIST_PREFIX_RE = re.compile(
    r"^\s*(?:objects?|visible objects?|salient objects?|entities?|visible entities?)\s*:\s*",
    flags=re.IGNORECASE,
)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("id") or row.get("image_id") or row.get("question_id") or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int_flag(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def sorted_preview(values: Iterable[str], limit: int) -> str:
    vals = sorted({str(v) for v in values if str(v).strip()})
    if limit > 0:
        vals = vals[:limit]
    return "|".join(vals)


def split_inventory_items(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
    text = LIST_PREFIX_RE.sub("", text)
    text = re.sub(r"\b(?:and|or)\s+others?\b", "", text, flags=re.IGNORECASE)
    raw_parts = re.split(r"[,;\n]+", text)
    parts: List[str] = []
    for part in raw_parts:
        part = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip()
        part = LIST_PREFIX_RE.sub("", part)
        part = part.strip(" .")
        if part:
            parts.append(part)
    return parts or [text]


def inventory_summary(text: str) -> Dict[str, Any]:
    token_units: List[str] = []
    phrase_units: List[str] = []
    items = split_inventory_items(text)
    for item in items:
        toks = content_tokens(item)
        item_tokens, item_phrases = semantic_units(toks, max_gap=3)
        token_units.extend(item_tokens)
        phrase_units.extend(item_phrases)
    all_units = token_units + phrase_units
    return {
        "text": str(text or ""),
        "items": items,
        "token_units": token_units,
        "phrase_units": phrase_units,
        "all_units": all_units,
        "unique_token_units": set(token_units),
        "unique_phrase_units": set(phrase_units),
        "unique_all_units": set(all_units),
    }


def auc_high(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    good = 0.0
    total = 0
    for a in pos:
        for b in neg:
            total += 1
            if a > b:
                good += 1.0
            elif a == b:
                good += 0.5
    return good / float(total) if total else None


def average_precision(items: Sequence[Tuple[int, float]]) -> Optional[float]:
    ranked = sorted(items, key=lambda item: item[1], reverse=True)
    n_pos = sum(label for label, _ in ranked)
    if n_pos <= 0:
        return None
    hits = 0
    total = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label:
            hits += 1
            total += hits / float(rank)
    return total / float(n_pos)


def precision_at(items: Sequence[Tuple[int, float]], k: int) -> Optional[float]:
    if not items:
        return None
    top = sorted(items, key=lambda item: item[1], reverse=True)[:k]
    if not top:
        return None
    return sum(label for label, _ in top) / float(len(top))


def numeric_feature_metrics(rows: Sequence[Dict[str, Any]], target_col: str) -> List[Dict[str, Any]]:
    if not rows:
        return []
    excluded = {"id", "image", "base_caption", "int_caption", "base_inventory_text", "int_inventory_text"}
    metrics: List[Dict[str, Any]] = []
    for feature in rows[0].keys():
        if feature in excluded or feature.endswith("_units") or feature.endswith("_items"):
            continue
        pairs: List[Tuple[int, float]] = []
        for row in rows:
            value = safe_float(row.get(feature))
            if value is None:
                continue
            pairs.append((safe_int_flag(row.get(target_col)), value))
        if len(pairs) < max(10, int(0.8 * len(rows))):
            continue
        vals = {round(score, 12) for _, score in pairs}
        if len(vals) < 3:
            continue
        pos = [score for label, score in pairs if label]
        neg = [score for label, score in pairs if not label]
        auc = auc_high(pos, neg)
        if auc is None:
            continue
        direction = "high" if auc >= 0.5 else "low"
        oriented = [(label, score if direction == "high" else -score) for label, score in pairs]
        metrics.append(
            {
                "feature": feature,
                "direction": direction,
                "auroc": max(float(auc), float(1.0 - auc)),
                "auroc_high": float(auc),
                "ap": average_precision(oriented),
                "p_at_10": precision_at(oriented, 10),
                "p_at_25": precision_at(oriented, 25),
                "p_at_50": precision_at(oriented, 50),
                "n": len(pairs),
                "n_pos": sum(label for label, _ in pairs),
                "pos_mean": sum(pos) / float(len(pos)) if pos else "",
                "neg_mean": sum(neg) / float(len(neg)) if neg else "",
            }
        )
    metrics.sort(key=lambda row: (float(row.get("auroc") or 0.0), float(row.get("ap") or 0.0)), reverse=True)
    return metrics


def frac(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def add_set_features(
    row: Dict[str, Any],
    *,
    prefix: str,
    base_set: set[str],
    int_set: set[str],
    base_inventory_set: set[str],
    int_inventory_set: Optional[set[str]],
    preview_limit: int,
) -> None:
    base_only = base_set - int_set
    int_only = int_set - base_set
    shared_caption = base_set & int_set
    base_supported = base_set & base_inventory_set
    int_supported_by_base_inv = int_set & base_inventory_set
    base_only_supported = base_only & base_inventory_set
    base_only_not_supported = base_only - base_inventory_set
    int_only_in_base_inv = int_only & base_inventory_set

    row[f"{prefix}_base_caption_unit_count"] = len(base_set)
    row[f"{prefix}_int_caption_unit_count"] = len(int_set)
    row[f"{prefix}_base_inventory_unit_count"] = len(base_inventory_set)
    row[f"{prefix}_caption_shared_count"] = len(shared_caption)
    row[f"{prefix}_caption_jaccard"] = frac(len(shared_caption), len(base_set | int_set))
    row[f"{prefix}_base_only_count"] = len(base_only)
    row[f"{prefix}_int_only_count"] = len(int_only)
    row[f"{prefix}_base_caption_supported_by_base_inventory_count"] = len(base_supported)
    row[f"{prefix}_base_caption_supported_by_base_inventory_rate"] = frac(len(base_supported), len(base_set))
    row[f"{prefix}_int_caption_supported_by_base_inventory_count"] = len(int_supported_by_base_inv)
    row[f"{prefix}_int_caption_supported_by_base_inventory_rate"] = frac(len(int_supported_by_base_inv), len(int_set))
    row[f"{prefix}_base_only_supported_by_base_inventory_count"] = len(base_only_supported)
    row[f"{prefix}_base_only_supported_by_base_inventory_rate"] = frac(len(base_only_supported), len(base_only))
    row[f"{prefix}_base_only_not_in_base_inventory_count"] = len(base_only_not_supported)
    row[f"{prefix}_int_only_in_base_inventory_count"] = len(int_only_in_base_inv)
    row[f"{prefix}_base_only_supported_minus_risk"] = len(base_only_supported) - len(base_only_not_supported)
    row[f"{prefix}_base_only_supported_x_caption_jaccard_gap"] = len(base_only_supported) * (1.0 - row[f"{prefix}_caption_jaccard"])

    row[f"{prefix}_base_only_units"] = sorted_preview(base_only, preview_limit)
    row[f"{prefix}_base_only_supported_by_base_inventory_units"] = sorted_preview(base_only_supported, preview_limit)
    row[f"{prefix}_base_only_not_in_base_inventory_units"] = sorted_preview(base_only_not_supported, preview_limit)

    if int_inventory_set is not None:
        inv_shared = base_inventory_set & int_inventory_set
        inv_base_only = base_inventory_set - int_inventory_set
        inv_int_only = int_inventory_set - base_inventory_set
        suppressed_caption_units = inv_base_only & base_set
        suppressed_base_only_units = inv_base_only & base_only
        base_only_supported_and_suppressed = base_only_supported & inv_base_only
        row[f"{prefix}_int_inventory_unit_count"] = len(int_inventory_set)
        row[f"{prefix}_inventory_jaccard"] = frac(len(inv_shared), len(base_inventory_set | int_inventory_set))
        row[f"{prefix}_base_inventory_only_count"] = len(inv_base_only)
        row[f"{prefix}_int_inventory_only_count"] = len(inv_int_only)
        row[f"{prefix}_suppressed_caption_unit_count"] = len(suppressed_caption_units)
        row[f"{prefix}_suppressed_base_only_count"] = len(suppressed_base_only_units)
        row[f"{prefix}_base_only_supported_and_inventory_suppressed_count"] = len(
            base_only_supported_and_suppressed
        )
        row[f"{prefix}_base_inventory_only_units"] = sorted_preview(inv_base_only, preview_limit)
        row[f"{prefix}_base_only_supported_and_inventory_suppressed_units"] = sorted_preview(
            base_only_supported_and_suppressed, preview_limit
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze CHAIR-free self-inventory proxy features. The target labels may "
            "come from validation/diagnostic CHAIR oracles, but no CHAIR/GT fields are "
            "used in the inference-side features."
        )
    )
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--baseline_inventory_pred_jsonl", required=True)
    ap.add_argument("--intervention_inventory_pred_jsonl", default="")
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_feature_metrics_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--preview_limit", type=int, default=80)
    args = ap.parse_args()

    baseline = read_prediction_map(args.baseline_pred_jsonl)
    intervention = read_prediction_map(args.intervention_pred_jsonl)
    base_inventory = read_prediction_map(args.baseline_inventory_pred_jsonl)
    int_inventory = (
        read_prediction_map(args.intervention_inventory_pred_jsonl)
        if str(args.intervention_inventory_pred_jsonl or "").strip()
        else {}
    )
    oracle_by_id = {safe_id(row): row for row in read_csv_rows(args.oracle_rows_csv)}

    ids = sorted(
        set(baseline.keys())
        & set(intervention.keys())
        & set(base_inventory.keys())
        & set(oracle_by_id.keys()),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    if int_inventory:
        ids = [sid for sid in ids if sid in int_inventory]

    rows: List[Dict[str, Any]] = []
    for sid in ids:
        b = unit_summary(baseline[sid]["text"])
        i = unit_summary(intervention[sid]["text"])
        bi = inventory_summary(base_inventory[sid]["text"])
        ii = inventory_summary(int_inventory[sid]["text"]) if sid in int_inventory else None
        row: Dict[str, Any] = {
            "id": sid,
            "image": baseline[sid].get("image") or intervention[sid].get("image") or base_inventory[sid].get("image", ""),
            args.target_col: safe_int_flag(oracle_by_id[sid].get(args.target_col)),
            "base_caption": baseline[sid]["text"],
            "int_caption": intervention[sid]["text"],
            "base_inventory_text": base_inventory[sid]["text"],
            "int_inventory_text": int_inventory[sid]["text"] if sid in int_inventory else "",
            "inv_base_inventory_item_count": len(bi["items"]),
            "inv_base_inventory_word_count": len(bi["token_units"]),
            "inv_int_inventory_item_count": len(ii["items"]) if ii is not None else "",
            "inv_int_inventory_word_count": len(ii["token_units"]) if ii is not None else "",
        }
        add_set_features(
            row,
            prefix="inv_tok",
            base_set=set(b["unique_token_units"]),
            int_set=set(i["unique_token_units"]),
            base_inventory_set=set(bi["unique_token_units"]),
            int_inventory_set=set(ii["unique_token_units"]) if ii is not None else None,
            preview_limit=int(args.preview_limit),
        )
        add_set_features(
            row,
            prefix="inv_all",
            base_set=set(b["unique_all_units"]),
            int_set=set(i["unique_all_units"]),
            base_inventory_set=set(bi["unique_all_units"]),
            int_inventory_set=set(ii["unique_all_units"]) if ii is not None else None,
            preview_limit=int(args.preview_limit),
        )
        rows.append(row)

    metrics = numeric_feature_metrics(rows, args.target_col)
    target_count = sum(safe_int_flag(row.get(args.target_col)) for row in rows)
    summary = {
        "inputs": {
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "baseline_inventory_pred_jsonl": os.path.abspath(args.baseline_inventory_pred_jsonl),
            "intervention_inventory_pred_jsonl": (
                os.path.abspath(args.intervention_inventory_pred_jsonl)
                if args.intervention_inventory_pred_jsonl
                else ""
            ),
            "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
            "target_col": args.target_col,
        },
        "counts": {
            "n_rows": len(rows),
            "n_target": int(target_count),
            "target_rate": frac(target_count, len(rows)),
            "n_feature_metrics": len(metrics),
        },
        "top_feature_metrics": metrics[:30],
        "outputs": {
            "features_csv": os.path.abspath(args.out_csv),
            "feature_metrics_csv": os.path.abspath(args.out_feature_metrics_csv),
            "summary_json": os.path.abspath(args.out_summary_json),
        },
    }

    write_csv(args.out_csv, rows)
    write_csv(args.out_feature_metrics_csv, metrics)
    write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_feature_metrics_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))
    for metric in metrics[:10]:
        print(
            "[metric]",
            metric["feature"],
            "dir=",
            metric["direction"],
            "auc=",
            metric["auroc"],
            "ap=",
            metric["ap"],
        )


if __name__ == "__main__":
    main()
