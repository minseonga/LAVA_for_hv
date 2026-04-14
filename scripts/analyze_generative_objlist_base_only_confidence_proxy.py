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
    normalize_token,
    read_prediction_map,
    sorted_preview,
    write_csv,
    write_json,
)
from analyze_generative_base_only_confidence_proxy import (
    add_trace_stats,
    auc_high,
    average_precision,
    feature_metrics,
    flag,
    load_trace_rows,
    parse_word_trace,
    precision_at,
    read_csv_rows,
    safe_id,
)


LIST_PREFIX_RE = re.compile(
    r"^\s*(?:objects?|visible objects?|salient objects?|entities?|visible entities?)\s*:\s*",
    flags=re.IGNORECASE,
)


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


def inventory_tokens(text: str) -> List[str]:
    out: List[str] = []
    for item in split_inventory_items(text):
        for raw in re.findall(r"[a-zA-Z][a-zA-Z0-9']*", item.lower()):
            tok = normalize_token(raw)
            if tok and len(tok) > 1:
                out.append(tok)
    return out


def unique_set(items: Iterable[str]) -> set[str]:
    return {str(x) for x in items if str(x).strip()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze baseline_obj_list - intervention_obj_list confidence. "
            "The trace is teacher-forced on baseline object-list outputs, and "
            "features aggregate lp/gap/entropy over baseline-only list units."
        )
    )
    ap.add_argument("--baseline_objlist_jsonl", required=True)
    ap.add_argument("--intervention_objlist_jsonl", required=True)
    ap.add_argument("--baseline_objlist_trace_csv", required=True)
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_feature_metrics_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--preview_limit", type=int, default=80)
    args = ap.parse_args()

    baseline = read_prediction_map(args.baseline_objlist_jsonl)
    intervention = read_prediction_map(args.intervention_objlist_jsonl)
    traces = load_trace_rows(args.baseline_objlist_trace_csv)
    oracle = {safe_id(row): row for row in read_csv_rows(args.oracle_rows_csv)}

    ids = sorted(
        set(baseline) & set(intervention) & set(traces) & set(oracle),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        base_units = unique_set(inventory_tokens(baseline[sid]["text"]))
        int_units = unique_set(inventory_tokens(intervention[sid]["text"]))
        base_only_units = base_units - int_units
        shared_units = base_units & int_units
        int_only_units = int_units - base_units
        word_trace = parse_word_trace(traces[sid])
        base_only_trace = [t for t in word_trace if str(t["word"]) in base_only_units]
        shared_trace = [t for t in word_trace if str(t["word"]) in shared_units]

        bo_words = {str(t["word"]) for t in base_only_trace}
        row: Dict[str, Any] = {
            "id": sid,
            "image": baseline[sid].get("image") or intervention[sid].get("image") or "",
            args.target_col: flag(oracle[sid].get(args.target_col)),
            "base_objlist_text": baseline[sid]["text"],
            "int_objlist_text": intervention[sid]["text"],
            "obj_bo_units": sorted_preview(base_only_units, int(args.preview_limit)),
            "obj_bo_trace_words": sorted_preview(bo_words, int(args.preview_limit)),
            "obj_base_unit_count": len(base_units),
            "obj_int_unit_count": len(int_units),
            "obj_shared_unit_count": len(shared_units),
            "obj_bo_unit_count": len(base_only_units),
            "obj_int_only_unit_count": len(int_only_units),
            "obj_jaccard": len(shared_units) / float(max(1, len(base_units | int_units))),
            "obj_bo_trace_coverage_rate": len(bo_words) / float(max(1, len(base_only_units))),
            "obj_bo_count_x_trace_coverage": len(base_only_units)
            * (len(bo_words) / float(max(1, len(base_only_units)))),
        }
        add_trace_stats(row, "obj_bo_conf", base_only_trace)
        add_trace_stats(row, "obj_shared_conf", shared_trace)
        row["obj_bo_minus_shared_lp_mean"] = float(
            row["obj_bo_conf_lp_mean"] - row["obj_shared_conf_lp_mean"]
        )
        row["obj_bo_minus_shared_gap_mean"] = float(
            row["obj_bo_conf_gap_mean"] - row["obj_shared_conf_gap_mean"]
        )
        row["obj_bo_minus_shared_ent_mean"] = float(
            row["obj_bo_conf_ent_mean"] - row["obj_shared_conf_ent_mean"]
        )
        row["obj_bo_high_gap_count_x_bo_count"] = float(
            row["obj_bo_conf_high_gap_count_ge_000"] * len(base_only_units)
        )
        row["obj_bo_confidence_score_x_bo_count"] = float(
            row["obj_bo_conf_confidence_score_gap_pos"] * len(base_only_units)
        )
        rows.append(row)

    metrics = feature_metrics(rows, args.target_col)
    n_target = sum(flag(row.get(args.target_col)) for row in rows)
    write_csv(args.out_csv, rows)
    write_csv(args.out_feature_metrics_csv, metrics)
    write_json(
        args.out_summary_json,
        {
            "inputs": {
                "baseline_objlist_jsonl": os.path.abspath(args.baseline_objlist_jsonl),
                "intervention_objlist_jsonl": os.path.abspath(args.intervention_objlist_jsonl),
                "baseline_objlist_trace_csv": os.path.abspath(args.baseline_objlist_trace_csv),
                "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
                "target_col": args.target_col,
            },
            "counts": {
                "n_rows": len(rows),
                "n_target": int(n_target),
                "target_rate": float(n_target / float(max(1, len(rows)))),
                "n_feature_metrics": len(metrics),
            },
            "top_feature_metrics": metrics[:30],
            "outputs": {
                "features_csv": os.path.abspath(args.out_csv),
                "feature_metrics_csv": os.path.abspath(args.out_feature_metrics_csv),
                "summary_json": os.path.abspath(args.out_summary_json),
            },
        },
    )
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
