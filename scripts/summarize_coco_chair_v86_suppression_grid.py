#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


METRICS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len")


def read_json(path: str) -> Any:
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(str(key))
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def maybe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out else None


def pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return 100.0 * float(value)


def entries_by_method(summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in summary.get("entries", []) or []:
        method = str(row.get("method", "")).strip()
        if method:
            out[method] = dict(row)
    return out


def parse_action_tag(path: str) -> Dict[str, Any]:
    name = os.path.basename(path)
    m = re.search(r"_(single_token|first_token|all_tokens)_bias([^_]+)_yp([^.]*(?:\.[^.]*)?)\.json$", name)
    if not m:
        return {}
    return {
        "suppress_mode": m.group(1),
        "suppress_bias": m.group(2),
        "threshold": m.group(3),
    }


def first_json(pattern: str) -> Tuple[str, Dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return "", {}
    path = paths[0]
    try:
        return path, read_json(path)
    except Exception:
        return path, {}


def summarize_run(run_dir: str) -> Optional[Dict[str, Any]]:
    summary_paths = sorted(glob.glob(os.path.join(run_dir, "summary", "chair_v82_object_token_suppression_*.json")))
    if not summary_paths:
        return None
    summary_path = summary_paths[0]
    summary = read_json(summary_path)
    entries = entries_by_method(summary)
    intervention = entries.get("intervention", {})
    ours = entries.get("object_token_suppression", {})
    if not intervention or not ours:
        return None

    selected_path, selected = first_json(os.path.join(run_dir, "test", "pred_object_token_suppression_selected_*.summary.json"))
    merged_path, merged = first_json(os.path.join(run_dir, "test", "pred_object_token_suppression_merged_*.summary.json"))
    tag = parse_action_tag(summary_path)
    selected_inputs = selected.get("inputs", {}) if isinstance(selected, dict) else {}
    selected_counts = selected.get("counts", {}) if isinstance(selected, dict) else {}
    merged_counts = merged.get("counts", {}) if isinstance(merged, dict) else {}

    row: Dict[str, Any] = {
        "run_dir": os.path.abspath(run_dir),
        "summary_json": os.path.abspath(summary_path),
        "selected_summary_json": os.path.abspath(selected_path) if selected_path else "",
        "merged_summary_json": os.path.abspath(merged_path) if merged_path else "",
        "threshold": tag.get("threshold") or selected_inputs.get("risk_max_yes_prob", ""),
        "suppress_mode": tag.get("suppress_mode") or selected_inputs.get("suppress_mode", ""),
        "suppress_bias": tag.get("suppress_bias") or selected_inputs.get("suppress_bias", ""),
        "selected": selected_counts.get("n_selected_by_threshold", ""),
        "written": selected_counts.get("n_written", ""),
        "skipped_no_token_ids": selected_counts.get("n_skipped_no_token_ids", ""),
        "changed": merged_counts.get("n_repaired", ""),
    }
    for metric in METRICS:
        base = maybe_float(intervention.get(metric))
        val = maybe_float(ours.get(metric))
        row[f"intervention_{metric}"] = pct(base)
        row[metric] = pct(val)
        row[f"delta_{metric}"] = None if base is None or val is None else pct(val - base)
    precision = maybe_float(ours.get("Precision"))
    base_precision = maybe_float(intervention.get("Precision"))
    row["Ci_unique"] = None if precision is None else pct(1.0 - precision)
    row["intervention_Ci_unique"] = None if base_precision is None else pct(1.0 - base_precision)
    row["delta_Ci_unique"] = (
        None if precision is None or base_precision is None else pct((1.0 - precision) - (1.0 - base_precision))
    )
    return row


def rank_key(row: Mapping[str, Any]) -> Tuple[int, float, float, float]:
    d_f1 = maybe_float(row.get("delta_F1")) or -999.0
    d_rec = maybe_float(row.get("delta_Recall")) or -999.0
    d_chairs = maybe_float(row.get("delta_CHAIRs")) or 999.0
    d_chairi = maybe_float(row.get("delta_CHAIRi")) or 999.0
    recall_safe = int(d_rec >= -0.30)
    return (recall_safe, d_f1, -d_chairs, -d_chairi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize v86 COCO-CHAIR token-suppression grid runs.")
    ap.add_argument("--grid_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    run_dirs = [p for p in sorted(glob.glob(os.path.join(os.path.abspath(args.grid_root), "*"))) if os.path.isdir(p)]
    rows = [row for row in (summarize_run(p) for p in run_dirs) if row is not None]
    rows_sorted = sorted(rows, key=rank_key, reverse=True)

    write_csv(args.out_csv, rows_sorted)
    write_json(
        args.out_json,
        {
            "inputs": {"grid_root": os.path.abspath(args.grid_root)},
            "counts": {"n_runs_found": len(run_dirs), "n_summarized": len(rows_sorted)},
            "top_rows": rows_sorted[: int(args.top_k)],
            "outputs": {"csv": os.path.abspath(args.out_csv), "json": os.path.abspath(args.out_json)},
        },
    )
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_json))
    for row in rows_sorted[: int(args.top_k)]:
        print(
            "[top]",
            "th=", row.get("threshold"),
            "mode=", row.get("suppress_mode"),
            "bias=", row.get("suppress_bias"),
            "selected=", row.get("selected"),
            "changed=", row.get("changed"),
            "dCs=", row.get("delta_CHAIRs"),
            "dCi=", row.get("delta_CHAIRi"),
            "dR=", row.get("delta_Recall"),
            "dF1=", row.get("delta_F1"),
        )


if __name__ == "__main__":
    main()
