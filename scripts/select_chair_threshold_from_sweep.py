#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def maybe_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def load_json(path: str) -> Any:
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def find_one(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else ""


def threshold_from_text(text: str) -> Optional[float]:
    basename = os.path.basename(str(text))
    patterns = [
        r"_yp([0-9]+(?:\.[0-9]+)?)",
        r"yp([0-9]+(?:\.[0-9]+)?)",
        r"TH([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, basename)
        if m:
            return float(m.group(1))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", basename)
    if m:
        return float(m.group(1))
    return None


def parse_threshold_dir(spec: str) -> Tuple[Optional[float], str]:
    if "::" in spec:
        th_s, root = spec.split("::", 1)
        return float(th_s), os.path.abspath(root)
    root = os.path.abspath(spec)
    return threshold_from_text(root), root


def load_chair_summary(root: str, explicit_path: str = "") -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    path = explicit_path
    if not path:
        path = find_one(os.path.join(root, "summary", "chair_v82_object_token_suppression_*.json"))
    if not path:
        raise FileNotFoundError(f"Could not find chair_v82 summary under {root}")
    obj = load_json(path)
    entries = obj.get("entries", [])
    intervention = None
    repaired = None
    for row in entries:
        method = str(row.get("method", "")).strip()
        if method == "intervention":
            intervention = row
        elif method in {"object_token_suppression", "risk_object_recaption", "oracle_negative_recaption"}:
            repaired = row
    if intervention is None or repaired is None:
        raise ValueError(f"Summary is missing intervention/repaired entries: {path}")
    return os.path.abspath(path), dict(intervention), dict(repaired)


def load_count_summaries(root: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    suppress_path = find_one(os.path.join(root, "*", "pred_object_token_suppression_selected_*.summary.json"))
    merge_path = find_one(os.path.join(root, "*", "pred_object_token_suppression_merged_*.summary.json"))
    if suppress_path:
        suppress = load_json(suppress_path)
        counts = suppress.get("counts", {})
        out["suppression_summary_json"] = os.path.abspath(suppress_path)
        out["selected"] = int(counts.get("n_selected_by_threshold", 0) or 0)
        out["written"] = int(counts.get("n_written", 0) or 0)
        out["skipped_no_token_ids"] = int(counts.get("n_skipped_no_token_ids", 0) or 0)
    if merge_path:
        merged = load_json(merge_path)
        counts = merged.get("counts", {})
        out["merge_summary_json"] = os.path.abspath(merge_path)
        out["changed"] = int(counts.get("n_repaired", 0) or 0)
    return out


def row_from_root(root_spec: str) -> Dict[str, Any]:
    threshold, root = parse_threshold_dir(root_spec)
    chair_summary_json, base, repaired = load_chair_summary(root)
    if threshold is None:
        threshold = threshold_from_text(chair_summary_json)
    if threshold is None:
        raise ValueError(f"Could not infer threshold from {root_spec}")
    row: Dict[str, Any] = {
        "threshold": float(threshold),
        "root": root,
        "chair_summary_json": chair_summary_json,
    }
    row.update(load_count_summaries(root))
    for key in ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1", "Len"):
        b = maybe_float(base.get(key))
        r = maybe_float(repaired.get(key))
        row[key] = r
        row[f"base_{key}"] = b
        row[f"delta_{key}"] = None if b is None or r is None else float(r - b)
    return row


def feasible(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if float(row.get("delta_Recall") or 0.0) < -float(args.max_recall_drop):
        return False
    if float(row.get("delta_F1") or 0.0) < float(args.min_delta_f1):
        return False
    if float(row.get("delta_CHAIRi") or 0.0) > float(args.max_delta_chair_i):
        return False
    if float(row.get("delta_CHAIRs") or 0.0) > float(args.max_delta_chair_s):
        return False
    if int(row.get("changed", row.get("written", 0)) or 0) < int(args.min_changed):
        return False
    return True


def score_row(row: Dict[str, Any], objective: str) -> Tuple[float, ...]:
    d_f1 = float(row.get("delta_F1") or 0.0)
    d_recall = float(row.get("delta_Recall") or 0.0)
    d_ci = float(row.get("delta_CHAIRi") or 0.0)
    d_cs = float(row.get("delta_CHAIRs") or 0.0)
    selected = float(row.get("selected", 0) or 0)
    changed = float(row.get("changed", row.get("written", 0)) or 0)
    if objective == "chairi_then_f1":
        return (-d_ci, d_f1, -d_cs, d_recall, changed, -selected)
    if objective == "chairs_then_f1":
        return (-d_cs, d_f1, -d_ci, d_recall, changed, -selected)
    if objective == "balanced":
        utility = d_f1 - 0.5 * max(0.0, -d_recall) - 0.25 * max(0.0, d_ci) - 0.1 * max(0.0, d_cs)
        return (utility, d_f1, -d_ci, -d_cs, d_recall, changed, -selected)
    return (d_f1, -d_ci, -d_cs, d_recall, changed, -selected)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select a validation threshold from CHAIR suppression sweep outputs.")
    ap.add_argument("--threshold_dir", action="append", default=[], help="Either /path/to/root or threshold::/path/to/root")
    ap.add_argument("--root_glob", type=str, default="", help="Glob over threshold output roots, e.g. out/val_sweep/yp*")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--objective", choices=["f1_then_chair", "chairi_then_f1", "chairs_then_f1", "balanced"], default="f1_then_chair")
    ap.add_argument("--max_recall_drop", type=float, default=0.002)
    ap.add_argument("--min_delta_f1", type=float, default=0.0)
    ap.add_argument("--max_delta_chair_i", type=float, default=0.0)
    ap.add_argument("--max_delta_chair_s", type=float, default=0.0)
    ap.add_argument("--min_changed", type=int, default=1)
    args = ap.parse_args()

    specs = list(args.threshold_dir)
    if str(args.root_glob or "").strip():
        specs.extend(sorted(glob.glob(str(args.root_glob))))
    if not specs:
        raise SystemExit("No threshold dirs supplied.")

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for spec in specs:
        try:
            rows.append(row_from_root(spec))
        except Exception as exc:
            errors.append({"spec": spec, "error": str(exc)})
    rows.sort(key=lambda r: float(r["threshold"]))
    candidates = [row for row in rows if feasible(row, args)]
    relaxed = False
    if not candidates:
        candidates = rows
        relaxed = True
    if not candidates:
        raise SystemExit("No valid threshold summaries found.")
    best = max(candidates, key=lambda r: score_row(r, str(args.objective)))
    write_json(
        args.out_json,
        {
            "selected_threshold": float(best["threshold"]),
            "selected": best,
            "relaxed_constraints": bool(relaxed),
            "constraints": {
                "objective": str(args.objective),
                "max_recall_drop": float(args.max_recall_drop),
                "min_delta_f1": float(args.min_delta_f1),
                "max_delta_chair_i": float(args.max_delta_chair_i),
                "max_delta_chair_s": float(args.max_delta_chair_s),
                "min_changed": int(args.min_changed),
            },
            "all_rows": rows,
            "errors": errors,
        },
    )
    print("[selected]", best["threshold"], best)
    print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()

