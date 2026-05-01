#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


YES_NO = {"yes", "no"}


def safe_str(value: Any) -> str:
    return str("" if value is None else value).strip()


def object_name(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(safe_str(v) for v in value if safe_str(v))
    return safe_str(value)


def sortable_int(value: Any) -> int:
    try:
        return int(safe_str(value))
    except Exception:
        return 0


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def row_id(row: Mapping[str, Any]) -> str:
    return safe_str(row.get("question_id") or row.get("id"))


def pick_text(row: Mapping[str, Any], key: str) -> str:
    mode = safe_str(key).lower()
    if mode and mode != "auto":
        return safe_str(row.get(mode))
    for col in ("text", "output", "answer", "caption"):
        text = safe_str(row.get(col))
        if text:
            return text
    return ""


def parse_yes_no(text: Any) -> str:
    s = safe_str(text)
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip().lower() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    head = s.lower().lstrip()
    if re.match(r"^no\b", head):
        return "no"
    if re.match(r"^yes\b", head):
        return "yes"
    return "yes"


def maybe_float(value: Any) -> Optional[float]:
    s = safe_str(value)
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def infer_category(qid: str) -> str:
    try:
        i = int(qid)
    except Exception:
        return ""
    if 0 <= i < 3000:
        return "adversarial"
    if 3000 <= i < 6000:
        return "popular"
    if 6000 <= i < 9000:
        return "random"
    return ""


def load_gt(path: str, id_col: str, label_col: str, category_col: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv_rows(path):
        qid = safe_str(row.get(id_col) or row.get("question_id") or row.get("id"))
        label = safe_str(row.get(label_col) or row.get("answer") or row.get("label")).lower()
        if qid and label in YES_NO:
            out[qid] = {
                "gt_label": label,
                "category": safe_str(row.get(category_col) or row.get("category")).lower(),
            }
    return out


def load_questions(path: str) -> Dict[str, Dict[str, Any]]:
    if not safe_str(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = row_id(row)
        if not qid:
            continue
        out[qid] = {
            "question": safe_str(row.get("question") or row.get("text") or row.get("prompt")),
            "image": safe_str(row.get("image") or row.get("image_id")),
            "object": object_name(row.get("object") or row.get("obj") or row.get("target_object")),
            "category": safe_str(row.get("category")).lower(),
        }
    return out


def load_preds(path: str, text_key: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_jsonl(path):
        qid = row_id(row)
        if not qid:
            continue
        text = pick_text(row, text_key)
        out[qid] = {
            "text": text,
            "label": parse_yes_no(text),
        }
    return out


def load_route_rows(path: str) -> Dict[str, Dict[str, Any]]:
    if not safe_str(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv_rows(path):
        qid = safe_str(row.get("id") or row.get("question_id"))
        if qid:
            out[qid] = dict(row)
    return out


def classify_outcome(gt: str, baseline: str, intervention: str) -> Tuple[str, str]:
    if gt not in YES_NO or baseline not in YES_NO or intervention not in YES_NO:
        return "missing", "missing"
    bc = baseline == gt
    ic = intervention == gt
    transition = f"{baseline}->{intervention}"
    if bc and not ic:
        if transition == "yes->no" and gt == "yes":
            return "harm", "harm_false_deletion"
        if transition == "no->yes" and gt == "no":
            return "harm", "harm_false_insertion"
        return "harm", "harm_other"
    if (not bc) and ic:
        if transition == "yes->no" and gt == "no":
            return "help", "help_hallucination_suppression"
        if transition == "no->yes" and gt == "yes":
            return "help", "help_miss_recovery"
        return "help", "help_other"
    if bc and ic:
        return "neutral", "both_correct"
    return "neutral", "both_wrong"


def confusion_counts(rows: Sequence[Mapping[str, Any]], label_col: str) -> Dict[str, Any]:
    tp = fp = tn = fn = missing = 0
    for row in rows:
        gt = safe_str(row.get("gt_label"))
        pred = safe_str(row.get(label_col))
        if gt not in YES_NO or pred not in YES_NO:
            missing += 1
        elif gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n,
        "acc": (tp + tn) / n if n else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "yes_ratio": (tp + fp) / n if n else 0.0,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "missing_pred": missing,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    harm = sum(1 for row in rows if row.get("outcome") == "harm")
    help_ = sum(1 for row in rows if row.get("outcome") == "help")
    neutral = sum(1 for row in rows if row.get("outcome") == "neutral")
    changed = sum(1 for row in rows if row.get("changed_answer") == 1)
    selected = sum(1 for row in rows if row.get("route") == "baseline")
    selected_harm = sum(1 for row in rows if row.get("route") == "baseline" and row.get("outcome") == "harm")
    selected_help = sum(1 for row in rows if row.get("route") == "baseline" and row.get("outcome") == "help")
    return {
        "n": n,
        "changed": changed,
        "harm": harm,
        "help": help_,
        "neutral": neutral,
        "net_harm_minus_help": harm - help_,
        "selected_count": selected,
        "selected_harm": selected_harm,
        "selected_help": selected_help,
        "selected_net": selected_harm - selected_help,
        "selected_harm_precision": selected_harm / selected if selected else 0.0,
    }


def make_group_rows(rows: Sequence[Mapping[str, Any]], dims: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(safe_str(row.get(dim) or "unknown") for dim in dims)
        groups[key].append(row)
    out: List[Dict[str, Any]] = []
    for key, vals in groups.items():
        payload = {dim: key[i] for i, dim in enumerate(dims)}
        payload.update(summarize(vals))
        out.append(payload)
    return sorted(out, key=lambda r: (-int(r["harm"]), -int(r["help"]), -int(r["changed"]), tuple(str(r.get(dim, "")) for dim in dims)))


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def score_distribution_rows(rows: Sequence[Mapping[str, Any]], score_cols: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    buckets = ["harm", "help", "neutral"]
    for col in score_cols:
        if not any(maybe_float(row.get(col)) is not None for row in rows):
            continue
        for bucket in buckets:
            vals = [float(v) for row in rows if row.get("outcome") == bucket and (v := maybe_float(row.get(col))) is not None]
            out.append(
                {
                    "score": col,
                    "outcome": bucket,
                    "n": len(vals),
                    "mean": sum(vals) / len(vals) if vals else None,
                    "q10": quantile(vals, 0.10),
                    "q25": quantile(vals, 0.25),
                    "q50": quantile(vals, 0.50),
                    "q75": quantile(vals, 0.75),
                    "q90": quantile(vals, 0.90),
                }
            )
    return out


def short_text(value: Any, limit: int) -> str:
    text = " ".join(safe_str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description="Break down POPE harm/help cases by transition, split, object, and optional routing scores.")
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--question_jsonl", default="")
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--route_rows_csv", default="")
    ap.add_argument("--baseline_pred_text_key", default="auto", choices=["auto", "text", "output", "answer", "caption"])
    ap.add_argument("--intervention_pred_text_key", default="auto", choices=["auto", "text", "output", "answer", "caption"])
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--label_col", default="answer")
    ap.add_argument("--category_col", default="category")
    ap.add_argument("--infer_category_by_id", default="true", choices=["true", "false"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_examples_per_type", type=int, default=40)
    ap.add_argument("--text_limit", type=int, default=180)
    args = ap.parse_args()

    gt = load_gt(args.gt_csv, args.id_col, args.label_col, args.category_col)
    questions = load_questions(args.question_jsonl)
    baseline = load_preds(args.baseline_pred_jsonl, args.baseline_pred_text_key)
    intervention = load_preds(args.intervention_pred_jsonl, args.intervention_pred_text_key)
    routes = load_route_rows(args.route_rows_csv)
    infer_by_id = str(args.infer_category_by_id).lower() == "true"

    rows: List[Dict[str, Any]] = []
    for qid in sorted(gt, key=lambda x: (len(str(x)), str(x))):
        gt_row = gt[qid]
        q_row = questions.get(qid, {})
        b = baseline.get(qid, {"text": "", "label": ""})
        v = intervention.get(qid, {"text": "", "label": ""})
        category = safe_str(gt_row.get("category") or q_row.get("category")).lower()
        if not category and infer_by_id:
            category = infer_category(qid)
        baseline_label = safe_str(b.get("label"))
        intervention_label = safe_str(v.get("label"))
        gt_label = safe_str(gt_row.get("gt_label"))
        outcome, error_type = classify_outcome(gt_label, baseline_label, intervention_label)
        route = routes.get(qid, {})
        route_name = safe_str(route.get("route")).lower()
        row = {
            "id": qid,
            "category": category or "unknown",
            "image": safe_str(q_row.get("image") or route.get("image")),
            "object": safe_str(q_row.get("object")),
            "question": short_text(q_row.get("question") or route.get("question"), int(args.text_limit)),
            "gt_label": gt_label,
            "baseline_label": baseline_label,
            "intervention_label": intervention_label,
            "transition": f"{baseline_label}->{intervention_label}",
            "changed_answer": int(baseline_label in YES_NO and intervention_label in YES_NO and baseline_label != intervention_label),
            "baseline_correct": int(baseline_label == gt_label) if baseline_label in YES_NO else "",
            "intervention_correct": int(intervention_label == gt_label) if intervention_label in YES_NO else "",
            "outcome": outcome,
            "error_type": error_type,
            "route": route_name,
            "baseline_text": short_text(b.get("text"), int(args.text_limit)),
            "intervention_text": short_text(v.get("text"), int(args.text_limit)),
        }
        for col in ("score", "raw_score", "c_score", "d_score", "object_score", "fusion_score"):
            if col in route:
                row[col] = route.get(col)
        rows.append(row)

    changed_rows = [row for row in rows if int(row["changed_answer"]) == 1]
    harm_rows = [row for row in changed_rows if row["outcome"] == "harm"]
    help_rows = [row for row in changed_rows if row["outcome"] == "help"]

    summary = {
        "inputs": {
            "gt_csv": os.path.abspath(args.gt_csv),
            "question_jsonl": os.path.abspath(args.question_jsonl) if safe_str(args.question_jsonl) else "",
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "route_rows_csv": os.path.abspath(args.route_rows_csv) if safe_str(args.route_rows_csv) else "",
        },
        "overall": summarize(rows),
        "changed": summarize(changed_rows),
        "harm": summarize(harm_rows),
        "help": summarize(help_rows),
        "baseline_metrics": confusion_counts(rows, "baseline_label"),
        "intervention_metrics": confusion_counts(rows, "intervention_label"),
        "outcome_counts": dict(Counter(row["outcome"] for row in changed_rows)),
        "error_type_counts": dict(Counter(row["error_type"] for row in changed_rows)),
        "category_counts_changed": dict(Counter(row["category"] for row in changed_rows)),
        "transition_counts_changed": dict(Counter(row["transition"] for row in changed_rows)),
    }

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    segment_rows: List[Dict[str, Any]] = []
    for dims in (
        ("category",),
        ("transition",),
        ("error_type",),
        ("category", "transition"),
        ("category", "error_type"),
        ("transition", "error_type"),
    ):
        for row in make_group_rows(changed_rows, dims):
            segment_rows.append({"group": "+".join(dims), **row})

    object_rows = make_group_rows(changed_rows, ("object",))
    object_harm_rows = make_group_rows(harm_rows, ("object", "error_type"))
    score_rows = score_distribution_rows(changed_rows, ("score", "raw_score", "c_score", "d_score", "object_score", "fusion_score"))

    limit = int(args.max_examples_per_type)
    examples: List[Dict[str, Any]] = []
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in changed_rows:
        by_type[str(row["error_type"])].append(dict(row))
    for error_type in sorted(by_type):
        vals = by_type[error_type]
        vals = sorted(vals, key=lambda r: (safe_str(r.get("category")), safe_str(r.get("object")), sortable_int(r.get("id"))))
        for row in vals[:limit]:
            examples.append({"example_type": error_type, **row})

    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_csv(os.path.join(out_dir, "changed_rows.csv"), changed_rows)
    write_csv(os.path.join(out_dir, "harm_rows.csv"), harm_rows)
    write_csv(os.path.join(out_dir, "help_rows.csv"), help_rows)
    write_csv(os.path.join(out_dir, "segments.csv"), segment_rows)
    write_csv(os.path.join(out_dir, "object_segments.csv"), object_rows)
    write_csv(os.path.join(out_dir, "harm_object_segments.csv"), object_harm_rows)
    write_csv(os.path.join(out_dir, "score_distributions.csv"), score_rows)
    write_csv(os.path.join(out_dir, "examples_by_error_type.csv"), examples)
    print(json.dumps(summary["changed"], ensure_ascii=False, indent=2))
    print("[saved]", os.path.join(out_dir, "summary.json"))
    print("[saved]", os.path.join(out_dir, "segments.csv"))
    print("[saved]", os.path.join(out_dir, "harm_rows.csv"))


if __name__ == "__main__":
    main()
