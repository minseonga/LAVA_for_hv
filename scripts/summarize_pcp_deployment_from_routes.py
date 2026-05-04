#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict


def safe_id(value: Any) -> str:
    return str("" if value is None else value).strip()


def parse_yes_no(text: Any) -> str:
    s = str("" if text is None else text).strip()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip().lower() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def pick_text(row: Dict[str, Any], key: str) -> str:
    mode = str(key or "auto").strip().lower()
    if mode != "auto":
        return str(row.get(mode, "")).strip()
    for k in ("text", "output", "answer", "caption"):
        text = str(row.get(k, "")).strip()
        if text:
            return text
    return ""


def load_gt(path: str, id_col: str, label_col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            qid = safe_id(row.get(id_col) or row.get("question_id"))
            label = safe_id(row.get(label_col)).lower()
            if qid and label in {"yes", "no"}:
                out[qid] = label
    return out


def load_pred(path: str, key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = safe_id(row.get("question_id") or row.get("id"))
            if not qid or qid.lower() in {"none", "null", "nan"}:
                continue
            out[qid] = parse_yes_no(pick_text(row, key))
    return out


def load_routes(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            qid = safe_id(row.get("id") or row.get("question_id"))
            if qid:
                out[qid] = safe_id(row.get("route")).lower()
    return out


def init_binary_counts() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def update_binary_counts(counts: Dict[str, int], pred: str, gold: str) -> None:
    if pred == "yes" and gold == "yes":
        counts["tp"] += 1
    elif pred == "yes" and gold == "no":
        counts["fp"] += 1
    elif pred == "no" and gold == "yes":
        counts["fn"] += 1
    elif pred == "no" and gold == "no":
        counts["tn"] += 1


def safe_div(num: float, den: float) -> float:
    return float(num / den) if float(den) else 0.0


def binary_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tn = int(counts["tn"])
    n = tp + fp + fn + tn
    precision = safe_div(float(tp), float(tp + fp))
    recall = safe_div(float(tp), float(tp + fn))
    return {
        "n": float(n),
        "acc": safe_div(float(tp + tn), float(n)),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(float(2.0 * precision * recall), float(precision + recall)),
        "yes_ratio": safe_div(float(tp + fp), float(n)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize full POPE deployment metrics from partial PCP route rows.")
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--route_rows_csv", required=True)
    ap.add_argument("--baseline_pred_text_key", default="auto", choices=["auto", "text", "output", "answer", "caption"])
    ap.add_argument("--intervention_pred_text_key", default="auto", choices=["auto", "text", "output", "answer", "caption"])
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--label_col", default="answer")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    gt = load_gt(os.path.abspath(args.gt_csv), str(args.id_col), str(args.label_col))
    baseline = load_pred(os.path.abspath(args.baseline_pred_jsonl), str(args.baseline_pred_text_key))
    intervention = load_pred(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_text_key))
    routes = load_routes(os.path.abspath(args.route_rows_csv))

    n = 0
    base_correct_total = 0
    int_correct_total = 0
    final_correct_total = 0
    base_counts = init_binary_counts()
    int_counts = init_binary_counts()
    final_counts = init_binary_counts()
    total_harm = 0
    total_help = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0
    selected_count = 0
    actual_fallback = 0
    flagged_unchanged = 0

    for qid, answer in gt.items():
        b = baseline.get(qid, "")
        i = intervention.get(qid, "")
        if b not in {"yes", "no"} or i not in {"yes", "no"}:
            continue
        n += 1
        bc = int(b == answer)
        ic = int(i == answer)
        base_correct_total += bc
        int_correct_total += ic
        update_binary_counts(base_counts, b, answer)
        update_binary_counts(int_counts, i, answer)
        harm = int(bc == 1 and ic == 0)
        help_ = int(bc == 0 and ic == 1)
        total_harm += harm
        total_help += help_

        route = routes.get(qid, "method")
        use_baseline = route == "baseline"
        final_label = b if use_baseline else i
        update_binary_counts(final_counts, final_label, answer)
        if use_baseline:
            selected_count += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int(harm == 0 and help_ == 0)
            actual_fallback += int(b != i)
            flagged_unchanged += int(b == i)
            final_correct_total += bc
        else:
            final_correct_total += ic

    base_metrics = binary_metrics(base_counts)
    int_metrics = binary_metrics(int_counts)
    final_metrics = binary_metrics(final_counts)

    summary = {
        "n": int(n),
        "baseline_acc": base_correct_total / n if n else 0.0,
        "baseline_precision": base_metrics["precision"],
        "baseline_recall": base_metrics["recall"],
        "baseline_f1": base_metrics["f1"],
        "baseline_yes_ratio": base_metrics["yes_ratio"],
        "intervention_acc": int_correct_total / n if n else 0.0,
        "intervention_precision": int_metrics["precision"],
        "intervention_recall": int_metrics["recall"],
        "intervention_f1": int_metrics["f1"],
        "intervention_yes_ratio": int_metrics["yes_ratio"],
        "pcp_deploy_acc": final_correct_total / n if n else 0.0,
        "pcp_deploy_precision": final_metrics["precision"],
        "pcp_deploy_recall": final_metrics["recall"],
        "pcp_deploy_f1": final_metrics["f1"],
        "pcp_deploy_yes_ratio": final_metrics["yes_ratio"],
        "delta_vs_intervention": (final_correct_total - int_correct_total) / n if n else 0.0,
        "delta_f1_vs_intervention": final_metrics["f1"] - int_metrics["f1"],
        "delta_f1_vs_baseline": final_metrics["f1"] - base_metrics["f1"],
        "confusion": {
            "baseline": {k: int(v) for k, v in base_counts.items()},
            "intervention": {k: int(v) for k, v in int_counts.items()},
            "pcp_deploy": {k: int(v) for k, v in final_counts.items()},
        },
        "baseline_generated": int(selected_count),
        "actual_fallback": int(actual_fallback),
        "flagged_unchanged": int(flagged_unchanged),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "net": int(selected_harm - selected_help),
    }

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[saved]", out_json)


if __name__ == "__main__":
    main()
