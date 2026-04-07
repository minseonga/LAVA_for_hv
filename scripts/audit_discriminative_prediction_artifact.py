#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_id(x: Any) -> str:
    return str(x).strip()


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def first_word(text: str) -> str:
    s = (text or "").strip().lower()
    if not s:
        return ""
    return s.split()[0]


def extract_text(row: Dict[str, Any], key: str) -> str:
    mode = (key or "auto").strip().lower()
    if mode == "text":
        return str(row.get("text", "") or "")
    if mode == "output":
        return str(row.get("output", "") or "")
    if mode == "answer":
        return str(row.get("answer", "") or "")
    txt = str(row.get("text", "") or "")
    if txt.strip():
        return txt
    txt = str(row.get("output", "") or "")
    if txt.strip():
        return txt
    return str(row.get("answer", "") or "")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def load_gt(path_csv: str, id_col: str = "id", label_col: str = "answer") -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            sid = safe_id(row.get(id_col))
            label = safe_id(row.get(label_col)).lower()
            if not sid or label not in {"yes", "no"}:
                continue
            out[sid] = {
                "gt_label": label,
                "question": safe_id(row.get("question")),
                "image_id": safe_id(row.get("image_id")),
                "category": safe_id(row.get("category")),
            }
    return out


def load_pred_rows(path_jsonl: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for row in read_jsonl(path_jsonl):
        sid = safe_id(row.get("question_id", row.get("id", "")))
        if not sid:
            continue
        if sid in by_id:
            duplicates.append(sid)
        by_id[sid] = row
    return by_id, duplicates


def eval_metrics(gt: Dict[str, Dict[str, str]], pred_rows: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    missing = 0
    pred_yes = 0
    empty_text = 0
    for sid, gt_row in gt.items():
        prow = pred_rows.get(sid)
        if prow is None:
            missing += 1
            continue
        txt = extract_text(prow, key)
        if not txt.strip():
            empty_text += 1
            missing += 1
            continue
        pred = parse_yes_no(txt)
        pred_yes += int(pred == "yes")
        gold = gt_row["gt_label"]
        if gold == "yes" and pred == "yes":
            tp += 1
        elif gold == "no" and pred == "yes":
            fp += 1
        elif gold == "no" and pred == "no":
            tn += 1
        elif gold == "yes" and pred == "no":
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": int(n),
        "acc": float((tp + tn) / float(max(1, n))),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "yes_ratio": float(pred_yes / float(max(1, n))),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "missing_pred": int(missing),
        "empty_text": int(empty_text),
    }


def summarize_field(pred_rows: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    texts = [extract_text(row, key) for row in pred_rows.values()]
    nonempty = [t for t in texts if str(t).strip()]
    parsed = [parse_yes_no(t) for t in nonempty]
    prefix = [first_word(t) for t in nonempty]
    exact_yes_no = [t.strip().lower() in {"yes", "no"} for t in nonempty]
    counts = Counter(parsed)
    top_outputs = Counter(t.strip().lower() for t in nonempty if t.strip()).most_common(10)
    word_lengths = [len(t.strip().split()) for t in nonempty]
    char_lengths = [len(t.strip()) for t in nonempty]
    explicit_prefix = [w in {"yes", "no"} for w in prefix]
    return {
        "n_rows": int(len(texts)),
        "n_nonempty": int(len(nonempty)),
        "nonempty_rate": float(len(nonempty) / float(max(1, len(texts)))),
        "parsed_yes_ratio": float(counts.get("yes", 0) / float(max(1, len(parsed)))),
        "explicit_yes_no_prefix_rate": float(sum(int(x) for x in explicit_prefix) / float(max(1, len(explicit_prefix)))),
        "exact_yes_no_only_rate": float(sum(int(x) for x in exact_yes_no) / float(max(1, len(exact_yes_no)))),
        "mean_word_len": float(mean(word_lengths)) if word_lengths else 0.0,
        "mean_char_len": float(mean(char_lengths)) if char_lengths else 0.0,
        "top_outputs": [{"text": text, "count": int(count)} for text, count in top_outputs],
    }


def compare_text_fields(pred_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    both = 0
    exact_match = 0
    parsed_match = 0
    output_nonempty = 0
    text_nonempty = 0
    for row in pred_rows.values():
        text = str(row.get("text", "") or "").strip()
        output = str(row.get("output", "") or "").strip()
        text_nonempty += int(bool(text))
        output_nonempty += int(bool(output))
        if text and output:
            both += 1
            exact_match += int(text == output)
            parsed_match += int(parse_yes_no(text) == parse_yes_no(output))
    return {
        "n_rows": int(len(pred_rows)),
        "text_nonempty_rate": float(text_nonempty / float(max(1, len(pred_rows)))),
        "output_nonempty_rate": float(output_nonempty / float(max(1, len(pred_rows)))),
        "both_nonempty_count": int(both),
        "exact_match_rate": float(exact_match / float(max(1, both))),
        "parsed_match_rate": float(parsed_match / float(max(1, both))),
    }


def build_pairwise_rows(
    gt: Dict[str, Dict[str, str]],
    baseline_rows: Dict[str, Dict[str, Any]],
    intervention_rows: Dict[str, Dict[str, Any]],
    baseline_key: str,
    intervention_key: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sid, gt_row in gt.items():
        brow = baseline_rows.get(sid)
        irow = intervention_rows.get(sid)
        if brow is None or irow is None:
            continue
        btxt = extract_text(brow, baseline_key)
        itxt = extract_text(irow, intervention_key)
        if not btxt.strip() or not itxt.strip():
            continue
        gold = gt_row["gt_label"]
        blabel = parse_yes_no(btxt)
        ilabel = parse_yes_no(itxt)
        bc = int(blabel == gold)
        ic = int(ilabel == gold)
        harm = int((bc == 1) and (ic == 0))
        help_ = int((bc == 0) and (ic == 1))
        both_correct = int((bc == 1) and (ic == 1))
        both_wrong = int((bc == 0) and (ic == 0))
        out.append(
            {
                "id": sid,
                "question": gt_row.get("question", ""),
                "image_id": gt_row.get("image_id", ""),
                "category": gt_row.get("category", ""),
                "gt_label": gold,
                "baseline_text": btxt,
                "intervention_text": itxt,
                "baseline_label": blabel,
                "intervention_label": ilabel,
                "baseline_correct": bc,
                "intervention_correct": ic,
                "harm": harm,
                "help": help_,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "baseline_word_len": len(btxt.strip().split()),
                "intervention_word_len": len(itxt.strip().split()),
                "baseline_prefix_yesno": int(first_word(btxt) in {"yes", "no"}),
                "intervention_prefix_yesno": int(first_word(itxt) in {"yes", "no"}),
            }
        )
    return out


def summarize_pairwise(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    harm = sum(int(r["harm"]) for r in rows)
    help_ = sum(int(r["help"]) for r in rows)
    both_correct = sum(int(r["both_correct"]) for r in rows)
    both_wrong = sum(int(r["both_wrong"]) for r in rows)
    bc = sum(int(r["baseline_correct"]) for r in rows)
    ic = sum(int(r["intervention_correct"]) for r in rows)
    return {
        "n_rows": int(n),
        "baseline_acc": float(bc / float(max(1, n))),
        "intervention_acc": float(ic / float(max(1, n))),
        "harm_rate": float(harm / float(max(1, n))),
        "help_rate": float(help_ / float(max(1, n))),
        "both_correct_rate": float(both_correct / float(max(1, n))),
        "both_wrong_rate": float(both_wrong / float(max(1, n))),
        "harm_count": int(harm),
        "help_count": int(help_),
        "both_correct_count": int(both_correct),
        "both_wrong_count": int(both_wrong),
    }


def sample_rows(rows: List[Dict[str, Any]], key: str, limit: int) -> List[Dict[str, Any]]:
    picked = [r for r in rows if int(r.get(key, 0)) == 1]
    picked.sort(key=lambda r: (-int(r["baseline_word_len"] + r["intervention_word_len"]), str(r["id"])))
    return picked[:limit]


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
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


def add_flag(flags: List[str], cond: bool, msg: str) -> None:
    if cond:
        flags.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit held-out/discovery yes-no prediction artifacts for discriminative interventions.")
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--baseline_name", type=str, default="baseline")
    ap.add_argument("--intervention_name", type=str, default="intervention")
    ap.add_argument("--baseline_pred_text_key", type=str, default="text")
    ap.add_argument("--intervention_pred_text_key", type=str, default="output")
    ap.add_argument("--discovery_gt_csv", type=str, default="")
    ap.add_argument("--discovery_baseline_pred_jsonl", type=str, default="")
    ap.add_argument("--discovery_intervention_pred_jsonl", type=str, default="")
    ap.add_argument("--discovery_baseline_pred_text_key", type=str, default="text")
    ap.add_argument("--discovery_intervention_pred_text_key", type=str, default="output")
    ap.add_argument("--examples_per_group", type=int, default=50)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    gt = load_gt(os.path.abspath(args.gt_csv))
    baseline_rows, baseline_dups = load_pred_rows(os.path.abspath(args.baseline_pred_jsonl))
    intervention_rows, intervention_dups = load_pred_rows(os.path.abspath(args.intervention_pred_jsonl))

    heldout_baseline_metrics = {
        key: eval_metrics(gt, baseline_rows, key)
        for key in ["text", "output", "auto"]
    }
    heldout_intervention_metrics = {
        key: eval_metrics(gt, intervention_rows, key)
        for key in ["text", "output", "auto"]
    }
    baseline_field_summary = summarize_field(baseline_rows, args.baseline_pred_text_key)
    intervention_field_summary = summarize_field(intervention_rows, args.intervention_pred_text_key)
    baseline_consistency = compare_text_fields(baseline_rows)
    intervention_consistency = compare_text_fields(intervention_rows)

    pairwise_rows = build_pairwise_rows(
        gt=gt,
        baseline_rows=baseline_rows,
        intervention_rows=intervention_rows,
        baseline_key=args.baseline_pred_text_key,
        intervention_key=args.intervention_pred_text_key,
    )
    pairwise_summary = summarize_pairwise(pairwise_rows)

    discovery_summary: Optional[Dict[str, Any]] = None
    flags: List[str] = []
    if args.discovery_gt_csv and args.discovery_baseline_pred_jsonl and args.discovery_intervention_pred_jsonl:
        d_gt = load_gt(os.path.abspath(args.discovery_gt_csv))
        d_base_rows, d_base_dups = load_pred_rows(os.path.abspath(args.discovery_baseline_pred_jsonl))
        d_int_rows, d_int_dups = load_pred_rows(os.path.abspath(args.discovery_intervention_pred_jsonl))
        discovery_summary = {
            "baseline_metrics": eval_metrics(d_gt, d_base_rows, args.discovery_baseline_pred_text_key),
            "intervention_metrics": eval_metrics(d_gt, d_int_rows, args.discovery_intervention_pred_text_key),
            "baseline_field_summary": summarize_field(d_base_rows, args.discovery_baseline_pred_text_key),
            "intervention_field_summary": summarize_field(d_int_rows, args.discovery_intervention_pred_text_key),
            "baseline_duplicates": int(len(d_base_dups)),
            "intervention_duplicates": int(len(d_int_dups)),
        }
        add_flag(
            flags,
            discovery_summary["intervention_metrics"]["acc"] > discovery_summary["baseline_metrics"]["acc"]
            and heldout_intervention_metrics[args.intervention_pred_text_key]["acc"] < heldout_baseline_metrics[args.baseline_pred_text_key]["acc"],
            "intervention_above_baseline_on_discovery_but_below_baseline_on_heldout",
        )
        add_flag(
            flags,
            abs(
                discovery_summary["intervention_metrics"]["yes_ratio"]
                - heldout_intervention_metrics[args.intervention_pred_text_key]["yes_ratio"]
            ) >= 0.08,
            "large_intervention_yes_ratio_shift_discovery_to_heldout",
        )

    add_flag(
        flags,
        heldout_intervention_metrics[args.intervention_pred_text_key]["acc"] < heldout_baseline_metrics[args.baseline_pred_text_key]["acc"],
        "heldout_intervention_below_baseline",
    )
    add_flag(flags, len(baseline_dups) > 0, "baseline_duplicate_question_ids")
    add_flag(flags, len(intervention_dups) > 0, "intervention_duplicate_question_ids")
    add_flag(
        flags,
        intervention_consistency["both_nonempty_count"] > 0 and intervention_consistency["parsed_match_rate"] < 1.0,
        "intervention_text_output_parse_mismatch",
    )
    add_flag(
        flags,
        intervention_field_summary["explicit_yes_no_prefix_rate"] < 0.95,
        "intervention_low_explicit_yes_no_prefix_rate",
    )
    add_flag(
        flags,
        abs(
            heldout_baseline_metrics[args.baseline_pred_text_key]["yes_ratio"]
            - heldout_intervention_metrics[args.intervention_pred_text_key]["yes_ratio"]
        ) >= 0.08,
        "large_heldout_yes_ratio_shift_vs_baseline",
    )

    harmful_examples = sample_rows(pairwise_rows, "harm", int(args.examples_per_group))
    helpful_examples = sample_rows(pairwise_rows, "help", int(args.examples_per_group))
    anomalous_examples = [
        row for row in pairwise_rows
        if (row["intervention_prefix_yesno"] == 0 or row["baseline_prefix_yesno"] == 0)
    ][: int(args.examples_per_group)]

    write_csv(os.path.join(out_dir, "pairwise_rows.csv"), pairwise_rows)
    write_csv(os.path.join(out_dir, "harm_examples.csv"), harmful_examples)
    write_csv(os.path.join(out_dir, "help_examples.csv"), helpful_examples)
    write_csv(os.path.join(out_dir, "parse_anomaly_examples.csv"), anomalous_examples)

    summary = {
        "inputs": {
            "gt_csv": os.path.abspath(args.gt_csv),
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "baseline_name": args.baseline_name,
            "intervention_name": args.intervention_name,
            "baseline_pred_text_key": args.baseline_pred_text_key,
            "intervention_pred_text_key": args.intervention_pred_text_key,
            "discovery_gt_csv": (os.path.abspath(args.discovery_gt_csv) if args.discovery_gt_csv else ""),
            "discovery_baseline_pred_jsonl": (
                os.path.abspath(args.discovery_baseline_pred_jsonl) if args.discovery_baseline_pred_jsonl else ""
            ),
            "discovery_intervention_pred_jsonl": (
                os.path.abspath(args.discovery_intervention_pred_jsonl) if args.discovery_intervention_pred_jsonl else ""
            ),
        },
        "heldout": {
            "baseline_metrics_by_key": heldout_baseline_metrics,
            "intervention_metrics_by_key": heldout_intervention_metrics,
            "baseline_field_summary": baseline_field_summary,
            "intervention_field_summary": intervention_field_summary,
            "baseline_text_output_consistency": baseline_consistency,
            "intervention_text_output_consistency": intervention_consistency,
            "baseline_duplicate_question_ids": int(len(baseline_dups)),
            "intervention_duplicate_question_ids": int(len(intervention_dups)),
            "pairwise_summary": pairwise_summary,
        },
        "discovery": discovery_summary,
        "flags": flags,
        "outputs": {
            "pairwise_rows_csv": os.path.join(out_dir, "pairwise_rows.csv"),
            "harm_examples_csv": os.path.join(out_dir, "harm_examples.csv"),
            "help_examples_csv": os.path.join(out_dir, "help_examples.csv"),
            "parse_anomaly_examples_csv": os.path.join(out_dir, "parse_anomaly_examples.csv"),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print("[saved]", os.path.join(out_dir, "summary.json"))
    print("[saved]", os.path.join(out_dir, "pairwise_rows.csv"))
    print("[saved]", os.path.join(out_dir, "harm_examples.csv"))
    print("[saved]", os.path.join(out_dir, "help_examples.csv"))
    print("[saved]", os.path.join(out_dir, "parse_anomaly_examples.csv"))


if __name__ == "__main__":
    main()
