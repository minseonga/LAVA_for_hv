#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


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


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def load_gt(path: str, id_col: str, label_col: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            qid = safe_id(row.get(id_col) or row.get("question_id"))
            label = safe_id(row.get(label_col)).lower()
            if qid and label in {"yes", "no"}:
                out[qid] = dict(row)
    return out


def load_pred(path: str, key: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_jsonl(path):
        qid = safe_id(row.get("question_id") or row.get("id"))
        if not qid or qid.lower() in {"none", "null", "nan"}:
            continue
        text = pick_text(row, key)
        out[qid] = {
            "text": text,
            "label": parse_yes_no(text),
        }
    return out


def is_selected(mode: str, base_label: str, int_label: str) -> bool:
    if base_label not in {"yes", "no"} or int_label not in {"yes", "no"}:
        return False
    if mode == "changed_answer":
        return base_label != int_label
    if mode == "yes_to_no":
        return base_label == "yes" and int_label == "no"
    if mode == "no_to_yes":
        return base_label == "no" and int_label == "yes"
    raise ValueError(f"Unsupported mode={mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize POPE questions where baseline/intervention predictions differ.")
    ap.add_argument("--question_jsonl", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--baseline_pred_text_key", default="auto", choices=["auto", "text", "output", "answer", "caption"])
    ap.add_argument("--intervention_pred_text_key", default="auto", choices=["auto", "text", "output", "answer", "caption"])
    ap.add_argument("--mode", default="changed_answer", choices=["changed_answer", "yes_to_no", "no_to_yes"])
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--label_col", default="answer")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    question_rows = read_jsonl(os.path.abspath(args.question_jsonl))
    questions = {safe_id(r.get("question_id") or r.get("id")): r for r in question_rows}
    gt = load_gt(os.path.abspath(args.gt_csv), str(args.id_col), str(args.label_col))
    baseline = load_pred(os.path.abspath(args.baseline_pred_jsonl), str(args.baseline_pred_text_key))
    intervention = load_pred(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_text_key))

    selected_ids: List[str] = []
    audit_rows: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    effects: Counter[str] = Counter()

    for qid in questions:
        if qid not in gt:
            counts["missing_gt"] += 1
            continue
        b = baseline.get(qid, {})
        i = intervention.get(qid, {})
        base_label = b.get("label", "")
        int_label = i.get("label", "")
        if base_label not in {"yes", "no"} or int_label not in {"yes", "no"}:
            counts["missing_or_invalid_pred"] += 1
            continue
        changed = base_label != int_label
        counts["changed" if changed else "unchanged"] += 1
        if changed:
            transitions[f"{base_label}->{int_label}"] += 1

        answer = safe_id(gt[qid].get(args.label_col)).lower()
        base_correct = int(base_label == answer)
        int_correct = int(int_label == answer)
        effect = "neutral"
        if base_correct == 1 and int_correct == 0:
            effect = "harm"
        elif base_correct == 0 and int_correct == 1:
            effect = "help"
        if changed:
            effects[effect] += 1

        select = is_selected(str(args.mode), base_label, int_label)
        audit_rows.append(
            {
                "id": qid,
                "answer": answer,
                "baseline_label": base_label,
                "intervention_label": int_label,
                "baseline_text": b.get("text", ""),
                "intervention_text": i.get("text", ""),
                "gt_label": answer,
                "changed": int(changed),
                "selected": int(select),
                "effect": effect if changed else "",
                "harm": int(changed and effect == "harm"),
                "help": int(changed and effect == "help"),
                "neutral": int(changed and effect == "neutral"),
                "baseline_correct": base_correct,
                "intervention_correct": int_correct,
                "category": gt[qid].get("category", ""),
                "image_id": gt[qid].get("image_id", questions[qid].get("image_id", "")),
            }
        )
        if select:
            selected_ids.append(qid)

    selected_set = set(selected_ids)
    out_dir = os.path.abspath(args.out_dir)
    q_out = os.path.join(out_dir, "changed_q_with_object.jsonl")
    gt_out = os.path.join(out_dir, "changed_gt.csv")
    audit_out = os.path.join(out_dir, "changed_audit.csv")
    summary_out = os.path.join(out_dir, "summary.json")

    write_jsonl(q_out, (questions[qid] for qid in selected_ids if qid in questions))
    write_csv(gt_out, [gt[qid] for qid in selected_ids if qid in gt])
    write_csv(audit_out, audit_rows)

    summary = {
        "inputs": {
            "question_jsonl": os.path.abspath(args.question_jsonl),
            "gt_csv": os.path.abspath(args.gt_csv),
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "baseline_pred_text_key": args.baseline_pred_text_key,
            "intervention_pred_text_key": args.intervention_pred_text_key,
            "mode": args.mode,
        },
        "counts": {
            "n_questions": len(question_rows),
            "n_gt": len(gt),
            "n_selected": len(selected_set),
            **dict(counts),
        },
        "changed_transitions": dict(transitions),
        "changed_effects": dict(effects),
        "outputs": {
            "question_jsonl": q_out,
            "gt_csv": gt_out,
            "audit_csv": audit_out,
            "summary_json": summary_out,
        },
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
