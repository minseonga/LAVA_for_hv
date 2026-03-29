#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from typing import Dict, Any


def parse_yes_no(text: str) -> str:
    s = (text or "").strip().lower()
    if s == "":
        return ""
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    if re.search(r"\b(no|not)\b", s):
        return "no"
    if re.search(r"\byes\b", s):
        return "yes"
    return "yes"


def pick_pred_text(rec: Dict[str, Any], mode: str) -> str:
    m = (mode or "auto").strip().lower()
    if m == "text":
        return str(rec.get("text", "")).strip()
    if m == "output":
        return str(rec.get("output", "")).strip()
    if m == "answer":
        return str(rec.get("answer", "")).strip()

    txt = str(rec.get("text", "")).strip()
    if txt:
        return txt
    txt = str(rec.get("output", "")).strip()
    if txt:
        return txt
    return str(rec.get("answer", "")).strip()


def read_gt(path_csv: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id", "")).strip()
            if sid == "":
                continue
            out[sid] = r
    return out


def read_pred(path_jsonl: str, pred_text_key: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r.get("question_id", "")).strip()
            if qid == "":
                continue
            txt = pick_pred_text(r, pred_text_key)
            yn = parse_yes_no(txt)
            out[qid] = {
                "pred_text": txt,
                "pred_answer_eval": yn,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build POPE-style samples_csv from subset_gt.csv + pred_jsonl.")
    ap.add_argument("--subset_gt_csv", type=str, required=True)
    ap.add_argument("--pred_jsonl", type=str, required=True)
    ap.add_argument("--pred_text_key", type=str, default="auto", choices=["auto", "text", "output", "answer"])
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary", type=str, default="")
    args = ap.parse_args()

    gt = read_gt(os.path.abspath(args.subset_gt_csv))
    pred = read_pred(os.path.abspath(args.pred_jsonl), pred_text_key=args.pred_text_key)

    rows = []
    n_missing_pred = 0
    n_valid = 0
    n_yes = 0
    n_no = 0
    n_fp = 0
    n_tp = 0
    n_tn = 0
    n_fn = 0

    for sid, g in gt.items():
        p = pred.get(sid)
        if p is None:
            n_missing_pred += 1
            continue

        answer = str(g.get("answer", "")).strip().lower()
        pred_yn = str(p.get("pred_answer_eval", "")).strip().lower()
        if answer not in {"yes", "no"} or pred_yn not in {"yes", "no"}:
            n_missing_pred += 1
            continue

        is_correct = int(pred_yn == answer)
        is_fp = int(pred_yn == "yes" and answer == "no")
        is_tp = int(pred_yn == "yes" and answer == "yes")
        is_tn = int(pred_yn == "no" and answer == "no")
        is_fn = int(pred_yn == "no" and answer == "yes")

        row = {
            "id": sid,
            "question": str(g.get("question", "")).strip(),
            "answer": answer,
            "image_id": str(g.get("image_id", "")).strip(),
            "category": str(g.get("category", "")).strip(),
            "group": str(g.get("group", "")).strip(),
            "pred_answer_eval": pred_yn,
            "pred_text": str(p.get("pred_text", "")).strip(),
            "is_correct": is_correct,
            "is_fp_hallucination": is_fp,
            "is_tp_yes": is_tp,
            "is_tn_no": is_tn,
            "is_fn_miss": is_fn,
        }
        rows.append(row)
        n_valid += 1
        if pred_yn == "yes":
            n_yes += 1
        else:
            n_no += 1
        n_fp += is_fp
        n_tp += is_tp
        n_tn += is_tn
        n_fn += is_fn

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "answer",
        "image_id",
        "category",
        "group",
        "pred_answer_eval",
        "pred_text",
        "is_correct",
        "is_fp_hallucination",
        "is_tp_yes",
        "is_tn_no",
        "is_fn_miss",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)

    summary = {
        "inputs": {
            "subset_gt_csv": os.path.abspath(args.subset_gt_csv),
            "pred_jsonl": os.path.abspath(args.pred_jsonl),
            "pred_text_key": args.pred_text_key,
        },
        "counts": {
            "n_gt": int(len(gt)),
            "n_pred": int(len(pred)),
            "n_valid_rows": int(n_valid),
            "n_missing_or_invalid_pred": int(n_missing_pred),
            "n_pred_yes": int(n_yes),
            "n_pred_no": int(n_no),
            "n_fp_hall": int(n_fp),
            "n_tp_yes": int(n_tp),
            "n_tn_no": int(n_tn),
            "n_fn_miss": int(n_fn),
        },
        "outputs": {
            "samples_csv": out_csv,
        },
    }

    out_summary = str(args.out_summary or "").strip()
    if out_summary != "":
        out_summary = os.path.abspath(out_summary)
        os.makedirs(os.path.dirname(out_summary), exist_ok=True)
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[saved]", out_summary)

    print("[saved]", out_csv)
    print("[summary]", json.dumps(summary["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()

