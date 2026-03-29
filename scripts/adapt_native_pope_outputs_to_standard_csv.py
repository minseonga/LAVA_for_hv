#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def normalize_yesno_from_text(text: str) -> str:
    s = str(text).strip()
    low = s.lower()
    tokens = low.replace(".", " ").replace(",", " ").split()
    if "no" in tokens or "not" in tokens or "n't" in low:
        return "no"
    return "yes"


def label_to_yesno(x: Any) -> str:
    s = str(x).strip().lower()
    if s in {"1", "yes", "true"}:
        return "yes"
    return "no"


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Adapt native VISTA/VHR/EAZY POPE outputs to a standard CSV schema.")
    ap.add_argument("--source", type=str, required=True, choices=["vista", "vhr", "eazy"])
    ap.add_argument("--pred_jsonl", type=str, required=True)
    ap.add_argument("--input_jsonl", type=str, required=True, help="Reference POPE jsonl used for that run.")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    source = str(args.source).strip().lower()
    pred_rows = read_jsonl(os.path.abspath(args.pred_jsonl))
    in_rows = read_jsonl(os.path.abspath(args.input_jsonl))

    by_qid: Dict[str, Dict[str, Any]] = {str(r.get("question_id")): r for r in in_rows if r.get("question_id") is not None}

    out: List[Dict[str, Any]] = []

    if source == "vhr":
        for p in pred_rows:
            qid = str(p.get("question_id"))
            src = by_qid.get(qid, {})
            question = str(p.get("question", src.get("text", "")))
            answer_gt = label_to_yesno(p.get("gt", src.get("label", "")))
            pred_text = str(p.get("text", ""))
            pred_eval = normalize_yesno_from_text(pred_text)
            image = str(src.get("image", ""))
            out.append(
                {
                    "source": "vhr",
                    "id": qid,
                    "image": image,
                    "question": question,
                    "answer_gt": answer_gt,
                    "pred_text": pred_text,
                    "pred_answer_eval": pred_eval,
                    "is_correct": int(pred_eval == answer_gt),
                }
            )
    else:
        # vista/eazy: usually no stable qid in output -> align by order.
        n = min(len(pred_rows), len(in_rows))
        for i in range(n):
            p = pred_rows[i]
            src = in_rows[i]
            qid = str(src.get("question_id", i + 1))
            question = str(src.get("text", p.get("query", "")))
            answer_gt = label_to_yesno(src.get("label", p.get("label", "")))
            pred_text = str(p.get("ans", p.get("text", "")))
            pred_eval = normalize_yesno_from_text(pred_text)
            image = str(src.get("image", ""))
            out.append(
                {
                    "source": source,
                    "id": qid,
                    "image": image,
                    "question": question,
                    "answer_gt": answer_gt,
                    "pred_text": pred_text,
                    "pred_answer_eval": pred_eval,
                    "is_correct": int(pred_eval == answer_gt),
                }
            )

    write_csv(os.path.abspath(args.out_csv), out)
    summary = {
        "inputs": {
            "source": source,
            "pred_jsonl": os.path.abspath(args.pred_jsonl),
            "input_jsonl": os.path.abspath(args.input_jsonl),
        },
        "counts": {
            "n_pred_rows": len(pred_rows),
            "n_input_rows": len(in_rows),
            "n_out_rows": len(out),
        },
        "outputs": {
            "out_csv": os.path.abspath(args.out_csv),
        },
    }
    sp = os.path.splitext(os.path.abspath(args.out_csv))[0] + "_summary.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", sp)


if __name__ == "__main__":
    main()

