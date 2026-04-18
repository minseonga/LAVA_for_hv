#!/usr/bin/env python
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def safe_id(x) -> str:
    return str(x).strip()


def load_gt(path_csv: str, id_col: str, label_col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            k = safe_id(r.get(id_col))
            v = safe_id(r.get(label_col)).lower()
            if v in {"yes", "no"}:
                out[k] = v
    return out


def load_pred(path_jsonl: str, pred_text_key: str = "auto") -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qraw = r.get("question_id")
            if qraw is None:
                continue
            qid = safe_id(qraw)
            ql = qid.lower()
            if (not qid) or (ql in {"none", "null", "nan"}):
                continue
            txt = ""
            mode = (pred_text_key or "auto").strip().lower()
            if mode == "text":
                txt = r.get("text", "")
            elif mode == "output":
                txt = r.get("output", "")
            else:
                # auto: prefer text, then output, then answer
                txt = r.get("text", "")
                if not str(txt).strip():
                    txt = r.get("output", "")
                if not str(txt).strip():
                    txt = r.get("answer", "")
            out[qid] = parse_yes_no(txt)
    return out


def eval_conf(gt: Dict[str, str], pred: Dict[str, str]) -> Tuple[dict, list]:
    tp = fp = tn = fn = 0
    missing = []
    for k, y in gt.items():
        p = pred.get(k)
        if p not in {"yes", "no"}:
            missing.append(k)
            continue
        if y == "yes" and p == "yes":
            tp += 1
        elif y == "no" and p == "yes":
            fp += 1
        elif y == "no" and p == "no":
            tn += 1
        elif y == "yes" and p == "no":
            fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    yes_ratio = (tp + fp) / n if n else 0.0
    return {
        "n": int(n),
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "yes_ratio": float(yes_ratio),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "missing_pred": int(len(missing)),
    }, missing


def eval_by_group(path_csv: str, id_col: str, label_col: str, group_col: str, pred: Dict[str, str]) -> Dict[str, dict]:
    grouped_gt: Dict[str, Dict[str, str]] = defaultdict(dict)
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            k = safe_id(r.get(id_col))
            y = safe_id(r.get(label_col)).lower()
            group = safe_id(r.get(group_col)).lower()
            if k and y in {"yes", "no"} and group:
                grouped_gt[group][k] = y
    out: Dict[str, dict] = {}
    for group in sorted(grouped_gt):
        metrics, _ = eval_conf(grouped_gt[group], pred)
        out[group] = metrics
    return out


def write_group_csv(path: str, overall: dict, by_group: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields: List[str] = ["group", "n", "acc", "f1", "precision", "recall", "yes_ratio", "TP", "FP", "TN", "FN", "missing_pred"]
    rows = [{"group": "overall", **overall}]
    rows.extend({"group": group, **metrics} for group, metrics in by_group.items())
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--pred_jsonl", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--label_col", type=str, default="answer")
    ap.add_argument("--group_col", type=str, default="")
    ap.add_argument("--pred_text_key", type=str, default="auto", choices=["auto", "text", "output"])
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    gt = load_gt(args.gt_csv, id_col=args.id_col, label_col=args.label_col)
    pred = load_pred(args.pred_jsonl, pred_text_key=args.pred_text_key)
    metrics, missing = eval_conf(gt, pred)
    by_group = eval_by_group(args.gt_csv, args.id_col, args.label_col, args.group_col, pred) if args.group_col else {}

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if by_group:
        print(json.dumps(by_group, ensure_ascii=False, indent=2))
    if missing:
        print(f"[warn] missing predictions for {len(missing)} ids")

    if args.out_json.strip():
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "inputs": {
                "gt_csv": os.path.abspath(args.gt_csv),
                "pred_jsonl": os.path.abspath(args.pred_jsonl),
                "id_col": args.id_col,
                "label_col": args.label_col,
                "pred_text_key": args.pred_text_key,
            },
            "metrics": metrics,
            "metrics_by_group": by_group,
            "missing_ids_preview": missing[:50],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("[saved]", out_path)
    if args.out_csv.strip():
        out_csv = os.path.abspath(args.out_csv)
        write_group_csv(out_csv, metrics, by_group)
        print("[saved]", out_csv)


if __name__ == "__main__":
    main()
