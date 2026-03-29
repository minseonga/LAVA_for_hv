#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def parse_yes_no(text: str) -> str:
    s = (text or "").strip().lower()
    if s == "":
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    # POPE yes/no eval fallback: default yes
    return "yes"


def read_gt(path: str, id_col: str, label_col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get(id_col, "")).strip()
            y = str(r.get(label_col, "")).strip().lower()
            if sid and y in {"yes", "no"}:
                out[sid] = y
    return out


def read_split(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if path.strip() == "":
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id", "")).strip()
            sp = str(r.get("split", "")).strip().lower()
            if sid:
                out[sid] = sp if sp in {"calib", "eval"} else "all"
    return out


def pick_pred_text(obj: Dict[str, Any], pred_text_key: str) -> str:
    k = str(pred_text_key).strip().lower()
    if k == "text":
        return str(obj.get("text", ""))
    if k == "output":
        return str(obj.get("output", ""))
    if k == "answer":
        return str(obj.get("answer", ""))
    # auto
    for kk in ("text", "output", "answer"):
        v = str(obj.get(kk, "")).strip()
        if v != "":
            return v
    return ""


def read_pred(path: str, pred_text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == "":
                continue
            obj = json.loads(s)
            sid = str(obj.get("question_id", "")).strip()
            if sid == "":
                continue
            txt = pick_pred_text(obj, pred_text_key=pred_text_key)
            out[sid] = parse_yes_no(txt)
    return out


def confusion(gt: Dict[str, str], pred: Dict[str, str], ids: List[str]) -> Dict[str, Any]:
    tp = fp = tn = fn = miss = 0
    for sid in ids:
        y = gt.get(sid)
        p = pred.get(sid)
        if y not in {"yes", "no"}:
            continue
        if p not in {"yes", "no"}:
            miss += 1
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
    acc = (tp + tn) / float(n) if n > 0 else 0.0
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "n": int(n),
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "missing_pred": int(miss),
    }


def compare_changes(
    gt: Dict[str, str],
    base_pred: Dict[str, str],
    new_pred: Dict[str, str],
    ids: List[str],
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    changed = gain = harm = 0
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        y = gt.get(sid, "")
        b = base_pred.get(sid, "")
        n = new_pred.get(sid, "")
        b_ok = int(b in {"yes", "no"} and y in {"yes", "no"} and b == y)
        n_ok = int(n in {"yes", "no"} and y in {"yes", "no"} and n == y)
        if b != n:
            changed += 1
        if b_ok == 0 and n_ok == 1:
            gain += 1
            ctype = "repaired"
        elif b_ok == 1 and n_ok == 0:
            harm += 1
            ctype = "regressed"
        elif b_ok == 0 and n_ok == 0:
            ctype = "both_wrong"
        else:
            ctype = "both_correct"
        rows.append(
            {
                "id": sid,
                "answer": y,
                "base_pred": b,
                "new_pred": n,
                "base_correct": int(b_ok),
                "new_correct": int(n_ok),
                "change_type": ctype,
            }
        )
    return {
        "changed_pred": int(changed),
        "gain": int(gain),
        "harm": int(harm),
        "net_gain": int(gain - harm),
    }, rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline vs RF-HAR yes/no runs (overall + split).")
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--base_pred_jsonl", type=str, required=True)
    ap.add_argument("--new_pred_jsonl", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--label_col", type=str, default="answer")
    ap.add_argument("--pred_text_key", type=str, default="auto", choices=["auto", "text", "output", "answer"])
    ap.add_argument("--id_split_csv", type=str, default="")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_fail_csv", type=str, required=True)
    args = ap.parse_args()

    gt = read_gt(os.path.abspath(args.gt_csv), id_col=args.id_col, label_col=args.label_col)
    base_pred = read_pred(os.path.abspath(args.base_pred_jsonl), pred_text_key=args.pred_text_key)
    new_pred = read_pred(os.path.abspath(args.new_pred_jsonl), pred_text_key=args.pred_text_key)
    split_map = read_split(os.path.abspath(args.id_split_csv)) if str(args.id_split_csv).strip() != "" else {}

    ids_all = sorted(gt.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    ids_by_split: Dict[str, List[str]] = {"all": ids_all, "calib": [], "eval": []}
    if len(split_map) > 0:
        for sid in ids_all:
            sp = split_map.get(sid, "all")
            if sp in {"calib", "eval"}:
                ids_by_split[sp].append(sid)

    per_split = {}
    for sp, ids in ids_by_split.items():
        if sp in {"calib", "eval"} and len(ids) == 0:
            continue
        bmet = confusion(gt, base_pred, ids)
        nmet = confusion(gt, new_pred, ids)
        cmpm, _rows = compare_changes(gt, base_pred, new_pred, ids)
        per_split[sp] = {
            "baseline": bmet,
            "new": nmet,
            "delta": {
                "acc": float(nmet["acc"] - bmet["acc"]),
                "f1": float(nmet["f1"] - bmet["f1"]),
            },
            "changes": cmpm,
        }

    cmp_all, fail_rows = compare_changes(gt, base_pred, new_pred, ids_all)
    split_col = []
    if len(split_map) > 0:
        for r in fail_rows:
            sid = str(r.get("id", ""))
            rr = dict(r)
            rr["split"] = split_map.get(sid, "all")
            split_col.append(rr)
    else:
        for r in fail_rows:
            rr = dict(r)
            rr["split"] = "all"
            split_col.append(rr)

    write_csv(os.path.abspath(args.out_fail_csv), split_col)

    summary = {
        "inputs": {
            "gt_csv": os.path.abspath(args.gt_csv),
            "base_pred_jsonl": os.path.abspath(args.base_pred_jsonl),
            "new_pred_jsonl": os.path.abspath(args.new_pred_jsonl),
            "id_split_csv": (os.path.abspath(args.id_split_csv) if str(args.id_split_csv).strip() != "" else ""),
            "id_col": str(args.id_col),
            "label_col": str(args.label_col),
            "pred_text_key": str(args.pred_text_key),
        },
        "counts": {
            "n_gt": int(len(gt)),
            "n_pred_base": int(len(base_pred)),
            "n_pred_new": int(len(new_pred)),
        },
        "overall": per_split.get("all", {}),
        "splits": {k: v for k, v in per_split.items() if k != "all"},
        "change_counts": cmp_all,
        "outputs": {
            "out_json": os.path.abspath(args.out_json),
            "out_fail_csv": os.path.abspath(args.out_fail_csv),
        },
    }

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_json)
    print("[saved]", os.path.abspath(args.out_fail_csv))
    print("[summary]", json.dumps(summary["overall"].get("delta", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
