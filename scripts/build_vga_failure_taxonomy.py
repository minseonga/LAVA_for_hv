#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


RELATION_KEYWORDS = [
    "left",
    "right",
    "behind",
    "in front of",
    "front of",
    "next to",
    "beside",
    "near",
    "under",
    "below",
    "above",
    "on top of",
    "between",
]

CONTEXT_KEYWORDS = [
    "with",
    "without",
    "wearing",
    "holding",
    "made of",
    "color",
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "small",
    "large",
    "tall",
    "short",
]


def parse_yes_no(text: str) -> str:
    t = (text or "").strip().lower()
    m = re.match(r"^(yes|no)\b", t)
    if m:
        return m.group(1)
    if re.search(r"\b(no|not)\b", t):
        return "no"
    if re.search(r"\byes\b", t):
        return "yes"
    return "yes"


def load_pred(path_jsonl: str, pred_text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    mode = (pred_text_key or "auto").strip().lower()
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r.get("question_id", "")).strip()
            if not qid:
                continue
            txt = ""
            if mode == "text":
                txt = r.get("text", "")
            elif mode == "output":
                txt = r.get("output", "")
            else:
                txt = r.get("text", "")
                if not str(txt).strip():
                    txt = r.get("output", "")
                if not str(txt).strip():
                    txt = r.get("answer", "")
            out[qid] = parse_yes_no(txt)
    return out


def load_gt(path_csv: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            qid = str(r.get("id", "")).strip()
            ans = str(r.get("answer", "")).strip().lower()
            if not qid or ans not in {"yes", "no"}:
                continue
            out[qid] = {
                "answer": ans,
                "question": str(r.get("question", "")).strip(),
                "image_id": str(r.get("image_id", "")).strip(),
                "category": str(r.get("category", "")).strip(),
            }
    return out


def extract_object_phrase(question: str) -> Tuple[str, bool]:
    q = (question or "").strip().lower().rstrip("?").strip()
    m = re.match(r"^is there (?:a|an)\s+(.+?)\s+in the image$", q)
    if m:
        return m.group(1).strip(), True
    m2 = re.match(r"^is there\s+(.+?)\s+in the image$", q)
    if m2:
        return m2.group(1).strip(), False
    return "", False


def has_any(text: str, kws: List[str]) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in kws)


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Build VGA failure taxonomy vs vanilla baseline on POPE.")
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--vga_pred_jsonl", type=str, required=True)
    ap.add_argument("--baseline_pred_text_key", type=str, default="auto", choices=["auto", "text", "output"])
    ap.add_argument("--vga_pred_text_key", type=str, default="auto", choices=["auto", "text", "output"])
    ap.add_argument("--object_prior_thr", type=float, default=0.55)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    gt = load_gt(args.gt_csv)
    pred_base = load_pred(args.baseline_pred_jsonl, args.baseline_pred_text_key)
    pred_vga = load_pred(args.vga_pred_jsonl, args.vga_pred_text_key)

    # Object prior from GT labels.
    obj_yes = Counter()
    obj_total = Counter()
    qid_obj = {}
    qid_simple = {}
    for qid, g in gt.items():
        obj, is_simple = extract_object_phrase(g["question"])
        qid_obj[qid] = obj
        qid_simple[qid] = is_simple
        if obj:
            obj_total[obj] += 1
            if g["answer"] == "yes":
                obj_yes[obj] += 1

    def obj_prior(obj: str) -> float:
        if not obj or obj_total[obj] == 0:
            return 0.5
        return float(obj_yes[obj] / obj_total[obj])

    per_case = []
    regressions = []
    counter_case = Counter()
    counter_tax = Counter()
    by_tax_category = defaultdict(Counter)
    n_common = 0
    for qid, g in gt.items():
        pb = pred_base.get(qid)
        pv = pred_vga.get(qid)
        if pb is None or pv is None:
            continue
        n_common += 1
        y = g["answer"]
        base_ok = pb == y
        vga_ok = pv == y
        if base_ok and vga_ok:
            ctype = "both_correct"
        elif (not base_ok) and (not vga_ok):
            ctype = "both_wrong"
        elif base_ok and (not vga_ok):
            ctype = "vga_regression"
        else:
            ctype = "vga_improvement"
        counter_case[ctype] += 1

        q = g["question"]
        obj = qid_obj.get(qid, "")
        prior = obj_prior(obj)
        simple = bool(qid_simple.get(qid, False))
        rel = has_any(q, RELATION_KEYWORDS)
        ctx = (not simple) or has_any(q, CONTEXT_KEYWORDS)

        absence_sensitive = bool((ctype == "vga_regression") and (y == "no") and (pb == "no") and (pv == "yes"))
        relation_dependent = bool(ctype == "vga_regression" and rel)
        context_dependent = bool(ctype == "vga_regression" and ctx)
        object_prior_misleading = bool(ctype == "vga_regression" and (y == "no") and (pv == "yes") and (prior >= args.object_prior_thr))

        row = {
            "id": qid,
            "category": g["category"],
            "image_id": g["image_id"],
            "question": q,
            "object_phrase": obj,
            "object_yes_prior": f"{prior:.6f}",
            "object_count": int(obj_total[obj]) if obj else 0,
            "gt": y,
            "pred_baseline": pb,
            "pred_vga": pv,
            "baseline_ok": int(base_ok),
            "vga_ok": int(vga_ok),
            "case_type": ctype,
            "absence_sensitive": int(absence_sensitive),
            "context_dependent": int(context_dependent),
            "relation_dependent": int(relation_dependent),
            "object_prior_misleading": int(object_prior_misleading),
        }
        per_case.append(row)

        if ctype == "vga_regression":
            regressions.append(row)
            if absence_sensitive:
                counter_tax["absence_sensitive"] += 1
                by_tax_category["absence_sensitive"][g["category"]] += 1
            if context_dependent:
                counter_tax["context_dependent"] += 1
                by_tax_category["context_dependent"][g["category"]] += 1
            if relation_dependent:
                counter_tax["relation_dependent"] += 1
                by_tax_category["relation_dependent"][g["category"]] += 1
            if object_prior_misleading:
                counter_tax["object_prior_misleading"] += 1
                by_tax_category["object_prior_misleading"][g["category"]] += 1

    os.makedirs(args.out_dir, exist_ok=True)
    per_case_csv = os.path.join(args.out_dir, "per_case_compare.csv")
    reg_csv = os.path.join(args.out_dir, "vga_regression_cases.csv")
    tax_csv = os.path.join(args.out_dir, "taxonomy_counts.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")

    fields = [
        "id",
        "category",
        "image_id",
        "question",
        "object_phrase",
        "object_yes_prior",
        "object_count",
        "gt",
        "pred_baseline",
        "pred_vga",
        "baseline_ok",
        "vga_ok",
        "case_type",
        "absence_sensitive",
        "context_dependent",
        "relation_dependent",
        "object_prior_misleading",
    ]
    write_csv(per_case_csv, per_case, fields)
    write_csv(reg_csv, regressions, fields)

    tax_rows = []
    for k in ["absence_sensitive", "context_dependent", "relation_dependent", "object_prior_misleading"]:
        tax_rows.append(
            {
                "taxonomy": k,
                "count": int(counter_tax.get(k, 0)),
                "ratio_in_regression": float(counter_tax.get(k, 0) / max(1, len(regressions))),
                "adversarial": int(by_tax_category[k].get("adversarial", 0)),
                "popular": int(by_tax_category[k].get("popular", 0)),
                "random": int(by_tax_category[k].get("random", 0)),
            }
        )
    write_csv(
        tax_csv,
        tax_rows,
        ["taxonomy", "count", "ratio_in_regression", "adversarial", "popular", "random"],
    )

    summary = {
        "inputs": {
            "gt_csv": os.path.abspath(args.gt_csv),
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "vga_pred_jsonl": os.path.abspath(args.vga_pred_jsonl),
            "baseline_pred_text_key": args.baseline_pred_text_key,
            "vga_pred_text_key": args.vga_pred_text_key,
            "object_prior_thr": args.object_prior_thr,
        },
        "counts": {
            "n_gt": int(len(gt)),
            "n_pred_baseline": int(len(pred_base)),
            "n_pred_vga": int(len(pred_vga)),
            "n_common": int(n_common),
            "both_correct": int(counter_case["both_correct"]),
            "both_wrong": int(counter_case["both_wrong"]),
            "vga_regression": int(counter_case["vga_regression"]),
            "vga_improvement": int(counter_case["vga_improvement"]),
        },
        "taxonomy_counts": {k: int(counter_tax.get(k, 0)) for k in ["absence_sensitive", "context_dependent", "relation_dependent", "object_prior_misleading"]},
        "outputs": {
            "per_case_csv": per_case_csv,
            "vga_regression_csv": reg_csv,
            "taxonomy_counts_csv": tax_csv,
            "summary_json": summary_json,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", per_case_csv)
    print("[saved]", reg_csv)
    print("[saved]", tax_csv)
    print("[saved]", summary_json)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
