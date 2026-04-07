#!/usr/bin/env python
import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def safe_id(x) -> str:
    return str(x).strip()


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def load_gt(path_csv: str, id_col: str, label_col: str, category_col: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            qid = safe_id(row.get(id_col))
            ans = safe_id(row.get(label_col)).lower()
            category = safe_id(row.get(category_col)).lower()
            if ans in {"yes", "no"}:
                out[qid] = {
                    "answer": ans,
                    "category": category,
                }
    return out


def load_pred(path_jsonl: str, pred_text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    mode = (pred_text_key or "auto").strip().lower()
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qraw = row.get("question_id")
            if qraw is None:
                continue
            qid = safe_id(qraw)
            text = ""
            if mode == "text":
                text = row.get("text", "")
            elif mode == "output":
                text = row.get("output", "")
            else:
                text = row.get("text", "")
                if not str(text).strip():
                    text = row.get("output", "")
                if not str(text).strip():
                    text = row.get("answer", "")
                if not str(text).strip():
                    text = row.get("caption", "")
            out[qid] = parse_yes_no(text)
    return out


def load_routes(path_csv: str, id_col: str = "id", route_col: str = "route") -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            qid = safe_id(row.get(id_col))
            route = safe_id(row.get(route_col)).lower()
            out[qid] = route
    return out


def choose_final_pred(route: str, base_pred: str, int_pred: str) -> str:
    route = (route or "").strip().lower()
    if route == "baseline":
        return base_pred
    if route in {"method", "intervention", "keep"}:
        return int_pred
    return int_pred


def eval_conf(rows: List[Tuple[str, str]]) -> dict:
    tp = fp = tn = fn = 0
    missing = 0
    for gold, pred in rows:
        if pred is None:
            missing += 1
            continue
        if gold == "yes" and pred == "yes":
            tp += 1
        elif gold == "no" and pred == "yes":
            fp += 1
        elif gold == "no" and pred == "no":
            tn += 1
        elif gold == "yes" and pred == "no":
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
        "missing_pred": int(missing),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--meta_route_rows_csv", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--label_col", type=str, default="answer")
    ap.add_argument("--category_col", type=str, default="category")
    ap.add_argument("--baseline_pred_text_key", type=str, default="auto", choices=["auto", "text", "output"])
    ap.add_argument("--intervention_pred_text_key", type=str, default="auto", choices=["auto", "text", "output"])
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    gt = load_gt(args.gt_csv, args.id_col, args.label_col, args.category_col)
    base_pred = load_pred(args.baseline_pred_jsonl, args.baseline_pred_text_key)
    int_pred = load_pred(args.intervention_pred_jsonl, args.intervention_pred_text_key)
    routes = load_routes(args.meta_route_rows_csv)

    categories = ["adversarial", "popular", "random"]
    grouped = {name: {"baseline": [], "intervention": [], "final": []} for name in categories + ["overall"]}
    route_counts = {name: {"baseline": 0, "method": 0, "other": 0} for name in categories + ["overall"]}

    for qid, row in gt.items():
        cat = row["category"]
        if cat not in grouped:
            grouped[cat] = {"baseline": [], "intervention": [], "final": []}
            route_counts[cat] = {"baseline": 0, "method": 0, "other": 0}

        gold = row["answer"]
        bpred = base_pred.get(qid)
        ipred = int_pred.get(qid)
        route = routes.get(qid, "method")
        fpred = choose_final_pred(route, bpred, ipred)

        grouped[cat]["baseline"].append((gold, bpred))
        grouped[cat]["intervention"].append((gold, ipred))
        grouped[cat]["final"].append((gold, fpred))
        grouped["overall"]["baseline"].append((gold, bpred))
        grouped["overall"]["intervention"].append((gold, ipred))
        grouped["overall"]["final"].append((gold, fpred))

        bucket = "other"
        if route == "baseline":
            bucket = "baseline"
        elif route in {"method", "intervention", "keep"}:
            bucket = "method"
        route_counts[cat][bucket] += 1
        route_counts["overall"][bucket] += 1

    summary = {
        "inputs": {
            "gt_csv": os.path.abspath(args.gt_csv),
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "meta_route_rows_csv": os.path.abspath(args.meta_route_rows_csv),
            "id_col": args.id_col,
            "label_col": args.label_col,
            "category_col": args.category_col,
            "baseline_pred_text_key": args.baseline_pred_text_key,
            "intervention_pred_text_key": args.intervention_pred_text_key,
        },
        "per_category": {},
    }

    csv_rows = []
    for category in categories + ["overall"]:
        base_metrics = eval_conf(grouped[category]["baseline"])
        int_metrics = eval_conf(grouped[category]["intervention"])
        final_metrics = eval_conf(grouped[category]["final"])
        counts = route_counts[category]

        payload = {
            "baseline": base_metrics,
            "intervention": int_metrics,
            "final": final_metrics,
            "route_counts": counts,
            "delta_final_vs_baseline_acc": final_metrics["acc"] - base_metrics["acc"],
            "delta_final_vs_intervention_acc": final_metrics["acc"] - int_metrics["acc"],
            "delta_final_vs_baseline_f1": final_metrics["f1"] - base_metrics["f1"],
            "delta_final_vs_intervention_f1": final_metrics["f1"] - int_metrics["f1"],
        }
        summary["per_category"][category] = payload

        csv_rows.append({
            "category": category,
            "baseline_acc": base_metrics["acc"],
            "baseline_f1": base_metrics["f1"],
            "intervention_acc": int_metrics["acc"],
            "intervention_f1": int_metrics["f1"],
            "final_acc": final_metrics["acc"],
            "final_f1": final_metrics["f1"],
            "delta_final_vs_baseline_acc": payload["delta_final_vs_baseline_acc"],
            "delta_final_vs_intervention_acc": payload["delta_final_vs_intervention_acc"],
            "delta_final_vs_baseline_f1": payload["delta_final_vs_baseline_f1"],
            "delta_final_vs_intervention_f1": payload["delta_final_vs_intervention_f1"],
            "route_baseline_count": counts["baseline"],
            "route_method_count": counts["method"],
            "route_other_count": counts["other"],
        })

    print(json.dumps(summary["per_category"], ensure_ascii=False, indent=2))

    if args.out_json.strip():
        out_json = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[saved]", out_json)

    if args.out_csv.strip():
        out_csv = os.path.abspath(args.out_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            wr.writeheader()
            wr.writerows(csv_rows)
        print("[saved]", out_csv)


if __name__ == "__main__":
    main()
