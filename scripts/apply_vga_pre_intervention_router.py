#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import load_prediction_text_map, parse_yes_no, safe_id, write_json


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
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


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def load_gt_label_map(gt_csv: str, id_col: str = "id", label_col: str = "answer") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_csv_rows(gt_csv):
        sid = safe_id(row.get(id_col))
        label = safe_id(row.get(label_col)).lower()
        if sid and label in {"yes", "no"}:
            out[sid] = label
    return out


def load_amber_disc_gt_map(amber_root: str) -> Dict[str, str]:
    query_json = os.path.join(amber_root, "data", "query", "query_discriminative.json")
    annotations_json = os.path.join(amber_root, "data", "annotations.json")
    queries = read_json(query_json)
    annotations = read_json(annotations_json)
    ann_by_id = {safe_id(row.get("id")): row for row in annotations if safe_id(row.get("id"))}
    out: Dict[str, str] = {}
    for item in queries:
        sid = safe_id(item.get("id"))
        ann = ann_by_id.get(sid)
        if not sid or ann is None:
            continue
        label = safe_id(ann.get("truth")).lower()
        if label in {"yes", "no"}:
            out[sid] = label
    return out


def compute_metrics(gt_map: Dict[str, str], pred_map: Dict[str, str]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    missing = 0
    for sid, gt in gt_map.items():
        pred = pred_map.get(sid)
        if pred is None:
            missing += 1
            continue
        if gt == "yes" and pred == "yes":
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
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": int(n),
        "acc": float((tp + tn) / n) if n else 0.0,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "yes_ratio": float((tp + fp) / n) if n else 0.0,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "missing_pred": int(missing),
    }


def parse_anchor_yes(text: Any) -> int:
    return 1 if str(text or "").strip().lower() == "yes" else 0


def build_router_features(row: Dict[str, Any], router_meta: Dict[str, Any]) -> Dict[str, float]:
    frg_off = float(row["frg"])
    g_top5_mass = float(row.get("g_top5_mass", 0.0))
    probe_anchor_yes = float(parse_anchor_yes(row.get("probe_anchor", "")))
    tau = float(router_meta.get("tau", 0.0))
    return {
        "frg_off": frg_off,
        "g_top5_mass": g_top5_mass,
        "probe_anchor_yes": probe_anchor_yes,
        "abs_frg_to_tau": abs(frg_off - tau),
        "frg_x_mass": frg_off * g_top5_mass,
    }


def load_router(router_dir: str) -> Tuple[Any, Dict[str, Any]]:
    model_path = os.path.join(router_dir, "router_model.pkl")
    metadata_path = os.path.join(router_dir, "router_metadata.json")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def load_probe_rows(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id"))
        if sid:
            out[sid] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a pre-intervention VGA utility router on a held-out split.")
    ap.add_argument("--router_dir", type=str, required=True)
    ap.add_argument("--probe_features_csv", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, default="")
    ap.add_argument("--amber_root", type=str, default="")
    ap.add_argument("--baseline_pred_text_key", type=str, default="text")
    ap.add_argument("--intervention_pred_text_key", type=str, default="output")
    ap.add_argument("--benchmark_name", type=str, default="pope")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, meta = load_router(os.path.abspath(args.router_dir))
    feature_cols = [str(x) for x in meta.get("feature_cols", [])]
    cutoff = float(meta["deployment_cutoff"])
    probe_rows = load_probe_rows(os.path.abspath(args.probe_features_csv))
    baseline_map = load_prediction_text_map(os.path.abspath(args.baseline_pred_jsonl), args.baseline_pred_text_key)
    intervention_map = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), args.intervention_pred_text_key)
    if args.gt_csv:
        gt_map = load_gt_label_map(os.path.abspath(args.gt_csv))
    elif args.amber_root:
        gt_map = load_amber_disc_gt_map(os.path.abspath(args.amber_root))
    else:
        raise ValueError("Need either --gt_csv or --amber_root.")

    rows: List[Dict[str, Any]] = []
    pred_baseline: Dict[str, str] = {}
    pred_intervention: Dict[str, str] = {}
    pred_final: Dict[str, str] = {}
    route_method = 0
    route_baseline = 0
    selected_help = 0
    selected_harm = 0
    selected_neutral = 0

    for sid, gt in gt_map.items():
        probe = probe_rows.get(sid)
        baseline_text = str(baseline_map.get(sid, "")).strip()
        intervention_text = str(intervention_map.get(sid, "")).strip()
        if probe is None or not baseline_text or not intervention_text:
            continue

        feat_map = build_router_features(probe, meta)
        feat_vec = [[float(feat_map[col]) for col in feature_cols]]
        utility_score = float(model.predict(feat_vec)[0])
        route = "method" if utility_score >= cutoff else "baseline"
        baseline_label = parse_yes_no(baseline_text)
        intervention_label = parse_yes_no(intervention_text)
        final_label = intervention_label if route == "method" else baseline_label

        baseline_correct = int(baseline_label == gt)
        intervention_correct = int(intervention_label == gt)
        help_ = int((baseline_correct == 0) and (intervention_correct == 1))
        harm = int((baseline_correct == 1) and (intervention_correct == 0))
        utility_true = int(intervention_correct - baseline_correct)
        if route == "method":
            route_method += 1
            selected_help += help_
            selected_harm += harm
            selected_neutral += int(utility_true == 0)
        else:
            route_baseline += 1

        pred_baseline[sid] = baseline_label
        pred_intervention[sid] = intervention_label
        pred_final[sid] = final_label
        row = {
            "id": sid,
            "gt_label": gt,
            "baseline_text": baseline_text,
            "intervention_text": intervention_text,
            "baseline_label": baseline_label,
            "intervention_label": intervention_label,
            "final_label": final_label,
            "baseline_correct": baseline_correct,
            "intervention_correct": intervention_correct,
            "final_correct": int(final_label == gt),
            "help": help_,
            "harm": harm,
            "utility_true": utility_true,
            "utility_score": utility_score,
            "deployment_cutoff": cutoff,
            "route": route,
            "route_method": int(route == "method"),
            "route_baseline": int(route == "baseline"),
        }
        for key, value in probe.items():
            if key == "id":
                continue
            row[f"probe_{key}"] = value
        rows.append(row)

    rows.sort(key=lambda r: int(str(r["id"])))
    decision_csv = os.path.join(args.out_dir, "decision_rows.csv")
    write_csv(decision_csv, rows)

    baseline_metrics = compute_metrics(gt_map, pred_baseline)
    intervention_metrics = compute_metrics(gt_map, pred_intervention)
    final_metrics = compute_metrics(gt_map, pred_final)

    summary = {
        "inputs": {
            "router_dir": os.path.abspath(args.router_dir),
            "probe_features_csv": os.path.abspath(args.probe_features_csv),
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "gt_csv": os.path.abspath(args.gt_csv) if args.gt_csv else "",
            "amber_root": os.path.abspath(args.amber_root) if args.amber_root else "",
            "benchmark_name": args.benchmark_name,
        },
        "router": {
            "backend_name": meta.get("backend_name", ""),
            "feature_variant": meta.get("feature_variant", ""),
            "feature_cols": feature_cols,
            "tau": meta.get("tau", None),
            "deployment_budget": meta.get("deployment_budget", None),
            "deployment_cutoff": cutoff,
            "score_policy": meta.get("score_policy", ""),
        },
        "evaluation": {
            "n_eval": int(len(rows)),
            "method_rate": float(route_method / float(max(1, len(rows)))),
            "baseline_rate": float(route_baseline / float(max(1, len(rows)))),
            "selected_help": int(selected_help),
            "selected_harm": int(selected_harm),
            "selected_neutral": int(selected_neutral),
            "selected_help_precision": float(selected_help / float(max(1, route_method))),
            "selected_harm_precision": float(selected_harm / float(max(1, route_method))),
            "baseline_metrics": baseline_metrics,
            "intervention_metrics": intervention_metrics,
            "final_metrics": final_metrics,
            "delta_vs_baseline_acc": float(final_metrics["acc"] - baseline_metrics["acc"]),
            "delta_vs_intervention_acc": float(final_metrics["acc"] - intervention_metrics["acc"]),
            "delta_vs_baseline_f1": float(final_metrics["f1"] - baseline_metrics["f1"]),
            "delta_vs_intervention_f1": float(final_metrics["f1"] - intervention_metrics["f1"]),
        },
        "outputs": {
            "decision_rows_csv": os.path.abspath(decision_csv),
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    write_json(summary_path, summary)
    print("[saved]", os.path.abspath(decision_csv))
    print("[saved]", os.path.abspath(summary_path))


if __name__ == "__main__":
    main()
