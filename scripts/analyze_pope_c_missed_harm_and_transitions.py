#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def parse_yes_no(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip().lower() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def safe_id(value: object) -> str:
    return str(value or "").strip()


def parse_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def parse_int01(value: object) -> Optional[int]:
    if value in {1, 1.0, True, "1", "true", "True"}:
        return 1
    if value in {0, 0.0, False, "0", "false", "False"}:
        return 0
    return None


def load_gt(path: str, id_col: str, label_col: str, group_col: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            sid = safe_id(row.get(id_col))
            label = safe_id(row.get(label_col)).lower()
            if not sid or label not in {"yes", "no"}:
                continue
            out[sid] = {
                "gt_label": label,
                "category": safe_id(row.get(group_col)) if group_col else "",
                "question": safe_id(row.get("question")),
            }
    return out


def load_pred(path: str, pred_key: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = safe_id(row.get("question_id", row.get("id")))
            if not sid or sid.lower() in {"none", "null", "nan"}:
                continue
            if pred_key == "auto":
                text = row.get("text", "") or row.get("output", "") or row.get("answer", "")
            else:
                text = row.get(pred_key, "")
            out[sid] = {"text": str(text or ""), "label": parse_yes_no(str(text or ""))}
    return out


def load_csv_rows(path: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            sid = safe_id(row.get("id", row.get("question_id")))
            if sid:
                rows[sid] = dict(row)
    return rows


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def pstdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return float(math.sqrt(max(0.0, var))) or 1.0


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = float(i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def infer_feature_specs(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_prefixes: Sequence[str],
    top_k: int,
    min_present_rate: float,
) -> List[Dict[str, Any]]:
    if not rows:
        return []

    candidate_cols: List[str] = []
    for key in rows[0].keys():
        if any(str(key).startswith(prefix) for prefix in feature_prefixes):
            n_present = sum(1 for row in rows if parse_float(row.get(key)) is not None)
            if n_present >= max(5, int(float(min_present_rate) * len(rows))):
                candidate_cols.append(str(key))

    labels = [int(row.get("harm", 0)) for row in rows]
    specs: List[Dict[str, Any]] = []
    for col in candidate_cols:
        paired = [(parse_float(row.get(col)), int(row.get("harm", 0))) for row in rows]
        paired = [(x, y) for x, y in paired if x is not None]
        if len(paired) < 5:
            continue
        vals = [float(x) for x, _ in paired]
        ys = [int(y) for _, y in paired]
        auc_high = auroc(vals, ys)
        if auc_high is None:
            continue
        auc_low = 1.0 - auc_high
        direction = "high" if auc_high >= auc_low else "low"
        oriented = vals if direction == "high" else [-x for x in vals]
        specs.append(
            {
                "feature": col,
                "direction": direction,
                "auroc": max(float(auc_high), float(auc_low)),
                "mu": mean(oriented),
                "sd": pstdev(oriented),
            }
        )

    specs.sort(key=lambda x: float(x["auroc"]), reverse=True)
    return specs[: int(top_k)]


def score_row(row: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> Optional[float]:
    vals: List[float] = []
    for spec in specs:
        x = parse_float(row.get(str(spec["feature"])))
        if x is None:
            return None
        if str(spec.get("direction")) == "low":
            x = -x
        vals.append((float(x) - float(spec["mu"])) / float(spec["sd"] or 1.0))
    return mean(vals)


def choose_threshold(rows: Sequence[Mapping[str, Any]], scores: Mapping[str, float]) -> Optional[float]:
    vals = sorted(set(float(v) for v in scores.values()))
    if not vals:
        return None
    thresholds = [max(vals) + 1e-9] + vals + [min(vals) - 1e-9]
    best_key = None
    best_tau = None
    for tau in thresholds:
        selected = harm_fixed = help_lost = correct = 0
        for row in rows:
            sid = str(row["id"])
            selected_here = float(scores[sid]) >= float(tau)
            if selected_here:
                selected += 1
                correct += int(row["baseline_correct"])
                harm_fixed += int(row["harm"])
                help_lost += int(row["help"])
            else:
                correct += int(row["intervention_correct"])
        final_acc = correct / len(rows) if rows else 0.0
        net = harm_fixed - help_lost
        key = (final_acc, net, -selected)
        if best_key is None or key > best_key:
            best_key = key
            best_tau = tau
    return best_tau


def count_outcomes(rows: Iterable[Mapping[str, Any]], route_key: Optional[str] = None) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    changed: Counter[str] = Counter()
    by_category: Dict[str, Counter[str]] = defaultdict(Counter)
    missed_harm: Counter[str] = Counter()
    caught_harm: Counter[str] = Counter()

    for row in rows:
        b = str(row.get("baseline_label", ""))
        m = str(row.get("intervention_label", ""))
        out = str(row.get("outcome", ""))
        cat = str(row.get("category", ""))
        trans = f"{b}->{m}"
        counts[out] += 1
        transitions[f"{trans}:{out}"] += 1
        by_category[cat][out] += 1
        if b != m:
            changed[f"{trans}:{out}"] += 1
        if route_key:
            route = str(row.get(route_key, ""))
            if int(row.get("harm", 0)) == 1:
                if route == "baseline":
                    caught_harm[trans] += 1
                else:
                    missed_harm[trans] += 1

    return {
        "outcome_counts": dict(counts),
        "transition_outcome_counts": dict(transitions),
        "changed_transition_counts": dict(changed),
        "by_category": {k: dict(v) for k, v in sorted(by_category.items())},
        "caught_harm_by_transition": dict(caught_harm),
        "missed_harm_by_transition": dict(missed_harm),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose C-controller caught/missed harm by POPE transition.")
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--feature_rows_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--label_col", default="answer")
    ap.add_argument("--group_col", default="category")
    ap.add_argument("--baseline_pred_key", default="auto")
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--feature_prefixes", default="cheap_")
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min_present_rate", type=float, default=0.8)
    ap.add_argument("--max_examples", type=int, default=20)
    args = ap.parse_args()

    gt = load_gt(args.gt_csv, args.id_col, args.label_col, args.group_col)
    baseline = load_pred(args.baseline_pred_jsonl, args.baseline_pred_key)
    intervention = load_pred(args.intervention_pred_jsonl, args.intervention_pred_key)
    feature_rows = load_csv_rows(args.feature_rows_csv)

    rows: List[Dict[str, Any]] = []
    for sid, g in gt.items():
        b = baseline.get(sid, {})
        m = intervention.get(sid, {})
        baseline_label = b.get("label", "")
        intervention_label = m.get("label", "")
        if baseline_label not in {"yes", "no"} or intervention_label not in {"yes", "no"}:
            continue
        bc = int(baseline_label == g["gt_label"])
        ic = int(intervention_label == g["gt_label"])
        if bc and ic:
            outcome = "both_correct"
        elif (not bc) and (not ic):
            outcome = "both_wrong"
        elif (not bc) and ic:
            outcome = "help"
        else:
            outcome = "harm"
        row: Dict[str, Any] = {
            "id": sid,
            "gt_label": g["gt_label"],
            "category": g["category"],
            "question": g["question"],
            "baseline_label": baseline_label,
            "intervention_label": intervention_label,
            "baseline_text": b.get("text", ""),
            "intervention_text": m.get("text", ""),
            "baseline_correct": bc,
            "intervention_correct": ic,
            "harm": int(outcome == "harm"),
            "help": int(outcome == "help"),
            "outcome": outcome,
        }
        row.update(feature_rows.get(sid, {}))
        row["id"] = sid
        rows.append(row)

    feature_prefixes = [x.strip() for x in str(args.feature_prefixes).split(",") if x.strip()]
    specs = infer_feature_specs(
        rows,
        feature_prefixes=feature_prefixes,
        top_k=int(args.top_k),
        min_present_rate=float(args.min_present_rate),
    )
    scores = {str(row["id"]): score_row(row, specs) for row in rows}
    scores = {k: float(v) for k, v in scores.items() if v is not None}
    threshold = float(args.threshold) if args.threshold is not None else choose_threshold(rows, scores)

    selected_count = selected_harm = selected_help = selected_neutral = final_correct = 0
    missed_examples: List[Dict[str, Any]] = []
    caught_examples: List[Dict[str, Any]] = []
    for row in rows:
        sid = str(row["id"])
        score = scores.get(sid)
        route = "method"
        if score is not None and threshold is not None and float(score) >= float(threshold):
            route = "baseline"
            selected_count += 1
            selected_harm += int(row["harm"])
            selected_help += int(row["help"])
            if not int(row["harm"]) and not int(row["help"]):
                selected_neutral += 1
            final_correct += int(row["baseline_correct"])
        else:
            final_correct += int(row["intervention_correct"])
        row["diagnostic_c_score"] = score
        row["diagnostic_route"] = route

        if int(row["harm"]) == 1:
            example = {
                "id": sid,
                "category": row["category"],
                "transition": f"{row['baseline_label']}->{row['intervention_label']}",
                "score": score,
                "gt_label": row["gt_label"],
                "question": row["question"],
                "baseline_text": row["baseline_text"],
                "intervention_text": row["intervention_text"],
            }
            for spec in specs:
                feat = str(spec["feature"])
                example[feat] = row.get(feat)
            if route == "baseline" and len(caught_examples) < int(args.max_examples):
                caught_examples.append(example)
            if route != "baseline" and len(missed_examples) < int(args.max_examples):
                missed_examples.append(example)

    base_correct = sum(int(row["baseline_correct"]) for row in rows)
    method_correct = sum(int(row["intervention_correct"]) for row in rows)
    n = len(rows)
    summary = {
        "n": n,
        "baseline_acc": base_correct / n if n else 0.0,
        "intervention_acc": method_correct / n if n else 0.0,
        "diagnostic_final_acc": final_correct / n if n else 0.0,
        "diagnostic_delta_vs_intervention": (final_correct - method_correct) / n if n else 0.0,
        "selected": selected_count,
        "selected_harm": selected_harm,
        "selected_help": selected_help,
        "selected_neutral": selected_neutral,
        "missed_harm": sum(int(row["harm"]) == 1 and row["diagnostic_route"] != "baseline" for row in rows),
        "feature_specs": specs,
        "threshold": threshold,
    }
    analysis = count_outcomes(rows, route_key="diagnostic_route")
    out = {
        "inputs": vars(args),
        "summary": summary,
        "analysis": analysis,
        "missed_harm_examples": missed_examples,
        "caught_harm_examples": caught_examples,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("n", summary["n"])
    print("baseline_acc", summary["baseline_acc"])
    print("intervention_acc", summary["intervention_acc"])
    print("diagnostic_final_acc", summary["diagnostic_final_acc"])
    print("delta_vs_intervention", summary["diagnostic_delta_vs_intervention"])
    print("selected", selected_count, "harm", selected_harm, "help", selected_help, "neutral", selected_neutral)
    print("missed_harm", summary["missed_harm"])
    print("threshold", threshold)
    print("feature_specs")
    for spec in specs:
        print(spec["feature"], "dir", spec["direction"], "auroc", round(float(spec["auroc"]), 6))
    print("missed_harm_by_transition", analysis["missed_harm_by_transition"])
    print("caught_harm_by_transition", analysis["caught_harm_by_transition"])
    print("[saved]", args.out_json)


if __name__ == "__main__":
    main()
