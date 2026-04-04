#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Sequence


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


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_float(value: object):
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def maybe_int(value: object):
    v = maybe_float(value)
    if v is None:
        return None
    return int(round(v))


def oriented_value(raw: float, direction: str) -> float:
    return float(raw) if direction == "high" else float(-raw)


def build_score_row(row: Dict[str, str], features: Sequence[Dict[str, Any]]):
    zs: List[float] = []
    for feat in features:
        raw = maybe_float(row.get(str(feat["feature"])))
        if raw is None:
            return None
        oriented = oriented_value(raw, str(feat["direction"]))
        mu = float(feat["mu"])
        sd = max(float(feat["sd"]), 1e-6)
        zs.append((oriented - mu) / sd)
    if not zs:
        return None
    return float(sum(zs) / float(len(zs)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a pre-intervention harm gate controller to a held-out table.")
    ap.add_argument("--table_csv", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    rows = read_csv_rows(args.table_csv)
    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)
    features = list(policy["features"])
    tau = float(policy["tau"])

    decision_rows: List[Dict[str, Any]] = []
    n = 0
    route_baseline = 0
    final_correct = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    oracle_correct_total = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0

    for row in rows:
        score = build_score_row(row, features)
        if score is None:
            continue
        n += 1
        baseline_correct = int(maybe_int(row.get("baseline_correct")) or 0)
        intervention_correct = int(maybe_int(row.get("intervention_correct")) or 0)
        oracle_correct = int(maybe_int(row.get("oracle_correct")) or max(baseline_correct, intervention_correct))
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        use_baseline = bool(score >= tau)
        route = "baseline" if use_baseline else "method"
        final_correct_row = baseline_correct if use_baseline else intervention_correct
        if use_baseline:
            route_baseline += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int((harm == 0) and (help_ == 0))
        final_correct += final_correct_row
        baseline_correct_total += baseline_correct
        intervention_correct_total += intervention_correct
        oracle_correct_total += oracle_correct

        out = dict(row)
        out["harm_score"] = score
        out["tau"] = tau
        out["route"] = route
        out["final_correct"] = int(final_correct_row)
        decision_rows.append(out)

    baseline_rate = float(route_baseline / float(max(1, n)))
    method_rate = float(1.0 - baseline_rate)
    summary = {
        "inputs": {
            "table_csv": os.path.abspath(args.table_csv),
            "policy_json": os.path.abspath(args.policy_json),
        },
        "policy": policy,
        "evaluation": {
            "n_eval": int(n),
            "baseline_rate": baseline_rate,
            "method_rate": method_rate,
            "baseline_acc": float(baseline_correct_total / float(max(1, n))),
            "intervention_acc": float(intervention_correct_total / float(max(1, n))),
            "oracle_posthoc_acc": float(oracle_correct_total / float(max(1, n))),
            "pregate_acc": float(final_correct / float(max(1, n))),
            "delta_vs_baseline": float((final_correct - baseline_correct_total) / float(max(1, n))),
            "delta_vs_intervention": float((final_correct - intervention_correct_total) / float(max(1, n))),
            "gap_to_oracle_posthoc": float((oracle_correct_total - final_correct) / float(max(1, n))),
            "selected_harm": int(selected_harm),
            "selected_help": int(selected_help),
            "selected_neutral": int(selected_neutral),
            "selected_harm_precision": float(selected_harm / float(max(1, route_baseline))),
            "selected_help_precision": float(selected_help / float(max(1, route_baseline))),
        },
        "outputs": {
            "decision_rows_csv": os.path.abspath(os.path.join(args.out_dir, "decision_rows.csv")),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    decision_csv = os.path.join(args.out_dir, "decision_rows.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_csv(decision_csv, decision_rows)
    write_json(summary_json, summary)
    print("[saved]", os.path.abspath(decision_csv))
    print("[saved]", os.path.abspath(summary_json))


if __name__ == "__main__":
    main()
