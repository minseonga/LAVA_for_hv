#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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


def mean(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 1.0
    mu = mean(seq)
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(max(math.sqrt(max(var, 0.0)), 1e-6))


def build_scores(rows: Sequence[Dict[str, str]], feature_specs: Sequence[Dict[str, Any]]) -> List[float]:
    cols: List[List[float]] = []
    for spec in feature_specs:
        feature = str(spec["feature"])
        direction = str(spec["direction"])
        vals = [float(maybe_float(row.get(feature))) for row in rows]
        oriented = vals if direction == "high" else [-float(v) for v in vals]
        mu = mean(oriented)
        sd = std(oriented)
        cols.append([(float(v) - mu) / sd for v in oriented])
    return [mean(zvals) for zvals in zip(*cols)]


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a generative post-hoc controller to a claim-aware table.")
    ap.add_argument("--table_csv", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    rows = read_csv_rows(args.table_csv)
    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)
    feature_specs = list(policy["feature_specs"])
    tau = float(policy["tau"])

    valid_rows: List[Dict[str, str]] = []
    for row in rows:
        ok = True
        for spec in feature_specs:
            if maybe_float(row.get(str(spec["feature"]))) is None:
                ok = False
                break
        if maybe_float(row.get("baseline_claim_utility")) is None or maybe_float(row.get("intervention_claim_utility")) is None:
            ok = False
        if ok:
            valid_rows.append(row)

    scores = build_scores(valid_rows, feature_specs)
    decision_rows: List[Dict[str, Any]] = []
    base_utils: List[float] = []
    int_utils: List[float] = []
    final_utils: List[float] = []
    veto_harm = veto_help = veto_neutral = 0

    for row, score in zip(valid_rows, scores):
        veto = bool(float(score) >= float(tau))
        base_u = float(maybe_float(row.get("baseline_claim_utility")))
        int_u = float(maybe_float(row.get("intervention_claim_utility")))
        final_u = base_u if veto else int_u
        harm = int(round(float(maybe_float(row.get("harm")) or 0.0)))
        help_ = int(round(float(maybe_float(row.get("help")) or 0.0)))
        if veto:
            veto_harm += harm
            veto_help += help_
            veto_neutral += int((harm == 0) and (help_ == 0))
        base_utils.append(base_u)
        int_utils.append(int_u)
        final_utils.append(final_u)
        out = dict(row)
        out["posthoc_risk_score"] = float(score)
        out["tau"] = float(tau)
        out["route"] = "baseline" if veto else "method"
        out["final_claim_utility"] = float(final_u)
        decision_rows.append(out)

    veto_count = veto_harm + veto_help + veto_neutral
    summary = {
        "inputs": {
            "table_csv": os.path.abspath(args.table_csv),
            "policy_json": os.path.abspath(args.policy_json),
        },
        "policy": policy,
        "evaluation": {
            "n_eval": int(len(valid_rows)),
            "baseline_rate": safe_div(float(veto_count), float(max(1, len(valid_rows)))),
            "method_rate": safe_div(float(len(valid_rows) - veto_count), float(max(1, len(valid_rows)))),
            "baseline_claim_utility": mean(base_utils),
            "intervention_claim_utility": mean(int_utils),
            "final_claim_utility": mean(final_utils),
            "delta_vs_intervention": float(mean(final_utils) - mean(int_utils)),
            "delta_vs_baseline": float(mean(final_utils) - mean(base_utils)),
            "veto_count": int(veto_count),
            "veto_harm": int(veto_harm),
            "veto_help": int(veto_help),
            "veto_neutral": int(veto_neutral),
            "veto_harm_precision": safe_div(float(veto_harm), float(max(1, veto_count))),
            "veto_help_precision": safe_div(float(veto_help), float(max(1, veto_count))),
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
