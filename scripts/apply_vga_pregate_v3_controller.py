#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Sequence

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable


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


def build_family_score(row: Dict[str, str], features: Sequence[Dict[str, Any]]):
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
    ap = argparse.ArgumentParser(description="Apply unified pre-gating v3 controller to a held-out table.")
    ap.add_argument("--table_csv", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    rows = read_csv_rows(args.table_csv)
    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)
    help_features = list(policy["help_features"])
    harm_features = list(policy["harm_features"])
    tau = float(policy["tau"])
    lam = float(policy["lambda_harm"])

    decision_rows: List[Dict[str, Any]] = []
    n = 0
    method_count = 0
    final_correct = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    oracle_correct_total = 0
    applied_harm = 0
    applied_help = 0
    applied_neutral = 0

    for row in tqdm(rows, desc="v3-apply", unit="sample"):
        help_score = build_family_score(row, help_features)
        harm_score = build_family_score(row, harm_features)
        if help_score is None or harm_score is None:
            continue
        n += 1
        apply_score = float(help_score - lam * harm_score)
        baseline_correct = int(maybe_int(row.get("baseline_correct")) or 0)
        intervention_correct = int(maybe_int(row.get("intervention_correct")) or 0)
        oracle_correct = int(maybe_int(row.get("oracle_correct")) or max(baseline_correct, intervention_correct))
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)

        use_method = bool(apply_score > tau)
        route = "method" if use_method else "baseline"
        final_correct_row = intervention_correct if use_method else baseline_correct

        if use_method:
            method_count += 1
            applied_harm += harm
            applied_help += help_
            applied_neutral += int((harm == 0) and (help_ == 0))
        final_correct += final_correct_row
        baseline_correct_total += baseline_correct
        intervention_correct_total += intervention_correct
        oracle_correct_total += oracle_correct

        out = dict(row)
        out["help_score"] = float(help_score)
        out["harm_score"] = float(harm_score)
        out["apply_score"] = float(apply_score)
        out["tau"] = float(tau)
        out["lambda_harm"] = float(lam)
        out["route"] = route
        out["final_correct"] = int(final_correct_row)
        decision_rows.append(out)

    method_rate = float(method_count / float(max(1, n)))
    baseline_rate = float(1.0 - method_rate)
    summary = {
        "inputs": {
            "table_csv": os.path.abspath(args.table_csv),
            "policy_json": os.path.abspath(args.policy_json),
        },
        "policy": policy,
        "evaluation": {
            "n_eval": int(n),
            "method_rate": method_rate,
            "baseline_rate": baseline_rate,
            "baseline_acc": float(baseline_correct_total / float(max(1, n))),
            "intervention_acc": float(intervention_correct_total / float(max(1, n))),
            "oracle_posthoc_acc": float(oracle_correct_total / float(max(1, n))),
            "pregate_acc": float(final_correct / float(max(1, n))),
            "delta_vs_baseline": float((final_correct - baseline_correct_total) / float(max(1, n))),
            "delta_vs_intervention": float((final_correct - intervention_correct_total) / float(max(1, n))),
            "gap_to_oracle_posthoc": float((oracle_correct_total - final_correct) / float(max(1, n))),
            "applied_harm": int(applied_harm),
            "applied_help": int(applied_help),
            "applied_neutral": int(applied_neutral),
            "applied_help_precision": float(applied_help / float(max(1, method_count))),
            "applied_harm_precision": float(applied_harm / float(max(1, method_count))),
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
