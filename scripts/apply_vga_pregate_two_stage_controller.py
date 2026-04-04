#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

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


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def maybe_int(value: object) -> Optional[int]:
    v = maybe_float(value)
    if v is None:
        return None
    return int(round(v))


def oriented_value(raw: float, direction: str) -> float:
    return float(raw) if direction == "high" else float(-raw)


def build_score(row: Dict[str, str], features: Sequence[Dict[str, Any]]) -> Optional[float]:
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


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a two-stage pre-intervention VGA controller to a held-out table.")
    ap.add_argument("--table_csv", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    rows = read_csv_rows(args.table_csv)
    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)
    p1_features = list(policy["p1"]["features"])
    p2_features = list(policy["p2"]["features"])
    tau_p1 = float(policy["p1"]["tau"])
    tau_p2 = float(policy["p2"]["tau"])

    decision_rows: List[Dict[str, Any]] = []
    n = 0
    baseline_count = 0
    method_count = 0
    final_correct = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    oracle_correct_total = 0

    sensitive_true_total = 0
    predicted_sensitive_total = 0
    p1_tp = 0
    p1_fp = 0
    p1_fn = 0

    blocked_harm = 0
    blocked_help = 0
    blocked_neutral = 0
    method_harm = 0
    method_help = 0
    method_neutral = 0

    for row in tqdm(rows, desc="two-stage-apply", unit="sample"):
        p1_score = build_score(row, p1_features)
        p2_score = build_score(row, p2_features)
        if p1_score is None or p2_score is None:
            continue
        n += 1
        baseline_correct = int(maybe_int(row.get("baseline_correct")) or 0)
        intervention_correct = int(maybe_int(row.get("intervention_correct")) or 0)
        oracle_correct = int(maybe_int(row.get("oracle_correct")) or max(baseline_correct, intervention_correct))
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        sensitive = int(harm or help_)

        baseline_correct_total += baseline_correct
        intervention_correct_total += intervention_correct
        oracle_correct_total += oracle_correct
        sensitive_true_total += sensitive

        predicted_sensitive = int(p1_score >= tau_p1)
        predicted_sensitive_total += predicted_sensitive
        if predicted_sensitive and sensitive:
            p1_tp += 1
        elif predicted_sensitive and not sensitive:
            p1_fp += 1
        elif (not predicted_sensitive) and sensitive:
            p1_fn += 1

        run_method = bool(predicted_sensitive and (p2_score < tau_p2))
        route = "method" if run_method else "baseline"
        final_correct_row = intervention_correct if run_method else baseline_correct
        final_correct += final_correct_row

        if run_method:
            method_count += 1
            if harm:
                method_harm += 1
            elif help_:
                method_help += 1
            else:
                method_neutral += 1
        else:
            baseline_count += 1
            if harm:
                blocked_harm += 1
            elif help_:
                blocked_help += 1
            else:
                blocked_neutral += 1

        out = dict(row)
        out["p1_sensitive_score"] = float(p1_score)
        out["p2_harm_score"] = float(p2_score)
        out["tau_p1"] = float(tau_p1)
        out["tau_p2"] = float(tau_p2)
        out["predicted_sensitive"] = int(predicted_sensitive)
        out["route"] = route
        out["final_correct"] = int(final_correct_row)
        decision_rows.append(out)

    baseline_rate = safe_div(float(baseline_count), float(max(1, n)))
    method_rate = safe_div(float(method_count), float(max(1, n)))
    p1_prec = safe_div(float(p1_tp), float(max(1, predicted_sensitive_total)))
    p1_rec = safe_div(float(p1_tp), float(max(1, sensitive_true_total)))
    p1_f1 = safe_div(2.0 * p1_prec * p1_rec, p1_prec + p1_rec)

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
            "baseline_acc": safe_div(float(baseline_correct_total), float(max(1, n))),
            "intervention_acc": safe_div(float(intervention_correct_total), float(max(1, n))),
            "oracle_posthoc_acc": safe_div(float(oracle_correct_total), float(max(1, n))),
            "pregate_acc": safe_div(float(final_correct), float(max(1, n))),
            "delta_vs_baseline": safe_div(float(final_correct - baseline_correct_total), float(max(1, n))),
            "delta_vs_intervention": safe_div(float(final_correct - intervention_correct_total), float(max(1, n))),
            "gap_to_oracle_posthoc": safe_div(float(oracle_correct_total - final_correct), float(max(1, n))),
            "p1_predicted_sensitive_count": int(predicted_sensitive_total),
            "p1_predicted_sensitive_rate": safe_div(float(predicted_sensitive_total), float(max(1, n))),
            "p1_true_sensitive_count": int(sensitive_true_total),
            "p1_sensitive_precision": p1_prec,
            "p1_sensitive_recall": p1_rec,
            "p1_sensitive_f1": p1_f1,
            "blocked_harm": int(blocked_harm),
            "blocked_help": int(blocked_help),
            "blocked_neutral": int(blocked_neutral),
            "method_harm": int(method_harm),
            "method_help": int(method_help),
            "method_neutral": int(method_neutral),
            "method_help_precision": safe_div(float(method_help), float(max(1, method_count))),
            "method_harm_precision": safe_div(float(method_harm), float(max(1, method_count))),
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
