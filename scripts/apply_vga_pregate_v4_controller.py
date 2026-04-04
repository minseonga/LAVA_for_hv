#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def build_z_vector(row: Dict[str, str], features: Sequence[Dict[str, Any]]):
    xs: List[float] = []
    for feat in features:
        raw = maybe_float(row.get(str(feat["feature"])))
        if raw is None:
            return None
        oriented = oriented_value(raw, str(feat["direction"]))
        mu = float(feat["mu"])
        sd = max(float(feat["sd"]), 1e-6)
        xs.append((oriented - mu) / sd)
    return xs


def score_head(row: Dict[str, str], model: Dict[str, Any]):
    z = build_z_vector(row, list(model["features"]))
    if z is None:
        return None
    bias = float(model["bias"])
    weights = [float(x) for x in list(model["weights"])]
    logit = bias + sum(w * zi for w, zi in zip(weights, z))
    prob = sigmoid(max(min(logit, 30.0), -30.0))
    return float(logit), float(prob)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply unified pre-gating v4 controller to a held-out table.")
    ap.add_argument("--table_csv", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    rows = read_csv_rows(args.table_csv)
    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)
    help_model = dict(policy["help_model"])
    harm_model = dict(policy["harm_model"])
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
    total_harm = 0
    total_help = 0
    harm_help_scores: List[float] = []
    harm_help_labels_harm: List[int] = []

    for row in tqdm(rows, desc="v4-apply", unit="sample"):
        help_pack = score_head(row, help_model)
        harm_pack = score_head(row, harm_model)
        if help_pack is None or harm_pack is None:
            continue
        help_logit, help_prob = help_pack
        harm_logit, harm_prob = harm_pack
        apply_score = float(help_prob - lam * harm_prob)
        baseline_correct = int(maybe_int(row.get("baseline_correct")) or 0)
        intervention_correct = int(maybe_int(row.get("intervention_correct")) or 0)
        oracle_correct = int(maybe_int(row.get("oracle_correct")) or max(baseline_correct, intervention_correct))
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        total_harm += harm
        total_help += help_
        if harm or help_:
            harm_help_scores.append(float(-apply_score))
            harm_help_labels_harm.append(int(harm))

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
        n += 1

        out = dict(row)
        out["help_logit"] = float(help_logit)
        out["help_prob"] = float(help_prob)
        out["harm_logit"] = float(harm_logit)
        out["harm_prob"] = float(harm_prob)
        out["apply_score"] = float(apply_score)
        out["tau"] = float(tau)
        out["lambda_harm"] = float(lam)
        out["route"] = route
        out["final_correct"] = int(final_correct_row)
        decision_rows.append(out)

    method_rate = float(method_count / float(max(1, n)))
    baseline_rate = float(1.0 - method_rate)
    total_neutral = int(n - total_harm - total_help)
    veto_count = int(n - method_count)
    veto_harm = int(total_harm - applied_harm)
    veto_help = int(total_help - applied_help)
    veto_neutral = int(total_neutral - applied_neutral)
    harm_vs_help_auroc = None
    if harm_help_scores:
        from_summary = 0  # placeholder to keep local scope simple
        harm_vs_help_auroc = None
        # inline AUROC
        n_pos = sum(harm_help_labels_harm)
        n_neg = len(harm_help_labels_harm) - n_pos
        if n_pos > 0 and n_neg > 0:
            indexed = sorted(enumerate(harm_help_scores), key=lambda x: x[1])
            ranks = [0.0] * len(harm_help_scores)
            i = 0
            while i < len(indexed):
                j = i + 1
                while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                    j += 1
                avg_rank = (float(i + 1) + float(j)) / 2.0
                for k in range(i, j):
                    ranks[indexed[k][0]] = avg_rank
                i = j
            rank_sum_pos = sum(rank for rank, y in zip(ranks, harm_help_labels_harm) if int(y) == 1)
            harm_vs_help_auroc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)

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
            "harm_vs_help_auroc": None if harm_vs_help_auroc is None else float(harm_vs_help_auroc),
            "help_vs_harm_auroc": None if harm_vs_help_auroc is None else float(harm_vs_help_auroc),
            "veto_count": int(veto_count),
            "veto_harm": int(veto_harm),
            "veto_help": int(veto_help),
            "veto_neutral": int(veto_neutral),
            "veto_harm_precision": float(veto_harm / float(max(1, veto_count))),
            "veto_help_precision": float(veto_help / float(max(1, veto_count))),
            "veto_harm_recall": float(veto_harm / float(max(1, total_harm))),
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
