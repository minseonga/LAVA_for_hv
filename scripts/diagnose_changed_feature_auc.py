#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_yes_no(text: object) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("yes"):
        return "yes"
    return ""


def maybe_float(value: object) -> Optional[float]:
    try:
        x = float(str(value).strip())
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def maybe_int(value: object) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = sorted(set().union(*(set(row.keys()) for row in rows))) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        if not keys:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_route_candidate(row: Dict[str, str], mode: str) -> bool:
    if mode == "all":
        return True
    baseline_label = str(row.get("baseline_label", "")).strip().lower()
    intervention_label = str(row.get("intervention_label", "")).strip().lower()
    if baseline_label not in {"yes", "no"}:
        baseline_label = parse_yes_no(row.get("baseline_text", ""))
    if intervention_label not in {"yes", "no"}:
        intervention_label = parse_yes_no(row.get("intervention_text", ""))
    if mode == "changed_answer":
        return baseline_label in {"yes", "no"} and intervention_label in {"yes", "no"} and baseline_label != intervention_label
    if mode == "yes_to_no":
        return baseline_label == "yes" and intervention_label == "no"
    raise ValueError(f"Unsupported candidate_filter={mode!r}")


def auc_score(values: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n = len(values)
    n_pos = int(sum(labels))
    n_neg = int(n - n_pos)
    if n == 0 or n_pos == 0 or n_neg == 0:
        return None

    pairs = sorted(zip(values, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        rank += j - i
        i = j
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def threshold_best_net(values: Sequence[float], labels: Sequence[int], direction: str) -> Dict[str, Any]:
    high = direction == "high"
    ordered = sorted(zip(values, labels), key=lambda item: item[0], reverse=high)
    best_net = -10**9
    best_count = 0
    best_harm = 0
    best_help = 0
    harm = 0
    help_ = 0
    best_tau = None
    for idx, (value, label) in enumerate(ordered, start=1):
        if int(label) == 1:
            harm += 1
        else:
            help_ += 1
        net = harm - help_
        if net > best_net:
            best_net = net
            best_count = idx
            best_harm = harm
            best_help = help_
            best_tau = value
    return {
        "oracle_best_net": int(best_net),
        "oracle_selected_count": int(best_count),
        "oracle_selected_harm": int(best_harm),
        "oracle_selected_help": int(best_help),
        "oracle_tau": best_tau,
    }


def parse_feature_cols(rows: Sequence[Dict[str, str]], feature_cols: str, feature_prefixes: str) -> List[str]:
    if not rows:
        return []
    cols = list(rows[0].keys())
    explicit = [x.strip() for x in str(feature_cols or "").split(",") if x.strip()]
    if explicit:
        return explicit
    prefixes = [x.strip() for x in str(feature_prefixes or "").split(",") if x.strip()]
    if not prefixes:
        prefixes = ["cheap_"]
    blocked = {"cheap_question", "cheap_decision_candidate_label"}
    return [col for col in cols if col not in blocked and any(col.startswith(prefix) for prefix in prefixes)]


def normalize_labels(rows: Sequence[Dict[str, str]]) -> Tuple[int, int, int]:
    n_harm = 0
    n_help = 0
    n_neutral = 0
    for row in rows:
        harm = maybe_int(row.get("harm"))
        help_ = maybe_int(row.get("help"))
        if harm is None or help_ is None:
            bc = maybe_int(row.get("baseline_correct"))
            ic = maybe_int(row.get("intervention_correct"))
            if bc is not None and ic is not None:
                harm = int(bc == 1 and ic == 0)
                help_ = int(bc == 0 and ic == 1)
            else:
                harm = 0
                help_ = 0
        row["harm"] = str(int(harm or 0))
        row["help"] = str(int(help_ or 0))
        if int(harm or 0):
            n_harm += 1
        elif int(help_ or 0):
            n_help += 1
        else:
            n_neutral += 1
    return n_harm, n_help, n_neutral


def feature_metrics(rows: Sequence[Dict[str, str]], features: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for feature in features:
        values: List[float] = []
        labels: List[int] = []
        n_present = 0
        for row in rows:
            value = maybe_float(row.get(feature))
            if value is None:
                continue
            n_present += 1
            values.append(float(value))
            labels.append(1 if str(row.get("harm", "0")).strip() == "1" else 0)
        if not values:
            continue
        auc = auc_score(values, labels)
        if auc is None:
            continue
        if auc >= 0.5:
            direction = "high"
            oriented_auc = auc
        else:
            direction = "low"
            oriented_auc = 1.0 - auc
        best = threshold_best_net(values, labels, direction)
        harm_values = [v for v, label in zip(values, labels) if label == 1]
        nonharm_values = [v for v, label in zip(values, labels) if label == 0]
        out.append(
            {
                "feature": feature,
                "direction": direction,
                "auroc": float(oriented_auc),
                "raw_auroc_high": float(auc),
                "n": int(len(values)),
                "n_present": int(n_present),
                "n_pos": int(sum(labels)),
                "n_neg": int(len(labels) - sum(labels)),
                "harm_mean": sum(harm_values) / max(1, len(harm_values)),
                "nonharm_mean": sum(nonharm_values) / max(1, len(nonharm_values)),
                **best,
            }
        )
    out.sort(key=lambda row: (-float(row["auroc"]), -float(row["oracle_best_net"]), str(row["feature"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank replay features by harm AUROC on a changed-answer discovery set.")
    ap.add_argument("--rows_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--candidate_filter", type=str, default="changed_answer", choices=["all", "changed_answer", "yes_to_no"])
    ap.add_argument("--feature_cols", type=str, default="")
    ap.add_argument("--feature_prefixes", type=str, default="cheap_hidden_")
    ap.add_argument("--top_k", type=int, default=30)
    args = ap.parse_args()

    rows_all = read_csv_rows(os.path.abspath(args.rows_csv))
    normalize_labels(rows_all)
    rows = [row for row in rows_all if is_route_candidate(row, str(args.candidate_filter))]
    n_harm, n_help, n_neutral = normalize_labels(rows)
    features = parse_feature_cols(rows, str(args.feature_cols), str(args.feature_prefixes))
    metrics = feature_metrics(rows, features)

    write_csv(os.path.abspath(args.out_csv), metrics)
    summary = {
        "inputs": {
            "rows_csv": os.path.abspath(args.rows_csv),
            "candidate_filter": str(args.candidate_filter),
            "feature_cols": str(args.feature_cols),
            "feature_prefixes": str(args.feature_prefixes),
        },
        "counts": {
            "n_all_rows": int(len(rows_all)),
            "n_eval_rows": int(len(rows)),
            "n_harm": int(n_harm),
            "n_help": int(n_help),
            "n_neutral": int(n_neutral),
            "n_features": int(len(features)),
        },
        "top_features": metrics[: int(args.top_k)],
        "outputs": {
            "out_csv": os.path.abspath(args.out_csv),
            "out_json": os.path.abspath(args.out_json) if str(args.out_json or "").strip() else "",
        },
    }
    if str(args.out_json or "").strip():
        write_json(os.path.abspath(args.out_json), summary)

    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    for row in metrics[: int(args.top_k)]:
        print(
            f"{row['feature']:55s} AUROC={float(row['auroc']):.4f} "
            f"dir={row['direction']:4s} best_net={row['oracle_best_net']} "
            f"selected={row['oracle_selected_count']}"
        )
    print("[saved]", os.path.abspath(args.out_csv))
    if str(args.out_json or "").strip():
        print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
