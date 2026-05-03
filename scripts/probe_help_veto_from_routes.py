#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_float(value: object) -> Optional[float]:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        value_f = float(text)
    except Exception:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def maybe_int(value: object) -> Optional[int]:
    value_f = maybe_float(value)
    if value_f is None:
        return None
    return int(round(value_f))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if float(den) else 0.0


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(int(x) for x in labels)
    n_neg = len(labels) - n_pos
    if len(scores) != len(labels) or n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if int(label) == 1)
    return float((rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg))


def parse_prefixes(text: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in str(text or "").split(",") if x.strip())


def label01(value: object, target: str) -> int:
    return int(str(value or "").strip().lower() == str(target))


def feature_allowed(name: str, prefixes: Tuple[str, ...]) -> bool:
    if name in {"id", "question_id", "image"}:
        return False
    if name in {"harm", "help", "baseline_correct", "intervention_correct"}:
        return False
    if name.endswith("_text") or "question" in name or "object_terms" in name:
        return False
    return not prefixes or any(name.startswith(prefix) for prefix in prefixes)


def selected_fallback_rows(features_csv: str, routes_csv: str) -> List[Dict[str, Any]]:
    features = read_csv(features_csv)
    routes = read_csv(routes_csv)
    feature_by_id = {str(row.get("id", row.get("question_id", ""))).strip(): row for row in features}
    merged: List[Dict[str, Any]] = []
    for route in routes:
        if str(route.get("route", "")).strip() != "baseline":
            continue
        sid = str(route.get("id", route.get("question_id", ""))).strip()
        if not sid:
            continue
        row: Dict[str, Any] = dict(feature_by_id.get(sid, {}))
        row.update(
            {
                "id": sid,
                "route_score": route.get("score", ""),
                "route_c_score": route.get("c_score", ""),
                "route_d_score": route.get("d_score", ""),
                "route_family": route.get("family", ""),
                "route_tau": route.get("tau", ""),
                "route_alpha": route.get("alpha", ""),
                "route_final_source": route.get("final_source", ""),
            }
        )
        for key in ("harm", "help", "baseline_correct", "intervention_correct"):
            if key in route:
                row[key] = route.get(key)
        baseline_label = str(row.get("baseline_label", "")).strip().lower()
        intervention_label = str(row.get("intervention_label", "")).strip().lower()
        row["transition_yes_to_no"] = int(baseline_label == "yes" and intervention_label == "no")
        row["transition_no_to_yes"] = int(baseline_label == "no" and intervention_label == "yes")
        row["baseline_is_yes"] = int(baseline_label == "yes")
        row["baseline_is_no"] = int(baseline_label == "no")
        row["intervention_is_yes"] = int(intervention_label == "yes")
        row["intervention_is_no"] = int(intervention_label == "no")
        merged.append(row)
    return merged


def candidate_features(rows: Sequence[Dict[str, Any]], prefixes: Tuple[str, ...], min_present_rate: float) -> List[str]:
    keys = sorted({key for row in rows for key in row.keys() if feature_allowed(str(key), prefixes)})
    out: List[str] = []
    n = max(1, len(rows))
    for key in keys:
        present = sum(int(maybe_float(row.get(key)) is not None) for row in rows)
        if safe_div(float(present), float(n)) >= float(min_present_rate):
            out.append(key)
    return out


def threshold_values(values: Sequence[float]) -> List[float]:
    uniq = sorted(set(float(x) for x in values if math.isfinite(float(x))))
    if not uniq:
        return []
    thresholds: List[float] = [float("-inf")]
    thresholds.extend(uniq)
    thresholds.append(float("inf"))
    return thresholds


def eval_veto(rows: Sequence[Dict[str, Any]], feature: str, direction: str, tau: float) -> Dict[str, Any]:
    selected_harm = sum(int(maybe_int(row.get("harm")) or 0) for row in rows)
    selected_help = sum(int(maybe_int(row.get("help")) or 0) for row in rows)
    selected_neutral = int(len(rows) - selected_harm - selected_help)
    veto_count = 0
    veto_harm = 0
    veto_help = 0
    veto_neutral = 0
    for row in rows:
        x = maybe_float(row.get(feature))
        if x is None:
            continue
        veto = bool(float(x) >= float(tau)) if direction == "high" else bool(float(x) <= float(tau))
        if not veto:
            continue
        veto_count += 1
        harm = int(maybe_int(row.get("harm")) or 0)
        help_ = int(maybe_int(row.get("help")) or 0)
        veto_harm += harm
        veto_help += help_
        veto_neutral += int((harm == 0) and (help_ == 0))
    kept_harm = int(selected_harm - veto_harm)
    kept_help = int(selected_help - veto_help)
    kept_neutral = int(selected_neutral - veto_neutral)
    original_net = int(selected_harm - selected_help)
    kept_net = int(kept_harm - kept_help)
    return {
        "feature": feature,
        "direction": direction,
        "tau": float(tau),
        "selected_count": int(len(rows)),
        "original_harm": int(selected_harm),
        "original_help": int(selected_help),
        "original_neutral": int(selected_neutral),
        "original_net": int(original_net),
        "veto_count": int(veto_count),
        "veto_harm": int(veto_harm),
        "veto_help": int(veto_help),
        "veto_neutral": int(veto_neutral),
        "kept_harm": int(kept_harm),
        "kept_help": int(kept_help),
        "kept_neutral": int(kept_neutral),
        "kept_net": int(kept_net),
        "delta_net": int(kept_net - original_net),
        "veto_help_precision": safe_div(float(veto_help), float(veto_count)),
        "veto_harm_precision": safe_div(float(veto_harm), float(veto_count)),
        "veto_help_recall": safe_div(float(veto_help), float(selected_help)),
        "veto_harm_recall": safe_div(float(veto_harm), float(selected_harm)),
        "kept_harm_recall": safe_div(float(kept_harm), float(selected_harm)),
        "kept_help_recall": safe_div(float(kept_help), float(selected_help)),
    }


def metric_key(row: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        float(row.get("delta_net") or 0.0),
        float(row.get("kept_net") or 0.0),
        float(row.get("veto_help_precision") or 0.0),
        float(row.get("veto_help_recall") or 0.0),
        -float(row.get("veto_harm_recall") or 0.0),
    )


def fit_policy(
    rows: Sequence[Dict[str, Any]],
    *,
    prefixes: Tuple[str, ...],
    min_present_rate: float,
    min_veto_count: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    features = candidate_features(rows, prefixes, min_present_rate)
    metrics: List[Dict[str, Any]] = []
    for feat in features:
        xs: List[float] = []
        ys: List[int] = []
        for row in rows:
            x = maybe_float(row.get(feat))
            if x is None:
                continue
            xs.append(float(x))
            ys.append(int(maybe_int(row.get("help")) or 0))
        auc_high = binary_auroc(xs, ys)
        auc_low = binary_auroc([-x for x in xs], ys) if xs else None
        if auc_high is None or auc_low is None:
            continue
        direction = "high" if float(auc_high) >= float(auc_low) else "low"
        feature_auc = max(float(auc_high), float(auc_low))
        for tau in threshold_values(xs):
            result = eval_veto(rows, feat, direction, tau)
            if int(result["veto_count"]) < int(min_veto_count):
                continue
            result["help_auroc"] = feature_auc
            result["n_present"] = int(len(xs))
            metrics.append(result)
    if not metrics:
        raise RuntimeError("No help-veto candidates were found.")
    metrics.sort(key=metric_key, reverse=True)
    return metrics[0], metrics


def apply_policy(rows: Sequence[Dict[str, Any]], policy: Dict[str, Any]) -> Dict[str, Any]:
    return eval_veto(rows, str(policy["feature"]), str(policy["direction"]), float(policy["tau"]))


def parse_apply_spec(spec: str) -> Tuple[str, str, str]:
    parts = str(spec).split(":", 2)
    if len(parts) != 3:
        raise ValueError("--apply must be name:features_csv:routes_csv")
    return parts[0], parts[1], parts[2]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe a second-stage help-veto rule on already selected baseline fallback candidates."
    )
    ap.add_argument("--fit_features_csv", required=True)
    ap.add_argument("--fit_routes_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--feature_prefixes",
        default="cheap_,route_,transition_,baseline_is_,intervention_is_",
        help="Comma-separated feature prefixes to search. Add baseline_cheap_ once baseline teacher-force features exist.",
    )
    ap.add_argument("--min_present_rate", type=float, default=0.8)
    ap.add_argument("--min_veto_count", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--apply", action="append", default=[], help="Optional name:features_csv:routes_csv")
    args = ap.parse_args()

    prefixes = parse_prefixes(args.feature_prefixes)
    fit_rows = selected_fallback_rows(args.fit_features_csv, args.fit_routes_csv)
    if not fit_rows:
        raise RuntimeError("No selected baseline fallback rows in fit_routes_csv.")
    best, metrics = fit_policy(
        fit_rows,
        prefixes=prefixes,
        min_present_rate=float(args.min_present_rate),
        min_veto_count=int(args.min_veto_count),
    )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    metrics_csv = os.path.join(out_dir, "help_veto_feature_sweep.csv")
    policy_json = os.path.join(out_dir, "selected_help_veto_policy.json")
    summary_json = os.path.join(out_dir, "summary.json")
    summary_md = os.path.join(out_dir, "summary.md")
    write_csv(metrics_csv, metrics[: max(1, int(args.top_k))])

    policy = {
        "policy_type": "selected_fallback_help_veto",
        "route_policy": "after harm router selects baseline, veto to method if feature crosses threshold",
        "feature": best["feature"],
        "direction": best["direction"],
        "tau": best["tau"],
        "fit_result": best,
        "feature_prefixes": list(prefixes),
    }
    write_json(policy_json, policy)

    apply_rows: List[Dict[str, Any]] = [{"split": "discovery", **apply_policy(fit_rows, policy)}]
    for spec in args.apply:
        name, features_csv, routes_csv = parse_apply_spec(spec)
        rows = selected_fallback_rows(features_csv, routes_csv)
        apply_rows.append({"split": name, **apply_policy(rows, policy)})

    write_csv(os.path.join(out_dir, "apply_summary.csv"), apply_rows)
    write_json(
        summary_json,
        {
            "inputs": {
                "fit_features_csv": os.path.abspath(args.fit_features_csv),
                "fit_routes_csv": os.path.abspath(args.fit_routes_csv),
                "feature_prefixes": list(prefixes),
                "min_present_rate": float(args.min_present_rate),
                "min_veto_count": int(args.min_veto_count),
            },
            "selected_policy": policy,
            "apply_summary": apply_rows,
            "outputs": {
                "metrics_csv": metrics_csv,
                "policy_json": policy_json,
                "apply_summary_csv": os.path.join(out_dir, "apply_summary.csv"),
            },
        },
    )
    lines = [
        "| Split | Fallback | Original H/G/Net | Veto H/G/N | Kept H/G/Net | dNet | Veto Help Prec | Veto Help Rec | Veto Harm Rec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in apply_rows:
        lines.append(
            f"| {row['split']} | {row['selected_count']} | "
            f"{row['original_harm']}/{row['original_help']}/{row['original_net']} | "
            f"{row['veto_harm']}/{row['veto_help']}/{row['veto_neutral']} | "
            f"{row['kept_harm']}/{row['kept_help']}/{row['kept_net']} | "
            f"{row['delta_net']:+d} | "
            f"{100*float(row['veto_help_precision']):.2f} | "
            f"{100*float(row['veto_help_recall']):.2f} | "
            f"{100*float(row['veto_harm_recall']):.2f} |"
        )
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"selected_policy": policy, "apply_summary": apply_rows}, ensure_ascii=False, indent=2))
    print("[saved]", metrics_csv)
    print("[saved]", policy_json)
    print("[saved]", summary_json)
    print("[saved]", summary_md)


if __name__ == "__main__":
    main()
