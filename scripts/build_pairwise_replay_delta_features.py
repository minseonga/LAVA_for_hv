#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


def maybe_float(value: object) -> Optional[float]:
    try:
        text = str(value).strip()
        if not text:
            return None
        out = float(text)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def maybe_int(value: object) -> Optional[int]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
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


def normalized_label(row: Dict[str, str], label_key: str, text_key: str) -> str:
    label = str(row.get(label_key, "")).strip().lower()
    if label in {"yes", "no"}:
        return label
    return parse_yes_no(row.get(text_key, ""))


def is_route_candidate(row: Dict[str, str], mode: str) -> bool:
    if mode == "all":
        return True
    baseline_label = normalized_label(row, "baseline_label", "baseline_text")
    intervention_label = normalized_label(row, "intervention_label", "intervention_text")
    if mode == "changed_answer":
        return baseline_label in {"yes", "no"} and intervention_label in {"yes", "no"} and baseline_label != intervention_label
    if mode == "yes_to_no":
        return baseline_label == "yes" and intervention_label == "no"
    if mode == "no_to_yes":
        return baseline_label == "no" and intervention_label == "yes"
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


def numeric_feature_cols(
    intervention_rows: Sequence[Dict[str, str]],
    baseline_by_id: Dict[str, Dict[str, str]],
    *,
    feature_cols: str,
    feature_prefixes: str,
) -> List[str]:
    if not intervention_rows:
        return []
    explicit = [x.strip() for x in str(feature_cols or "").split(",") if x.strip()]
    if explicit:
        return explicit
    prefixes = [x.strip() for x in str(feature_prefixes or "").split(",") if x.strip()]
    if not prefixes:
        prefixes = ["cheap_"]
    blocked = {
        "cheap_question",
        "cheap_decision_candidate_label",
    }
    cols = list(intervention_rows[0].keys())
    out: List[str] = []
    for col in cols:
        if col in blocked or not any(col.startswith(prefix) for prefix in prefixes):
            continue
        has_pair_value = False
        for row in intervention_rows[: min(200, len(intervention_rows))]:
            sid = str(row.get("id", "")).strip()
            base_row = baseline_by_id.get(sid)
            if base_row is None:
                continue
            if maybe_float(row.get(col)) is not None and maybe_float(base_row.get(col)) is not None:
                has_pair_value = True
                break
        if has_pair_value:
            out.append(col)
    return out


def normalize_effect_labels(row: Dict[str, str]) -> Tuple[int, int]:
    harm = maybe_int(row.get("harm"))
    help_ = maybe_int(row.get("help"))
    if harm is not None and help_ is not None:
        return int(harm or 0), int(help_ or 0)
    bc = maybe_int(row.get("baseline_correct"))
    ic = maybe_int(row.get("intervention_correct"))
    if bc is None or ic is None:
        return 0, 0
    return int(bc == 1 and ic == 0), int(bc == 0 and ic == 1)


def build_delta_rows(
    intervention_rows: Sequence[Dict[str, str]],
    baseline_rows: Sequence[Dict[str, str]],
    *,
    id_col: str,
    candidate_filter: str,
    feature_cols: str,
    feature_prefixes: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    baseline_by_id = {str(row.get(id_col, row.get("id", ""))).strip(): row for row in baseline_rows}
    features = numeric_feature_cols(
        intervention_rows,
        baseline_by_id,
        feature_cols=feature_cols,
        feature_prefixes=feature_prefixes,
    )
    out: List[Dict[str, Any]] = []
    n_missing_baseline = 0
    n_not_candidate = 0
    n_no_delta = 0
    n_harm = 0
    n_help = 0
    n_neutral = 0

    for int_row in intervention_rows:
        sid = str(int_row.get(id_col, int_row.get("id", ""))).strip()
        base_row = baseline_by_id.get(sid)
        if base_row is None:
            n_missing_baseline += 1
            continue
        if not is_route_candidate(int_row, candidate_filter):
            n_not_candidate += 1
            continue
        harm, help_ = normalize_effect_labels(int_row)
        n_harm += int(harm)
        n_help += int(help_)
        n_neutral += int((harm == 0) and (help_ == 0))

        row: Dict[str, Any] = {
            "id": sid,
            "image": int_row.get("image", ""),
            "question": int_row.get("question", ""),
            "baseline_text": int_row.get("baseline_text", ""),
            "intervention_text": int_row.get("intervention_text", ""),
            "baseline_label": normalized_label(int_row, "baseline_label", "baseline_text"),
            "intervention_label": normalized_label(int_row, "intervention_label", "intervention_text"),
            "baseline_correct": int_row.get("baseline_correct", ""),
            "intervention_correct": int_row.get("intervention_correct", ""),
            "harm": int(harm),
            "help": int(help_),
        }
        n_delta = 0
        for feature in features:
            int_value = maybe_float(int_row.get(feature))
            base_value = maybe_float(base_row.get(feature))
            if int_value is None or base_value is None:
                continue
            row[f"delta_baseline_minus_intervention__{feature}"] = float(base_value - int_value)
            row[f"delta_intervention_minus_baseline__{feature}"] = float(int_value - base_value)
            n_delta += 2
        if n_delta == 0:
            n_no_delta += 1
            continue
        out.append(row)

    summary = {
        "n_intervention_rows": int(len(intervention_rows)),
        "n_baseline_rows": int(len(baseline_rows)),
        "n_missing_baseline_rows": int(n_missing_baseline),
        "n_not_candidate": int(n_not_candidate),
        "n_no_delta": int(n_no_delta),
        "n_delta_rows": int(len(out)),
        "n_harm": int(n_harm),
        "n_help": int(n_help),
        "n_neutral": int(n_neutral),
        "n_source_features": int(len(features)),
        "n_delta_features": int(len(features) * 2),
    }
    return out, features, summary


def feature_metrics(rows: Sequence[Dict[str, Any]], features: Sequence[str], min_present_rate: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    min_present = int(math.ceil(float(min_present_rate) * float(len(rows))))
    for feature in features:
        values: List[float] = []
        labels: List[int] = []
        for row in rows:
            value = maybe_float(row.get(feature))
            if value is None:
                continue
            values.append(float(value))
            labels.append(1 if int(row.get("harm", 0) or 0) == 1 else 0)
        if len(values) < min_present:
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
        harm_values = [v for v, label in zip(values, labels) if label == 1]
        nonharm_values = [v for v, label in zip(values, labels) if label == 0]
        out.append(
            {
                "feature": feature,
                "direction": direction,
                "auroc": float(oriented_auc),
                "raw_auroc_high": float(auc),
                "n": int(len(values)),
                "n_present": int(len(values)),
                "n_pos": int(sum(labels)),
                "n_neg": int(len(labels) - sum(labels)),
                "harm_mean": sum(harm_values) / max(1, len(harm_values)),
                "nonharm_mean": sum(nonharm_values) / max(1, len(nonharm_values)),
                **threshold_best_net(values, labels, direction),
            }
        )
    out.sort(key=lambda row: (-float(row["auroc"]), -float(row["oracle_best_net"]), str(row["feature"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Join intervention-answer replay rows with baseline-answer replay rows, "
            "then rank pairwise delta features by harm AUROC."
        )
    )
    ap.add_argument("--intervention_rows_csv", required=True)
    ap.add_argument("--baseline_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--candidate_filter", default="changed_answer", choices=["all", "changed_answer", "yes_to_no", "no_to_yes"])
    ap.add_argument("--feature_cols", default="")
    ap.add_argument("--feature_prefixes", default="cheap_")
    ap.add_argument("--min_present_rate", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    intervention_rows = read_csv_rows(os.path.abspath(args.intervention_rows_csv))
    baseline_rows = read_csv_rows(os.path.abspath(args.baseline_rows_csv))
    delta_rows, source_features, counts = build_delta_rows(
        intervention_rows,
        baseline_rows,
        id_col=str(args.id_col),
        candidate_filter=str(args.candidate_filter),
        feature_cols=str(args.feature_cols),
        feature_prefixes=str(args.feature_prefixes),
    )
    delta_features = [
        key
        for key in (list(delta_rows[0].keys()) if delta_rows else [])
        if key.startswith("delta_baseline_minus_intervention__") or key.startswith("delta_intervention_minus_baseline__")
    ]
    metrics = feature_metrics(delta_rows, delta_features, float(args.min_present_rate))

    out_dir = os.path.abspath(args.out_dir)
    delta_csv = os.path.join(out_dir, "pairwise_delta_rows.csv")
    metrics_csv = os.path.join(out_dir, "pairwise_delta_feature_metrics.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(delta_csv, delta_rows)
    write_csv(metrics_csv, metrics)
    summary = {
        "inputs": {
            "intervention_rows_csv": os.path.abspath(args.intervention_rows_csv),
            "baseline_rows_csv": os.path.abspath(args.baseline_rows_csv),
            "candidate_filter": str(args.candidate_filter),
            "feature_cols": str(args.feature_cols),
            "feature_prefixes": str(args.feature_prefixes),
            "min_present_rate": float(args.min_present_rate),
        },
        "counts": counts,
        "source_features": source_features,
        "top_features": metrics[: int(args.top_k)],
        "outputs": {
            "pairwise_delta_rows_csv": delta_csv,
            "pairwise_delta_feature_metrics_csv": metrics_csv,
            "summary_json": summary_json,
        },
    }
    write_json(summary_json, summary)

    print(json.dumps({"counts": counts, "top_features": metrics[: int(args.top_k)]}, ensure_ascii=False, indent=2))
    print("[saved]", delta_csv)
    print("[saved]", metrics_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
