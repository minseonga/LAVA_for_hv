#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num) / float(den)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def canonical_object(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[-1]).strip()
    return str(value).strip()


def object_list(values: Iterable[Any]) -> List[str]:
    return [canonical_object(value) for value in values if canonical_object(value)]


def object_set(values: Iterable[Any]) -> Set[str]:
    return {canonical_object(value) for value in values if canonical_object(value)}


def image_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("image_id", "")).strip()
    try:
        return str(int(raw))
    except (TypeError, ValueError):
        return raw


def per_sample_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    generated = object_list(row.get("mscoco_generated_words", []))
    generated_unique = set(generated)
    gt = object_set(row.get("mscoco_gt_words", []))
    hallucinated = object_list(row.get("mscoco_hallucinated_words", []))
    if not hallucinated and generated:
        hallucinated = [obj for obj in generated if obj not in gt]
    supported = {obj for obj in generated if obj in gt}
    words = row.get("words")
    if isinstance(words, list):
        n_words = len(words)
    else:
        n_words = len(str(row.get("caption", "")).split())

    precision = safe_div(float(len(supported)), float(len(generated_unique)))
    recall = safe_div(float(len(supported)), float(len(gt)))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "caption": str(row.get("caption", "")),
        "n_words": int(n_words),
        "n_gt_objects": int(len(gt)),
        "n_generated_instances": int(len(generated)),
        "n_generated_unique": int(len(generated_unique)),
        "n_supported_unique": int(len(supported)),
        "n_hallucinated_instances": int(len(hallucinated)),
        "n_duplicate_object_mentions": int(len(generated) - len(generated_unique)),
        "chair_s": float(bool(hallucinated)),
        "chair_i": safe_div(float(len(hallucinated)), float(len(generated))),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def load_sentence_map(path: str) -> Dict[str, Dict[str, Any]]:
    payload = json.load(open(path, "r", encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("sentences", []):
        sid = image_id(row)
        if sid:
            out[sid] = row
    return out


def build_rows(
    baseline_chair_json: str,
    intervention_chair_json: str,
    chair_i_eps: float,
    chair_s_eps: float,
    min_recall_gain: float,
    require_f1_nondecrease: bool,
) -> List[Dict[str, Any]]:
    base_map = load_sentence_map(baseline_chair_json)
    int_map = load_sentence_map(intervention_chair_json)
    rows: List[Dict[str, Any]] = []
    for sid in sorted(set(base_map) & set(int_map), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        b = per_sample_metrics(base_map[sid])
        i = per_sample_metrics(int_map[sid])
        hall_cost = max(0.0, float(b["n_hallucinated_instances"] - i["n_hallucinated_instances"]))
        chair_i_cost = float(b["chair_i"] - i["chair_i"])
        chair_s_cost = float(b["chair_s"] - i["chair_s"])
        recall_gain = float(b["recall"] - i["recall"])
        f1_gain = float(b["f1"] - i["f1"])
        supported_gain = float(b["n_supported_unique"] - i["n_supported_unique"])
        generated_unique_gain = float(b["n_generated_unique"] - i["n_generated_unique"])
        teacher_positive = (
            recall_gain > float(min_recall_gain)
            and chair_i_cost <= float(chair_i_eps)
            and chair_s_cost <= float(chair_s_eps)
            and (not require_f1_nondecrease or f1_gain >= -1e-12)
        )
        row: Dict[str, Any] = {
            "image_id": sid,
            "teacher_positive": int(teacher_positive),
            "pair_recall_gain": recall_gain,
            "pair_f1_gain": f1_gain,
            "pair_supported_gain": supported_gain,
            "pair_generated_unique_gain": generated_unique_gain,
            "pair_hall_cost": hall_cost,
            "pair_chair_i_cost": chair_i_cost,
            "pair_chair_s_cost": chair_s_cost,
            "pair_recall_gain_per_hall_cost": recall_gain / (1.0 + hall_cost),
            "pair_supported_gain_per_hall_cost": supported_gain / (1.0 + hall_cost),
            "pair_f1_gain_per_hall_cost": f1_gain / (1.0 + hall_cost),
            "int_unique_per_word": safe_div(float(i["n_generated_unique"]), float(i["n_words"])),
            "int_duplicate_rate": safe_div(float(i["n_duplicate_object_mentions"]), float(i["n_generated_instances"])),
        }
        for key, value in b.items():
            if key != "caption":
                row[f"base_{key}"] = value
        for key, value in i.items():
            if key != "caption":
                row[f"int_{key}"] = value
        row["base_caption"] = b["caption"]
        row["int_caption"] = i["caption"]
        rows.append(row)
    return rows


def aggregate_counts(rows: Sequence[Dict[str, Any]], route_fn: Callable[[Dict[str, Any]], str]) -> Dict[str, Any]:
    n_caps = 0
    n_hall_caps = 0.0
    n_hall_instances = 0.0
    n_generated_instances = 0.0
    n_generated_unique = 0.0
    n_gt_objects = 0.0
    n_supported_unique = 0.0
    n_words = 0.0
    baseline_count = 0
    for row in rows:
        route = route_fn(row)
        prefix = "base_" if route == "baseline" else "int_"
        baseline_count += int(route == "baseline")
        n_caps += 1
        n_hall_caps += float(row[f"{prefix}chair_s"])
        n_hall_instances += float(row[f"{prefix}n_hallucinated_instances"])
        n_generated_instances += float(row[f"{prefix}n_generated_instances"])
        n_generated_unique += float(row[f"{prefix}n_generated_unique"])
        n_gt_objects += float(row[f"{prefix}n_gt_objects"])
        n_supported_unique += float(row[f"{prefix}n_supported_unique"])
        n_words += float(row[f"{prefix}n_words"])
    precision = safe_div(n_supported_unique, n_generated_unique)
    recall = safe_div(n_supported_unique, n_gt_objects)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "n_eval": int(n_caps),
        "baseline_rate": safe_div(float(baseline_count), float(n_caps)),
        "method_rate": 1.0 - safe_div(float(baseline_count), float(n_caps)),
        "chair_s": safe_div(n_hall_caps, float(n_caps)),
        "chair_i": safe_div(n_hall_instances, n_generated_instances),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "len_words": safe_div(n_words, float(n_caps)),
        "avg_generated_unique_objects": safe_div(n_generated_unique, float(n_caps)),
        "avg_supported_unique_objects": safe_div(n_supported_unique, float(n_caps)),
        "avg_hallucinated_object_mentions": safe_div(n_hall_instances, float(n_caps)),
    }


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(int(x) for x in labels)
    n_neg = len(labels) - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(int(label) for _, label in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def quantile_thresholds(values: Sequence[float], quantiles: Sequence[float]) -> List[float]:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return []
    out = set()
    n = len(vals)
    for q in quantiles:
        qq = min(1.0, max(0.0, float(q)))
        pos = qq * float(n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.add(vals[lo])
        else:
            w = pos - float(lo)
            out.add((1.0 - w) * vals[lo] + w * vals[hi])
    return sorted(out)


def score_row(row: Dict[str, Any], feature: str) -> float:
    return float(row.get(feature, 0.0))


def evaluate_feature(
    rows: Sequence[Dict[str, Any]],
    feature: str,
    max_baseline_rate: float,
    chair_i_eps_vs_int: float,
    chair_s_eps_vs_int: float,
    min_recall_gain_vs_int: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scores = [score_row(row, feature) for row in rows]
    labels = [int(row["teacher_positive"]) for row in rows]
    thresholds = quantile_thresholds(scores, [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    baseline = aggregate_counts(rows, lambda _: "baseline")
    intervention = aggregate_counts(rows, lambda _: "method")
    sweep: List[Dict[str, Any]] = []
    for tau in thresholds:
        selected = [row for row in rows if score_row(row, feature) >= tau]
        result = aggregate_counts(rows, lambda row, tau=tau: "baseline" if score_row(row, feature) >= tau else "method")
        tp = sum(int(row["teacher_positive"]) for row in selected)
        result.update(
            {
                "feature": feature,
                "tau": float(tau),
                "teacher_precision": safe_div(float(tp), float(len(selected))),
                "teacher_recall": safe_div(float(tp), float(sum(labels))),
                "delta_chair_i_vs_int": float(result["chair_i"] - intervention["chair_i"]),
                "delta_chair_s_vs_int": float(result["chair_s"] - intervention["chair_s"]),
                "delta_recall_vs_int": float(result["recall"] - intervention["recall"]),
                "delta_f1_vs_int": float(result["f1"] - intervention["f1"]),
                "feasible": int(
                    result["baseline_rate"] <= float(max_baseline_rate)
                    and result["chair_i"] <= float(intervention["chair_i"] + chair_i_eps_vs_int)
                    and result["chair_s"] <= float(intervention["chair_s"] + chair_s_eps_vs_int)
                    and result["recall"] >= float(intervention["recall"] + min_recall_gain_vs_int)
                ),
            }
        )
        sweep.append(result)
    feasible = [row for row in sweep if int(row["feasible"]) == 1]
    candidates = feasible or sweep
    best = sorted(
        candidates,
        key=lambda row: (
            float(row["delta_recall_vs_int"]),
            float(row["delta_f1_vs_int"]),
            -float(row["delta_chair_i_vs_int"]),
            -float(row["baseline_rate"]),
        ),
        reverse=True,
    )[0]
    return sweep, {
        "feature": feature,
        "teacher_auroc": auroc(scores, labels),
        "best": best,
        "used_feasible": bool(feasible),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate pairwise CHAIR rollback features for recall recovery.")
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--chair_i_eps", type=float, default=0.01)
    ap.add_argument("--chair_s_eps", type=float, default=0.0)
    ap.add_argument("--min_recall_gain", type=float, default=0.0)
    ap.add_argument("--require_f1_nondecrease", action="store_true")
    ap.add_argument("--max_baseline_rate", type=float, default=0.08)
    ap.add_argument("--feature", action="append", default=[])
    args = ap.parse_args()

    features = args.feature or [
        "pair_recall_gain_per_hall_cost",
        "pair_supported_gain_per_hall_cost",
        "pair_recall_gain",
        "pair_supported_gain",
        "pair_f1_gain_per_hall_cost",
        "pair_f1_gain",
    ]
    rows = build_rows(
        os.path.abspath(args.baseline_chair_json),
        os.path.abspath(args.intervention_chair_json),
        chair_i_eps=float(args.chair_i_eps),
        chair_s_eps=float(args.chair_s_eps),
        min_recall_gain=float(args.min_recall_gain),
        require_f1_nondecrease=bool(args.require_f1_nondecrease),
    )
    if not rows:
        raise RuntimeError("No overlapping CHAIR rows.")

    baseline = aggregate_counts(rows, lambda _: "baseline")
    intervention = aggregate_counts(rows, lambda _: "method")
    oracle = aggregate_counts(rows, lambda row: "baseline" if int(row["teacher_positive"]) == 1 else "method")
    feature_summaries: List[Dict[str, Any]] = []
    all_sweeps: List[Dict[str, Any]] = []
    for feature in features:
        sweep, summary = evaluate_feature(
            rows,
            feature,
            max_baseline_rate=float(args.max_baseline_rate),
            chair_i_eps_vs_int=float(args.chair_i_eps),
            chair_s_eps_vs_int=float(args.chair_s_eps),
            min_recall_gain_vs_int=float(args.min_recall_gain),
        )
        feature_summaries.append(summary)
        all_sweeps.extend(sweep)
    best_summary = sorted(
        feature_summaries,
        key=lambda item: (
            bool(item["used_feasible"]),
            float(item["best"]["delta_recall_vs_int"]),
            float(item["best"]["delta_f1_vs_int"]),
            -float(item["best"]["delta_chair_i_vs_int"]),
        ),
        reverse=True,
    )[0]

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "decision_rows.csv"), rows)
    write_csv(os.path.join(out_dir, "tau_sweep.csv"), all_sweeps)
    summary = {
        "inputs": {
            "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
            "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
            "chair_i_eps": float(args.chair_i_eps),
            "chair_s_eps": float(args.chair_s_eps),
            "min_recall_gain": float(args.min_recall_gain),
            "require_f1_nondecrease": bool(args.require_f1_nondecrease),
            "max_baseline_rate": float(args.max_baseline_rate),
            "features": features,
        },
        "counts": {
            "n_rows": len(rows),
            "teacher_positive_rate": safe_div(float(sum(int(r["teacher_positive"]) for r in rows)), float(len(rows))),
        },
        "baseline": baseline,
        "intervention": intervention,
        "teacher_oracle": oracle,
        "feature_summaries": feature_summaries,
        "best_feature": best_summary,
        "outputs": {
            "decision_rows_csv": os.path.join(out_dir, "decision_rows.csv"),
            "tau_sweep_csv": os.path.join(out_dir, "tau_sweep.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
