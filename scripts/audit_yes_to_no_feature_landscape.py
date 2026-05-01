#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


META_EXACT = {
    "",
    "id",
    "qid",
    "question_id",
    "answer_id",
    "layer_index",
    "layer_frac",
    "is_final_layer",
    "image",
    "image_id",
    "question",
    "prompt",
    "text",
    "output",
    "answer",
    "caption",
    "category",
    "object",
    "objects",
    "gt_label",
    "label",
    "baseline_label",
    "intervention_label",
    "candidate_label",
    "baseline_text",
    "intervention_text",
    "final_text",
    "route",
    "family",
    "source",
    "score_error",
    "score_error_traceback",
    "baseline_correct",
    "intervention_correct",
    "final_correct",
    "harm",
    "help",
    "neutral",
    "changed_answer",
    "route_candidate",
}

META_SUFFIXES = (
    "_correct",
    "_id",
    "_label",
    "_text",
    "_error",
    "_traceback",
)


def maybe_float(value: Any) -> Optional[float]:
    s = str("" if value is None else value).strip()
    if not s or s.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def maybe_int(value: Any) -> Optional[int]:
    x = maybe_float(value)
    if x is None:
        return None
    return int(round(x))


def kl_uniform(p: float) -> float:
    eps = 1e-12
    p = min(1.0 - eps, max(eps, float(p)))
    q = 1.0 - p
    return float(p * math.log(2.0 * p) + q * math.log(2.0 * q))


def augment_derived_features(row: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)
    for key, value in list(row.items()):
        if not str(key).endswith("candidate_prob_binary"):
            continue
        p = maybe_float(value)
        if p is None:
            continue
        derived = str(key)[: -len("candidate_prob_binary")] + "candidate_kl_uniform"
        if str(out.get(derived, "")).strip() == "":
            out[derived] = str(kl_uniform(float(p)))
    return out


def safe_id(value: Any) -> str:
    raw = str("" if value is None else value).strip()
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def parse_yes_no(text: Any) -> str:
    s = str("" if text is None else text).strip()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip().lower() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    return ""


def label_value(row: Mapping[str, Any], key: str, text_key: str) -> str:
    label = str(row.get(key, "")).strip().lower()
    if label in {"yes", "no"}:
        return label
    return parse_yes_no(row.get(text_key, ""))


def is_candidate(row: Mapping[str, Any], mode: str) -> bool:
    mode = str(mode or "yes_to_no").strip().lower()
    if mode == "all":
        return True
    base = label_value(row, "baseline_label", "baseline_text")
    intervention = label_value(row, "intervention_label", "intervention_text")
    if mode == "changed_answer":
        return base in {"yes", "no"} and intervention in {"yes", "no"} and base != intervention
    if mode == "yes_to_no":
        return base == "yes" and intervention == "no"
    raise ValueError(f"Unsupported candidate_filter={mode!r}")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return [augment_derived_features(dict(row)) for row in csv.DictReader(f)]


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(max(0.0, sum((x - mu) ** 2 for x in values) / float(len(values)))))


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(int(y) for y in labels)
    n_neg = len(labels) - n_pos
    if len(scores) != len(labels) or n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if int(pairs[k][1]) == 1:
                rank_sum += avg_rank
        i = j
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def feature_allowed(name: str, include_regex: Optional[re.Pattern[str]], exclude_regex: Optional[re.Pattern[str]]) -> bool:
    key = str(name or "").strip()
    if key in META_EXACT:
        return False
    if key.endswith("_label_lp"):
        pass
    elif any(key.endswith(suffix) for suffix in META_SUFFIXES):
        return False
    if key == "layer_index":
        return False
    if include_regex and not include_regex.search(key):
        return False
    if exclude_regex and exclude_regex.search(key):
        return False
    return True


def numeric_feature_names(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_regex: Optional[re.Pattern[str]],
    exclude_regex: Optional[re.Pattern[str]],
    min_present: int,
) -> List[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if not feature_allowed(key, include_regex, exclude_regex):
                continue
            if maybe_float(value) is not None:
                counts[str(key)] += 1
    return sorted([key for key, count in counts.items() if count >= int(min_present)])


def threshold_grid(values: Sequence[float]) -> List[float]:
    finite = sorted(float(x) for x in values if math.isfinite(float(x)))
    return sorted(set(finite))


def best_tail(oriented_scores: Sequence[float], labels: Sequence[int], *, min_selected_count: int) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    pairs = list(zip([float(x) for x in oriented_scores], [int(y) for y in labels]))
    total_harm = sum(labels)
    total_help = len(labels) - total_harm
    for tau in threshold_grid(oriented_scores):
        selected = [(score, y) for score, y in pairs if score >= float(tau)]
        if len(selected) < int(min_selected_count):
            continue
        harm = sum(int(y) for _, y in selected)
        help_ = len(selected) - harm
        result = {
            "tau": float(tau),
            "selected_count": int(len(selected)),
            "selected_harm": int(harm),
            "selected_help": int(help_),
            "net": int(harm - help_),
            "selected_harm_precision": float(harm / max(1, len(selected))),
            "selected_harm_recall_in_scope": float(harm / max(1, total_harm)),
            "selected_help_recall_in_scope": float(help_ / max(1, total_help)),
        }
        key = (
            int(result["net"]),
            int(result["selected_harm"]),
            -int(result["selected_help"]),
            -int(result["selected_count"]),
        )
        if best is None or key > (
            int(best["net"]),
            int(best["selected_harm"]),
            -int(best["selected_help"]),
            -int(best["selected_count"]),
        ):
            best = result
    if best is not None:
        return best
    return {
        "tau": "",
        "selected_count": 0,
        "selected_harm": 0,
        "selected_help": 0,
        "net": 0,
        "selected_harm_precision": 0.0,
        "selected_harm_recall_in_scope": 0.0,
        "selected_help_recall_in_scope": 0.0,
    }


def top_frac_summary(oriented_scores: Sequence[float], labels: Sequence[int], frac: float) -> Dict[str, Any]:
    n = len(oriented_scores)
    k = max(1, int(round(float(frac) * float(n)))) if n else 0
    pairs = sorted(zip([float(x) for x in oriented_scores], [int(y) for y in labels]), key=lambda x: x[0], reverse=True)[:k]
    harm = sum(y for _, y in pairs)
    help_ = len(pairs) - harm
    return {
        f"top{int(round(100 * frac)):02d}_count": int(len(pairs)),
        f"top{int(round(100 * frac)):02d}_harm": int(harm),
        f"top{int(round(100 * frac)):02d}_help": int(help_),
        f"top{int(round(100 * frac)):02d}_net": int(harm - help_),
        f"top{int(round(100 * frac)):02d}_precision": float(harm / max(1, len(pairs))),
    }


def evaluate_feature(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    feature: str,
    layer_index: str,
    candidate_filter: str,
    min_selected_count: int,
    top_fracs: Sequence[float],
) -> Optional[Dict[str, Any]]:
    xs: List[float] = []
    ys: List[int] = []
    categories: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    for row in rows:
        if not is_candidate(row, candidate_filter):
            continue
        x = maybe_float(row.get(feature))
        harm = maybe_int(row.get("harm"))
        help_ = maybe_int(row.get("help"))
        if x is None or harm not in {0, 1} or help_ not in {0, 1}:
            continue
        if int(harm) == 0 and int(help_) == 0:
            continue
        base = label_value(row, "baseline_label", "baseline_text")
        intervention = label_value(row, "intervention_label", "intervention_text")
        xs.append(float(x))
        ys.append(int(harm))
        categories[str(row.get("category", "") or "unknown")] += 1
        transitions[f"{base}->{intervention}"] += 1
    if len(xs) < 2 or sum(ys) == 0 or sum(ys) == len(ys):
        return None
    auc_high = binary_auroc(xs, ys)
    auc_low = binary_auroc([-x for x in xs], ys)
    if auc_high is None or auc_low is None:
        return None
    direction = "high" if float(auc_high) >= float(auc_low) else "low"
    oriented = [x if direction == "high" else -x for x in xs]
    tail = best_tail(oriented, ys, min_selected_count=min_selected_count)
    harm_values = [x for x, y in zip(xs, ys) if int(y) == 1]
    help_values = [x for x, y in zip(xs, ys) if int(y) == 0]
    out: Dict[str, Any] = {
        "family": family,
        "layer_index": layer_index,
        "feature": feature,
        "n": int(len(xs)),
        "n_harm": int(sum(ys)),
        "n_help": int(len(ys) - sum(ys)),
        "auroc": float(max(float(auc_high), float(auc_low))),
        "direction": direction,
        "raw_auroc_high": float(auc_high),
        "harm_mean": mean(harm_values),
        "help_mean": mean(help_values),
        "harm_std": std(harm_values),
        "help_std": std(help_values),
        "category_counts": json.dumps(dict(categories), ensure_ascii=False, sort_keys=True),
        "transition_counts": json.dumps(dict(transitions), ensure_ascii=False, sort_keys=True),
        **tail,
    }
    for frac in top_fracs:
        out.update(top_frac_summary(oriented, ys, float(frac)))
    return out


def group_rows(rows: Sequence[Mapping[str, Any]], layerwise: bool) -> Dict[str, List[Dict[str, Any]]]:
    if not layerwise:
        return {"": [dict(row) for row in rows]}
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        layer = str(row.get("layer_index", "")).strip()
        if layer == "":
            continue
        out[layer].append(dict(row))
    return dict(out)


def evaluate_source(
    *,
    family: str,
    path: str,
    layerwise: bool,
    candidate_filter: str,
    min_present: int,
    min_selected_count: int,
    top_fracs: Sequence[float],
    include_regex: Optional[re.Pattern[str]],
    exclude_regex: Optional[re.Pattern[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_csv_rows(path)
    metrics: List[Dict[str, Any]] = []
    counts = {
        "family": family,
        "path": os.path.abspath(path),
        "n_rows": len(rows),
        "n_candidates": sum(1 for row in rows if is_candidate(row, candidate_filter)),
    }
    for layer_index, layer_rows in group_rows(rows, layerwise).items():
        features = numeric_feature_names(
            layer_rows,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            min_present=min_present,
        )
        for feature in features:
            result = evaluate_feature(
                layer_rows,
                family=family,
                feature=feature,
                layer_index=layer_index,
                candidate_filter=candidate_filter,
                min_selected_count=min_selected_count,
                top_fracs=top_fracs,
            )
            if result is not None:
                metrics.append(result)
    return metrics, counts


def parse_source_specs(args: argparse.Namespace) -> List[Tuple[str, str, bool]]:
    specs: List[Tuple[str, str, bool]] = []
    fixed = [
        ("d_layered", args.d_trajectory_long_csv, True),
        ("c_vis", args.c_vis_rows_csv, False),
        ("object_token", args.object_trajectory_long_csv, True),
        ("visual_grounding", args.visual_grounding_long_csv, True),
        ("content_final_d", args.online_feature_rows_csv, False),
    ]
    for family, path, layerwise in fixed:
        if str(path or "").strip():
            specs.append((family, str(path), bool(layerwise)))
    for item in args.source_csv:
        parts = str(item).split("=", 1)
        if len(parts) != 2:
            raise ValueError("--source_csv must be formatted as family=/path/to/file.csv")
        family, rest = parts[0].strip(), parts[1].strip()
        layerwise = False
        if rest.endswith(":layerwise"):
            rest = rest[: -len(":layerwise")]
            layerwise = True
        specs.append((family, rest, layerwise))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit feature landscape under POPE yes->no suppression scope.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_filter", default="yes_to_no", choices=["all", "changed_answer", "yes_to_no"])
    ap.add_argument("--d_trajectory_long_csv", default="")
    ap.add_argument("--c_vis_rows_csv", default="")
    ap.add_argument("--object_trajectory_long_csv", default="")
    ap.add_argument("--visual_grounding_long_csv", default="")
    ap.add_argument("--online_feature_rows_csv", default="")
    ap.add_argument(
        "--source_csv",
        action="append",
        default=[],
        help="Additional source as family=/path.csv or family=/path.csv:layerwise.",
    )
    ap.add_argument("--include_regex", default="")
    ap.add_argument("--exclude_regex", default="")
    ap.add_argument("--min_present", type=int, default=20)
    ap.add_argument("--min_selected_count", type=int, default=5)
    ap.add_argument("--top_k_per_family", type=int, default=40)
    ap.add_argument("--top_fracs", default="0.05,0.10,0.20,0.30")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    include_regex = re.compile(str(args.include_regex)) if str(args.include_regex or "").strip() else None
    exclude_regex = re.compile(str(args.exclude_regex)) if str(args.exclude_regex or "").strip() else None
    top_fracs = [float(x.strip()) for x in str(args.top_fracs).split(",") if x.strip()]

    all_metrics: List[Dict[str, Any]] = []
    source_counts: List[Dict[str, Any]] = []
    for family, path, layerwise in parse_source_specs(args):
        if not os.path.isfile(os.path.abspath(path)):
            raise FileNotFoundError(path)
        metrics, counts = evaluate_source(
            family=family,
            path=path,
            layerwise=layerwise,
            candidate_filter=str(args.candidate_filter),
            min_present=int(args.min_present),
            min_selected_count=int(args.min_selected_count),
            top_fracs=top_fracs,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
        )
        all_metrics.extend(metrics)
        source_counts.append(counts)

    all_metrics.sort(
        key=lambda row: (
            -float(row.get("auroc", 0.0)),
            -int(row.get("net", 0)),
            -float(row.get("selected_harm_precision", 0.0)),
            str(row.get("family", "")),
            str(row.get("layer_index", "")),
            str(row.get("feature", "")),
        )
    )

    top_by_family: List[Dict[str, Any]] = []
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_metrics:
        by_family[str(row.get("family", ""))].append(row)
    for family in sorted(by_family):
        top_by_family.extend(by_family[family][: int(args.top_k_per_family)])

    write_csv(os.path.join(out_dir, "feature_landscape.csv"), all_metrics)
    write_csv(os.path.join(out_dir, "top_by_family.csv"), top_by_family)
    write_json(
        os.path.join(out_dir, "summary.json"),
        {
            "mode": "yes_to_no_feature_landscape",
            "candidate_filter": str(args.candidate_filter),
            "inputs": {
                "d_trajectory_long_csv": os.path.abspath(args.d_trajectory_long_csv)
                if str(args.d_trajectory_long_csv or "").strip()
                else "",
                "c_vis_rows_csv": os.path.abspath(args.c_vis_rows_csv) if str(args.c_vis_rows_csv or "").strip() else "",
                "object_trajectory_long_csv": os.path.abspath(args.object_trajectory_long_csv)
                if str(args.object_trajectory_long_csv or "").strip()
                else "",
                "visual_grounding_long_csv": os.path.abspath(args.visual_grounding_long_csv)
                if str(args.visual_grounding_long_csv or "").strip()
                else "",
                "online_feature_rows_csv": os.path.abspath(args.online_feature_rows_csv)
                if str(args.online_feature_rows_csv or "").strip()
                else "",
                "source_csv": list(args.source_csv),
            },
            "settings": {
                "include_regex": str(args.include_regex),
                "exclude_regex": str(args.exclude_regex),
                "min_present": int(args.min_present),
                "min_selected_count": int(args.min_selected_count),
                "top_k_per_family": int(args.top_k_per_family),
                "top_fracs": top_fracs,
            },
            "sources": source_counts,
            "n_features_evaluated": int(len(all_metrics)),
            "top_overall": all_metrics[: int(args.top_k_per_family)],
        },
    )
    print("[saved]", os.path.join(out_dir, "feature_landscape.csv"))
    print("[saved]", os.path.join(out_dir, "top_by_family.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
