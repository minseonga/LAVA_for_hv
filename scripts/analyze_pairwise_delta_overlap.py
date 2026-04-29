#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


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


def label(row: Dict[str, str], label_key: str, text_key: str) -> str:
    out = str(row.get(label_key, "")).strip().lower()
    if out in {"yes", "no"}:
        return out
    return parse_yes_no(row.get(text_key, ""))


def is_route_candidate(row: Dict[str, str], mode: str) -> bool:
    if mode == "all":
        return True
    b = label(row, "baseline_label", "baseline_text")
    v = label(row, "intervention_label", "intervention_text")
    if mode == "changed_answer":
        return b in {"yes", "no"} and v in {"yes", "no"} and b != v
    if mode == "yes_to_no":
        return b == "yes" and v == "no"
    if mode == "no_to_yes":
        return b == "no" and v == "yes"
    raise ValueError(f"Unsupported candidate_filter={mode!r}")


def choose_policy(bundle: Dict[str, Any], family: str) -> Dict[str, Any]:
    if family != "selected":
        best = bundle.get("best_results") or {}
        if family not in best:
            raise RuntimeError(f"family={family!r} is not available in policy bundle")
        return dict(best[family])
    if isinstance(bundle.get("selected_policy"), dict):
        return dict(bundle["selected_policy"])
    if isinstance(bundle.get("policy"), dict) and isinstance(bundle["policy"].get("applied_policy"), dict):
        return dict(bundle["policy"]["applied_policy"])
    if {"family", "alpha", "tau"}.issubset(bundle):
        return dict(bundle)
    raise RuntimeError("Could not find selected policy in policy JSON")


def feature_list(bundle: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = bundle.get(key)
    if isinstance(value, list):
        return [dict(x) for x in value if isinstance(x, dict)]
    policy = bundle.get("policy")
    if isinstance(policy, dict) and isinstance(policy.get(key), list):
        return [dict(x) for x in policy[key] if isinstance(x, dict)]
    return []


def oriented_z(row: Dict[str, str], feat: Dict[str, Any]) -> Optional[float]:
    name = str(feat.get("feature", "")).strip()
    value = maybe_float(row.get(name))
    mu = maybe_float(feat.get("mu"))
    sd = maybe_float(feat.get("sd"))
    if value is None or mu is None or sd is None or abs(sd) < 1e-12:
        return None
    z = (float(value) - float(mu)) / float(sd)
    direction = str(feat.get("direction", "high")).strip().lower()
    return float(-z if direction == "low" else z)


def mean_z(row: Dict[str, str], features: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not features:
        return None
    vals: List[float] = []
    for feat in features:
        z = oriented_z(row, feat)
        if z is None:
            return None
        vals.append(float(z))
    return sum(vals) / float(len(vals)) if vals else None


def score_row(row: Dict[str, str], policy: Dict[str, Any], c_features: Sequence[Dict[str, Any]], d_features: Sequence[Dict[str, Any]]) -> Optional[float]:
    family = str(policy.get("family", "")).strip()
    alpha = float(policy.get("alpha", 0.0))
    c_score = mean_z(row, c_features)
    d_score = mean_z(row, d_features)
    if family == "c_only":
        return c_score
    if family == "d_only":
        return d_score
    if family == "cd_fusion":
        if c_score is None or d_score is None:
            return None
        return float((1.0 - alpha) * c_score + alpha * d_score)
    raise RuntimeError(f"Unsupported policy family={family!r}")


def selected_by_content_policy(
    rows: Sequence[Dict[str, str]],
    bundle: Dict[str, Any],
    *,
    family: str,
    candidate_filter: str,
) -> Tuple[Set[str], Dict[str, Any]]:
    policy = choose_policy(bundle, family)
    c_features = feature_list(bundle, "selected_c_features")
    d_features = feature_list(bundle, "selected_d_features")
    tau = float(policy["tau"])
    selected: Set[str] = set()
    for row in rows:
        if not is_route_candidate(row, candidate_filter):
            continue
        score = score_row(row, policy, c_features, d_features)
        if score is not None and float(score) >= tau:
            selected.add(str(row.get("id", "")).strip())
    return selected, {
        "family": policy.get("family"),
        "alpha": policy.get("alpha"),
        "tau": tau,
        "n_c_features": len(c_features),
        "n_d_features": len(d_features),
        "c_features": [f.get("feature") for f in c_features],
        "d_features": [f.get("feature") for f in d_features],
    }


def choose_pairwise_metric(metrics: Sequence[Dict[str, str]], feature: str) -> Dict[str, str]:
    if not metrics:
        raise RuntimeError("pairwise metrics CSV is empty")
    if feature:
        for row in metrics:
            if str(row.get("feature", "")).strip() == feature:
                return row
        raise RuntimeError(f"pairwise feature not found: {feature}")
    return metrics[0]


def selected_by_pairwise_metric(delta_rows: Sequence[Dict[str, str]], metric: Dict[str, str]) -> Tuple[Set[str], Dict[str, Any]]:
    feature = str(metric["feature"]).strip()
    direction = str(metric["direction"]).strip().lower()
    tau = float(metric["oracle_tau"])
    selected: Set[str] = set()
    for row in delta_rows:
        value = maybe_float(row.get(feature))
        if value is None:
            continue
        use = value >= tau if direction == "high" else value <= tau
        if use:
            selected.add(str(row.get("id", "")).strip())
    return selected, {
        "feature": feature,
        "direction": direction,
        "tau": tau,
        "auroc": maybe_float(metric.get("auroc")),
        "oracle_best_net": maybe_int(metric.get("oracle_best_net")),
        "oracle_selected_count": maybe_int(metric.get("oracle_selected_count")),
        "oracle_selected_harm": maybe_int(metric.get("oracle_selected_harm")),
        "oracle_selected_help": maybe_int(metric.get("oracle_selected_help")),
    }


def effect_by_id(rows: Iterable[Dict[str, str]]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for row in rows:
        sid = str(row.get("id", "")).strip()
        if not sid:
            continue
        harm = maybe_int(row.get("harm"))
        help_ = maybe_int(row.get("help"))
        if harm is None or help_ is None:
            bc = maybe_int(row.get("baseline_correct"))
            ic = maybe_int(row.get("intervention_correct"))
            if bc is None or ic is None:
                harm, help_ = 0, 0
            else:
                harm, help_ = int(bc == 1 and ic == 0), int(bc == 0 and ic == 1)
        out[sid] = (int(harm or 0), int(help_ or 0))
    return out


def summarize_set(ids: Set[str], effects: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
    harm = 0
    help_ = 0
    neutral = 0
    missing = 0
    for sid in ids:
        if sid not in effects:
            missing += 1
            continue
        h, hp = effects[sid]
        harm += int(h)
        help_ += int(hp)
        neutral += int((h == 0) and (hp == 0))
    return {
        "count": len(ids),
        "harm": int(harm),
        "help": int(help_),
        "neutral": int(neutral),
        "missing_effect": int(missing),
        "net": int(harm - help_),
        "harm_precision": float(harm / max(1, len(ids))),
    }


def sorted_ids(ids: Set[str], limit: int) -> List[str]:
    def key(x: str) -> Tuple[int, str]:
        try:
            return (0, f"{int(x):012d}")
        except Exception:
            return (1, x)

    return sorted(ids, key=key)[: max(0, int(limit))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare overlap between a content/RAPIC policy and a pairwise delta oracle policy.")
    ap.add_argument("--intervention_rows_csv", required=True)
    ap.add_argument("--content_policy_json", required=True)
    ap.add_argument("--pairwise_delta_rows_csv", required=True)
    ap.add_argument("--pairwise_metrics_csv", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--content_family", default="selected", choices=["selected", "c_only", "d_only", "cd_fusion"])
    ap.add_argument("--content_candidate_filter", default="changed_answer", choices=["all", "changed_answer", "yes_to_no", "no_to_yes"])
    ap.add_argument("--pairwise_feature", default="", help="Default: first/top row in pairwise metrics CSV.")
    ap.add_argument("--print_ids", type=int, default=20)
    args = ap.parse_args()

    int_rows = read_csv_rows(os.path.abspath(args.intervention_rows_csv))
    delta_rows = read_csv_rows(os.path.abspath(args.pairwise_delta_rows_csv))
    metrics = read_csv_rows(os.path.abspath(args.pairwise_metrics_csv))
    with open(os.path.abspath(args.content_policy_json), "r", encoding="utf-8") as f:
        bundle = json.load(f)

    content_ids, content_meta = selected_by_content_policy(
        int_rows,
        bundle,
        family=str(args.content_family),
        candidate_filter=str(args.content_candidate_filter),
    )
    pair_metric = choose_pairwise_metric(metrics, str(args.pairwise_feature))
    pair_ids, pair_meta = selected_by_pairwise_metric(delta_rows, pair_metric)
    effects = effect_by_id(int_rows)

    inter = content_ids & pair_ids
    union = content_ids | pair_ids
    content_only = content_ids - pair_ids
    pair_only = pair_ids - content_ids

    result = {
        "inputs": {
            "intervention_rows_csv": os.path.abspath(args.intervention_rows_csv),
            "content_policy_json": os.path.abspath(args.content_policy_json),
            "pairwise_delta_rows_csv": os.path.abspath(args.pairwise_delta_rows_csv),
            "pairwise_metrics_csv": os.path.abspath(args.pairwise_metrics_csv),
            "content_candidate_filter": str(args.content_candidate_filter),
            "pairwise_feature_requested": str(args.pairwise_feature),
        },
        "content_policy": content_meta,
        "pairwise_policy": pair_meta,
        "sets": {
            "content": summarize_set(content_ids, effects),
            "pairwise": summarize_set(pair_ids, effects),
            "intersection": summarize_set(inter, effects),
            "content_only": summarize_set(content_only, effects),
            "pairwise_only": summarize_set(pair_only, effects),
            "union": summarize_set(union, effects),
        },
        "ids": {
            "intersection": sorted_ids(inter, int(args.print_ids)),
            "content_only": sorted_ids(content_only, int(args.print_ids)),
            "pairwise_only": sorted_ids(pair_only, int(args.print_ids)),
        },
    }

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.out_json:
        write_json(os.path.abspath(args.out_json), result)
        print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
