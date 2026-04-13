#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METRICS = ["chair_i", "chair_s", "recall", "precision", "f1"]


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    if fieldnames:
        cols = list(fieldnames)
    else:
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int_flag(value: Any) -> int:
    return int(str(value).strip() in {"1", "true", "True", "yes"})


def norm_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("id") or row.get("image_id") or row.get("question_id") or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def load_feature_rows(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(os.path.abspath(path)):
        sid = norm_id(row)
        if sid:
            out[sid] = row
    return out


def join_rows(net_harm_csv: str, feature_csv: str) -> List[Dict[str, str]]:
    features = load_feature_rows(feature_csv)
    rows: List[Dict[str, str]] = []
    for row in read_csv_rows(os.path.abspath(net_harm_csv)):
        sid = norm_id(row)
        if not sid or sid not in features:
            continue
        merged = dict(row)
        for key, value in features[sid].items():
            if key in {"id", "image_id", "question_id"}:
                continue
            merged[key] = value
        rows.append(merged)
    return rows


def is_numeric_feature(rows: Sequence[Dict[str, str]], feature: str, min_valid: int) -> bool:
    values = [safe_float(row.get(feature)) for row in rows]
    values = [v for v in values if v is not None]
    return len(values) >= min_valid and len(set(round(v, 12) for v in values)) >= 3


def discover_features(rows: Sequence[Dict[str, str]], feature_prefix: str, requested: Sequence[str]) -> List[str]:
    if requested:
        return list(requested)
    if not rows:
        return []
    excluded_exact = {"id", "image", "image_id", "question_id"}
    excluded_substrings = ("text", "units", "caption")
    min_valid = max(10, int(0.8 * len(rows)))
    out: List[str] = []
    for feature in rows[0]:
        if feature in excluded_exact:
            continue
        if feature_prefix and not feature.startswith(feature_prefix):
            continue
        if any(part in feature for part in excluded_substrings):
            continue
        if is_numeric_feature(rows, feature, min_valid):
            out.append(feature)
    return out


def auc_high(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    good = 0.0
    total = 0
    for a in pos:
        for b in neg:
            total += 1
            if a > b:
                good += 1.0
            elif a == b:
                good += 0.5
    return good / float(total) if total else None


def ap_score(items: Sequence[Tuple[int, float]]) -> Optional[float]:
    ranked = sorted(items, key=lambda x: x[1], reverse=True)
    n_pos = sum(label for label, _ in ranked)
    if n_pos <= 0:
        return None
    hits = 0
    total = 0.0
    for rank, (label, score) in enumerate(ranked, start=1):
        del score
        if label:
            hits += 1
            total += hits / float(rank)
    return total / float(n_pos)


def feature_metrics(rows: Sequence[Dict[str, str]], features: Sequence[str], target_col: str) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for feature in features:
        pairs: List[Tuple[int, float]] = []
        for row in rows:
            val = safe_float(row.get(feature))
            if val is None:
                continue
            pairs.append((safe_int_flag(row.get(target_col)), val))
        if not pairs:
            continue
        pos = [val for label, val in pairs if label]
        neg = [val for label, val in pairs if not label]
        auc = auc_high(pos, neg)
        if auc is None:
            continue
        direction = "high" if auc >= 0.5 else "low"
        oriented = [(label, val if direction == "high" else -val) for label, val in pairs]
        metrics.append(
            {
                "feature": feature,
                "direction": direction,
                "auroc": max(auc, 1.0 - auc),
                "auroc_high": auc,
                "ap": ap_score(oriented),
                "n": len(pairs),
                "n_pos": sum(label for label, _ in pairs),
                "pos_mean": sum(pos) / float(len(pos)) if pos else "",
                "neg_mean": sum(neg) / float(len(neg)) if neg else "",
            }
        )
    metrics.sort(key=lambda x: (float(x.get("auroc") or 0.0), float(x.get("ap") or 0.0)), reverse=True)
    return metrics


RuleSpec = Tuple[str, str, float]


def threshold_mask(rows: Sequence[Dict[str, str]], spec: RuleSpec) -> List[bool]:
    feature, direction, tau = spec
    mask = []
    for row in rows:
        val = safe_float(row.get(feature))
        if val is None:
            mask.append(False)
        elif direction == "low":
            mask.append(val <= tau)
        else:
            mask.append(val >= tau)
    return mask


def make_rule_specs(
    rows: Sequence[Dict[str, str]],
    features: Sequence[str],
    quantiles: Sequence[float],
    min_valid: int,
) -> List[RuleSpec]:
    specs: List[RuleSpec] = []
    seen = set()
    for feature in features:
        values = sorted(v for row in rows if (v := safe_float(row.get(feature))) is not None)
        if len(values) < max(1, int(min_valid)) or len(set(round(v, 12) for v in values)) < 3:
            continue
        for direction in ("low", "high"):
            for q in quantiles:
                qq = q if direction == "low" else 1.0 - q
                idx = int(max(0.0, min(1.0, qq)) * (len(values) - 1))
                tau = values[max(0, min(len(values) - 1, idx))]
                key = (feature, direction, round(tau, 10))
                if key in seen:
                    continue
                seen.add(key)
                specs.append((feature, direction, tau))
    return specs


def intervention_means(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    return {
        metric: sum(safe_float(row.get(f"intervention_{metric}")) or 0.0 for row in rows) / float(max(1, len(rows)))
        for metric in METRICS
    }


def eval_mask(
    rows: Sequence[Dict[str, str]],
    mask: Sequence[bool],
    target_col: str,
    safe_col: str,
) -> Dict[str, Any]:
    int_means = intervention_means(rows)
    sums = {metric: 0.0 for metric in METRICS}
    selected_ids: List[str] = []
    counts = {
        "n_selected": 0,
        "n_target": 0,
        "n_safe": 0,
        "n_ignore_v2": 0,
        "n_parser_artifact": 0,
        "n_type_o": 0,
        "n_type_r": 0,
    }
    total_target = sum(safe_int_flag(row.get(target_col)) for row in rows)
    for use, row in zip(mask, rows):
        if use:
            counts["n_selected"] += 1
            selected_ids.append(norm_id(row))
            counts["n_target"] += safe_int_flag(row.get(target_col))
            counts["n_safe"] += safe_int_flag(row.get(safe_col))
            counts["n_ignore_v2"] += safe_int_flag(row.get("net_ignore_v2"))
            counts["n_parser_artifact"] += safe_int_flag(row.get("parser_artifact_only_sample"))
            counts["n_type_o"] += safe_int_flag(row.get("net_harm_type_o"))
            counts["n_type_r"] += safe_int_flag(row.get("net_harm_type_r"))
        for metric in METRICS:
            base_value = safe_float(row.get(f"baseline_{metric}"))
            int_value = safe_float(row.get(f"intervention_{metric}"))
            sums[metric] += base_value if use and base_value is not None else (int_value or 0.0)
    means = {metric: sums[metric] / float(max(1, len(rows))) for metric in METRICS}
    deltas = {f"delta_{metric}_vs_int": means[metric] - int_means[metric] for metric in METRICS}
    target_precision = counts["n_target"] / float(counts["n_selected"]) if counts["n_selected"] else 0.0
    target_recall = counts["n_target"] / float(total_target) if total_target else 0.0
    score = (
        1.0 * deltas["delta_f1_vs_int"]
        + 0.5 * deltas["delta_recall_vs_int"]
        - 2.0 * max(0.0, deltas["delta_chair_i_vs_int"])
        - 0.7 * max(0.0, deltas["delta_chair_s_vs_int"])
        + 0.01 * target_precision
    )
    return {
        **counts,
        **means,
        **deltas,
        "target_precision": target_precision,
        "target_recall": target_recall,
        "target_rate": total_target / float(max(1, len(rows))),
        "score": score,
        "selected_ids": "|".join(selected_ids[:100]),
    }


def rule_name(specs: Sequence[RuleSpec]) -> str:
    return " & ".join(f"{f} {d} {t:.8g}" for f, d, t in specs)


def mask_for_rule(rows: Sequence[Dict[str, str]], specs: Sequence[RuleSpec]) -> List[bool]:
    masks = [threshold_mask(rows, spec) for spec in specs]
    if not masks:
        return [False] * len(rows)
    return [all(parts) for parts in zip(*masks)]


def search_rules(
    rows: Sequence[Dict[str, str]],
    features: Sequence[str],
    target_col: str,
    safe_col: str,
    quantiles: Sequence[float],
    max_combo_size: int,
    max_rule_specs: int,
    require_benefit_cost_combo: bool,
    benefit_prefix: str,
    cost_prefix: str,
    min_feature_valid: int,
    constraints: Dict[str, float],
    top_k: int,
) -> List[Dict[str, Any]]:
    specs = make_rule_specs(rows, features, quantiles, int(min_feature_valid))
    scored_specs: List[Tuple[float, RuleSpec]] = []
    for spec in specs:
        res = eval_mask(rows, threshold_mask(rows, spec), target_col, safe_col)
        scored_specs.append((float(res["score"]), spec))
    if max_rule_specs > 0 and len(scored_specs) > max_rule_specs:
        specs = [spec for _, spec in sorted(scored_specs, reverse=True)[:max_rule_specs]]
    candidates: List[Dict[str, Any]] = []

    def consider(rule_specs: Sequence[RuleSpec]) -> None:
        if require_benefit_cost_combo:
            has_benefit = any(spec[0].startswith(benefit_prefix) for spec in rule_specs)
            has_cost = any(spec[0].startswith(cost_prefix) for spec in rule_specs)
            if not (has_benefit and has_cost):
                return
        mask = mask_for_rule(rows, rule_specs)
        result = eval_mask(rows, mask, target_col, safe_col)
        if result["n_selected"] < constraints["min_selected"]:
            return
        if result["n_selected"] > constraints["max_selected"]:
            return
        if result["delta_f1_vs_int"] < constraints["min_delta_f1"]:
            return
        if result["delta_recall_vs_int"] < constraints["min_delta_recall"]:
            return
        if result["delta_chair_i_vs_int"] > constraints["max_delta_chair_i"]:
            return
        if result["delta_chair_s_vs_int"] > constraints["max_delta_chair_s"]:
            return
        if result["target_precision"] < constraints["min_target_precision"]:
            return
        candidates.append(
            {
                "rule": rule_name(rule_specs),
                "policy_specs": json.dumps(
                    [{"feature": f, "direction": d, "tau": t} for f, d, t in rule_specs],
                    sort_keys=True,
                ),
                **result,
            }
        )

    for spec in specs:
        consider([spec])
    if max_combo_size >= 2:
        for combo in combinations(specs, 2):
            if len({spec[0] for spec in combo}) != len(combo):
                continue
            consider(combo)
    if max_combo_size >= 3:
        # Bound the cubic pass to the most promising univariate specs.
        scored_specs = []
        for spec in specs:
            res = eval_mask(rows, threshold_mask(rows, spec), target_col, safe_col)
            scored_specs.append((float(res["score"]), spec))
        for combo in combinations([spec for _, spec in sorted(scored_specs, reverse=True)[:80]], 3):
            if len({spec[0] for spec in combo}) != len(combo):
                continue
            consider(combo)
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    return candidates[:top_k]


def apply_policy(rows: Sequence[Dict[str, str]], policy: Dict[str, Any], target_col: str, safe_col: str) -> Dict[str, Any]:
    specs = [(str(s["feature"]), str(s["direction"]), float(s["tau"])) for s in policy.get("rule_specs", [])]
    mask = mask_for_rule(rows, specs)
    return eval_mask(rows, mask, target_col, safe_col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search CHAIR-free semantic pairwise fallback rules.")
    parser.add_argument("--net_harm_rows_csv", required=True)
    parser.add_argument("--feature_rows_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--apply_net_harm_rows_csv", default="")
    parser.add_argument("--apply_feature_rows_csv", default="")
    parser.add_argument("--target_col", default="net_harm_strict_train")
    parser.add_argument("--safe_col", default="net_safe_strict_v2")
    parser.add_argument("--feature_prefix", default="sem_")
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--quantile", type=float, action="append", default=[])
    parser.add_argument("--max_combo_size", type=int, default=2)
    parser.add_argument("--max_rule_specs", type=int, default=240)
    parser.add_argument("--require_benefit_cost_combo", action="store_true")
    parser.add_argument("--benefit_prefix", default="sem_benefit_")
    parser.add_argument("--cost_prefix", default="sem_cost_")
    parser.add_argument("--min_feature_valid_count", type=int, default=0)
    parser.add_argument("--min_feature_valid_frac", type=float, default=0.8)
    parser.add_argument("--min_selected", type=int, default=5)
    parser.add_argument("--max_selected", type=int, default=80)
    parser.add_argument("--min_delta_recall", type=float, default=0.0)
    parser.add_argument("--min_delta_f1", type=float, default=0.0)
    parser.add_argument("--max_delta_chair_i", type=float, default=0.005)
    parser.add_argument("--max_delta_chair_s", type=float, default=0.02)
    parser.add_argument("--min_target_precision", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rows = join_rows(os.path.abspath(args.net_harm_rows_csv), os.path.abspath(args.feature_rows_csv))
    features = discover_features(rows, str(args.feature_prefix), args.feature)
    quantiles = args.quantile or [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    constraints = {
        "min_selected": float(args.min_selected),
        "max_selected": float(args.max_selected),
        "min_delta_recall": float(args.min_delta_recall),
        "min_delta_f1": float(args.min_delta_f1),
        "max_delta_chair_i": float(args.max_delta_chair_i),
        "max_delta_chair_s": float(args.max_delta_chair_s),
        "min_target_precision": float(args.min_target_precision),
    }
    min_feature_valid = int(args.min_feature_valid_count)
    if min_feature_valid <= 0:
        min_feature_valid = max(10, int(float(args.min_feature_valid_frac) * len(rows)))
    metrics = feature_metrics(rows, features, str(args.target_col))
    candidates = search_rules(
        rows,
        features,
        str(args.target_col),
        str(args.safe_col),
        quantiles,
        int(args.max_combo_size),
        int(args.max_rule_specs),
        bool(args.require_benefit_cost_combo),
        str(args.benefit_prefix),
        str(args.cost_prefix),
        int(min_feature_valid),
        constraints,
        int(args.top_k),
    )

    metrics_csv = os.path.join(out_dir, "feature_metrics.csv")
    candidates_csv = os.path.join(out_dir, "rule_candidates.csv")
    write_csv(metrics_csv, metrics)
    candidate_fields = [
        "rule",
        "score",
        "n_selected",
        "target_precision",
        "target_recall",
        "n_target",
        "n_safe",
        "n_ignore_v2",
        "n_parser_artifact",
        "n_type_o",
        "n_type_r",
        "chair_i",
        "chair_s",
        "recall",
        "precision",
        "f1",
        "delta_chair_i_vs_int",
        "delta_chair_s_vs_int",
        "delta_recall_vs_int",
        "delta_precision_vs_int",
        "delta_f1_vs_int",
        "target_rate",
        "selected_ids",
        "policy_specs",
    ]
    write_csv(candidates_csv, candidates, candidate_fields)

    selected_policy = None
    apply_result = None
    apply_csv = ""
    if candidates:
        selected_specs = json.loads(candidates[0]["policy_specs"])
        selected_policy = {
            "policy_type": "semantic_pairwise_rule_v1",
            "rule": candidates[0]["rule"],
            "rule_specs": selected_specs,
            "target_col": str(args.target_col),
            "safe_col": str(args.safe_col),
            "feature_prefix": str(args.feature_prefix),
            "uses_chair_parser": False,
            "uses_coco_ontology": False,
            "uses_mscoco_generated_words": False,
        }
        write_json(os.path.join(out_dir, "selected_policy.json"), selected_policy)
        if args.apply_net_harm_rows_csv and args.apply_feature_rows_csv:
            apply_rows = join_rows(os.path.abspath(args.apply_net_harm_rows_csv), os.path.abspath(args.apply_feature_rows_csv))
            mask = mask_for_rule(
                apply_rows,
                [(str(s["feature"]), str(s["direction"]), float(s["tau"])) for s in selected_specs],
            )
            apply_result = apply_policy(apply_rows, selected_policy, str(args.target_col), str(args.safe_col))
            selected_rows = [
                {
                    "id": norm_id(row),
                    "route": "baseline" if use else "intervention",
                    "target": safe_int_flag(row.get(args.target_col)),
                    "safe": safe_int_flag(row.get(args.safe_col)),
                }
                for row, use in zip(apply_rows, mask)
            ]
            apply_csv = os.path.join(out_dir, "apply_decision_rows.csv")
            write_csv(apply_csv, selected_rows)

    summary = {
        "inputs": {
            "net_harm_rows_csv": os.path.abspath(args.net_harm_rows_csv),
            "feature_rows_csv": os.path.abspath(args.feature_rows_csv),
            "apply_net_harm_rows_csv": os.path.abspath(args.apply_net_harm_rows_csv) if args.apply_net_harm_rows_csv else "",
            "apply_feature_rows_csv": os.path.abspath(args.apply_feature_rows_csv) if args.apply_feature_rows_csv else "",
            "target_col": str(args.target_col),
            "safe_col": str(args.safe_col),
            "feature_prefix": str(args.feature_prefix),
            "constraints": constraints,
            "quantiles": quantiles,
            "max_combo_size": int(args.max_combo_size),
            "max_rule_specs": int(args.max_rule_specs),
            "require_benefit_cost_combo": bool(args.require_benefit_cost_combo),
            "benefit_prefix": str(args.benefit_prefix),
            "cost_prefix": str(args.cost_prefix),
            "min_feature_valid_count": int(min_feature_valid),
            "min_feature_valid_frac": float(args.min_feature_valid_frac),
        },
        "counts": {
            "n_rows": len(rows),
            "n_features": len(features),
            "n_candidates": len(candidates),
            "n_target": sum(safe_int_flag(row.get(args.target_col)) for row in rows),
            "n_safe": sum(safe_int_flag(row.get(args.safe_col)) for row in rows),
        },
        "best": candidates[0] if candidates else None,
        "apply_result": apply_result,
        "outputs": {
            "feature_metrics_csv": metrics_csv,
            "rule_candidates_csv": candidates_csv,
            "selected_policy_json": os.path.join(out_dir, "selected_policy.json") if selected_policy else "",
            "apply_decision_rows_csv": apply_csv,
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
