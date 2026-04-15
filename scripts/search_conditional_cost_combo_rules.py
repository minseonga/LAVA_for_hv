#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_TARGET_COL = "oracle_recall_gain_f1_nondecrease_ci_unique_noworse"
DEFAULT_GATE_FEATURE = "capobjyn_base_only_x_jaccard_gap"
DEFAULT_FEATURE_SPECS = (
    "capcost_base_object_density_per_content:low",
    "capcost_base_divergence_oververbose_score:low",
    "capcost_base_object_overgeneration:low",
    "capcost_delta_object_density_base_minus_int:low",
    "capcost_base_object_density_per_word:low",
    "capobjyn_base_minus_int_object_count:low",
    "capobjyn_base_to_int_count_ratio:low",
    "capobj_base_only_object_count:low",
    "capobj_base_object_count:low",
    "capobj_base_compound_object_rate:high",
    "capobj_base_only_compound_object_rate:high",
    "capobj_base_high_freq_object_rate:high",
    "capobj_base_only_high_freq_object_rate:high",
    "capcost_int_noise_score:low",
)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row.keys():
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
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("id") or row.get("image_id") or row.get("question_id") or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def flag(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def parse_feature_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for spec in specs:
        parts = str(spec).split(":")
        if len(parts) != 2:
            raise ValueError(f"Feature spec must be feature:high|low, got {spec}")
        feature, direction = parts[0].strip(), parts[1].strip().lower()
        if direction not in {"high", "low"}:
            raise ValueError(f"Direction must be high or low, got {direction}")
        out.append((feature, direction))
    return out


def select_by_threshold(
    rows: Sequence[Dict[str, Any]],
    feature: str,
    direction: str,
    threshold: float,
) -> List[int]:
    ids: List[int] = []
    for idx, row in enumerate(rows):
        value = safe_float(row.get(feature))
        if value is None:
            continue
        if direction == "high" and value >= threshold:
            ids.append(idx)
        elif direction == "low" and value <= threshold:
            ids.append(idx)
    return ids


def condition_threshold(values: Sequence[float], direction: str, selected_fraction: float) -> float:
    selected_fraction = max(0.0, min(1.0, float(selected_fraction)))
    if direction == "high":
        return percentile(values, 1.0 - selected_fraction)
    return percentile(values, selected_fraction)


def selected_stats(
    selected_ids: Iterable[int],
    rows: Sequence[Dict[str, Any]],
    target_col: str,
    gate_pos: int,
    global_pos: int,
) -> Dict[str, Any]:
    ids = list(selected_ids)
    n = len(ids)
    n_pos = sum(flag(rows[idx].get(target_col)) for idx in ids)
    return {
        "n_selected": n,
        "n_target": n_pos,
        "target_precision": n_pos / float(n) if n else 0.0,
        "target_recall_in_gate": n_pos / float(gate_pos) if gate_pos else 0.0,
        "target_recall_global": n_pos / float(global_pos) if global_pos else 0.0,
        "selected_ids": "|".join(safe_id(rows[idx]) for idx in ids[:200]),
    }


def build_conditions(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Tuple[str, str]],
    selected_fractions: Sequence[float],
    min_selected: int,
    max_selected: int,
    target_col: str,
    gate_pos: int,
    global_pos: int,
) -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    for feature, direction in feature_specs:
        values = [safe_float(row.get(feature)) for row in rows]
        vals = [float(v) for v in values if v is not None]
        if len(vals) < max(4, min_selected):
            continue
        for fraction in selected_fractions:
            threshold = condition_threshold(vals, direction, fraction)
            ids = select_by_threshold(rows, feature, direction, threshold)
            if len(ids) < min_selected or len(ids) > max_selected:
                continue
            row = {
                "rule_type": "single",
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "selected_fraction": fraction,
                "condition": f"{feature} {direction} {threshold:.8g}",
                "ids": set(ids),
            }
            row.update(selected_stats(ids, rows, target_col, gate_pos, global_pos))
            conditions.append(row)
    return conditions


def combo_rules(
    rows: Sequence[Dict[str, Any]],
    conditions: Sequence[Dict[str, Any]],
    combo_sizes: Sequence[int],
    min_selected: int,
    max_selected: int,
    target_col: str,
    gate_pos: int,
    global_pos: int,
) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for size in combo_sizes:
        if size <= 1:
            continue
        for combo in itertools.combinations(conditions, int(size)):
            features = [str(cond["feature"]) for cond in combo]
            if len(set(features)) < len(features):
                continue
            ids = set(combo[0]["ids"])
            for cond in combo[1:]:
                ids &= set(cond["ids"])
            if len(ids) < min_selected or len(ids) > max_selected:
                continue
            rule = {
                "rule_type": f"and_{size}",
                "combo_size": size,
                "condition": " AND ".join(str(cond["condition"]) for cond in combo),
                "features": "|".join(features),
                "directions": "|".join(str(cond["direction"]) for cond in combo),
                "thresholds": "|".join(str(cond["threshold"]) for cond in combo),
                "selected_fractions": "|".join(str(cond["selected_fraction"]) for cond in combo),
            }
            rule.update(selected_stats(sorted(ids), rows, target_col, gate_pos, global_pos))
            rules.append(rule)
    return rules


def zscore_rank_rules(
    rows: Sequence[Dict[str, Any]],
    feature_specs: Sequence[Tuple[str, str]],
    combo_sizes: Sequence[int],
    topk_values: Sequence[int],
    target_col: str,
    gate_pos: int,
    global_pos: int,
) -> List[Dict[str, Any]]:
    usable: List[Tuple[str, str, float, float]] = []
    for feature, direction in feature_specs:
        vals = [safe_float(row.get(feature)) for row in rows]
        values = [float(v) for v in vals if v is not None]
        if len(values) < 5:
            continue
        mu = sum(values) / float(len(values))
        sd = math.sqrt(sum((value - mu) ** 2 for value in values) / float(len(values))) or 1.0
        usable.append((feature, direction, mu, sd))

    out: List[Dict[str, Any]] = []
    for size in combo_sizes:
        for combo in itertools.combinations(usable, int(size)):
            scores: List[Tuple[float, int]] = []
            for idx, row in enumerate(rows):
                total = 0.0
                ok = True
                for feature, direction, mu, sd in combo:
                    value = safe_float(row.get(feature))
                    if value is None:
                        ok = False
                        break
                    z = (value - mu) / sd
                    total += -z if direction == "low" else z
                if ok:
                    scores.append((total, idx))
            scores.sort(reverse=True)
            for topk in topk_values:
                selected = [idx for _, idx in scores[: int(topk)]]
                if not selected:
                    continue
                rule = {
                    "rule_type": f"zsum_top{int(topk)}",
                    "combo_size": size,
                    "features": "|".join(feature for feature, _, _, _ in combo),
                    "directions": "|".join(direction for _, direction, _, _ in combo),
                    "condition": " + ".join(f"z({feature},{direction})" for feature, direction, _, _ in combo),
                    "topk": int(topk),
                }
                rule.update(selected_stats(selected, rows, target_col, gate_pos, global_pos))
                out.append(rule)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Search AND and rank-combo cost rules inside a benefit gate.")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--target_col", default=DEFAULT_TARGET_COL)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gate_feature", default=DEFAULT_GATE_FEATURE)
    ap.add_argument("--gate_direction", choices=["high", "low"], default="high")
    ap.add_argument("--gate_quantile", type=float, default=0.80)
    ap.add_argument("--gate_threshold", type=float, default=None)
    ap.add_argument("--feature_spec", action="append", default=[], help="feature:high|low. Defaults to v74 cost features.")
    ap.add_argument("--selected_fractions", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60")
    ap.add_argument("--combo_sizes", default="2,3")
    ap.add_argument("--topk", default="5,8,10,15,20")
    ap.add_argument("--min_selected", type=int, default=4)
    ap.add_argument("--max_selected", type=int, default=120)
    ap.add_argument("--max_output_rows", type=int, default=5000)
    args = ap.parse_args()

    rows_all = read_csv_rows(args.features_csv)
    target_col = str(args.target_col)
    feature_specs = parse_feature_specs(args.feature_spec or list(DEFAULT_FEATURE_SPECS))
    gate_values = [safe_float(row.get(args.gate_feature)) for row in rows_all]
    gate_vals = [float(v) for v in gate_values if v is not None]
    if args.gate_threshold is None:
        gate_threshold = percentile(gate_vals, float(args.gate_quantile))
    else:
        gate_threshold = float(args.gate_threshold)
    gate_ids = select_by_threshold(rows_all, args.gate_feature, args.gate_direction, gate_threshold)
    rows = [rows_all[idx] for idx in gate_ids]
    global_pos = sum(flag(row.get(target_col)) for row in rows_all)
    gate_pos = sum(flag(row.get(target_col)) for row in rows)
    selected_fractions = [float(x.strip()) for x in str(args.selected_fractions).split(",") if x.strip()]
    combo_sizes = [int(x.strip()) for x in str(args.combo_sizes).split(",") if x.strip()]
    topk_values = [int(x.strip()) for x in str(args.topk).split(",") if x.strip()]

    conditions = build_conditions(
        rows,
        feature_specs,
        selected_fractions,
        int(args.min_selected),
        int(args.max_selected),
        target_col,
        gate_pos,
        global_pos,
    )
    rules = list(conditions)
    rules.extend(
        combo_rules(
            rows,
            conditions,
            combo_sizes,
            int(args.min_selected),
            int(args.max_selected),
            target_col,
            gate_pos,
            global_pos,
        )
    )
    zrules = zscore_rank_rules(rows, feature_specs, combo_sizes, topk_values, target_col, gate_pos, global_pos)
    rules.sort(
        key=lambda row: (
            float(row.get("target_precision", 0.0)),
            float(row.get("n_target", 0.0)),
            -float(row.get("n_selected", 0.0)),
            float(row.get("target_recall_global", 0.0)),
        ),
        reverse=True,
    )
    zrules.sort(
        key=lambda row: (
            float(row.get("target_precision", 0.0)),
            float(row.get("n_target", 0.0)),
            -float(row.get("n_selected", 0.0)),
            float(row.get("target_recall_global", 0.0)),
        ),
        reverse=True,
    )

    n_rules_total = len(rules)
    n_zrules_total = len(zrules)
    max_output_rows = max(1, int(args.max_output_rows))
    rules_out = rules[:max_output_rows]
    zrules_out = zrules[:max_output_rows]

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rules_csv = os.path.join(out_dir, "combo_rule_candidates.csv")
    zrules_csv = os.path.join(out_dir, "combo_rank_candidates.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(rules_csv, rules_out)
    write_csv(zrules_csv, zrules_out)
    summary = {
        "inputs": {
            "features_csv": os.path.abspath(args.features_csv),
            "target_col": target_col,
            "gate_feature": args.gate_feature,
            "gate_direction": args.gate_direction,
            "gate_quantile": float(args.gate_quantile),
            "gate_threshold": gate_threshold,
            "feature_specs": [f"{feature}:{direction}" for feature, direction in feature_specs],
            "selected_fractions": selected_fractions,
            "combo_sizes": combo_sizes,
            "topk": topk_values,
            "min_selected": int(args.min_selected),
            "max_selected": int(args.max_selected),
        },
        "counts": {
            "n_rows": len(rows_all),
            "n_target": global_pos,
            "n_gate_rows": len(rows),
            "n_gate_target": gate_pos,
            "gate_precision": gate_pos / float(len(rows)) if rows else 0.0,
            "gate_recall_global": gate_pos / float(global_pos) if global_pos else 0.0,
            "n_conditions": len(conditions),
            "n_combo_rules_total": n_rules_total,
            "n_rank_rules_total": n_zrules_total,
            "n_combo_rules_written": len(rules_out),
            "n_rank_rules_written": len(zrules_out),
        },
        "top_combo_rules": rules_out[:40],
        "top_rank_rules": zrules_out[:40],
        "outputs": {
            "combo_rule_candidates_csv": rules_csv,
            "combo_rank_candidates_csv": zrules_csv,
            "summary_json": summary_json,
        },
    }
    write_json(summary_json, summary)
    print("[saved]", rules_csv)
    print("[saved]", zrules_csv)
    print("[saved]", summary_json)
    for row in rules[:15]:
        print(
            "[combo]",
            row.get("rule_type"),
            "n=",
            row.get("n_selected"),
            "target=",
            row.get("n_target"),
            "prec=",
            row.get("target_precision"),
            "global_rec=",
            row.get("target_recall_global"),
            "::",
            row.get("condition"),
        )
    for row in zrules[:10]:
        print(
            "[rank]",
            row.get("rule_type"),
            "n=",
            row.get("n_selected"),
            "target=",
            row.get("n_target"),
            "prec=",
            row.get("target_precision"),
            "::",
            row.get("condition"),
        )


if __name__ == "__main__":
    main()
