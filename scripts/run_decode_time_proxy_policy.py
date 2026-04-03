#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_PROXY_FEATURES = [
    "proxy_gap_content_mean",
    "proxy_gap_content_std",
    "proxy_gap_content_lastk_mean",
    "proxy_gap_content_lastk_std",
    "proxy_faithful_content_std",
    "proxy_lp_content_min",
    "proxy_margin_content_min",
    "proxy_entropy_content_mean",
    "proxy_low_gap_ratio_content",
    "proxy_low_margin_ratio_content",
    "proxy_gap_sign_flip_count_content",
]


def canonical_label_name(label_key: str) -> str:
    return str(label_key or "").strip().lower()


def maybe_float(value: object) -> Optional[float]:
    s = str(value or "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: object) -> Optional[int]:
    f = maybe_float(value)
    if f is None:
        return None
    return int(round(f))


def mean(values: Iterable[float]) -> Optional[float]:
    seq = [float(v) for v in values]
    if not seq:
        return None
    return float(sum(seq) / float(len(seq)))


def std(values: Iterable[float]) -> Optional[float]:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0 if seq else None
    mu = mean(seq)
    if mu is None:
        return None
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(math.sqrt(max(0.0, var)))


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def assign_avg_ranks(scores: Sequence[float]) -> List[float]:
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and float(scores[order[j]]) == float(scores[order[i]]):
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pos = sum(1 for y in labels if int(y) == 1)
    neg = sum(1 for y in labels if int(y) == 0)
    if pos == 0 or neg == 0:
        return None
    ranks = assign_avg_ranks(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if int(y) == 1)
    u = rank_sum_pos - float(pos * (pos + 1)) / 2.0
    return float(u / float(pos * neg))


def threshold_grid(values: Sequence[float]) -> List[float]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return [0.0]
    if len(finite) == 1:
        base = float(finite[0])
        eps = max(1e-9, 1e-6 * max(1.0, abs(base)))
        return [base - eps, base, base + eps]
    lo = float(finite[0])
    hi = float(finite[-1])
    scale = max(1.0, abs(lo), abs(hi), abs(hi - lo))
    eps = max(1e-9, 1e-6 * scale)
    out = {lo - eps, lo, hi, hi + eps}
    for q in [i / 100.0 for i in range(1, 100)]:
        pos = q * float(len(finite) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.add(finite[lo])
        else:
            w = pos - float(lo)
            out.add((1.0 - w) * finite[lo] + w * finite[hi])
    return sorted(out)


def select_feature_names(rows: Sequence[Dict[str, Any]], feature_cols: str) -> List[str]:
    names = [part.strip() for part in str(feature_cols or "").split(",") if part.strip()]
    if not names:
        names = list(DEFAULT_PROXY_FEATURES)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return [name for name in names if name in keys]


def merge_rows(
    feature_rows: Sequence[Dict[str, str]],
    reference_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, Any]]:
    ref_map = {str(row.get("id", "")).strip(): row for row in reference_rows}
    merged: List[Dict[str, Any]] = []
    for row in feature_rows:
        sid = str(row.get("id", "")).strip()
        ref = ref_map.get(sid)
        if ref is None:
            continue
        out: Dict[str, Any] = dict(row)
        out["id"] = sid
        out["reference_rescue"] = maybe_int(ref.get("rescue"))
        out["baseline_correct"] = maybe_int(ref.get("baseline_correct"))
        out["intervention_correct"] = maybe_int(ref.get("intervention_correct"))
        out["reference_final_correct"] = maybe_int(ref.get("final_correct"))
        out["reference_case_type"] = str(ref.get("case_type", "")).strip()
        bc = out.get("baseline_correct")
        ic = out.get("intervention_correct")
        out["actual_rescue"] = None if bc is None or ic is None else int(int(bc) == 1 and int(ic) == 0)
        merged.append(out)
    return merged


def orient_for_rescue(rows: Sequence[Dict[str, Any]], feature: str, label_key: str) -> Tuple[str, Optional[float]]:
    feat_rows = [row for row in rows if maybe_float(row.get(feature)) is not None and row.get(label_key) is not None]
    scores = [float(row[feature]) for row in feat_rows]
    labels = [int(row[label_key]) for row in feat_rows]
    high_auc = binary_auc(scores, labels)
    low_auc = binary_auc([-float(v) for v in scores], labels)
    if (low_auc or -1.0) > (high_auc or -1.0):
        return "low", low_auc
    return "high", high_auc


def eval_against_label(
    rows: Sequence[Dict[str, Any]],
    rescue_flags: Sequence[int],
    *,
    label_key: str,
    prefix: str,
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    n = 0
    for row, rescue in zip(rows, rescue_flags):
        label = row.get(label_key)
        if label is None:
            continue
        n += 1
        if int(rescue) == 1 and int(label) == 1:
            tp += 1
        elif int(rescue) == 1 and int(label) == 0:
            fp += 1
        elif int(rescue) == 0 and int(label) == 0:
            tn += 1
        elif int(rescue) == 0 and int(label) == 1:
            fn += 1
    precision = None if (tp + fp) == 0 else float(tp / float(tp + fp))
    recall = None if (tp + fn) == 0 else float(tp / float(tp + fn))
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {
        f"{prefix}_n": int(n),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_tn": int(tn),
        f"{prefix}_fn": int(fn),
    }


def eval_decision(
    rows: Sequence[Dict[str, Any]],
    rescue_flags: Sequence[int],
    *,
    target_label: str,
) -> Dict[str, Any]:
    n = 0
    base_sum = 0
    int_sum = 0
    final_sum = 0
    for row, rescue in zip(rows, rescue_flags):
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        if bc is None or ic is None:
            continue
        n += 1
        base_sum += int(bc)
        int_sum += int(ic)
        final_sum += int(bc) if int(rescue) == 1 else int(ic)

    out = {
        "n_eval": int(n),
        "target_label": canonical_label_name(target_label),
        "rescue_rate": (None if n == 0 else float(sum(int(x) for x in rescue_flags) / float(n))),
        "baseline_acc": (None if n == 0 else float(base_sum / float(n))),
        "intervention_acc": (None if n == 0 else float(int_sum / float(n))),
        "final_acc": (None if n == 0 else float(final_sum / float(n))),
        "delta_vs_intervention": (None if n == 0 else float((final_sum - int_sum) / float(n))),
    }
    out.update(eval_against_label(rows, rescue_flags, label_key="reference_rescue", prefix="reference"))
    out.update(eval_against_label(rows, rescue_flags, label_key="actual_rescue", prefix="actual"))
    primary_prefix = "actual" if canonical_label_name(target_label) == "actual_rescue" else "reference"
    out["target_precision"] = out.get(f"{primary_prefix}_precision")
    out["target_recall"] = out.get(f"{primary_prefix}_recall")
    out["target_f1"] = out.get(f"{primary_prefix}_f1")
    out["target_tp"] = out.get(f"{primary_prefix}_tp")
    out["target_fp"] = out.get(f"{primary_prefix}_fp")
    out["target_tn"] = out.get(f"{primary_prefix}_tn")
    out["target_fn"] = out.get(f"{primary_prefix}_fn")
    return out


def build_single_oriented_scores(rows: Sequence[Dict[str, Any]], feature: str, direction: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for row in rows:
        raw = maybe_float(row.get(feature))
        if raw is None:
            out.append(None)
            continue
        out.append(-float(raw) if direction == "low" else float(raw))
    return out


def build_pair_scores(
    rows: Sequence[Dict[str, Any]],
    feat_a: str,
    dir_a: str,
    feat_b: str,
    dir_b: str,
    mu_a: float,
    sd_a: float,
    mu_b: float,
    sd_b: float,
) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    sd_a = float(max(sd_a, 1e-6))
    sd_b = float(max(sd_b, 1e-6))
    for row in rows:
        va = maybe_float(row.get(feat_a))
        vb = maybe_float(row.get(feat_b))
        if va is None or vb is None:
            out.append(None)
            continue
        oa = -float(va) if dir_a == "low" else float(va)
        ob = -float(vb) if dir_b == "low" else float(vb)
        za = (oa - float(mu_a)) / sd_a
        zb = (ob - float(mu_b)) / sd_b
        out.append(float(za + zb))
    return out


def calibrate(args: argparse.Namespace) -> None:
    feature_rows = load_csv_rows(args.features_csv)
    reference_rows = load_csv_rows(args.reference_decisions_csv)
    rows = merge_rows(feature_rows, reference_rows)
    target_label = canonical_label_name(args.target_label)
    feature_names = select_feature_names(rows, args.feature_cols)
    if not rows:
        raise RuntimeError("No merged rows for calibration.")
    if not feature_names:
        available = sorted({k for row in rows for k in row.keys()})
        proxy_like = [k for k in available if str(k).startswith("proxy_")]
        raise RuntimeError(
            "No requested proxy features were found after merging. "
            f"Check --feature_cols names against decode_time_proxy_features.csv. "
            f"Available proxy-like columns: {proxy_like[:40]}"
        )

    candidates: List[Dict[str, Any]] = []
    single_rank: List[Tuple[str, float]] = []

    for feature in feature_names:
        direction, auc = orient_for_rescue(rows, feature, target_label)
        single_rank.append((feature, -1.0 if auc is None else float(auc)))
        oriented = build_single_oriented_scores(rows, feature, direction)
        valid = [float(v) for v in oriented if v is not None]
        for tau in threshold_grid(valid):
            flags = [0 if v is None else int(float(v) >= float(tau)) for v in oriented]
            metrics = eval_decision(rows, flags, target_label=target_label)
            if metrics["rescue_rate"] is not None and float(metrics["rescue_rate"]) > float(args.max_rescue_rate):
                continue
            candidates.append(
                {
                    "policy_type": "single",
                    "target_label": target_label,
                    "feature_a": feature,
                    "direction_a": direction,
                    "feature_b": "",
                    "direction_b": "",
                    "mu_a": "",
                    "sd_a": "",
                    "mu_b": "",
                    "sd_b": "",
                    "tau": float(tau),
                    "orient_auc": auc,
                    **metrics,
                }
            )

    single_rank = sorted(single_rank, key=lambda x: float(x[1]), reverse=True)
    top_pair_feats = [name for name, _ in single_rank[: max(2, int(args.pair_feature_topn))]]
    for feat_a, feat_b in combinations(top_pair_feats, 2):
        dir_a, auc_a = orient_for_rescue(rows, feat_a, target_label)
        dir_b, auc_b = orient_for_rescue(rows, feat_b, target_label)
        oa = [v for v in build_single_oriented_scores(rows, feat_a, dir_a) if v is not None]
        ob = [v for v in build_single_oriented_scores(rows, feat_b, dir_b) if v is not None]
        mu_a = mean(oa)
        sd_a = std(oa)
        mu_b = mean(ob)
        sd_b = std(ob)
        if mu_a is None or sd_a is None or mu_b is None or sd_b is None:
            continue
        pair_scores = build_pair_scores(rows, feat_a, dir_a, feat_b, dir_b, mu_a, sd_a, mu_b, sd_b)
        valid = [float(v) for v in pair_scores if v is not None]
        for tau in threshold_grid(valid):
            flags = [0 if v is None else int(float(v) >= float(tau)) for v in pair_scores]
            metrics = eval_decision(rows, flags, target_label=target_label)
            if metrics["rescue_rate"] is not None and float(metrics["rescue_rate"]) > float(args.max_rescue_rate):
                continue
            candidates.append(
                {
                    "policy_type": "pair_sum_z",
                    "target_label": target_label,
                    "feature_a": feat_a,
                    "direction_a": dir_a,
                    "feature_b": feat_b,
                    "direction_b": dir_b,
                    "mu_a": float(mu_a),
                    "sd_a": float(max(sd_a, 1e-6)),
                    "mu_b": float(mu_b),
                    "sd_b": float(max(sd_b, 1e-6)),
                    "tau": float(tau),
                    "orient_auc": None if auc_a is None or auc_b is None else float((auc_a + auc_b) / 2.0),
                    **metrics,
                }
            )

    if not candidates:
        raise RuntimeError("No feasible proxy-policy candidate found.")

    def score_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (
            -1.0 if row["target_f1"] is None else float(row["target_f1"]),
            -1.0 if row["final_acc"] is None else float(row["final_acc"]),
            -1.0 if row["target_precision"] is None else float(row["target_precision"]),
            -float(row["rescue_rate"] or 0.0),
        )

    best = max(candidates, key=score_key)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    candidates_csv = os.path.join(out_dir, "calibration_candidates.csv")
    selected_json = os.path.join(out_dir, "selected_policy.json")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(candidates_csv, candidates)
    selected_policy = {
        "target_label": target_label,
        "policy_type": str(best["policy_type"]),
        "feature_a": str(best["feature_a"]),
        "direction_a": str(best["direction_a"]),
        "feature_b": str(best.get("feature_b", "")),
        "direction_b": str(best.get("direction_b", "")),
        "mu_a": (None if best.get("mu_a", "") == "" else float(best["mu_a"])),
        "sd_a": (None if best.get("sd_a", "") == "" else float(best["sd_a"])),
        "mu_b": (None if best.get("mu_b", "") == "" else float(best["mu_b"])),
        "sd_b": (None if best.get("sd_b", "") == "" else float(best["sd_b"])),
        "tau": float(best["tau"]),
    }
    write_json(selected_json, selected_policy)
    write_json(
        summary_json,
        {
            "mode": "calibrate",
            "inputs": {
                "features_csv": os.path.abspath(args.features_csv),
                "reference_decisions_csv": os.path.abspath(args.reference_decisions_csv),
                "feature_cols": feature_names,
                "target_label": target_label,
                "pair_feature_topn": int(args.pair_feature_topn),
                "max_rescue_rate": float(args.max_rescue_rate),
            },
            "selected_policy": {
                **selected_policy,
                "calibration_metrics": {
                    key: best[key]
                    for key in [
                        "final_acc",
                        "delta_vs_intervention",
                        "rescue_rate",
                        "target_precision",
                        "target_recall",
                        "target_f1",
                        "actual_precision",
                        "actual_recall",
                        "actual_f1",
                        "reference_precision",
                        "reference_recall",
                        "reference_f1",
                    ]
                },
            },
            "outputs": {
                "candidates_csv": os.path.abspath(candidates_csv),
                "selected_policy_json": os.path.abspath(selected_json),
            },
        },
    )
    print("[saved]", candidates_csv)
    print("[saved]", selected_json)
    print("[saved]", summary_json)


def apply(args: argparse.Namespace) -> None:
    feature_rows = load_csv_rows(args.features_csv)
    reference_rows = load_csv_rows(args.reference_decisions_csv)
    rows = merge_rows(feature_rows, reference_rows)
    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)
    target_label = canonical_label_name(policy.get("target_label", args.target_label))

    flags: List[int] = []
    decision_rows: List[Dict[str, Any]] = []
    for row in rows:
        value_a = maybe_float(row.get(policy["feature_a"]))
        score = None
        if policy["policy_type"] == "single":
            if value_a is not None:
                score = -float(value_a) if policy["direction_a"] == "low" else float(value_a)
        elif policy["policy_type"] == "pair_sum_z":
            value_b = maybe_float(row.get(policy["feature_b"]))
            if value_a is not None and value_b is not None:
                oa = -float(value_a) if policy["direction_a"] == "low" else float(value_a)
                ob = -float(value_b) if policy["direction_b"] == "low" else float(value_b)
                score = float((oa - float(policy["mu_a"])) / float(max(policy["sd_a"], 1e-6)))
                score += float((ob - float(policy["mu_b"])) / float(max(policy["sd_b"], 1e-6)))
        rescue = int(score is not None and float(score) >= float(policy["tau"]))
        flags.append(rescue)
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        final_correct = None if bc is None or ic is None else (int(bc) if rescue == 1 else int(ic))
        decision_rows.append(
            {
                "id": row.get("id"),
                "reference_case_type": row.get("reference_case_type"),
                "reference_rescue": row.get("reference_rescue"),
                "actual_rescue": row.get("actual_rescue"),
                "proxy_score": score,
                "proxy_rescue": rescue,
                "feature_a": policy["feature_a"],
                "feature_a_raw": value_a,
                "feature_b": policy.get("feature_b", ""),
                "feature_b_raw": (None if not policy.get("feature_b") else maybe_float(row.get(policy["feature_b"]))),
                "baseline_correct": bc,
                "intervention_correct": ic,
                "final_correct": final_correct,
            }
        )

    metrics = eval_decision(rows, flags, target_label=target_label)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    decisions_csv = os.path.join(out_dir, "decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    write_csv(decisions_csv, decision_rows)
    write_json(
        summary_json,
        {
            "mode": "apply",
            "inputs": {
                "features_csv": os.path.abspath(args.features_csv),
                "reference_decisions_csv": os.path.abspath(args.reference_decisions_csv),
                "policy_json": os.path.abspath(args.policy_json),
            },
            "policy": policy,
            "evaluation": metrics,
            "outputs": {
                "decision_rows_csv": os.path.abspath(decisions_csv),
            },
        },
    )
    print("[saved]", decisions_csv)
    print("[saved]", summary_json)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate/apply a 1-pass decode-time proxy rescue policy.")
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_cal = sub.add_parser("calibrate")
    ap_cal.add_argument("--features_csv", type=str, required=True)
    ap_cal.add_argument("--reference_decisions_csv", type=str, required=True)
    ap_cal.add_argument("--feature_cols", type=str, default="")
    ap_cal.add_argument("--target_label", type=str, default="reference_rescue")
    ap_cal.add_argument("--pair_feature_topn", type=int, default=6)
    ap_cal.add_argument("--max_rescue_rate", type=float, default=0.03)
    ap_cal.add_argument("--out_dir", type=str, required=True)

    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--features_csv", type=str, required=True)
    ap_apply.add_argument("--reference_decisions_csv", type=str, required=True)
    ap_apply.add_argument("--policy_json", type=str, required=True)
    ap_apply.add_argument("--target_label", type=str, default="reference_rescue")
    ap_apply.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()
    if args.mode == "calibrate":
        calibrate(args)
    elif args.mode == "apply":
        apply(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
