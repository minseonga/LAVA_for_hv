#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from evaluate_chair_pairwise_rollback import load_sentence_map, per_sample_metrics, safe_div


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
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def object_set(values: Iterable[Any]) -> Set[str]:
    out = set()
    for value in values:
        if isinstance(value, (list, tuple)) and value:
            out.add(str(value[-1]).strip())
        else:
            out.add(str(value).strip())
    return {x for x in out if x}


def hallucination_set(values: Iterable[Any]) -> Set[str]:
    out = set()
    for value in values:
        if isinstance(value, (list, tuple)) and value:
            out.add(str(value[0]).strip())
        else:
            out.add(str(value).strip())
    return {x for x in out if x}


def image_id_from_row(row: Dict[str, Any]) -> str:
    raw = str(row.get("image_id", row.get("id", ""))).strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def safe_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    return val if math.isfinite(val) else None


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
    ap = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label:
            hits += 1
            ap += hits / float(rank)
    return ap / float(n_pos)


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / float(len(values)) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def load_audit_features(path: str) -> Tuple[Set[str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    if not path:
        return set(), {}, {}
    rows = read_csv_rows(path)
    parser_by_sample: Dict[str, int] = {}
    total_by_sample: Dict[str, int] = {}
    labels_by_sample: Dict[str, List[str]] = {}
    features_by_sample: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "n_nonartifact_intervention_only_hall": 0,
            "n_parser_artifact_suspect": 0,
            "has_trace_unstable_semantic_suspect": 0,
            "has_confident_scene_prior_suspect": 0,
            "has_unresolved_manual_needed": 0,
            "nonartifact_intervention_only_hall_objects": set(),
            "parser_artifact_objects": set(),
        }
    )
    for row in rows:
        iid = str(row.get("image_id", "")).strip()
        if not iid:
            continue
        label = str(row.get("proxy_label", ""))
        obj = str(row.get("object", "")).strip()
        total_by_sample[iid] = total_by_sample.get(iid, 0) + 1
        if str(row.get("v50_parser_artifact_exclude", "")).strip() == "1" or label == "parser_artifact_suspect":
            parser_by_sample[iid] = parser_by_sample.get(iid, 0) + 1
            features_by_sample[iid]["n_parser_artifact_suspect"] += 1
            if obj:
                features_by_sample[iid]["parser_artifact_objects"].add(obj)
        else:
            features_by_sample[iid]["n_nonartifact_intervention_only_hall"] += 1
            if obj:
                features_by_sample[iid]["nonartifact_intervention_only_hall_objects"].add(obj)
        if label == "trace_unstable_semantic_suspect":
            features_by_sample[iid]["has_trace_unstable_semantic_suspect"] = 1
        if label == "confident_scene_prior_suspect":
            features_by_sample[iid]["has_confident_scene_prior_suspect"] = 1
        if label in {"middle_needs_manual", "unmatched_needs_manual"}:
            features_by_sample[iid]["has_unresolved_manual_needed"] = 1
        labels_by_sample.setdefault(iid, []).append(str(row.get("proxy_label", "")))
    excluded = {iid for iid, total in total_by_sample.items() if total > 0 and parser_by_sample.get(iid, 0) == total}
    labels = {iid: "|".join(vals) for iid, vals in labels_by_sample.items()}
    serializable: Dict[str, Dict[str, Any]] = {}
    for iid, vals in features_by_sample.items():
        item = dict(vals)
        item["nonartifact_intervention_only_hall_objects"] = "|".join(sorted(item["nonartifact_intervention_only_hall_objects"]))
        item["parser_artifact_objects"] = "|".join(sorted(item["parser_artifact_objects"]))
        serializable[iid] = item
    return excluded, labels, serializable


def build_metric_rows(
    baseline_chair_json: str,
    intervention_chair_json: str,
    *,
    recall_eps: float,
    f1_eps: float,
    ci_gain_min: float,
    cs_gain_min: float,
    soft_recall_eps: float,
    soft_f1_eps: float,
    soft_ci_gain_min: float,
    parser_artifact_object_rows_csv: str,
    tau_clean_ci: float,
    tau_clean_h: float,
    type_o_recall_gain: float,
    type_o_f1_gain: float,
    type_o_supported_gain: float,
    type_o_ci_eps: float,
    type_r_f1_gain: float,
    type_r_recall_gain: float,
    type_r_ci_delta: float,
    safe_recall_gain: float,
    safe_f1_gain: float,
    safe_ci_eps: float,
) -> List[Dict[str, Any]]:
    base_map = load_sentence_map(os.path.abspath(baseline_chair_json))
    int_map = load_sentence_map(os.path.abspath(intervention_chair_json))
    artifact_only_samples, artifact_labels, audit_features = load_audit_features(parser_artifact_object_rows_csv)
    rows: List[Dict[str, Any]] = []
    for sid in sorted(set(base_map) & set(int_map), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        b_row = base_map[sid]
        i_row = int_map[sid]
        b = per_sample_metrics(b_row)
        i = per_sample_metrics(i_row)
        gt = object_set(i_row.get("mscoco_gt_words", []))
        b_gen = object_set(b_row.get("mscoco_generated_words", []))
        i_gen = object_set(i_row.get("mscoco_generated_words", []))
        b_supported = b_gen & gt
        i_supported = i_gen & gt
        dropped_supported = b_supported - i_supported
        gained_supported = i_supported - b_supported
        delta_recall = float(i["recall"] - b["recall"])
        delta_f1 = float(i["f1"] - b["f1"])
        delta_ci = float(i["chair_i"] - b["chair_i"])
        delta_cs = float(i["chair_s"] - b["chair_s"])
        ci_gain = float(b["chair_i"] - i["chair_i"])
        cs_gain = float(b["chair_s"] - i["chair_s"])
        delta_r_base_minus_int = -delta_recall
        delta_f_base_minus_int = -delta_f1
        delta_s_base_minus_int = float(len(b_supported) - len(i_supported))
        b_hall_set = hallucination_set(b_row.get("mscoco_hallucinated_words", []))
        i_hall_set = hallucination_set(i_row.get("mscoco_hallucinated_words", []))
        intervention_only_hall_set = i_hall_set - b_hall_set
        baseline_only_hall_set = b_hall_set - i_hall_set
        both_hall_set = b_hall_set & i_hall_set
        f1_nondecrease_no_more_hall_set = bool(
            delta_f_base_minus_int >= 0.0
            and len(b_hall_set) <= len(i_hall_set)
        )
        recall_gain_f1_nondecrease_no_more_hall_set = bool(
            delta_r_base_minus_int > 0.0
            and delta_f_base_minus_int >= 0.0
            and len(b_hall_set) <= len(i_hall_set)
        )
        any_intervention_only_hall = bool(intervention_only_hall_set)
        pure_regression_base_clean_to_int_hall = bool(not b_hall_set and i_hall_set)
        harm_score = (
            max(0.0, -delta_recall)
            + max(0.0, -delta_f1)
            - max(0.0, ci_gain)
            - 0.2 * max(0.0, cs_gain)
        )
        strict = (
            delta_recall <= -float(recall_eps)
            and delta_f1 <= float(f1_eps)
            and ci_gain <= float(ci_gain_min)
            and cs_gain <= float(cs_gain_min)
        )
        soft = (
            delta_recall <= -float(soft_recall_eps)
            and delta_f1 <= float(soft_f1_eps)
            and ci_gain <= float(soft_ci_gain_min)
        )
        parser_artifact_only = sid in artifact_only_samples
        audit = audit_features.get(sid, {})
        n_nonartifact_hall = int(audit.get("n_nonartifact_intervention_only_hall", 0) or 0)
        n_parser_artifact = int(audit.get("n_parser_artifact_suspect", 0) or 0)
        has_trace_unstable = int(audit.get("has_trace_unstable_semantic_suspect", 0) or 0)
        has_confident_scene_prior = int(audit.get("has_confident_scene_prior_suspect", 0) or 0)
        has_unresolved = int(audit.get("has_unresolved_manual_needed", 0) or 0)
        base_nonartifact_h = int(b["n_hallucinated_instances"])
        baseline_clean = int(float(b["chair_i"]) <= float(tau_clean_ci) and base_nonartifact_h <= float(tau_clean_h))
        type_o = (
            (not parser_artifact_only)
            and baseline_clean
            and delta_r_base_minus_int >= float(type_o_recall_gain)
            and delta_f_base_minus_int >= float(type_o_f1_gain)
            and delta_s_base_minus_int >= float(type_o_supported_gain)
            and delta_ci >= -float(type_o_ci_eps)
        )
        type_r = (
            (not parser_artifact_only)
            and baseline_clean
            and n_nonartifact_hall >= 1
            and bool(has_trace_unstable or has_confident_scene_prior)
            and (
                delta_f_base_minus_int >= float(type_r_f1_gain)
                or delta_r_base_minus_int >= float(type_r_recall_gain)
                or delta_ci >= float(type_r_ci_delta)
            )
        )
        strict_harm = bool(type_o or type_r)
        safe = (
            (not parser_artifact_only)
            and (
                delta_recall >= float(safe_recall_gain)
                or delta_f1 >= float(safe_f1_gain)
            )
            and float(i["chair_i"]) <= float(b["chair_i"]) + float(safe_ci_eps)
            and n_nonartifact_hall == 0
        )
        ignore = not strict_harm and not safe
        nh_score = (
            1.0 * max(0.0, delta_r_base_minus_int - float(type_o_recall_gain))
            + 1.2 * max(0.0, delta_f_base_minus_int - float(type_o_f1_gain))
            + 0.8 * max(0.0, delta_s_base_minus_int - float(type_o_supported_gain))
            + 0.8 * max(0.0, delta_ci - float(type_r_ci_delta))
            + 1.0 * float(n_nonartifact_hall)
            + 0.7 * float(has_trace_unstable)
            + 0.5 * float(has_confident_scene_prior)
            - 1.2 * max(0.0, float(b["chair_i"]) - float(tau_clean_ci))
            - 1.0 * max(0.0, float(base_nonartifact_h) - float(tau_clean_h))
            - 3.0 * float(parser_artifact_only)
        )
        rows.append(
            {
                "id": sid,
                "image_id": sid,
                "baseline_chair_i": b["chair_i"],
                "intervention_chair_i": i["chair_i"],
                "baseline_chair_s": b["chair_s"],
                "intervention_chair_s": i["chair_s"],
                "baseline_recall": b["recall"],
                "intervention_recall": i["recall"],
                "baseline_precision": b["precision"],
                "intervention_precision": i["precision"],
                "baseline_f1": b["f1"],
                "intervention_f1": i["f1"],
                "delta_recall_int_minus_base": delta_recall,
                "delta_f1_int_minus_base": delta_f1,
                "delta_chair_i_int_minus_base": delta_ci,
                "delta_chair_s_int_minus_base": delta_cs,
                "delta_recall_base_minus_int": delta_r_base_minus_int,
                "delta_f1_base_minus_int": delta_f_base_minus_int,
                "delta_supported_base_minus_int": delta_s_base_minus_int,
                "chair_i_gain_base_minus_int": ci_gain,
                "chair_s_gain_base_minus_int": cs_gain,
                "n_dropped_supported": len(dropped_supported),
                "n_gained_supported": len(gained_supported),
                "dropped_supported": "|".join(sorted(dropped_supported)),
                "gained_supported": "|".join(sorted(gained_supported)),
                "baseline_nonartifact_hall_count": base_nonartifact_h,
                "baseline_hall_set_count": len(b_hall_set),
                "intervention_hall_set_count": len(i_hall_set),
                "intervention_only_hall_set_count": len(intervention_only_hall_set),
                "baseline_only_hall_set_count": len(baseline_only_hall_set),
                "both_hall_set_count": len(both_hall_set),
                "intervention_only_hall_set": "|".join(sorted(intervention_only_hall_set)),
                "baseline_only_hall_set": "|".join(sorted(baseline_only_hall_set)),
                "both_hall_set": "|".join(sorted(both_hall_set)),
                "baseline_clean": baseline_clean,
                "n_nonartifact_intervention_only_hall": n_nonartifact_hall,
                "n_parser_artifact_suspect": n_parser_artifact,
                "has_trace_unstable_semantic_suspect": has_trace_unstable,
                "has_confident_scene_prior_suspect": has_confident_scene_prior,
                "has_unresolved_manual_needed": has_unresolved,
                "nonartifact_intervention_only_hall_objects": audit.get("nonartifact_intervention_only_hall_objects", ""),
                "parser_artifact_objects": audit.get("parser_artifact_objects", ""),
                "sample_harm_score": harm_score,
                "net_harm_score_v2": nh_score,
                "parser_artifact_only_sample": int(parser_artifact_only),
                "artifact_proxy_labels": artifact_labels.get(sid, ""),
                "net_harm_type_o": int(type_o),
                "net_harm_type_r": int(type_r),
                "net_harm_strict_v2": int(strict_harm),
                "net_safe_strict_v2": int(safe),
                "net_ignore_v2": int(ignore),
                "net_harm_strict_raw": int(strict),
                "net_harm_soft_raw": int(soft),
                "net_harm_strict_train": int(strict and not parser_artifact_only),
                "net_harm_soft_train": int(soft and not parser_artifact_only),
                "oracle_f1_nondecrease_no_more_hall_set": int(f1_nondecrease_no_more_hall_set),
                "oracle_recall_gain_f1_nondecrease_no_more_hall_set": int(recall_gain_f1_nondecrease_no_more_hall_set),
                "fallback_any_intervention_only_hall": int(any_intervention_only_hall),
                "fallback_pure_regression_base_clean_to_int_hall": int(pure_regression_base_clean_to_int_hall),
            }
        )
    return rows


def evaluate_feature_auc(
    target_rows: Sequence[Dict[str, Any]],
    feature_rows_csv: str,
    *,
    target_col: str,
    feature_prefix: str,
    top_ks: Sequence[int],
) -> List[Dict[str, Any]]:
    if not feature_rows_csv:
        return []
    feature_rows = read_csv_rows(os.path.abspath(feature_rows_csv))
    target_by_id = {str(row["id"]): int(row[target_col]) for row in target_rows}
    feature_rows = [row for row in feature_rows if str(row.get("id", row.get("image_id", ""))).strip() in target_by_id]
    if not feature_rows:
        return []
    exclude = {"id", "image", "question", "probe_decoded_text", "probe_mention_texts", "probe_selector", "probe_selector_fallback", "probe_task_mode", "score_error"}
    metrics: List[Dict[str, Any]] = []
    for feature in feature_rows[0].keys():
        if feature in exclude:
            continue
        if feature_prefix and not feature.startswith(feature_prefix):
            continue
        pairs: List[Tuple[int, float, str]] = []
        for row in feature_rows:
            sid = str(row.get("id", row.get("image_id", ""))).strip()
            val = safe_float(row.get(feature))
            if val is None or sid not in target_by_id:
                continue
            pairs.append((target_by_id[sid], val, sid))
        if not pairs:
            continue
        n_pos = sum(label for label, _, _ in pairs)
        n_neg = len(pairs) - n_pos
        if n_pos <= 0 or n_neg <= 0:
            continue
        pos = [val for label, val, _ in pairs if label]
        neg = [val for label, val, _ in pairs if not label]
        auc = auc_high(pos, neg)
        if auc is None:
            continue
        direction = "high" if auc >= 0.5 else "low"
        oriented = [(label, val if direction == "high" else -val, sid) for label, val, sid in pairs]
        oriented.sort(key=lambda x: x[1], reverse=True)
        rec: Dict[str, Any] = {
            "target_col": target_col,
            "feature": feature,
            "direction": direction,
            "auroc": max(auc, 1.0 - auc),
            "auroc_high": auc,
            "ap": ap_score([(label, score) for label, score, _ in oriented]),
            "n": len(pairs),
            "n_pos": n_pos,
            "pos_mean": mean(pos),
            "neg_mean": mean(neg),
            "pos_median": median(pos),
            "neg_median": median(neg),
        }
        for k in top_ks:
            kk = min(int(k), len(oriented))
            if kk <= 0:
                continue
            top = oriented[:kk]
            rec[f"precision_at_{k}"] = sum(label for label, _, _ in top) / float(kk)
            rec[f"hits_at_{k}"] = sum(label for label, _, _ in top)
            rec[f"top_pos_ids_at_{k}"] = "|".join(sid for label, _, sid in top if label)
        metrics.append(rec)
    metrics.sort(key=lambda row: (float(row.get("auroc") or 0.0), float(row.get("ap") or 0.0)), reverse=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Build sample-level generative net-harm targets and optional feature AUC tables.")
    ap.add_argument("--baseline_chair_json", required=True)
    ap.add_argument("--intervention_chair_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--parser_artifact_object_rows_csv", default="")
    ap.add_argument("--feature_rows_csv", default="")
    ap.add_argument("--feature_prefix", default="probe_")
    ap.add_argument("--recall_eps", type=float, default=0.05)
    ap.add_argument("--f1_eps", type=float, default=0.0)
    ap.add_argument("--ci_gain_min", type=float, default=0.01)
    ap.add_argument("--cs_gain_min", type=float, default=0.05)
    ap.add_argument("--soft_recall_eps", type=float, default=0.03)
    ap.add_argument("--soft_f1_eps", type=float, default=0.005)
    ap.add_argument("--soft_ci_gain_min", type=float, default=0.03)
    ap.add_argument("--tau_clean_ci", type=float, default=0.10)
    ap.add_argument("--tau_clean_h", type=float, default=1.0)
    ap.add_argument("--type_o_recall_gain", type=float, default=0.10)
    ap.add_argument("--type_o_f1_gain", type=float, default=0.08)
    ap.add_argument("--type_o_supported_gain", type=float, default=1.0)
    ap.add_argument("--type_o_ci_eps", type=float, default=0.02)
    ap.add_argument("--type_r_f1_gain", type=float, default=0.05)
    ap.add_argument("--type_r_recall_gain", type=float, default=0.05)
    ap.add_argument("--type_r_ci_delta", type=float, default=0.03)
    ap.add_argument("--safe_recall_gain", type=float, default=0.03)
    ap.add_argument("--safe_f1_gain", type=float, default=0.03)
    ap.add_argument("--safe_ci_eps", type=float, default=0.01)
    ap.add_argument("--top_ks", default="5,10,20,50,100")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    target_rows = build_metric_rows(
        os.path.abspath(args.baseline_chair_json),
        os.path.abspath(args.intervention_chair_json),
        recall_eps=float(args.recall_eps),
        f1_eps=float(args.f1_eps),
        ci_gain_min=float(args.ci_gain_min),
        cs_gain_min=float(args.cs_gain_min),
        soft_recall_eps=float(args.soft_recall_eps),
        soft_f1_eps=float(args.soft_f1_eps),
        soft_ci_gain_min=float(args.soft_ci_gain_min),
        parser_artifact_object_rows_csv=str(args.parser_artifact_object_rows_csv),
        tau_clean_ci=float(args.tau_clean_ci),
        tau_clean_h=float(args.tau_clean_h),
        type_o_recall_gain=float(args.type_o_recall_gain),
        type_o_f1_gain=float(args.type_o_f1_gain),
        type_o_supported_gain=float(args.type_o_supported_gain),
        type_o_ci_eps=float(args.type_o_ci_eps),
        type_r_f1_gain=float(args.type_r_f1_gain),
        type_r_recall_gain=float(args.type_r_recall_gain),
        type_r_ci_delta=float(args.type_r_ci_delta),
        safe_recall_gain=float(args.safe_recall_gain),
        safe_f1_gain=float(args.safe_f1_gain),
        safe_ci_eps=float(args.safe_ci_eps),
    )
    target_csv = os.path.join(out_dir, "net_harm_rows.csv")
    write_csv(target_csv, target_rows)

    top_ks = [int(x) for x in str(args.top_ks).split(",") if x.strip()]
    auc_outputs: Dict[str, str] = {}
    best: Dict[str, Any] = {}
    for target_col in (
        "net_harm_strict_v2",
        "net_harm_type_o",
        "net_harm_type_r",
        "net_safe_strict_v2",
        "net_harm_strict_raw",
        "net_harm_strict_train",
        "net_harm_soft_raw",
        "net_harm_soft_train",
        "oracle_f1_nondecrease_no_more_hall_set",
        "oracle_recall_gain_f1_nondecrease_no_more_hall_set",
        "fallback_any_intervention_only_hall",
        "fallback_pure_regression_base_clean_to_int_hall",
    ):
        metrics = evaluate_feature_auc(
            target_rows,
            str(args.feature_rows_csv),
            target_col=target_col,
            feature_prefix=str(args.feature_prefix),
            top_ks=top_ks,
        )
        out_csv = os.path.join(out_dir, f"feature_auc_{target_col}.csv")
        if metrics:
            write_csv(out_csv, metrics)
            auc_outputs[target_col] = out_csv
            best[target_col] = metrics[:20]

    counts = {
        "n_rows": len(target_rows),
        "net_harm_strict_raw": sum(int(row["net_harm_strict_raw"]) for row in target_rows),
        "net_harm_strict_train": sum(int(row["net_harm_strict_train"]) for row in target_rows),
        "net_harm_soft_raw": sum(int(row["net_harm_soft_raw"]) for row in target_rows),
        "net_harm_soft_train": sum(int(row["net_harm_soft_train"]) for row in target_rows),
        "parser_artifact_only_sample": sum(int(row["parser_artifact_only_sample"]) for row in target_rows),
        "net_harm_strict_v2": sum(int(row["net_harm_strict_v2"]) for row in target_rows),
        "net_harm_type_o": sum(int(row["net_harm_type_o"]) for row in target_rows),
        "net_harm_type_r": sum(int(row["net_harm_type_r"]) for row in target_rows),
        "net_safe_strict_v2": sum(int(row["net_safe_strict_v2"]) for row in target_rows),
        "net_ignore_v2": sum(int(row["net_ignore_v2"]) for row in target_rows),
        "baseline_clean": sum(int(row["baseline_clean"]) for row in target_rows),
        "oracle_f1_nondecrease_no_more_hall_set": sum(int(row["oracle_f1_nondecrease_no_more_hall_set"]) for row in target_rows),
        "oracle_recall_gain_f1_nondecrease_no_more_hall_set": sum(int(row["oracle_recall_gain_f1_nondecrease_no_more_hall_set"]) for row in target_rows),
        "fallback_any_intervention_only_hall": sum(int(row["fallback_any_intervention_only_hall"]) for row in target_rows),
        "fallback_pure_regression_base_clean_to_int_hall": sum(int(row["fallback_pure_regression_base_clean_to_int_hall"]) for row in target_rows),
    }
    summary = {
        "inputs": {
            "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
            "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
            "parser_artifact_object_rows_csv": os.path.abspath(args.parser_artifact_object_rows_csv) if args.parser_artifact_object_rows_csv else "",
            "feature_rows_csv": os.path.abspath(args.feature_rows_csv) if args.feature_rows_csv else "",
            "thresholds": {
                "recall_eps": float(args.recall_eps),
                "f1_eps": float(args.f1_eps),
                "ci_gain_min": float(args.ci_gain_min),
                "cs_gain_min": float(args.cs_gain_min),
                "soft_recall_eps": float(args.soft_recall_eps),
                "soft_f1_eps": float(args.soft_f1_eps),
                "soft_ci_gain_min": float(args.soft_ci_gain_min),
                "tau_clean_ci": float(args.tau_clean_ci),
                "tau_clean_h": float(args.tau_clean_h),
                "type_o_recall_gain": float(args.type_o_recall_gain),
                "type_o_f1_gain": float(args.type_o_f1_gain),
                "type_o_supported_gain": float(args.type_o_supported_gain),
                "type_o_ci_eps": float(args.type_o_ci_eps),
                "type_r_f1_gain": float(args.type_r_f1_gain),
                "type_r_recall_gain": float(args.type_r_recall_gain),
                "type_r_ci_delta": float(args.type_r_ci_delta),
                "safe_recall_gain": float(args.safe_recall_gain),
                "safe_f1_gain": float(args.safe_f1_gain),
                "safe_ci_eps": float(args.safe_ci_eps),
            },
        },
        "counts": counts,
        "best_feature_auc": best,
        "outputs": {
            "target_rows_csv": target_csv,
            "feature_auc_csvs": auc_outputs,
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
