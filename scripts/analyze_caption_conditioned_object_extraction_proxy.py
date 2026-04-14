#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from extract_generative_semantic_pairwise_features import read_prediction_map, write_csv, write_json


LIST_PREFIX_RE = re.compile(r"^\s*(?:objects?|object list|answer|output)\s*:\s*", flags=re.IGNORECASE)

ABSTRACT_OR_NONOBJECT = {
    "action",
    "activity",
    "addition",
    "appearance",
    "area",
    "arrangement",
    "atmosphere",
    "background",
    "break",
    "color",
    "comfort",
    "direction",
    "edge",
    "environment",
    "expertise",
    "focus",
    "foreground",
    "game",
    "journey",
    "landscape",
    "location",
    "moment",
    "none",
    "object",
    "objects",
    "other",
    "part",
    "piece",
    "portion",
    "position",
    "process",
    "progress",
    "project",
    "scene",
    "setting",
    "size",
    "skill",
    "space",
    "standing",
    "state",
    "style",
    "talent",
    "task",
    "thing",
    "things",
    "time",
    "total",
    "touch",
    "unity",
    "use",
    "variety",
    "view",
    "warmth",
    "way",
}

PEOPLE_ALIASES = {
    "adult",
    "boy",
    "child",
    "children",
    "girl",
    "kid",
    "man",
    "men",
    "passenger",
    "pedestrian",
    "people",
    "person",
    "player",
    "rider",
    "skier",
    "snowboarder",
    "surfer",
    "teammate",
    "woman",
    "women",
    "worker",
}

ALIASES = {
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "phone": "cell phone",
    "controller": "remote",
    "remote controller": "remote",
    "wii remote": "remote",
    "tennis ball": "sports ball",
    "sport ball": "sports ball",
    "ball": "sports ball",
    "table": "dining table",
    "plant": "potted plant",
}


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def safe_int_flag(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def normalize_text(value: str) -> str:
    value = str(value or "").lower()
    value = value.replace("break pedal", "brake pedal")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def singularize_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ves"):
        return word[:-3] + "f"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize_object(raw: str) -> str:
    text = LIST_PREFIX_RE.sub("", str(raw or "")).strip()
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
    text = normalize_text(text)
    if not text:
        return ""
    words = [singularize_word(w) for w in text.split() if w]
    if not words:
        return ""
    if words[-1] in PEOPLE_ALIASES:
        return "person"
    text = " ".join(words)
    text = ALIASES.get(text, text)
    if text in ABSTRACT_OR_NONOBJECT:
        return ""
    parts = [p for p in text.split() if p not in {"a", "an", "the"}]
    if not parts:
        return ""
    if parts[-1] in ABSTRACT_OR_NONOBJECT:
        return ""
    if all(p in ABSTRACT_OR_NONOBJECT for p in parts):
        return ""
    return " ".join(parts)


def split_object_list(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
    text = LIST_PREFIX_RE.sub("", text)
    text = re.sub(r"\b(?:and|or)\s+others?\b", "", text, flags=re.IGNORECASE)
    raw_parts = re.split(r"[,;\n]+", text)
    out: List[str] = []
    seen = set()
    for part in raw_parts:
        obj = normalize_object(part)
        if not obj or obj in seen:
            continue
        seen.add(obj)
        out.append(obj)
    return out


def split_pipe_objects(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for part in str(text or "").split("|"):
        obj = normalize_object(part)
        if not obj or obj in seen:
            continue
        seen.add(obj)
        out.append(obj)
    return out


def object_matches(gold: str, pred: str) -> bool:
    gold = normalize_object(gold)
    pred = normalize_object(pred)
    if not gold or not pred:
        return False
    if gold == pred:
        return True
    gp = gold.split()
    pp = pred.split()
    if gp[-1] == pp[-1]:
        return True
    if set(gp).issubset(set(pp)) or set(pp).issubset(set(gp)):
        return True
    return False


def auc_high(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for a in pos:
        for b in neg:
            total += 1
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / float(total) if total else None


def average_precision(items: Sequence[Tuple[int, float]]) -> Optional[float]:
    ranked = sorted(items, key=lambda item: item[1], reverse=True)
    n_pos = sum(label for label, _ in ranked)
    if n_pos <= 0:
        return None
    hits = 0
    total = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label:
            hits += 1
            total += hits / float(rank)
    return total / float(n_pos)


def precision_at(items: Sequence[Tuple[int, float]], k: int) -> Optional[float]:
    top = sorted(items, key=lambda item: item[1], reverse=True)[:k]
    if not top:
        return None
    return sum(label for label, _ in top) / float(len(top))


def numeric_feature_metrics(rows: Sequence[Dict[str, Any]], target_col: str) -> List[Dict[str, Any]]:
    if not rows:
        return []
    excluded = {"id", target_col, "base_objects", "int_objects", "base_only_objects", "audit_base_only_supported"}
    metrics: List[Dict[str, Any]] = []
    for feature in rows[0].keys():
        if feature in excluded or feature.endswith("_text"):
            continue
        pairs: List[Tuple[int, float]] = []
        for row in rows:
            value = safe_float(row.get(feature))
            if value is None:
                continue
            pairs.append((safe_int_flag(row.get(target_col)), value))
        if len(pairs) < max(10, int(0.8 * len(rows))):
            continue
        if len({round(score, 12) for _, score in pairs}) < 3:
            continue
        pos = [score for label, score in pairs if label]
        neg = [score for label, score in pairs if not label]
        auc = auc_high(pos, neg)
        if auc is None:
            continue
        direction = "high" if auc >= 0.5 else "low"
        oriented = [(label, score if direction == "high" else -score) for label, score in pairs]
        metrics.append(
            {
                "feature": feature,
                "direction": direction,
                "auroc": max(float(auc), float(1.0 - auc)),
                "auroc_high": float(auc),
                "ap": average_precision(oriented),
                "p_at_10": precision_at(oriented, 10),
                "p_at_25": precision_at(oriented, 25),
                "p_at_50": precision_at(oriented, 50),
                "n": len(pairs),
                "n_pos": sum(label for label, _ in pairs),
                "pos_mean": sum(pos) / float(len(pos)) if pos else "",
                "neg_mean": sum(neg) / float(len(neg)) if neg else "",
            }
        )
    metrics.sort(key=lambda row: (float(row.get("auroc") or 0.0), float(row.get("ap") or 0.0)), reverse=True)
    return metrics


def preview(values: Iterable[str], limit: int) -> str:
    vals = list(values)
    if int(limit) > 0:
        vals = vals[: int(limit)]
    return " | ".join(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze caption-conditioned object extraction proxy.")
    ap.add_argument("--baseline_object_pred_jsonl", required=True)
    ap.add_argument("--intervention_object_pred_jsonl", required=True)
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_feature_metrics_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--preview_limit", type=int, default=80)
    args = ap.parse_args()

    base_pred = read_prediction_map(args.baseline_object_pred_jsonl)
    int_pred = read_prediction_map(args.intervention_object_pred_jsonl)
    oracle_by_id = {safe_id(row): row for row in read_csv_rows(args.oracle_rows_csv)}
    ids = sorted(
        set(base_pred.keys()) & set(int_pred.keys()) & set(oracle_by_id.keys()),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )

    rows: List[Dict[str, Any]] = []
    overlap_stats = {
        "target_samples_with_gold": 0,
        "target_samples_hit": 0,
        "target_gold_objects": 0,
        "target_gold_object_hits": 0,
        "nontarget_loss_samples_with_gold": 0,
        "nontarget_loss_samples_hit": 0,
        "nontarget_loss_gold_objects": 0,
        "nontarget_loss_gold_object_hits": 0,
    }
    for sid in ids:
        oracle = oracle_by_id[sid]
        base_objects = split_object_list(base_pred[sid].get("text", ""))
        int_objects = split_object_list(int_pred[sid].get("text", ""))
        base_set = set(base_objects)
        int_set = set(int_objects)
        base_only = [obj for obj in base_objects if obj not in int_set]
        int_only = [obj for obj in int_objects if obj not in base_set]
        shared = base_set & int_set
        gold = split_pipe_objects(oracle.get("base_only_supported_unique", ""))
        gold_hits = [g for g in gold if any(object_matches(g, pred) for pred in base_only)]
        sample_hit = int(bool(gold_hits))
        is_target = safe_int_flag(oracle.get(args.target_col))
        if gold:
            if is_target:
                overlap_stats["target_samples_with_gold"] += 1
                overlap_stats["target_samples_hit"] += sample_hit
                overlap_stats["target_gold_objects"] += len(gold)
                overlap_stats["target_gold_object_hits"] += len(gold_hits)
            else:
                overlap_stats["nontarget_loss_samples_with_gold"] += 1
                overlap_stats["nontarget_loss_samples_hit"] += sample_hit
                overlap_stats["nontarget_loss_gold_objects"] += len(gold)
                overlap_stats["nontarget_loss_gold_object_hits"] += len(gold_hits)

        row: Dict[str, Any] = {
            "id": sid,
            args.target_col: is_target,
            "base_objects": preview(base_objects, int(args.preview_limit)),
            "int_objects": preview(int_objects, int(args.preview_limit)),
            "base_only_objects": preview(base_only, int(args.preview_limit)),
            "audit_base_only_supported": preview(gold, int(args.preview_limit)),
            "capobj_base_object_count": len(base_objects),
            "capobj_int_object_count": len(int_objects),
            "capobj_shared_object_count": len(shared),
            "capobj_object_jaccard": len(shared) / float(max(1, len(base_set | int_set))),
            "capobj_base_only_object_count": len(base_only),
            "capobj_int_only_object_count": len(int_only),
            "capobj_base_minus_int_object_count": len(base_objects) - len(int_objects),
            "capobj_base_only_x_jaccard_gap": len(base_only) * (1.0 - len(shared) / float(max(1, len(base_set | int_set)))),
            "capobj_gold_base_only_supported_count": len(gold),
            "capobj_gold_hit_count": len(gold_hits),
            "capobj_gold_hit_rate": len(gold_hits) / float(max(1, len(gold))),
            "capobj_gold_sample_hit": sample_hit,
        }
        rows.append(row)

    metrics = numeric_feature_metrics(rows, args.target_col)
    target_count = sum(safe_int_flag(row.get(args.target_col)) for row in rows)
    summary = {
        "inputs": {
            "baseline_object_pred_jsonl": os.path.abspath(args.baseline_object_pred_jsonl),
            "intervention_object_pred_jsonl": os.path.abspath(args.intervention_object_pred_jsonl),
            "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
            "target_col": args.target_col,
        },
        "counts": {
            "n_rows": len(rows),
            "n_target": int(target_count),
            "target_rate": target_count / float(len(rows)) if rows else 0.0,
            "n_feature_metrics": len(metrics),
        },
        "overlap_stats": {
            **overlap_stats,
            "target_sample_hit_rate": overlap_stats["target_samples_hit"] / float(max(1, overlap_stats["target_samples_with_gold"])),
            "target_object_hit_rate": overlap_stats["target_gold_object_hits"] / float(max(1, overlap_stats["target_gold_objects"])),
            "nontarget_loss_sample_hit_rate": overlap_stats["nontarget_loss_samples_hit"]
            / float(max(1, overlap_stats["nontarget_loss_samples_with_gold"])),
            "nontarget_loss_object_hit_rate": overlap_stats["nontarget_loss_gold_object_hits"]
            / float(max(1, overlap_stats["nontarget_loss_gold_objects"])),
        },
        "top_feature_metrics": metrics[:30],
        "outputs": {
            "features_csv": os.path.abspath(args.out_csv),
            "feature_metrics_csv": os.path.abspath(args.out_feature_metrics_csv),
            "summary_json": os.path.abspath(args.out_summary_json),
        },
    }

    write_csv(args.out_csv, rows)
    write_csv(args.out_feature_metrics_csv, metrics)
    write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_feature_metrics_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))
    print("[overlap]", json.dumps(summary["overlap_stats"], sort_keys=True))
    for metric in metrics[:15]:
        print("[metric]", metric["feature"], "dir=", metric["direction"], "auc=", metric["auroc"], "ap=", metric["ap"])


if __name__ == "__main__":
    main()
