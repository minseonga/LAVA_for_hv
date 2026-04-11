#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any, Counter, Dict, Iterable, List, Optional, Sequence, Set


RATE_KEYS = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1")


def normalize_rate(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if abs(out) > 1.0:
        out /= 100.0
    return out


def canonical_object(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[-1]).strip()
    return str(value).strip()


def canonical_set(values: Iterable[Any]) -> Set[str]:
    return {canonical_object(value) for value in values if canonical_object(value)}


def canonical_list(values: Iterable[Any]) -> List[str]:
    return [canonical_object(value) for value in values if canonical_object(value)]


def f1(precision: float, recall: float) -> float:
    denom = float(precision) + float(recall)
    if denom <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def compute(sentences: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n_caps = 0
    n_hall_caps = 0
    n_hall_instances = 0
    n_generated_instances = 0
    n_generated_unique = 0
    n_gt_objects = 0
    n_supported_unique = 0
    n_words = 0
    n_sentences_with_duplicate_objects = 0
    generated_counter: Counter[str] = collections.Counter()
    hallucinated_counter: Counter[str] = collections.Counter()
    supported_counter: Counter[str] = collections.Counter()

    for row in sentences:
        caption = str(row.get("caption", "")).strip()
        generated = canonical_list(row.get("mscoco_generated_words", []))
        gt = canonical_set(row.get("mscoco_gt_words", []))
        hallucinated_saved = row.get("mscoco_hallucinated_words", [])
        hallucinated = canonical_list(hallucinated_saved)
        if not hallucinated and generated:
            hallucinated = [obj for obj in generated if obj not in gt]

        supported = {obj for obj in generated if obj in gt}
        unique_generated = set(generated)
        if len(generated) > len(unique_generated):
            n_sentences_with_duplicate_objects += 1
        generated_counter.update(generated)
        hallucinated_counter.update(hallucinated)
        supported_counter.update(supported)
        words = row.get("words")
        if isinstance(words, list):
            n_words += len(words)
        elif caption:
            n_words += len(caption.split())

        n_caps += 1
        n_hall_caps += int(bool(hallucinated))
        n_hall_instances += len(hallucinated)
        n_generated_instances += len(generated)
        n_generated_unique += len(unique_generated)
        n_gt_objects += len(gt)
        n_supported_unique += len(supported)

    chair_s = n_hall_caps / float(n_caps) if n_caps else 0.0
    chair_i = n_hall_instances / float(n_generated_instances) if n_generated_instances else 0.0
    recall = n_supported_unique / float(n_gt_objects) if n_gt_objects else 0.0
    precision = n_supported_unique / float(n_generated_unique) if n_generated_unique else 0.0
    length = n_words / float(n_caps) if n_caps else 0.0

    return {
        "counts": {
            "n_caps": n_caps,
            "n_hall_caps": n_hall_caps,
            "n_hall_instances": n_hall_instances,
            "n_generated_instances": n_generated_instances,
            "n_generated_unique": n_generated_unique,
            "n_gt_objects": n_gt_objects,
            "n_supported_unique": n_supported_unique,
            "n_duplicate_object_mentions": n_generated_instances - n_generated_unique,
            "n_sentences_with_duplicate_objects": n_sentences_with_duplicate_objects,
        },
        "metrics": {
            "CHAIRs": chair_s,
            "CHAIRi": chair_i,
            "Recall": recall,
            "Precision": precision,
            "F1": f1(precision, recall),
            "Precision_1_minus_CHAIRi": 1.0 - chair_i,
            "F1_1_minus_CHAIRi": f1(1.0 - chair_i, recall),
            "Len": length,
            "avg_generated_object_mentions": n_generated_instances / float(n_caps) if n_caps else 0.0,
            "avg_generated_unique_objects": n_generated_unique / float(n_caps) if n_caps else 0.0,
            "avg_gt_objects": n_gt_objects / float(n_caps) if n_caps else 0.0,
            "avg_supported_unique_objects": n_supported_unique / float(n_caps) if n_caps else 0.0,
            "avg_hallucinated_object_mentions": n_hall_instances / float(n_caps) if n_caps else 0.0,
            "duplicate_object_mention_rate": (
                (n_generated_instances - n_generated_unique) / float(n_generated_instances)
                if n_generated_instances
                else 0.0
            ),
            "duplicate_sentence_rate": n_sentences_with_duplicate_objects / float(n_caps) if n_caps else 0.0,
        },
        "top_generated_objects": generated_counter.most_common(25),
        "top_supported_objects": supported_counter.most_common(25),
        "top_hallucinated_objects": hallucinated_counter.most_common(25),
    }


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.4f}"


def audit(path: str) -> Dict[str, Any]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    overall = obj.get("overall_metrics", {})
    sentences = obj.get("sentences", [])
    recomputed = compute(sentences)
    reported = {key: normalize_rate(overall.get(key)) for key in RATE_KEYS}
    reported["Len"] = overall.get("Len")
    return {
        "path": os.path.abspath(path),
        "reported": reported,
        "recomputed": recomputed,
    }


def print_audit(result: Dict[str, Any]) -> None:
    print("path:", result["path"])
    print("counts:", json.dumps(result["recomputed"]["counts"], sort_keys=True))
    print("metric,reported_pct,recomputed_pct,diff_pct")
    reported = result["reported"]
    metrics = result["recomputed"]["metrics"]
    for key in RATE_KEYS:
        rep = reported.get(key)
        rec = metrics.get(key)
        diff = None if rep is None else float(rec) - float(rep)
        print(f"{key},{fmt_rate(rep)},{fmt_rate(rec)},{fmt_rate(diff)}")
    print(f"Precision_1_minus_CHAIRi,NA,{fmt_rate(metrics['Precision_1_minus_CHAIRi'])},NA")
    print(f"F1_1_minus_CHAIRi,NA,{fmt_rate(metrics['F1_1_minus_CHAIRi'])},NA")
    print(f"Len,{reported.get('Len', 'NA')},{metrics['Len']:.4f},NA")
    print("diagnostics:")
    for key in (
        "avg_generated_object_mentions",
        "avg_generated_unique_objects",
        "avg_gt_objects",
        "avg_supported_unique_objects",
        "avg_hallucinated_object_mentions",
        "duplicate_object_mention_rate",
        "duplicate_sentence_rate",
    ):
        print(f"  {key}: {metrics[key]:.6f}")
    for key in ("top_generated_objects", "top_supported_objects", "top_hallucinated_objects"):
        print(f"  {key}: {result['recomputed'][key][:10]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit CHAIR object-count Recall/Precision/F1 from saved CHAIR JSON.")
    ap.add_argument("chair_json", nargs="+")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    results = [audit(path) for path in args.chair_json]
    for idx, result in enumerate(results):
        if idx:
            print()
        print_audit(result)

    if args.out_json:
        out_path = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
        print("[saved]", out_path)


if __name__ == "__main__":
    main()
