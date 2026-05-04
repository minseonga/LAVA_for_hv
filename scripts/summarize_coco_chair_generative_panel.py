#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TARGETS: Dict[str, Tuple[str, str]] = {
    "llava15_vga": ("LLaVA-1.5", "VGA"),
    "llava15_pai_attn": ("LLaVA-1.5", "PAI-attn"),
    "llava15_vaf": ("LLaVA-1.5", "VAF"),
    "llava_next_vga": ("LLaVA-NeXT", "VGA"),
    "llava_next_pai_attn": ("LLaVA-NeXT", "PAI-attn"),
    "llava_next_vaf": ("LLaVA-NeXT", "VAF"),
    "qwen25_vga": ("Qwen2.5-VL", "VGA"),
    "qwen25_pai_attn": ("Qwen2.5-VL", "PAI-attn"),
    "qwen25_vaf": ("Qwen2.5-VL", "VAF"),
}

BACKBONE_TO_BASELINE_KEY = {
    "LLaVA-1.5": "llava15",
    "LLaVA-NeXT": "llava_next",
    "Qwen2.5-VL": "qwen25",
}


def normalize_rate(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out / 100.0 if abs(out) > 1.0 else out


def canonical_object(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[-1]).strip()
    return str(value).strip()


def compute_object_pr(sentences: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    n_supported_unique = 0
    n_generated_unique = 0
    n_gt_objects = 0
    for row in sentences:
        generated = [canonical_object(value) for value in row.get("mscoco_generated_words", [])]
        generated = [value for value in generated if value]
        gt_objects = {canonical_object(value) for value in row.get("mscoco_gt_words", [])}
        gt_objects = {value for value in gt_objects if value}
        n_supported_unique += len({value for value in generated if value in gt_objects})
        n_generated_unique += len(set(generated))
        n_gt_objects += len(gt_objects)
    if n_generated_unique <= 0 or n_gt_objects <= 0:
        return None, None
    return n_supported_unique / n_generated_unique, n_supported_unique / n_gt_objects


def sentence_image_id(row: Dict[str, Any]) -> str:
    value = str(row.get("image_id", "")).strip()
    if value:
        try:
            return str(int(value))
        except Exception:
            return value
    return ""


def load_chair_json(path: Path) -> Tuple[Dict[str, float], int]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    overall = obj.get("overall_metrics", {})
    sentences = obj.get("sentences", [])
    precision, recall_from_sentences = compute_object_pr(sentences)
    recall = normalize_rate(overall.get("Recall"))
    if recall <= 0.0 and recall_from_sentences is not None:
        recall = recall_from_sentences
    precision_value = normalize_rate(overall.get("Precision"))
    if precision_value <= 0.0 and precision is not None:
        precision_value = precision
    if precision_value <= 0.0:
        precision_value = 1.0 - normalize_rate(overall.get("CHAIRi"))
    f1 = normalize_rate(overall.get("F1"))
    if f1 <= 0.0 and precision_value + recall > 0.0:
        f1 = 2.0 * precision_value * recall / (precision_value + recall)
    return (
        {
            "CHAIRs": normalize_rate(overall.get("CHAIRs")),
            "CHAIRi": normalize_rate(overall.get("CHAIRi")),
            "Recall": recall,
            "Precision": precision_value,
            "F1": f1,
            "Len": float(overall.get("Len", 0.0) or 0.0),
        },
        len(sentences),
    )


def load_sentence_metric_map(path: Path, metric: str) -> Dict[str, float]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    out: Dict[str, float] = {}
    for row in obj.get("sentences", []):
        image_id = sentence_image_id(row)
        if not image_id:
            continue
        metrics = row.get("metrics", {})
        if metric not in metrics:
            continue
        try:
            out[image_id] = float(metrics[metric])
        except (TypeError, ValueError):
            continue
    return out


def chair_delta_counts(baseline_path: Path, target_path: Path, metric: str, eps: float) -> Dict[str, Any]:
    base = load_sentence_metric_map(baseline_path, metric)
    target = load_sentence_metric_map(target_path, metric)
    common = sorted(set(base) & set(target))
    harm = 0
    gain = 0
    neutral = 0
    utility = 0.0
    for image_id in common:
        delta = float(base[image_id]) - float(target[image_id])
        utility += delta
        if delta > float(eps):
            gain += 1
        elif delta < -float(eps):
            harm += 1
        else:
            neutral += 1
    return {
        "delta_metric": metric,
        "delta_n": len(common),
        "harm": harm,
        "gain": gain,
        "neutral": neutral,
        "net": gain - harm,
        "mean_utility": utility / max(1, len(common)),
    }


def load_ours_csv(path: Path) -> Tuple[Dict[str, float], int]:
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    chosen = None
    for row in rows:
        if row.get("method") == "object_token_suppression":
            chosen = row
            break
    if chosen is None:
        raise ValueError(f"missing object_token_suppression row in {path}")
    return (
        {
            "CHAIRs": normalize_rate(chosen.get("CHAIRs")),
            "CHAIRi": normalize_rate(chosen.get("CHAIRi")),
            "Recall": normalize_rate(chosen.get("Recall")),
            "Precision": normalize_rate(chosen.get("Precision")),
            "F1": normalize_rate(chosen.get("F1")),
            "Len": float(chosen.get("Len", 0.0) or 0.0),
        },
        int(float(chosen.get("n", 0) or 0)),
    )


def load_ours_chair_json(path: Path) -> Optional[Path]:
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    for row in rows:
        if row.get("method") == "object_token_suppression":
            chair_json = str(row.get("chair_json", "")).strip()
            if chair_json:
                return Path(chair_json).expanduser().resolve()
    return None


def latest_match(pattern: str) -> Optional[Path]:
    paths = [Path(p) for p in glob.glob(pattern)]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def parse_existing(spec: str) -> Tuple[str, str, Path, Optional[Path]]:
    parts = spec.split("::")
    if len(parts) not in {3, 4}:
        raise ValueError("existing entry must be target::label::raw_chair_json[::ours_csv]")
    target, label, raw = parts[:3]
    ours = Path(parts[3]).expanduser().resolve() if len(parts) == 4 and parts[3] else None
    return target.strip(), label.strip(), Path(raw).expanduser().resolve(), ours


def parse_baseline_entry(spec: str) -> Tuple[str, Path]:
    parts = spec.split("::", 1)
    if len(parts) != 2:
        raise ValueError("baseline entry must be baseline_key::chair_json")
    return parts[0].strip(), Path(parts[1]).expanduser().resolve()


def add_metric_row(
    rows: List[Dict[str, Any]],
    *,
    target: str,
    backbone: str,
    method: str,
    variant: str,
    metrics: Dict[str, float],
    n: int,
    source: Path,
    delta: Optional[Dict[str, Any]] = None,
) -> None:
    delta = delta or {}
    rows.append(
        {
            "target": target,
            "Backbone": backbone,
            "Method": method if variant == "raw" else f"Ours ({method})",
            "Variant": variant,
            "n": n,
            "Harm": int(delta.get("harm", 0)),
            "Gain": int(delta.get("gain", 0)),
            "Neutral": int(delta.get("neutral", 0)),
            "Net": int(delta.get("net", 0)),
            "delta_n": int(delta.get("delta_n", 0)),
            "mean_utility": float(delta.get("mean_utility", 0.0)),
            "CHAIRs": metrics["CHAIRs"],
            "CHAIRi": metrics["CHAIRi"],
            "Recall": metrics["Recall"],
            "Precision": metrics["Precision"],
            "F1": metrics["F1"],
            "Len": metrics["Len"],
            "source": str(source),
        }
    )


def fmt_pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "target",
        "Backbone",
        "Method",
        "Variant",
        "n",
        "Harm",
        "Gain",
        "Neutral",
        "Net",
        "delta_n",
        "mean_utility",
        "CHAIRs",
        "CHAIRi",
        "Recall",
        "Precision",
        "F1",
        "Len",
        "source",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Backbone | Method | Variant | H/G/Net | CHAIRs ↓ | CHAIRi ↓ | Recall ↑ | Precision ↑ | F1 ↑ | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Backbone']} | {row['Method']} | {row['Variant']} | "
            f"{row['Harm']}/{row['Gain']}/{row['Net']} | "
            f"{fmt_pct(row['CHAIRs'])} | {fmt_pct(row['CHAIRi'])} | {fmt_pct(row['Recall'])} | "
            f"{fmt_pct(row['Precision'])} | {fmt_pct(row['F1'])} | {row['n']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize multibackbone CHAIR raw-method and Ours panel outputs.")
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--ours_root", required=True)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--existing_entry", action="append", default=[], help="target::label::raw_chair_json[::ours_csv]")
    ap.add_argument("--baseline_entry", action="append", default=[], help="baseline_key::chair_json, e.g. llava_next::/path/chair.json")
    ap.add_argument("--delta_metric", default="CHAIRi")
    ap.add_argument("--epsilon", type=float, default=1e-12)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    ours_root = Path(args.ours_root).expanduser().resolve()
    targets = args.target or list(TARGETS)
    baseline_paths: Dict[str, Path] = {}
    for spec in args.baseline_entry:
        key, path = parse_baseline_entry(spec)
        baseline_paths[key] = path

    rows: List[Dict[str, Any]] = []
    seen_existing: set[str] = set()

    for spec in args.existing_entry:
        target, label, raw_path, ours_csv = parse_existing(spec)
        backbone, method = TARGETS.get(target, ("", label))
        if " / " in label:
            method, backbone = label.split(" / ", 1)
        metrics, n = load_chair_json(raw_path)
        baseline = baseline_paths.get(BACKBONE_TO_BASELINE_KEY.get(backbone, ""))
        delta = chair_delta_counts(baseline, raw_path, args.delta_metric, args.epsilon) if baseline and baseline.exists() else None
        add_metric_row(rows, target=target, backbone=backbone, method=method, variant="raw", metrics=metrics, n=n, source=raw_path, delta=delta)
        if ours_csv is not None and ours_csv.exists():
            metrics, n = load_ours_csv(ours_csv)
            ours_chair = load_ours_chair_json(ours_csv)
            delta = (
                chair_delta_counts(baseline, ours_chair, args.delta_metric, args.epsilon)
                if baseline and baseline.exists() and ours_chair and ours_chair.exists()
                else None
            )
            add_metric_row(rows, target=target, backbone=backbone, method=method, variant="ours", metrics=metrics, n=n, source=ours_csv, delta=delta)
        seen_existing.add(target)

    missing: List[str] = []
    for target in targets:
        if target in seen_existing:
            continue
        if target not in TARGETS:
            missing.append(f"{target}: unknown target")
            continue
        backbone, method = TARGETS[target]
        raw_path = raw_root / target / "test" / f"chair_{target}.json"
        baseline = baseline_paths.get(BACKBONE_TO_BASELINE_KEY.get(backbone, ""))
        if raw_path.exists():
            metrics, n = load_chair_json(raw_path)
            delta = chair_delta_counts(baseline, raw_path, args.delta_metric, args.epsilon) if baseline and baseline.exists() else None
            add_metric_row(rows, target=target, backbone=backbone, method=method, variant="raw", metrics=metrics, n=n, source=raw_path, delta=delta)
        else:
            missing.append(f"{target}: missing raw {raw_path}")

        ours_csv = latest_match(str(ours_root / target / "test_apply_*" / "summary" / "*.csv"))
        if ours_csv is not None:
            metrics, n = load_ours_csv(ours_csv)
            ours_chair = load_ours_chair_json(ours_csv)
            delta = (
                chair_delta_counts(baseline, ours_chair, args.delta_metric, args.epsilon)
                if baseline and baseline.exists() and ours_chair and ours_chair.exists()
                else None
            )
            add_metric_row(rows, target=target, backbone=backbone, method=method, variant="ours", metrics=metrics, n=n, source=ours_csv, delta=delta)
        else:
            missing.append(f"{target}: missing ours under {ours_root / target}")

    order = {target: idx for idx, target in enumerate(TARGETS)}
    rows.sort(key=lambda r: (order.get(str(r["target"]), 999), 0 if r["Variant"] == "raw" else 1))
    write_csv(Path(args.out_csv), rows)
    write_md(Path(args.out_md), rows)
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_md))
    if missing:
        print("[warn] missing:")
        for item in missing:
            print("  -", item)


if __name__ == "__main__":
    main()
