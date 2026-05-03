#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATASETS = ("mscoco", "aokvqa", "gqa")

LABELS = {
    "vga_llava15": "VGA / LLaVA-1.5",
    "vga_llava_next": "VGA / LLaVA-NeXT",
    "vga_qwen25_vl_7b": "VGA / Qwen2.5-VL-7B",
    "llava15_vaf": "VAF / LLaVA-1.5",
    "llava15_pai_attn": "PAI-attn / LLaVA-1.5",
    "llava_next_vaf": "VAF / LLaVA-NeXT",
    "llava_next_pai_attn": "PAI-attn / LLaVA-NeXT",
    "qwen25_vaf": "VAF / Qwen2.5-VL-7B",
    "qwen25_pai_attn": "PAI-attn / Qwen2.5-VL-7B",
}

QUESTION_OBJECT_RE = re.compile(
    r"Is there (?:a |an |any |the )?(.+?) in the image\?",
    re.IGNORECASE,
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sid(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("question_id") or "").strip()


def to_int(value: Any) -> int:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return 0
        return int(round(float(text)))
    except Exception:
        return 0


def to_float(value: Any) -> Optional[float]:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return None
        out = float(text)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def quantile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    xs = sorted(x for x in values if x is not None)
    if not xs:
        return None
    return xs[int((len(xs) - 1) * q)]


def mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def norm_obj(question: str) -> str:
    match = QUESTION_OBJECT_RE.search(str(question or ""))
    if not match:
        return str(question or "")[:80].strip().lower()
    obj = match.group(1).strip().lower()
    return obj.rstrip(". ")


def discover_targets(split_root: Path) -> List[str]:
    apply_root = split_root / "apply"
    if not apply_root.exists():
        return []
    return sorted(p.name for p in apply_root.iterdir() if p.is_dir())


def feature_path_for(source_root: Path, target: str, dataset: str) -> Path:
    return source_root / "methods" / target / f"apply_{dataset}" / "features" / "online_feature_rows.csv"


def route_path_for(split_root: Path, target: str, dataset: str) -> Path:
    return split_root / "apply" / target / dataset / "pcp_route_rows.csv"


def deployment_path_for(split_root: Path, target: str, dataset: str) -> Path:
    return split_root / "apply" / target / dataset / "deployment_summary.json"


def feature_gaps(
    *,
    help_feats: Sequence[Dict[str, str]],
    harm_feats: Sequence[Dict[str, str]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not help_feats or not harm_feats:
        return []
    skip = {
        "id",
        "question_id",
        "image",
        "question",
        "baseline_text",
        "intervention_text",
        "final_text",
        "baseline_correct",
        "intervention_correct",
        "final_correct",
        "harm",
        "help",
    }
    rows: List[Dict[str, Any]] = []
    for col in help_feats[0].keys():
        if col in skip:
            continue
        hv = [to_float(row.get(col)) for row in help_feats]
        hm = [to_float(row.get(col)) for row in harm_feats]
        hv2 = [x for x in hv if x is not None]
        hm2 = [x for x in hm if x is not None]
        if len(hv2) < 5 or len(hm2) < 5:
            continue
        pooled = pstdev(hv2 + hm2) or 1.0
        gap = (mean(hv2) - mean(hm2)) / pooled
        rows.append(
            {
                "feature": col,
                "z_gap_help_minus_harm": gap,
                "abs_z_gap": abs(gap),
                "help_mean": mean(hv2),
                "harm_mean": mean(hm2),
                "n_help_present": len(hv2),
                "n_harm_present": len(hm2),
            }
        )
    rows.sort(key=lambda r: float(r["abs_z_gap"]), reverse=True)
    return rows[: int(top_k)]


def audit_one(
    *,
    source_root: Path,
    split_root: Path,
    target: str,
    dataset: str,
    top_objects: int,
    top_features: int,
    examples: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    route_path = route_path_for(split_root, target, dataset)
    feature_path = feature_path_for(source_root, target, dataset)
    deploy_path = deployment_path_for(split_root, target, dataset)
    if not route_path.exists() or not feature_path.exists():
        raise FileNotFoundError(f"missing route/features for {target}/{dataset}: {route_path} {feature_path}")

    routes = read_csv(route_path)
    feats = {sid(row): row for row in read_csv(feature_path)}
    selected = [row for row in routes if str(row.get("route", "")).strip() == "baseline"]
    selected_help = [row for row in selected if to_int(row.get("help")) == 1]
    selected_harm = [row for row in selected if to_int(row.get("harm")) == 1]
    selected_neutral = [
        row for row in selected if to_int(row.get("help")) == 0 and to_int(row.get("harm")) == 0
    ]
    help_feats = [feats[sid(row)] for row in selected_help if sid(row) in feats]
    harm_feats = [feats[sid(row)] for row in selected_harm if sid(row) in feats]

    transition_help: Counter[str] = Counter()
    transition_harm: Counter[str] = Counter()
    object_help: Counter[str] = Counter()
    object_harm: Counter[str] = Counter()
    direction_help: Counter[str] = Counter()
    direction_harm: Counter[str] = Counter()
    for row in selected_help:
        fr = feats.get(sid(row), {})
        trans = f"{fr.get('baseline_label', '')}->{fr.get('intervention_label', '')}"
        transition_help[trans] += 1
        direction_help[str(row.get("route_policy_direction", ""))] += 1
        object_help[norm_obj(fr.get("question", row.get("question", "")))] += 1
    for row in selected_harm:
        fr = feats.get(sid(row), {})
        trans = f"{fr.get('baseline_label', '')}->{fr.get('intervention_label', '')}"
        transition_harm[trans] += 1
        direction_harm[str(row.get("route_policy_direction", ""))] += 1
        object_harm[norm_obj(fr.get("question", row.get("question", "")))] += 1

    def qstats(rows: Sequence[Dict[str, str]], col: str) -> Dict[str, Any]:
        values = [to_float(row.get(col)) for row in rows]
        return {
            "n": sum(x is not None for x in values),
            "p10": quantile(values, 0.1),
            "p50": quantile(values, 0.5),
            "p90": quantile(values, 0.9),
        }

    deployment = {}
    if deploy_path.exists():
        with deploy_path.open("r", encoding="utf-8") as f:
            deployment = json.load(f)

    summary = {
        "target": target,
        "label": LABELS.get(target, target),
        "dataset": dataset,
        "n_selected": len(selected),
        "selected_harm": len(selected_harm),
        "selected_help": len(selected_help),
        "selected_neutral": len(selected_neutral),
        "selected_help_rate": len(selected_help) / max(1, len(selected)),
        "selected_harm_rate": len(selected_harm) / max(1, len(selected)),
        "net": len(selected_harm) - len(selected_help),
        "transition_help_top": dict(transition_help.most_common(4)),
        "transition_harm_top": dict(transition_harm.most_common(4)),
        "route_direction_help_top": dict(direction_help.most_common(4)),
        "route_direction_harm_top": dict(direction_harm.most_common(4)),
        "help_objects_top": dict(object_help.most_common(top_objects)),
        "harm_objects_top": dict(object_harm.most_common(top_objects)),
        "help_d_score": qstats(selected_help, "d_score"),
        "harm_d_score": qstats(selected_harm, "d_score"),
        "help_c_score": qstats(selected_help, "c_score"),
        "harm_c_score": qstats(selected_harm, "c_score"),
        "deployment": deployment,
        "route_rows_csv": str(route_path),
        "feature_rows_csv": str(feature_path),
    }

    gaps = [
        {
            "target": target,
            "dataset": dataset,
            **row,
        }
        for row in feature_gaps(help_feats=help_feats, harm_feats=harm_feats, top_k=top_features)
    ]
    ex_rows: List[Dict[str, Any]] = []
    for row in selected_help[: int(examples)]:
        fr = feats.get(sid(row), {})
        ex_rows.append(
            {
                "target": target,
                "dataset": dataset,
                "id": sid(row),
                "route_policy_direction": row.get("route_policy_direction", ""),
                "transition": f"{fr.get('baseline_label', '')}->{fr.get('intervention_label', '')}",
                "object": norm_obj(fr.get("question", row.get("question", ""))),
                "question": fr.get("question", row.get("question", "")),
                "baseline_text": fr.get("baseline_text", ""),
                "intervention_text": fr.get("intervention_text", ""),
                "d_score": row.get("d_score", ""),
                "c_score": row.get("c_score", ""),
            }
        )
    return summary, gaps, ex_rows


def format_md(summaries: Sequence[Dict[str, Any]], gaps: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Method / Backbone | Dataset | Selected | H/G/Net | Help% | Top Help Transition | Top Help Route | Help d p50 | Harm d p50 | Top Help Objects |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in summaries:
        help_trans = next(iter(row["transition_help_top"].items()), ("", 0))
        help_dir = next(iter(row["route_direction_help_top"].items()), ("", 0))
        help_objs = ", ".join(f"{k}:{v}" for k, v in list(row["help_objects_top"].items())[:5])
        lines.append(
            f"| {row['label']} | {row['dataset']} | {row['n_selected']} | "
            f"{row['selected_harm']}/{row['selected_help']}/{row['net']} | "
            f"{100*float(row['selected_help_rate']):.1f} | "
            f"{help_trans[0]} ({help_trans[1]}) | {help_dir[0]} ({help_dir[1]}) | "
            f"{fmt(row['help_d_score']['p50'])} | {fmt(row['harm_d_score']['p50'])} | {help_objs} |"
        )

    lines += [
        "",
        "## Top Feature Gaps",
        "",
        "| Method / Backbone | Dataset | Feature | z_gap Help-Harm | Help Mean | Harm Mean |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in gaps:
        lines.append(
            f"| {LABELS.get(row['target'], row['target'])} | {row['dataset']} | {row['feature']} | "
            f"{float(row['z_gap_help_minus_harm']):+.3f} | {float(row['help_mean']):.4g} | {float(row['harm_mean']):.4g} |"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Audit common patterns among selected-help RaPiC fallbacks.")
    ap.add_argument("--source_root", required=True, help="Root containing methods/<target>/apply_<dataset>/features.")
    ap.add_argument("--split_root", required=True, help="Root containing split-calibrated apply/<target>/<dataset> routes.")
    ap.add_argument("--target", action="append", default=None, help="Target name to audit. Defaults to all under split_root/apply.")
    ap.add_argument("--dataset", action="append", choices=DATASETS, default=None, help="Dataset to audit. Defaults to all.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_objects", type=int, default=12)
    ap.add_argument("--top_features", type=int, default=8)
    ap.add_argument("--examples", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    split_root = Path(args.split_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    targets = args.target or discover_targets(split_root)
    datasets = args.dataset or list(DATASETS)

    summaries: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for target in targets:
        for dataset in datasets:
            try:
                summary, gap_rows, ex_rows = audit_one(
                    source_root=source_root,
                    split_root=split_root,
                    target=target,
                    dataset=dataset,
                    top_objects=int(args.top_objects),
                    top_features=int(args.top_features),
                    examples=int(args.examples),
                )
            except Exception as exc:
                errors.append({"target": target, "dataset": dataset, "error": str(exc)})
                continue
            summaries.append(summary)
            gaps.extend(gap_rows)
            examples.extend(ex_rows)

    write_json(out_dir / "selected_help_pattern_audit.json", {"summaries": summaries, "feature_gaps": gaps, "examples": examples, "errors": errors})
    write_csv(out_dir / "selected_help_pattern_summary.csv", summaries)
    write_csv(out_dir / "selected_help_feature_gaps.csv", gaps)
    write_csv(out_dir / "selected_help_examples.csv", examples)
    md = format_md(summaries, gaps[: max(1, len(summaries)) * 3])
    (out_dir / "selected_help_pattern_audit.md").write_text(md, encoding="utf-8")
    print(md)
    print("[saved]", out_dir)
    if errors:
        print("[warn] errors:")
        for err in errors:
            print(err)


if __name__ == "__main__":
    main()
