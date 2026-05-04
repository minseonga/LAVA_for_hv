#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
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


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def as_int(obj: Any, key: str) -> int:
    try:
        return int(round(float(obj.get(key, 0) or 0)))
    except Exception:
        return 0


def as_float(obj: Any, key: str) -> float:
    try:
        return float(obj.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def maybe_float(obj: Any, key: str) -> Optional[float]:
    try:
        value = obj.get(key)
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def maybe_pct(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return ""
    out = 100.0 * float(value)
    return f"{out:+.2f}" if signed else f"{out:.2f}"


def signed_pct(value: float) -> str:
    return f"{100.0 * float(value):+.2f}"


def hgn(h: int, g: int) -> str:
    return f"{h}/{g}/{h - g}"


def short_policy(policy: Dict[str, Any]) -> str:
    family = str(policy.get("family", ""))
    if not family:
        return ""
    if family == "noop" or policy.get("disabled"):
        return "noop"
    return f"{family}@{float(policy.get('tau', 0.0) or 0.0):.3f}"


def policy_text(run_root: Optional[Path], target: str) -> str:
    if run_root is None:
        return ""
    yes_path = run_root / "policies" / target / "yes_to_no" / "selected_policy.json"
    no_path = run_root / "policies" / target / "no_to_yes" / "selected_policy.json"
    if yes_path.exists() or no_path.exists():
        yes = short_policy(read_json(yes_path).get("selected_policy") or {}) if yes_path.exists() else ""
        no = short_policy(read_json(no_path).get("selected_policy") or {}) if no_path.exists() else ""
        return f"Y:{yes or 'missing'} / N:{no or 'missing'}"

    single_path = run_root / "policies" / target / "selected_policy.json"
    if single_path.exists():
        return short_policy(read_json(single_path).get("selected_policy") or {})
    return ""


def infer_run_root(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part == "apply":
            return Path(*parts[:idx]) if idx else Path("/")
    return None


def infer_target_dataset(path: Path) -> Tuple[str, str]:
    dataset = path.parent.name
    target = path.parent.parent.name
    return target, dataset


def find_deployment_jsons(roots: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for root in roots:
        if root.is_file() and root.name == "deployment_summary.json":
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("deployment_summary.json"))
        else:
            candidates = []
        for path in candidates:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def build_row(path: Path, explicit_run_root: Optional[Path] = None) -> Dict[str, Any]:
    target, dataset = infer_target_dataset(path)
    d = read_json(path)
    total_h = as_int(d, "total_harm")
    total_g = as_int(d, "total_help")
    selected_h = as_int(d, "selected_harm")
    selected_g = as_int(d, "selected_help")
    final_h = total_h - selected_h
    final_g = total_g - selected_g
    fallback = as_int(d, "baseline_generated")
    total_hrec = selected_h / total_h if total_h else 0.0
    total_grec = selected_g / total_g if total_g else 0.0
    run_root = explicit_run_root or infer_run_root(path)

    base = as_float(d, "baseline_acc")
    method = as_float(d, "intervention_acc")
    ours = as_float(d, "pcp_deploy_acc")
    base_f1 = maybe_float(d, "baseline_f1")
    method_f1 = maybe_float(d, "intervention_f1")
    ours_f1 = maybe_float(d, "pcp_deploy_f1")
    delta_f1_vs_method = None if method_f1 is None or ours_f1 is None else float(ours_f1 - method_f1)
    delta_f1_vs_base = None if base_f1 is None or ours_f1 is None else float(ours_f1 - base_f1)

    return {
        "target": target,
        "method_backbone": LABELS.get(target, target),
        "dataset": dataset,
        "policies": policy_text(run_root, target),
        "baseline_acc": base,
        "method_acc": method,
        "ours_acc": ours,
        "delta_vs_method": ours - method,
        "delta_vs_baseline": ours - base,
        "baseline_f1": base_f1,
        "method_f1": method_f1,
        "ours_f1": ours_f1,
        "delta_f1_vs_method": delta_f1_vs_method,
        "delta_f1_vs_baseline": delta_f1_vs_base,
        "method_harm": total_h,
        "method_gain": total_g,
        "method_net_h_minus_g": total_h - total_g,
        "fallback": fallback,
        "fallback_harm": selected_h,
        "fallback_gain": selected_g,
        "fallback_net_h_minus_g": selected_h - selected_g,
        "final_harm": final_h,
        "final_gain": final_g,
        "final_net_h_minus_g": final_h - final_g,
        "hrec": total_hrec,
        "grec": total_grec,
        "deployment_summary_json": str(path.resolve()),
    }


def sort_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    method = str(row["method_backbone"])
    dataset = str(row["dataset"])
    ds_order = {name: idx for idx, name in enumerate(DATASETS)}
    return method, ds_order.get(dataset, 99), dataset


def markdown(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Method / Backbone | Dataset | Policies | Base Acc | Method Acc | Ours Acc | dMethod Acc | dBase Acc | Base F1 | Method F1 | Ours F1 | dMethod F1 | dBase F1 | Method H/G/Net | Fallback H/G/Net | Final H/G/Net | Fallback | Hrec | Grec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_backbone']} | {row['dataset']} | {row.get('policies') or ''} | "
            f"{pct(row['baseline_acc'])} | {pct(row['method_acc'])} | {pct(row['ours_acc'])} | "
            f"{signed_pct(row['delta_vs_method'])} | {signed_pct(row['delta_vs_baseline'])} | "
            f"{maybe_pct(row.get('baseline_f1'))} | {maybe_pct(row.get('method_f1'))} | "
            f"{maybe_pct(row.get('ours_f1'))} | {maybe_pct(row.get('delta_f1_vs_method'), signed=True)} | "
            f"{maybe_pct(row.get('delta_f1_vs_baseline'), signed=True)} | "
            f"{hgn(int(row['method_harm']), int(row['method_gain']))} | "
            f"{hgn(int(row['fallback_harm']), int(row['fallback_gain']))} | "
            f"{hgn(int(row['final_harm']), int(row['final_gain']))} | "
            f"{row['fallback']} | {pct(row['hrec'])} | {pct(row['grec'])} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize discriminative RaPiC deployment JSONs with method-level, "
            "fallback-selected, and final remaining harm/gain/net counts."
        )
    )
    ap.add_argument("--apply_root", action="append", required=True, help="Root containing deployment_summary.json files.")
    ap.add_argument("--run_root", default="", help="Optional experiment root used to resolve policy text.")
    ap.add_argument("--target", action="append", default=None, help="Only include these target names.")
    ap.add_argument("--exclude_target", action="append", default=None, help="Exclude these target names.")
    ap.add_argument("--dataset", action="append", choices=DATASETS, default=None, help="Only include these datasets.")
    ap.add_argument("--out_md", default="")
    ap.add_argument("--out_csv", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p).resolve() for p in args.apply_root]
    explicit_run_root = Path(args.run_root).resolve() if args.run_root else None
    target_filter = set(args.target or [])
    exclude_targets = set(args.exclude_target or [])
    dataset_filter = set(args.dataset or [])

    rows: List[Dict[str, Any]] = []
    for path in find_deployment_jsons(roots):
        target, dataset = infer_target_dataset(path)
        if target_filter and target not in target_filter:
            continue
        if target in exclude_targets:
            continue
        if dataset_filter and dataset not in dataset_filter:
            continue
        rows.append(build_row(path, explicit_run_root=explicit_run_root))
    rows.sort(key=sort_key)

    table = markdown(rows)
    print(table)
    if args.out_md:
        out_md = Path(args.out_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(table + "\n", encoding="utf-8")
        print("[saved]", out_md)
    if args.out_csv:
        out_csv = Path(args.out_csv).resolve()
        write_csv(out_csv, rows)
        print("[saved]", out_csv)


if __name__ == "__main__":
    main()
