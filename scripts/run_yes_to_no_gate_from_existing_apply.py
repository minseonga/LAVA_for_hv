#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATASETS = ("mscoco", "aokvqa", "gqa")


LABELS = {
    "vga_llava15": "VGA / LLaVA-1.5",
    "vga_llava_next": "VGA / LLaVA-NeXT",
    "vga_qwen25_vl_7b": "VGA / Qwen2.5-VL-7B",
    "llava15_vaf": "VAF / LLaVA-1.5",
    "llava15_pai_attn": "PAI-attn / LLaVA-1.5",
    "qwen25_vaf": "VAF / Qwen2.5-VL-7B",
    "qwen25_pai_attn": "PAI-attn / Qwen2.5-VL-7B",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def maybe_int(value: object) -> int:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return 0
        return int(round(float(text)))
    except Exception:
        return 0


def discover_apply_jobs(apply_roots: Iterable[Path]) -> List[Tuple[str, str, Path]]:
    jobs: List[Tuple[str, str, Path]] = []
    seen = set()
    for root in apply_roots:
        if not root.exists():
            continue
        for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            target = target_dir.name
            for dataset in DATASETS:
                app_dir = target_dir / dataset
                if not (app_dir / "summary.json").exists() or not (app_dir / "deployment_summary.json").exists():
                    continue
                key = (target, dataset, str(app_dir.resolve()))
                if key in seen:
                    continue
                seen.add(key)
                jobs.append((target, dataset, app_dir))
    return jobs


def selected_policy(policy_json: Path) -> Dict[str, Any]:
    bundle = read_json(policy_json)
    return dict(bundle.get("selected_policy") or {})


def summarize_from_existing_deployment(
    *,
    old_deploy: Dict[str, Any],
    route_rows_csv: Path,
) -> Dict[str, Any]:
    rows = read_csv(route_rows_csv)
    selected = [row for row in rows if str(row.get("route", "")).strip() == "baseline"]
    selected_harm = sum(maybe_int(row.get("harm")) for row in selected)
    selected_help = sum(maybe_int(row.get("help")) for row in selected)
    selected_neutral = max(0, len(selected) - selected_harm - selected_help)
    net = selected_harm - selected_help
    n = int(old_deploy["n"])
    pcp_acc = float(old_deploy["intervention_acc"]) + float(net) / float(n)
    return {
        "n": n,
        "baseline_acc": float(old_deploy["baseline_acc"]),
        "intervention_acc": float(old_deploy["intervention_acc"]),
        "pcp_deploy_acc": float(pcp_acc),
        "delta_vs_intervention": float(pcp_acc) - float(old_deploy["intervention_acc"]),
        "baseline_generated": int(len(selected)),
        "actual_fallback": int(len(selected)),
        "flagged_unchanged": int(selected_neutral),
        "total_harm": int(old_deploy.get("total_harm", 0)),
        "total_help": int(old_deploy.get("total_help", 0)),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "net": int(net),
    }


def apply_yes_to_no_gate(
    *,
    cal: Path,
    cal_py: str,
    old_apply_dir: Path,
    out_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    apply_summary = read_json(old_apply_dir / "summary.json")
    old_deploy = read_json(old_apply_dir / "deployment_summary.json")
    inputs = apply_summary.get("inputs") or {}
    rows_csv = Path(str(inputs.get("rows_csv", "")))
    policy_json = Path(str(inputs.get("policy_json", "")))
    if not rows_csv.exists():
        raise FileNotFoundError(f"rows_csv missing in {old_apply_dir}: {rows_csv}")
    if not policy_json.exists():
        raise FileNotFoundError(f"policy_json missing in {old_apply_dir}: {policy_json}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        cal_py,
        str(cal / "scripts" / "apply_pcp_c_d_controller.py"),
        "--rows_csv",
        str(rows_csv),
        "--policy_json",
        str(policy_json),
        "--out_dir",
        str(out_dir),
        "--family",
        "selected",
        "--candidate_filter",
        "yes_to_no",
        "--derive_decision_kl",
        "true",
    ]
    subprocess.check_call(cmd, cwd=str(cal))
    deploy = summarize_from_existing_deployment(
        old_deploy=old_deploy,
        route_rows_csv=out_dir / "pcp_route_rows.csv",
    )
    write_json(out_dir / "deployment_summary.json", deploy)
    return deploy, selected_policy(policy_json)


def format_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Method / Backbone | Dataset | Family | Alpha | Tau | Base | Method | RaPiC yes->no | dMethod | dBase | Fallback | H/G/Net | Hrec | Grec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        d = row["deployment"]
        p = row["policy"]
        hrec = float(d["selected_harm"]) / float(d["total_harm"]) if int(d["total_harm"]) else 0.0
        grec = float(d["selected_help"]) / float(d["total_help"]) if int(d["total_help"]) else 0.0
        lines.append(
            f"| {row['label']} | {row['dataset']} | {p.get('family', '')} | "
            f"{float(p.get('alpha', 0.0) or 0.0):.3f} | {float(p.get('tau', 0.0) or 0.0):.4f} | "
            f"{100*float(d['baseline_acc']):.2f} | {100*float(d['intervention_acc']):.2f} | "
            f"{100*float(d['pcp_deploy_acc']):.2f} | {100*float(d['delta_vs_intervention']):+.2f} | "
            f"{100*(float(d['pcp_deploy_acc']) - float(d['baseline_acc'])):+.2f} | "
            f"{d['baseline_generated']} | {d['selected_harm']}/{d['selected_help']}/{d['net']} | "
            f"{100*hrec:.2f} | {100*grec:.2f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Apply an unlabeled yes->no transition gate to existing RaPiC apply outputs. "
            "This keeps each existing discovery-calibrated policy, but allows fallback only "
            "for rows whose baseline/method transition is yes->no."
        )
    )
    ap.add_argument("--cal", default=os.environ.get("CAL", "/home/kms/LLaVA_calibration"))
    ap.add_argument("--cal_py", default=os.environ.get("CAL_PY", "/home/kms/miniconda3/envs/vga_base/bin/python"))
    ap.add_argument(
        "--apply_root",
        action="append",
        default=None,
        help="Existing apply root containing target/dataset/summary.json directories. Can be repeated.",
    )
    ap.add_argument(
        "--out_root",
        default=os.environ.get("OUT_ROOT", ""),
        help="Output root. Defaults to $CAL/experiments/paper_pcp_cd_yes_to_no_apply_gate_existing.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cal = Path(args.cal).resolve()
    apply_roots = [Path(p).resolve() for p in (args.apply_root or [])]
    if not apply_roots:
        apply_roots = [
            cal / "experiments" / "paper_pcp_cd_finalacc_alpha0p025_main" / "apply" / "vga",
            cal / "experiments" / "paper_pcp_cd_finalacc_alpha0p025_pai_vaf_main" / "apply",
        ]
    out_root = Path(args.out_root).resolve() if args.out_root else (
        cal / "experiments" / "paper_pcp_cd_yes_to_no_apply_gate_existing"
    )

    jobs = discover_apply_jobs(apply_roots)
    if not jobs:
        raise SystemExit(f"No apply jobs found under: {', '.join(str(p) for p in apply_roots)}")

    table_rows: List[Dict[str, Any]] = []
    for target, dataset, old_apply_dir in jobs:
        label = LABELS.get(target, target)
        out_dir = out_root / target / dataset
        print(f"== {label} / {dataset}")
        deploy, policy = apply_yes_to_no_gate(
            cal=cal,
            cal_py=str(args.cal_py),
            old_apply_dir=old_apply_dir,
            out_dir=out_dir,
        )
        print(json.dumps(deploy, ensure_ascii=False, indent=2))
        table_rows.append(
            {
                "target": target,
                "label": label,
                "dataset": dataset,
                "deployment": deploy,
                "policy": policy,
            }
        )

    table = format_table(table_rows)
    out_md = out_root / "yes_to_no_apply_gate_summary.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")
    print("\n== summary table ==")
    print(table)
    print("[saved]", out_md)


if __name__ == "__main__":
    main()
