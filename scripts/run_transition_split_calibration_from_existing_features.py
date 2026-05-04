#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATASETS = ("mscoco", "aokvqa", "gqa")
DIRECTIONS = ("yes_to_no", "no_to_yes")

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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def maybe_int(value: object) -> int:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return 0
        return int(round(float(text)))
    except Exception:
        return 0


def label(row: Dict[str, Any], key: str) -> str:
    return str(row.get(key, "")).strip().lower()


def direction_match(row: Dict[str, Any], direction: str) -> bool:
    b = label(row, "baseline_label")
    i = label(row, "intervention_label")
    if direction == "yes_to_no":
        return b == "yes" and i == "no"
    if direction == "no_to_yes":
        return b == "no" and i == "yes"
    raise ValueError(direction)


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


def policy_paths_from_apply(old_apply_dir: Path) -> Tuple[Path, Path, Path]:
    summary = read_json(old_apply_dir / "summary.json")
    inputs = summary.get("inputs") or {}
    rows_csv = Path(str(inputs.get("rows_csv", "")))
    policy_json = Path(str(inputs.get("policy_json", "")))
    if not rows_csv.exists():
        raise FileNotFoundError(f"rows_csv missing in {old_apply_dir}: {rows_csv}")
    if not policy_json.exists():
        raise FileNotFoundError(f"policy_json missing in {old_apply_dir}: {policy_json}")
    policy_summary = policy_json.parent / "summary.json"
    if not policy_summary.exists():
        raise FileNotFoundError(f"policy summary missing next to {policy_json}: {policy_summary}")
    return rows_csv, policy_json, policy_summary


def count_direction_rows(rows_csv: Path, direction: str) -> int:
    return sum(int(direction_match(row, direction)) for row in read_csv(rows_csv))


def noop_policy_from_original(
    *,
    original_policy_json: Path,
    rows_csv: Path,
    out_dir: Path,
    direction: str,
    reason: str,
) -> None:
    bundle = read_json(original_policy_json)
    rows = read_csv(rows_csv)
    harm = sum(maybe_int(row.get("harm")) for row in rows)
    help_ = sum(maybe_int(row.get("help")) for row in rows)
    candidates = [row for row in rows if direction_match(row, direction)]
    cand_harm = sum(maybe_int(row.get("harm")) for row in candidates)
    cand_help = sum(maybe_int(row.get("help")) for row in candidates)
    noop = {
        "family": "noop",
        "alpha": 0.0,
        "tau": 0.0,
        "disabled": True,
        "n_eval": int(len(rows)),
        "baseline_rate": 0.0,
        "method_rate": 1.0,
        "final_acc": None,
        "baseline_acc": None,
        "intervention_acc": None,
        "delta_vs_intervention": 0.0,
        "selected_count": 0,
        "total_harm": int(harm),
        "total_help": int(help_),
        "n_route_candidates": int(len(candidates)),
        "n_route_candidate_harm": int(cand_harm),
        "n_route_candidate_help": int(cand_help),
        "selected_harm": 0,
        "selected_help": 0,
        "selected_neutral": 0,
        "net": 0,
        "selected_harm_precision": 0.0,
        "selected_help_precision": 0.0,
        "selected_harm_recall": 0.0,
        "selected_help_recall": 0.0,
        "selected_harm_f1": 0.0,
        "reason": reason,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bundle = {
        "rows_csv": str(rows_csv.resolve()),
        "candidate_filter": direction,
        "selected_c_features": bundle.get("selected_c_features") or [],
        "selected_d_features": bundle.get("selected_d_features") or [],
        "best_results": {"noop": noop},
        "selected_policy": noop,
    }
    write_json(out_dir / "selected_policy.json", out_bundle)
    write_json(
        out_dir / "summary.json",
        {
            "mode": "transition_split_noop_policy",
            "inputs": {
                "rows_csv": str(rows_csv.resolve()),
                "original_policy_json": str(original_policy_json.resolve()),
                "candidate_filter": direction,
            },
            "counts": {
                "n_rows": int(len(rows)),
                "n_route_candidates": int(len(candidates)),
                "n_route_candidate_harm": int(cand_harm),
                "n_route_candidate_help": int(cand_help),
            },
            "selected_policy": noop,
        },
    )


def build_direction_policy(
    *,
    cal: Path,
    cal_py: str,
    original_policy_json: Path,
    original_policy_summary: Path,
    out_dir: Path,
    direction: str,
    min_selected_count: int,
    allow_noop_policy: bool,
    max_help_recall: float,
) -> None:
    if (out_dir / "selected_policy.json").exists():
        return

    original_summary = read_json(original_policy_summary)
    inputs = original_summary.get("inputs") or {}
    rows_csv = Path(str(inputs.get("rows_csv", "")))
    if not rows_csv.exists():
        raise FileNotFoundError(f"policy rows_csv missing: {rows_csv}")

    n_direction = count_direction_rows(rows_csv, direction)
    if n_direction == 0 and allow_noop_policy:
        noop_policy_from_original(
            original_policy_json=original_policy_json,
            rows_csv=rows_csv,
            out_dir=out_dir,
            direction=direction,
            reason="no_rows_for_direction",
        )
        return

    cmd = [
        cal_py,
        str(cal / "scripts" / "build_pcp_c_d_controller.py"),
        "--rows_csv",
        str(rows_csv),
        "--c_feature_cols",
        ",".join(inputs.get("c_feature_cols") or []),
        "--d_feature_cols",
        ",".join(inputs.get("d_feature_cols") or []),
        "--derive_decision_kl",
        str(bool(inputs.get("derive_decision_kl", True))).lower(),
        "--min_present_rate",
        str(inputs.get("min_present_rate", 0.8)),
        "--min_feature_auroc",
        str(inputs.get("min_feature_auroc", 0.55)),
        "--top_k_c",
        str(inputs.get("top_k_c", 3)),
        "--top_k_d",
        str(inputs.get("top_k_d", 4)),
        "--alpha_grid",
        ",".join(str(x) for x in (inputs.get("alpha_grid") or [])),
        "--tau_objective",
        str(inputs.get("tau_objective", "final_acc")),
        "--min_baseline_rate",
        str(inputs.get("min_baseline_rate", 0.0)),
        "--max_baseline_rate",
        str(inputs.get("max_baseline_rate", 1.0)),
        "--min_selected_count",
        str(min_selected_count),
        "--min_harm_precision",
        str(inputs.get("min_harm_precision", 0.0)),
        "--min_harm_recall",
        str(inputs.get("min_harm_recall", 0.0)),
        "--max_help_recall",
        str(max_help_recall),
        "--allow_noop_policy",
        str(bool(allow_noop_policy)).lower(),
        "--candidate_filter",
        direction,
        "--out_dir",
        str(out_dir),
    ]
    subprocess.check_call(cmd, cwd=str(cal))


def apply_direction_policy(
    *,
    cal: Path,
    cal_py: str,
    rows_csv: Path,
    policy_json: Path,
    direction: str,
    out_dir: Path,
) -> Path:
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
        direction,
        "--derive_decision_kl",
        "true",
    ]
    subprocess.check_call(cmd, cwd=str(cal))
    return out_dir / "pcp_route_rows.csv"


def merge_direction_routes(
    *,
    yes_routes_csv: Path,
    no_routes_csv: Path,
    out_csv: Path,
) -> None:
    yes_rows = read_csv(yes_routes_csv)
    no_rows = read_csv(no_routes_csv)
    no_by_id = {str(row.get("id") or row.get("question_id", "")).strip(): row for row in no_rows}
    merged: List[Dict[str, Any]] = []
    for yrow in yes_rows:
        sid = str(yrow.get("id") or yrow.get("question_id", "")).strip()
        nrow = no_by_id.get(sid, {})
        ybase = str(yrow.get("route", "")).strip() == "baseline"
        nbase = str(nrow.get("route", "")).strip() == "baseline"
        out = dict(yrow)
        out["route_yes_to_no"] = yrow.get("route", "")
        out["route_no_to_yes"] = nrow.get("route", "")
        if ybase:
            out["route"] = "baseline"
            out["route_policy_direction"] = "yes_to_no"
        elif nbase:
            out["route"] = "baseline"
            out["route_policy_direction"] = "no_to_yes"
        else:
            out["route"] = "method"
            out["route_policy_direction"] = "method"
        merged.append(out)
    write_csv(out_csv, merged)


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


def selected_policy_summary(policy_dir: Path) -> Dict[str, Any]:
    path = policy_dir / "selected_policy.json"
    if not path.exists():
        return {}
    return dict(read_json(path).get("selected_policy") or {})


def format_policy_pair(yes_policy: Dict[str, Any], no_policy: Dict[str, Any]) -> str:
    def short(policy: Dict[str, Any]) -> str:
        fam = str(policy.get("family", ""))
        if fam == "noop" or policy.get("disabled"):
            return "noop"
        return f"{fam}@{float(policy.get('tau', 0.0) or 0.0):.3f}"

    return f"Y:{short(yes_policy)} / N:{short(no_policy)}"


def format_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| Method / Backbone | Dataset | Policies | Base | Method | Split-Calib RaPiC | dMethod | dBase | Method H/G/Net | Fallback H/G/Net | Final H/G/Net | Fallback | Hrec | Grec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        d = row["deployment"]
        hrec = float(d["selected_harm"]) / float(d["total_harm"]) if int(d["total_harm"]) else 0.0
        grec = float(d["selected_help"]) / float(d["total_help"]) if int(d["total_help"]) else 0.0
        method_h = int(d["total_harm"])
        method_g = int(d["total_help"])
        fallback_h = int(d["selected_harm"])
        fallback_g = int(d["selected_help"])
        final_h = method_h - fallback_h
        final_g = method_g - fallback_g
        lines.append(
            f"| {row['label']} | {row['dataset']} | {row['policies']} | "
            f"{100*float(d['baseline_acc']):.2f} | {100*float(d['intervention_acc']):.2f} | "
            f"{100*float(d['pcp_deploy_acc']):.2f} | {100*float(d['delta_vs_intervention']):+.2f} | "
            f"{100*(float(d['pcp_deploy_acc']) - float(d['baseline_acc'])):+.2f} | "
            f"{method_h}/{method_g}/{method_h - method_g} | "
            f"{fallback_h}/{fallback_g}/{fallback_h - fallback_g} | "
            f"{final_h}/{final_g}/{final_h - final_g} | "
            f"{d['baseline_generated']} | "
            f"{100*hrec:.2f} | {100*grec:.2f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build separate discovery-calibrated RaPiC policies for yes->no and no->yes "
            "transitions, then apply and merge them on existing test feature rows."
        )
    )
    ap.add_argument("--cal", default=os.environ.get("CAL", "/home/kms/LLaVA_calibration"))
    ap.add_argument("--cal_py", default=os.environ.get("CAL_PY", "/home/kms/miniconda3/envs/vga_base/bin/python"))
    ap.add_argument("--apply_root", action="append", default=None)
    ap.add_argument(
        "--target",
        action="append",
        default=None,
        help="Only process these target directory names, e.g. llava_next_vaf. Can be repeated.",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        default=None,
        help="Only process these datasets. Can be repeated.",
    )
    ap.add_argument(
        "--out_root",
        default=os.environ.get("OUT_ROOT", ""),
        help="Output root. Defaults to $CAL/experiments/paper_pcp_cd_transition_split_calib_existing.",
    )
    ap.add_argument("--min_selected_count", type=int, default=int(os.environ.get("MIN_SELECTED_COUNT", "5")))
    ap.add_argument("--max_help_recall", type=float, default=float(os.environ.get("MAX_HELP_RECALL", "1.0")))
    ap.add_argument(
        "--allow_noop_policy",
        default=os.environ.get("ALLOW_NOOP_POLICY", "true").lower() in {"1", "true", "yes", "y", "on"},
        action=argparse.BooleanOptionalAction,
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cal = Path(args.cal).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else (
        cal / "experiments" / "paper_pcp_cd_transition_split_calib_existing"
    )
    apply_roots = [Path(p).resolve() for p in (args.apply_root or [])]
    if not apply_roots:
        apply_roots = [
            cal / "experiments" / "paper_pcp_cd_finalacc_alpha0p025_main" / "apply" / "vga",
            cal / "experiments" / "paper_pcp_cd_finalacc_alpha0p025_pai_vaf_main" / "apply",
        ]
    jobs = discover_apply_jobs(apply_roots)
    if args.target:
        targets = set(args.target)
        jobs = [job for job in jobs if job[0] in targets]
    if args.dataset:
        datasets = set(args.dataset)
        jobs = [job for job in jobs if job[1] in datasets]
    if not jobs:
        raise SystemExit(f"No apply jobs found under: {', '.join(str(p) for p in apply_roots)}")

    table_rows: List[Dict[str, Any]] = []
    built = set()
    for target, dataset, old_apply_dir in jobs:
        label = LABELS.get(target, target)
        rows_csv, original_policy_json, original_policy_summary = policy_paths_from_apply(old_apply_dir)
        policy_parent = out_root / "policies" / target
        for direction in DIRECTIONS:
            key = (target, direction)
            if key in built:
                continue
            build_direction_policy(
                cal=cal,
                cal_py=str(args.cal_py),
                original_policy_json=original_policy_json,
                original_policy_summary=original_policy_summary,
                out_dir=policy_parent / direction,
                direction=direction,
                min_selected_count=int(args.min_selected_count),
                allow_noop_policy=bool(args.allow_noop_policy),
                max_help_recall=float(args.max_help_recall),
            )
            built.add(key)

        apply_dir = out_root / "apply" / target / dataset
        print(f"== {label} / {dataset}")
        yes_routes = apply_direction_policy(
            cal=cal,
            cal_py=str(args.cal_py),
            rows_csv=rows_csv,
            policy_json=policy_parent / "yes_to_no" / "selected_policy.json",
            direction="yes_to_no",
            out_dir=apply_dir / "yes_to_no",
        )
        no_routes = apply_direction_policy(
            cal=cal,
            cal_py=str(args.cal_py),
            rows_csv=rows_csv,
            policy_json=policy_parent / "no_to_yes" / "selected_policy.json",
            direction="no_to_yes",
            out_dir=apply_dir / "no_to_yes",
        )
        merged_routes = apply_dir / "pcp_route_rows.csv"
        merge_direction_routes(
            yes_routes_csv=yes_routes,
            no_routes_csv=no_routes,
            out_csv=merged_routes,
        )
        deploy = summarize_from_existing_deployment(
            old_deploy=read_json(old_apply_dir / "deployment_summary.json"),
            route_rows_csv=merged_routes,
        )
        write_json(apply_dir / "deployment_summary.json", deploy)
        yes_policy = selected_policy_summary(policy_parent / "yes_to_no")
        no_policy = selected_policy_summary(policy_parent / "no_to_yes")
        print(json.dumps(deploy, ensure_ascii=False, indent=2))
        table_rows.append(
            {
                "target": target,
                "label": label,
                "dataset": dataset,
                "deployment": deploy,
                "policies": format_policy_pair(yes_policy, no_policy),
            }
        )

    table = format_table(table_rows)
    out_md = out_root / "transition_split_calib_summary.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")
    print("\n== summary table ==")
    print(table)
    print("[saved]", out_md)


if __name__ == "__main__":
    main()
