#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from typing import Any, Dict, List


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def best_single_by_family(single_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in single_rows:
        fam = str(r.get("family", "")).strip()
        if fam not in {"A", "B", "C", "D", "E"}:
            continue
        auc = float(r.get("auc_best_dir") or 0.0)
        if fam not in out or auc > float(out[fam].get("auc_best_dir") or 0.0):
            out[fam] = r
    return out


def best_composite_by_family(comp_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # eval_feature_families_screening.py writes family composites in column "set_name"
    # where A/B/C/D/E are family-only sets and A+E/C+D/... are combos.
    out: Dict[str, Dict[str, Any]] = {}
    for r in comp_rows:
        fam = str(r.get("set_name", "")).strip()
        if fam not in {"A", "B", "C", "D", "E"}:
            continue
        auc = float(r.get("auc_best_dir") or 0.0)
        if fam not in out or auc > float(out[fam].get("auc_best_dir") or 0.0):
            out[fam] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Structure-first family screening (A-E): FV vs VF / FP vs TP / incorrect vs correct."
    )
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--repo_root", type=str, default="/home/kms/LLaVA_calibration")
    ap.add_argument("--python_bin", type=str, default="python")
    ap.add_argument("--split_filter", type=str, default="all", choices=["all", "calib", "eval"])
    args = ap.parse_args()

    features_csv = os.path.abspath(args.features_csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = read_csv(features_csv)
    for r in rows:
        g = str(r.get("group", "")).strip()
        # FV = vanilla fail / VGA win = vcs_wrong_vga_correct
        # VF = vanilla win / VGA fail = vcs_correct_vga_wrong
        r["target_fv_vs_vf"] = "1" if g == "vcs_wrong_vga_correct" else ("0" if g == "vcs_correct_vga_wrong" else "")
        r["target_vf_vs_fv"] = "1" if g == "vcs_correct_vga_wrong" else ("0" if g == "vcs_wrong_vga_correct" else "")

    derived_csv = os.path.join(out_dir, "features_with_pair_targets.csv")
    write_csv(derived_csv, rows)

    eval_script = os.path.join(os.path.abspath(args.repo_root), "scripts", "eval_feature_families_screening.py")
    tasks = [
        ("fp_vs_tp_yes", ["--target_mode", "fp_vs_tp_yes"]),
        ("fv_vs_vf", ["--target_mode", "custom_col", "--target_col", "target_fv_vs_vf"]),
        ("incorrect_vs_correct", ["--target_mode", "incorrect_vs_correct"]),
    ]

    run_meta: List[Dict[str, Any]] = []
    for task, mode_args in tasks:
        task_dir = os.path.join(out_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        cmd = [
            args.python_bin,
            eval_script,
            "--features_csv",
            derived_csv,
            *mode_args,
            "--split_filter",
            args.split_filter,
            "--out_dir",
            task_dir,
        ]
        proc = subprocess.run(cmd, cwd=os.path.abspath(args.repo_root), capture_output=True, text=True)
        run_meta.append(
            {
                "task": task,
                "returncode": int(proc.returncode),
                "cmd": " ".join(cmd),
                "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
                "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Task '{task}' failed (rc={proc.returncode}). See run_meta in summary.")

    summary_rows: List[Dict[str, Any]] = []
    for task, _ in tasks:
        task_dir = os.path.join(out_dir, task)
        comp_path = os.path.join(task_dir, "family_composite_metrics.csv")
        single_path = os.path.join(task_dir, "family_single_feature_metrics.csv")
        comp_rows = read_csv(comp_path) if os.path.isfile(comp_path) else []
        single_rows = read_csv(single_path) if os.path.isfile(single_path) else []

        best_comp = best_composite_by_family(comp_rows)
        best_single = best_single_by_family(single_rows)

        for fam in ["A", "B", "C", "D", "E"]:
            c = best_comp.get(fam, {})
            s = best_single.get(fam, {})
            summary_rows.append(
                {
                    "task": task,
                    "family": fam,
                    "composite_auc_best_dir": c.get("auc_best_dir", ""),
                    "composite_pr_auc_best_dir": c.get("ap_best_dir", ""),
                    "composite_ks_best_dir": c.get("ks", ""),
                    "single_feature": s.get("feature", ""),
                    "single_auc_best_dir": s.get("auc_best_dir", ""),
                    "single_pr_auc_best_dir": s.get("ap_best_dir", ""),
                    "single_ks_best_dir": s.get("ks", ""),
                }
            )

    summary_csv = os.path.join(out_dir, "family_structure_summary.csv")
    write_csv(summary_csv, summary_rows)

    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "features_csv": features_csv,
                    "repo_root": os.path.abspath(args.repo_root),
                    "python_bin": args.python_bin,
                    "split_filter": args.split_filter,
                },
                "runs": run_meta,
                "outputs": {
                    "derived_features_csv": derived_csv,
                    "family_structure_summary_csv": summary_csv,
                    "summary_json": summary_json,
                    "task_dirs": {task: os.path.join(out_dir, task) for task, _ in tasks},
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[saved]", derived_csv)
    print("[saved]", summary_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()

