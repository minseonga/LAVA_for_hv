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
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})


def yn(x: Any) -> str:
    t = str(x or "").strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return t


def map_case_to_group(case_type: str) -> str:
    c = str(case_type or "").strip()
    mp = {
        "both_correct": "both_correct",
        "both_wrong": "both_wrong",
        "vga_improvement": "vcs_wrong_vga_correct",  # FV
        "vga_regression": "vcs_correct_vga_wrong",   # VF
    }
    return mp.get(c, c)


def main() -> None:
    ap = argparse.ArgumentParser(description="Join per_case taxonomy with feature table and run family-structure screening.")
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--per_case_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--pred_col", type=str, default="pred_vga")
    ap.add_argument("--gt_col", type=str, default="gt")
    ap.add_argument("--case_col", type=str, default="case_type")
    ap.add_argument("--python_bin", type=str, default="python")
    ap.add_argument("--repo_root", type=str, default="/home/kms/LLaVA_calibration")
    ap.add_argument("--split_filter", type=str, default="all", choices=["all", "calib", "eval"])
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    feat_rows = read_csv(os.path.abspath(args.features_csv))
    case_rows = read_csv(os.path.abspath(args.per_case_csv))

    id_col = str(args.id_col)
    pred_col = str(args.pred_col)
    gt_col = str(args.gt_col)
    case_col = str(args.case_col)

    case_map: Dict[str, Dict[str, Any]] = {}
    for r in case_rows:
        rid = str(r.get(id_col, "")).strip()
        if not rid:
            continue
        case_map[rid] = r

    merged: List[Dict[str, Any]] = []
    for fr in feat_rows:
        rid = str(fr.get(id_col, "")).strip()
        if not rid or rid not in case_map:
            continue
        cr = case_map[rid]

        pred = yn(cr.get(pred_col, ""))
        gt = yn(cr.get(gt_col, ""))

        rr = dict(fr)
        rr["group"] = map_case_to_group(cr.get(case_col, ""))

        rr["target_is_correct"] = 1 if (pred == gt and gt in {"yes", "no"}) else 0
        rr["target_is_fp_hallucination"] = 1 if (pred == "yes" and gt == "no") else 0
        rr["target_is_tp_yes"] = 1 if (pred == "yes" and gt == "yes") else 0
        rr["target_is_tn_no"] = 1 if (pred == "no" and gt == "no") else 0
        rr["target_is_fn_miss"] = 1 if (pred == "no" and gt == "yes") else 0

        ctype = str(cr.get(case_col, "")).strip()
        if ctype == "vga_improvement":
            rr["target_fv_vs_vf"] = 1
            rr["target_vf_vs_fv"] = 0
        elif ctype == "vga_regression":
            rr["target_fv_vs_vf"] = 0
            rr["target_vf_vs_fv"] = 1
        else:
            rr["target_fv_vs_vf"] = ""
            rr["target_vf_vs_fv"] = ""

        rr["target_pred_answer_eval"] = pred
        rr["answer_gt"] = gt

        merged.append(rr)

    merged_csv = os.path.join(out_dir, "features_model_targets.csv")
    write_csv(merged_csv, merged)

    structure_dir = os.path.join(out_dir, "family_structure")
    os.makedirs(structure_dir, exist_ok=True)

    eval_script = os.path.join(os.path.abspath(args.repo_root), "scripts", "eval_feature_families_structure.py")
    cmd = [
        args.python_bin,
        eval_script,
        "--features_csv",
        merged_csv,
        "--out_dir",
        structure_dir,
        "--repo_root",
        os.path.abspath(args.repo_root),
        "--python_bin",
        args.python_bin,
        "--split_filter",
        args.split_filter,
    ]
    proc = subprocess.run(cmd, cwd=os.path.abspath(args.repo_root), capture_output=True, text=True)

    summary = {
        "inputs": {
            "features_csv": os.path.abspath(args.features_csv),
            "per_case_csv": os.path.abspath(args.per_case_csv),
            "id_col": id_col,
            "pred_col": pred_col,
            "gt_col": gt_col,
            "case_col": case_col,
            "repo_root": os.path.abspath(args.repo_root),
            "python_bin": args.python_bin,
            "split_filter": args.split_filter,
        },
        "counts": {
            "n_features_rows": len(feat_rows),
            "n_case_rows": len(case_rows),
            "n_merged_rows": len(merged),
            "n_unique_ids_merged": len({str(r.get(id_col, "")) for r in merged if str(r.get(id_col, "")).strip()}),
        },
        "family_structure_run": {
            "returncode": int(proc.returncode),
            "cmd": " ".join(cmd),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-30:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-30:]),
        },
        "outputs": {
            "merged_features_csv": merged_csv,
            "family_structure_dir": structure_dir,
        },
    }

    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", merged_csv)
    print("[saved]", summary_json)

    if proc.returncode != 0:
        raise RuntimeError("eval_feature_families_structure.py failed; check summary.json tails")


if __name__ == "__main__":
    main()
