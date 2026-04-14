#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_vga_free_run_trajectory_drift import read_guidance_selected_samples
from diagnose_vga_lost_object_guidance import mode_for_oracle_row, read_csv_rows, safe_float, safe_id


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                cols.append(key)
                seen.add(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def row_out(row: Dict[str, str], group: str) -> Dict[str, Any]:
    return {
        "id": safe_id(row),
        "group": group,
        "mode": mode_for_oracle_row(row),
        "base_n_gt_objects": safe_float(row.get("base_n_gt_objects")),
        "n_base_only_supported_unique": safe_float(row.get("n_base_only_supported_unique")),
        "delta_recall_base_minus_int": safe_float(row.get("delta_recall_base_minus_int")),
        "delta_f1_unique_base_minus_int": safe_float(row.get("delta_f1_unique_base_minus_int")),
        "delta_ci_unique_base_minus_int": safe_float(row.get("delta_ci_unique_base_minus_int")),
        "base_only_supported_unique": row.get("base_only_supported_unique", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build target/control sample manifest for VGA prefix drift comparison.")
    ap.add_argument("--oracle-rows-csv", required=True)
    ap.add_argument("--guidance-rows-csv", default="")
    ap.add_argument("--target-col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--n-target", type=int, default=10)
    ap.add_argument("--n-safe-control", type=int, default=10)
    ap.add_argument("--n-loss-control", type=int, default=10)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rows = [r for r in read_csv_rows(args.oracle_rows_csv) if safe_id(r)]
    by_id = {safe_id(r): r for r in rows}
    target_rows = [r for r in rows if int(safe_float(r.get(args.target_col))) == 1]
    target_ids: List[str]
    if args.guidance_rows_csv:
        target_ids = [sid for sid in read_guidance_selected_samples(args.guidance_rows_csv, limit_samples=int(args.n_target) * 4) if sid in by_id and int(safe_float(by_id[sid].get(args.target_col))) == 1]
        target_ids = target_ids[: int(args.n_target)]
    else:
        target_ids = [safe_id(r) for r in target_rows[: int(args.n_target)]]

    target_gt = [safe_float(by_id[sid].get("base_n_gt_objects")) for sid in target_ids if sid in by_id]
    target_gt_mid = sorted(target_gt)[len(target_gt) // 2] if target_gt else 3.0

    def control_score(row: Dict[str, str]) -> tuple:
        # Prefer scenes with similar object count to target; stable deterministic tie-break.
        return (
            abs(safe_float(row.get("base_n_gt_objects")) - target_gt_mid),
            -safe_float(row.get("base_n_gt_objects")),
            safe_id(row),
        )

    target_set = set(target_ids)
    safe_controls = [
        r
        for r in rows
        if safe_id(r) not in target_set
        and int(safe_float(r.get(args.target_col))) == 0
        and safe_float(r.get("n_base_only_supported_unique")) == 0
        and safe_float(r.get("int_f1_unique")) >= safe_float(r.get("base_f1_unique"))
    ]
    if len(safe_controls) < int(args.n_safe_control):
        safe_controls = [
            r
            for r in rows
            if safe_id(r) not in target_set
            and int(safe_float(r.get(args.target_col))) == 0
            and safe_float(r.get("n_base_only_supported_unique")) == 0
        ]
    safe_controls = sorted(safe_controls, key=control_score)[: int(args.n_safe_control)]

    used = target_set | {safe_id(r) for r in safe_controls}
    loss_controls = [
        r
        for r in rows
        if safe_id(r) not in used
        and int(safe_float(r.get(args.target_col))) == 0
        and safe_float(r.get("n_base_only_supported_unique")) > 0
    ]
    loss_controls = sorted(loss_controls, key=control_score)[: int(args.n_loss_control)]

    out_rows: List[Dict[str, Any]] = []
    for sid in target_ids:
        out_rows.append(row_out(by_id[sid], "target_recoverable"))
    for row in safe_controls:
        out_rows.append(row_out(row, "safe_control"))
    for row in loss_controls:
        out_rows.append(row_out(row, "loss_nonrecoverable_control"))

    write_csv(args.out_csv, out_rows)
    write_json(
        args.out_json,
        {
            "inputs": vars(args),
            "counts": {
                "n_rows": len(out_rows),
                "n_target": sum(r["group"] == "target_recoverable" for r in out_rows),
                "n_safe_control": sum(r["group"] == "safe_control" for r in out_rows),
                "n_loss_control": sum(r["group"] == "loss_nonrecoverable_control" for r in out_rows),
            },
            "target_gt_mid": target_gt_mid,
            "rows": out_rows,
            "outputs": {"csv": os.path.abspath(args.out_csv), "json": os.path.abspath(args.out_json)},
        },
    )
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_json))
    print("[counts]", len(out_rows))


if __name__ == "__main__":
    main()
