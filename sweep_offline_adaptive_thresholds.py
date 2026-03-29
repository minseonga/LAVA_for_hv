#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional


def parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for x in str(raw or "").split(","):
        xx = str(x).strip()
        if xx == "":
            continue
        out.append(float(xx))
    if len(out) == 0:
        raise ValueError("empty float list")
    return out


def parse_str_list_semicolon(raw: str) -> List[str]:
    out = [x.strip() for x in str(raw or "").split(";") if x.strip() != ""]
    if len(out) == 0:
        raise ValueError("empty string list")
    return out


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def pick_gate_row(table_csv: str, gate_name: str) -> Dict[str, Any]:
    rows = list(csv.DictReader(open(table_csv, encoding="utf-8")))
    for r in rows:
        if str(r.get("gate", "")) == str(gate_name):
            return dict(r)
    # fallback to best non-greedy
    non_greedy = [r for r in rows if str(r.get("gate", "")) != "greedy_only"]
    if len(non_greedy) > 0:
        non_greedy.sort(key=lambda x: (safe_float(x.get("delta_acc")) or -1e9), reverse=True)
        return dict(non_greedy[0])
    if len(rows) > 0:
        return dict(rows[0])
    raise RuntimeError(f"no rows in {table_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline sweep for adaptive gate/selector/trigger thresholds.")
    ap.add_argument("--greedy_dir", type=str, required=True)
    ap.add_argument("--expand_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--extra_candidates_cost", type=int, default=5)
    ap.add_argument("--twostage_script", type=str, default="/home/kms/LLaVA_calibration/analyze_artrap_twostage_minimal.py")
    ap.add_argument("--subset_json", type=str, default="", help="Optional subset IDs json for fast validation.")

    ap.add_argument("--gate_modes", type=str, default="vpmi_only")
    ap.add_argument("--tau_vpmi_values", type=str, default="-6.5,-6.0,-5.8,-5.5,-5.2,-5.0")
    ap.add_argument("--tau_sfull_values", type=str, default="-9.5,-9.0,-8.5,-8.0")

    ap.add_argument("--policy_values", type=str, default="agree_vminpm_wmin_dfull_le:-0.05")
    ap.add_argument("--trigger_values", type=str, default="P3;P3V_dv:0.02;P3V_dv:0.05;P3V_dv:0.10")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    gate_modes = [x.strip() for x in str(args.gate_modes).split(",") if x.strip() != ""]
    tau_v_vals = parse_float_list(args.tau_vpmi_values)
    tau_s_vals = parse_float_list(args.tau_sfull_values)
    policy_vals = parse_str_list_semicolon(args.policy_values)
    trigger_vals = parse_str_list_semicolon(args.trigger_values)

    runs: List[Dict[str, Any]] = []
    k = 0
    total = 0
    for gm in gate_modes:
        if gm == "and":
            total += len(tau_v_vals) * len(tau_s_vals) * len(policy_vals) * len(trigger_vals)
        else:
            total += len(tau_v_vals) * len(policy_vals) * len(trigger_vals)

    for gm in gate_modes:
        gm_l = str(gm).strip().lower()
        if gm_l not in {"vpmi_only", "and"}:
            raise ValueError(f"Unsupported gate mode in sweep: {gm}")
        s_grid = tau_s_vals if gm_l == "and" else [None]
        for tau_v in tau_v_vals:
            for tau_s in s_grid:
                tau_s_arg = "none" if tau_s is None else f"{float(tau_s):.6g}"
                gate_name = f"gate_{gm_l}_v{float(tau_v):.6g}_s{tau_s_arg}"
                gates_arg = f"{gate_name}|{gm_l}|{float(tau_v):.6g}|{tau_s_arg}"
                for pol in policy_vals:
                    for trig in trigger_vals:
                        k += 1
                        run_name = (
                            f"{gate_name}__pol_{str(pol).replace(':', '_').replace('/', '_')}"
                            f"__trig_{str(trig).replace(':', '_').replace('/', '_')}"
                        )
                        run_name = run_name[:220]
                        run_out = os.path.join(os.path.abspath(args.out_root), run_name)
                        os.makedirs(run_out, exist_ok=True)

                        cmd = [
                            sys.executable,
                            str(args.twostage_script),
                            "--greedy_dir", str(args.greedy_dir),
                            "--expand_dir", str(args.expand_dir),
                            "--out_dir", str(run_out),
                            "--eval_mode", str(args.eval_mode),
                            "--gates", str(gates_arg),
                            "--policy", str(pol),
                            "--trigger", str(trig),
                            "--extra_candidates_cost", str(int(args.extra_candidates_cost)),
                        ]
                        if str(args.subset_json).strip() != "":
                            cmd += ["--subset_json", str(args.subset_json).strip()]
                        print(f"[{k}/{total}] {run_name}")
                        subprocess.run(cmd, check=True)

                        table_csv = os.path.join(run_out, "adaptive_two_stage_table.csv")
                        row = pick_gate_row(table_csv, gate_name=gate_name)
                        out = {
                            "run_name": run_name,
                            "gate_name": gate_name,
                            "gate_mode": gm_l,
                            "tau_vpmi": float(tau_v),
                            "tau_sfull": (None if tau_s is None else float(tau_s)),
                            "policy": str(pol),
                            "trigger": str(trig),
                            "base_acc": safe_float(row.get("base_acc")),
                            "final_acc": safe_float(row.get("final_acc")),
                            "delta_acc": safe_float(row.get("delta_acc")),
                            "expand_rate": safe_float(row.get("expand_rate")),
                            "switch_rate": safe_float(row.get("switch_rate")),
                            "gain": safe_float(row.get("gain")),
                            "harm": safe_float(row.get("harm")),
                            "precision_gain": safe_float(row.get("precision_gain")),
                            "avg_cost_rel": safe_float(row.get("avg_cost_rel")),
                            "speedup_vs_fixed6": safe_float(row.get("speedup_vs_fixed6")),
                            "out_dir": run_out,
                            "table_csv": table_csv,
                            "summary_json": os.path.join(run_out, "summary.json"),
                        }
                        runs.append(out)

    runs_sorted = sorted(
        runs,
        key=lambda x: (
            -(safe_float(x.get("delta_acc")) or -1e9),
            (safe_float(x.get("harm")) or 1e9),
            (safe_float(x.get("expand_rate")) or 1e9),
        ),
    )
    out_csv = os.path.join(os.path.abspath(args.out_root), "offline_sweep_results.csv")
    out_json = os.path.join(os.path.abspath(args.out_root), "offline_sweep_results.json")
    write_csv(out_csv, runs_sorted)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "greedy_dir": os.path.abspath(args.greedy_dir),
                    "expand_dir": os.path.abspath(args.expand_dir),
                    "eval_mode": str(args.eval_mode),
                    "extra_candidates_cost": int(args.extra_candidates_cost),
                    "gate_modes": gate_modes,
                    "tau_vpmi_values": tau_v_vals,
                    "tau_sfull_values": tau_s_vals,
                    "policy_values": policy_vals,
                    "trigger_values": trigger_vals,
                    "twostage_script": os.path.abspath(args.twostage_script),
                    "subset_json": (None if str(args.subset_json).strip() == "" else os.path.abspath(str(args.subset_json).strip())),
                },
                "n_runs": int(len(runs_sorted)),
                "results_sorted": runs_sorted,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("[saved]", out_csv)
    print("[saved]", out_json)


if __name__ == "__main__":
    main()
