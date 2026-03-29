#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for x in str(raw or "").split(","):
        xx = str(x).strip()
        if xx == "":
            continue
        vals.append(int(xx))
    if len(vals) == 0:
        raise ValueError("empty integer list")
    return vals


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


def first_gate_name(gates: str) -> str:
    rows = [x.strip() for x in str(gates).split(";") if str(x).strip() != ""]
    if len(rows) == 0:
        return "gate_and_vpmi_m5.5_and_sfull_m6"
    return str(rows[0]).split("|")[0].strip()


def pick_gate_row(table_csv: str, gate_name: str) -> Dict[str, Any]:
    rows = list(csv.DictReader(open(table_csv, encoding="utf-8")))
    for r in rows:
        if str(r.get("gate", "")) == str(gate_name):
            return dict(r)
    non_greedy = [r for r in rows if str(r.get("gate", "")) != "greedy_only"]
    if len(non_greedy) > 0:
        non_greedy.sort(key=lambda x: (safe_float(x.get("delta_acc")) or -1e9), reverse=True)
        return dict(non_greedy[0])
    if len(rows) > 0:
        return dict(rows[0])
    raise RuntimeError(f"no rows in {table_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid sweep for adaptive one-pass (beam, max_new_tokens)")
    ap.add_argument("--questions_json", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="")
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--gates", type=str, default="gate_and_vpmi_m5.5_and_sfull_m6|and|-5.5|-6")
    ap.add_argument("--policy", type=str, default="agree_vminpm_wmin_dfull_le:-0.05")
    ap.add_argument("--trigger", type=str, default="P3")
    ap.add_argument("--beam_values", type=str, default="4,6,8")
    ap.add_argument("--max_new_tokens_values", type=str, default="16,24,32")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")
    ap.add_argument("--enable_tqdm", action="store_true")
    ap.add_argument("--keep_pairwise_dirs", action="store_true")
    ap.add_argument(
        "--extra_candidates_cost_mode",
        type=str,
        default="beam_minus_1",
        choices=["beam_minus_1", "fixed"],
    )
    ap.add_argument("--extra_candidates_cost_fixed", type=int, default=5)
    ap.add_argument("--twostage_script", type=str, default="/home/kms/LLaVA_calibration/analyze_artrap_twostage_minimal.py")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    beam_vals = parse_int_list(args.beam_values)
    tok_vals = parse_int_list(args.max_new_tokens_values)
    gate_name = first_gate_name(args.gates)

    grid_rows: List[Dict[str, Any]] = []
    run_meta: List[Dict[str, Any]] = []
    total = int(len(beam_vals) * len(tok_vals))
    k = 0
    for beam in beam_vals:
        for tok in tok_vals:
            k += 1
            run_name = f"beam{int(beam)}_tok{int(tok)}"
            run_out = os.path.join(os.path.abspath(args.out_root), run_name)
            os.makedirs(run_out, exist_ok=True)
            ecc = int(args.extra_candidates_cost_fixed)
            if str(args.extra_candidates_cost_mode) == "beam_minus_1":
                ecc = int(max(0, int(beam) - 1))

            cmd: List[str] = [
                sys.executable,
                str(args.twostage_script),
                "--questions_json", str(args.questions_json),
                "--image_root", str(args.image_root),
                "--out_dir", str(run_out),
                "--model_path", str(args.model_path),
                "--eval_mode", str(args.eval_mode),
                "--gates", str(args.gates),
                "--policy", str(args.policy),
                "--trigger", str(args.trigger),
                "--num_samples", str(int(args.num_samples)),
                "--seed", str(int(args.seed)),
                "--max_new_tokens", str(int(tok)),
                "--expand_num_beams", str(int(beam)),
                "--expand_num_return_sequences", str(int(beam)),
                "--extra_candidates_cost", str(int(ecc)),
                "--attn_impl", str(args.attn_impl),
            ]
            if str(args.model_base).strip() != "":
                cmd += ["--model_base", str(args.model_base)]
            if str(args.conv_mode).strip() != "":
                cmd += ["--conv_mode", str(args.conv_mode)]
            if bool(args.use_flash_attn):
                cmd.append("--use_flash_attn")
            if bool(args.enable_tqdm):
                cmd.append("--enable_tqdm")
            if bool(args.keep_pairwise_dirs):
                cmd.append("--keep_pairwise_dirs")

            print(f"[{k}/{total}] run {run_name}")
            subprocess.run(cmd, check=True)

            table_csv = os.path.join(run_out, "adaptive_two_stage_table.csv")
            summary_json = os.path.join(run_out, "summary.json")
            row = pick_gate_row(table_csv, gate_name=gate_name)
            out: Dict[str, Any] = {
                "run_name": run_name,
                "beam": int(beam),
                "max_new_tokens": int(tok),
                "extra_candidates_cost": int(ecc),
                "gate": row.get("gate"),
                "base_acc": safe_float(row.get("base_acc")),
                "final_acc": safe_float(row.get("final_acc")),
                "delta_acc": safe_float(row.get("delta_acc")),
                "expand_rate": safe_float(row.get("expand_rate")),
                "switch_rate": safe_float(row.get("switch_rate")),
                "gain": safe_float(row.get("gain")),
                "harm": safe_float(row.get("harm")),
                "avg_cost_rel": safe_float(row.get("avg_cost_rel")),
                "speedup_vs_fixed6": safe_float(row.get("speedup_vs_fixed6")),
                "out_dir": run_out,
                "table_csv": table_csv,
                "summary_json": summary_json,
            }
            rt_path = None
            if os.path.isfile(summary_json):
                sm = json.load(open(summary_json, encoding="utf-8"))
                gdir = str(sm.get("inputs", {}).get("greedy_dir", ""))
                if gdir != "":
                    rt_path = os.path.join(os.path.dirname(gdir), "onepass_runtime_stats.json")
            if rt_path is not None and os.path.isfile(rt_path):
                rt = json.load(open(rt_path, encoding="utf-8"))
                lat = rt.get("latency_ms", {})
                cnt = rt.get("counts", {})
                out["runtime_json"] = rt_path
                out["wall_total_ms"] = safe_float(lat.get("wall_total_ms"))
                out["greedy_wall_mean_ms"] = safe_float(lat.get("greedy_wall_mean_ms"))
                out["expand_wall_mean_ms"] = safe_float(lat.get("expand_wall_mean_ms"))
                out["n_expand_calls"] = safe_float(cnt.get("n_expand_calls"))
            grid_rows.append(out)
            run_meta.append(
                {
                    "run_name": run_name,
                    "cmd": cmd,
                    "out_dir": run_out,
                    "runtime_json": rt_path,
                }
            )

    grid_rows.sort(
        key=lambda x: (
            -(safe_float(x.get("final_acc")) or -1e9),
            (safe_float(x.get("wall_total_ms")) or 1e18),
        )
    )
    results_csv = os.path.join(os.path.abspath(args.out_root), "grid_results.csv")
    results_json = os.path.join(os.path.abspath(args.out_root), "grid_results.json")
    write_csv(results_csv, grid_rows)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "questions_json": os.path.abspath(args.questions_json),
                    "image_root": os.path.abspath(args.image_root),
                    "out_root": os.path.abspath(args.out_root),
                    "beam_values": [int(x) for x in beam_vals],
                    "max_new_tokens_values": [int(x) for x in tok_vals],
                    "num_samples": int(args.num_samples),
                    "seed": int(args.seed),
                    "gates": str(args.gates),
                    "policy": str(args.policy),
                    "trigger": str(args.trigger),
                    "extra_candidates_cost_mode": str(args.extra_candidates_cost_mode),
                    "extra_candidates_cost_fixed": int(args.extra_candidates_cost_fixed),
                },
                "runs": run_meta,
                "results_sorted": grid_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("[saved]", results_csv)
    print("[saved]", results_json)


if __name__ == "__main__":
    main()

