#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


YES_SET = {"yes", "y", "1", "true"}
NO_SET = {"no", "n", "0", "false"}


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def read_csv_map(path: str, key_col: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            key = str(row.get(key_col, "")).strip()
            if key != "":
                out[key] = row
    return out


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx <= 0.0 or deny <= 0.0:
        return float("nan")
    return num / (denx * deny)


def quantiles(xs: List[float], ps: Iterable[float]) -> Dict[str, float]:
    arr = sorted(xs)
    out: Dict[str, float] = {}
    if not arr:
        return out
    for p in ps:
        idx = max(0, min(len(arr) - 1, int((len(arr) - 1) * p)))
        out[f"p{int(round(p * 100)):02d}"] = float(arr[idx])
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare runtime probe outputs against offline controller features.")
    ap.add_argument("--runtime_route_csv", type=str, required=True)
    ap.add_argument("--offline_controller_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--runtime_id_col", type=str, default="id")
    ap.add_argument("--offline_id_col", type=str, default="id")
    ap.add_argument("--runtime_frg_col", type=str, default="frg")
    ap.add_argument("--offline_frg_col", type=str, default="__C__")
    ap.add_argument("--offline_case_col", type=str, default="case_type")
    ap.add_argument("--offline_baseline_col", type=str, default="pred_baseline")
    ap.add_argument("--runtime_branch_text_col", type=str, default="baseline_preview_text")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runtime_rows = read_csv_map(args.runtime_route_csv, key_col=args.runtime_id_col)
    offline_rows = read_csv_map(args.offline_controller_csv, key_col=args.offline_id_col)
    ids = sorted(set(runtime_rows) & set(offline_rows), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if not ids:
        raise RuntimeError("No overlapping ids between runtime and offline tables.")

    xs: List[float] = []
    ys: List[float] = []
    abs_delta_rows: List[Dict[str, Any]] = []
    runtime_only_rows: List[Dict[str, Any]] = []
    offline_only_rows: List[Dict[str, Any]] = []
    case_groups: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    baseline_match = 0
    baseline_total = 0

    tau = float(args.tau) if args.tau is not None else None

    for sid in ids:
        runtime_row = runtime_rows[sid]
        offline_row = offline_rows[sid]
        runtime_frg = float(runtime_row[args.runtime_frg_col])
        offline_frg = float(offline_row[args.offline_frg_col])
        case_name = str(offline_row.get(args.offline_case_col, ""))
        xs.append(runtime_frg)
        ys.append(offline_frg)
        case_groups[case_name].append((runtime_frg, offline_frg))
        abs_delta_rows.append(
            {
                "id": sid,
                "case_type": case_name,
                "runtime_frg": runtime_frg,
                "offline_frg": offline_frg,
                "delta_runtime_minus_offline": runtime_frg - offline_frg,
                "runtime_route": runtime_row.get("route", ""),
                "probe_source": runtime_row.get("probe_source", ""),
                "probe_branch_source": runtime_row.get("probe_branch_source", ""),
                "probe_impl": runtime_row.get("probe_impl", ""),
            }
        )
        branch_text = str(runtime_row.get(args.runtime_branch_text_col, "")).strip()
        baseline_label = str(offline_row.get(args.offline_baseline_col, "")).strip().lower()
        if branch_text != "" and baseline_label in YES_SET | NO_SET:
            baseline_total += 1
            if parse_yes_no(branch_text) == baseline_label:
                baseline_match += 1
        if tau is not None:
            runtime_ge = runtime_frg >= tau
            offline_ge = offline_frg >= tau
            if runtime_ge and not offline_ge:
                runtime_only_rows.append(abs_delta_rows[-1])
            if offline_ge and not runtime_ge:
                offline_only_rows.append(abs_delta_rows[-1])

    abs_delta_rows.sort(key=lambda row: abs(float(row["delta_runtime_minus_offline"])), reverse=True)
    runtime_only_rows.sort(key=lambda row: float(row["delta_runtime_minus_offline"]), reverse=True)
    offline_only_rows.sort(key=lambda row: float(row["delta_runtime_minus_offline"]))

    summary: Dict[str, Any] = {
        "inputs": {
            "runtime_route_csv": os.path.abspath(args.runtime_route_csv),
            "offline_controller_csv": os.path.abspath(args.offline_controller_csv),
            "tau": tau,
            "runtime_frg_col": args.runtime_frg_col,
            "offline_frg_col": args.offline_frg_col,
        },
        "counts": {
            "runtime_rows": len(runtime_rows),
            "offline_rows": len(offline_rows),
            "overlap_ids": len(ids),
        },
        "frg_compare": {
            "pearson": pearson(xs, ys),
            "runtime_mean": sum(xs) / len(xs),
            "offline_mean": sum(ys) / len(ys),
            "delta_runtime_minus_offline_mean": (sum(xs) - sum(ys)) / len(xs),
            "runtime_quantiles": quantiles(xs, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
            "offline_quantiles": quantiles(ys, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
            "sign_flip_count": int(sum((x >= 0.0) != (y >= 0.0) for x, y in zip(xs, ys))),
            "runtime_pos_offline_neg": int(sum((x >= 0.0) and (y < 0.0) for x, y in zip(xs, ys))),
            "runtime_neg_offline_pos": int(sum((x < 0.0) and (y >= 0.0) for x, y in zip(xs, ys))),
        },
        "baseline_branch": {
            "match_count": int(baseline_match),
            "compare_count": int(baseline_total),
            "match_rate": (baseline_match / baseline_total) if baseline_total else float("nan"),
        },
        "case_breakdown": {},
    }
    if tau is not None:
        summary["tau_compare"] = {
            "runtime_ge_tau": int(sum(x >= tau for x in xs)),
            "offline_ge_tau": int(sum(y >= tau for y in ys)),
            "both_ge_tau": int(sum((x >= tau) and (y >= tau) for x, y in zip(xs, ys))),
            "runtime_only_ge_tau": int(len(runtime_only_rows)),
            "offline_only_ge_tau": int(len(offline_only_rows)),
        }
    for case_name, pairs in case_groups.items():
        xr = [x for x, _ in pairs]
        yr = [y for _, y in pairs]
        item = {
            "n": len(pairs),
            "runtime_mean": sum(xr) / len(xr),
            "offline_mean": sum(yr) / len(yr),
            "sign_flip_count": int(sum((x >= 0.0) != (y >= 0.0) for x, y in pairs)),
        }
        if tau is not None:
            item["runtime_ge_tau"] = int(sum(x >= tau for x in xr))
            item["offline_ge_tau"] = int(sum(y >= tau for y in yr))
        summary["case_breakdown"][case_name] = item

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(os.path.join(args.out_dir, "top_abs_delta.csv"), abs_delta_rows[:200])
    if tau is not None:
        write_csv(os.path.join(args.out_dir, "tau_disagree_runtime_only.csv"), runtime_only_rows[:200])
        write_csv(os.path.join(args.out_dir, "tau_disagree_offline_only.csv"), offline_only_rows[:200])
    print("[saved]", summary_path)
    print("[saved]", os.path.join(args.out_dir, "top_abs_delta.csv"))
    if tau is not None:
        print("[saved]", os.path.join(args.out_dir, "tau_disagree_runtime_only.csv"))
        print("[saved]", os.path.join(args.out_dir, "tau_disagree_offline_only.csv"))
    print(
        "[summary]",
        json.dumps(
            {
                "pearson": summary["frg_compare"]["pearson"],
                "runtime_mean": summary["frg_compare"]["runtime_mean"],
                "offline_mean": summary["frg_compare"]["offline_mean"],
                "baseline_match_rate": summary["baseline_branch"]["match_rate"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
