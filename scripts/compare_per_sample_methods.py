#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable


def parse_bool(x: Any) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def safe_float(x: Any):
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
            wr.writerow({k: r.get(k) for k in keys})


def load_map(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid == "":
                continue
            if str(r.get("error") or "").strip() != "":
                continue
            out[sid] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline vs candidate per-sample outputs.")
    ap.add_argument("--baseline_csv", type=str, required=True)
    ap.add_argument("--candidate_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    base = load_map(os.path.abspath(args.baseline_csv))
    cand = load_map(os.path.abspath(args.candidate_csv))
    ids = sorted(set(base.keys()) & set(cand.keys()))
    if len(ids) == 0:
        raise RuntimeError("No overlapping ids between baseline and candidate CSVs.")

    gain = 0
    harm = 0
    same = 0
    changed_pred = 0
    rows: List[Dict[str, Any]] = []
    pbar = tqdm(ids, total=len(ids), desc="compare-rows", dynamic_ncols=True)
    for sid in pbar:
        b = base[sid]
        c = cand[sid]
        b_ok = parse_bool(b.get("is_success"))
        c_ok = parse_bool(c.get("is_success"))
        b_pred = str(b.get("pred_answer_eval") or "")
        c_pred = str(c.get("pred_answer_eval") or "")
        changed = bool(b_pred != c_pred)
        if changed:
            changed_pred += 1

        outcome = "same"
        if b_ok != c_ok:
            if (not b_ok) and c_ok:
                gain += 1
                outcome = "gain"
            elif b_ok and (not c_ok):
                harm += 1
                outcome = "harm"
        else:
            same += 1

        rows.append(
            {
                "id": sid,
                "baseline_ok": bool(b_ok),
                "candidate_ok": bool(c_ok),
                "changed_prediction": bool(changed),
                "baseline_pred": b_pred,
                "candidate_pred": c_pred,
                "outcome": outcome,
                "baseline_tokens": safe_float(b.get("n_gen_tokens")),
                "candidate_tokens": safe_float(c.get("n_gen_tokens")),
                "baseline_latency_sec": safe_float(b.get("elapsed_sec_sample")),
                "candidate_latency_sec": safe_float(c.get("elapsed_sec_sample")),
            }
        )
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(gain=int(gain), harm=int(harm), net=int(gain - harm))

    n = len(ids)
    base_acc = float(sum(1 for sid in ids if parse_bool(base[sid].get("is_success"))) / max(1, n))
    cand_acc = float(sum(1 for sid in ids if parse_bool(cand[sid].get("is_success"))) / max(1, n))
    base_lat = [safe_float(base[sid].get("elapsed_sec_sample")) for sid in ids]
    cand_lat = [safe_float(cand[sid].get("elapsed_sec_sample")) for sid in ids]
    base_lat = [float(x) for x in base_lat if x is not None]
    cand_lat = [float(x) for x in cand_lat if x is not None]
    mean_base_lat = (None if len(base_lat) == 0 else float(sum(base_lat) / len(base_lat)))
    mean_cand_lat = (None if len(cand_lat) == 0 else float(sum(cand_lat) / len(cand_lat)))

    summary = {
        "inputs": {
            "baseline_csv": os.path.abspath(args.baseline_csv),
            "candidate_csv": os.path.abspath(args.candidate_csv),
            "n_common_ids": int(n),
        },
        "metrics": {
            "base_acc": base_acc,
            "candidate_acc": cand_acc,
            "delta_acc": float(cand_acc - base_acc),
            "gain": int(gain),
            "harm": int(harm),
            "net": int(gain - harm),
            "same": int(same),
            "changed_prediction_count": int(changed_pred),
            "changed_prediction_rate": float(changed_pred / max(1, n)),
            "switch_precision": (None if changed_pred == 0 else float(gain / changed_pred)),
            "mean_latency_sec_baseline": mean_base_lat,
            "mean_latency_sec_candidate": mean_cand_lat,
            "latency_ratio_candidate_over_baseline": (
                None
                if mean_base_lat is None or mean_cand_lat is None or mean_base_lat <= 0.0
                else float(mean_cand_lat / mean_base_lat)
            ),
        },
        "outputs": {
            "comparison_csv": os.path.join(out_dir, "comparison_rows.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "comparison_rows.csv"), rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "comparison_rows.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
