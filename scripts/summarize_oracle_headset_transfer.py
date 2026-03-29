#!/usr/bin/env python
import argparse
import csv
import os
from typing import Dict, List


def _f(x, d=0.0):
    try:
        if x is None or x == "":
            return float(d)
        return float(x)
    except Exception:
        return float(d)


def _i(x, d=0):
    try:
        if x is None or x == "":
            return int(d)
        return int(float(x))
    except Exception:
        return int(d)


def _pick_best(rows: List[Dict[str, str]]) -> Dict[str, str]:
    if not rows:
        return {}
    # Prefer higher delta_acc, then higher net_gain, then higher delta_f1.
    return sorted(
        rows,
        key=lambda r: (
            _f(r.get("delta_acc_vs_base"), 0.0),
            _i(r.get("net_gain"), 0),
            _f(r.get("delta_f1_vs_base"), 0.0),
        ),
        reverse=True,
    )[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    with open(args.ablation_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Empty ablation csv.")

    base = None
    runs = []
    for r in rows:
        if str(r.get("name", "")) == "baseline":
            base = r
        else:
            runs.append(r)
    if base is None:
        raise RuntimeError("Missing baseline row.")

    print("[base]", {
        "acc": _f(base.get("acc")),
        "f1": _f(base.get("f1")),
        "TP": _i(base.get("TP")),
        "FP": _i(base.get("FP")),
        "TN": _i(base.get("TN")),
        "FN": _i(base.get("FN")),
    })

    by_control: Dict[str, List[Dict[str, str]]] = {}
    for r in runs:
        c = str(r.get("control_type", "unknown"))
        by_control.setdefault(c, []).append(r)

    summary_rows: List[Dict[str, object]] = []
    print("[best_by_control]")
    for c in sorted(by_control.keys()):
        b = _pick_best(by_control[c])
        out = {
            "control_type": c,
            "name": str(b.get("name", "")),
            "arm": str(b.get("arm", "")),
            "lambda_pos": _f(b.get("lambda_pos"), 0.0),
            "lambda_neg": _f(b.get("lambda_neg"), 0.0),
            "acc": _f(b.get("acc"), 0.0),
            "f1": _f(b.get("f1"), 0.0),
            "delta_acc_vs_base": _f(b.get("delta_acc_vs_base"), 0.0),
            "delta_f1_vs_base": _f(b.get("delta_f1_vs_base"), 0.0),
            "changed_pred": _i(b.get("changed_pred"), 0),
            "gain": _i(b.get("gain"), 0),
            "harm": _i(b.get("harm"), 0),
            "net_gain": _i(b.get("net_gain"), 0),
            "TP": _i(b.get("TP"), 0),
            "FP": _i(b.get("FP"), 0),
            "TN": _i(b.get("TN"), 0),
            "FN": _i(b.get("FN"), 0),
            "trigger_frac_batch_mean": _f(b.get("trigger_frac_batch_mean"), 0.0),
            "late_trigger_fraction_step_mean": _f(b.get("late_trigger_fraction_step_mean"), 0.0),
            "penalty_img_mean": _f(b.get("penalty_img_mean"), 0.0),
            "harmful_per_cell_dose_mean": _f(b.get("harmful_per_cell_dose_mean"), 0.0),
        }
        summary_rows.append(out)
        print(" ", out)

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        keys = [
            "control_type",
            "name",
            "arm",
            "lambda_pos",
            "lambda_neg",
            "acc",
            "f1",
            "delta_acc_vs_base",
            "delta_f1_vs_base",
            "changed_pred",
            "gain",
            "harm",
            "net_gain",
            "TP",
            "FP",
            "TN",
            "FN",
            "trigger_frac_batch_mean",
            "late_trigger_fraction_step_mean",
            "penalty_img_mean",
            "harmful_per_cell_dose_mean",
        ]
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in summary_rows:
                wr.writerow(r)
        print("[saved]", os.path.abspath(args.out_csv))


if __name__ == "__main__":
    main()
