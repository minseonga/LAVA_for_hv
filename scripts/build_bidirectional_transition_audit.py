#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_transition_counts(path: Path):
    counts = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["gt"].strip().lower(), r["pair"].strip().lower())
            counts[key] = int(r["n"])
    return counts


def c(counts, gt, pair):
    return int(counts.get((gt, pair), 0))


def main():
    ap = argparse.ArgumentParser(
        description="Build No->Yes and bidirectional transition audit tables from transition_by_gt.csv."
    )
    ap.add_argument(
        "--root_dir",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/pope_full_9000/all_models_full_strict",
    )
    ap.add_argument("--models", type=str, default="vga,vista,eazy")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/pope_full_9000/all_models_full_strict/transition_audit",
    )
    args = ap.parse_args()

    root = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows_no2yes = []
    rows_bidirectional = []

    for m in models:
        p = root / m / "d1d2_audit" / "transition_by_gt.csv"
        if not p.exists():
            continue
        cnt = load_transition_counts(p)

        # Baseline -> model transitions conditioned by GT
        no_no2yes = c(cnt, "no", "no->yes")
        yes_no2yes = c(cnt, "yes", "no->yes")
        no_yes2no = c(cnt, "no", "yes->no")
        yes_yes2no = c(cnt, "yes", "yes->no")

        no2yes_changed = no_no2yes + yes_no2yes
        no2yes_beneficial = yes_no2yes
        no2yes_harmful = no_no2yes
        no2yes_beneficial_ratio = (
            float(no2yes_beneficial) / float(no2yes_changed) if no2yes_changed > 0 else 0.0
        )
        no2yes_harmful_ratio = (
            float(no2yes_harmful) / float(no2yes_changed) if no2yes_changed > 0 else 0.0
        )

        rows_no2yes.append(
            {
                "model": m,
                "no_to_yes_changed": no2yes_changed,
                "beneficial": no2yes_beneficial,
                "harmful": no2yes_harmful,
                "beneficial_ratio": no2yes_beneficial_ratio,
                "harmful_ratio": no2yes_harmful_ratio,
            }
        )

        yes2no_beneficial = no_yes2no
        yes2no_harmful = yes_yes2no

        net = (yes2no_beneficial + no2yes_beneficial) - (yes2no_harmful + no2yes_harmful)

        rows_bidirectional.append(
            {
                "model": m,
                "yes_to_no_beneficial": yes2no_beneficial,
                "yes_to_no_harmful": yes2no_harmful,
                "no_to_yes_beneficial": no2yes_beneficial,
                "no_to_yes_harmful": no2yes_harmful,
                "net": net,
            }
        )

    no2yes_csv = out_dir / "no2yes_transition_audit.csv"
    with no2yes_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "no_to_yes_changed",
                "beneficial",
                "harmful",
                "beneficial_ratio",
                "harmful_ratio",
            ],
        )
        w.writeheader()
        for r in rows_no2yes:
            w.writerow(r)

    bidir_csv = out_dir / "bidirectional_transition_table.csv"
    with bidir_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "yes_to_no_beneficial",
                "yes_to_no_harmful",
                "no_to_yes_beneficial",
                "no_to_yes_harmful",
                "net",
            ],
        )
        w.writeheader()
        for r in rows_bidirectional:
            w.writerow(r)

    summary = {
        "root_dir": str(root.resolve()),
        "models": models,
        "outputs": {
            "no2yes_transition_audit_csv": str(no2yes_csv.resolve()),
            "bidirectional_transition_table_csv": str(bidir_csv.resolve()),
        },
    }
    sjson = out_dir / "summary.json"
    sjson.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[saved]", no2yes_csv)
    print("[saved]", bidir_csv)
    print("[saved]", sjson)


if __name__ == "__main__":
    main()
