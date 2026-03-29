#!/usr/bin/env python3
import argparse
import csv
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser(description="Make id split csv with calib/eval assignment.")
    ap.add_argument("--subset_ids_csv", type=str, required=True, help="CSV with at least id column.")
    ap.add_argument("--calib_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    ids = []
    with open(args.subset_ids_csv, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id", "")).strip()
            if sid:
                ids.append(sid)

    rng = random.Random(int(args.seed))
    rng.shuffle(ids)
    k = int(round(len(ids) * float(args.calib_ratio)))
    calib = set(ids[:k])

    out_rows = []
    for sid in ids:
        out_rows.append({"id": sid, "split": ("calib" if sid in calib else "eval")})

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "split"])
        wr.writeheader()
        wr.writerows(out_rows)

    n_calib = sum(1 for r in out_rows if r["split"] == "calib")
    n_eval = len(out_rows) - n_calib
    print("[saved]", out_csv)
    print(f"[counts] n_total={len(out_rows)} n_calib={n_calib} n_eval={n_eval}")


if __name__ == "__main__":
    main()

