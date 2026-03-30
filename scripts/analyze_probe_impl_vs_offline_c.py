#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_csv_by_id(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {str(row["id"]): row for row in csv.DictReader(f)}


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or not x:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sum((a - mx) ** 2 for a in x)
    den_y = sum((b - my) ** 2 for b in y)
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return float("nan")
    return num / den


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare offline C and probe FRG, optionally filtered by probe_impl.")
    ap.add_argument("--probe_csv", type=str, required=True)
    ap.add_argument("--offline_features_csv", type=str, required=True)
    ap.add_argument("--taxonomy_csv", type=str, default="")
    ap.add_argument("--probe_impl", type=str, default="offline_style_helper")
    ap.add_argument("--tau_c", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    probe_rows = load_csv_by_id(args.probe_csv)
    offline_rows = load_csv_by_id(args.offline_features_csv)
    tax_rows = load_csv_by_id(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else {}

    common_ids = sorted(set(probe_rows) & set(offline_rows), key=lambda x: int(x))
    if not common_ids:
        raise SystemExit("no overlapping ids between probe and offline feature tables")

    probe_impl_counts = Counter(str(probe_rows[i].get("probe_impl", "") or "") for i in common_ids)
    selected_ids = [
        i for i in common_ids
        if str(probe_rows[i].get("probe_impl", "") or "") == str(args.probe_impl)
    ]

    rows: List[Tuple[str, float, float, str, str]] = []
    for i in selected_ids:
        off_c = float(offline_rows[i]["faithful_minus_global_attn"])
        probe_c = float(probe_rows[i]["frg"])
        case_type = str(tax_rows.get(i, {}).get("case_type", ""))
        question = str(offline_rows[i].get("question", ""))
        rows.append((i, off_c, probe_c, case_type, question))

    summary = {
        "inputs": {
            "probe_csv": os.path.abspath(args.probe_csv),
            "offline_features_csv": os.path.abspath(args.offline_features_csv),
            "taxonomy_csv": os.path.abspath(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else "",
            "probe_impl": args.probe_impl,
            "tau_c": args.tau_c,
            "top_k": int(args.top_k),
        },
        "counts": {
            "n_common": int(len(common_ids)),
            "probe_impl_counts": dict(probe_impl_counts),
            "n_selected": int(len(rows)),
        },
    }

    if rows:
        x = [r[1] for r in rows]
        y = [r[2] for r in rows]
        by_case = defaultdict(lambda: {"offline": [], "probe": []})
        for _, off_c, probe_c, case_type, _ in rows:
            by_case[case_type]["offline"].append(off_c)
            by_case[case_type]["probe"].append(probe_c)
        case_means = {}
        for case_type, vals in sorted(by_case.items()):
            case_means[case_type] = {
                "n": int(len(vals["offline"])),
                "offline_mean": float(sum(vals["offline"]) / len(vals["offline"])),
                "probe_mean": float(sum(vals["probe"]) / len(vals["probe"])),
            }

        mismatches = sorted(
            [
                {
                    "id": i,
                    "abs_diff": float(abs(off_c - probe_c)),
                    "offline_c": float(off_c),
                    "probe_c": float(probe_c),
                    "case_type": case_type,
                    "question": question,
                }
                for i, off_c, probe_c, case_type, question in rows
            ],
            key=lambda r: r["abs_diff"],
            reverse=True,
        )

        summary["stats"] = {
            "pearson": float(pearson(x, y)),
            "offline_mean": float(sum(x) / len(x)),
            "probe_mean": float(sum(y) / len(y)),
            "case_means": case_means,
            "top_abs_diff": mismatches[: int(args.top_k)],
        }
        if args.tau_c is not None:
            agree = sum(((off_c >= args.tau_c) == (probe_c >= args.tau_c)) for _, off_c, probe_c, _, _ in rows)
            disagree = []
            for i, off_c, probe_c, case_type, question in rows:
                if (off_c >= args.tau_c) != (probe_c >= args.tau_c):
                    disagree.append(
                        {
                            "id": i,
                            "offline_c": float(off_c),
                            "probe_c": float(probe_c),
                            "case_type": case_type,
                            "question": question,
                        }
                    )
            summary["stats"]["threshold_agreement"] = {
                "tau_c": float(args.tau_c),
                "agree_count": int(agree),
                "agree_rate": float(agree / len(rows)),
                "n_disagree": int(len(disagree)),
                "top_disagree": sorted(
                    disagree,
                    key=lambda r: abs(r["offline_c"] - r["probe_c"]),
                    reverse=True,
                )[: int(args.top_k)],
            }

    if str(args.out_json).strip():
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[saved]", os.path.abspath(args.out_json))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
