#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List


def load_csv_by_id(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {str(row["id"]): row for row in csv.DictReader(f)}


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, None) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit offline-vs-probe C threshold disagreements, especially D2 regressions.")
    ap.add_argument("--probe_csv", type=str, required=True)
    ap.add_argument("--offline_features_csv", type=str, required=True)
    ap.add_argument("--taxonomy_csv", type=str, required=True)
    ap.add_argument("--tau_c", type=float, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--probe_impl", type=str, default="")
    args = ap.parse_args()

    probe_rows = load_csv_by_id(args.probe_csv)
    offline_rows = load_csv_by_id(args.offline_features_csv)
    tax_rows = load_csv_by_id(args.taxonomy_csv)

    common_ids = sorted(set(probe_rows) & set(offline_rows) & set(tax_rows), key=lambda x: int(x))
    if str(args.probe_impl).strip():
        common_ids = [
            i for i in common_ids
            if str(probe_rows[i].get("probe_impl", "") or "") == str(args.probe_impl)
        ]

    os.makedirs(args.out_dir, exist_ok=True)

    disagreement_rows: List[Dict[str, object]] = []
    d2_rows: List[Dict[str, object]] = []
    d1_rows: List[Dict[str, object]] = []
    per_case = defaultdict(lambda: {"n": 0, "offline_veto": 0, "probe_veto": 0, "disagree": 0})
    probe_impl_counts = Counter(str(probe_rows[i].get("probe_impl", "") or "") for i in common_ids)

    for i in common_ids:
        probe = probe_rows[i]
        off = offline_rows[i]
        tax = tax_rows[i]
        off_c = float(off["faithful_minus_global_attn"])
        probe_c = float(probe["frg"])
        case_type = str(tax["case_type"])
        off_veto = bool(off_c >= args.tau_c)
        probe_veto = bool(probe_c >= args.tau_c)

        per_case[case_type]["n"] += 1
        per_case[case_type]["offline_veto"] += int(off_veto)
        per_case[case_type]["probe_veto"] += int(probe_veto)
        per_case[case_type]["disagree"] += int(off_veto != probe_veto)

        row = {
            "id": i,
            "case_type": case_type,
            "question": off.get("question", ""),
            "offline_c": off_c,
            "probe_c": probe_c,
            "abs_diff": abs(off_c - probe_c),
            "offline_veto": int(off_veto),
            "probe_veto": int(probe_veto),
            "probe_impl": probe.get("probe_impl", ""),
            "probe_impl_error": probe.get("probe_impl_error", ""),
            "probe_source": probe.get("probe_source", ""),
            "probe_anchor": probe.get("probe_anchor", ""),
            "probe_anchor_token_idx": probe.get("probe_anchor_token_idx", ""),
            "probe_decision_pos": probe.get("probe_decision_pos", ""),
            "baseline_preview_fallback": probe.get("baseline_preview_fallback", ""),
        }

        if off_veto != probe_veto:
            disagreement_rows.append(row)
        if case_type == "vga_regression":
            d2_rows.append(row)
        if case_type == "vga_improvement":
            d1_rows.append(row)

    disagreement_rows.sort(key=lambda r: float(r["abs_diff"]), reverse=True)
    d2_rows.sort(key=lambda r: float(r["abs_diff"]), reverse=True)
    d1_rows.sort(key=lambda r: float(r["abs_diff"]), reverse=True)

    d2_missed = [r for r in d2_rows if int(r["offline_veto"]) == 1 and int(r["probe_veto"]) == 0]
    d2_extra = [r for r in d2_rows if int(r["offline_veto"]) == 0 and int(r["probe_veto"]) == 1]
    d1_harm = [r for r in d1_rows if int(r["offline_veto"]) == 0 and int(r["probe_veto"]) == 1]
    d1_spared = [r for r in d1_rows if int(r["offline_veto"]) == 1 and int(r["probe_veto"]) == 0]

    summary = {
        "inputs": {
            "probe_csv": os.path.abspath(args.probe_csv),
            "offline_features_csv": os.path.abspath(args.offline_features_csv),
            "taxonomy_csv": os.path.abspath(args.taxonomy_csv),
            "tau_c": float(args.tau_c),
            "probe_impl": str(args.probe_impl),
        },
        "counts": {
            "n_common": int(len(common_ids)),
            "probe_impl_counts": dict(probe_impl_counts),
            "n_disagreements": int(len(disagreement_rows)),
            "n_d2_total": int(len(d2_rows)),
            "n_d2_missed_by_probe": int(len(d2_missed)),
            "n_d2_extra_veto_by_probe": int(len(d2_extra)),
            "n_d1_total": int(len(d1_rows)),
            "n_d1_harm_by_probe": int(len(d1_harm)),
            "n_d1_spared_by_probe": int(len(d1_spared)),
        },
        "per_case": {},
        "outputs": {
            "threshold_disagreements_csv": os.path.join(args.out_dir, "threshold_disagreements.csv"),
            "d2_cases_csv": os.path.join(args.out_dir, "d2_cases.csv"),
            "d2_missed_by_probe_csv": os.path.join(args.out_dir, "d2_missed_by_probe.csv"),
            "d1_harm_by_probe_csv": os.path.join(args.out_dir, "d1_harm_by_probe.csv"),
        },
        "examples": {
            "top_d2_missed_by_probe": d2_missed[:10],
            "top_d1_harm_by_probe": d1_harm[:10],
        },
    }

    for case_type, stats in sorted(per_case.items()):
        n = int(stats["n"])
        summary["per_case"][case_type] = {
            "n": n,
            "offline_veto": int(stats["offline_veto"]),
            "probe_veto": int(stats["probe_veto"]),
            "disagree": int(stats["disagree"]),
            "offline_veto_rate": float(stats["offline_veto"] / n) if n else 0.0,
            "probe_veto_rate": float(stats["probe_veto"] / n) if n else 0.0,
            "disagree_rate": float(stats["disagree"] / n) if n else 0.0,
        }

    write_csv(summary["outputs"]["threshold_disagreements_csv"], disagreement_rows)
    write_csv(summary["outputs"]["d2_cases_csv"], d2_rows)
    write_csv(summary["outputs"]["d2_missed_by_probe_csv"], d2_missed)
    write_csv(summary["outputs"]["d1_harm_by_probe_csv"], d1_harm)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", summary_path)
    for key, value in summary["outputs"].items():
        print("[saved]", value)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
