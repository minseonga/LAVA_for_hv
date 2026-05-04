#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


ORDER = [
    "NOOP",
    "Always rollback",
    "Random fallback",
    "C-only both directions",
    "D-only both directions",
    "Fusion-only both directions",
    "RAPIC selected policy",
]


def maybe_float(value: Any) -> float | None:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def fmt_pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return ""
    pct = 100.0 * float(value)
    return f"{pct:+.2f}" if signed else f"{pct:.2f}"


def fmt_num(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Average family-ablation CSVs across datasets.")
    ap.add_argument("--ablation_dir", action="append", required=True, help="Directory containing family_ablation.csv.")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    rows: List[Dict[str, str]] = []
    for item in args.ablation_dir:
        path = Path(item).resolve() / "family_ablation.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_rows(path))

    by_variant: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant", "")), []).append(row)

    out_rows: List[Dict[str, Any]] = []
    for variant in ORDER:
        group = by_variant.get(variant, [])
        if not group:
            continue
        numeric_keys = [
            "acc",
            "delta_vs_intervention",
            "f1",
            "delta_f1_vs_intervention",
            "fallback",
            "selected_harm",
            "selected_help",
            "net",
            "hrec",
            "grec",
        ]
        agg: Dict[str, Any] = {
            "variant": variant,
            "policy": " / ".join(sorted({str(row.get("policy", "")) for row in group if str(row.get("policy", "")).strip()})),
            "n_datasets": len(group),
            "datasets": ",".join(str(row.get("dataset", "")) for row in group),
        }
        for key in numeric_keys:
            vals = [maybe_float(row.get(key)) for row in group]
            vals_f = [v for v in vals if v is not None]
            agg[key] = mean(vals_f) if vals_f else None
        out_rows.append(agg)

    md_lines = [
        "| Variant | Policy | Avg Acc | Avg dMethod | Avg F1 | Avg dMethod F1 | Avg Fallback | Avg H/G/Net | Avg Hrec | Avg Grec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in out_rows:
        h = row.get("selected_harm")
        g = row.get("selected_help")
        net = row.get("net")
        md_lines.append(
            f"| {row['variant']} | {row['policy']} | "
            f"{fmt_pct(row.get('acc'))} | {fmt_pct(row.get('delta_vs_intervention'), signed=True)} | "
            f"{fmt_pct(row.get('f1'))} | {fmt_pct(row.get('delta_f1_vs_intervention'), signed=True)} | "
            f"{fmt_num(row.get('fallback'))} | {fmt_num(h)}/{fmt_num(g)}/{fmt_num(net)} | "
            f"{fmt_pct(row.get('hrec'))} | {fmt_pct(row.get('grec'))} |"
        )

    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    out_csv = Path(args.out_csv).resolve() if args.out_csv else out_md.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else ["variant"]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in out_rows:
            wr.writerow(row)

    print("\n".join(md_lines))
    print("[saved]", out_md)
    print("[saved]", out_csv)


if __name__ == "__main__":
    main()
