#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List


def safe_id(value: Any) -> str:
    return str("" if value is None else value).strip()


def is_baseline(row: Dict[str, Any] | None) -> bool:
    return str((row or {}).get("route", "")).strip().lower() == "baseline"


def read_rows(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            qid = safe_id(row.get("id") or row.get("question_id"))
            if qid:
                rows[qid] = dict(row)
    return rows


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    preferred = [
        "id",
        "image",
        "question",
        "route",
        "merge_mode",
        "merge_source",
        "primary_route",
        "secondary_route",
        "primary_baseline",
        "secondary_baseline",
        "harm",
        "help",
        "baseline_correct",
        "intervention_correct",
        "final_source",
        "final_text",
    ]
    for key in preferred:
        seen.add(key)
        fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def choose_route(mode: str, primary_base: bool, secondary_base: bool) -> tuple[str, str]:
    if mode in {"union", "primary_plus_secondary"}:
        if primary_base and secondary_base:
            return "baseline", "both"
        if primary_base:
            return "baseline", "primary"
        if secondary_base:
            return "baseline", "secondary"
        return "method", "method"
    if mode == "intersection":
        return ("baseline", "both") if primary_base and secondary_base else ("method", "method")
    if mode == "primary":
        return ("baseline", "primary") if primary_base else ("method", "method")
    if mode == "secondary":
        return ("baseline", "secondary") if secondary_base else ("method", "method")
    if mode == "secondary_minus_primary":
        return ("baseline", "secondary") if secondary_base and not primary_base else ("method", "method")
    raise ValueError(f"Unsupported mode={mode!r}")


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = str("" if value is None else value)
        if text.strip():
            return text
    return ""


def merged_text(route: str, source: str, p: Dict[str, Any] | None, s: Dict[str, Any] | None) -> tuple[str, str]:
    if route == "baseline":
        row = p if source == "primary" else s if source == "secondary" else p or s
        return (
            first_nonempty((row or {}).get("final_source"), "baseline_cached"),
            first_nonempty((row or {}).get("final_text")),
        )
    row = p or s
    return (
        first_nonempty((row or {}).get("final_source"), "method"),
        first_nonempty((row or {}).get("final_text")),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge two PCP route CSVs at decision level.")
    ap.add_argument("--primary_route_rows_csv", required=True)
    ap.add_argument("--secondary_route_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--mode",
        default="primary_plus_secondary",
        choices=["union", "primary_plus_secondary", "intersection", "primary", "secondary", "secondary_minus_primary"],
    )
    ap.add_argument("--primary_name", default="primary")
    ap.add_argument("--secondary_name", default="secondary")
    args = ap.parse_args()

    primary = read_rows(args.primary_route_rows_csv)
    secondary = read_rows(args.secondary_route_rows_csv)
    ids = sorted(set(primary) | set(secondary), key=lambda x: (len(x), x))

    rows: List[Dict[str, Any]] = []
    counts = {
        "n_ids": len(ids),
        "primary_rows": len(primary),
        "secondary_rows": len(secondary),
        "primary_baseline": 0,
        "secondary_baseline": 0,
        "merged_baseline": 0,
        "both_baseline": 0,
        "primary_only_baseline": 0,
        "secondary_only_baseline": 0,
    }

    for qid in ids:
        p = primary.get(qid)
        s = secondary.get(qid)
        p_base = is_baseline(p)
        s_base = is_baseline(s)
        counts["primary_baseline"] += int(p_base)
        counts["secondary_baseline"] += int(s_base)
        counts["both_baseline"] += int(p_base and s_base)
        counts["primary_only_baseline"] += int(p_base and not s_base)
        counts["secondary_only_baseline"] += int(s_base and not p_base)

        route, source = choose_route(str(args.mode), p_base, s_base)
        counts["merged_baseline"] += int(route == "baseline")
        final_source, final_text = merged_text(route, source, p, s)
        base_row = dict(p or s or {})
        base_row.update(
            {
                "id": qid,
                "route": route,
                "merge_mode": str(args.mode),
                "merge_source": source,
                "primary_route": str((p or {}).get("route", "method")).strip().lower() or "method",
                "secondary_route": str((s or {}).get("route", "method")).strip().lower() or "method",
                "primary_baseline": int(p_base),
                "secondary_baseline": int(s_base),
                "primary_name": str(args.primary_name),
                "secondary_name": str(args.secondary_name),
                "final_source": final_source,
                "final_text": final_text,
            }
        )
        rows.append(base_row)

    out_dir = os.path.abspath(args.out_dir)
    route_path = os.path.join(out_dir, "pcp_route_rows.csv")
    summary_path = os.path.join(out_dir, "merge_summary.json")
    write_csv(route_path, rows)
    write_json(
        summary_path,
        {
            "mode": str(args.mode),
            "inputs": {
                "primary_route_rows_csv": os.path.abspath(args.primary_route_rows_csv),
                "secondary_route_rows_csv": os.path.abspath(args.secondary_route_rows_csv),
                "primary_name": str(args.primary_name),
                "secondary_name": str(args.secondary_name),
            },
            "counts": counts,
            "outputs": {
                "pcp_route_rows_csv": route_path,
                "summary_json": summary_path,
            },
        },
    )
    print(json.dumps({"counts": counts, "outputs": {"pcp_route_rows_csv": route_path}}, ensure_ascii=False, indent=2))
    print("[saved]", route_path)
    print("[saved]", summary_path)


if __name__ == "__main__":
    main()
