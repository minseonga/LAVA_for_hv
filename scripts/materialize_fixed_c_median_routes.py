#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import build_pcp_c_d_controller as pcp
import build_transition_split_fixed_c_median_ensemble as fixed


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def safe_id(row: Dict[str, Any]) -> str:
    for key in ("id", "question_id", "qid"):
        value = str(row.get(key, "")).strip()
        if value and value.lower() not in {"none", "null", "nan"}:
            return value
    return ""


def route_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    yes_policy: Dict[str, Any],
    no_policy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        qid = safe_id(row)
        if not qid:
            continue
        y_route, y_score = fixed.compute_route(row, yes_policy, "yes_to_no")
        n_route, n_score = fixed.compute_route(row, no_policy, "no_to_yes")
        route = "baseline" if y_route == "baseline" or n_route == "baseline" else "method"
        direction = ""
        score = ""
        if y_route == "baseline":
            direction = "yes_to_no"
            score = y_score
        elif n_route == "baseline":
            direction = "no_to_yes"
            score = n_score
        out.append(
            {
                "id": qid,
                "question_id": qid,
                "route": route,
                "selected_direction": direction,
                "selected_score": "" if score is None else float(score),
                "yes_to_no_route": y_route,
                "yes_to_no_score": "" if y_score is None else float(y_score),
                "no_to_yes_route": n_route,
                "no_to_yes_score": "" if n_score is None else float(n_score),
                "baseline_correct": row.get("baseline_correct", ""),
                "intervention_correct": row.get("intervention_correct", ""),
                "harm": row.get("harm", ""),
                "help": row.get("help", ""),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize route rows from fixed-C median ensemble output.")
    ap.add_argument("--fixed_json", required=True, help="fixed_c_median_ensemble.json")
    ap.add_argument("--out_apply_root", required=True, help="Output root containing target/dataset/pcp_route_rows.csv.")
    ap.add_argument("--target", action="append", default=[], help="Optional target filter; repeatable.")
    ap.add_argument("--dataset", action="append", default=[], help="Optional dataset filter; repeatable.")
    ap.add_argument("--derive_decision_kl", action="store_true", default=True)
    args = ap.parse_args()

    fixed_json = Path(args.fixed_json).expanduser().resolve()
    out_root = Path(args.out_apply_root).expanduser().resolve()
    payload = read_json(fixed_json)
    per_dataset = list(payload.get("per_dataset") or [])
    if not per_dataset:
        raise RuntimeError(f"No per_dataset rows in {fixed_json}")

    target_filter = set(args.target)
    dataset_filter = set(args.dataset)
    written: List[Dict[str, Any]] = []
    for item in per_dataset:
        target = str(item.get("target", "")).strip()
        dataset = str(item.get("dataset", "")).strip()
        if target_filter and target not in target_filter:
            continue
        if dataset_filter and dataset not in dataset_filter:
            continue
        rows_csv = Path(str(item.get("apply_rows_csv") or "")).expanduser().resolve()
        if not rows_csv.exists():
            raise FileNotFoundError(rows_csv)
        yes_policy = item.get("yes_policy_json") or {}
        no_policy = item.get("no_policy_json") or {}
        if not yes_policy or not no_policy:
            raise RuntimeError(f"Missing policy JSON for {target}/{dataset}")

        rows = pcp.load_rows(str(rows_csv), derive_decision_kl=bool(args.derive_decision_kl))
        routes = route_rows(rows, yes_policy=yes_policy, no_policy=no_policy)
        out_dir = out_root / target / dataset
        route_csv = out_dir / "pcp_route_rows.csv"
        write_csv(route_csv, routes)
        write_json(
            out_dir / "fixed_c_median_route_metadata.json",
            {
                "fixed_json": str(fixed_json),
                "target": target,
                "dataset": dataset,
                "rows_csv": str(rows_csv),
                "policy": str(item.get("policy", "")),
                "n_route_rows": len(routes),
            },
        )
        written.append({"target": target, "dataset": dataset, "route_rows_csv": str(route_csv), "n": len(routes)})
        print("[saved]", route_csv, "n=", len(routes))

    write_json(out_root / "materialize_fixed_c_median_routes_summary.json", {"written": written})
    print("[saved]", out_root / "materialize_fixed_c_median_routes_summary.json")


if __name__ == "__main__":
    main()
