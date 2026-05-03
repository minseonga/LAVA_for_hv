#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_float(value: object) -> Optional[float]:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def maybe_int(value: object) -> Optional[int]:
    value = maybe_float(value)
    if value is None:
        return None
    return int(round(value))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if float(den) else 0.0


def load_selected_rate_from_policy(policy_json: str) -> float:
    with open(os.path.abspath(policy_json), "r", encoding="utf-8") as f:
        bundle = json.load(f)
    policy = bundle.get("selected_policy") or {}
    selected = int(policy.get("selected_count") or 0)
    candidates = int(policy.get("n_route_candidates") or 0)
    if candidates <= 0:
        n_eval = int(policy.get("n_eval") or 0)
        candidates = n_eval
    return safe_div(float(selected), float(candidates))


def score_key(row: Dict[str, str]) -> float:
    score = maybe_float(row.get("score"))
    return float("-inf") if score is None else float(score)


def route_with_cap(rows: Sequence[Dict[str, str]], *, rate: float, k: Optional[int]) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if str(row.get("route_candidate", "")).strip() in {"1", "true", "True"}]
    if k is None:
        k_eff = int(round(float(rate) * float(len(candidates))))
    else:
        k_eff = int(k)
    k_eff = max(0, min(k_eff, len(candidates)))
    selected_ids = {
        str(row.get("id", row.get("question_id", ""))).strip()
        for row in sorted(candidates, key=score_key, reverse=True)[:k_eff]
    }
    out: List[Dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("id", row.get("question_id", ""))).strip()
        route = "baseline" if sid in selected_ids else "method"
        final_text = row.get("final_text", "")
        if route == "method":
            # Existing route rows do not always include intervention_text, but
            # summarize_pcp_deployment_from_routes only needs route decisions.
            final_text = ""
        copied: Dict[str, Any] = dict(row)
        copied["route_original"] = row.get("route", "")
        copied["route"] = route
        copied["rate_cap"] = float(rate)
        copied["rate_cap_k"] = int(k_eff)
        copied["final_source"] = "baseline_cached" if route == "baseline" else "method"
        copied["final_text"] = final_text
        out.append(copied)
    return out


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    route_baseline = [row for row in rows if str(row.get("route")) == "baseline"]
    total_harm = sum(int(maybe_int(row.get("harm")) or 0) for row in rows)
    total_help = sum(int(maybe_int(row.get("help")) or 0) for row in rows)
    selected_harm = sum(int(maybe_int(row.get("harm")) or 0) for row in route_baseline)
    selected_help = sum(int(maybe_int(row.get("help")) or 0) for row in route_baseline)
    selected_neutral = int(len(route_baseline) - selected_harm - selected_help)
    return {
        "n_route_rows": int(n),
        "n_route_candidates": sum(int(str(row.get("route_candidate", "")).strip() in {"1", "true", "True"}) for row in rows),
        "fallback": int(len(route_baseline)),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "net": int(selected_harm - selected_help),
        "selected_harm_precision": safe_div(float(selected_harm), float(len(route_baseline))),
        "selected_help_precision": safe_div(float(selected_help), float(len(route_baseline))),
        "selected_harm_recall": safe_div(float(selected_harm), float(total_harm)),
        "selected_help_recall": safe_div(float(selected_help), float(total_help)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-apply PCP route rows using a calibrated top-score fallback rate cap.")
    ap.add_argument("--route_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--policy_json", default="", help="Use selected_count/n_route_candidates from policy_json.")
    ap.add_argument("--rate", type=float, default=None, help="Fallback fraction among route candidates.")
    ap.add_argument("--k", type=int, default=None, help="Absolute number of fallback route candidates.")
    args = ap.parse_args()

    if args.k is None and args.rate is None and not str(args.policy_json).strip():
        raise ValueError("Provide one of --k, --rate, or --policy_json.")
    rate = float(args.rate) if args.rate is not None else load_selected_rate_from_policy(str(args.policy_json))
    rows = read_csv(args.route_rows_csv)
    capped = route_with_cap(rows, rate=rate, k=args.k)
    summary = summarize(capped)
    summary["rate"] = float(rate)
    summary["k_arg"] = args.k
    summary["policy_json"] = os.path.abspath(args.policy_json) if str(args.policy_json).strip() else ""

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "pcp_route_rows.csv")
    out_summary = os.path.join(out_dir, "summary.json")
    write_csv(out_csv, capped)
    write_json(
        out_summary,
        {
            "mode": "apply_pcp_routes_with_rate_cap",
            "inputs": {
                "route_rows_csv": os.path.abspath(args.route_rows_csv),
                "policy_json": summary["policy_json"],
                "rate": float(rate),
                "k": args.k,
            },
            "summary": summary,
            "outputs": {
                "pcp_route_rows_csv": out_csv,
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[saved]", out_csv)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()
