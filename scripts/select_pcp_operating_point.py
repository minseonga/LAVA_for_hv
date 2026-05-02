#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def maybe_int(value: Any) -> Optional[int]:
    f = maybe_float(value)
    if f is None:
        return None
    return int(f)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def numeric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    int_keys = {
        "n_eval",
        "selected_count",
        "total_harm",
        "total_help",
        "n_route_candidates",
        "n_route_candidate_harm",
        "n_route_candidate_help",
        "n_route_candidate_neutral",
        "selected_harm",
        "selected_help",
        "selected_neutral",
        "net",
        "calibration_score_count",
    }
    for key, value in row.items():
        if key in {"family", "family_key", "score_space", "fusion_name"}:
            out[key] = value
            continue
        if key in int_keys:
            parsed_i = maybe_int(value)
            out[key] = parsed_i if parsed_i is not None else value
            continue
        parsed_f = maybe_float(value)
        out[key] = parsed_f if parsed_f is not None else value
    return out


def safe_div(num: float, den: float) -> float:
    return float(num / den) if float(den) != 0.0 else 0.0


def gain_preserving_score(row: Dict[str, Any], lambda_gain: float) -> float:
    selected_harm = float(maybe_float(row.get("selected_harm")) or 0.0)
    selected_help = float(maybe_float(row.get("selected_help")) or 0.0)
    total_harm = float(maybe_float(row.get("total_harm")) or 0.0)
    total_help = float(maybe_float(row.get("total_help")) or 0.0)
    harm_recall = safe_div(selected_harm, total_harm)
    help_destroy = safe_div(selected_help, total_help)
    return float(harm_recall - float(lambda_gain) * help_destroy)


def selection_key(row: Dict[str, Any], objective: str, lambda_gain: float) -> Tuple[float, float, float, float]:
    if objective == "gain_preserving_harm_recall":
        return (
            gain_preserving_score(row, lambda_gain),
            float(maybe_float(row.get("net")) or 0.0),
            float(maybe_float(row.get("selected_harm_precision")) or 0.0),
            -float(maybe_float(row.get("baseline_rate")) or 0.0),
        )
    if objective == "net":
        return (
            float(maybe_float(row.get("net")) or 0.0),
            float(maybe_float(row.get("final_acc")) or 0.0),
            float(maybe_float(row.get("selected_harm_precision")) or 0.0),
            -float(maybe_float(row.get("baseline_rate")) or 0.0),
        )
    if objective == "harm_precision":
        return (
            float(maybe_float(row.get("selected_harm_precision")) or 0.0),
            float(maybe_float(row.get("selected_harm_recall")) or 0.0),
            float(maybe_float(row.get("net")) or 0.0),
            -float(maybe_float(row.get("baseline_rate")) or 0.0),
        )
    if objective == "harm_recall":
        return (
            float(maybe_float(row.get("selected_harm_recall")) or 0.0),
            float(maybe_float(row.get("selected_harm_precision")) or 0.0),
            float(maybe_float(row.get("net")) or 0.0),
            -float(maybe_float(row.get("baseline_rate")) or 0.0),
        )
    if objective == "harm_f1":
        return (
            float(maybe_float(row.get("selected_harm_f1")) or 0.0),
            float(maybe_float(row.get("selected_harm_precision")) or 0.0),
            float(maybe_float(row.get("net")) or 0.0),
            -float(maybe_float(row.get("baseline_rate")) or 0.0),
        )
    return (
        float(maybe_float(row.get("final_acc")) or 0.0),
        float(maybe_float(row.get("net")) or 0.0),
        float(maybe_float(row.get("selected_harm_precision")) or 0.0),
        -float(maybe_float(row.get("baseline_rate")) or 0.0),
    )


def row_matches(row: Dict[str, Any], *, family: str, alpha: Optional[float], alpha_tol: float) -> bool:
    if str(row.get("family", "")).strip() != str(family):
        return False
    if alpha is None:
        return True
    row_alpha = maybe_float(row.get("alpha"))
    if row_alpha is None:
        return False
    return abs(float(row_alpha) - float(alpha)) <= float(alpha_tol)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select a fixed-family PCP operating point from tau_sweep.csv.")
    ap.add_argument("--policy_json", required=True, help="Base build_pcp_c_d_controller selected_policy.json.")
    ap.add_argument("--tau_sweep_csv", default="", help="Defaults to tau_sweep.csv next to policy_json.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--family", required=True, choices=["c_only", "d_only", "cd_fusion"])
    ap.add_argument("--alpha", type=float, default=None, help="Optional fixed alpha. Use 0 for c_only, 1 for d_only.")
    ap.add_argument("--alpha_tol", type=float, default=1e-9)
    ap.add_argument(
        "--objective",
        default="gain_preserving_harm_recall",
        choices=["final_acc", "net", "harm_precision", "harm_recall", "harm_f1", "gain_preserving_harm_recall"],
    )
    ap.add_argument("--lambda_gain", type=float, default=1.0)
    ap.add_argument("--min_selected_count", type=int, default=0)
    ap.add_argument("--min_harm_precision", type=float, default=0.0)
    ap.add_argument("--min_harm_recall", type=float, default=0.0)
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    args = ap.parse_args()

    policy_path = os.path.abspath(args.policy_json)
    tau_path = os.path.abspath(args.tau_sweep_csv) if str(args.tau_sweep_csv).strip() else os.path.join(os.path.dirname(policy_path), "tau_sweep.csv")
    with open(policy_path, "r", encoding="utf-8") as f:
        bundle: Dict[str, Any] = json.load(f)
    rows = read_csv_rows(tau_path)

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if not row_matches(row, family=str(args.family), alpha=args.alpha, alpha_tol=float(args.alpha_tol)):
            continue
        if int(maybe_int(row.get("selected_count")) or 0) < int(args.min_selected_count):
            continue
        if float(maybe_float(row.get("selected_harm_precision")) or 0.0) < float(args.min_harm_precision):
            continue
        if float(maybe_float(row.get("selected_harm_recall")) or 0.0) < float(args.min_harm_recall):
            continue
        baseline_rate = float(maybe_float(row.get("baseline_rate")) or 0.0)
        if baseline_rate < float(args.min_baseline_rate) or baseline_rate > float(args.max_baseline_rate):
            continue
        candidates.append(row)

    if not candidates:
        raise RuntimeError(
            "No candidates after filters: "
            f"family={args.family}, alpha={args.alpha}, objective={args.objective}"
        )

    best = max(candidates, key=lambda r: selection_key(r, str(args.objective), float(args.lambda_gain)))
    selected_policy = numeric_row(best)
    selected_policy["family"] = str(args.family)
    if args.alpha is not None:
        selected_policy["alpha"] = float(args.alpha)

    bundle["selected_policy"] = selected_policy
    bundle.setdefault("best_results", {})[str(args.family)] = selected_policy
    bundle["operating_point_selection"] = {
        "mode": "fixed_family_tau_sweep",
        "policy_json": policy_path,
        "tau_sweep_csv": tau_path,
        "family": str(args.family),
        "alpha": args.alpha,
        "objective": str(args.objective),
        "lambda_gain": float(args.lambda_gain),
        "min_selected_count": int(args.min_selected_count),
        "min_harm_precision": float(args.min_harm_precision),
        "min_harm_recall": float(args.min_harm_recall),
        "min_baseline_rate": float(args.min_baseline_rate),
        "max_baseline_rate": float(args.max_baseline_rate),
        "n_candidates": int(len(candidates)),
        "selection_key": list(selection_key(best, str(args.objective), float(args.lambda_gain))),
    }

    out_dir = os.path.abspath(args.out_dir)
    out_policy = os.path.join(out_dir, "selected_policy.json")
    out_summary = os.path.join(out_dir, "summary.json")
    write_json(out_policy, bundle)
    write_json(
        out_summary,
        {
            "inputs": bundle["operating_point_selection"],
            "selected_policy": selected_policy,
            "outputs": {
                "selected_policy_json": out_policy,
                "summary_json": out_summary,
            },
        },
    )
    print(json.dumps({"selected_policy": selected_policy, "outputs": {"selected_policy_json": out_policy}}, ensure_ascii=False, indent=2))
    print("[saved]", out_policy)
    print("[saved]", out_summary)


if __name__ == "__main__":
    main()
