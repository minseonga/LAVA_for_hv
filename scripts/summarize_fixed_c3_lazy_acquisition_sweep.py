#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def safe_id(row: Dict[str, Any]) -> str:
    for key in ("id", "question_id", "qid"):
        value = str(row.get(key, "")).strip()
        if value and value.lower() not in {"none", "null", "nan"}:
            return value
    return ""


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: Any) -> Optional[int]:
    x = maybe_float(value)
    return None if x is None else int(round(x))


def bool_int(value: Any) -> int:
    text = str(value).strip().lower()
    return int(text in {"1", "1.0", "true", "yes", "y"})


def parse_float_list(text: str, default: Sequence[float]) -> List[float]:
    if not str(text or "").strip():
        return list(default)
    out: List[float] = []
    for item in str(text).replace(",", " ").split():
        out.append(float(item))
    return out


def fmt_pct(value: Any) -> str:
    x = maybe_float(value)
    return "" if x is None else f"{100.0 * x:.2f}"


def fmt_float(value: Any, ndigits: int = 4) -> str:
    x = maybe_float(value)
    return "" if x is None else f"{x:.{ndigits}f}"


def score_excess(row: Dict[str, Any], score_key: str, tau_key: str) -> Optional[float]:
    score = maybe_float(row.get(score_key))
    tau = maybe_float(row.get(tau_key))
    if score is None or tau is None:
        return None
    return float(score - tau)


def acquisition_score(row: Dict[str, Any]) -> float:
    vals = [
        score_excess(row, "yes_to_no_score", "yes_to_no_tau"),
        score_excess(row, "no_to_yes_score", "no_to_yes_tau"),
    ]
    nums = [v for v in vals if v is not None]
    return float(max(nums)) if nums else float("-inf")


def top_budget_ids(rows: Sequence[Dict[str, Any]], budget: float) -> Set[str]:
    n = len(rows)
    k = int(math.ceil(float(max(0.0, min(1.0, budget))) * n))
    ranked = sorted(
        ((acquisition_score(row), safe_id(row)) for row in rows if safe_id(row)),
        key=lambda x: x[0],
        reverse=True,
    )
    return {qid for _, qid in ranked[:k]}


def numeric_mean(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    nums = [x for x in (maybe_float(row.get(key)) for row in rows) if x is not None]
    return None if not nums else float(sum(nums) / float(len(nums)))


def simulate(
    rows: Sequence[Dict[str, Any]],
    *,
    acquired_ids: Optional[Set[str]] = None,
    margin_threshold: Optional[float] = None,
    score_only_mean_sec: Optional[float],
    baseline_mean_sec: Optional[float],
    always_mean_sec: Optional[float],
) -> Dict[str, Any]:
    n = len(rows)
    final_correct = 0
    method_correct = 0
    acquired = 0
    route_baseline = 0
    total_harm = 0
    total_help = 0
    selected_harm = 0
    selected_help = 0

    for row in rows:
        qid = safe_id(row)
        if acquired_ids is not None:
            use_acquisition = qid in acquired_ids
        else:
            use_acquisition = acquisition_score(row) >= float(margin_threshold or 0.0)

        intervention_correct = maybe_int(row.get("intervention_correct"))
        if intervention_correct is None:
            intervention_correct = maybe_int(row.get("method_correct"))
        baseline_correct = maybe_int(row.get("baseline_correct"))
        if intervention_correct is None:
            continue

        if baseline_correct is not None:
            total_harm += int(baseline_correct == 1 and intervention_correct == 0)
            total_help += int(baseline_correct == 0 and intervention_correct == 1)
        method_correct += int(intervention_correct)
        if use_acquisition:
            acquired += 1
        use_baseline = bool(use_acquisition and str(row.get("route", "")).strip() == "baseline")
        if use_baseline and baseline_correct is not None:
            route_baseline += 1
            final_correct += int(baseline_correct)
            selected_harm += int(baseline_correct == 1 and intervention_correct == 0)
            selected_help += int(baseline_correct == 0 and intervention_correct == 1)
        else:
            final_correct += int(intervention_correct)

    denom = float(max(1, n))
    trigger_rate = acquired / denom
    route_rate = route_baseline / denom
    method_acc = method_correct / denom
    final_acc = final_correct / denom
    selected_harm_precision = selected_harm / float(max(1, route_baseline))
    selected_help_precision = selected_help / float(max(1, route_baseline))
    selected_harm_recall = selected_harm / float(max(1, total_harm))
    selected_help_recall = selected_help / float(max(1, total_help))
    lazy_mean_sec = None
    speedup = None
    savings = None
    if score_only_mean_sec is not None and baseline_mean_sec is not None:
        lazy_mean_sec = float(score_only_mean_sec + trigger_rate * baseline_mean_sec)
        if always_mean_sec is not None and lazy_mean_sec > 0:
            speedup = float(always_mean_sec / lazy_mean_sec)
            savings = float(100.0 * (1.0 - lazy_mean_sec / always_mean_sec))

    return {
        "n": n,
        "baseline_generated": acquired,
        "baseline_trigger_rate": trigger_rate,
        "route_baseline": route_baseline,
        "route_baseline_rate": route_rate,
        "method_acc": method_acc,
        "final_acc": final_acc,
        "delta_vs_method": final_acc - method_acc,
        "total_harm": total_harm,
        "total_help": total_help,
        "selected_harm": selected_harm,
        "selected_help": selected_help,
        "net": selected_harm - selected_help,
        "selected_harm_precision": selected_harm_precision,
        "selected_help_precision": selected_help_precision,
        "selected_harm_recall": selected_harm_recall,
        "selected_help_recall": selected_help_recall,
        "lazy_mean_sec_per_sample": lazy_mean_sec,
        "always_mean_sec_per_sample": always_mean_sec,
        "speedup_vs_always": speedup,
        "latency_savings_pct": savings,
    }


def load_inputs(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any], Path]:
    summary: Dict[str, Any] = {}
    feature_rows_path = Path(str(args.feature_rows_csv)).expanduser()
    if str(args.summary_json or "").strip():
        summary_path = Path(str(args.summary_json)).expanduser().resolve()
        summary = read_json(summary_path)
        if not str(args.feature_rows_csv or "").strip():
            feature_rows_path = Path(str((summary.get("outputs") or {}).get("online_feature_rows_csv", "")))
    if not str(feature_rows_path):
        raise SystemExit("Pass --summary_json or --feature_rows_csv")
    feature_rows_path = feature_rows_path.expanduser().resolve()
    rows = read_csv(feature_rows_path)
    rows = [row for row in rows if safe_id(row) and not str(row.get("score_error", "")).strip()]
    return rows, summary, feature_rows_path


def markdown(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Mode | Param | Generated % | Route % | Final Acc | Delta | Harm recall | Help recall | Harm precision | Lazy sec | Speedup | Saved % | H/G/Net |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['param']} | {fmt_pct(row['baseline_trigger_rate'])} | "
            f"{fmt_pct(row['route_baseline_rate'])} | {fmt_pct(row['final_acc'])} | "
            f"{fmt_pct(row['delta_vs_method'])} | {fmt_pct(row['selected_harm_recall'])} | "
            f"{fmt_pct(row['selected_help_recall'])} | {fmt_pct(row['selected_harm_precision'])} | "
            f"{fmt_float(row['lazy_mean_sec_per_sample'])} | "
            f"{fmt_float(row['speedup_vs_always'], 3)} | {fmt_float(row['latency_savings_pct'], 2)} | "
            f"{row['selected_harm']}/{row['selected_help']}/{row['net']} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Offline latency/accuracy sweep for fixed-C3 lazy deploy. It reuses a run "
            "where baseline was already generated for candidate rows, then simulates "
            "stricter baseline-acquisition budgets or score margins."
        )
    )
    ap.add_argument("--summary_json", default="", help="summary.json from run_llava15_fixed_c3_lazy_deploy.py.")
    ap.add_argument("--feature_rows_csv", default="", help="online_feature_rows.csv; optional if summary_json is set.")
    ap.add_argument("--budgets", default="1.0 0.75 0.50 0.25 0.10 0.05 0.01 0.0")
    ap.add_argument("--margins", default="0.0 0.5 1.0 1.5 2.0 2.5 3.0")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary, feature_rows_path = load_inputs(args)
    timing = summary.get("timing") or {}
    score_only_mean_sec = maybe_float(timing.get("estimated_score_only_no_baseline_mean_sec_per_sample"))
    baseline_mean_sec = maybe_float(timing.get("mean_baseline_generated_sec"))
    always_mean_sec = maybe_float(timing.get("estimated_always_baseline_mean_sec_per_sample"))

    out_rows: List[Dict[str, Any]] = []
    for budget in parse_float_list(args.budgets, []):
        acquired_ids = top_budget_ids(rows, float(budget))
        result = simulate(
            rows,
            acquired_ids=acquired_ids,
            score_only_mean_sec=score_only_mean_sec,
            baseline_mean_sec=baseline_mean_sec,
            always_mean_sec=always_mean_sec,
        )
        result.update({"mode": "budget", "param": f"{float(budget):.3f}", "feature_rows_csv": str(feature_rows_path)})
        out_rows.append(result)

    for margin in parse_float_list(args.margins, []):
        result = simulate(
            rows,
            margin_threshold=float(margin),
            score_only_mean_sec=score_only_mean_sec,
            baseline_mean_sec=baseline_mean_sec,
            always_mean_sec=always_mean_sec,
        )
        result.update({"mode": "margin", "param": f"{float(margin):.3f}", "feature_rows_csv": str(feature_rows_path)})
        out_rows.append(result)

    out_csv = Path(args.out_csv).expanduser().resolve()
    write_csv(out_csv, out_rows)
    print("[saved]", out_csv)
    md = markdown(out_rows)
    print(md)
    if str(args.out_md or "").strip():
        out_md = Path(args.out_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md + "\n", encoding="utf-8")
        print("[saved]", out_md)


if __name__ == "__main__":
    main()
