#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build_pcp_c_d_controller as pcp
import build_posthoc_b_c_fusion_controller as base


DIRECTIONS = ("yes_to_no", "no_to_yes")
DEFAULT_C_FEATURES = (
    "cheap_lp_content_min",
    "cheap_target_gap_content_min",
    "cheap_first_target_gap",
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


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
        for row in rows:
            wr.writerow(row)


def read_jobs(path: Path, *, target: str = "") -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        sample = f.readline()
        f.seek(0)
        has_header = "target" in sample and "dataset" in sample
        if has_header:
            for row in csv.DictReader(f, delimiter="\t"):
                jobs.append({k: str(v) for k, v in row.items()})
        else:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    raise ValueError(f"job_tsv rows need at least 4 columns: {line!r}")
                job = {
                    "target": parts[0],
                    "label": parts[1],
                    "dataset": parts[2],
                    "rows_csv": parts[3],
                }
                if len(parts) >= 5:
                    job["discovery_rows_csv"] = parts[4]
                if len(parts) >= 6:
                    job["policy_root"] = parts[5]
                if len(parts) >= 7:
                    job["deployment_summary_json"] = parts[6]
                jobs.append(job)
    if target:
        jobs = [job for job in jobs if str(job.get("target", "")) == target]
    return jobs


def direction_rows(rows: Sequence[Dict[str, Any]], direction: str) -> List[Dict[str, Any]]:
    return [row for row in rows if pcp.is_route_candidate(row, direction)]


def median_z_score(row: Dict[str, Any], features: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not features:
        return None
    zs: List[float] = []
    for feat in features:
        z = base.oriented_z(row, feat)
        if z is None:
            return None
        zs.append(float(z))
    return float(median(zs))


def median_score_values(rows: Sequence[Dict[str, Any]], features: Sequence[Dict[str, Any]], direction: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        if not pcp.is_route_candidate(row, direction):
            continue
        score = median_z_score(row, features)
        if score is not None:
            out.append(float(score))
    return out


def evaluate_median_policy(
    rows: Sequence[Dict[str, Any]],
    *,
    c_features: Sequence[Dict[str, Any]],
    tau: float,
    candidate_filter: str,
) -> Dict[str, Any]:
    n = 0
    n_route_candidates = 0
    selected = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    final_correct_total = 0
    total_harm = 0
    total_help = 0
    route_candidate_harm = 0
    route_candidate_help = 0
    route_candidate_neutral = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0

    for row in rows:
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        if bc is None or ic is None:
            continue
        score = median_z_score(row, c_features)
        if score is None:
            continue

        harm = int(base.maybe_int(row.get("harm")) or 0)
        help_ = int(base.maybe_int(row.get("help")) or 0)
        n += 1
        total_harm += harm
        total_help += help_
        baseline_correct_total += int(bc)
        intervention_correct_total += int(ic)

        can_route = pcp.is_route_candidate(row, str(candidate_filter))
        if can_route:
            n_route_candidates += 1
            route_candidate_harm += harm
            route_candidate_help += help_
            route_candidate_neutral += int((harm == 0) and (help_ == 0))

        use_baseline = bool(can_route and float(score) >= float(tau))
        if use_baseline:
            selected += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int((harm == 0) and (help_ == 0))
            final_correct_total += int(bc)
        else:
            final_correct_total += int(ic)

    precision = base.safe_div(float(selected_harm), float(max(1, selected)))
    recall = base.safe_div(float(selected_harm), float(max(1, total_harm)))
    f1 = base.safe_div(2.0 * precision * recall, precision + recall)
    return {
        "family": "c_median",
        "aggregation": "median",
        "alpha": 0.0,
        "tau": float(tau),
        "n_eval": int(n),
        "baseline_rate": base.safe_div(float(selected), float(max(1, n))),
        "method_rate": 1.0 - base.safe_div(float(selected), float(max(1, n))),
        "final_acc": base.safe_div(float(final_correct_total), float(max(1, n))),
        "baseline_acc": base.safe_div(float(baseline_correct_total), float(max(1, n))),
        "intervention_acc": base.safe_div(float(intervention_correct_total), float(max(1, n))),
        "delta_vs_intervention": base.safe_div(float(final_correct_total - intervention_correct_total), float(max(1, n))),
        "selected_count": int(selected),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "n_route_candidates": int(n_route_candidates),
        "n_route_candidate_harm": int(route_candidate_harm),
        "n_route_candidate_help": int(route_candidate_help),
        "n_route_candidate_neutral": int(route_candidate_neutral),
        "route_candidate_baseline_rate": base.safe_div(float(selected), float(max(1, n_route_candidates))),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "net": int(selected_harm - selected_help),
        "selected_harm_precision": precision,
        "selected_help_precision": base.safe_div(float(selected_help), float(max(1, selected))),
        "selected_harm_recall": recall,
        "selected_help_recall": base.safe_div(float(selected_help), float(max(1, total_help))),
        "selected_harm_recall_in_scope": base.safe_div(float(selected_harm), float(max(1, route_candidate_harm))),
        "selected_help_recall_in_scope": base.safe_div(float(selected_help), float(max(1, route_candidate_help))),
        "selected_harm_f1": f1,
    }


def disabled_policy(rows: Sequence[Dict[str, Any]], *, direction: str, reason: str) -> Dict[str, Any]:
    out = pcp.evaluate_noop_policy(rows)
    out.update(
        {
            "family": "noop",
            "aggregation": "median",
            "disabled": True,
            "candidate_filter": direction,
            "selected_c_features": [],
            "reason": reason,
        }
    )
    return out


def calibrate_median_policy(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_names: Sequence[str],
    direction: str,
    tau_objective: str,
    lambda_gain: float,
    min_present_rate: float,
    min_selected_count: int,
    min_harm_precision: float,
    min_harm_recall: float,
    max_help_recall: float,
    allow_noop_policy: bool,
) -> Dict[str, Any]:
    fit_rows = direction_rows(rows, direction)
    if not fit_rows:
        return disabled_policy(rows, direction=direction, reason="no_rows_for_direction")

    specs: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for feature in feature_names:
        present = pcp.feature_present_count(fit_rows, feature)
        present_rate = float(present) / float(max(1, len(fit_rows)))
        if present_rate < float(min_present_rate):
            rejected.append({"feature": feature, "reason": "low_present_rate", "present_rate": present_rate})
            continue
        spec = base.orient_feature(fit_rows, feature, target="harm")
        if spec is None:
            rejected.append({"feature": feature, "reason": "cannot_orient", "present_rate": present_rate})
            continue
        spec["present_rate"] = present_rate
        specs.append(spec)

    if len(specs) != len(feature_names):
        return disabled_policy(
            rows,
            direction=direction,
            reason=f"missing_or_unusable_features:{rejected}",
        )

    values = median_score_values(rows, specs, direction)
    if not values:
        return disabled_policy(rows, direction=direction, reason="no_score_values")

    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for tau in pcp.threshold_grid(values):
        result = evaluate_median_policy(rows, c_features=specs, tau=float(tau), candidate_filter=direction)
        result["candidate_filter"] = direction
        result["selected_c_features"] = specs
        candidates.append(result)
        if int(result["selected_count"]) < int(min_selected_count):
            continue
        if float(result["selected_harm_precision"]) < float(min_harm_precision):
            continue
        if float(result["selected_harm_recall"]) < float(min_harm_recall):
            continue
        if float(result["selected_help_recall"]) > float(max_help_recall):
            continue
        if best is None or pcp.selection_key(result, tau_objective, lambda_gain) > pcp.selection_key(
            best, tau_objective, lambda_gain
        ):
            best = result

    if bool(allow_noop_policy):
        noop = disabled_policy(rows, direction=direction, reason="allowed_noop_candidate")
        if best is None or pcp.selection_key(noop, tau_objective, lambda_gain) >= pcp.selection_key(
            best, tau_objective, lambda_gain
        ):
            best = noop

    if best is None:
        best = disabled_policy(rows, direction=direction, reason="no_policy_after_constraints")
    best["tau_sweep_count"] = len(candidates)
    best["selected_c_features"] = specs if not bool(best.get("disabled")) else []
    best["rejected_features"] = rejected
    return best


def compute_route(row: Dict[str, Any], policy: Dict[str, Any], direction: str) -> Tuple[str, Optional[float]]:
    if bool(policy.get("disabled")) or str(policy.get("family", "")) == "noop":
        return "method", None
    features = list(policy.get("selected_c_features") or [])
    score = median_z_score(row, features)
    if score is None:
        return "method", None
    if pcp.is_route_candidate(row, direction) and float(score) >= float(policy.get("tau", 0.0)):
        return "baseline", float(score)
    return "method", float(score)


def summarize_application(
    rows: Sequence[Dict[str, Any]],
    *,
    yes_policy: Dict[str, Any],
    no_policy: Dict[str, Any],
    old_deploy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n = 0
    baseline_correct = 0
    method_correct = 0
    final_correct = 0
    total_harm = 0
    total_help = 0
    selected = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0

    for row in rows:
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        if bc is None or ic is None:
            continue
        harm = int(base.maybe_int(row.get("harm")) or 0)
        help_ = int(base.maybe_int(row.get("help")) or 0)
        y_route, _ = compute_route(row, yes_policy, "yes_to_no")
        n_route, _ = compute_route(row, no_policy, "no_to_yes")
        use_baseline = y_route == "baseline" or n_route == "baseline"

        n += 1
        baseline_correct += int(bc)
        method_correct += int(ic)
        total_harm += harm
        total_help += help_
        if use_baseline:
            selected += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int((harm == 0) and (help_ == 0))
            final_correct += int(bc)
        else:
            final_correct += int(ic)

    baseline_acc = float(baseline_correct) / float(max(1, n))
    method_acc = float(method_correct) / float(max(1, n))
    final_acc = float(final_correct) / float(max(1, n))
    report_n = int(n)
    report_total_harm = int(total_harm)
    report_total_help = int(total_help)

    if old_deploy is not None:
        report_n = int(old_deploy.get("n", n) or n)
        baseline_acc = float(old_deploy.get("baseline_acc", baseline_acc) or baseline_acc)
        method_acc = float(old_deploy.get("intervention_acc", method_acc) or method_acc)
        report_total_harm = int(old_deploy.get("total_harm", total_harm) or total_harm)
        report_total_help = int(old_deploy.get("total_help", total_help) or total_help)
        final_acc = method_acc + float(selected_harm - selected_help) / float(max(1, report_n))

    return {
        "n": int(report_n),
        "n_route_rows": int(n),
        "baseline_acc": baseline_acc,
        "intervention_acc": method_acc,
        "final_acc": final_acc,
        "delta_vs_intervention": final_acc - method_acc,
        "selected_count": int(selected),
        "total_harm": int(report_total_harm),
        "total_help": int(report_total_help),
        "route_rows_total_harm": int(total_harm),
        "route_rows_total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "net": int(selected_harm - selected_help),
        "selected_harm_recall": float(selected_harm) / float(max(1, report_total_harm)),
        "selected_help_recall": float(selected_help) / float(max(1, report_total_help)),
        "accuracy_source": "deployment_summary" if old_deploy is not None else "rows_csv",
    }


def fmt_pct(value: Any, *, signed: bool = False) -> str:
    try:
        v = 100.0 * float(value)
    except Exception:
        return ""
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def fmt_num(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if math.isclose(v, round(v), abs_tol=1e-9):
        return str(int(round(v)))
    return f"{v:.2f}"


def policy_short(policy: Dict[str, Any]) -> str:
    if bool(policy.get("disabled")) or str(policy.get("family", "")) == "noop":
        return "noop"
    tau = float(policy.get("tau", 0.0) or 0.0)
    feats = ",".join(str(f.get("feature", "")) for f in policy.get("selected_c_features", []))
    dirs = ",".join(str(f.get("direction", "")) for f in policy.get("selected_c_features", []))
    return f"median_z[{feats};{dirs}]>={tau:.3f}"


def average_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["target"]), str(row["label"])), []).append(row)

    out: List[Dict[str, Any]] = []
    for (target, label), group in sorted(grouped.items()):
        h = mean(float(r["selected_harm"]) for r in group)
        g = mean(float(r["selected_help"]) for r in group)
        out.append(
            {
                "target": target,
                "label": label,
                "dataset": "avg",
                "base": mean(float(r["base"]) for r in group),
                "method": mean(float(r["method"]) for r in group),
                "median_c": mean(float(r["median_c"]) for r in group),
                "d_method": mean(float(r["d_method"]) for r in group),
                "fallback": mean(float(r["fallback"]) for r in group),
                "selected_harm": h,
                "selected_help": g,
                "net": h - g,
                "hrec": mean(float(r["hrec"]) for r in group),
                "grec": mean(float(r["grec"]) for r in group),
                "n_datasets": len(group),
            }
        )
    return out


def total_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["target"]), str(row["label"])), []).append(row)

    out: List[Dict[str, Any]] = []
    for (target, label), group in sorted(grouped.items()):
        n_total = sum(int(r["n"]) for r in group)
        h_total = sum(int(r["total_harm"]) for r in group)
        g_total = sum(int(r["total_help"]) for r in group)
        selected_h = sum(int(r["selected_harm"]) for r in group)
        selected_g = sum(int(r["selected_help"]) for r in group)
        out.append(
            {
                "target": target,
                "label": label,
                "dataset": "total",
                "n": n_total,
                "base": sum(float(r["base"]) * int(r["n"]) for r in group) / float(max(1, n_total)),
                "method": sum(float(r["method"]) * int(r["n"]) for r in group) / float(max(1, n_total)),
                "median_c": sum(float(r["median_c"]) * int(r["n"]) for r in group) / float(max(1, n_total)),
                "d_method": sum(float(r["d_method"]) * int(r["n"]) for r in group) / float(max(1, n_total)),
                "fallback": sum(int(r["fallback"]) for r in group),
                "selected_harm": selected_h,
                "selected_help": selected_g,
                "net": selected_h - selected_g,
                "hrec": float(selected_h) / float(max(1, h_total)),
                "grec": float(selected_g) / float(max(1, g_total)),
                "n_datasets": len(group),
            }
        )
    return out


def md_table(rows: Sequence[Dict[str, Any]], *, include_label: bool, average: bool = False) -> str:
    dataset_col = "Datasets" if average else "Dataset"
    label_prefix = "| Method / Backbone " if include_label else "|"
    header_prefix = "| --- " if include_label else "|"
    lines = [
        f"{label_prefix}| {dataset_col} | Base | Method | Median-C | dMethod | Fallback | H/G/Net | Hrec | Grec |",
        f"{header_prefix}| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        prefix = f"| {row.get('label', row.get('target', ''))} " if include_label else "|"
        dataset = str(row.get("n_datasets", "")) if average else str(row.get("dataset", ""))
        h = float(row.get("selected_harm", 0.0) or 0.0)
        g = float(row.get("selected_help", 0.0) or 0.0)
        lines.append(
            f"{prefix}| {dataset} | {fmt_pct(row.get('base'))} | {fmt_pct(row.get('method'))} | "
            f"{fmt_pct(row.get('median_c'))} | {fmt_pct(row.get('d_method'), signed=True)} | "
            f"{fmt_num(row.get('fallback'))} | {fmt_num(h)}/{fmt_num(g)}/{fmt_num(h - g)} | "
            f"{fmt_pct(row.get('hrec'))} | {fmt_pct(row.get('grec'))} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Calibrate a fixed C-feature median ensemble on discovery rows and apply "
            "the frozen median threshold to held-out datasets."
        )
    )
    ap.add_argument("--job_tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", default="")
    ap.add_argument("--discovery_rows_csv", default="")
    ap.add_argument("--c_features", default=",".join(DEFAULT_C_FEATURES))
    ap.add_argument(
        "--tau_objective",
        default="final_acc",
        choices=["final_acc", "net", "harm_precision", "harm_recall", "harm_f1", "gain_preserving_harm_recall"],
    )
    ap.add_argument("--lambda_gain", type=float, default=1.0)
    ap.add_argument("--min_present_rate", type=float, default=0.8)
    ap.add_argument("--min_selected_count", type=int, default=5)
    ap.add_argument("--min_harm_precision", type=float, default=0.0)
    ap.add_argument("--min_harm_recall", type=float, default=0.0)
    ap.add_argument("--max_help_recall", type=float, default=1.0)
    ap.add_argument("--allow_noop_policy", type=parse_bool, default=True)
    ap.add_argument("--derive_decision_kl", type=parse_bool, default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    jobs = read_jobs(Path(args.job_tsv).resolve(), target=str(args.target))
    if not jobs:
        raise RuntimeError("No jobs found.")

    feature_names = [x.strip() for x in str(args.c_features).split(",") if x.strip()]
    if len(feature_names) < 1:
        raise RuntimeError("No C features supplied.")

    policy_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    per_dataset_rows: List[Dict[str, Any]] = []

    for job in jobs:
        target = str(job.get("target", ""))
        label = str(job.get("label", target))
        dataset = str(job.get("dataset", ""))
        rows_csv_text = str(job.get("rows_csv") or job.get("rows") or job.get("apply_rows_csv") or "")
        discovery_csv_text = str(
            job.get("discovery_rows_csv") or job.get("discovery_rows") or args.discovery_rows_csv
        )
        deploy_json_text = str(job.get("deployment_summary_json") or job.get("deploy_json") or "")
        rows_csv = Path(rows_csv_text).resolve()
        discovery_csv = Path(discovery_csv_text).resolve()
        deploy_json = Path(deploy_json_text).resolve() if deploy_json_text.strip() else None
        if not rows_csv.exists():
            raise FileNotFoundError(rows_csv)
        if not discovery_csv.exists():
            raise FileNotFoundError(discovery_csv)
        old_deploy = read_json(deploy_json) if deploy_json is not None and deploy_json.exists() else None

        apply_rows = pcp.load_rows(str(rows_csv), derive_decision_kl=bool(args.derive_decision_kl))
        discovery_rows = pcp.load_rows(str(discovery_csv), derive_decision_kl=bool(args.derive_decision_kl))

        for direction in DIRECTIONS:
            key = (str(discovery_csv), direction)
            if key not in policy_cache:
                policy_cache[key] = calibrate_median_policy(
                    discovery_rows,
                    feature_names=feature_names,
                    direction=direction,
                    tau_objective=str(args.tau_objective),
                    lambda_gain=float(args.lambda_gain),
                    min_present_rate=float(args.min_present_rate),
                    min_selected_count=int(args.min_selected_count),
                    min_harm_precision=float(args.min_harm_precision),
                    min_harm_recall=float(args.min_harm_recall),
                    max_help_recall=float(args.max_help_recall),
                    allow_noop_policy=bool(args.allow_noop_policy),
                )

        yes_policy = policy_cache[(str(discovery_csv), "yes_to_no")]
        no_policy = policy_cache[(str(discovery_csv), "no_to_yes")]
        summary = summarize_application(apply_rows, yes_policy=yes_policy, no_policy=no_policy, old_deploy=old_deploy)
        policy_label = f"Y:{policy_short(yes_policy)} / N:{policy_short(no_policy)}"
        per_dataset_rows.append(
            {
                "target": target,
                "label": label,
                "dataset": dataset,
                "policy": policy_label,
                "features": ",".join(feature_names),
                "n": summary["n"],
                "base": summary["baseline_acc"],
                "method": summary["intervention_acc"],
                "median_c": summary["final_acc"],
                "d_method": summary["delta_vs_intervention"],
                "fallback": summary["selected_count"],
                "selected_harm": summary["selected_harm"],
                "selected_help": summary["selected_help"],
                "net": summary["net"],
                "total_harm": summary["total_harm"],
                "total_help": summary["total_help"],
                "hrec": summary["selected_harm_recall"],
                "grec": summary["selected_help_recall"],
                "apply_rows_csv": str(rows_csv),
                "discovery_rows_csv": str(discovery_csv),
                "deployment_summary_json": str(deploy_json) if deploy_json is not None else "",
                "accuracy_source": summary["accuracy_source"],
                "yes_policy_json": yes_policy,
                "no_policy_json": no_policy,
            }
        )

    avg = average_rows(per_dataset_rows)
    total = total_rows(per_dataset_rows)
    write_csv(out_dir / "fixed_c_median_ensemble.csv", per_dataset_rows)
    write_csv(out_dir / "fixed_c_median_ensemble_avg.csv", avg)
    write_csv(out_dir / "fixed_c_median_ensemble_total.csv", total)
    write_json(
        out_dir / "fixed_c_median_ensemble.json",
        {
            "inputs": {
                "job_tsv": str(Path(args.job_tsv).resolve()),
                "target": str(args.target),
                "discovery_rows_csv": str(Path(args.discovery_rows_csv).resolve()) if args.discovery_rows_csv else "",
                "c_features": feature_names,
                "aggregation": "median",
                "tau_objective": str(args.tau_objective),
                "min_present_rate": float(args.min_present_rate),
                "min_selected_count": int(args.min_selected_count),
                "allow_noop_policy": bool(args.allow_noop_policy),
            },
            "per_dataset": per_dataset_rows,
            "average": avg,
            "total": total,
        },
    )
    md = "## Average Across Datasets\n\n" + md_table(avg, include_label=True, average=True)
    md += "\n\n## Total Counts Across Datasets\n\n" + md_table(total, include_label=True, average=True)
    md += "\n\n## Per Dataset\n\n" + md_table(per_dataset_rows, include_label=True, average=False)
    (out_dir / "fixed_c_median_ensemble.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print("[saved]", out_dir / "fixed_c_median_ensemble.md")


if __name__ == "__main__":
    main()
