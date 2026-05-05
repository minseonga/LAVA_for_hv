#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import build_pcp_c_d_controller as pcp
import build_posthoc_b_c_fusion_controller as base


DIRECTIONS = ("yes_to_no", "no_to_yes")
DATASETS = ("mscoco", "aokvqa", "gqa")


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
                jobs.append(job)
    if target:
        jobs = [job for job in jobs if str(job.get("target", "")) == target]
    return jobs


def feature_names_from_policy_root(policy_root: Path) -> List[str]:
    names: List[str] = []
    seen = set()
    for direction in DIRECTIONS:
        path = policy_root / direction / "selected_policy.json"
        if not path.exists():
            continue
        bundle = read_json(path)
        for feat in bundle.get("selected_c_features") or []:
            name = str(feat.get("feature", "")).strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def policy_short(policy: Dict[str, Any]) -> str:
    if bool(policy.get("disabled")) or str(policy.get("family", "")) == "noop":
        return "noop"
    feat = str(((policy.get("selected_c_features") or [{}])[0]).get("feature", "c"))
    direction = str(((policy.get("selected_c_features") or [{}])[0]).get("direction", "high"))
    tau = float(policy.get("tau", 0.0) or 0.0)
    return f"z[{feat},{direction}]>={tau:.3f}"


def direction_fit_rows(rows: Sequence[Dict[str, Any]], direction: str) -> List[Dict[str, Any]]:
    return [row for row in rows if pcp.is_route_candidate(row, direction)]


def score_values(rows: Sequence[Dict[str, Any]], feature_spec: Dict[str, Any], direction: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        if not pcp.is_route_candidate(row, direction):
            continue
        z = base.oriented_z(row, feature_spec)
        if z is not None:
            out.append(float(z))
    return out


def disabled_policy(
    *,
    rows: Sequence[Dict[str, Any]],
    direction: str,
    feature: str,
    reason: str,
) -> Dict[str, Any]:
    out = pcp.evaluate_noop_policy(rows)
    out.update(
        {
            "family": "noop",
            "disabled": True,
            "candidate_filter": direction,
            "feature": feature,
            "selected_c_features": [],
            "reason": reason,
        }
    )
    return out


def calibrate_one_feature(
    rows: Sequence[Dict[str, Any]],
    *,
    feature: str,
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
    fit_rows = direction_fit_rows(rows, direction)
    if not fit_rows:
        return disabled_policy(rows=rows, direction=direction, feature=feature, reason="no_rows_for_direction")

    present = pcp.feature_present_count(fit_rows, feature)
    present_rate = float(present) / float(max(1, len(fit_rows)))
    if present_rate < float(min_present_rate):
        return disabled_policy(
            rows=rows,
            direction=direction,
            feature=feature,
            reason=f"present_rate<{min_present_rate}",
        )

    spec = base.orient_feature(fit_rows, feature, target="harm")
    if spec is None:
        return disabled_policy(rows=rows, direction=direction, feature=feature, reason="cannot_orient_feature")
    spec["present_rate"] = present_rate

    values = score_values(rows, spec, direction)
    if not values:
        return disabled_policy(rows=rows, direction=direction, feature=feature, reason="no_score_values")

    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for tau in pcp.threshold_grid(values):
        result = pcp.evaluate_policy(
            rows,
            c_features=[spec],
            d_features=[],
            family="c_only",
            alpha=0.0,
            tau=float(tau),
            candidate_filter=direction,
        )
        result["candidate_filter"] = direction
        result["feature"] = feature
        result["selected_c_features"] = [spec]
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
        noop = disabled_policy(rows=rows, direction=direction, feature=feature, reason="allowed_noop_candidate")
        if best is None or pcp.selection_key(noop, tau_objective, lambda_gain) >= pcp.selection_key(
            best, tau_objective, lambda_gain
        ):
            best = noop

    if best is None:
        best = disabled_policy(rows=rows, direction=direction, feature=feature, reason="no_policy_after_constraints")
    best["tau_sweep_count"] = len(candidates)
    best["feature_metrics"] = spec
    return best


def compute_route(row: Dict[str, Any], policy: Dict[str, Any], direction: str) -> Tuple[str, Optional[float]]:
    if bool(policy.get("disabled")) or str(policy.get("family", "")) == "noop":
        return "method", None
    features = list(policy.get("selected_c_features") or [])
    score = pcp.mean_z_score(row, features)
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

    method_acc = float(method_correct) / float(max(1, n))
    final_acc = float(final_correct) / float(max(1, n))
    return {
        "n": int(n),
        "baseline_acc": float(baseline_correct) / float(max(1, n)),
        "intervention_acc": method_acc,
        "final_acc": final_acc,
        "delta_vs_intervention": final_acc - method_acc,
        "selected_count": int(selected),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "net": int(selected_harm - selected_help),
        "selected_harm_recall": float(selected_harm) / float(max(1, total_harm)),
        "selected_help_recall": float(selected_help) / float(max(1, total_help)),
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


def average_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["target"]), str(row["label"]), str(row["feature"]))
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (target, label, feature), group in sorted(grouped.items()):
        if not group:
            continue
        h = mean(float(r["selected_harm"]) for r in group)
        g = mean(float(r["selected_help"]) for r in group)
        out.append(
            {
                "target": target,
                "label": label,
                "feature": feature,
                "dataset": "avg",
                "policy": group[0].get("policy", ""),
                "base": mean(float(r["base"]) for r in group),
                "method": mean(float(r["method"]) for r in group),
                "single_c": mean(float(r["single_c"]) for r in group),
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


def md_table(rows: Sequence[Dict[str, Any]], *, average: bool = False) -> str:
    dataset_col = "Dataset" if not average else "Datasets"
    lines = [
        f"| Feature | {dataset_col} | Policy | Base | Method | Single-C | dMethod | Fallback | H/G/Net | Hrec | Grec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if average:
            dataset = str(row.get("n_datasets", ""))
        h = float(row.get("selected_harm", 0.0) or 0.0)
        g = float(row.get("selected_help", 0.0) or 0.0)
        lines.append(
            f"| {row.get('feature', '')} | {dataset} | {row.get('policy', '')} | "
            f"{fmt_pct(row.get('base'))} | {fmt_pct(row.get('method'))} | "
            f"{fmt_pct(row.get('single_c'))} | {fmt_pct(row.get('d_method'), signed=True)} | "
            f"{fmt_num(row.get('fallback'))} | {fmt_num(h)}/{fmt_num(g)}/{fmt_num(h - g)} | "
            f"{fmt_pct(row.get('hrec'))} | {fmt_pct(row.get('grec'))} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Calibrate each C feature independently on discovery rows, freeze its tau, "
            "and apply it to three held-out datasets."
        )
    )
    ap.add_argument("--job_tsv", required=True, help="TSV with target,label,dataset,rows_csv[,discovery_rows_csv].")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target", default="", help="Optional target filter for job_tsv.")
    ap.add_argument("--discovery_rows_csv", default="", help="Discovery rows used when job_tsv has no 5th column.")
    ap.add_argument("--policy_root", default="", help="Optional split policy root used to infer selected C features.")
    ap.add_argument("--c_features", default="", help="Comma-separated C features. Overrides --policy_root feature discovery.")
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
    if not feature_names and str(args.policy_root).strip():
        feature_names = feature_names_from_policy_root(Path(args.policy_root).resolve())
    if not feature_names:
        raise RuntimeError("No C features supplied. Use --c_features or --policy_root.")

    policy_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    per_dataset_rows: List[Dict[str, Any]] = []

    for job in jobs:
        target = str(job.get("target", ""))
        label = str(job.get("label", target))
        dataset = str(job.get("dataset", ""))
        rows_csv_text = str(job.get("rows_csv") or job.get("rows") or job.get("apply_rows_csv") or "")
        discovery_csv_text = str(
            job.get("discovery_rows_csv") or job.get("discovery_rows") or args.discovery_rows_csv
        )
        rows_csv = Path(rows_csv_text).resolve()
        discovery_csv = Path(discovery_csv_text).resolve()
        if not rows_csv.exists():
            raise FileNotFoundError(rows_csv)
        if not discovery_csv.exists():
            raise FileNotFoundError(discovery_csv)

        apply_rows = pcp.load_rows(str(rows_csv), derive_decision_kl=bool(args.derive_decision_kl))
        discovery_rows = pcp.load_rows(str(discovery_csv), derive_decision_kl=bool(args.derive_decision_kl))

        for feature in feature_names:
            for direction in DIRECTIONS:
                key = (str(discovery_csv), feature, direction)
                if key not in policy_cache:
                    policy_cache[key] = calibrate_one_feature(
                        discovery_rows,
                        feature=feature,
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

            yes_policy = policy_cache[(str(discovery_csv), feature, "yes_to_no")]
            no_policy = policy_cache[(str(discovery_csv), feature, "no_to_yes")]
            summary = summarize_application(apply_rows, yes_policy=yes_policy, no_policy=no_policy)
            policy_label = f"Y:{policy_short(yes_policy)} / N:{policy_short(no_policy)}"
            row = {
                "target": target,
                "label": label,
                "dataset": dataset,
                "feature": feature,
                "policy": policy_label,
                "base": summary["baseline_acc"],
                "method": summary["intervention_acc"],
                "single_c": summary["final_acc"],
                "d_method": summary["delta_vs_intervention"],
                "fallback": summary["selected_count"],
                "selected_harm": summary["selected_harm"],
                "selected_help": summary["selected_help"],
                "net": summary["net"],
                "hrec": summary["selected_harm_recall"],
                "grec": summary["selected_help_recall"],
                "apply_rows_csv": str(rows_csv),
                "discovery_rows_csv": str(discovery_csv),
                "yes_policy": yes_policy,
                "no_policy": no_policy,
            }
            per_dataset_rows.append(row)

    avg = average_rows(per_dataset_rows)
    write_csv(out_dir / "single_c_feature_ablation.csv", per_dataset_rows)
    write_csv(out_dir / "single_c_feature_ablation_avg.csv", avg)
    write_json(
        out_dir / "single_c_feature_ablation.json",
        {
            "inputs": {
                "job_tsv": str(Path(args.job_tsv).resolve()),
                "target": str(args.target),
                "discovery_rows_csv": str(Path(args.discovery_rows_csv).resolve()) if args.discovery_rows_csv else "",
                "policy_root": str(Path(args.policy_root).resolve()) if args.policy_root else "",
                "c_features": feature_names,
                "tau_objective": str(args.tau_objective),
                "min_present_rate": float(args.min_present_rate),
                "min_selected_count": int(args.min_selected_count),
                "allow_noop_policy": bool(args.allow_noop_policy),
            },
            "per_dataset": per_dataset_rows,
            "average": avg,
        },
    )
    md = "## Average Across Datasets\n\n" + md_table(avg, average=True)
    md += "\n\n## Per Dataset\n\n" + md_table(per_dataset_rows, average=False)
    (out_dir / "single_c_feature_ablation.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print("[saved]", out_dir / "single_c_feature_ablation.md")


if __name__ == "__main__":
    main()
