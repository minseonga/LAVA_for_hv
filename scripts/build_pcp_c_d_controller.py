#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build_posthoc_b_c_fusion_controller as base


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_yes_no(text: object) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("yes"):
        return "yes"
    return ""


def add_decision_kl_features(row: Dict[str, Any]) -> None:
    candidate_p = base.maybe_float(row.get("cheap_decision_candidate_prob_binary"))
    if candidate_p is not None:
        eps = 1e-12
        p = min(1.0 - eps, max(eps, float(candidate_p)))
        q = 1.0 - p
        row["cheap_decision_candidate_kl_uniform"] = float(p * math.log(2.0 * p) + q * math.log(2.0 * q))
        row["cheap_decision_candidate_entropy"] = float(-(p * math.log(p) + q * math.log(q)))
        row["cheap_decision_candidate_conf_abs"] = float(abs(p - 0.5))

    suffix = "cheap_decision_candidate_prob_binary"
    for key, value in list(row.items()):
        if key == suffix or not key.endswith(suffix):
            continue
        prefixed_p = base.maybe_float(value)
        if prefixed_p is None:
            continue
        prefix = key[: -len(suffix)]
        eps = 1e-12
        p = min(1.0 - eps, max(eps, float(prefixed_p)))
        q = 1.0 - p
        row[f"{prefix}cheap_decision_candidate_kl_uniform"] = float(p * math.log(2.0 * p) + q * math.log(2.0 * q))
        row[f"{prefix}cheap_decision_candidate_entropy"] = float(-(p * math.log(p) + q * math.log(q)))
        row[f"{prefix}cheap_decision_candidate_conf_abs"] = float(abs(p - 0.5))


def load_rows(rows_csv: str, *, derive_decision_kl: bool) -> List[Dict[str, Any]]:
    rows = base.read_csv_rows(os.path.abspath(rows_csv))
    out: List[Dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        merged["id"] = str(row.get("id", "")).strip()
        merged["baseline_correct"] = base.maybe_int(row.get("baseline_correct"))
        merged["intervention_correct"] = base.maybe_int(row.get("intervention_correct"))
        harm = base.maybe_int(row.get("harm"))
        help_ = base.maybe_int(row.get("help"))
        if harm is None or help_ is None:
            bc = merged["baseline_correct"]
            ic = merged["intervention_correct"]
            if bc is not None and ic is not None:
                harm = int(int(bc) == 1 and int(ic) == 0)
                help_ = int(int(bc) == 0 and int(ic) == 1)
        merged["harm"] = int(harm or 0)
        merged["help"] = int(help_ or 0)
        if bool(derive_decision_kl):
            add_decision_kl_features(merged)
        out.append(merged)
    return out


def is_route_candidate(row: Dict[str, Any], candidate_filter: str) -> bool:
    mode = str(candidate_filter or "all")
    if mode == "all":
        return True
    baseline_label = str(row.get("baseline_label", "")).strip().lower()
    intervention_label = str(row.get("intervention_label", "")).strip().lower()
    if baseline_label not in {"yes", "no"}:
        baseline_label = parse_yes_no(row.get("baseline_text", ""))
    if intervention_label not in {"yes", "no"}:
        intervention_label = parse_yes_no(row.get("intervention_text", ""))
    if mode == "changed_answer":
        return baseline_label in {"yes", "no"} and intervention_label in {"yes", "no"} and baseline_label != intervention_label
    if mode == "yes_to_no":
        return baseline_label == "yes" and intervention_label == "no"
    raise ValueError(f"Unsupported candidate_filter={candidate_filter!r}")


def feature_present_count(rows: Sequence[Dict[str, Any]], feature: str) -> int:
    return sum(int(base.maybe_float(row.get(feature)) is not None) for row in rows)


def orient_feature_list(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    *,
    target: str,
    min_present_rate: float,
    min_feature_auroc: float,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n_rows = max(1, len(rows))
    metrics: List[Dict[str, Any]] = []
    for feat in feature_names:
        present_rate = float(feature_present_count(rows, feat)) / float(n_rows)
        if present_rate < float(min_present_rate):
            continue
        result = base.orient_feature(rows, feat, target=target)
        if result is None:
            continue
        result["present_rate"] = present_rate
        metrics.append(result)
    metrics.sort(key=lambda r: (-float(r["auroc"]), str(r["feature"])))
    selected = [r for r in metrics if float(r["auroc"]) >= float(min_feature_auroc)]
    if int(top_k) > 0:
        selected = selected[: int(top_k)]
    return metrics, selected


def mean_z_score(row: Dict[str, Any], features: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not features:
        return None
    vals: List[float] = []
    for feat in features:
        z = base.oriented_z(row, feat)
        if z is None:
            return None
        vals.append(float(z))
    return float(sum(vals) / float(len(vals))) if vals else None


def threshold_grid(values: Sequence[float]) -> List[float]:
    return base.threshold_grid(values)


def evaluate_policy(
    rows: Sequence[Dict[str, Any]],
    *,
    c_features: Sequence[Dict[str, Any]],
    d_features: Sequence[Dict[str, Any]],
    family: str,
    alpha: float,
    tau: float,
    candidate_filter: str = "all",
) -> Dict[str, Any]:
    n = 0
    selected = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    final_correct_total = 0
    total_harm = 0
    total_help = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0

    for row in rows:
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        if bc is None or ic is None:
            continue
        c_score = mean_z_score(row, c_features)
        d_score = mean_z_score(row, d_features)
        if family == "c_only":
            score = c_score
        elif family == "d_only":
            score = d_score
        else:
            if c_score is None or d_score is None:
                score = None
            else:
                score = float((1.0 - float(alpha)) * float(c_score) + float(alpha) * float(d_score))
        if score is None:
            continue

        harm = int(base.maybe_int(row.get("harm")) or 0)
        help_ = int(base.maybe_int(row.get("help")) or 0)
        n += 1
        total_harm += harm
        total_help += help_
        baseline_correct_total += int(bc)
        intervention_correct_total += int(ic)

        can_route = is_route_candidate(row, str(candidate_filter))
        use_baseline = bool(can_route and float(score) >= float(tau))
        if use_baseline:
            selected += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int((harm == 0) and (help_ == 0))
            final_correct_total += int(bc)
        else:
            final_correct_total += int(ic)

    baseline_rate = base.safe_div(float(selected), float(max(1, n)))
    precision = base.safe_div(float(selected_harm), float(max(1, selected)))
    recall = base.safe_div(float(selected_harm), float(max(1, total_harm)))
    f1 = base.safe_div(2.0 * precision * recall, precision + recall)
    return {
        "family": str(family),
        "alpha": float(alpha),
        "tau": float(tau),
        "n_eval": int(n),
        "baseline_rate": baseline_rate,
        "method_rate": float(1.0 - baseline_rate),
        "final_acc": base.safe_div(float(final_correct_total), float(max(1, n))),
        "baseline_acc": base.safe_div(float(baseline_correct_total), float(max(1, n))),
        "intervention_acc": base.safe_div(float(intervention_correct_total), float(max(1, n))),
        "delta_vs_intervention": base.safe_div(float(final_correct_total - intervention_correct_total), float(max(1, n))),
        "selected_count": int(selected),
        "total_harm": int(total_harm),
        "total_help": int(total_help),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "net": int(selected_harm - selected_help),
        "selected_harm_precision": precision,
        "selected_help_precision": base.safe_div(float(selected_help), float(max(1, selected))),
        "selected_harm_recall": recall,
        "selected_harm_f1": f1,
    }


def selection_key(result: Dict[str, Any], objective: str) -> Tuple[float, float, float, float]:
    if objective == "net":
        return (
            float(result["net"]),
            float(result["final_acc"]),
            float(result["selected_harm_precision"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_f1":
        return (
            float(result["selected_harm_f1"]),
            float(result["selected_harm_precision"]),
            float(result["net"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_precision":
        return (
            float(result["selected_harm_precision"]),
            float(result["selected_harm_recall"]),
            float(result["net"]),
            -float(result["baseline_rate"]),
        )
    if objective == "harm_recall":
        return (
            float(result["selected_harm_recall"]),
            float(result["selected_harm_precision"]),
            float(result["net"]),
            -float(result["baseline_rate"]),
        )
    return (
        float(result["final_acc"]),
        float(result["net"]),
        float(result["selected_harm_precision"]),
        -float(result["baseline_rate"]),
    )


def search_family(
    rows: Sequence[Dict[str, Any]],
    *,
    c_features: Sequence[Dict[str, Any]],
    d_features: Sequence[Dict[str, Any]],
    family: str,
    alpha_grid: Sequence[float],
    objective: str,
    min_baseline_rate: float,
    max_baseline_rate: float,
    min_selected_count: int,
    candidate_filter: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    if family == "c_only":
        alphas = [0.0]
    elif family == "d_only":
        alphas = [1.0]
    else:
        alphas = [float(a) for a in alpha_grid if 0.0 < float(a) < 1.0]

    for alpha in alphas:
        score_values: List[float] = []
        for row in rows:
            if not is_route_candidate(row, str(candidate_filter)):
                continue
            c_score = mean_z_score(row, c_features)
            d_score = mean_z_score(row, d_features)
            if family == "c_only":
                score = c_score
            elif family == "d_only":
                score = d_score
            else:
                if c_score is None or d_score is None:
                    score = None
                else:
                    score = float((1.0 - float(alpha)) * float(c_score) + float(alpha) * float(d_score))
            if score is not None:
                score_values.append(float(score))
        if not score_values:
            continue
        for tau in threshold_grid(score_values):
            result = evaluate_policy(
                rows,
                c_features=c_features,
                d_features=d_features,
                family=family,
                alpha=float(alpha),
                tau=float(tau),
                candidate_filter=str(candidate_filter),
            )
            candidates.append(result)
            if int(result["selected_count"]) < int(min_selected_count):
                continue
            if float(result["baseline_rate"]) < float(min_baseline_rate):
                continue
            if float(result["baseline_rate"]) > float(max_baseline_rate):
                continue
            if best is None or selection_key(result, objective) > selection_key(best, objective):
                best = result
    return best, candidates


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a PCP-only C/D controller from online feature rows.")
    ap.add_argument("--rows_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--c_feature_cols",
        type=str,
        default="cheap_target_gap_content_min,cheap_lp_content_min,cheap_lp_content_std",
    )
    ap.add_argument(
        "--d_feature_cols",
        type=str,
        default=(
            "cheap_decision_candidate_minus_alt,cheap_decision_candidate_prob_binary,"
            "cheap_decision_candidate_label_lp,cheap_decision_candidate_kl_uniform"
        ),
    )
    ap.add_argument("--derive_decision_kl", type=parse_bool, default=True)
    ap.add_argument("--min_present_rate", type=float, default=0.8)
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k_c", type=int, default=3)
    ap.add_argument("--top_k_d", type=int, default=4)
    ap.add_argument("--alpha_grid", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument(
        "--tau_objective",
        type=str,
        default="final_acc",
        choices=["final_acc", "net", "harm_precision", "harm_recall", "harm_f1"],
    )
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    ap.add_argument("--min_selected_count", type=int, default=0)
    ap.add_argument(
        "--candidate_filter",
        type=str,
        default="all",
        choices=["all", "changed_answer", "yes_to_no"],
        help=(
            "Rows eligible for fallback during calibration. changed_answer uses only "
            "samples where baseline and intervention yes/no labels differ; yes_to_no "
            "uses the subset where the intervention suppresses a baseline yes answer. "
            "Both are deployable because they use predictions, not ground truth."
        ),
    )
    args = ap.parse_args()

    rows = load_rows(os.path.abspath(args.rows_csv), derive_decision_kl=bool(args.derive_decision_kl))
    candidate_filter = str(args.candidate_filter)
    fit_rows = [row for row in rows if is_route_candidate(row, candidate_filter)]
    if not fit_rows:
        raise RuntimeError(f"No rows remain after candidate_filter={candidate_filter!r}.")
    c_feature_names = [x.strip() for x in str(args.c_feature_cols).split(",") if x.strip()]
    d_feature_names = [x.strip() for x in str(args.d_feature_cols).split(",") if x.strip()]
    alpha_grid = [float(x.strip()) for x in str(args.alpha_grid).split(",") if x.strip()]

    c_metrics, selected_c = orient_feature_list(
        fit_rows,
        c_feature_names,
        target="harm",
        min_present_rate=float(args.min_present_rate),
        min_feature_auroc=float(args.min_feature_auroc),
        top_k=int(args.top_k_c),
    )
    d_metrics, selected_d = orient_feature_list(
        fit_rows,
        d_feature_names,
        target="harm",
        min_present_rate=float(args.min_present_rate),
        min_feature_auroc=float(args.min_feature_auroc),
        top_k=int(args.top_k_d),
    )

    family_results: Dict[str, Dict[str, Any]] = {}
    sweep_rows: List[Dict[str, Any]] = []

    if selected_c:
        best, cand = search_family(
            rows,
            c_features=selected_c,
            d_features=[],
            family="c_only",
            alpha_grid=alpha_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
            candidate_filter=candidate_filter,
        )
        sweep_rows.extend(cand)
        if best is not None:
            family_results["c_only"] = best

    if selected_d:
        best, cand = search_family(
            rows,
            c_features=[],
            d_features=selected_d,
            family="d_only",
            alpha_grid=alpha_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
            candidate_filter=candidate_filter,
        )
        sweep_rows.extend(cand)
        if best is not None:
            family_results["d_only"] = best

    if selected_c and selected_d:
        best, cand = search_family(
            rows,
            c_features=selected_c,
            d_features=selected_d,
            family="cd_fusion",
            alpha_grid=alpha_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
            candidate_filter=candidate_filter,
        )
        sweep_rows.extend(cand)
        if best is not None:
            family_results["cd_fusion"] = best

    selected_policy = None
    for result in family_results.values():
        if selected_policy is None or selection_key(result, str(args.tau_objective)) > selection_key(selected_policy, str(args.tau_objective)):
            selected_policy = result

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    c_metrics_csv = os.path.join(out_dir, "c_feature_metrics.csv")
    d_metrics_csv = os.path.join(out_dir, "d_feature_metrics.csv")
    tau_sweep_csv = os.path.join(out_dir, "tau_sweep.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    policy_json = os.path.join(out_dir, "selected_policy.json")

    base.write_csv(c_metrics_csv, c_metrics)
    base.write_csv(d_metrics_csv, d_metrics)
    base.write_csv(tau_sweep_csv, sweep_rows)
    policy_obj = {
        "rows_csv": os.path.abspath(args.rows_csv),
        "candidate_filter": candidate_filter,
        "selected_c_features": selected_c,
        "selected_d_features": selected_d,
        "best_results": family_results,
        "selected_policy": selected_policy,
    }
    write_json(policy_json, policy_obj)
    write_json(
        summary_json,
        {
            "inputs": {
                "rows_csv": os.path.abspath(args.rows_csv),
                "c_feature_cols": c_feature_names,
                "d_feature_cols": d_feature_names,
                "derive_decision_kl": bool(args.derive_decision_kl),
                "min_present_rate": float(args.min_present_rate),
                "min_feature_auroc": float(args.min_feature_auroc),
                "top_k_c": int(args.top_k_c),
                "top_k_d": int(args.top_k_d),
                "alpha_grid": alpha_grid,
                "tau_objective": str(args.tau_objective),
                "min_baseline_rate": float(args.min_baseline_rate),
                "max_baseline_rate": float(args.max_baseline_rate),
                "min_selected_count": int(args.min_selected_count),
                "candidate_filter": candidate_filter,
            },
            "counts": {
                "n_rows": int(len(rows)),
                "n_harm": int(sum(int(row.get("harm", 0) or 0) for row in rows)),
                "n_help": int(sum(int(row.get("help", 0) or 0) for row in rows)),
                "n_route_candidates": int(len(fit_rows)),
                "n_route_candidate_harm": int(sum(int(row.get("harm", 0) or 0) for row in fit_rows)),
                "n_route_candidate_help": int(sum(int(row.get("help", 0) or 0) for row in fit_rows)),
            },
            "selected_c_features": selected_c,
            "selected_d_features": selected_d,
            "best_results": family_results,
            "selected_policy": selected_policy,
            "outputs": {
                "c_feature_metrics_csv": c_metrics_csv,
                "d_feature_metrics_csv": d_metrics_csv,
                "tau_sweep_csv": tau_sweep_csv,
                "selected_policy_json": policy_json,
            },
        },
    )
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
