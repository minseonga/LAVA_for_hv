#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import build_pcp_c_d_controller as pcp
import build_posthoc_b_c_fusion_controller as base
import run_transition_split_calibration_from_existing_features as split_calib


DIRECTIONS = ("yes_to_no", "no_to_yes")
SELECTORS = ("answer_logprob", "answer_prob", "yesno_margin", "yesno_entropy", "first_token_entropy")

CONFIDENCE_SPECS: Dict[str, Dict[str, str]] = {
    "answer_logprob": {
        "feature": "cheap_decision_candidate_label_lp",
        "risk_when": "low",
        "label": "Answer log-prob",
    },
    "answer_prob": {
        "feature": "cheap_decision_candidate_prob_binary",
        "risk_when": "low",
        "label": "Answer yes/no prob",
    },
    "yesno_margin": {
        "feature": "cheap_decision_margin_abs",
        "risk_when": "low",
        "label": "Yes/no margin",
    },
    "yesno_entropy": {
        "feature": "cheap_decision_candidate_entropy",
        "risk_when": "high",
        "label": "Yes/no entropy",
    },
    "first_token_entropy": {
        "feature": "cheap_first_entropy",
        "risk_when": "high",
        "label": "First-token entropy",
    },
}


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


def maybe_float(value: object) -> Optional[float]:
    x = base.maybe_float(value)
    if x is None or not math.isfinite(float(x)):
        return None
    return float(x)


def maybe_int(value: object) -> int:
    x = base.maybe_int(value)
    return int(x or 0)


def risk_value(row: Dict[str, Any], spec: Dict[str, str]) -> Optional[float]:
    raw = maybe_float(row.get(spec["feature"]))
    if raw is None:
        return None
    return -float(raw) if spec["risk_when"] == "low" else float(raw)


def condition_label(policy: Dict[str, Any]) -> str:
    if policy.get("disabled"):
        return "noop"
    spec = CONFIDENCE_SPECS.get(str(policy.get("selector", "")), {})
    feature = str(spec.get("feature", policy.get("feature", "")))
    tau = float(policy.get("tau", 0.0) or 0.0)
    risk_when = str(spec.get("risk_when", policy.get("risk_when", "")))
    if risk_when == "low":
        return f"{policy.get('selector')}:{feature}<={-tau:.3f}"
    return f"{policy.get('selector')}:{feature}>={tau:.3f}"


def route_row_from_feature(
    row: Dict[str, Any],
    *,
    route: str,
    selector: str,
    direction: str,
    score: Any = "",
    tau: Any = "",
) -> Dict[str, Any]:
    baseline_label = str(row.get("baseline_label", "")).strip().lower()
    intervention_label = str(row.get("intervention_label", "")).strip().lower()
    if baseline_label not in {"yes", "no"}:
        baseline_label = pcp.parse_yes_no(row.get("baseline_text", ""))
    if intervention_label not in {"yes", "no"}:
        intervention_label = pcp.parse_yes_no(row.get("intervention_text", ""))
    final_label = baseline_label if route == "baseline" else intervention_label
    return {
        "id": str(row.get("id", "")),
        "image": str(row.get("image", "")),
        "question": str(row.get("question", "")),
        "gt_label": str(row.get("gt_label", "")).strip().lower(),
        "baseline_label": baseline_label,
        "intervention_label": intervention_label,
        "final_label": final_label,
        "route": route,
        "selector": selector,
        "score": score,
        "tau": tau,
        "route_candidate": int(pcp.is_route_candidate(row, "changed_answer")),
        "route_policy_direction": direction if route == "baseline" else "method",
        "harm": maybe_int(row.get("harm")),
        "help": maybe_int(row.get("help")),
        "baseline_correct": row.get("baseline_correct"),
        "intervention_correct": row.get("intervention_correct"),
        "final_source": "baseline_cached" if route == "baseline" else "method",
        "final_text": str(row.get("baseline_text", "")) if route == "baseline" else str(row.get("intervention_text", "")),
    }


def evaluate_scalar_policy(
    rows: Sequence[Dict[str, Any]],
    *,
    selector: str,
    tau: float,
    candidate_filter: str,
) -> Dict[str, Any]:
    spec = CONFIDENCE_SPECS[selector]
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
        harm = maybe_int(row.get("harm"))
        help_ = maybe_int(row.get("help"))
        n += 1
        total_harm += harm
        total_help += help_
        baseline_correct_total += int(bc)
        intervention_correct_total += int(ic)

        can_route = pcp.is_route_candidate(row, candidate_filter)
        score = risk_value(row, spec) if can_route else None
        if can_route:
            n_route_candidates += 1
            route_candidate_harm += harm
            route_candidate_help += help_
            route_candidate_neutral += int((harm == 0) and (help_ == 0))

        use_baseline = bool(can_route and score is not None and float(score) >= float(tau))
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
        "selector": selector,
        "selector_label": CONFIDENCE_SPECS[selector]["label"],
        "feature": spec["feature"],
        "risk_when": spec["risk_when"],
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


def noop_policy(rows: Sequence[Dict[str, Any]], candidate_filter: str) -> Dict[str, Any]:
    result = pcp.evaluate_noop_policy(rows)
    candidates = [row for row in rows if pcp.is_route_candidate(row, candidate_filter)]
    result.update(
        {
            "selector": "noop",
            "selector_label": "NOOP",
            "feature": "",
            "risk_when": "",
            "tau": 0.0,
            "disabled": True,
            "n_route_candidates": len(candidates),
            "n_route_candidate_harm": sum(maybe_int(row.get("harm")) for row in candidates),
            "n_route_candidate_help": sum(maybe_int(row.get("help")) for row in candidates),
        }
    )
    return result


def calibrate_direction(
    rows: Sequence[Dict[str, Any]],
    *,
    candidate_filter: str,
    selectors: Sequence[str],
    objective: str,
    lambda_gain: float,
    min_selected_count: int,
    min_harm_precision: float,
    min_harm_recall: float,
    max_help_recall: float,
    allow_noop_policy: bool,
) -> Dict[str, Any]:
    best_results: Dict[str, Dict[str, Any]] = {}
    all_candidates: List[Dict[str, Any]] = []
    selected_best: Optional[Dict[str, Any]] = None

    for selector in selectors:
        spec = CONFIDENCE_SPECS[selector]
        values = [
            risk_value(row, spec)
            for row in rows
            if pcp.is_route_candidate(row, candidate_filter)
        ]
        values = [float(x) for x in values if x is not None]
        if not values:
            continue
        selector_best: Optional[Dict[str, Any]] = None
        for tau in pcp.threshold_grid(values):
            result = evaluate_scalar_policy(
                rows,
                selector=selector,
                tau=float(tau),
                candidate_filter=candidate_filter,
            )
            all_candidates.append(result)
            if int(result["selected_count"]) < int(min_selected_count):
                continue
            if float(result["selected_harm_precision"]) < float(min_harm_precision):
                continue
            if float(result["selected_harm_recall"]) < float(min_harm_recall):
                continue
            if float(result["selected_help_recall"]) > float(max_help_recall):
                continue
            if selector_best is None or pcp.selection_key(result, objective, lambda_gain) > pcp.selection_key(
                selector_best, objective, lambda_gain
            ):
                selector_best = result
        if selector_best is not None:
            best_results[selector] = selector_best
            if selected_best is None or pcp.selection_key(selector_best, objective, lambda_gain) > pcp.selection_key(
                selected_best, objective, lambda_gain
            ):
                selected_best = selector_best

    if selected_best is None:
        selected_best = noop_policy(rows, candidate_filter)
    elif allow_noop_policy:
        noop = noop_policy(rows, candidate_filter)
        if pcp.selection_key(noop, objective, lambda_gain) > pcp.selection_key(selected_best, objective, lambda_gain):
            selected_best = noop

    return {
        "candidate_filter": candidate_filter,
        "selectors": list(selectors),
        "best_results": best_results,
        "selected_policy": selected_best,
        "n_candidates_evaluated": len(all_candidates),
    }


def load_policy(path: Path) -> Dict[str, Any]:
    return read_json(path)


def select_policy(bundle: Dict[str, Any], selector: str) -> Dict[str, Any]:
    if selector == "best_confidence":
        return dict(bundle.get("selected_policy") or {})
    best = bundle.get("best_results") or {}
    if selector in best:
        return dict(best[selector])
    return dict(bundle.get("noop_policy") or {"disabled": True, "selector": "noop", "tau": 0.0})


def build_routes(
    rows: Sequence[Dict[str, Any]],
    *,
    yes_bundle: Dict[str, Any],
    no_bundle: Dict[str, Any],
    selector: str,
) -> List[Dict[str, Any]]:
    yes_policy = select_policy(yes_bundle, selector)
    no_policy = select_policy(no_bundle, selector)
    out: List[Dict[str, Any]] = []
    for row in rows:
        route = "method"
        chosen = None
        direction = "method"
        for candidate_filter, policy in (("yes_to_no", yes_policy), ("no_to_yes", no_policy)):
            if policy.get("disabled") or not pcp.is_route_candidate(row, candidate_filter):
                continue
            spec = CONFIDENCE_SPECS.get(str(policy.get("selector", "")))
            if not spec:
                continue
            score = risk_value(row, spec)
            if score is not None and float(score) >= float(policy.get("tau", 0.0) or 0.0):
                route = "baseline"
                chosen = (policy, score)
                direction = candidate_filter
                break
        if chosen is None:
            row_out = route_row_from_feature(row, route=route, selector=selector, direction=direction)
        else:
            policy, score = chosen
            row_out = route_row_from_feature(
                row,
                route=route,
                selector=str(policy.get("selector", selector)),
                direction=direction,
                score=score,
                tau=policy.get("tau", ""),
            )
        row_out["yes_policy"] = condition_label(yes_policy)
        row_out["no_policy"] = condition_label(no_policy)
        out.append(row_out)
    return out


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


def format_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Selector | Method / Backbone | Dataset | Policies | Base | Method | Conf-RaPiC | dMethod | Fallback | H/G/Net | Hrec | Grec |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        s = row["summary"]
        h = float(s.get("selected_harm", 0) or 0)
        g = float(s.get("selected_help", 0) or 0)
        th = float(s.get("total_harm", 0) or 0)
        tg = float(s.get("total_help", 0) or 0)
        lines.append(
            f"| {row['selector_label']} | {row['label']} | {row['dataset']} | {row['policies']} | "
            f"{fmt_pct(s.get('baseline_acc'))} | {fmt_pct(s.get('intervention_acc'))} | "
            f"{fmt_pct(s.get('pcp_deploy_acc'))} | {fmt_pct(s.get('delta_vs_intervention'), signed=True)} | "
            f"{fmt_num(s.get('baseline_generated'))} | {fmt_num(h)}/{fmt_num(g)}/{fmt_num(h - g)} | "
            f"{fmt_pct(h / th if th else 0.0)} | {fmt_pct(g / tg if tg else 0.0)} |"
        )
    return "\n".join(lines)


def discover_jobs(apply_roots: Iterable[Path], targets: Optional[Sequence[str]], datasets: Optional[Sequence[str]]) -> List[Tuple[str, str, Path]]:
    jobs = split_calib.discover_apply_jobs(apply_roots)
    if targets:
        target_set = set(targets)
        jobs = [job for job in jobs if job[0] in target_set]
    if datasets:
        dataset_set = set(datasets)
        jobs = [job for job in jobs if job[1] in dataset_set]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build direction-split scalar confidence fallback baselines from existing RAPIC feature rows. "
            "These are replay-scored confidence baselines, not generation-time no-replay logits."
        )
    )
    ap.add_argument("--apply_root", action="append", default=None)
    ap.add_argument("--target", action="append", default=None)
    ap.add_argument("--dataset", action="append", choices=split_calib.DATASETS, default=None)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--selector", action="append", choices=(*SELECTORS, "best_confidence"), default=None)
    ap.add_argument(
        "--tau_objective",
        default="final_acc",
        choices=["final_acc", "net", "harm_precision", "harm_recall", "harm_f1", "gain_preserving_harm_recall"],
    )
    ap.add_argument("--lambda_gain", type=float, default=1.0)
    ap.add_argument("--min_selected_count", type=int, default=5)
    ap.add_argument("--min_harm_precision", type=float, default=0.0)
    ap.add_argument("--min_harm_recall", type=float, default=0.0)
    ap.add_argument("--max_help_recall", type=float, default=1.0)
    ap.add_argument("--allow_noop_policy", type=pcp.parse_bool, default=True)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    apply_roots = [Path(p).resolve() for p in (args.apply_root or [])]
    if not apply_roots:
        raise SystemExit("--apply_root is required")
    selectors = list(args.selector or ["best_confidence", *SELECTORS])
    calibration_selectors = [s for s in SELECTORS if s in selectors or "best_confidence" in selectors]

    jobs = discover_jobs(apply_roots, args.target, args.dataset)
    if not jobs:
        raise SystemExit(f"No apply jobs found under: {', '.join(str(p) for p in apply_roots)}")

    built = set()
    table_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for target, dataset, apply_dir in jobs:
        label = split_calib.LABELS.get(target, target)
        rows_csv, _policy_json, policy_summary = split_calib.policy_paths_from_apply(apply_dir)
        policy_inputs = read_json(policy_summary).get("inputs") or {}
        cal_rows_csv = Path(str(policy_inputs.get("rows_csv", ""))).resolve()
        if not cal_rows_csv.exists():
            raise FileNotFoundError(f"discovery rows_csv missing for {target}: {cal_rows_csv}")

        policy_root = out_root / "policies" / target
        if target not in built:
            discovery_rows = pcp.load_rows(str(cal_rows_csv), derive_decision_kl=True)
            for direction in DIRECTIONS:
                bundle = calibrate_direction(
                    discovery_rows,
                    candidate_filter=direction,
                    selectors=calibration_selectors,
                    objective=str(args.tau_objective),
                    lambda_gain=float(args.lambda_gain),
                    min_selected_count=int(args.min_selected_count),
                    min_harm_precision=float(args.min_harm_precision),
                    min_harm_recall=float(args.min_harm_recall),
                    max_help_recall=float(args.max_help_recall),
                    allow_noop_policy=bool(args.allow_noop_policy),
                )
                bundle["rows_csv"] = str(cal_rows_csv)
                bundle["tau_objective"] = str(args.tau_objective)
                bundle["noop_policy"] = noop_policy(discovery_rows, direction)
                write_json(policy_root / direction / "selected_policy.json", bundle)
            built.add(target)

        test_rows = pcp.load_rows(str(rows_csv), derive_decision_kl=True)
        old_deploy = read_json(apply_dir / "deployment_summary.json")
        yes_bundle = load_policy(policy_root / "yes_to_no" / "selected_policy.json")
        no_bundle = load_policy(policy_root / "no_to_yes" / "selected_policy.json")

        for selector in selectors:
            route_rows = build_routes(
                test_rows,
                yes_bundle=yes_bundle,
                no_bundle=no_bundle,
                selector=selector,
            )
            selector_dir = out_root / "apply" / target / dataset / selector
            route_csv = selector_dir / "pcp_route_rows.csv"
            write_csv(route_csv, route_rows)
            summary = split_calib.summarize_from_existing_deployment(
                old_deploy=old_deploy,
                route_rows_csv=route_csv,
            )
            write_json(selector_dir / "deployment_summary.json", summary)
            yes_policy = select_policy(yes_bundle, selector)
            no_policy = select_policy(no_bundle, selector)
            policies = f"Y:{condition_label(yes_policy)} / N:{condition_label(no_policy)}"
            row = {
                "selector": selector,
                "selector_label": "Best scalar confidence" if selector == "best_confidence" else CONFIDENCE_SPECS[selector]["label"],
                "target": target,
                "label": label,
                "dataset": dataset,
                "policies": policies,
                "summary": summary,
            }
            table_rows.append(row)
            csv_rows.append(
                {
                    "selector": selector,
                    "selector_label": row["selector_label"],
                    "target": target,
                    "label": label,
                    "dataset": dataset,
                    "policies": policies,
                    "source_apply_dir": str(apply_dir),
                    "apply_rows_csv": str(rows_csv),
                    "discovery_rows_csv": str(cal_rows_csv),
                    "baseline_acc": summary.get("baseline_acc"),
                    "intervention_acc": summary.get("intervention_acc"),
                    "pcp_deploy_acc": summary.get("pcp_deploy_acc"),
                    "delta_vs_intervention": summary.get("delta_vs_intervention"),
                    "fallback": summary.get("baseline_generated"),
                    "selected_harm": summary.get("selected_harm"),
                    "selected_help": summary.get("selected_help"),
                    "net": summary.get("net"),
                    "hrec": (
                        float(summary.get("selected_harm", 0) or 0) / float(summary.get("total_harm", 0) or 1)
                    ),
                    "grec": (
                        float(summary.get("selected_help", 0) or 0) / float(summary.get("total_help", 0) or 1)
                    ),
                }
            )

    write_csv(out_root / "confidence_baseline_summary.csv", csv_rows)
    md = format_summary_table(table_rows)
    (out_root / "confidence_baseline_summary.md").write_text(md + "\n", encoding="utf-8")
    selected_rows = [row for row in table_rows if row["selector"] == "best_confidence"]
    selected_md = format_summary_table(selected_rows)
    (out_root / "best_confidence_baseline_summary.md").write_text(selected_md + "\n", encoding="utf-8")
    print(md)
    print("[saved]", out_root / "confidence_baseline_summary.md")
    print("[saved]", out_root / "best_confidence_baseline_summary.md")


if __name__ == "__main__":
    main()
