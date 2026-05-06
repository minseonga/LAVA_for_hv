#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import build_pcp_c_d_controller as pcp
import build_transition_split_fixed_c_median_ensemble as fixed
import build_transition_split_single_c_feature_ablation as single


DIRECTIONS = ("yes_to_no", "no_to_yes")
DEFAULT_SINGLE_FEATURES = (
    "cheap_entropy_content_mean",
    "cheap_lp_content_min",
    "cheap_target_gap_content_min",
    "cheap_first_target_gap",
)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def ffloat(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def fint(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(round(float(value)))
    except Exception:
        return default


def fmt_pct(value: Any, *, signed: bool = False) -> str:
    try:
        out = 100.0 * float(value)
    except Exception:
        return ""
    return f"{out:+.2f}" if signed else f"{out:.2f}"


def fmt_num(value: Any) -> str:
    try:
        out = float(value)
    except Exception:
        return ""
    if math.isclose(out, round(out), abs_tol=1e-9):
        return str(int(round(out)))
    return f"{out:.2f}"


def fmt_tau(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return ""


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def score_summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "score_n": 0,
            "score_mean": "",
            "score_q10": "",
            "score_q25": "",
            "score_q50": "",
            "score_q75": "",
            "score_q90": "",
        }
    return {
        "score_n": len(values),
        "score_mean": mean(float(v) for v in values),
        "score_q10": quantile(values, 0.10),
        "score_q25": quantile(values, 0.25),
        "score_q50": quantile(values, 0.50),
        "score_q75": quantile(values, 0.75),
        "score_q90": quantile(values, 0.90),
    }


def policy_from_item(item: Dict[str, Any], direction: str) -> Dict[str, Any]:
    key = "yes_policy_json" if direction == "yes_to_no" else "no_policy_json"
    fallback_key = "yes_policy" if direction == "yes_to_no" else "no_policy"
    policy = item.get(key, item.get(fallback_key, {}))
    if isinstance(policy, str):
        try:
            return json.loads(policy)
        except Exception:
            return {}
    return dict(policy or {})


def is_disabled(policy: Dict[str, Any]) -> bool:
    return bool(policy.get("disabled")) or str(policy.get("family", "")) == "noop"


def route_candidate_rows(rows: Sequence[Dict[str, Any]], direction: str) -> List[Dict[str, Any]]:
    return [row for row in rows if pcp.is_route_candidate(row, direction)]


def hg_counts(rows: Sequence[Dict[str, Any]]) -> Tuple[int, int, int]:
    h = sum(int(row.get("harm", 0) or 0) for row in rows)
    g = sum(int(row.get("help", 0) or 0) for row in rows)
    return h, g, h - g


def selected_stats(
    rows: Sequence[Dict[str, Any]],
    *,
    use_baseline: Callable[[Dict[str, Any]], bool],
) -> Dict[str, Any]:
    selected_rows = [row for row in rows if use_baseline(row)]
    h, g, net = hg_counts(selected_rows)
    return {
        "fallback": len(selected_rows),
        "selected_harm": h,
        "selected_help": g,
        "selected_net": net,
    }


def report_totals(item: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    route_h, route_g, _ = hg_counts(rows)
    return {
        "n": fint(item.get("n"), len(rows)),
        "base_acc": ffloat(item.get("base"), ffloat(item.get("baseline_acc"))),
        "method_acc": ffloat(item.get("method"), ffloat(item.get("intervention_acc"))),
        "total_harm": fint(item.get("total_harm"), route_h),
        "total_help": fint(item.get("total_help"), route_g),
    }


def final_metrics(item: Dict[str, Any], rows: Sequence[Dict[str, Any]], selected: Dict[str, Any]) -> Dict[str, Any]:
    totals = report_totals(item, rows)
    selected_h = ffloat(selected.get("selected_harm"))
    selected_g = ffloat(selected.get("selected_help"))
    final_acc = totals["method_acc"] + (selected_h - selected_g) / float(max(1, totals["n"]))
    return {
        "base_acc": totals["base_acc"],
        "method_acc": totals["method_acc"],
        "final_acc": final_acc,
        "d_method": final_acc - totals["method_acc"],
        "hrec": selected_h / float(max(1, totals["total_harm"])),
        "grec": selected_g / float(max(1, totals["total_help"])),
    }


def fixed_use_baseline(row: Dict[str, Any], yes_policy: Dict[str, Any], no_policy: Dict[str, Any]) -> bool:
    yes_route, _ = fixed.compute_route(row, yes_policy, "yes_to_no")
    no_route, _ = fixed.compute_route(row, no_policy, "no_to_yes")
    return yes_route == "baseline" or no_route == "baseline"


def single_use_baseline(row: Dict[str, Any], yes_policy: Dict[str, Any], no_policy: Dict[str, Any]) -> bool:
    yes_route, _ = single.compute_route(row, yes_policy, "yes_to_no")
    no_route, _ = single.compute_route(row, no_policy, "no_to_yes")
    return yes_route == "baseline" or no_route == "baseline"


def any_changed(row: Dict[str, Any]) -> bool:
    return pcp.is_route_candidate(row, "yes_to_no") or pcp.is_route_candidate(row, "no_to_yes")


def random_expected_selector(
    rows: Sequence[Dict[str, Any]],
    *,
    fallback_count: int,
) -> Dict[str, Any]:
    candidates = [row for row in rows if any_changed(row)]
    n = len(candidates)
    if n == 0 or fallback_count <= 0:
        return {
            "fallback": 0,
            "selected_harm": 0.0,
            "selected_help": 0.0,
            "selected_net": 0.0,
        }
    selected = min(int(fallback_count), n)
    h, g, _ = hg_counts(candidates)
    exp_h = float(selected) * float(h) / float(n)
    exp_g = float(selected) * float(g) / float(n)
    return {
        "fallback": selected,
        "selected_harm": exp_h,
        "selected_help": exp_g,
        "selected_net": exp_h - exp_g,
    }


def calibrate_single_policies(
    discovery_rows: Sequence[Dict[str, Any]],
    *,
    feature: str,
    settings: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kwargs = {
        "tau_objective": str(settings.get("tau_objective", "final_acc")),
        "lambda_gain": ffloat(settings.get("lambda_gain"), 1.0),
        "min_present_rate": ffloat(settings.get("min_present_rate"), 0.8),
        "min_selected_count": fint(settings.get("min_selected_count"), 5),
        "min_harm_precision": ffloat(settings.get("min_harm_precision"), 0.0),
        "min_harm_recall": ffloat(settings.get("min_harm_recall"), 0.0),
        "max_help_recall": ffloat(settings.get("max_help_recall"), 1.0),
        "allow_noop_policy": bool(settings.get("allow_noop_policy", True)),
    }
    yes = single.calibrate_one_feature(discovery_rows, feature=feature, direction="yes_to_no", **kwargs)
    no = single.calibrate_one_feature(discovery_rows, feature=feature, direction="no_to_yes", **kwargs)
    return yes, no


def fixed_transition_rows(item: Dict[str, Any], discovery_rows: Sequence[Dict[str, Any]], apply_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dataset = str(item.get("dataset", ""))
    for direction in DIRECTIONS:
        policy = policy_from_item(item, direction)
        discovery_candidates = route_candidate_rows(discovery_rows, direction)
        apply_candidates = route_candidate_rows(apply_rows, direction)
        disc_h, disc_g, disc_net = hg_counts(discovery_candidates)
        app_h, app_g, app_net = hg_counts(apply_candidates)

        features = list(policy.get("selected_c_features") or [])
        values = fixed.median_score_values(discovery_rows, features, direction) if features else []
        dist = score_summary(values)

        def selected_in_direction(row: Dict[str, Any]) -> bool:
            route, _ = fixed.compute_route(row, policy, direction)
            return route == "baseline"

        disc_sel = selected_stats(discovery_candidates, use_baseline=selected_in_direction)
        app_sel = selected_stats(apply_candidates, use_baseline=selected_in_direction)
        out.append(
            {
                "dataset": dataset,
                "direction": direction,
                "policy": fixed.policy_short(policy),
                "disabled": int(is_disabled(policy)),
                "tau": "" if is_disabled(policy) else policy.get("tau", ""),
                "disc_candidates": len(discovery_candidates),
                "disc_harm": disc_h,
                "disc_help": disc_g,
                "disc_net": disc_net,
                **dist,
                "disc_selected": disc_sel["fallback"],
                "disc_sel_harm": disc_sel["selected_harm"],
                "disc_sel_help": disc_sel["selected_help"],
                "disc_sel_net": disc_sel["selected_net"],
                "apply_candidates": len(apply_candidates),
                "apply_harm": app_h,
                "apply_help": app_g,
                "apply_net": app_net,
                "apply_selected": app_sel["fallback"],
                "apply_sel_harm": app_sel["selected_harm"],
                "apply_sel_help": app_sel["selected_help"],
                "apply_sel_net": app_sel["selected_net"],
                "apply_selected_rate": ffloat(app_sel["fallback"]) / float(max(1, len(apply_candidates))),
                "apply_hrec_in_direction": ffloat(app_sel["selected_harm"]) / float(max(1, app_h)),
                "apply_grec_in_direction": ffloat(app_sel["selected_help"]) / float(max(1, app_g)),
            }
        )
    return out


def selector_rows_for_item(
    item: Dict[str, Any],
    *,
    discovery_rows: Sequence[Dict[str, Any]],
    apply_rows: Sequence[Dict[str, Any]],
    single_features: Sequence[str],
    settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dataset = str(item.get("dataset", ""))
    yes_fixed = policy_from_item(item, "yes_to_no")
    no_fixed = policy_from_item(item, "no_to_yes")

    def add_selector(selector: str, selected: Dict[str, Any], note: str = "") -> None:
        metrics = final_metrics(item, apply_rows, selected)
        out.append(
            {
                "dataset": dataset,
                "selector": selector,
                "base_acc": metrics["base_acc"],
                "method_acc": metrics["method_acc"],
                "final_acc": metrics["final_acc"],
                "d_method": metrics["d_method"],
                "fallback": selected["fallback"],
                "selected_harm": selected["selected_harm"],
                "selected_help": selected["selected_help"],
                "selected_net": selected["selected_net"],
                "hrec": metrics["hrec"],
                "grec": metrics["grec"],
                "note": note,
            }
        )

    fixed_selected = selected_stats(
        apply_rows,
        use_baseline=lambda row: fixed_use_baseline(row, yes_fixed, no_fixed),
    )
    add_selector("fixed3_median", fixed_selected, "fixed replay-C median")

    for feature in single_features:
        yes_single, no_single = calibrate_single_policies(discovery_rows, feature=feature, settings=settings)
        selected = selected_stats(
            apply_rows,
            use_baseline=lambda row, y=yes_single, n=no_single: single_use_baseline(row, y, n),
        )
        add_selector(f"single:{feature}", selected, f"Y:{single.policy_short(yes_single)} / N:{single.policy_short(no_single)}")

    random_selected = random_expected_selector(apply_rows, fallback_count=fint(fixed_selected["fallback"]))
    add_selector("random@fixed3_count", random_selected, "expected random fallback at fixed3 budget")

    always_selected = selected_stats(apply_rows, use_baseline=any_changed)
    add_selector("always_rollback", always_selected, "fallback all changed answers")
    return out


def md_transition_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Direction | Policy | Tau | Disc H/G/Net | Score q10/q50/q90 | Apply H/G/Net | Fallback | Sel H/G/Net | Sel% | Hrec | Grec |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['direction']} | {row['policy']} | {fmt_tau(row['tau'])} | "
            f"{fmt_num(row['disc_harm'])}/{fmt_num(row['disc_help'])}/{fmt_num(row['disc_net'])} | "
            f"{fmt_num(row['score_q10'])}/{fmt_num(row['score_q50'])}/{fmt_num(row['score_q90'])} | "
            f"{fmt_num(row['apply_harm'])}/{fmt_num(row['apply_help'])}/{fmt_num(row['apply_net'])} | "
            f"{fmt_num(row['apply_selected'])} | "
            f"{fmt_num(row['apply_sel_harm'])}/{fmt_num(row['apply_sel_help'])}/{fmt_num(row['apply_sel_net'])} | "
            f"{fmt_pct(row['apply_selected_rate'])} | {fmt_pct(row['apply_hrec_in_direction'])} | "
            f"{fmt_pct(row['apply_grec_in_direction'])} |"
        )
    return "\n".join(lines)


def md_selector_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Dataset | Selector | Base | Method | Final | dMethod | Fallback | Sel H/G/Net | Hrec | Grec | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    order = {"mscoco": 0, "aokvqa": 1, "gqa": 2}
    sorted_rows = sorted(rows, key=lambda r: (order.get(str(r["dataset"]), 99), -ffloat(r["final_acc"]), str(r["selector"])))
    for row in sorted_rows:
        lines.append(
            f"| {row['dataset']} | {row['selector']} | {fmt_pct(row['base_acc'])} | "
            f"{fmt_pct(row['method_acc'])} | {fmt_pct(row['final_acc'])} | "
            f"{fmt_pct(row['d_method'], signed=True)} | {fmt_num(row['fallback'])} | "
            f"{fmt_num(row['selected_harm'])}/{fmt_num(row['selected_help'])}/{fmt_num(row['selected_net'])} | "
            f"{fmt_pct(row['hrec'])} | {fmt_pct(row['grec'])} | {row.get('note', '')} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate transition-split calibration diagnostics and selector baselines from fixed-C median output."
    )
    ap.add_argument("--fixed_json", required=True, help="Path to fixed_c_median_ensemble.json.")
    ap.add_argument("--target", default="vga_llava15")
    ap.add_argument("--out_dir", default="", help="Defaults to fixed_json parent.")
    ap.add_argument("--single_features", default=",".join(DEFAULT_SINGLE_FEATURES))
    ap.add_argument("--derive_decision_kl", type=pcp.parse_bool, default=True)
    args = ap.parse_args()

    fixed_json = Path(args.fixed_json).resolve()
    bundle = read_json(fixed_json)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else fixed_json.parent
    target = str(args.target)
    single_features = [x.strip() for x in str(args.single_features).split(",") if x.strip()]
    settings = dict(bundle.get("inputs") or {})

    items = [dict(row) for row in bundle.get("per_dataset", []) if str(row.get("target", "")) == target]
    if not items:
        raise RuntimeError(f"No rows for target={target!r} in {fixed_json}")

    rows_cache: Dict[str, List[Dict[str, Any]]] = {}

    def load_cached(path_text: str) -> List[Dict[str, Any]]:
        path = str(Path(path_text).resolve())
        if path not in rows_cache:
            rows_cache[path] = pcp.load_rows(path, derive_decision_kl=bool(args.derive_decision_kl))
        return rows_cache[path]

    transition_rows: List[Dict[str, Any]] = []
    selector_rows: List[Dict[str, Any]] = []
    for item in items:
        apply_path = str(item.get("apply_rows_csv") or item.get("rows_csv") or "")
        discovery_path = str(item.get("discovery_rows_csv") or "")
        if not apply_path or not discovery_path:
            raise RuntimeError(f"Missing rows paths for target={target} dataset={item.get('dataset')}")
        apply_rows = load_cached(apply_path)
        discovery_rows = load_cached(discovery_path)
        transition_rows.extend(fixed_transition_rows(item, discovery_rows, apply_rows))
        selector_rows.extend(
            selector_rows_for_item(
                item,
                discovery_rows=discovery_rows,
                apply_rows=apply_rows,
                single_features=single_features,
                settings=settings,
            )
        )

    prefix = target.replace("/", "_")
    transition_csv = out_dir / f"{prefix}_transition_calibration_stats.csv"
    transition_md = out_dir / f"{prefix}_transition_calibration_stats.md"
    selector_csv = out_dir / f"{prefix}_direct_selector_comparison.csv"
    selector_md = out_dir / f"{prefix}_direct_selector_comparison.md"
    write_csv(transition_csv, transition_rows)
    write_csv(selector_csv, selector_rows)
    transition_md.write_text(md_transition_table(transition_rows) + "\n", encoding="utf-8")
    selector_md.write_text(md_selector_table(selector_rows) + "\n", encoding="utf-8")
    print("[saved]", transition_md)
    print("[saved]", selector_md)


if __name__ == "__main__":
    main()
