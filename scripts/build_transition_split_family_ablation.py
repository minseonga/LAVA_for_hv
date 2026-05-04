#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import apply_pcp_c_d_controller as apply_pcp
import build_pcp_c_d_controller as pcp
import build_posthoc_b_c_fusion_controller as base
import run_transition_split_calibration_from_existing_features as split_calib


VARIANTS = (
    ("noop", "NOOP"),
    ("always", "Always rollback"),
    ("random", "Random fallback"),
    ("c_only", "C-only both directions"),
    ("d_only", "D-only both directions"),
    ("cd_fusion", "Fusion-only both directions"),
    ("selected", "RAPIC selected policy"),
)


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


def rows_csv_from_apply_dir(source_apply_dir: Path) -> Path:
    summary = read_json(source_apply_dir / "summary.json")
    rows_csv = Path(str((summary.get("inputs") or {}).get("rows_csv", "")))
    if not rows_csv.exists():
        raise FileNotFoundError(f"rows_csv missing from {source_apply_dir}/summary.json: {rows_csv}")
    return rows_csv


def policy_short(policy: Dict[str, Any]) -> str:
    family = str(policy.get("family", ""))
    if family == "noop" or policy.get("disabled"):
        return "noop"
    return f"{family}@{float(policy.get('tau', 0.0) or 0.0):.3f}"


def variant_policy_label(policy_root: Path, family: str) -> str:
    if family == "noop":
        return "noop"
    if family == "always":
        return "all changed candidates"
    if family == "random":
        return "random matched count"
    yes_bundle = read_json(policy_root / "yes_to_no" / "selected_policy.json")
    no_bundle = read_json(policy_root / "no_to_yes" / "selected_policy.json")
    yes_policy = apply_pcp.choose_policy(yes_bundle, family)
    no_policy = apply_pcp.choose_policy(no_bundle, family)
    return f"Y:{policy_short(yes_policy)} / N:{policy_short(no_policy)}"


def policy_provenance(policy_root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for direction in ("yes_to_no", "no_to_yes"):
        path = policy_root / direction / "selected_policy.json"
        if not path.exists():
            continue
        bundle = read_json(path)
        out[f"{direction}_policy_json"] = str(path)
        out[f"{direction}_rows_csv"] = str(bundle.get("rows_csv", ""))
        if "testcalib" in str(path).lower() or "testcalib" in str(bundle.get("rows_csv", "")).lower():
            out[f"{direction}_warning"] = "policy path looks like a test-calibration root"
    return out


def parse_label(row: Dict[str, Any], key: str, text_key: str) -> str:
    label = str(row.get(key, "")).strip().lower()
    if label in {"yes", "no"}:
        return label
    return pcp.parse_yes_no(row.get(text_key, ""))


def route_row_from_feature(row: Dict[str, Any], route: str, *, family: str, score: Any = "") -> Dict[str, Any]:
    baseline_label = parse_label(row, "baseline_label", "baseline_text")
    intervention_label = parse_label(row, "intervention_label", "intervention_text")
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
        "family": family,
        "alpha": "",
        "tau": "",
        "score": score,
        "route_candidate": int(pcp.is_route_candidate(row, "changed_answer")),
        "route_policy_direction": "forced" if route == "baseline" else "method",
        "harm": int(base.maybe_int(row.get("harm")) or 0),
        "help": int(base.maybe_int(row.get("help")) or 0),
        "baseline_correct": row.get("baseline_correct"),
        "intervention_correct": row.get("intervention_correct"),
        "final_source": "baseline_cached" if route == "baseline" else "method",
        "final_text": str(row.get("baseline_text", "")) if route == "baseline" else str(row.get("intervention_text", "")),
    }


def apply_direction_family(
    rows: Sequence[Dict[str, Any]],
    *,
    policy_json: Path,
    family: str,
    direction: str,
) -> Dict[str, Dict[str, Any]]:
    bundle = read_json(policy_json)
    policy = apply_pcp.choose_policy(bundle, family)
    c_features = list(bundle.get("selected_c_features") or [])
    d_features = list(bundle.get("selected_d_features") or [])
    applied_family = str(policy.get("family", "noop"))
    disabled = bool(policy.get("disabled")) or applied_family == "noop"
    alpha = float(policy.get("alpha", 0.0) or 0.0)
    tau = float(policy.get("tau", 0.0) or 0.0)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("id", "")).strip()
        can_route = pcp.is_route_candidate(row, direction)
        score = apply_pcp.compute_score(
            row,
            c_features=c_features,
            d_features=d_features,
            family=applied_family,
            alpha=alpha,
        )
        route = "method"
        if not disabled and can_route and score is not None and float(score) >= tau:
            route = "baseline"
        out[sid] = {
            "route": route,
            "score": score,
            "family": applied_family,
            "alpha": alpha,
            "tau": tau,
            "direction": direction,
        }
    return out


def build_family_routes(
    rows: Sequence[Dict[str, Any]],
    *,
    policy_root: Path,
    family: str,
) -> List[Dict[str, Any]]:
    yes = apply_direction_family(
        rows,
        policy_json=policy_root / "yes_to_no" / "selected_policy.json",
        family=family,
        direction="yes_to_no",
    )
    no = apply_direction_family(
        rows,
        policy_json=policy_root / "no_to_yes" / "selected_policy.json",
        family=family,
        direction="no_to_yes",
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("id", "")).strip()
        yr = yes.get(sid, {})
        nr = no.get(sid, {})
        route = "method"
        chosen = {}
        policy_direction = "method"
        if yr.get("route") == "baseline":
            route = "baseline"
            chosen = yr
            policy_direction = "yes_to_no"
        elif nr.get("route") == "baseline":
            route = "baseline"
            chosen = nr
            policy_direction = "no_to_yes"
        route_row = route_row_from_feature(row, route, family=str(chosen.get("family", family)), score=chosen.get("score", ""))
        route_row.update(
            {
                "alpha": chosen.get("alpha", ""),
                "tau": chosen.get("tau", ""),
                "route_yes_to_no": yr.get("route", ""),
                "route_no_to_yes": nr.get("route", ""),
                "route_policy_direction": policy_direction,
            }
        )
        out.append(route_row)
    return out


def build_forced_routes(rows: Sequence[Dict[str, Any]], *, mode: str, rng: Optional[random.Random], k: int) -> List[Dict[str, Any]]:
    selected_ids = set()
    if mode == "always":
        selected_ids = {str(row.get("id", "")).strip() for row in rows if pcp.is_route_candidate(row, "changed_answer")}
    elif mode == "random":
        candidates = [str(row.get("id", "")).strip() for row in rows if pcp.is_route_candidate(row, "changed_answer")]
        if rng is None:
            raise RuntimeError("random mode requires rng")
        selected_ids = set(rng.sample(candidates, min(int(k), len(candidates))))
    elif mode == "noop":
        selected_ids = set()
    else:
        raise ValueError(mode)
    out = []
    for row in rows:
        sid = str(row.get("id", "")).strip()
        route = "baseline" if sid in selected_ids else "method"
        out.append(route_row_from_feature(row, route, family=mode))
    return out


def summarize_variant(
    *,
    old_deploy: Dict[str, Any],
    route_rows: Sequence[Dict[str, Any]],
    out_csv: Path,
) -> Dict[str, Any]:
    write_csv(out_csv, route_rows)
    summary = split_calib.summarize_from_existing_deployment(
        old_deploy=old_deploy,
        route_rows_csv=out_csv,
    )
    total_harm = sum(int(base.maybe_int(row.get("harm")) or 0) for row in route_rows)
    total_help = sum(int(base.maybe_int(row.get("help")) or 0) for row in route_rows)
    selected = [row for row in route_rows if str(row.get("route", "")).strip() == "baseline"]
    selected_harm = sum(int(base.maybe_int(row.get("harm")) or 0) for row in selected)
    selected_help = sum(int(base.maybe_int(row.get("help")) or 0) for row in selected)
    selected_neutral = max(0, len(selected) - selected_harm - selected_help)

    n = int(old_deploy.get("n", 0) or 0)
    baseline_acc = float(old_deploy.get("baseline_acc", 0.0) or 0.0)
    intervention_acc_from_rows = baseline_acc + float(total_help - total_harm) / float(n) if n else 0.0
    pcp_acc_from_rows = intervention_acc_from_rows + float(selected_harm - selected_help) / float(n) if n else 0.0

    stale_intervention_acc = abs(
        float(old_deploy.get("intervention_acc", intervention_acc_from_rows) or 0.0) - intervention_acc_from_rows
    ) > 1e-9
    summary.update(
        {
            "baseline_acc": baseline_acc,
            "intervention_acc": intervention_acc_from_rows,
            "pcp_deploy_acc": pcp_acc_from_rows,
            "delta_vs_intervention": pcp_acc_from_rows - intervention_acc_from_rows,
            "total_harm": int(total_harm),
            "total_help": int(total_help),
            "selected_harm": int(selected_harm),
            "selected_help": int(selected_help),
            "selected_neutral": int(selected_neutral),
            "baseline_generated": int(len(selected)),
            "actual_fallback": int(len(selected)),
            "flagged_unchanged": int(selected_neutral),
            "net": int(selected_harm - selected_help),
        }
    )
    if stale_intervention_acc:
        summary["source_summary_warning"] = (
            "source deployment_summary intervention_acc does not match rows_csv H/G totals; "
            "accuracy and H/G metrics were recomputed from rows_csv totals and baseline_acc."
        )
        for key in (
            "intervention_f1",
            "pcp_deploy_f1",
            "delta_f1_vs_intervention",
            "delta_f1_vs_baseline",
        ):
            summary[key] = None
    return summary


def metric_value(summary: Dict[str, Any], key: str) -> float:
    return float(summary.get(key, 0.0) or 0.0)


def aggregate_random(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        return {}
    keys = [
        "pcp_deploy_acc",
        "delta_vs_intervention",
        "delta_f1_vs_intervention",
        "baseline_generated",
        "selected_harm",
        "selected_help",
        "net",
    ]
    out = dict(summaries[0])
    out["random_repeats"] = len(summaries)
    for key in keys:
        vals = [metric_value(s, key) for s in summaries]
        out[key] = mean(vals)
        out[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
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


def format_md(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Variant | Policy | Acc | dMethod | F1 | dMethod F1 | Fallback | H/G/Net | Hrec | Grec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        s = row["summary"]
        h = float(s.get("selected_harm", 0) or 0)
        g = float(s.get("selected_help", 0) or 0)
        total_h = float(s.get("total_harm", 0) or 0)
        total_g = float(s.get("total_help", 0) or 0)
        lines.append(
            f"| {row['variant']} | {row['policy']} | "
            f"{fmt_pct(s.get('pcp_deploy_acc'))} | {fmt_pct(s.get('delta_vs_intervention'), signed=True)} | "
            f"{fmt_pct(s.get('pcp_deploy_f1'))} | {fmt_pct(s.get('delta_f1_vs_intervention'), signed=True)} | "
            f"{fmt_num(s.get('baseline_generated'))} | "
            f"{fmt_num(h)}/{fmt_num(g)}/{fmt_num(h - g)} | "
            f"{fmt_pct(h / total_h if total_h else 0.0)} | {fmt_pct(g / total_g if total_g else 0.0)} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build transition-split C/D/fusion ablation table for one target/dataset.")
    ap.add_argument("--source_apply_dir", required=True, help="Existing apply dir with summary.json and deployment_summary.json.")
    ap.add_argument("--policy_root", required=True, help="Transition-split policy dir containing yes_to_no/no_to_yes.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--random_k", type=int, default=0)
    ap.add_argument("--random_repeats", type=int, default=100)
    ap.add_argument("--random_seed", type=int, default=17)
    args = ap.parse_args()

    source_apply_dir = Path(args.source_apply_dir).resolve()
    policy_root = Path(args.policy_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    rows_csv = rows_csv_from_apply_dir(source_apply_dir)
    old_deploy = read_json(source_apply_dir / "deployment_summary.json")
    rows = pcp.load_rows(str(rows_csv), derive_decision_kl=True)
    provenance = policy_provenance(policy_root)
    if any(key.endswith("_warning") for key in provenance):
        print("[warn] policy root may be test-calibrated:", policy_root)

    selected_policy = variant_policy_label(policy_root, "selected")
    random_k = int(args.random_k) if int(args.random_k) > 0 else int(
        (old_deploy.get("baseline_generated") or 0)
    )
    if random_k <= 0:
        # Default to selected policy fallback count.
        selected_routes = build_family_routes(rows, policy_root=policy_root, family="selected")
        selected_summary = summarize_variant(
            old_deploy=old_deploy,
            route_rows=selected_routes,
            out_csv=out_dir / "selected_for_random_count" / "pcp_route_rows.csv",
        )
        random_k = int(round(float(selected_summary.get("baseline_generated", 0) or 0)))

    table_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for key, label in VARIANTS:
        summaries: List[Dict[str, Any]] = []
        policy = ""
        if key in {"noop", "always"}:
            route_rows = build_forced_routes(rows, mode=key, rng=None, k=random_k)
            summary = summarize_variant(
                old_deploy=old_deploy,
                route_rows=route_rows,
                out_csv=out_dir / key / "pcp_route_rows.csv",
            )
            summaries = [summary]
            policy = variant_policy_label(policy_root, key)
        elif key == "random":
            policy = f"random k={random_k}"
            for i in range(max(1, int(args.random_repeats))):
                rng = random.Random(int(args.random_seed) + i)
                route_rows = build_forced_routes(rows, mode="random", rng=rng, k=random_k)
                summary = summarize_variant(
                    old_deploy=old_deploy,
                    route_rows=route_rows,
                    out_csv=out_dir / key / f"seed_{int(args.random_seed) + i}" / "pcp_route_rows.csv",
                )
                summaries.append(summary)
            summary = aggregate_random(summaries)
        else:
            try:
                route_rows = build_family_routes(rows, policy_root=policy_root, family=key)
                summary = summarize_variant(
                    old_deploy=old_deploy,
                    route_rows=route_rows,
                    out_csv=out_dir / key / "pcp_route_rows.csv",
                )
                summaries = [summary]
                policy = variant_policy_label(policy_root, key)
            except RuntimeError as exc:
                summary = {"error": str(exc)}
                policy = "unavailable"

        write_json(out_dir / key / "summary.json", summary)
        table_row = {
            "variant": label,
            "policy": policy,
            "summary": summary,
        }
        table_rows.append(table_row)
        csv_rows.append(
            {
                "label": str(args.label),
                "dataset": str(args.dataset),
                "variant": label,
                "policy": policy,
                "source_apply_dir": str(source_apply_dir),
                "apply_rows_csv": str(rows_csv),
                "policy_root": str(policy_root),
                "yes_to_no_policy_rows_csv": provenance.get("yes_to_no_rows_csv", ""),
                "no_to_yes_policy_rows_csv": provenance.get("no_to_yes_rows_csv", ""),
                "random_k": random_k if key == "random" else "",
                "random_repeats": int(args.random_repeats) if key == "random" else "",
                "acc": summary.get("pcp_deploy_acc"),
                "delta_vs_intervention": summary.get("delta_vs_intervention"),
                "f1": summary.get("pcp_deploy_f1"),
                "delta_f1_vs_intervention": summary.get("delta_f1_vs_intervention"),
                "fallback": summary.get("baseline_generated"),
                "selected_harm": summary.get("selected_harm"),
                "selected_help": summary.get("selected_help"),
                "net": summary.get("net"),
                "hrec": (
                    float(summary.get("selected_harm", 0) or 0) / float(summary.get("total_harm", 0) or 1)
                    if not summary.get("error")
                    else ""
                ),
                "grec": (
                    float(summary.get("selected_help", 0) or 0) / float(summary.get("total_help", 0) or 1)
                    if not summary.get("error")
                    else ""
                ),
                "error": summary.get("error", ""),
            }
        )

    write_csv(out_dir / "family_ablation.csv", csv_rows)
    write_json(
        out_dir / "provenance.json",
        {
            "source_apply_dir": str(source_apply_dir),
            "apply_rows_csv": str(rows_csv),
            "policy_root": str(policy_root),
            "policy_provenance": provenance,
        },
    )
    md = format_md(table_rows)
    header = ""
    if args.label or args.dataset:
        header = f"## {args.label} / {args.dataset}\n\n"
    header += (
        f"- Apply rows: `{rows_csv}`\n"
        f"- Policy root: `{policy_root}`\n"
        f"- yes->no calibration rows: `{provenance.get('yes_to_no_rows_csv', '')}`\n"
        f"- no->yes calibration rows: `{provenance.get('no_to_yes_rows_csv', '')}`\n\n"
    )
    (out_dir / "family_ablation.md").write_text(header + md + "\n", encoding="utf-8")
    print(header + md)
    print("[saved]", out_dir / "family_ablation.md")


if __name__ == "__main__":
    main()
