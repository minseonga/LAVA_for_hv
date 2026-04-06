#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import build_posthoc_b_c_fusion_controller as base


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if s == "":
            continue
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


def build_score_maps(
    rows: Sequence[Dict[str, Any]],
    b_feat: Optional[Dict[str, Any]],
    c_feats: Sequence[Dict[str, Any]],
    fusion_result: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    w_b_fusion = 0.0 if fusion_result is None else float(fusion_result.get("w_b", 0.0))
    w_c_fusion = 0.0 if fusion_result is None else float(fusion_result.get("w_c", 0.0))
    for row in rows:
        sid = str(row.get("id", "")).strip()
        b_score = None if b_feat is None else base.oriented_z(row, b_feat)
        c_vals = [base.oriented_z(row, feat) for feat in c_feats]
        if any(v is None for v in c_vals):
            c_score = None
        else:
            c_score = None if not c_vals else float(sum(float(v) for v in c_vals if v is not None) / float(len(c_vals)))
        f_score = None
        if b_score is not None or c_score is not None:
            f_score = float(w_b_fusion * float(b_score or 0.0) + w_c_fusion * float(c_score or 0.0))
        out[sid] = {
            "b_score": b_score,
            "c_score": c_score,
            "f_score": f_score,
        }
    return out


def expert_route(score: Optional[float], tau: float) -> Optional[str]:
    if score is None:
        return None
    return "baseline" if float(score) >= float(tau) else "method"


def sign(v: Optional[float]) -> int:
    if v is None:
        return 0
    if float(v) > 0:
        return 1
    if float(v) < 0:
        return -1
    return 0


def choose_expert(
    b_score: Optional[float],
    c_score: Optional[float],
    delta: float,
    mode: str,
) -> str:
    b_ok = b_score is not None
    c_ok = c_score is not None
    if b_ok and not c_ok:
        return "b_only"
    if c_ok and not b_ok:
        return "c_only"
    if not b_ok and not c_ok:
        return "none"

    assert b_score is not None and c_score is not None
    abs_b = abs(float(b_score))
    abs_c = abs(float(c_score))
    if abs_b - abs_c >= float(delta):
        return "b_only"
    if abs_c - abs_b >= float(delta):
        return "c_only"

    if mode == "delta_then_fusion":
        return "fusion"
    if mode == "delta_then_stronger":
        return "b_only" if abs_b >= abs_c else "c_only"
    if mode == "agree_fusion_else_stronger":
        return "fusion" if sign(b_score) == sign(c_score) else ("b_only" if abs_b >= abs_c else "c_only")
    return "fusion"


def evaluate_meta(
    rows: Sequence[Dict[str, Any]],
    score_map: Dict[str, Dict[str, Optional[float]]],
    best_b: Optional[Dict[str, Any]],
    best_c: Optional[Dict[str, Any]],
    best_f: Optional[Dict[str, Any]],
    *,
    delta: float,
    mode: str,
) -> Dict[str, Any]:
    n = 0
    baseline_correct_total = 0
    intervention_correct_total = 0
    final_correct_total = 0
    total_harm = 0
    total_help = 0
    selected_harm = 0
    selected_help = 0
    selected_neutral = 0
    selected_count = 0
    expert_counts = {"b_only": 0, "c_only": 0, "fusion": 0, "none": 0}
    route_rows: List[Dict[str, Any]] = []

    for row in rows:
        bc = row.get("baseline_correct")
        ic = row.get("intervention_correct")
        sid = str(row.get("id", "")).strip()
        if bc is None or ic is None or sid == "":
            continue
        b_score = score_map.get(sid, {}).get("b_score")
        c_score = score_map.get(sid, {}).get("c_score")
        f_score = score_map.get(sid, {}).get("f_score")
        expert = choose_expert(b_score, c_score, delta=float(delta), mode=mode)
        expert_counts[expert] = int(expert_counts.get(expert, 0)) + 1

        if expert == "b_only":
            score = b_score
            tau = None if best_b is None else float(best_b["tau"])
        elif expert == "c_only":
            score = c_score
            tau = None if best_c is None else float(best_c["tau"])
        elif expert == "fusion":
            score = f_score
            tau = None if best_f is None else float(best_f["tau"])
        else:
            score = None
            tau = None

        route = expert_route(score, tau) if tau is not None else None
        if route is None:
            route = "method"

        harm = int(base.maybe_int(row.get("harm")) or 0)
        help_ = int(base.maybe_int(row.get("help")) or 0)
        total_harm += harm
        total_help += help_
        baseline_correct_total += int(bc)
        intervention_correct_total += int(ic)
        if route == "baseline":
            selected_count += 1
            selected_harm += harm
            selected_help += help_
            selected_neutral += int((harm == 0) and (help_ == 0))
            final_correct_total += int(bc)
        else:
            final_correct_total += int(ic)
        n += 1
        route_rows.append(
            {
                "id": sid,
                "expert": expert,
                "route": route,
                "b_score": b_score,
                "c_score": c_score,
                "f_score": f_score,
                "harm": harm,
                "help": help_,
                "baseline_correct": int(bc),
                "intervention_correct": int(ic),
                "final_correct": int(bc) if route == "baseline" else int(ic),
            }
        )

    baseline_rate = base.safe_div(float(selected_count), float(max(1, n)))
    precision = base.safe_div(float(selected_harm), float(max(1, selected_count)))
    recall = base.safe_div(float(selected_harm), float(max(1, total_harm)))
    f1 = base.safe_div(2.0 * precision * recall, precision + recall)
    return {
        "mode": str(mode),
        "delta": float(delta),
        "n_eval": int(n),
        "baseline_rate": baseline_rate,
        "method_rate": float(1.0 - baseline_rate),
        "final_acc": base.safe_div(float(final_correct_total), float(max(1, n))),
        "baseline_acc": base.safe_div(float(baseline_correct_total), float(max(1, n))),
        "intervention_acc": base.safe_div(float(intervention_correct_total), float(max(1, n))),
        "delta_vs_intervention": base.safe_div(float(final_correct_total - intervention_correct_total), float(max(1, n))),
        "selected_count": int(selected_count),
        "selected_harm": int(selected_harm),
        "selected_help": int(selected_help),
        "selected_neutral": int(selected_neutral),
        "selected_harm_precision": precision,
        "selected_harm_recall": recall,
        "selected_harm_f1": f1,
        "expert_b_only_rate": base.safe_div(float(expert_counts["b_only"]), float(max(1, n))),
        "expert_c_only_rate": base.safe_div(float(expert_counts["c_only"]), float(max(1, n))),
        "expert_fusion_rate": base.safe_div(float(expert_counts["fusion"]), float(max(1, n))),
        "route_rows": route_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a dynamic post-hoc meta-controller that arbitrates between B-only, C-only, and fusion.")
    ap.add_argument("--scores_csv", type=str, required=True)
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--b_feature_cols", type=str, default="stage_b_score")
    ap.add_argument(
        "--c_feature_cols",
        type=str,
        default="cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy",
    )
    ap.add_argument("--min_feature_auroc", type=float, default=0.55)
    ap.add_argument("--top_k_c", type=int, default=3)
    ap.add_argument("--weight_grid", type=str, default="0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--tau_objective", type=str, default="final_acc", choices=["final_acc", "harm_f1", "harm_precision", "harm_recall"])
    ap.add_argument("--min_baseline_rate", type=float, default=0.0)
    ap.add_argument("--max_baseline_rate", type=float, default=1.0)
    ap.add_argument("--min_selected_count", type=int, default=0)
    ap.add_argument("--delta_grid", type=str, default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--meta_modes", type=str, default="delta_then_fusion,delta_then_stronger,agree_fusion_else_stronger")
    args = ap.parse_args()

    rows = base.load_merged_rows(os.path.abspath(args.scores_csv), os.path.abspath(args.features_csv))
    log(f"[meta] loaded rows={len(rows)}")
    b_feature_names = [x.strip() for x in str(args.b_feature_cols).split(",") if x.strip()]
    c_feature_names = [x.strip() for x in str(args.c_feature_cols).split(",") if x.strip()]
    weight_grid = [float(x.strip()) for x in str(args.weight_grid).split(",") if x.strip()]
    delta_grid = parse_float_list(args.delta_grid)
    meta_modes = [x.strip() for x in str(args.meta_modes).split(",") if x.strip()]
    log(f"[meta] b_feature_cols={b_feature_names}")
    log(f"[meta] c_feature_candidates={len(c_feature_names)}")

    b_metrics: List[Dict[str, Any]] = []
    for feat in b_feature_names:
        result = base.orient_feature(rows, feat, target="harm")
        if result is not None:
            b_metrics.append(result)
    b_metrics.sort(key=lambda r: (-float(r["auroc"]), str(r["feature"])))
    best_b_feat = b_metrics[0] if b_metrics else None
    log(f"[meta] best_b_feature={None if best_b_feat is None else best_b_feat['feature']}")

    c_metrics: List[Dict[str, Any]] = []
    for feat in c_feature_names:
        result = base.orient_feature(rows, feat, target="harm")
        if result is not None:
            c_metrics.append(result)
    c_metrics.sort(key=lambda r: (-float(r["auroc"]), str(r["feature"])))
    selected_c = [r for r in c_metrics if float(r["auroc"]) >= float(args.min_feature_auroc)]
    if int(args.top_k_c) > 0:
        selected_c = selected_c[: int(args.top_k_c)]
    log(f"[meta] selected_c={[str(r['feature']) for r in selected_c]}")

    best_results: Dict[str, Dict[str, Any]] = {}
    fusion_candidates: List[Dict[str, Any]] = []

    if best_b_feat is not None:
        log("[meta] search b_only")
        best, cand = base.search_family(
            rows,
            b_feat=best_b_feat,
            c_feats=[],
            family="b_only",
            weight_grid=weight_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
        )
        fusion_candidates.extend(cand)
        if best is not None:
            best_results["b_only"] = best
            log(f"[meta] best b_only delta={float(best['delta_vs_intervention']):.6f}")

    if selected_c:
        log("[meta] search c_only")
        best, cand = base.search_family(
            rows,
            b_feat=None,
            c_feats=selected_c,
            family="c_only",
            weight_grid=weight_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
        )
        fusion_candidates.extend(cand)
        if best is not None:
            best_results["c_only"] = best
            log(f"[meta] best c_only delta={float(best['delta_vs_intervention']):.6f}")

    if best_b_feat is not None and selected_c:
        log("[meta] search fusion")
        best, cand = base.search_family(
            rows,
            b_feat=best_b_feat,
            c_feats=selected_c,
            family="fusion",
            weight_grid=weight_grid,
            objective=str(args.tau_objective),
            min_baseline_rate=float(args.min_baseline_rate),
            max_baseline_rate=float(args.max_baseline_rate),
            min_selected_count=int(args.min_selected_count),
        )
        fusion_candidates.extend(cand)
        if best is not None:
            best_results["fusion"] = best
            log(f"[meta] best fusion delta={float(best['delta_vs_intervention']):.6f}")

    if "b_only" not in best_results or "c_only" not in best_results or "fusion" not in best_results:
        raise RuntimeError("Need valid b_only, c_only, and fusion experts before meta arbitration.")

    score_map = build_score_maps(rows, best_b_feat, selected_c, best_results.get("fusion"))
    log("[meta] search meta policies")

    meta_candidates: List[Dict[str, Any]] = []
    best_meta: Optional[Dict[str, Any]] = None
    best_route_rows: List[Dict[str, Any]] = []
    for mode in meta_modes:
        for delta in delta_grid:
            result = evaluate_meta(
                rows,
                score_map,
                best_results.get("b_only"),
                best_results.get("c_only"),
                best_results.get("fusion"),
                delta=float(delta),
                mode=mode,
            )
            meta_candidates.append({k: v for k, v in result.items() if k != "route_rows"})
            if best_meta is None or base.selection_key(result, str(args.tau_objective)) > base.selection_key(best_meta, str(args.tau_objective)):
                best_meta = {k: v for k, v in result.items() if k != "route_rows"}
                best_route_rows = result["route_rows"]

    if best_meta is None:
        raise RuntimeError("Failed to select a meta-controller.")
    log(f"[meta] best mode={best_meta['mode']} delta={float(best_meta['delta_vs_intervention']):.6f}")

    candidates_csv = os.path.join(args.out_dir, "fusion_candidates.csv")
    meta_candidates_csv = os.path.join(args.out_dir, "meta_candidates.csv")
    route_rows_csv = os.path.join(args.out_dir, "meta_route_rows.csv")
    b_metrics_csv = os.path.join(args.out_dir, "b_feature_metrics.csv")
    c_metrics_csv = os.path.join(args.out_dir, "c_feature_metrics.csv")
    selected_json = os.path.join(args.out_dir, "selected_meta_policy.json")
    bundle_json = os.path.join(args.out_dir, "selected_meta_bundle.json")
    summary_json = os.path.join(args.out_dir, "summary.json")

    base.write_csv(candidates_csv, fusion_candidates)
    base.write_csv(meta_candidates_csv, meta_candidates)
    base.write_csv(route_rows_csv, best_route_rows)
    base.write_csv(b_metrics_csv, b_metrics)
    base.write_csv(c_metrics_csv, c_metrics)
    write_json(selected_json, best_meta)
    write_json(
        bundle_json,
        {
            "best_b_feature": best_b_feat,
            "selected_c_features": selected_c,
            "best_experts": best_results,
            "best_meta_policy": best_meta,
        },
    )
    write_json(
        summary_json,
        {
            "inputs": {
                "scores_csv": os.path.abspath(args.scores_csv),
                "features_csv": os.path.abspath(args.features_csv),
                "b_feature_cols": b_feature_names,
                "c_feature_cols": c_feature_names,
                "min_feature_auroc": float(args.min_feature_auroc),
                "top_k_c": int(args.top_k_c),
                "weight_grid": weight_grid,
                "tau_objective": str(args.tau_objective),
                "min_baseline_rate": float(args.min_baseline_rate),
                "max_baseline_rate": float(args.max_baseline_rate),
                "min_selected_count": int(args.min_selected_count),
                "delta_grid": delta_grid,
                "meta_modes": meta_modes,
            },
            "counts": {
                "n_rows": int(len(rows)),
                "n_harm": int(sum(int(row.get("harm", 0) or 0) for row in rows)),
                "n_help": int(sum(int(row.get("help", 0) or 0) for row in rows)),
            },
            "best_b_feature": best_b_feat,
            "selected_c_features": selected_c,
            "best_experts": best_results,
            "best_meta_policy": best_meta,
            "outputs": {
                "fusion_candidates_csv": os.path.abspath(candidates_csv),
                "meta_candidates_csv": os.path.abspath(meta_candidates_csv),
                "meta_route_rows_csv": os.path.abspath(route_rows_csv),
                "b_feature_metrics_csv": os.path.abspath(b_metrics_csv),
                "c_feature_metrics_csv": os.path.abspath(c_metrics_csv),
                "selected_meta_policy_json": os.path.abspath(selected_json),
                "selected_meta_bundle_json": os.path.abspath(bundle_json),
            },
        },
    )
    log(f"[saved] {os.path.abspath(summary_json)}")


if __name__ == "__main__":
    main()
