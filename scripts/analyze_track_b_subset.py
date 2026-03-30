#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


CASE_D2 = "vga_regression"
CASE_D1 = "vga_improvement"
CASE_BC = "both_correct"
CASE_BW = "both_wrong"
CASE_ORDER = [CASE_D2, CASE_D1, CASE_BC, CASE_BW]


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def auc_rank(y: Sequence[int], s: Sequence[float]) -> tuple[float, int]:
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(s, dtype=float)
    ok = np.isfinite(s_arr)
    y_arr = y_arr[ok]
    s_arr = s_arr[ok]
    n1 = int((y_arr == 1).sum())
    n0 = int((y_arr == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), 0
    ranks = pd.Series(s_arr).rank(method="average").to_numpy()
    auc = (ranks[y_arr == 1].sum() - n1 * (n1 + 1) / 2.0) / float(n1 * n0)
    return float(auc), int(len(y_arr))


def ks_stat(y: Sequence[int], s: Sequence[float]) -> float:
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(s, dtype=float)
    ok = np.isfinite(s_arr)
    y_arr = y_arr[ok]
    s_arr = s_arr[ok]
    a = np.sort(s_arr[y_arr == 1])
    b = np.sort(s_arr[y_arr == 0])
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.unique(np.concatenate([a, b]))
    cdfa = np.searchsorted(a, grid, side="right") / float(len(a))
    cdfb = np.searchsorted(b, grid, side="right") / float(len(b))
    return float(np.max(np.abs(cdfa - cdfb)))


def mean_of_mask(scores: np.ndarray, mask: np.ndarray) -> float:
    vals = scores[mask & np.isfinite(scores)]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def build_case_counts(df_case: pd.DataFrame) -> pd.DataFrame:
    cnt = Counter(str(x) for x in df_case["case_type"].astype(str))
    rows = []
    total = int(len(df_case))
    for case_type in CASE_ORDER:
        n = int(cnt.get(case_type, 0))
        rows.append(
            {
                "case_type": case_type,
                "n": n,
                "ratio": (float(n) / float(total) if total > 0 else float("nan")),
            }
        )
    return pd.DataFrame(rows)


def sweep_thresholds(
    cases: np.ndarray,
    scores: np.ndarray,
    descending: bool,
    lambda_d1: float,
    lambda_bc: float,
    max_d1_rate: float,
    max_bc_rate: float,
) -> Dict[str, Any] | None:
    ok = np.isfinite(scores)
    cases = cases[ok]
    scores = scores[ok]
    if len(scores) == 0:
        return None

    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    s = scores[order]
    c = cases[order]

    is_d2 = (c == CASE_D2).astype(np.int64)
    is_d1 = (c == CASE_D1).astype(np.int64)
    is_bc = (c == CASE_BC).astype(np.int64)
    is_bw = (c == CASE_BW).astype(np.int64)

    n_d2 = int(is_d2.sum())
    n_d1 = int(is_d1.sum())
    n_bc = int(is_bc.sum())
    n_bw = int(is_bw.sum())
    if n_d2 == 0:
        return None

    cum_d2 = np.cumsum(is_d2)
    cum_d1 = np.cumsum(is_d1)
    cum_bc = np.cumsum(is_bc)
    cum_bw = np.cumsum(is_bw)
    run_ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])

    best_any: Dict[str, Any] | None = None
    best_ok: Dict[str, Any] | None = None
    direction = "ge" if descending else "le"
    for end in run_ends:
        veto_n = int(end + 1)
        d2 = int(cum_d2[end])
        d1 = int(cum_d1[end])
        bc = int(cum_bc[end])
        bw = int(cum_bw[end])
        d2_rate = float(d2 / n_d2) if n_d2 > 0 else float("nan")
        d1_rate = float(d1 / n_d1) if n_d1 > 0 else 0.0
        bc_rate = float(bc / n_bc) if n_bc > 0 else 0.0
        obj = float(d2 - lambda_d1 * d1 - lambda_bc * bc)
        row = {
            "threshold": float(s[end]),
            "direction": direction,
            "veto_rate": float(veto_n / len(s)),
            "d2_correct_veto": d2,
            "d1_wrong_veto": d1,
            "bc_wrong_veto": bc,
            "bw_veto": bw,
            "d2_correct_veto_rate": d2_rate,
            "d1_wrong_veto_rate": d1_rate,
            "bc_wrong_veto_rate": bc_rate,
            "objective": obj,
            "meets_constraints": bool(d1_rate <= max_d1_rate and bc_rate <= max_bc_rate),
        }
        if (
            best_any is None
            or row["objective"] > best_any["objective"]
            or (
                row["objective"] == best_any["objective"]
                and row["d2_correct_veto"] > best_any["d2_correct_veto"]
            )
        ):
            best_any = dict(row)
        if row["meets_constraints"]:
            if (
                best_ok is None
                or row["objective"] > best_ok["objective"]
                or (
                    row["objective"] == best_ok["objective"]
                    and row["d2_correct_veto"] > best_ok["d2_correct_veto"]
                )
            ):
                best_ok = dict(row)
    return {"best_any": best_any, "best_ok": best_ok}


def summarize_pair(scores: np.ndarray, case_types: np.ndarray, pos_case: str, neg_case: str) -> Dict[str, Any]:
    mask = (case_types == pos_case) | (case_types == neg_case)
    if int(mask.sum()) == 0:
        return {
            "n": 0,
            "auc": float("nan"),
            "auc_best": float("nan"),
            "direction": "na",
            "ks": float("nan"),
        }
    yy = (case_types[mask] == pos_case).astype(int)
    ss = scores[mask]
    auc, n = auc_rank(yy, ss)
    return {
        "n": int(n),
        "auc": auc,
        "auc_best": (float(max(auc, 1.0 - auc)) if math.isfinite(auc) else float("nan")),
        "direction": ("higher_in_pos" if auc >= 0.5 else "lower_in_pos") if math.isfinite(auc) else "na",
        "ks": ks_stat(yy, ss),
    }


def analyze_table(
    df_in: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
    lambda_d1: float,
    lambda_bc: float,
    max_d1_rate: float,
    max_bc_rate: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouped = df_in.groupby(group_cols, dropna=False)
    for entity_key, df_ent in grouped:
        if not isinstance(entity_key, tuple):
            entity_key = (entity_key,)
        entity_meta = {col: entity_key[i] for i, col in enumerate(group_cols)}
        case_types = df_ent["case_type"].astype(str).to_numpy()
        for metric in metric_cols:
            scores = pd.to_numeric(df_ent[metric], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(scores)
            if int(finite.sum()) == 0:
                continue
            pair_d2d1 = summarize_pair(scores, case_types, CASE_D2, CASE_D1)
            pair_d2bc = summarize_pair(scores, case_types, CASE_D2, CASE_BC)
            sweep_hi = sweep_thresholds(
                case_types,
                scores,
                descending=True,
                lambda_d1=lambda_d1,
                lambda_bc=lambda_bc,
                max_d1_rate=max_d1_rate,
                max_bc_rate=max_bc_rate,
            )
            sweep_lo = sweep_thresholds(
                case_types,
                scores,
                descending=False,
                lambda_d1=lambda_d1,
                lambda_bc=lambda_bc,
                max_d1_rate=max_d1_rate,
                max_bc_rate=max_bc_rate,
            )

            def pick_best(which: str) -> Dict[str, Any] | None:
                cand = []
                if sweep_hi is not None and sweep_hi.get(which) is not None:
                    cand.append(sweep_hi[which])
                if sweep_lo is not None and sweep_lo.get(which) is not None:
                    cand.append(sweep_lo[which])
                if len(cand) == 0:
                    return None
                cand = sorted(
                    cand,
                    key=lambda x: (
                        float(x["objective"]),
                        float(x["d2_correct_veto"]),
                        -float(x["veto_rate"]),
                    ),
                    reverse=True,
                )
                return cand[0]

            best_any = pick_best("best_any")
            best_ok = pick_best("best_ok")
            case_arr = np.asarray(case_types, dtype=object)
            row = dict(entity_meta)
            row.update(
                {
                    "metric": metric,
                    "n_finite": int(finite.sum()),
                    "mean_d2": mean_of_mask(scores, case_arr == CASE_D2),
                    "mean_d1": mean_of_mask(scores, case_arr == CASE_D1),
                    "mean_bc": mean_of_mask(scores, case_arr == CASE_BC),
                    "mean_bw": mean_of_mask(scores, case_arr == CASE_BW),
                    "delta_d2_minus_d1": (
                        float(mean_of_mask(scores, case_arr == CASE_D2) - mean_of_mask(scores, case_arr == CASE_D1))
                        if math.isfinite(mean_of_mask(scores, case_arr == CASE_D2))
                        and math.isfinite(mean_of_mask(scores, case_arr == CASE_D1))
                        else float("nan")
                    ),
                    "delta_d2_minus_bc": (
                        float(mean_of_mask(scores, case_arr == CASE_D2) - mean_of_mask(scores, case_arr == CASE_BC))
                        if math.isfinite(mean_of_mask(scores, case_arr == CASE_D2))
                        and math.isfinite(mean_of_mask(scores, case_arr == CASE_BC))
                        else float("nan")
                    ),
                    "auc_d2_vs_d1": pair_d2d1["auc"],
                    "auc_best_d2_vs_d1": pair_d2d1["auc_best"],
                    "direction_d2_vs_d1": pair_d2d1["direction"],
                    "ks_d2_vs_d1": pair_d2d1["ks"],
                    "auc_d2_vs_bc": pair_d2bc["auc"],
                    "auc_best_d2_vs_bc": pair_d2bc["auc_best"],
                    "direction_d2_vs_bc": pair_d2bc["direction"],
                    "ks_d2_vs_bc": pair_d2bc["ks"],
                    "best_any_direction": (None if best_any is None else best_any["direction"]),
                    "best_any_threshold": (None if best_any is None else best_any["threshold"]),
                    "best_any_objective": (None if best_any is None else best_any["objective"]),
                    "best_any_veto_rate": (None if best_any is None else best_any["veto_rate"]),
                    "best_any_d2_correct_veto": (None if best_any is None else best_any["d2_correct_veto"]),
                    "best_any_d1_wrong_veto": (None if best_any is None else best_any["d1_wrong_veto"]),
                    "best_any_bc_wrong_veto": (None if best_any is None else best_any["bc_wrong_veto"]),
                    "best_any_d2_rate": (None if best_any is None else best_any["d2_correct_veto_rate"]),
                    "best_any_d1_rate": (None if best_any is None else best_any["d1_wrong_veto_rate"]),
                    "best_any_bc_rate": (None if best_any is None else best_any["bc_wrong_veto_rate"]),
                    "best_ok_direction": (None if best_ok is None else best_ok["direction"]),
                    "best_ok_threshold": (None if best_ok is None else best_ok["threshold"]),
                    "best_ok_objective": (None if best_ok is None else best_ok["objective"]),
                    "best_ok_veto_rate": (None if best_ok is None else best_ok["veto_rate"]),
                    "best_ok_d2_correct_veto": (None if best_ok is None else best_ok["d2_correct_veto"]),
                    "best_ok_d1_wrong_veto": (None if best_ok is None else best_ok["d1_wrong_veto"]),
                    "best_ok_bc_wrong_veto": (None if best_ok is None else best_ok["bc_wrong_veto"]),
                    "best_ok_d2_rate": (None if best_ok is None else best_ok["d2_correct_veto_rate"]),
                    "best_ok_d1_rate": (None if best_ok is None else best_ok["d1_wrong_veto_rate"]),
                    "best_ok_bc_rate": (None if best_ok is None else best_ok["bc_wrong_veto_rate"]),
                    "has_feasible_policy": int(best_ok is not None),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ["best_ok_objective", "auc_best_d2_vs_bc", "auc_best_d2_vs_d1"] if c in out.columns]
    return out.sort_values(sort_cols, ascending=False, na_position="last").reset_index(drop=True)


def select_head_candidates(
    df_head: pd.DataFrame,
    metric_name: str,
    topk_global: int,
    topk_per_layer: int,
    want_higher_in_d2: bool,
) -> List[Dict[str, Any]]:
    if df_head.empty:
        return []
    x = df_head[df_head["metric"] == metric_name].copy()
    if x.empty:
        return []
    if want_higher_in_d2:
        x = x[
            (x["direction_d2_vs_d1"] == "higher_in_pos")
            & (x["direction_d2_vs_bc"] == "higher_in_pos")
        ]
    else:
        x = x[
            (x["direction_d2_vs_d1"] == "lower_in_pos")
            & (x["direction_d2_vs_bc"] == "lower_in_pos")
        ]
    if x.empty:
        return []
    x = x.sort_values(
        ["has_feasible_policy", "best_ok_objective", "auc_best_d2_vs_bc", "auc_best_d2_vs_d1"],
        ascending=False,
        na_position="last",
    )
    out: List[Dict[str, Any]] = []
    per_layer_count: Dict[int, int] = {}
    for _, row in x.iterrows():
        layer = int(row["block_layer_idx"])
        head = int(row["head_idx"])
        if per_layer_count.get(layer, 0) >= int(topk_per_layer):
            continue
        out.append(
            {
                "layer": layer,
                "head": head,
                "metric": str(row["metric"]),
                "best_ok_objective": safe_float(row["best_ok_objective"]),
                "best_ok_direction": row["best_ok_direction"],
                "best_ok_threshold": safe_float(row["best_ok_threshold"]),
                "auc_best_d2_vs_d1": safe_float(row["auc_best_d2_vs_d1"]),
                "auc_best_d2_vs_bc": safe_float(row["auc_best_d2_vs_bc"]),
                "delta_d2_minus_d1": safe_float(row["delta_d2_minus_d1"]),
                "delta_d2_minus_bc": safe_float(row["delta_d2_minus_bc"]),
            }
        )
        per_layer_count[layer] = per_layer_count.get(layer, 0) + 1
        if len(out) >= int(topk_global):
            break
    return out


def to_layer_map(items: Iterable[Dict[str, Any]]) -> Dict[str, List[int]]:
    mp: Dict[str, List[int]] = {}
    for item in items:
        li = str(int(item["layer"]))
        hi = int(item["head"])
        mp.setdefault(li, []).append(hi)
    for k in list(mp.keys()):
        mp[k] = sorted(set(mp[k]))
    return mp


def write_summary_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Track B subset analysis for fast late-layer/head signals.")
    ap.add_argument("--per_case_csv", type=str, required=True, help="taxonomy per_case_compare.csv")
    ap.add_argument("--per_layer_trace_csv", type=str, default="", help="optional per_layer_yes_trace.csv")
    ap.add_argument("--per_head_trace_csv", type=str, default="", help="optional per_head_yes_trace.csv")
    ap.add_argument(
        "--layer_metrics",
        type=str,
        default="yes_attn_vis_ratio,yes_attn_vis_sum,yes_sim_objpatch_topk,yes_sim_objpatch_max,yes_sim_local_topk",
    )
    ap.add_argument(
        "--head_metrics",
        type=str,
        default="head_attn_vis_ratio,head_attn_vis_sum,head_attn_vis_peak,head_attn_vis_entropy",
    )
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--lambda_d1", type=float, default=1.0)
    ap.add_argument("--lambda_bc", type=float, default=0.5)
    ap.add_argument("--max_d1_rate", type=float, default=0.05)
    ap.add_argument("--max_bc_rate", type=float, default=0.20)
    ap.add_argument("--candidate_head_metric", type=str, default="head_attn_vis_ratio")
    ap.add_argument("--candidate_topk_global", type=int, default=16)
    ap.add_argument("--candidate_topk_per_layer", type=int, default=4)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df_case = pd.read_csv(args.per_case_csv)
    if "id" not in df_case.columns or "case_type" not in df_case.columns:
        raise RuntimeError("per_case_csv must contain at least columns: id, case_type")
    df_case["id"] = df_case["id"].astype(str)
    df_case = df_case.drop_duplicates(subset=["id"]).copy()

    case_counts = build_case_counts(df_case)
    case_counts.to_csv(os.path.join(out_dir, "case_counts.csv"), index=False)

    summary: Dict[str, Any] = {
        "inputs": {
            "per_case_csv": os.path.abspath(args.per_case_csv),
            "per_layer_trace_csv": os.path.abspath(args.per_layer_trace_csv) if str(args.per_layer_trace_csv).strip() else "",
            "per_head_trace_csv": os.path.abspath(args.per_head_trace_csv) if str(args.per_head_trace_csv).strip() else "",
            "late_range": [int(args.late_start), int(args.late_end)],
            "lambda_d1": float(args.lambda_d1),
            "lambda_bc": float(args.lambda_bc),
            "max_d1_rate": float(args.max_d1_rate),
            "max_bc_rate": float(args.max_bc_rate),
            "candidate_head_metric": str(args.candidate_head_metric),
            "candidate_topk_global": int(args.candidate_topk_global),
            "candidate_topk_per_layer": int(args.candidate_topk_per_layer),
        },
        "counts": {
            "n_cases": int(len(df_case)),
            "case_type_counts": {str(r["case_type"]): int(r["n"]) for _, r in case_counts.iterrows()},
        },
        "outputs": {
            "case_counts_csv": os.path.join(out_dir, "case_counts.csv"),
        },
    }

    if str(args.per_layer_trace_csv).strip():
        df_layer = pd.read_csv(args.per_layer_trace_csv)
        df_layer["id"] = df_layer["id"].astype(str)
        if "block_layer_idx" not in df_layer.columns:
            raise RuntimeError("per_layer_trace_csv must contain block_layer_idx")
        df_layer["block_layer_idx"] = pd.to_numeric(df_layer["block_layer_idx"], errors="coerce")
        df_layer = df_layer[df_layer["block_layer_idx"].between(int(args.late_start), int(args.late_end), inclusive="both")]
        df_layer = df_layer.merge(df_case[["id", "case_type"]], on="id", how="inner")
        layer_metrics = [c for c in parse_csv_list(args.layer_metrics) if c in df_layer.columns]
        if len(layer_metrics) == 0:
            raise RuntimeError("No requested layer metrics were found in per_layer_trace_csv")
        layer_result = analyze_table(
            df_in=df_layer,
            group_cols=["block_layer_idx"],
            metric_cols=layer_metrics,
            lambda_d1=float(args.lambda_d1),
            lambda_bc=float(args.lambda_bc),
            max_d1_rate=float(args.max_d1_rate),
            max_bc_rate=float(args.max_bc_rate),
        )
        layer_result.to_csv(os.path.join(out_dir, "layer_metric_analysis.csv"), index=False)
        layer_result.head(50).to_csv(os.path.join(out_dir, "top_layer_candidates.csv"), index=False)
        summary["outputs"]["layer_metric_analysis_csv"] = os.path.join(out_dir, "layer_metric_analysis.csv")
        summary["outputs"]["top_layer_candidates_csv"] = os.path.join(out_dir, "top_layer_candidates.csv")
        summary["layer_analysis"] = {
            "metrics_used": layer_metrics,
            "top_row": (None if layer_result.empty else layer_result.iloc[0].to_dict()),
        }

    head_result = pd.DataFrame()
    if str(args.per_head_trace_csv).strip():
        df_head = pd.read_csv(args.per_head_trace_csv)
        df_head["id"] = df_head["id"].astype(str)
        for col in ["block_layer_idx", "head_idx"]:
            if col not in df_head.columns:
                raise RuntimeError(f"per_head_trace_csv must contain {col}")
            df_head[col] = pd.to_numeric(df_head[col], errors="coerce")
        df_head = df_head[df_head["block_layer_idx"].between(int(args.late_start), int(args.late_end), inclusive="both")]
        df_head = df_head.merge(df_case[["id", "case_type"]], on="id", how="inner")
        head_metrics = [c for c in parse_csv_list(args.head_metrics) if c in df_head.columns]
        if len(head_metrics) == 0:
            raise RuntimeError("No requested head metrics were found in per_head_trace_csv")
        head_result = analyze_table(
            df_in=df_head,
            group_cols=["block_layer_idx", "head_idx"],
            metric_cols=head_metrics,
            lambda_d1=float(args.lambda_d1),
            lambda_bc=float(args.lambda_bc),
            max_d1_rate=float(args.max_d1_rate),
            max_bc_rate=float(args.max_bc_rate),
        )
        head_result.to_csv(os.path.join(out_dir, "head_metric_analysis.csv"), index=False)
        head_result.head(200).to_csv(os.path.join(out_dir, "top_head_candidates.csv"), index=False)
        summary["outputs"]["head_metric_analysis_csv"] = os.path.join(out_dir, "head_metric_analysis.csv")
        summary["outputs"]["top_head_candidates_csv"] = os.path.join(out_dir, "top_head_candidates.csv")
        summary["head_analysis"] = {
            "metrics_used": head_metrics,
            "top_row": (None if head_result.empty else head_result.iloc[0].to_dict()),
        }

        faithful = select_head_candidates(
            head_result,
            metric_name=str(args.candidate_head_metric),
            topk_global=int(args.candidate_topk_global),
            topk_per_layer=int(args.candidate_topk_per_layer),
            want_higher_in_d2=True,
        )
        harmful = select_head_candidates(
            head_result,
            metric_name=str(args.candidate_head_metric),
            topk_global=int(args.candidate_topk_global),
            topk_per_layer=int(args.candidate_topk_per_layer),
            want_higher_in_d2=False,
        )
        headset_payload = {
            "inputs": {
                "per_head_trace_csv": os.path.abspath(args.per_head_trace_csv),
                "per_case_csv": os.path.abspath(args.per_case_csv),
                "candidate_head_metric": str(args.candidate_head_metric),
                "late_range": [int(args.late_start), int(args.late_end)],
            },
            "selection_rule": (
                "Track B heuristic: faithful candidates are heads whose selected metric is consistently higher in "
                "vga_regression than both vga_improvement and both_correct. Harmful candidates are the reverse."
            ),
            "faithful_heads": faithful,
            "harmful_heads": harmful,
            "faithful_head_specs": [f"{int(x['layer'])}:{int(x['head'])}" for x in faithful],
            "harmful_head_specs": [f"{int(x['layer'])}:{int(x['head'])}" for x in harmful],
            "faithful_heads_by_layer": to_layer_map(faithful),
            "harmful_heads_by_layer": to_layer_map(harmful),
        }
        headset_json = os.path.join(out_dir, "track_b_candidate_headset.json")
        write_summary_json(headset_json, headset_payload)
        pd.DataFrame(faithful).to_csv(os.path.join(out_dir, "faithful_head_candidates.csv"), index=False)
        pd.DataFrame(harmful).to_csv(os.path.join(out_dir, "harmful_head_candidates.csv"), index=False)
        summary["outputs"]["candidate_headset_json"] = headset_json
        summary["outputs"]["faithful_head_candidates_csv"] = os.path.join(out_dir, "faithful_head_candidates.csv")
        summary["outputs"]["harmful_head_candidates_csv"] = os.path.join(out_dir, "harmful_head_candidates.csv")
        summary["candidate_headset"] = {
            "n_faithful": int(len(faithful)),
            "n_harmful": int(len(harmful)),
            "top_faithful": (None if len(faithful) == 0 else faithful[0]),
            "top_harmful": (None if len(harmful) == 0 else harmful[0]),
        }

    summary_json = os.path.join(out_dir, "summary.json")
    write_summary_json(summary_json, summary)

    print("[saved]", os.path.join(out_dir, "case_counts.csv"))
    if "layer_metric_analysis_csv" in summary["outputs"]:
        print("[saved]", summary["outputs"]["layer_metric_analysis_csv"])
        print("[saved]", summary["outputs"]["top_layer_candidates_csv"])
    if "head_metric_analysis_csv" in summary["outputs"]:
        print("[saved]", summary["outputs"]["head_metric_analysis_csv"])
        print("[saved]", summary["outputs"]["top_head_candidates_csv"])
    if "candidate_headset_json" in summary["outputs"]:
        print("[saved]", summary["outputs"]["candidate_headset_json"])
        print("[saved]", summary["outputs"]["faithful_head_candidates_csv"])
        print("[saved]", summary["outputs"]["harmful_head_candidates_csv"])
    print("[saved]", summary_json)
    print("[counts]", summary["counts"])
    if "layer_analysis" in summary and summary["layer_analysis"]["top_row"] is not None:
        print("[top_layer]", summary["layer_analysis"]["top_row"])
    if "head_analysis" in summary and summary["head_analysis"]["top_row"] is not None:
        print("[top_head]", summary["head_analysis"]["top_row"])


if __name__ == "__main__":
    main()
