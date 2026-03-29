#!/usr/bin/env python
import argparse
import csv
import json
import os
from typing import Callable, Dict, List, Optional


def load_csv(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd)


def safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def safe_int(v, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def pick_best(
    rows: List[Dict[str, str]],
    allow: Optional[Callable[[Dict[str, str]], bool]] = None,
    auc_key: str = "auc_best_dir",
) -> Optional[Dict[str, str]]:
    cand = []
    for r in rows:
        if allow is not None and not allow(r):
            continue
        cand.append(r)
    if len(cand) == 0:
        return None
    cand.sort(key=lambda x: safe_float(x.get(auc_key), -1.0), reverse=True)
    return cand[0]


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def build_row(
    *,
    axis: str,
    surrogate: str,
    source_exp: str,
    source_file: str,
    best: Optional[Dict[str, str]],
    metric_key: str = "metric",
    auc_key: str = "auc_best_dir",
    ks_key: str = "ks_hall_high",
    n_key: str = "n",
    direction_key: str = "direction",
    editable_now: str = "yes",
    intervention_target: str = "",
    evidence_note: str = "",
    gqa_pair_auc: Optional[float] = None,
    gqa_pair_ks: Optional[float] = None,
) -> Dict[str, object]:
    if best is None:
        return {
            "axis": axis,
            "surrogate": surrogate,
            "source_exp": source_exp,
            "source_file": source_file,
            "metric": None,
            "auc_best_dir": None,
            "ks": None,
            "direction": None,
            "n": 0,
            "editable_now": editable_now,
            "intervention_target": intervention_target,
            "evidence_note": evidence_note,
            "gqa_pair_absdiff_auc": gqa_pair_auc,
            "gqa_pair_absdiff_ks": gqa_pair_ks,
            "priority_score": None,
            "strength": "missing",
        }
    auc = safe_float(best.get(auc_key), 0.0)
    ks = safe_float(best.get(ks_key), 0.0)
    score = 0.5 * auc + 0.5 * ks
    if auc >= 0.75 and ks >= 0.35:
        strength = "strong"
    elif auc >= 0.65 and ks >= 0.25:
        strength = "moderate"
    else:
        strength = "weak"
    return {
        "axis": axis,
        "surrogate": surrogate,
        "source_exp": source_exp,
        "source_file": source_file,
        "metric": best.get(metric_key),
        "auc_best_dir": auc,
        "ks": ks,
        "direction": best.get(direction_key),
        "n": safe_int(best.get(n_key), 0),
        "editable_now": editable_now,
        "intervention_target": intervention_target,
        "evidence_note": evidence_note,
        "gqa_pair_absdiff_auc": gqa_pair_auc,
        "gqa_pair_absdiff_ks": gqa_pair_ks,
        "priority_score": score,
        "strength": strength,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_root", type=str, default="/home/kms/LLaVA_calibration/experiments")
    ap.add_argument("--out_dir", type=str, default="/home/kms/LLaVA_calibration/experiments/root_cause_surrogate_matrix")
    ap.add_argument("--pope_layer_eval_csv", type=str, default="")
    ap.add_argument("--pope_head_eval_csv", type=str, default="")
    ap.add_argument("--pope_ais_head_eval_csv", type=str, default="")
    ap.add_argument("--pope_ers_eval_csv", type=str, default="")
    ap.add_argument("--onset_feature_metrics_csv", type=str, default="")
    ap.add_argument("--gqa_pair_eval_csv", type=str, default="")
    args = ap.parse_args()

    exp = os.path.abspath(args.experiments_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pope_layer_eval_csv = args.pope_layer_eval_csv or os.path.join(
        exp, "pope_visual_disconnect_1000_alllayers_objpatch", "layer_eval_fp_vs_tp_yes.csv"
    )
    pope_head_eval_csv = args.pope_head_eval_csv or os.path.join(
        exp, "pope_visual_disconnect_1000_headscan_l10_24", "head_eval_fp_vs_tp_yes.csv"
    )
    pope_ais_head_eval_csv = args.pope_ais_head_eval_csv or os.path.join(
        exp, "pope_ais_decomposition_v1", "head_eval_ais_fp_vs_tp_yes.csv"
    )
    pope_ers_eval_csv = args.pope_ers_eval_csv or os.path.join(
        exp, "pope_ers_ais_pcs_v2_with_drop", "eval_ers_ais_pcs.csv"
    )
    onset_feature_metrics_csv = args.onset_feature_metrics_csv or os.path.join(
        exp, "onset_margin_entropy_1000", "feature_metrics.csv"
    )
    gqa_pair_eval_csv = args.gqa_pair_eval_csv or os.path.join(
        exp, "gqa_stage2_contrast_ais_1000_pairaware", "pair_eval.csv"
    )

    layer_rows = load_csv(pope_layer_eval_csv)
    head_rows = load_csv(pope_head_eval_csv)
    ais_head_rows = load_csv(pope_ais_head_eval_csv)
    ers_rows = load_csv(pope_ers_eval_csv)
    onset_rows = load_csv(onset_feature_metrics_csv)
    gqa_pair_rows = load_csv(gqa_pair_eval_csv)

    gqa_pair_absdiff = pick_best(
        gqa_pair_rows,
        allow=lambda r: (r.get("score_field") == "metric_absdiff" and r.get("target") == "pair_consistency_failure"),
        auc_key="auc_best_dir",
    )
    gqa_pair_auc = safe_float(gqa_pair_absdiff.get("auc_best_dir"), 0.0) if gqa_pair_absdiff else None
    gqa_pair_ks = safe_float(gqa_pair_absdiff.get("ks_pos_high"), 0.0) if gqa_pair_absdiff else None

    # Axis 1: vision-aware head failure (head-level visual attention collapse).
    best_head_vis_fail = pick_best(
        head_rows,
        allow=lambda r: str(r.get("metric", "")).startswith("head_attn_vis_")
        and str(r.get("direction", "")) == "lower_in_hallucination",
        auc_key="auc_best_dir",
    )

    # Axis 2: visual sink / artifact (late local/object sim inflation).
    best_visual_sink = pick_best(
        layer_rows,
        allow=lambda r: (
            str(r.get("metric", "")).startswith("yes_sim_local_max__")
            or str(r.get("metric", "")).startswith("yes_sim_objpatch_max__")
        )
        and str(r.get("direction", "")) == "higher_in_hallucination",
        auc_key="auc_best_dir",
    )

    # Axis 3: dynamic harmful heads (AIS contribution at head granularity).
    best_dynamic_harmful_head = pick_best(
        ais_head_rows,
        allow=lambda r: str(r.get("metric", "")).startswith("head_contrib_"),
        auc_key="auc_best_dir",
    )

    # Axis 4: text-prior hijack (margin/entropy onset weakness + ERS weak, include strongest text-only cue).
    best_text_prior = pick_best(
        onset_rows,
        allow=lambda r: str(r.get("feature", "")).startswith("entropy_")
        or str(r.get("feature", "")).startswith("margin_"),
        auc_key="auc_best_dir",
    )

    best_ers_drop = pick_best(
        ers_rows,
        allow=lambda r: str(r.get("metric", "")) == "ers_drop_gt_logit",
        auc_key="auc_best_dir",
    )

    rows: List[Dict[str, object]] = []
    rows.append(
        build_row(
            axis="vision-aware head failure",
            surrogate="head_attn_vis_ratio/sum (head-level)",
            source_exp="pope_visual_disconnect_1000_headscan_l10_24",
            source_file=pope_head_eval_csv,
            best=best_head_vis_fail,
            editable_now="yes",
            intervention_target="late attention path, per-head soft down-weighting",
            evidence_note="Hallucination shows lower image attention at specific late heads.",
            gqa_pair_auc=gqa_pair_auc,
            gqa_pair_ks=gqa_pair_ks,
        )
    )
    rows.append(
        build_row(
            axis="visual sink / artifact",
            surrogate="yes_sim_local_max / yes_sim_objpatch_max (layer-level)",
            source_exp="pope_visual_disconnect_1000_alllayers_objpatch",
            source_file=pope_layer_eval_csv,
            best=best_visual_sink,
            editable_now="yes",
            intervention_target="image-column biasing, patch-level late penalty",
            evidence_note="Late layers show over-attraction to specific visual patches in hallucination.",
            gqa_pair_auc=gqa_pair_auc,
            gqa_pair_ks=gqa_pair_ks,
        )
    )
    rows.append(
        build_row(
            axis="dynamic harmful heads",
            surrogate="head_contrib_l*_h* (AIS contribution)",
            source_exp="pope_ais_decomposition_v1",
            source_file=pope_ais_head_eval_csv,
            best=best_dynamic_harmful_head,
            editable_now="yes",
            intervention_target="dynamic per-head gating (late heads only)",
            evidence_note="Subset of late heads contributes disproportionately to hallucination score.",
            gqa_pair_auc=gqa_pair_auc,
            gqa_pair_ks=gqa_pair_ks,
        )
    )
    # text prior row merges onset + ers_drop signals
    text_row = build_row(
        axis="text-prior hijack",
        surrogate="entropy/margin onset + ers_drop_gt_logit",
        source_exp="onset_margin_entropy_1000 + pope_ers_ais_pcs_v2_with_drop",
        source_file=f"{onset_feature_metrics_csv} | {pope_ers_eval_csv}",
        best=best_text_prior,
        metric_key="feature",
        auc_key="auc_best_dir",
        ks_key="ks_onset_high",
        editable_now="partial",
        intervention_target="decoder-level safeguard (secondary), not primary root cause",
        evidence_note=(
            f"Text-only onset cue is weak/moderate; ERS_drop_gt_logit="
            f"{safe_float(best_ers_drop.get('auc_best_dir') if best_ers_drop else None, 0.0):.4f}"
        ),
        gqa_pair_auc=gqa_pair_auc,
        gqa_pair_ks=gqa_pair_ks,
    )
    rows.append(text_row)

    # Ranking: prefer editable and strong separability.
    ranked = []
    for r in rows:
        score = safe_float(r.get("priority_score"), 0.0)
        editable_bonus = 0.05 if str(r.get("editable_now")) == "yes" else 0.0
        gqa_bonus = 0.02 if safe_float(r.get("gqa_pair_absdiff_auc"), 0.0) >= 0.65 else 0.0
        final = score + editable_bonus + gqa_bonus
        rr = dict(r)
        rr["final_priority"] = final
        ranked.append(rr)
    ranked.sort(key=lambda x: safe_float(x.get("final_priority"), -1.0), reverse=True)

    out_matrix_csv = os.path.join(out_dir, "surrogate_matrix.csv")
    out_rank_csv = os.path.join(out_dir, "surrogate_ranking.csv")
    out_summary_json = os.path.join(out_dir, "summary.json")
    write_csv(out_matrix_csv, rows)
    write_csv(out_rank_csv, ranked)

    summary = {
        "inputs": {
            "pope_layer_eval_csv": pope_layer_eval_csv,
            "pope_head_eval_csv": pope_head_eval_csv,
            "pope_ais_head_eval_csv": pope_ais_head_eval_csv,
            "pope_ers_eval_csv": pope_ers_eval_csv,
            "onset_feature_metrics_csv": onset_feature_metrics_csv,
            "gqa_pair_eval_csv": gqa_pair_eval_csv,
        },
        "counts": {
            "n_layer_rows": len(layer_rows),
            "n_head_rows": len(head_rows),
            "n_ais_head_rows": len(ais_head_rows),
            "n_ers_rows": len(ers_rows),
            "n_onset_rows": len(onset_rows),
            "n_gqa_pair_rows": len(gqa_pair_rows),
        },
        "best_per_axis": rows,
        "ranking_top": ranked[:4],
        "outputs": {
            "surrogate_matrix_csv": out_matrix_csv,
            "surrogate_ranking_csv": out_rank_csv,
            "summary_json": out_summary_json,
        },
    }
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_matrix_csv)
    print("[saved]", out_rank_csv)
    print("[saved]", out_summary_json)


if __name__ == "__main__":
    main()

