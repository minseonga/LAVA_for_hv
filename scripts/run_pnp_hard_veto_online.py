#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pnp_controller.adapters.vga_online import VGAOnlineAdapter, VGAOnlineConfig
from pnp_controller.adapters.vga import VGAOfflineAdapter
from pnp_controller.core.controller import run_offline_hard_veto
from pnp_controller.core.schemas import HardVetoConfig, ThresholdCalibrationConfig


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            t = ln.strip()
            if t == "":
                continue
            out.append(json.loads(t))
    return out


def load_gt_csv(path_csv: str, id_col: str, label_col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            sid = str(row.get(id_col, "")).strip()
            lab = str(row.get(label_col, "")).strip().lower()
            if sid and lab in {"yes", "no"}:
                out[sid] = lab
    return out


def compute_metrics(gt_map: Dict[str, str], pred_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    pred_map: Dict[str, str] = {}
    for row in pred_rows:
        sid = row.get("question_id", row.get("id", None))
        if sid is None:
            continue
        pred_map[str(sid)] = parse_yes_no(str(row.get("output", "")))

    tp = fp = tn = fn = 0
    missing = 0
    for sid, gt in gt_map.items():
        pred = pred_map.get(str(sid), None)
        if pred is None:
            missing += 1
            continue
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": int(n),
        "acc": float((tp + tn) / n) if n else 0.0,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "yes_ratio": float((tp + fp) / n) if n else 0.0,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "missing_pred": int(missing),
    }


def resolve_thresholds(args) -> Tuple[float, float, Dict[str, Any]]:
    if args.controller_summary_json:
        obj = json.loads(open(args.controller_summary_json, "r", encoding="utf-8").read())
        th = obj.get("thresholds", {})
        if "tau_frg" in th and "tau_gmi" in th:
            tau_frg = float(th["tau_frg"])
            tau_gmi = float(th["tau_gmi"])
        elif "tau_c" in th and "tau_e" in th:
            # Backward compatibility for earlier offline controller summaries.
            tau_frg = float(th["tau_c"])
            tau_gmi = float(th["tau_e"])
        else:
            raise KeyError(
                "Could not find threshold keys in controller summary JSON. "
                "Expected either (tau_frg, tau_gmi) or legacy (tau_c, tau_e)."
            )
        return tau_frg, tau_gmi, {"mode": "summary_json", "path": os.path.abspath(args.controller_summary_json)}

    if args.tau_frg is not None and args.tau_gmi is not None:
        return float(args.tau_frg), float(args.tau_gmi), {"mode": "manual"}

    if args.per_case_csv and args.features_csv:
        controller_cfg = HardVetoConfig(
            frg_col=args.frg_col,
            gmi_col=args.gmi_col,
            improvement_case_value=args.improvement_case_value,
            regression_case_value=args.regression_case_value,
            fallback_when_missing_feature="method",
            tau_frg=None,
            tau_gmi=None,
            calibration=ThresholdCalibrationConfig(
                calib_ratio=float(args.calib_ratio),
                seed=int(args.seed),
                lambda_improvement=float(args.lambda_improvement),
                max_wrong_veto_rate=float(args.max_wrong_veto_rate),
                q_grid=[float(x.strip()) for x in str(args.q_grid).split(",") if x.strip()],
            ),
        )
        adapter = VGAOfflineAdapter(args.per_case_csv, args.features_csv)
        merged = adapter.load_merged(controller_cfg=controller_cfg)
        _, summary = run_offline_hard_veto(merged_df=merged, schema=adapter.schema, controller_cfg=controller_cfg)
        return (
            float(summary["thresholds"]["tau_frg"]),
            float(summary["thresholds"]["tau_gmi"]),
            {"mode": "offline_calibrated", "summary": summary},
        )

    raise ValueError("Provide either --controller_summary_json, both --tau_frg/--tau_gmi, or both --per_case_csv/--features_csv.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run online hard-veto probe-and-route controller.")
    ap.add_argument("--backend", type=str, default="vga", choices=["vga"])
    ap.add_argument("--vga_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--controller_summary_json", type=str, default="")
    ap.add_argument("--per_case_csv", type=str, default="")
    ap.add_argument("--features_csv", type=str, default="")
    ap.add_argument("--tau_frg", type=float, default=None)
    ap.add_argument("--tau_gmi", type=float, default=None)
    ap.add_argument("--frg_col", type=str, default="faithful_minus_global_attn")
    ap.add_argument("--gmi_col", type=str, default="guidance_mismatch_score")
    ap.add_argument("--improvement_case_value", type=str, default="vga_improvement")
    ap.add_argument("--regression_case_value", type=str, default="vga_regression")
    ap.add_argument("--calib_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_improvement", type=float, default=1.0)
    ap.add_argument("--max_wrong_veto_rate", type=float, default=0.35)
    ap.add_argument("--q_grid", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--sampling", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--max_gen_len", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--cd_alpha", type=float, default=0.02)
    ap.add_argument("--attn_coef", type=float, default=0.2)
    ap.add_argument("--start_layer", type=int, default=16)
    ap.add_argument("--end_layer", type=int, default=24)
    ap.add_argument("--head_balancing", type=str, default="simg")
    ap.add_argument("--attn_norm", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--probe_feature_mode", type=str, default="static_headset", choices=["static_headset", "aggregate"])
    ap.add_argument(
        "--aggregate_frg_metric",
        type=str,
        default="frg_shared_topk",
        choices=["frg_shared_topk", "frg_shared_mean", "c_agg_cos", "c_agg_ip"],
    )
    ap.add_argument(
        "--aggregate_gmi_metric",
        type=str,
        default="e_agg_js",
        choices=["e_agg_js", "e_agg_combo", "c_agg_cos", "c_agg_ip"],
    )
    ap.add_argument("--aggregate_topk", type=int, default=5)
    ap.add_argument("--aggregate_lambda", type=float, default=1.0)
    ap.add_argument("--headset_json", type=str, default="")
    ap.add_argument("--probe_position_mode", type=str, default="prompt_last", choices=["prompt_last", "baseline_yesno_preview"])
    ap.add_argument("--probe_preview_max_new_tokens", type=int, default=3)
    ap.add_argument("--probe_preview_reuse_baseline", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_preview_fallback_to_prompt_last", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--use_gmi", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--gt_csv", type=str, default="")
    ap.add_argument("--gt_id_col", type=str, default="id")
    ap.add_argument("--gt_label_col", type=str, default="answer")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tau_frg, tau_gmi, tau_info = resolve_thresholds(args)

    adapter = VGAOnlineAdapter(
        VGAOnlineConfig(
            vga_root=args.vga_root,
            model_path=args.model_path,
            image_folder=args.image_folder,
            conv_mode=args.conv_mode,
            model_base=args.model_base,
            device=args.device,
            temperature=args.temperature,
            top_p=args.top_p,
            sampling=args.sampling,
            max_gen_len=args.max_gen_len,
            num_beams=args.num_beams,
            cd_alpha=args.cd_alpha,
            attn_coef=args.attn_coef,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            head_balancing=args.head_balancing,
            attn_norm=args.attn_norm,
            late_start=args.late_start,
            late_end=args.late_end,
            probe_feature_mode=args.probe_feature_mode,
            aggregate_frg_metric=args.aggregate_frg_metric,
            aggregate_gmi_metric=args.aggregate_gmi_metric,
            aggregate_topk=args.aggregate_topk,
            aggregate_lambda=args.aggregate_lambda,
            headset_json=args.headset_json,
            probe_position_mode=args.probe_position_mode,
            probe_preview_max_new_tokens=args.probe_preview_max_new_tokens,
            probe_preview_reuse_baseline=bool(args.probe_preview_reuse_baseline),
            probe_preview_fallback_to_prompt_last=bool(args.probe_preview_fallback_to_prompt_last),
            use_gmi=bool(args.use_gmi),
            seed=args.seed,
        )
    )

    samples = load_jsonl(args.question_file)
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]

    preds: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    for sample in samples:
        probe = adapter.probe(sample)
        veto = bool((probe.frg >= tau_frg) or (probe.gmi >= tau_gmi))
        if veto:
            pred = adapter.predict_base(sample, probe_state=probe)
            route = "baseline"
        else:
            pred = adapter.predict_method(sample, probe_state=probe)
            route = "method"
        pred["controller_route"] = route
        pred["frg"] = float(probe.frg)
        pred["gmi"] = float(probe.gmi)
        pred["tau_frg"] = float(tau_frg)
        pred["tau_gmi"] = float(tau_gmi)
        preds.append(pred)

        extras = probe.extras
        route_rows.append(
            {
                "id": probe.sample_id,
                "route": route,
                "veto": int(veto),
                "frg": float(probe.frg),
                "gmi": float(probe.gmi),
                "probe_feature_mode": str(extras.get("probe_feature_mode", "")),
                "probe_position_mode": str(extras.get("probe_position_mode", "")),
                "probe_source": str(extras.get("probe_source", "")),
                "tau_frg": float(tau_frg),
                "tau_gmi": float(tau_gmi),
                "g_top5_mass": float(extras.get("g_top5_mass", 0.0)),
                "guidance_mode": extras.get("guidance_mode", ""),
                "image_start": int(extras.get("image_start", -1)),
                "image_end": int(extras.get("image_end", -1)),
                "probe_anchor": str(extras.get("probe_anchor", "")),
                "probe_anchor_token_idx": int(extras.get("probe_anchor_token_idx", -1)),
                "baseline_preview_found_anchor": int(bool(extras.get("baseline_preview_found_anchor", False))),
                "baseline_preview_fallback": int(bool(extras.get("baseline_preview_fallback", False))),
                "baseline_preview_reusable": int(bool(extras.get("baseline_preview_reusable", False))),
                "baseline_preview_text": str(extras.get("baseline_preview_text", "")),
                "late_head_vis_ratio_mean": float(extras.get("attn_stats", {}).get("late_head_vis_ratio_mean", 0.0)),
                "late_head_vis_ratio_topkmean": float(extras.get("attn_stats", {}).get("late_head_vis_ratio_topkmean", 0.0)),
                "frg_shared_mean": float(extras.get("attn_stats", {}).get("frg_shared_mean", 0.0)),
                "frg_shared_topk": float(extras.get("attn_stats", {}).get("frg_shared_topk", 0.0)),
                "c_agg_cos": float(extras.get("attn_stats", {}).get("c_agg_cos", 0.0)),
                "c_agg_ip": float(extras.get("attn_stats", {}).get("c_agg_ip", 0.0)),
                "e_agg_js": float(extras.get("attn_stats", {}).get("e_agg_js", 0.0)),
                "e_agg_combo": float(extras.get("attn_stats", {}).get("e_agg_combo", 0.0)),
                "topk_guidance_coverage": float(extras.get("attn_stats", {}).get("topk_guidance_coverage", 0.0)),
            }
        )

    pred_jsonl = os.path.join(args.out_dir, "pred_online_controller.jsonl")
    with open(pred_jsonl, "w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    route_csv = os.path.join(args.out_dir, "route_log.csv")
    if route_rows:
        keys = list(route_rows[0].keys())
        with open(route_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for row in route_rows:
                wr.writerow(row)

    summary: Dict[str, Any] = {
        "inputs": {
            "backend": args.backend,
            "vga_root": os.path.abspath(args.vga_root),
            "model_path": args.model_path,
            "image_folder": os.path.abspath(args.image_folder),
            "question_file": os.path.abspath(args.question_file),
            "probe_feature_mode": args.probe_feature_mode,
            "aggregate_frg_metric": args.aggregate_frg_metric,
            "aggregate_gmi_metric": args.aggregate_gmi_metric,
            "aggregate_topk": int(args.aggregate_topk),
            "aggregate_lambda": float(args.aggregate_lambda),
            "headset_json": os.path.abspath(args.headset_json) if str(args.headset_json).strip() else "",
            "probe_position_mode": args.probe_position_mode,
            "probe_preview_max_new_tokens": int(args.probe_preview_max_new_tokens),
            "probe_preview_reuse_baseline": bool(args.probe_preview_reuse_baseline),
            "probe_preview_fallback_to_prompt_last": bool(args.probe_preview_fallback_to_prompt_last),
            "use_gmi": bool(args.use_gmi),
            "max_samples": int(args.max_samples),
        },
        "thresholds": {
            "tau_frg": float(tau_frg),
            "tau_gmi": float(tau_gmi),
            "source": tau_info,
        },
        "counts": {
            "n_samples": int(len(samples)),
            "veto_count": int(sum(int(r["veto"]) for r in route_rows)),
            "veto_rate": float(sum(int(r["veto"]) for r in route_rows) / max(1, len(route_rows))),
        },
        "outputs": {
            "pred_jsonl": pred_jsonl,
            "route_csv": route_csv,
        },
    }

    if args.gt_csv:
        gt_map = load_gt_csv(args.gt_csv, id_col=args.gt_id_col, label_col=args.gt_label_col)
        summary["metrics"] = compute_metrics(gt_map, preds)

    summary_json = os.path.join(args.out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", pred_jsonl)
    print("[saved]", route_csv)
    print("[saved]", summary_json)
    if "metrics" in summary:
        print(
            "[summary]",
            json.dumps(
                {
                    "acc": summary["metrics"]["acc"],
                    "f1": summary["metrics"]["f1"],
                    "veto_rate": summary["counts"]["veto_rate"],
                },
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
