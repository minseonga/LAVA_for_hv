#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pnp_controller.adapters.vga_online import VGAOnlineAdapter, VGAOnlineConfig
from pnp_controller.adapters.vga import VGAOfflineAdapter
from pnp_controller.core.controller import run_offline_hard_veto
from pnp_controller.core.schemas import HardVetoConfig, ThresholdCalibrationConfig


def sample_id(sample: Dict[str, Any]) -> str:
    for key in ("question_id", "id", "qid", "image_id"):
        v = sample.get(key, None)
        if v is not None and str(v).strip() != "":
            return str(v)
    return ""


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


def score_from_verify(verify: Dict[str, Any], prefix: str, mode: str) -> float:
    key = f"{prefix}_{mode}"
    if key not in verify:
        raise KeyError(f"Missing verify score key: {key}")
    return float(verify[key])


def jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run method-first asymmetric FRG verifier with selective baseline rescue.")
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
    ap.add_argument(
        "--probe_position_mode",
        type=str,
        default="prompt_last",
        choices=["prompt_last", "baseline_yesno_preview", "baseline_yesno_offline_fullseq"],
    )
    ap.add_argument(
        "--probe_branch_source",
        type=str,
        default="preview",
        choices=["preview", "baseline_output"],
    )
    ap.add_argument("--probe_preview_max_new_tokens", type=int, default=3)
    ap.add_argument("--probe_preview_reuse_baseline", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_preview_fallback_to_prompt_last", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_force_manual_fullseq", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--use_gmi", type=lambda x: x.lower() == "true", default=True)

    ap.add_argument("--verify_enabled", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--verify_force_manual_fullseq", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--verify_frg_stat", type=str, default="mean_plus_std", choices=["mean", "max", "last", "mean_plus_std"])
    ap.add_argument("--verify_gmi_stat", type=str, default="mean_plus_std", choices=["mean", "max", "last", "mean_plus_std"])
    ap.add_argument("--tau_verify_frg", type=float, default=None)
    ap.add_argument("--tau_verify_gmi", type=float, default=None)
    ap.add_argument("--verify_fail_action", type=str, default="baseline", choices=["baseline", "method"])
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--gt_csv", type=str, default="")
    ap.add_argument("--gt_id_col", type=str, default="id")
    ap.add_argument("--gt_label_col", type=str, default="answer")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tau_frg, tau_gmi, tau_info = resolve_thresholds(args)
    tau_verify_frg = float(args.tau_verify_frg) if args.tau_verify_frg is not None else float(tau_frg)
    tau_verify_gmi = float(args.tau_verify_gmi) if args.tau_verify_gmi is not None else float(tau_gmi)

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
            probe_branch_source=args.probe_branch_source,
            probe_force_manual_fullseq=bool(args.probe_force_manual_fullseq),
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
    intervention_preds: List[Dict[str, Any]] = []
    rescue_preds: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    stage_a_trigger_count = 0
    verify_count = 0
    rescue_count = 0

    for sample in tqdm(samples, total=len(samples), desc="pnp-asym-verify", dynamic_ncols=True):
        probe = adapter.probe(sample)
        method_pred = adapter.predict_method(sample, probe_state=probe)
        intervention_preds.append(dict(method_pred))

        stage_a_trigger = bool((probe.frg >= tau_frg) or (probe.gmi >= tau_gmi))
        if stage_a_trigger:
            stage_a_trigger_count += 1

        verify: Dict[str, Any] = {"valid": False}
        verify_done = False
        verify_frg_score = float("nan")
        verify_gmi_score = float("nan")
        stage_b_trigger = False

        if stage_a_trigger and bool(args.verify_enabled):
            verify_done = True
            verify_count += 1
            verify = adapter.verify_candidate_path(
                sample=sample,
                candidate_text=str(method_pred.get("output", "")),
                probe_state=probe,
                force_manual_fullseq=bool(args.verify_force_manual_fullseq),
            )
            if bool(verify.get("valid", False)):
                verify_frg_score = score_from_verify(verify, "path_frg", str(args.verify_frg_stat))
                verify_gmi_score = score_from_verify(verify, "path_gmi", str(args.verify_gmi_stat))
                stage_b_trigger = bool((verify_frg_score >= tau_verify_frg) or (verify_gmi_score >= tau_verify_gmi))
            else:
                stage_b_trigger = bool(str(args.verify_fail_action).strip().lower() == "baseline")
        elif stage_a_trigger:
            stage_b_trigger = True

        if stage_a_trigger and stage_b_trigger:
            pred = adapter.predict_base(sample, probe_state=probe)
            pred["controller_route"] = "baseline_rescue"
            rescue_preds.append(dict(pred))
            rescue_count += 1
            final_route = "baseline_rescue"
        else:
            pred = dict(method_pred)
            pred["controller_route"] = "method_keep"
            final_route = "method_keep"

        pred["stage_a_frg"] = float(probe.frg)
        pred["stage_a_gmi"] = float(probe.gmi)
        pred["stage_a_trigger"] = int(stage_a_trigger)
        pred["stage_b_trigger"] = int(stage_b_trigger)
        pred["verify_done"] = int(verify_done)
        pred["tau_frg"] = float(tau_frg)
        pred["tau_gmi"] = float(tau_gmi)
        pred["tau_verify_frg"] = float(tau_verify_frg)
        pred["tau_verify_gmi"] = float(tau_verify_gmi)
        pred["verify_frg_score"] = None if math.isnan(verify_frg_score) else float(verify_frg_score)
        pred["verify_gmi_score"] = None if math.isnan(verify_gmi_score) else float(verify_gmi_score)
        preds.append(pred)

        extras = probe.extras
        route_rows.append(
            {
                "id": probe.sample_id,
                "final_route": final_route,
                "stage_a_trigger": int(stage_a_trigger),
                "stage_b_trigger": int(stage_b_trigger),
                "verify_done": int(verify_done),
                "rescue": int(final_route == "baseline_rescue"),
                "stage_a_frg": float(probe.frg),
                "stage_a_gmi": float(probe.gmi),
                "tau_frg": float(tau_frg),
                "tau_gmi": float(tau_gmi),
                "tau_verify_frg": float(tau_verify_frg),
                "tau_verify_gmi": float(tau_verify_gmi),
                "verify_frg_score": "" if math.isnan(verify_frg_score) else float(verify_frg_score),
                "verify_gmi_score": "" if math.isnan(verify_gmi_score) else float(verify_gmi_score),
                "verify_valid": int(bool(verify.get("valid", False))),
                "verify_impl": str(verify.get("debug", {}).get("verify_impl", "")),
                "verify_impl_error": str(verify.get("debug", {}).get("verify_impl_error", "")),
                "verify_n_tokens": int(verify.get("n_tokens", 0)),
                "path_frg_mean": float(verify.get("path_frg_mean", 0.0)),
                "path_frg_std": float(verify.get("path_frg_std", 0.0)),
                "path_frg_max": float(verify.get("path_frg_max", 0.0)),
                "path_frg_last": float(verify.get("path_frg_last", 0.0)),
                "path_frg_mean_plus_std": float(verify.get("path_frg_mean_plus_std", 0.0)),
                "path_gmi_mean": float(verify.get("path_gmi_mean", 0.0)),
                "path_gmi_std": float(verify.get("path_gmi_std", 0.0)),
                "path_gmi_max": float(verify.get("path_gmi_max", 0.0)),
                "path_gmi_last": float(verify.get("path_gmi_last", 0.0)),
                "path_gmi_mean_plus_std": float(verify.get("path_gmi_mean_plus_std", 0.0)),
                "probe_feature_mode": str(extras.get("probe_feature_mode", "")),
                "probe_position_mode": str(extras.get("probe_position_mode", "")),
                "probe_branch_source": str(extras.get("probe_branch_source", "")),
                "probe_force_manual_fullseq": int(bool(extras.get("probe_force_manual_fullseq", False))),
                "probe_source": str(extras.get("probe_source", "")),
                "probe_impl": str(extras.get("probe_impl", "")),
                "g_top5_mass": float(extras.get("g_top5_mass", 0.0)),
                "guidance_mode": extras.get("guidance_mode", ""),
                "baseline_preview_found_anchor": int(bool(extras.get("baseline_preview_found_anchor", False))),
                "baseline_preview_fallback": int(bool(extras.get("baseline_preview_fallback", False))),
            }
        )

    pred_jsonl = os.path.join(args.out_dir, "pred_online_asymmetric_controller.jsonl")
    intervention_jsonl = os.path.join(args.out_dir, "pred_intervention_first.jsonl")
    rescue_jsonl = os.path.join(args.out_dir, "pred_baseline_rescue.jsonl")
    route_csv = os.path.join(args.out_dir, "route_log.csv")

    jsonl_write(pred_jsonl, preds)
    jsonl_write(intervention_jsonl, intervention_preds)
    jsonl_write(rescue_jsonl, rescue_preds)
    if route_rows:
        keys = list(route_rows[0].keys())
        with open(route_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for row in route_rows:
                wr.writerow(row)

    n_samples = max(1, len(samples))
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
            "probe_branch_source": args.probe_branch_source,
            "probe_force_manual_fullseq": bool(args.probe_force_manual_fullseq),
            "probe_preview_max_new_tokens": int(args.probe_preview_max_new_tokens),
            "probe_preview_reuse_baseline": bool(args.probe_preview_reuse_baseline),
            "probe_preview_fallback_to_prompt_last": bool(args.probe_preview_fallback_to_prompt_last),
            "use_gmi": bool(args.use_gmi),
            "verify_enabled": bool(args.verify_enabled),
            "verify_force_manual_fullseq": bool(args.verify_force_manual_fullseq),
            "verify_frg_stat": str(args.verify_frg_stat),
            "verify_gmi_stat": str(args.verify_gmi_stat),
            "verify_fail_action": str(args.verify_fail_action),
            "max_samples": int(args.max_samples),
        },
        "thresholds": {
            "tau_frg": float(tau_frg),
            "tau_gmi": float(tau_gmi),
            "tau_verify_frg": float(tau_verify_frg),
            "tau_verify_gmi": float(tau_verify_gmi),
            "source": tau_info,
        },
        "counts": {
            "n_samples": int(len(samples)),
            "stage_a_trigger_count": int(stage_a_trigger_count),
            "stage_a_trigger_rate": float(stage_a_trigger_count / n_samples),
            "verify_count": int(verify_count),
            "verify_rate": float(verify_count / n_samples),
            "rescue_count": int(rescue_count),
            "rescue_rate": float(rescue_count / n_samples),
            "method_keep_count": int(len(samples) - rescue_count),
            "method_keep_rate": float((len(samples) - rescue_count) / n_samples),
        },
        "cost_proxy": {
            "method_generation_per_sample": 1.0,
            "avg_additional_verify_passes_proxy": float(verify_count / n_samples),
            "avg_additional_rescue_generations": float(rescue_count / n_samples),
            "avg_total_pass_proxy": float(1.0 + verify_count / n_samples + rescue_count / n_samples),
        },
        "outputs": {
            "pred_jsonl": pred_jsonl,
            "intervention_pred_jsonl": intervention_jsonl,
            "rescue_pred_jsonl": rescue_jsonl,
            "route_csv": route_csv,
        },
    }

    if args.gt_csv:
        gt_map = load_gt_csv(args.gt_csv, id_col=args.gt_id_col, label_col=args.gt_label_col)
        summary["metrics"] = compute_metrics(gt_map, preds)
        summary["metrics_intervention_first"] = compute_metrics(gt_map, intervention_preds)

    summary_json = os.path.join(args.out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", pred_jsonl)
    print("[saved]", intervention_jsonl)
    print("[saved]", rescue_jsonl)
    print("[saved]", route_csv)
    print("[saved]", summary_json)
    if "metrics" in summary:
        print(
            "[summary]",
            json.dumps(
                {
                    "acc": summary["metrics"]["acc"],
                    "intervention_acc": summary["metrics_intervention_first"]["acc"],
                    "rescue_rate": summary["counts"]["rescue_rate"],
                    "verify_rate": summary["counts"]["verify_rate"],
                },
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
