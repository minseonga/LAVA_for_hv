#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pnp_controller.adapters.vga_online import VGAOnlineAdapter, VGAOnlineConfig


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


def build_branch_text_map(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in load_jsonl(path):
        sid = sample_id(row)
        if sid is None or str(sid).strip() == "":
            continue
        text = str(row.get("output", row.get("text", ""))).strip()
        if str(sid).strip() != "" and text != "":
            out[str(sid)] = text
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


def load_router(router_dir: str) -> tuple[Any, Dict[str, Any]]:
    router_dir = os.path.abspath(router_dir)
    model_path = os.path.join(router_dir, "router_model.pkl")
    metadata_path = os.path.join(router_dir, "router_metadata.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing router model pickle: {model_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing router metadata JSON: {metadata_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def parse_anchor_yes(text: Any) -> int:
    return 1 if str(text or "").strip().lower() == "yes" else 0


def build_router_features(probe, router_meta: Dict[str, Any]) -> Dict[str, float]:
    extras = probe.extras if probe is not None else {}
    frg_off = float(probe.frg)
    g_top5_mass = float(extras.get("g_top5_mass", 0.0))
    probe_anchor_yes = int(parse_anchor_yes(extras.get("probe_anchor", "")))
    tau = float(router_meta.get("tau", 0.0))
    feature_map = {
        "frg_off": frg_off,
        "g_top5_mass": g_top5_mass,
        "probe_anchor_yes": float(probe_anchor_yes),
        "abs_frg_to_tau": abs(frg_off - tau),
        "frg_x_mass": frg_off * g_top5_mass,
    }
    return feature_map


def main() -> None:
    ap = argparse.ArgumentParser(description="Run actual online cost-aware gain routing with a saved utility-router artifact.")
    ap.add_argument("--backend", type=str, default="vga", choices=["vga"])
    ap.add_argument("--router_dir", type=str, required=True)
    ap.add_argument("--vga_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")

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
        default="baseline_yesno_preview",
        choices=["prompt_last", "baseline_yesno_preview", "baseline_yesno_offline_fullseq"],
    )
    ap.add_argument(
        "--probe_branch_source",
        type=str,
        default="preview",
        choices=["preview", "baseline_output", "baseline_jsonl"],
    )
    ap.add_argument("--branch_text_jsonl", type=str, default="")
    ap.add_argument("--probe_preview_max_new_tokens", type=int, default=3)
    ap.add_argument("--probe_preview_reuse_baseline", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_preview_fallback_to_prompt_last", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_force_manual_fullseq", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--use_gmi", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--gt_csv", type=str, default="")
    ap.add_argument("--gt_id_col", type=str, default="id")
    ap.add_argument("--gt_label_col", type=str, default="answer")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    router_model, router_meta = load_router(args.router_dir)
    feature_cols = [str(x) for x in router_meta.get("feature_cols", [])]
    if not feature_cols:
        raise ValueError("Router metadata is missing feature_cols.")
    router_cutoff = float(router_meta["deployment_cutoff"])
    branch_text_map: Dict[str, str] = {}
    if str(args.branch_text_jsonl).strip():
        branch_text_map = build_branch_text_map(args.branch_text_jsonl)

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
    route_rows: List[Dict[str, Any]] = []
    for sample in tqdm(samples, total=len(samples), desc="cost-aware-online", dynamic_ncols=True):
        sid = sample_id(sample)
        baseline_pred = None
        probe_branch_origin = "adapter_internal"
        branch_source = str(args.probe_branch_source).strip().lower()
        if branch_source == "baseline_output":
            baseline_pred = adapter.predict_base_direct(sample)
            baseline_pred["branch_source"] = "live_baseline_output"
            probe = adapter.probe(sample, branch_text=str(baseline_pred.get("output", "")))
            probe_branch_origin = "live_baseline_output"
        elif branch_source == "baseline_jsonl":
            branch_text = str(branch_text_map.get(sid, "")).strip()
            if branch_text == "":
                raise RuntimeError(f"Missing branch text for sample {sid} in --branch_text_jsonl.")
            probe = adapter.probe(sample, branch_text=branch_text)
            baseline_pred = adapter._build_prediction_row(
                sample=sample,
                prepared=probe.extras["prepared"],
                output_text=branch_text,
                use_add=False,
            )
            baseline_pred["branch_source"] = "external_probe_branch"
            probe_branch_origin = "external_probe_branch"
        else:
            probe = adapter.probe(sample)

        feature_map = build_router_features(probe=probe, router_meta=router_meta)
        feat_vec = [[float(feature_map[col]) for col in feature_cols]]
        utility_score = float(router_model.predict(feat_vec)[0])
        route = "method" if utility_score >= router_cutoff else "baseline"

        if route == "baseline":
            pred = baseline_pred if baseline_pred is not None else adapter.predict_base(sample, probe_state=probe)
        else:
            pred = adapter.predict_method(sample, probe_state=probe)

        pred["controller_route"] = route
        pred["utility_score"] = utility_score
        pred["router_cutoff"] = router_cutoff
        pred["router_feature_variant"] = str(router_meta.get("feature_variant", ""))
        preds.append(pred)

        extras = probe.extras
        row = {
            "id": probe.sample_id,
            "route": route,
            "use_method": int(route == "method"),
            "utility_score": utility_score,
            "router_cutoff": router_cutoff,
            "router_type": str(router_meta.get("router_type", "")),
            "feature_variant": str(router_meta.get("feature_variant", "")),
            "probe_feature_mode": str(extras.get("probe_feature_mode", "")),
            "probe_position_mode": str(extras.get("probe_position_mode", "")),
            "probe_branch_source": str(extras.get("probe_branch_source", "")),
            "probe_force_manual_fullseq": int(bool(extras.get("probe_force_manual_fullseq", False))),
            "probe_branch_origin": probe_branch_origin,
            "probe_source": str(extras.get("probe_source", "")),
            "probe_impl": str(extras.get("probe_impl", "")),
            "frg": float(probe.frg),
            "gmi": float(probe.gmi),
            "g_top5_mass": float(extras.get("g_top5_mass", 0.0)),
            "probe_anchor": str(extras.get("probe_anchor", "")),
            "probe_anchor_token_idx": int(extras.get("probe_anchor_token_idx", -1)),
            "probe_decision_pos": int(extras.get("probe_decision_pos", -1)),
            "baseline_preview_found_anchor": int(bool(extras.get("baseline_preview_found_anchor", False))),
            "baseline_preview_fallback": int(bool(extras.get("baseline_preview_fallback", False))),
            "baseline_preview_reusable": int(bool(extras.get("baseline_preview_reusable", False))),
            "probe_branch_text": str(extras.get("probe_branch_text", "")),
        }
        for col in feature_cols:
            row[col] = float(feature_map[col])
        route_rows.append(row)

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
            "router_dir": os.path.abspath(args.router_dir),
            "model_path": args.model_path,
            "image_folder": os.path.abspath(args.image_folder),
            "question_file": os.path.abspath(args.question_file),
            "headset_json": os.path.abspath(args.headset_json) if str(args.headset_json).strip() else "",
            "probe_feature_mode": args.probe_feature_mode,
            "probe_position_mode": args.probe_position_mode,
            "probe_branch_source": args.probe_branch_source,
            "probe_force_manual_fullseq": bool(args.probe_force_manual_fullseq),
            "branch_text_jsonl": os.path.abspath(args.branch_text_jsonl) if str(args.branch_text_jsonl).strip() else "",
            "probe_preview_max_new_tokens": int(args.probe_preview_max_new_tokens),
            "probe_preview_reuse_baseline": bool(args.probe_preview_reuse_baseline),
            "probe_preview_fallback_to_prompt_last": bool(args.probe_preview_fallback_to_prompt_last),
            "max_samples": int(args.max_samples),
        },
        "router": {
            "router_type": str(router_meta.get("router_type", "")),
            "backend_name": str(router_meta.get("backend_name", "")),
            "feature_variant": str(router_meta.get("feature_variant", "")),
            "feature_cols": feature_cols,
            "tau": float(router_meta.get("tau", 0.0)),
            "deployment_budget": float(router_meta.get("deployment_budget", 0.0)),
            "deployment_cutoff": router_cutoff,
            "score_policy": str(router_meta.get("score_policy", "")),
        },
        "counts": {
            "n_samples": int(len(samples)),
            "method_count": int(sum(int(r["use_method"]) for r in route_rows)),
            "baseline_count": int(sum(1 - int(r["use_method"]) for r in route_rows)),
            "method_rate": float(sum(int(r["use_method"]) for r in route_rows) / max(1, len(route_rows))),
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
                    "method_rate": summary["counts"]["method_rate"],
                },
                ensure_ascii=False,
            ),
        )


if __name__ == "__main__":
    main()
