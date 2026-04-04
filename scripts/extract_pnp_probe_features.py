#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **_: Any):
        return iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pnp_controller.adapters.vga_online import VGAOnlineAdapter, VGAOnlineConfig


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
        sid = row.get("question_id", row.get("id", row.get("qid", None)))
        if sid is None:
            continue
        text = str(row.get("output", row.get("text", ""))).strip()
        out[str(sid)] = text
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, None) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract probe-only PnP controller features without generation.")
    ap.add_argument("--backend", type=str, default="vga", choices=["vga"])
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
    ap.add_argument("--probe_feature_mode", type=str, default="aggregate", choices=["static_headset", "aggregate"])
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
        choices=["preview", "baseline_output", "baseline_jsonl"],
    )
    ap.add_argument("--branch_text_jsonl", type=str, default="")
    ap.add_argument("--probe_preview_max_new_tokens", type=int, default=3)
    ap.add_argument("--probe_preview_reuse_baseline", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_preview_fallback_to_prompt_last", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_force_manual_fullseq", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_base = str(args.model_base or "").strip() or None

    adapter = VGAOnlineAdapter(
        VGAOnlineConfig(
            vga_root=args.vga_root,
            model_path=args.model_path,
            image_folder=args.image_folder,
            conv_mode=args.conv_mode,
            model_base=model_base,
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
            seed=args.seed,
        )
    )

    samples = load_jsonl(args.question_file)
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    branch_text_map: Dict[str, str] = {}
    if str(args.branch_text_jsonl).strip():
        branch_text_map = build_branch_text_map(args.branch_text_jsonl)

    rows: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="probe", unit="sample"):
        baseline_pred = None
        branch_source = str(args.probe_branch_source).strip().lower()
        if branch_source == "baseline_output":
            baseline_pred = adapter.predict_base_direct(sample)
            probe = adapter.probe(sample, branch_text=str(baseline_pred.get("output", "")))
        elif branch_source == "baseline_jsonl":
            sid = str(sample.get("question_id", sample.get("id", sample.get("qid", "")))).strip()
            branch_text = str(branch_text_map.get(sid, "")).strip()
            probe = adapter.probe(sample, branch_text=branch_text)
        else:
            probe = adapter.probe(sample)
        extras = probe.extras
        attn_stats = extras.get("attn_stats", {}) if isinstance(extras, dict) else {}
        row = {
            "id": probe.sample_id,
            "frg": float(probe.frg),
            "gmi": float(probe.gmi),
            "probe_feature_mode": str(extras.get("probe_feature_mode", "")),
            "probe_position_mode": str(extras.get("probe_position_mode", "")),
            "probe_branch_source": str(extras.get("probe_branch_source", "")),
            "probe_force_manual_fullseq": int(bool(extras.get("probe_force_manual_fullseq", False))),
            "probe_source": str(extras.get("probe_source", "")),
            "probe_impl": str(extras.get("probe_impl", "")),
            "probe_impl_error": str(extras.get("probe_impl_error", "")),
            "probe_branch_text": str(extras.get("probe_branch_text", "")),
            "probe_anchor": str(extras.get("probe_anchor", "")),
            "probe_anchor_token_idx": int(extras.get("probe_anchor_token_idx", -1)),
            "probe_decision_pos": int(extras.get("probe_decision_pos", -1)),
            "baseline_preview_reusable": int(bool(extras.get("baseline_preview_reusable", False))),
            "baseline_preview_found_anchor": int(bool(extras.get("baseline_preview_found_anchor", False))),
            "baseline_preview_fallback": int(bool(extras.get("baseline_preview_fallback", False))),
            "guidance_mode": extras.get("guidance_mode", ""),
            "g_top5_mass": float(extras.get("g_top5_mass", 0.0)),
            "late_head_vis_ratio_mean": float(attn_stats.get("late_head_vis_ratio_mean", 0.0)),
            "late_head_vis_ratio_topkmean": float(attn_stats.get("late_head_vis_ratio_topkmean", 0.0)),
            "frg_shared_mean": float(attn_stats.get("frg_shared_mean", 0.0)),
            "frg_shared_topk": float(attn_stats.get("frg_shared_topk", 0.0)),
            "c_agg_cos": float(attn_stats.get("c_agg_cos", 0.0)),
            "c_agg_ip": float(attn_stats.get("c_agg_ip", 0.0)),
            "e_agg_js": float(attn_stats.get("e_agg_js", 0.0)),
            "e_agg_combo": float(attn_stats.get("e_agg_combo", 0.0)),
            "topk_guidance_coverage": float(attn_stats.get("topk_guidance_coverage", 0.0)),
        }
        rows.append(row)

    csv_path = os.path.join(args.out_dir, "probe_features.csv")
    write_csv(csv_path, rows)
    probe_impl_counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("probe_impl", "") or "")
        probe_impl_counts[key] = int(probe_impl_counts.get(key, 0)) + 1
    summary = {
        "inputs": {
            "backend": args.backend,
            "question_file": os.path.abspath(args.question_file),
            "branch_text_jsonl": os.path.abspath(args.branch_text_jsonl) if str(args.branch_text_jsonl).strip() else "",
            "probe_feature_mode": args.probe_feature_mode,
            "aggregate_frg_metric": args.aggregate_frg_metric,
            "aggregate_gmi_metric": args.aggregate_gmi_metric,
            "aggregate_topk": int(args.aggregate_topk),
            "aggregate_lambda": float(args.aggregate_lambda),
            "late_start": int(args.late_start),
            "late_end": int(args.late_end),
            "headset_json": os.path.abspath(args.headset_json) if str(args.headset_json).strip() else "",
            "probe_position_mode": args.probe_position_mode,
            "probe_branch_source": args.probe_branch_source,
            "probe_force_manual_fullseq": bool(args.probe_force_manual_fullseq),
            "probe_preview_max_new_tokens": int(args.probe_preview_max_new_tokens),
            "probe_preview_reuse_baseline": bool(args.probe_preview_reuse_baseline),
            "probe_preview_fallback_to_prompt_last": bool(args.probe_preview_fallback_to_prompt_last),
            "max_samples": int(args.max_samples),
        },
        "counts": {
            "n_rows": int(len(rows)),
            "probe_impl_counts": probe_impl_counts,
        },
        "outputs": {"probe_features_csv": csv_path},
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", csv_path)
    print("[saved]", summary_path)
    print(
        "[summary]",
        json.dumps(
            {
                "n_rows": len(rows),
                "probe_feature_mode": args.probe_feature_mode,
                "probe_impl_counts": probe_impl_counts,
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
