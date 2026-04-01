#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pnp_controller.adapters.vga_online import VGAOnlineAdapter, VGAOnlineConfig


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            t = ln.strip()
            if t:
                rows.append(json.loads(t))
    return rows


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_branch_text_map(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in load_jsonl(path):
        sid = row.get("question_id", row.get("id", row.get("qid", None)))
        if sid is None:
            continue
        out[str(sid)] = str(row.get("output", row.get("text", ""))).strip()
    return out


def first_row_by_id(rows: List[Dict[str, str]], sid: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("id", "")) == str(sid):
            return row
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump one-sample offline trace meta and runtime helper meta side by side.")
    ap.add_argument("--id", type=str, required=True)
    ap.add_argument("--vga_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--offline_per_head_trace_csv", type=str, required=True)
    ap.add_argument("--offline_per_layer_trace_csv", type=str, required=True)
    ap.add_argument("--offline_features_csv", type=str, default="")
    ap.add_argument("--taxonomy_csv", type=str, default="")
    ap.add_argument("--headset_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--sampling", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--max_gen_len", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--cd_alpha", type=float, default=0.02)
    ap.add_argument("--attn_coef", type=float, default=0.2)
    ap.add_argument("--start_layer", type=int, default=2)
    ap.add_argument("--end_layer", type=int, default=15)
    ap.add_argument("--head_balancing", type=str, default="simg")
    ap.add_argument("--attn_norm", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--probe_feature_mode", type=str, default="static_headset")
    ap.add_argument("--probe_preview_max_new_tokens", type=int, default=3)
    ap.add_argument("--probe_preview_reuse_baseline", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_preview_fallback_to_prompt_last", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--probe_force_manual_fullseq", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument(
        "--probe_branch_source",
        type=str,
        default="preview",
        choices=["preview", "baseline_output", "baseline_jsonl"],
    )
    ap.add_argument("--branch_text_jsonl", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sid = str(args.id)
    samples = {str(row.get("id", row.get("question_id", ""))): row for row in load_jsonl(args.question_file)}
    if sid not in samples:
        raise SystemExit(f"id={sid} not found in question_file")
    sample = samples[sid]

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
            headset_json=args.headset_json,
            probe_position_mode="baseline_yesno_offline_fullseq",
            probe_force_manual_fullseq=bool(args.probe_force_manual_fullseq),
            probe_preview_max_new_tokens=args.probe_preview_max_new_tokens,
            probe_preview_reuse_baseline=bool(args.probe_preview_reuse_baseline),
            probe_preview_fallback_to_prompt_last=bool(args.probe_preview_fallback_to_prompt_last),
            seed=args.seed,
        )
    )
    branch_source = str(args.probe_branch_source).strip().lower()
    branch_text: str | None = None
    if branch_source == "baseline_output":
        baseline_pred = adapter.predict_base_direct(sample)
        branch_text = str(baseline_pred.get("output", ""))
    elif branch_source == "baseline_jsonl":
        if str(args.branch_text_jsonl).strip() == "":
            raise SystemExit("--branch_text_jsonl is required when --probe_branch_source=baseline_jsonl")
        branch_text_map = build_branch_text_map(args.branch_text_jsonl)
        branch_text = str(branch_text_map.get(sid, ""))
    debug_pack = adapter.debug_probe_baseline_yesno_offline_fullseq(
        sample=sample,
        branch_text=branch_text,
        branch_source=branch_source,
    )

    offline_head_row = first_row_by_id(load_csv(args.offline_per_head_trace_csv), sid)
    offline_layer_row = first_row_by_id(load_csv(args.offline_per_layer_trace_csv), sid)
    offline_feat_row = first_row_by_id(load_csv(args.offline_features_csv), sid) if str(args.offline_features_csv).strip() else {}
    tax_row = first_row_by_id(load_csv(args.taxonomy_csv), sid) if str(args.taxonomy_csv).strip() else {}

    debug = debug_pack["debug"]
    out = {
        "sample_id": sid,
        "question_file_sample": sample,
        "offline_trace_meta": {
            "per_head_first_row": offline_head_row,
            "per_layer_first_row": offline_layer_row,
            "offline_feature_row_subset": {
                "faithful_minus_global_attn": offline_feat_row.get("faithful_minus_global_attn", ""),
                "guidance_mismatch_score": offline_feat_row.get("guidance_mismatch_score", ""),
                "question": offline_feat_row.get("question", ""),
                "image_id": offline_feat_row.get("image_id", ""),
                "target_pred_answer_eval": offline_feat_row.get("target_pred_answer_eval", ""),
            },
            "taxonomy_row": tax_row,
        },
        "runtime_helper_meta": {
            "question": debug_pack["question"],
            "image_id": debug_pack["image_id"],
            "object_phrase": debug_pack["object_phrase"],
            "guidance_mode": debug_pack["guidance_mode"],
            "g_top5_mass": debug_pack["g_top5_mass"],
            "runtime_frg": debug_pack["frg"],
            "runtime_gmi": debug_pack["gmi"],
            "debug": debug,
            "attn_stats": debug_pack["attn_stats"],
        },
        "quick_compare": {
            "offline_anchor_phrase": offline_head_row.get("anchor_phrase", ""),
            "runtime_probe_anchor": debug.get("probe_anchor", ""),
            "offline_yesno_token_idx": offline_head_row.get("yesno_token_idx", ""),
            "runtime_probe_anchor_token_idx": debug.get("probe_anchor_token_idx", ""),
            "offline_yesno_token_str": offline_head_row.get("yesno_token_str", ""),
            "runtime_probe_anchor_token_str": debug.get("probe_anchor_token_str", ""),
            "offline_pred_text": offline_head_row.get("pred_text", ""),
            "runtime_preview_text": debug.get("baseline_preview_text", ""),
            "runtime_probe_decision_pos": debug.get("probe_decision_pos", ""),
            "runtime_probe_decision_positions": debug.get("probe_decision_positions", []),
            "runtime_probe_cont_ids": debug.get("probe_cont_ids", []),
            "runtime_probe_cont_token_strs": debug.get("probe_cont_token_strs", []),
        },
    }

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[saved]", out_json)


if __name__ == "__main__":
    main()
