#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pnp_controller.adapters.vga_online import VGAOnlineAdapter, VGAOnlineConfig


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            t = ln.strip()
            if t:
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


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def parse_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if str(args.ids).strip():
        ids.extend([x.strip() for x in str(args.ids).split(",") if x.strip()])
    if str(args.ids_csv).strip():
        rows = load_csv_rows(args.ids_csv)
        for row in rows:
            sid = str(row.get(args.ids_col, "")).strip()
            if sid:
                ids.append(sid)
    seen = set()
    out: List[str] = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    if int(args.max_ids) > 0:
        out = out[: int(args.max_ids)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump runtime helper decision-row attention and align it with offline per-head trace.")
    ap.add_argument("--vga_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--offline_per_head_trace_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--ids", type=str, default="")
    ap.add_argument("--ids_csv", type=str, default="")
    ap.add_argument("--ids_col", type=str, default="id")
    ap.add_argument("--max_ids", type=int, default=-1)
    ap.add_argument("--offline_features_csv", type=str, default="")
    ap.add_argument("--taxonomy_csv", type=str, default="")
    ap.add_argument("--offline_branch_text_jsonl", type=str, default="")
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
    ap.add_argument("--headset_json", type=str, required=True)
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
    ap.add_argument("--head_layer_start", type=int, default=10)
    ap.add_argument("--head_layer_end", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    selected_ids = parse_ids(args)
    if not selected_ids:
        raise SystemExit("No IDs provided. Use --ids or --ids_csv.")

    samples = {str(row.get("id", row.get("question_id", ""))): row for row in load_jsonl(args.question_file)}
    missing_ids = [sid for sid in selected_ids if sid not in samples]
    if missing_ids:
        raise SystemExit(f"Some requested ids are missing from question_file: {missing_ids[:10]}")
    branch_text_map: Dict[str, str] = {}
    if str(args.branch_text_jsonl).strip():
        branch_text_map = build_branch_text_map(args.branch_text_jsonl)
    if str(args.probe_branch_source).strip().lower() == "baseline_jsonl" and not branch_text_map:
        raise SystemExit("--branch_text_jsonl is required when --probe_branch_source=baseline_jsonl")
    offline_branch_text_map: Dict[str, str] = {}
    if str(args.offline_branch_text_jsonl).strip():
        offline_branch_text_map = build_branch_text_map(args.offline_branch_text_jsonl)

    offline_trace_rows = load_csv_rows(args.offline_per_head_trace_csv)
    offline_trace_filtered = [row for row in offline_trace_rows if str(row.get("id", "")) in set(selected_ids)]
    offline_trace_map = {
        (str(row["id"]), int(row["block_layer_idx"]), int(row["head_idx"])): row
        for row in offline_trace_filtered
    }

    offline_feat_map: Dict[str, Dict[str, str]] = {}
    if str(args.offline_features_csv).strip():
        offline_feat_map = {str(r["id"]): r for r in load_csv_rows(args.offline_features_csv)}

    tax_map: Dict[str, Dict[str, str]] = {}
    if str(args.taxonomy_csv).strip():
        tax_map = {str(r["id"]): r for r in load_csv_rows(args.taxonomy_csv)}

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

    runtime_rows: List[Dict[str, Any]] = []
    aligned_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    per_sample_json_dir = os.path.join(args.out_dir, "per_sample_json")
    os.makedirs(per_sample_json_dir, exist_ok=True)

    for sid in selected_ids:
        sample = samples[sid]
        branch_source = str(args.probe_branch_source).strip().lower()
        branch_text: str | None = None
        if branch_source == "baseline_output":
            baseline_pred = adapter.predict_base_direct(sample)
            branch_text = str(baseline_pred.get("output", ""))
        elif branch_source == "baseline_jsonl":
            branch_text = str(branch_text_map.get(sid, ""))
            if branch_text.strip() == "":
                raise SystemExit(f"Missing branch text for id={sid} in --branch_text_jsonl")
        debug_pack = adapter.debug_probe_baseline_yesno_offline_fullseq(
            sample=sample,
            head_layer_start=args.head_layer_start,
            head_layer_end=args.head_layer_end,
            branch_text=branch_text,
            branch_source=branch_source,
        )
        debug = debug_pack["debug"]
        offline_feat = offline_feat_map.get(sid, {})
        tax = tax_map.get(sid, {})
        runtime_branch_text = str(debug.get("probe_branch_text", ""))
        offline_branch_text = str(offline_branch_text_map.get(sid, ""))
        runtime_branch_label = parse_yes_no(runtime_branch_text) if runtime_branch_text.strip() else ""
        offline_branch_label = parse_yes_no(offline_branch_text) if offline_branch_text.strip() else ""
        sample_runtime_rows: List[Dict[str, Any]] = []
        sample_offline_rows = [row for row in offline_trace_filtered if str(row.get("id", "")) == sid]
        sample_aligned_rows: List[Dict[str, Any]] = []
        sample_rows.append(
            {
                "id": sid,
                "question": debug_pack["question"],
                "image_id": debug_pack["image_id"],
                "object_phrase": debug_pack["object_phrase"],
                "probe_impl": debug.get("probe_impl", ""),
                "probe_impl_error": debug.get("probe_impl_error", ""),
                "probe_branch_source": debug.get("probe_branch_source", ""),
                "probe_branch_text": debug.get("probe_branch_text", ""),
                "probe_branch_label": runtime_branch_label,
                "offline_branch_text": offline_branch_text,
                "offline_branch_label": offline_branch_label,
                "branch_text_exact_match": int(runtime_branch_text == offline_branch_text) if offline_branch_text.strip() else "",
                "branch_label_match": int(runtime_branch_label == offline_branch_label) if offline_branch_label else "",
                "probe_anchor": debug.get("probe_anchor", ""),
                "probe_anchor_token_idx": debug.get("probe_anchor_token_idx", -1),
                "probe_cont_len": debug.get("probe_cont_len", -1),
                "probe_cont_label_positions": json.dumps(debug.get("probe_cont_label_positions", []), ensure_ascii=False),
                "probe_decision_pos": debug.get("probe_decision_pos", -1),
                "probe_decision_positions": json.dumps(debug.get("probe_decision_positions", []), ensure_ascii=False),
                "vision_positions": json.dumps(debug.get("vision_positions", []), ensure_ascii=False),
                "text_positions": json.dumps(debug.get("text_positions", []), ensure_ascii=False),
                "baseline_preview_fallback": int(bool(debug.get("baseline_preview_fallback", False))),
                "baseline_preview_text": debug.get("baseline_preview_text", ""),
                "guidance_mode": debug_pack["guidance_mode"],
                "g_top5_mass": debug_pack["g_top5_mass"],
                "runtime_frg": debug_pack["frg"],
                "runtime_gmi": debug_pack["gmi"],
                "offline_c": offline_feat.get("faithful_minus_global_attn", ""),
                "offline_guidance_mismatch": offline_feat.get("guidance_mismatch_score", ""),
                "case_type": tax.get("case_type", ""),
                "pred_baseline": tax.get("pred_baseline", ""),
                "pred_vga": tax.get("pred_vga", ""),
                "gt": tax.get("gt", ""),
            }
        )

        for row in debug_pack["per_head_rows"]:
            sample_runtime_rows.append(row)
            runtime_rows.append(row)
            key = (str(row["id"]), int(row["block_layer_idx"]), int(row["head_idx"]))
            offline = offline_trace_map.get(key)
            if offline is None:
                continue
            aligned = {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "image_id": str(row["image_id"]),
                "block_layer_idx": int(row["block_layer_idx"]),
                "head_idx": int(row["head_idx"]),
                "probe_impl": str(row.get("probe_impl", "")),
                "probe_anchor": str(row.get("anchor_phrase", "")),
                "probe_decision_pos": int(row.get("probe_decision_pos", -1)),
                "offline_head_attn_vis_sum": float(offline["head_attn_vis_sum"]),
                "runtime_head_attn_vis_sum": float(row["head_attn_vis_sum"]),
                "diff_head_attn_vis_sum": float(row["head_attn_vis_sum"]) - float(offline["head_attn_vis_sum"]),
                "offline_head_attn_vis_ratio": float(offline["head_attn_vis_ratio"]),
                "runtime_head_attn_vis_ratio": float(row["head_attn_vis_ratio"]),
                "diff_head_attn_vis_ratio": float(row["head_attn_vis_ratio"]) - float(offline["head_attn_vis_ratio"]),
                "offline_head_attn_vis_peak": float(offline["head_attn_vis_peak"]),
                "runtime_head_attn_vis_peak": float(row["head_attn_vis_peak"]),
                "diff_head_attn_vis_peak": float(row["head_attn_vis_peak"]) - float(offline["head_attn_vis_peak"]),
                "offline_head_attn_vis_entropy": float(offline["head_attn_vis_entropy"]),
                "runtime_head_attn_vis_entropy": float(row["head_attn_vis_entropy"]),
                "diff_head_attn_vis_entropy": float(row["head_attn_vis_entropy"]) - float(offline["head_attn_vis_entropy"]),
            }
            aligned_rows.append(aligned)
            sample_aligned_rows.append(aligned)

        if sample_aligned_rows:
            mean_abs_ratio_diff = sum(abs(float(r["diff_head_attn_vis_ratio"])) for r in sample_aligned_rows) / len(sample_aligned_rows)
            mean_abs_sum_diff = sum(abs(float(r["diff_head_attn_vis_sum"])) for r in sample_aligned_rows) / len(sample_aligned_rows)
            sample_rows[-1]["n_runtime_head_rows"] = int(len(sample_runtime_rows))
            sample_rows[-1]["n_offline_head_rows"] = int(len(sample_offline_rows))
            sample_rows[-1]["n_aligned_head_rows"] = int(len(sample_aligned_rows))
            sample_rows[-1]["mean_abs_diff_head_attn_vis_ratio"] = float(mean_abs_ratio_diff)
            sample_rows[-1]["mean_abs_diff_head_attn_vis_sum"] = float(mean_abs_sum_diff)
        else:
            sample_rows[-1]["n_runtime_head_rows"] = int(len(sample_runtime_rows))
            sample_rows[-1]["n_offline_head_rows"] = int(len(sample_offline_rows))
            sample_rows[-1]["n_aligned_head_rows"] = 0
            sample_rows[-1]["mean_abs_diff_head_attn_vis_ratio"] = ""
            sample_rows[-1]["mean_abs_diff_head_attn_vis_sum"] = ""

        sample_json = {
            "id": sid,
            "question_file_sample": sample,
            "runtime_debug_pack": debug_pack,
            "offline_feature_row": offline_feat,
            "taxonomy_row": tax,
            "runtime_branch_text": runtime_branch_text,
            "runtime_branch_label": runtime_branch_label,
            "offline_branch_text": offline_branch_text,
            "offline_branch_label": offline_branch_label,
            "branch_text_exact_match": bool(runtime_branch_text == offline_branch_text) if offline_branch_text.strip() else None,
            "branch_label_match": bool(runtime_branch_label == offline_branch_label) if offline_branch_label else None,
            "offline_head_rows": sample_offline_rows,
            "runtime_head_rows": sample_runtime_rows,
            "aligned_head_rows": sample_aligned_rows,
        }
        with open(os.path.join(per_sample_json_dir, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)

    runtime_csv = os.path.join(args.out_dir, "runtime_probe_head_rows.csv")
    offline_csv = os.path.join(args.out_dir, "offline_trace_head_rows.csv")
    aligned_csv = os.path.join(args.out_dir, "aligned_head_comparison.csv")
    sample_csv = os.path.join(args.out_dir, "sample_summary.csv")

    write_csv(runtime_csv, runtime_rows)
    write_csv(offline_csv, offline_trace_filtered)
    write_csv(aligned_csv, aligned_rows)
    write_csv(sample_csv, sample_rows)

    summary = {
        "inputs": {
            "question_file": os.path.abspath(args.question_file),
            "offline_per_head_trace_csv": os.path.abspath(args.offline_per_head_trace_csv),
            "offline_features_csv": os.path.abspath(args.offline_features_csv) if str(args.offline_features_csv).strip() else "",
            "taxonomy_csv": os.path.abspath(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else "",
            "offline_branch_text_jsonl": os.path.abspath(args.offline_branch_text_jsonl) if str(args.offline_branch_text_jsonl).strip() else "",
            "probe_branch_source": str(args.probe_branch_source),
            "branch_text_jsonl": os.path.abspath(args.branch_text_jsonl) if str(args.branch_text_jsonl).strip() else "",
            "head_layer_start": int(args.head_layer_start),
            "head_layer_end": int(args.head_layer_end),
            "selected_ids": selected_ids,
        },
        "counts": {
            "n_ids": int(len(selected_ids)),
            "n_runtime_rows": int(len(runtime_rows)),
            "n_offline_rows": int(len(offline_trace_filtered)),
            "n_aligned_rows": int(len(aligned_rows)),
        },
        "outputs": {
            "sample_summary_csv": sample_csv,
            "runtime_probe_head_rows_csv": runtime_csv,
            "offline_trace_head_rows_csv": offline_csv,
            "aligned_head_comparison_csv": aligned_csv,
            "per_sample_json_dir": per_sample_json_dir,
        },
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", summary_path)
    print("[saved]", sample_csv)
    print("[saved]", runtime_csv)
    print("[saved]", offline_csv)
    print("[saved]", aligned_csv)


if __name__ == "__main__":
    main()
