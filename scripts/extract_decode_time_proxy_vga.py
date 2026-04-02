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
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pred_text_map(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not str(path or "").strip():
        return out
    for row in load_jsonl(path):
        sid = str(row.get("question_id", row.get("id", row.get("qid", "")))).strip()
        if not sid:
            continue
        text = str(row.get("output", row.get("text", row.get("answer", "")))).strip()
        out[sid] = text
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract 1-pass decode-time proxy summaries during VGA generation.")
    ap.add_argument("--vga_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--headset_json", type=str, required=True)
    ap.add_argument("--reference_pred_jsonl", type=str, default="")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--sampling", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--max_gen_len", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--cd_alpha", type=float, default=0.02)
    ap.add_argument("--attn_coef", type=float, default=0.2)
    ap.add_argument("--start_layer", type=int, default=16)
    ap.add_argument("--end_layer", type=int, default=24)
    ap.add_argument("--head_balancing", type=str, default="simg")
    ap.add_argument("--attn_norm", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--proxy_trace_late_start", type=int, default=16)
    ap.add_argument("--proxy_trace_late_end", type=int, default=24)
    ap.add_argument("--proxy_trace_last_k", type=int, default=8)
    ap.add_argument("--proxy_trace_margin_low", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
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
            headset_json=args.headset_json,
            seed=args.seed,
            prefer_local_llava=True,
            proxy_trace_enabled=True,
            proxy_trace_late_start=args.proxy_trace_late_start,
            proxy_trace_late_end=args.proxy_trace_late_end,
            proxy_trace_last_k=args.proxy_trace_last_k,
            proxy_trace_margin_low=args.proxy_trace_margin_low,
        )
    )

    samples = load_jsonl(args.question_file)
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    ref_pred_map = build_pred_text_map(args.reference_pred_jsonl)

    pred_rows: List[Dict[str, Any]] = []
    proxy_rows: List[Dict[str, Any]] = []
    n_errors = 0
    match_total = 0
    match_same = 0

    for idx, sample in enumerate(samples):
        sid = str(sample.get("question_id", sample.get("id", sample.get("qid", "")))).strip()
        try:
            out = adapter.predict_method_with_proxy(sample)
            pred_row = dict(out["prediction"])
            proxy_row = dict(out["proxy"])
            ref_text = str(ref_pred_map.get(sid, "")).strip()
            if ref_text:
                match_total += 1
                same = int(str(pred_row.get("output", "")).strip() == ref_text)
                match_same += same
                pred_row["reference_pred_text"] = ref_text
                pred_row["prediction_matches_reference"] = same
                proxy_row["reference_pred_text"] = ref_text
                proxy_row["prediction_matches_reference"] = same
            pred_rows.append(pred_row)
            proxy_rows.append(proxy_row)
        except Exception as exc:
            n_errors += 1
            pred_rows.append(
                {
                    "question_id": sid,
                    "question": str(sample.get("question", sample.get("text", ""))).strip(),
                    "image": str(sample.get("image", "")).strip(),
                    "output": "",
                    "route_mode": "method",
                    "error": str(exc),
                }
            )
            proxy_rows.append(
                {
                    "id": sid,
                    "image": str(sample.get("image", "")).strip(),
                    "question": str(sample.get("question", sample.get("text", ""))).strip(),
                    "output": "",
                    "error": str(exc),
                }
            )
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[decode-proxy] {idx + 1}/{len(samples)}")

    pred_jsonl = os.path.join(args.out_dir, "pred_vga_decode_proxy.jsonl")
    proxy_csv = os.path.join(args.out_dir, "decode_time_proxy_features.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_jsonl(pred_jsonl, pred_rows)
    write_csv(proxy_csv, proxy_rows)
    write_json(
        summary_json,
        {
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "headset_json": os.path.abspath(args.headset_json),
                "reference_pred_jsonl": (os.path.abspath(args.reference_pred_jsonl) if str(args.reference_pred_jsonl).strip() else ""),
                "model_path": args.model_path,
                "conv_mode": args.conv_mode,
                "device": args.device,
                "max_gen_len": int(args.max_gen_len),
                "start_layer": int(args.start_layer),
                "end_layer": int(args.end_layer),
                "proxy_trace_late_start": int(args.proxy_trace_late_start),
                "proxy_trace_late_end": int(args.proxy_trace_late_end),
                "proxy_trace_last_k": int(args.proxy_trace_last_k),
            },
            "counts": {
                "n_samples": int(len(samples)),
                "n_pred_rows": int(len(pred_rows)),
                "n_proxy_rows": int(len(proxy_rows)),
                "n_errors": int(n_errors),
                "reference_match_total": int(match_total),
                "reference_match_same": int(match_same),
                "reference_match_rate": (None if match_total == 0 else float(match_same / float(match_total))),
            },
            "outputs": {
                "pred_jsonl": os.path.abspath(pred_jsonl),
                "proxy_csv": os.path.abspath(proxy_csv),
            },
        },
    )
    print("[saved]", pred_jsonl)
    print("[saved]", proxy_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
