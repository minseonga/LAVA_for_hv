#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


CASE_ORDER = ["both_correct", "both_wrong", "vga_improvement", "vga_regression"]


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


def load_headset(path: str) -> Dict[Tuple[int, int], str]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    out: Dict[Tuple[int, int], str] = {}
    for layer, heads in (obj.get("faithful_heads_by_layer") or {}).items():
        for head in heads:
            out[(int(layer), int(head))] = "faithful"
    for layer, heads in (obj.get("harmful_heads_by_layer") or {}).items():
        for head in heads:
            out[(int(layer), int(head))] = "harmful"
    return out


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return sum(vals) / len(vals) if vals else 0.0


def safe_get_case(means: Dict[str, float], case: str) -> float:
    return float(means.get(case, 0.0))


def build_shift_row(
    label_cols: Dict[str, Any],
    offline_means: Dict[str, float],
    runtime_means: Dict[str, float],
) -> Dict[str, Any]:
    row = dict(label_cols)
    for case in CASE_ORDER:
        off = safe_get_case(offline_means, case)
        run = safe_get_case(runtime_means, case)
        row[f"offline_{case}"] = off
        row[f"runtime_{case}"] = run
        row[f"shift_{case}"] = run - off

    row["offline_sep_d2_vs_bc"] = row["offline_vga_regression"] - row["offline_both_correct"]
    row["runtime_sep_d2_vs_bc"] = row["runtime_vga_regression"] - row["runtime_both_correct"]
    row["shift_sep_d2_vs_bc"] = row["runtime_sep_d2_vs_bc"] - row["offline_sep_d2_vs_bc"]

    row["offline_sep_d2_vs_d1"] = row["offline_vga_regression"] - row["offline_vga_improvement"]
    row["runtime_sep_d2_vs_d1"] = row["runtime_vga_regression"] - row["runtime_vga_improvement"]
    row["shift_sep_d2_vs_d1"] = row["runtime_sep_d2_vs_d1"] - row["offline_sep_d2_vs_d1"]
    return row


def aggregate_head_case_means(rows: List[Dict[str, Any]], head_type: str) -> List[Dict[str, Any]]:
    bucket: Dict[Tuple[int, int, str], Dict[str, List[float]]] = defaultdict(lambda: {"offline": [], "runtime": []})
    for row in rows:
        if row["head_type"] != head_type:
            continue
        key = (int(row["block_layer_idx"]), int(row["head_idx"]), str(row["case_type"]))
        bucket[key]["offline"].append(float(row["offline_val"]))
        bucket[key]["runtime"].append(float(row["runtime_val"]))

    grouped: Dict[Tuple[int, int], Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (layer, head, case_type), vals in bucket.items():
        grouped[(layer, head)][case_type] = {
            "offline": mean(vals["offline"]),
            "runtime": mean(vals["runtime"]),
        }

    out: List[Dict[str, Any]] = []
    for (layer, head), case_map in sorted(grouped.items()):
        offline_means = {case: float(v["offline"]) for case, v in case_map.items()}
        runtime_means = {case: float(v["runtime"]) for case, v in case_map.items()}
        row = build_shift_row(
            {"head_type": head_type, "block_layer_idx": layer, "head_idx": head},
            offline_means=offline_means,
            runtime_means=runtime_means,
        )
        out.append(row)
    return out


def aggregate_layer_case_means(rows: List[Dict[str, Any]], head_type: str) -> List[Dict[str, Any]]:
    per_sample_layer_case: Dict[Tuple[str, int, str], Dict[str, List[float]]] = defaultdict(lambda: {"offline": [], "runtime": []})
    for row in rows:
        if row["head_type"] != head_type:
            continue
        key = (str(row["id"]), int(row["block_layer_idx"]), str(row["case_type"]))
        per_sample_layer_case[key]["offline"].append(float(row["offline_val"]))
        per_sample_layer_case[key]["runtime"].append(float(row["runtime_val"]))

    bucket: Dict[Tuple[int, str], Dict[str, List[float]]] = defaultdict(lambda: {"offline": [], "runtime": []})
    for (sid, layer, case_type), vals in per_sample_layer_case.items():
        bucket[(layer, case_type)]["offline"].append(mean(vals["offline"]))
        bucket[(layer, case_type)]["runtime"].append(mean(vals["runtime"]))

    grouped: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (layer, case_type), vals in bucket.items():
        grouped[layer][case_type] = {
            "offline": mean(vals["offline"]),
            "runtime": mean(vals["runtime"]),
        }

    out: List[Dict[str, Any]] = []
    for layer, case_map in sorted(grouped.items()):
        offline_means = {case: float(v["offline"]) for case, v in case_map.items()}
        runtime_means = {case: float(v["runtime"]) for case, v in case_map.items()}
        row = build_shift_row(
            {"head_type": head_type, "block_layer_idx": layer},
            offline_means=offline_means,
            runtime_means=runtime_means,
        )
        out.append(row)
    return out


def rank_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = [dict(r) for r in rows]
    for row in ranked:
        row["bc_positive_shift"] = float(row.get("shift_both_correct", 0.0))
        row["d2_negative_shift"] = -float(row.get("shift_vga_regression", 0.0))
        row["harm_score_d2_vs_bc"] = -float(row.get("shift_sep_d2_vs_bc", 0.0))
        row["harm_score_d2_vs_d1"] = -float(row.get("shift_sep_d2_vs_d1", 0.0))
    ranked.sort(
        key=lambda r: (
            float(r.get("harm_score_d2_vs_bc", 0.0)),
            float(r.get("bc_positive_shift", 0.0)),
            float(r.get("d2_negative_shift", 0.0)),
        ),
        reverse=True,
    )
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze static-headset runtime/offline contribution shifts by head and layer.")
    ap.add_argument("--aligned_csv", type=str, required=True)
    ap.add_argument("--sample_csv", type=str, required=True)
    ap.add_argument("--headset_json", type=str, required=True)
    ap.add_argument("--metric", type=str, default="head_attn_vis_ratio", choices=["head_attn_vis_ratio", "head_attn_vis_sum", "head_attn_vis_peak", "head_attn_vis_entropy"])
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    head_type_map = load_headset(args.headset_json)
    sample_rows = load_csv_rows(args.sample_csv)
    case_map = {str(r.get("id", "")): str(r.get("case_type", "")) for r in sample_rows}
    aligned_rows = load_csv_rows(args.aligned_csv)

    metric = str(args.metric)
    offline_key = f"offline_{metric}"
    runtime_key = f"runtime_{metric}"

    enriched: List[Dict[str, Any]] = []
    for row in aligned_rows:
        layer = int(row["block_layer_idx"])
        head = int(row["head_idx"])
        head_type = head_type_map.get((layer, head), "other")
        if head_type == "other":
            continue
        sid = str(row["id"])
        enriched.append(
            {
                "id": sid,
                "case_type": case_map.get(sid, ""),
                "block_layer_idx": layer,
                "head_idx": head,
                "head_type": head_type,
                "offline_val": float(row[offline_key]),
                "runtime_val": float(row[runtime_key]),
            }
        )

    faithful_head = aggregate_head_case_means(enriched, "faithful")
    harmful_head = aggregate_head_case_means(enriched, "harmful")
    faithful_layer = aggregate_layer_case_means(enriched, "faithful")
    harmful_layer = aggregate_layer_case_means(enriched, "harmful")

    faithful_head_ranked = rank_rows(faithful_head)
    harmful_head_ranked = rank_rows(harmful_head)
    faithful_layer_ranked = rank_rows(faithful_layer)
    harmful_layer_ranked = rank_rows(harmful_layer)

    outputs = {
        "faithful_head_case_means_csv": os.path.join(args.out_dir, "faithful_head_case_means.csv"),
        "faithful_head_ranked_csv": os.path.join(args.out_dir, "faithful_head_ranked.csv"),
        "faithful_layer_case_means_csv": os.path.join(args.out_dir, "faithful_layer_case_means.csv"),
        "faithful_layer_ranked_csv": os.path.join(args.out_dir, "faithful_layer_ranked.csv"),
        "harmful_head_case_means_csv": os.path.join(args.out_dir, "harmful_head_case_means.csv"),
        "harmful_head_ranked_csv": os.path.join(args.out_dir, "harmful_head_ranked.csv"),
        "harmful_layer_case_means_csv": os.path.join(args.out_dir, "harmful_layer_case_means.csv"),
        "harmful_layer_ranked_csv": os.path.join(args.out_dir, "harmful_layer_ranked.csv"),
    }

    write_csv(outputs["faithful_head_case_means_csv"], faithful_head)
    write_csv(outputs["faithful_head_ranked_csv"], faithful_head_ranked)
    write_csv(outputs["faithful_layer_case_means_csv"], faithful_layer)
    write_csv(outputs["faithful_layer_ranked_csv"], faithful_layer_ranked)
    write_csv(outputs["harmful_head_case_means_csv"], harmful_head)
    write_csv(outputs["harmful_head_ranked_csv"], harmful_head_ranked)
    write_csv(outputs["harmful_layer_case_means_csv"], harmful_layer)
    write_csv(outputs["harmful_layer_ranked_csv"], harmful_layer_ranked)

    summary = {
        "inputs": {
            "aligned_csv": os.path.abspath(args.aligned_csv),
            "sample_csv": os.path.abspath(args.sample_csv),
            "headset_json": os.path.abspath(args.headset_json),
            "metric": metric,
        },
        "counts": {
            "n_aligned_rows": len(aligned_rows),
            "n_enriched_rows": len(enriched),
            "n_faithful_heads": len(faithful_head),
            "n_harmful_heads": len(harmful_head),
            "n_faithful_layers": len(faithful_layer),
            "n_harmful_layers": len(harmful_layer),
        },
        "top_findings": {
            "faithful_head_top3": faithful_head_ranked[:3],
            "faithful_layer_top3": faithful_layer_ranked[:3],
            "harmful_head_top3": harmful_head_ranked[:3],
            "harmful_layer_top3": harmful_layer_ranked[:3],
        },
        "outputs": outputs,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[saved]", summary_path)
    print(json.dumps(summary["top_findings"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
