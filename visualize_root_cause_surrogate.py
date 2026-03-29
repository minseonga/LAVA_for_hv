#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RE_VIS_HEAD = re.compile(r"^head_attn_vis_(?:ratio|sum)__layer_(\d+)__head_(\d+)$")
RE_HARM_HEAD = re.compile(r"^head_contrib_l(\d+)_h(\d+)$")


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def short_axis_name(x: str) -> str:
    m = {
        "vision-aware head failure": "Vision-Aware\nHead Failure",
        "dynamic harmful heads": "Dynamic\nHarmful Heads",
        "visual sink / artifact": "Visual Sink /\nArtifact",
        "text-prior hijack": "Text-Prior\nHijack",
    }
    return m.get(x, x)


def _make_matrix(
    triples: List[Tuple[int, int, float]],
) -> Tuple[np.ndarray, List[int], List[int]]:
    if len(triples) == 0:
        return np.zeros((1, 1), dtype=np.float32), [0], [0]
    layers = sorted(set(t[0] for t in triples))
    heads = sorted(set(t[1] for t in triples))
    l2i = {l: i for i, l in enumerate(layers)}
    h2i = {h: i for i, h in enumerate(heads)}
    mat = np.full((len(layers), len(heads)), np.nan, dtype=np.float32)
    for l, h, v in triples:
        mat[l2i[l], h2i[h]] = float(v)
    return mat, layers, heads


def _plot_heatmap(
    mat: np.ndarray,
    y_labels: List[int],
    x_labels: List[int],
    title: str,
    out_path: str,
    cmap: str = "viridis",
) -> None:
    plt.figure(figsize=(11, 5.5))
    shown = np.where(np.isfinite(mat), mat, np.nan)
    im = plt.imshow(shown, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar(im, label="AUC(best_dir)")
    plt.yticks(ticks=np.arange(len(y_labels)), labels=[str(x) for x in y_labels])
    # show sparse x ticks for readability
    xt = np.arange(len(x_labels))
    if len(xt) > 24:
        step = max(1, len(xt) // 24)
        xt = xt[::step]
    plt.xticks(ticks=xt, labels=[str(x_labels[i]) for i in xt], rotation=45, ha="right")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_summary_json",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/root_cause_surrogate_matrix_v1/summary.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/root_cause_surrogate_matrix_v1/viz",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    root_summary = json.load(open(args.root_summary_json, "r", encoding="utf-8"))
    root_dir = os.path.dirname(os.path.abspath(args.root_summary_json))

    mat_csv = os.path.join(root_dir, "surrogate_matrix.csv")
    rank_csv = os.path.join(root_dir, "surrogate_ranking.csv")
    rows = load_csv(mat_csv)
    rank_rows = load_csv(rank_csv)

    # 1) Axis-level AUC/KS bar
    axis_rows = sorted(rows, key=lambda r: safe_float(r.get("auc_best_dir"), -1.0), reverse=True)
    labels = [short_axis_name(str(r.get("axis", ""))) for r in axis_rows]
    auc = [safe_float(r.get("auc_best_dir"), 0.0) for r in axis_rows]
    ks = [safe_float(r.get("ks"), 0.0) for r in axis_rows]

    x = np.arange(len(labels))
    w = 0.36
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - w / 2, auc, width=w, label="AUC(best_dir)", color="#2a9d8f")
    plt.bar(x + w / 2, ks, width=w, label="KS", color="#e76f51")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Score")
    plt.title("Root-Cause Surrogate Separation (FP_hall vs TP_yes)")
    plt.legend()
    plt.tight_layout()
    out_axis = os.path.join(args.out_dir, "01_axis_auc_ks.png")
    plt.savefig(out_axis, dpi=180)
    plt.close()

    # 2) Priority ranking bar
    top = sorted(rank_rows, key=lambda r: safe_float(r.get("final_priority"), -1.0), reverse=True)
    labels2 = [short_axis_name(str(r.get("axis", ""))) for r in top]
    vals2 = [safe_float(r.get("final_priority"), 0.0) for r in top]
    plt.figure(figsize=(9.5, 4.6))
    y = np.arange(len(labels2))
    plt.barh(y, vals2, color="#264653")
    plt.yticks(y, labels2)
    plt.gca().invert_yaxis()
    plt.xlabel("Final Priority Score")
    plt.title("Intervention Priority Ranking")
    plt.tight_layout()
    out_rank = os.path.join(args.out_dir, "02_priority_ranking.png")
    plt.savefig(out_rank, dpi=180)
    plt.close()

    # 3) Vision-aware head failure heatmap (from source head eval)
    head_eval_csv = root_summary["inputs"]["pope_head_eval_csv"]
    head_rows = load_csv(head_eval_csv)
    vis_triples: List[Tuple[int, int, float]] = []
    for r in head_rows:
        if str(r.get("comparison", "")) != "fp_hall_vs_tp_yes":
            continue
        if str(r.get("direction", "")) != "lower_in_hallucination":
            continue
        m = RE_VIS_HEAD.match(str(r.get("metric", "")))
        if m is None:
            continue
        li, hi = int(m.group(1)), int(m.group(2))
        vis_triples.append((li, hi, safe_float(r.get("auc_best_dir"), 0.0)))
    vis_mat, vis_layers, vis_heads = _make_matrix(vis_triples)
    out_vis_heat = os.path.join(args.out_dir, "03_vision_head_auc_heatmap.png")
    _plot_heatmap(
        vis_mat,
        y_labels=vis_layers,
        x_labels=vis_heads,
        title="Vision-Aware Head Failure: AUC Heatmap (higher=stronger separation)",
        out_path=out_vis_heat,
        cmap="magma",
    )

    # 4) Dynamic harmful head heatmap (from AIS decomposition head eval)
    ais_head_eval_csv = root_summary["inputs"]["pope_ais_head_eval_csv"]
    ais_rows = load_csv(ais_head_eval_csv)
    harm_triples: List[Tuple[int, int, float]] = []
    for r in ais_rows:
        if str(r.get("comparison", "")) != "fp_hall_vs_tp_yes":
            continue
        if str(r.get("direction", "")) != "higher_in_hallucination":
            continue
        m = RE_HARM_HEAD.match(str(r.get("metric", "")))
        if m is None:
            continue
        li, hi = int(m.group(1)), int(m.group(2))
        harm_triples.append((li, hi, safe_float(r.get("auc_best_dir"), 0.0)))
    harm_mat, harm_layers, harm_heads = _make_matrix(harm_triples)
    out_harm_heat = os.path.join(args.out_dir, "04_harmful_head_auc_heatmap.png")
    _plot_heatmap(
        harm_mat,
        y_labels=harm_layers,
        x_labels=harm_heads,
        title="Dynamic Harmful Heads: AUC Heatmap (higher=more harmful separation)",
        out_path=out_harm_heat,
        cmap="inferno",
    )

    # 5) Visual sink / artifact layer curves
    layer_eval_csv = root_summary["inputs"]["pope_layer_eval_csv"]
    layer_rows = load_csv(layer_eval_csv)
    curves: Dict[str, Dict[int, float]] = {
        "yes_sim_local_max": {},
        "yes_sim_local_topk": {},
        "yes_sim_objpatch_max": {},
    }
    for r in layer_rows:
        if str(r.get("comparison", "")) != "fp_hall_vs_tp_yes":
            continue
        if str(r.get("direction", "")) != "higher_in_hallucination":
            continue
        base = str(r.get("metric_base", ""))
        li = int(safe_float(r.get("block_layer_idx"), -1))
        if base not in curves or li < 0:
            continue
        v = safe_float(r.get("auc_best_dir"), 0.0)
        prev = curves[base].get(li, None)
        if prev is None or v > prev:
            curves[base][li] = v

    plt.figure(figsize=(10.2, 4.8))
    colors = {
        "yes_sim_local_max": "#1d3557",
        "yes_sim_local_topk": "#457b9d",
        "yes_sim_objpatch_max": "#e63946",
    }
    for k, mp in curves.items():
        if len(mp) == 0:
            continue
        xs = sorted(mp.keys())
        ys = [mp[x] for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.8, label=k, color=colors.get(k, None))
    plt.ylim(0.5, 0.9)
    plt.xlabel("Layer")
    plt.ylabel("AUC(best_dir)")
    plt.title("Visual Sink/Artifact Surrogates by Layer (FP_hall vs TP_yes)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_sink_curve = os.path.join(args.out_dir, "05_visual_sink_layer_curves.png")
    plt.savefig(out_sink_curve, dpi=180)
    plt.close()

    # Save compact numeric table for quick reading.
    stats_out = os.path.join(args.out_dir, "viz_stats.csv")
    with open(stats_out, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["axis", "metric", "auc_best_dir", "ks", "direction", "n"])
        for r in axis_rows:
            wr.writerow(
                [
                    r.get("axis", ""),
                    r.get("metric", ""),
                    safe_float(r.get("auc_best_dir"), 0.0),
                    safe_float(r.get("ks"), 0.0),
                    r.get("direction", ""),
                    int(safe_float(r.get("n"), 0)),
                ]
            )

    summary_out = os.path.join(args.out_dir, "summary.json")
    summary = {
        "inputs": {
            "root_summary_json": os.path.abspath(args.root_summary_json),
            "surrogate_matrix_csv": mat_csv,
            "surrogate_ranking_csv": rank_csv,
            "head_eval_csv": head_eval_csv,
            "ais_head_eval_csv": ais_head_eval_csv,
            "layer_eval_csv": layer_eval_csv,
        },
        "outputs": {
            "axis_auc_ks_png": out_axis,
            "priority_png": out_rank,
            "vision_head_heatmap_png": out_vis_heat,
            "harmful_head_heatmap_png": out_harm_heat,
            "visual_sink_curve_png": out_sink_curve,
            "viz_stats_csv": stats_out,
            "summary_json": summary_out,
        },
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_axis)
    print("[saved]", out_rank)
    print("[saved]", out_vis_heat)
    print("[saved]", out_harm_heat)
    print("[saved]", out_sink_curve)
    print("[saved]", stats_out)
    print("[saved]", summary_out)


if __name__ == "__main__":
    main()
