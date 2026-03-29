#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def family_columns() -> Dict[str, List[str]]:
    return {
        "A": [
            "obj_token_prob_max",
            "obj_token_prob_mean",
            "obj_token_prob_lse",
            "obj_token_prob_topkmean",
            "entropy_score",
            "G_entropy",
            "G_top1_mass",
            "G_top5_mass",
            "G_effective_support_size",
        ],
        "B": [
            "early_attn_mean",
            "late_attn_mean",
            "late_uplift",
            "late_topk_persistence",
            "peak_layer_idx",
            "peak_before_final",
            "late_rank_std",
            "argmax_patch_flip_count",
            "persistence_after_peak",
        ],
        "C": [
            "faithful_head_attn_mean",
            "faithful_head_attn_topkmean",
            "faithful_head_coverage",
            "faithful_minus_global_attn",
            "faithful_n_points",
        ],
        "D": [
            "harmful_head_attn_mean",
            "harmful_head_attn_topkmean",
            "harmful_head_coverage",
            "harmful_minus_global_attn",
            "harmful_minus_faithful",
            "harmful_n_points",
        ],
        "E": [
            "supportive_outside_G",
            "harmful_inside_G",
            "guidance_mismatch_score",
            "context_need_score",
            "G_overfocus",
            "faithful_on_G_mass",
            "faithful_on_nonG_mass",
            "harmful_on_G_mass",
            "harmful_on_nonG_mass",
            "faithful_G_alignment",
            "harmful_G_alignment",
        ],
    }


def zscore(vs: List[float], eps: float = 1e-6) -> List[float]:
    if len(vs) == 0:
        return []
    m = sum(vs) / float(len(vs))
    if len(vs) <= 1:
        return [0.0 for _ in vs]
    var = sum((x - m) * (x - m) for x in vs) / float(len(vs) - 1)
    s = math.sqrt(max(var, eps))
    return [(x - m) / s for x in vs]


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    n = len(labels)
    if n == 0:
        return None
    pairs = [(int(labels[i]), float(scores[i])) for i in range(n)]
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    idx = sorted(range(n), key=lambda i: pairs[i][1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and pairs[idx[j]][1] == pairs[idx[i]][1]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[idx[k]] = avg_rank
        i = j
    sum_pos = sum(ranks[i] for i in range(n) if pairs[i][0] == 1)
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    n = len(labels)
    if n == 0:
        return None
    pairs = [(int(labels[i]), float(scores[i])) for i in range(n)]
    n_pos = sum(1 for y, _ in pairs if y == 1)
    if n_pos == 0:
        return None
    pairs.sort(key=lambda x: x[1], reverse=True)
    tp = 0
    sum_prec = 0.0
    for i, (y, _) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            sum_prec += tp / float(i)
    return float(sum_prec / float(n_pos))


def composite_score(rows: List[Dict[str, Any]], labels: List[int], cols: List[str]) -> Tuple[List[int], List[float], List[str]]:
    valid_cols: List[str] = []
    per_col_values: Dict[str, List[Optional[float]]] = {}
    for c in cols:
        vals = [safe_float(r.get(c), None) for r in rows]
        if sum(1 for v in vals if v is not None) >= 10:
            valid_cols.append(c)
            per_col_values[c] = vals
    if len(valid_cols) == 0:
        return [], [], []

    zvals_by_col: Dict[str, List[Optional[float]]] = {}
    sign_by_col: Dict[str, float] = {}
    for c in valid_cols:
        vals = per_col_values[c]
        idx = [i for i, v in enumerate(vals) if v is not None]
        arr = [float(vals[i]) for i in idx]
        zarr = zscore(arr)
        zfull: List[Optional[float]] = [None for _ in vals]
        for i, z in zip(idx, zarr):
            zfull[i] = float(z)
        zvals_by_col[c] = zfull

        pos = [float(vals[i]) for i in idx if labels[i] == 1]
        neg = [float(vals[i]) for i in idx if labels[i] == 0]
        if len(pos) == 0 or len(neg) == 0:
            sign_by_col[c] = 1.0
        else:
            sign_by_col[c] = 1.0 if (sum(pos) / len(pos)) >= (sum(neg) / len(neg)) else -1.0

    ys: List[int] = []
    xs: List[float] = []
    for i, y in enumerate(labels):
        parts = []
        for c in valid_cols:
            zc = zvals_by_col[c][i]
            if zc is not None:
                parts.append(sign_by_col[c] * float(zc))
        if len(parts) == 0:
            continue
        ys.append(int(y))
        xs.append(float(sum(parts) / float(len(parts))))
    return ys, xs, valid_cols


def roc_curve_points(labels: Sequence[int], scores: Sequence[float]) -> Tuple[List[float], List[float]]:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    p = sum(1 for _, y in pairs if y == 1)
    n = sum(1 for _, y in pairs if y == 0)
    if p == 0 or n == 0:
        return [0.0, 1.0], [0.0, 1.0]

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    for _, y in pairs:
        if int(y) == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / float(p))
        fpr.append(fp / float(n))
    return fpr, tpr


def pr_curve_points(labels: Sequence[int], scores: Sequence[float]) -> Tuple[List[float], List[float]]:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    p = sum(1 for _, y in pairs if y == 1)
    if p == 0:
        return [0.0], [1.0]

    precision: List[float] = []
    recall: List[float] = []
    tp = 0
    fp = 0
    for _, y in pairs:
        if int(y) == 1:
            tp += 1
        else:
            fp += 1
        precision.append(tp / float(max(1, tp + fp)))
        recall.append(tp / float(p))
    return recall, precision


def build_label(row: Dict[str, Any], task: str) -> Optional[int]:
    if task == "fp_vs_tp_yes":
        fp = safe_int(row.get("target_is_fp_hallucination"), None)
        tp = safe_int(row.get("target_is_tp_yes"), None)
        if fp == 1:
            return 1
        if tp == 1:
            return 0
        return None
    if task == "fv_vs_vf":
        v = safe_int(row.get("target_fv_vs_vf"), None)
        return v if v in {0, 1} else None
    if task == "incorrect_vs_correct":
        c = safe_int(row.get("target_is_correct"), None)
        if c is None:
            return None
        return 0 if c == 1 else 1
    raise ValueError(f"Unknown task: {task}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize family-level ROC/PR and top-feature bars for structure screening.")
    ap.add_argument("--structure_dir", type=str, required=True, help="Output directory from eval_feature_families_structure.py")
    ap.add_argument("--task", type=str, default="fv_vs_vf", choices=["fv_vs_vf", "fp_vs_tp_yes", "incorrect_vs_correct"])
    ap.add_argument("--split_filter", type=str, default="all", choices=["all", "calib", "eval"])
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    structure_dir = os.path.abspath(args.structure_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(structure_dir, args.task, "viz")
    os.makedirs(out_dir, exist_ok=True)

    features_csv = os.path.join(structure_dir, "features_with_pair_targets.csv")
    if not os.path.isfile(features_csv):
        raise RuntimeError(f"Missing required file: {features_csv}")

    rows = read_csv(features_csv)
    if args.split_filter != "all":
        rows = [r for r in rows if str(r.get("split", "")).strip().lower() == args.split_filter]

    use_rows: List[Dict[str, Any]] = []
    labels: List[int] = []
    for r in rows:
        y = build_label(r, args.task)
        if y is None:
            continue
        use_rows.append(r)
        labels.append(int(y))

    if len(use_rows) == 0:
        raise RuntimeError("No labeled rows after task/split filtering.")

    fam_cols = family_columns()
    curves: Dict[str, Dict[str, Any]] = {}
    for fam in ["A", "B", "C", "D", "E"]:
        ys, xs, used = composite_score(use_rows, labels, fam_cols[fam])
        if len(ys) < 20:
            continue
        auc_raw = auc_from_scores(ys, xs)
        if auc_raw is None:
            continue
        orient = "higher_in_pos"
        xs_use = xs
        if auc_raw < 0.5:
            orient = "lower_in_pos"
            xs_use = [-v for v in xs]
        auc_best = auc_from_scores(ys, xs_use) or 0.5
        ap_best = average_precision(ys, xs_use) or 0.0
        fpr, tpr = roc_curve_points(ys, xs_use)
        recall, prec = pr_curve_points(ys, xs_use)
        curves[fam] = {
            "auc_best_dir": float(auc_best),
            "ap_best_dir": float(ap_best),
            "orientation": orient,
            "n": len(ys),
            "features_used": used,
            "roc_fpr": fpr,
            "roc_tpr": tpr,
            "pr_recall": recall,
            "pr_precision": prec,
        }

    # 1) Family ROC
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.4), dpi=160)
    palette = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728", "E": "#9467bd"}
    for fam in ["A", "B", "C", "D", "E"]:
        c = curves.get(fam)
        if c is None:
            continue
        ax.plot(c["roc_fpr"], c["roc_tpr"], lw=1.8, color=palette[fam], label=f"{fam} (AUC={c['auc_best_dir']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Family ROC ({args.task}, split={args.split_filter})")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    roc_png = os.path.join(out_dir, "01_family_roc_composite.png")
    fig.savefig(roc_png)
    plt.close(fig)

    # 2) Family PR
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.4), dpi=160)
    for fam in ["A", "B", "C", "D", "E"]:
        c = curves.get(fam)
        if c is None:
            continue
        ax.plot(c["pr_recall"], c["pr_precision"], lw=1.8, color=palette[fam], label=f"{fam} (AP={c['ap_best_dir']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Family PR ({args.task}, split={args.split_filter})")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    pr_png = os.path.join(out_dir, "02_family_pr_composite.png")
    fig.savefig(pr_png)
    plt.close(fig)

    # 3) Top feature bar (single vs family composite)
    single_csv = os.path.join(structure_dir, args.task, "family_single_feature_metrics.csv")
    comp_csv = os.path.join(structure_dir, args.task, "family_composite_metrics.csv")
    single_rows = read_csv(single_csv) if os.path.isfile(single_csv) else []
    comp_rows = read_csv(comp_csv) if os.path.isfile(comp_csv) else []

    best_single: Dict[str, Dict[str, Any]] = {}
    for r in single_rows:
        fam = str(r.get("family", "")).strip()
        if fam not in {"A", "B", "C", "D", "E"}:
            continue
        auc = safe_float(r.get("auc_best_dir"), 0.0) or 0.0
        if fam not in best_single or auc > (safe_float(best_single[fam].get("auc_best_dir"), 0.0) or 0.0):
            best_single[fam] = r

    best_comp: Dict[str, Dict[str, Any]] = {}
    for r in comp_rows:
        fam = str(r.get("set_name", "")).strip()
        if fam not in {"A", "B", "C", "D", "E"}:
            continue
        auc = safe_float(r.get("auc_best_dir"), 0.0) or 0.0
        if fam not in best_comp or auc > (safe_float(best_comp[fam].get("auc_best_dir"), 0.0) or 0.0):
            best_comp[fam] = r

    families = ["A", "B", "C", "D", "E"]
    x = list(range(len(families)))
    single_auc = [safe_float(best_single.get(f, {}).get("auc_best_dir"), 0.0) or 0.0 for f in families]
    comp_auc = [safe_float(best_comp.get(f, {}).get("auc_best_dir"), 0.0) or 0.0 for f in families]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), dpi=160)
    w = 0.38
    ax.bar([i - w / 2 for i in x], single_auc, width=w, label="Best single feature AUC", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], comp_auc, width=w, label="Family composite AUC", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("AUC (best direction)")
    ax.set_title(f"Top Feature Bar ({args.task}, split={args.split_filter})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    bar_png = os.path.join(out_dir, "03_top_feature_bar_auc.png")
    fig.savefig(bar_png)
    plt.close(fig)

    # CSV summary for the bar figure
    bar_rows: List[Dict[str, Any]] = []
    for fam in families:
        bar_rows.append(
            {
                "family": fam,
                "single_feature": best_single.get(fam, {}).get("feature", ""),
                "single_auc_best_dir": safe_float(best_single.get(fam, {}).get("auc_best_dir"), None),
                "single_ap_best_dir": safe_float(best_single.get(fam, {}).get("ap_best_dir"), None),
                "single_ks": safe_float(best_single.get(fam, {}).get("ks"), None),
                "composite_auc_best_dir": safe_float(best_comp.get(fam, {}).get("auc_best_dir"), None),
                "composite_ap_best_dir": safe_float(best_comp.get(fam, {}).get("ap_best_dir"), None),
                "composite_ks": safe_float(best_comp.get(fam, {}).get("ks"), None),
            }
        )
    bar_csv = os.path.join(out_dir, "03_top_feature_bar_auc.csv")
    with open(bar_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(bar_rows[0].keys()))
        wr.writeheader()
        wr.writerows(bar_rows)

    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "structure_dir": structure_dir,
                    "task": args.task,
                    "split_filter": args.split_filter,
                    "n_rows_labeled": len(use_rows),
                },
                "family_curves": {
                    fam: {
                        "auc_best_dir": c["auc_best_dir"],
                        "ap_best_dir": c["ap_best_dir"],
                        "orientation": c["orientation"],
                        "n": c["n"],
                        "n_features_used": len(c["features_used"]),
                    }
                    for fam, c in curves.items()
                },
                "outputs": {
                    "roc_png": roc_png,
                    "pr_png": pr_png,
                    "top_bar_png": bar_png,
                    "top_bar_csv": bar_csv,
                    "summary_json": summary_json,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[saved]", roc_png)
    print("[saved]", pr_png)
    print("[saved]", bar_png)
    print("[saved]", bar_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()

