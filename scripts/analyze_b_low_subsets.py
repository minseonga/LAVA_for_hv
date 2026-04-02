#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_bool(text: object) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def maybe_float(value: object) -> Optional[float]:
    s = str(value or "").strip()
    if s == "" or s.lower() in {"nan", "none", "null", "inf", "-inf"}:
        return None
    try:
        out = float(s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: object) -> Optional[int]:
    f = maybe_float(value)
    if f is None:
        return None
    return int(round(f))


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
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
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def infer_case_type(base_correct: Optional[int], int_correct: Optional[int]) -> str:
    if base_correct is None or int_correct is None:
        return "unknown"
    if int(base_correct) == 1 and int(int_correct) == 0:
        return "vga_regression"
    if int(base_correct) == 0 and int(int_correct) == 1:
        return "vga_improvement"
    if int(base_correct) == 1 and int(int_correct) == 1:
        return "both_correct"
    if int(base_correct) == 0 and int(int_correct) == 0:
        return "both_wrong"
    return "unknown"


def assign_avg_ranks(scores: Sequence[float]) -> List[float]:
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and float(scores[order[j]]) == float(scores[order[i]]):
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pos = sum(1 for y in labels if int(y) == 1)
    neg = sum(1 for y in labels if int(y) == 0)
    if pos == 0 or neg == 0:
        return None
    ranks = assign_avg_ranks(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if int(y) == 1)
    u = rank_sum_pos - float(pos * (pos + 1)) / 2.0
    return float(u / float(pos * neg))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pos = sum(1 for y in labels if int(y) == 1)
    if pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    tp = 0
    ap = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(labels[idx]) == 1:
            tp += 1
            ap += float(tp) / float(rank)
    return float(ap / float(pos))


def mean(values: Iterable[float]) -> Optional[float]:
    seq = [float(v) for v in values]
    if not seq:
        return None
    return float(sum(seq) / float(len(seq)))


def std(values: Iterable[float]) -> Optional[float]:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0 if seq else None
    mu = float(sum(seq) / float(len(seq)))
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(math.sqrt(max(0.0, var)))


def parse_percents(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text or "").split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    return out or [1.0, 2.0, 5.0, 10.0]


def maybe_numeric_feature_names(rows: Sequence[Dict[str, Any]]) -> List[str]:
    excluded = {
        "id",
        "question",
        "image",
        "intervention_text",
        "baseline_text",
        "intervention_label",
        "baseline_label",
        "gt_label",
        "score_error",
        "case_type",
        "group",
        "answer_gt",
        "pred_baseline",
        "pred_vga",
        "gt",
        "category",
        "object_phrase",
        "image_id",
        "orig_question_id",
        "A_source",
        "A_source_repo",
        "B_source_repo",
        "C_source_repo",
        "D_source_repo",
    }
    names: List[str] = []
    if not rows:
        return names
    for key in rows[0].keys():
        if key in excluded:
            continue
        if key.startswith("stage_b_"):
            continue
        if key in {"baseline_correct", "intervention_correct", "baseline_ok", "vga_ok"}:
            continue
        good = 0
        bad = 0
        for row in rows[:200]:
            if key not in row:
                continue
            value = maybe_float(row.get(key))
            if value is None:
                continue
            good += 1
        bad = max(0, min(200, len(rows)) - good)
        if good >= 5 and good > bad:
            names.append(key)
    return sorted(set(names))


def merge_rows(
    score_rows: Sequence[Dict[str, str]],
    taxonomy_rows: Optional[Sequence[Dict[str, str]]],
    feature_rows: Optional[Sequence[Dict[str, str]]],
) -> List[Dict[str, Any]]:
    tax_map = {str(row.get("id", "")).strip(): row for row in (taxonomy_rows or [])}
    feat_map = {str(row.get("id", "")).strip(): row for row in (feature_rows or [])}
    merged: List[Dict[str, Any]] = []
    for row in score_rows:
        sid = str(row.get("id", "")).strip()
        base_correct = maybe_int(row.get("baseline_correct"))
        int_correct = maybe_int(row.get("intervention_correct"))
        merged_row: Dict[str, Any] = dict(row)
        merged_row["id"] = sid
        merged_row["baseline_correct"] = base_correct
        merged_row["intervention_correct"] = int_correct
        merged_row["case_type"] = infer_case_type(base_correct=base_correct, int_correct=int_correct)
        if sid in tax_map:
            for key, value in tax_map[sid].items():
                if key not in merged_row or merged_row[key] in {"", None}:
                    merged_row[key] = value
            if str(tax_map[sid].get("case_type", "")).strip():
                merged_row["case_type"] = str(tax_map[sid]["case_type"]).strip()
        if sid in feat_map:
            for key, value in feat_map[sid].items():
                if key == "id":
                    continue
                if key not in merged_row or merged_row[key] in {"", None}:
                    merged_row[key] = value
        merged.append(merged_row)
    return merged


def subset_by_bottom_percent(rows: Sequence[Dict[str, Any]], percent: float) -> List[Dict[str, Any]]:
    ordered = sorted(
        [row for row in rows if maybe_float(row.get("stage_b_score")) is not None and str(row.get("case_type", "")) != "unknown"],
        key=lambda row: float(row["stage_b_score"]),
    )
    if not ordered:
        return []
    k = max(1, int(math.ceil(float(percent) / 100.0 * float(len(ordered)))))
    return ordered[:k]


def subset_composition_row(name: str, rows: Sequence[Dict[str, Any]], percent: float) -> Dict[str, Any]:
    n = len(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[str(row["case_type"])] += 1
    return {
        "subset": name,
        "bottom_percent": float(percent),
        "n": int(n),
        "n_regression": int(counts["vga_regression"]),
        "n_improvement": int(counts["vga_improvement"]),
        "n_both_correct": int(counts["both_correct"]),
        "n_both_wrong": int(counts["both_wrong"]),
        "regression_share": (None if n == 0 else float(counts["vga_regression"] / float(n))),
        "improvement_share": (None if n == 0 else float(counts["vga_improvement"] / float(n))),
        "both_correct_share": (None if n == 0 else float(counts["both_correct"] / float(n))),
        "both_wrong_share": (None if n == 0 else float(counts["both_wrong"] / float(n))),
        "stage_b_score_mean": mean(maybe_float(row["stage_b_score"]) for row in rows if maybe_float(row.get("stage_b_score")) is not None),
        "stage_b_score_std": std(maybe_float(row["stage_b_score"]) for row in rows if maybe_float(row.get("stage_b_score")) is not None),
    }


def pairwise_metrics_for_subset(name: str, rows: Sequence[Dict[str, Any]], percent: float) -> List[Dict[str, Any]]:
    pairs = [
        ("regression_vs_non_rescue", {"vga_regression"}, {"vga_improvement", "both_correct", "both_wrong"}),
        ("regression_vs_improvement", {"vga_regression"}, {"vga_improvement"}),
        ("regression_vs_both_wrong", {"vga_regression"}, {"both_wrong"}),
        ("regression_vs_both_correct", {"vga_regression"}, {"both_correct"}),
    ]
    out: List[Dict[str, Any]] = []
    for comp, pos_set, neg_set in pairs:
        sub = [row for row in rows if row["case_type"] in pos_set or row["case_type"] in neg_set]
        labels = [1 if row["case_type"] in pos_set else 0 for row in sub]
        scores = [-float(row["stage_b_score"]) for row in sub]
        out.append(
            {
                "subset": name,
                "bottom_percent": float(percent),
                "comparison": comp,
                "n_total": int(len(sub)),
                "n_pos": int(sum(labels)),
                "n_neg": int(len(labels) - sum(labels)),
                "stage_b_risk_auroc": binary_auc(scores, labels),
                "stage_b_risk_ap": average_precision(scores, labels),
            }
        )
    return out


def oriented_scores(values: Sequence[float], labels: Sequence[int]) -> Tuple[str, List[float], Optional[float], Optional[float]]:
    auc_high = binary_auc(values, labels)
    auc_low = binary_auc([-float(v) for v in values], labels)
    ap_high = average_precision(values, labels)
    ap_low = average_precision([-float(v) for v in values], labels)
    if auc_high is None and auc_low is None:
        return "high", list(values), None, None
    if (auc_low or -1.0) > (auc_high or -1.0):
        return "low", [-float(v) for v in values], auc_low, ap_low
    return "high", list(values), auc_high, ap_high


def precision_at_k(oriented: Sequence[float], labels: Sequence[int], case_types: Sequence[str], k: int) -> Dict[str, Any]:
    if k <= 0 or not oriented:
        return {
            "precision_at_k": None,
            "regression_recall_at_k": None,
            "improvement_share_at_k": None,
            "both_wrong_share_at_k": None,
            "both_correct_share_at_k": None,
        }
    order = sorted(range(len(oriented)), key=lambda i: float(oriented[i]), reverse=True)
    top = order[: min(k, len(order))]
    reg_total = max(1, sum(1 for y in labels if int(y) == 1))
    reg = sum(1 for i in top if int(labels[i]) == 1)
    imp = sum(1 for i in top if case_types[i] == "vga_improvement")
    bw = sum(1 for i in top if case_types[i] == "both_wrong")
    bc = sum(1 for i in top if case_types[i] == "both_correct")
    denom = max(1, len(top))
    return {
        "precision_at_k": float(reg / float(denom)),
        "regression_recall_at_k": float(reg / float(reg_total)),
        "improvement_share_at_k": float(imp / float(denom)),
        "both_wrong_share_at_k": float(bw / float(denom)),
        "both_correct_share_at_k": float(bc / float(denom)),
    }


def feature_screen_rows(name: str, rows: Sequence[Dict[str, Any]], percent: float, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    reg_vs_keep = [row for row in rows if row["case_type"] in {"vga_regression", "vga_improvement", "both_correct", "both_wrong"}]
    reg_vs_imp = [row for row in rows if row["case_type"] in {"vga_regression", "vga_improvement"}]

    for feat in feature_names:
        keep_vals: List[float] = []
        keep_labels: List[int] = []
        keep_types: List[str] = []
        for row in reg_vs_keep:
            val = maybe_float(row.get(feat))
            if val is None:
                continue
            keep_vals.append(val)
            keep_labels.append(1 if row["case_type"] == "vga_regression" else 0)
            keep_types.append(str(row["case_type"]))
        imp_vals: List[float] = []
        imp_labels: List[int] = []
        for row in reg_vs_imp:
            val = maybe_float(row.get(feat))
            if val is None:
                continue
            imp_vals.append(val)
            imp_labels.append(1 if row["case_type"] == "vga_regression" else 0)

        direction, keep_oriented, keep_auc, keep_ap = oriented_scores(keep_vals, keep_labels)
        _, imp_oriented, imp_auc, imp_ap = oriented_scores(imp_vals, imp_labels)
        k = sum(1 for y in keep_labels if int(y) == 1)
        topk = precision_at_k(keep_oriented, keep_labels, keep_types, k=k)

        reg_mean = mean(maybe_float(row.get(feat)) for row in reg_vs_keep if row["case_type"] == "vga_regression" and maybe_float(row.get(feat)) is not None)
        imp_mean = mean(maybe_float(row.get(feat)) for row in reg_vs_keep if row["case_type"] == "vga_improvement" and maybe_float(row.get(feat)) is not None)
        keep_mean = mean(maybe_float(row.get(feat)) for row in reg_vs_keep if row["case_type"] != "vga_regression" and maybe_float(row.get(feat)) is not None)

        out.append(
            {
                "subset": name,
                "bottom_percent": float(percent),
                "feature": feat,
                "direction_for_regression": direction,
                "n_reg_vs_keep": int(len(keep_vals)),
                "n_reg_vs_improvement": int(len(imp_vals)),
                "auroc_reg_vs_non_rescue": keep_auc,
                "ap_reg_vs_non_rescue": keep_ap,
                "auroc_reg_vs_improvement": imp_auc,
                "ap_reg_vs_improvement": imp_ap,
                "regression_mean": reg_mean,
                "improvement_mean": imp_mean,
                "non_rescue_mean": keep_mean,
                **topk,
            }
        )
    return out


def top_feature_rows(feature_rows: Sequence[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[str(row["subset"])].append(row)
    out: List[Dict[str, Any]] = []
    for subset, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                -1.0 if row["auroc_reg_vs_non_rescue"] is None else float(row["auroc_reg_vs_non_rescue"]),
                -1.0 if row["auroc_reg_vs_improvement"] is None else float(row["auroc_reg_vs_improvement"]),
                -1.0 if row["precision_at_k"] is None else float(row["precision_at_k"]),
            ),
            reverse=False,
        )
        ranked = list(reversed(ranked))[:topk]
        out.extend(ranked)
    return out


def maybe_make_plots(out_dir: str, subset_rows: Sequence[Dict[str, Any]], top_rows: Sequence[Dict[str, Any]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    # composition plot
    ordered_subsets = sorted(subset_rows, key=lambda row: float(row["bottom_percent"]))
    labels = [f"bottom {float(row['bottom_percent']):g}%" for row in ordered_subsets]
    reg = [100.0 * float(row["regression_share"] or 0.0) for row in ordered_subsets]
    imp = [100.0 * float(row["improvement_share"] or 0.0) for row in ordered_subsets]
    bw = [100.0 * float(row["both_wrong_share"] or 0.0) for row in ordered_subsets]
    bc = [100.0 * float(row["both_correct_share"] or 0.0) for row in ordered_subsets]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.bar(labels, reg, color="#d73027", label="regression")
    ax.bar(labels, imp, bottom=reg, color="#1a9850", label="improvement")
    ax.bar(labels, bw, bottom=[a + b for a, b in zip(reg, imp)], color="#666666", label="both_wrong")
    ax.bar(labels, bc, bottom=[a + b + c for a, b, c in zip(reg, imp, bw)], color="#4575b4", label="both_correct")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Share within B-low subset (%)")
    ax.set_title("Composition of the B-low risky pool")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(out_dir, "b_low_subset_composition.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    # top-feature plots by subset
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in top_rows:
        grouped[str(row["subset"])].append(row)
    for subset, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                -1.0 if row["auroc_reg_vs_improvement"] is None else float(row["auroc_reg_vs_improvement"]),
                -1.0 if row["auroc_reg_vs_non_rescue"] is None else float(row["auroc_reg_vs_non_rescue"]),
            ),
            reverse=True,
        )[:12]
        if not ranked:
            continue
        labels = [str(row["feature"]) for row in ranked]
        vals = [0.0 if row["auroc_reg_vs_improvement"] is None else float(row["auroc_reg_vs_improvement"]) for row in ranked]
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        ax.barh(labels[::-1], vals[::-1], color="#7570b3")
        ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("AUROC (regression vs improvement)")
        ax.set_title(f"Top contamination-reducing features inside {subset}")
        fig.tight_layout()
        path = os.path.join(out_dir, f"{subset}_top_features_reg_vs_improvement.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze low-Stage-B subsets for contamination and candidate Stage-C features.")
    ap.add_argument("--scores_csv", type=str, required=True)
    ap.add_argument("--taxonomy_csv", type=str, default="")
    ap.add_argument("--features_csv", type=str, default="")
    ap.add_argument("--subset_percents", type=str, default="1,2,5,10")
    ap.add_argument("--topk_features", type=int, default=20)
    ap.add_argument("--make_plots", type=str, default="true")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    score_rows = load_csv_rows(args.scores_csv)
    taxonomy_rows = load_csv_rows(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else None
    feature_rows = load_csv_rows(args.features_csv) if str(args.features_csv).strip() else None
    merged_rows = merge_rows(score_rows=score_rows, taxonomy_rows=taxonomy_rows, feature_rows=feature_rows)
    feature_names = maybe_numeric_feature_names(merged_rows)

    percents = parse_percents(args.subset_percents)
    subset_summary_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    feature_rows_all: List[Dict[str, Any]] = []

    for percent in percents:
        subset_name = f"bottom_{str(percent).replace('.', 'p')}pct"
        subset = subset_by_bottom_percent(merged_rows, percent=percent)
        subset_summary_rows.append(subset_composition_row(subset_name, subset, percent=percent))
        pairwise_rows.extend(pairwise_metrics_for_subset(subset_name, subset, percent=percent))
        feature_rows_all.extend(feature_screen_rows(subset_name, subset, percent=percent, feature_names=feature_names))

    top_rows = top_feature_rows(feature_rows_all, topk=int(args.topk_features))
    plot_paths = maybe_make_plots(os.path.join(args.out_dir, "plots"), subset_summary_rows, top_rows) if parse_bool(args.make_plots) else []

    subset_csv = os.path.join(args.out_dir, "subset_composition.csv")
    pairwise_csv = os.path.join(args.out_dir, "subset_pairwise_metrics.csv")
    feature_csv = os.path.join(args.out_dir, "feature_screen.csv")
    top_csv = os.path.join(args.out_dir, "feature_top.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_csv(subset_csv, subset_summary_rows)
    write_csv(pairwise_csv, pairwise_rows)
    write_csv(feature_csv, feature_rows_all)
    write_csv(top_csv, top_rows)

    summary = {
        "inputs": {
            "scores_csv": os.path.abspath(args.scores_csv),
            "taxonomy_csv": (os.path.abspath(args.taxonomy_csv) if str(args.taxonomy_csv).strip() else ""),
            "features_csv": (os.path.abspath(args.features_csv) if str(args.features_csv).strip() else ""),
            "subset_percents": parse_percents(args.subset_percents),
            "n_feature_candidates": int(len(feature_names)),
        },
        "counts": {
            "n_rows": int(len(merged_rows)),
            "n_valid_stageb_rows": int(sum(1 for row in merged_rows if maybe_float(row.get("stage_b_score")) is not None and str(row.get("case_type")) != "unknown")),
        },
        "outputs": {
            "subset_composition_csv": os.path.abspath(subset_csv),
            "subset_pairwise_csv": os.path.abspath(pairwise_csv),
            "feature_screen_csv": os.path.abspath(feature_csv),
            "feature_top_csv": os.path.abspath(top_csv),
            "plot_paths": [os.path.abspath(p) for p in plot_paths],
        },
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", subset_csv)
    print("[saved]", pairwise_csv)
    print("[saved]", feature_csv)
    print("[saved]", top_csv)
    print("[saved]", summary_json)
    for path in plot_paths:
        print("[saved]", path)


if __name__ == "__main__":
    main()
