#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import write_csv, write_json


def maybe_int(value: object) -> Optional[int]:
    s = str(value or "").strip()
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return None
    return int(float(s))


def maybe_float(value: object) -> Optional[float]:
    s = str(value or "").strip()
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return None
    out = float(s)
    if not math.isfinite(out):
        return None
    return out


def mean_or_zero(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std_or_zero(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0
    mu = mean_or_zero(seq)
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(math.sqrt(max(0.0, var)))


def median(values: Sequence[float]) -> float:
    seq = sorted(float(v) for v in values)
    if not seq:
        return 0.0
    n = len(seq)
    m = n // 2
    if n % 2 == 1:
        return float(seq[m])
    return float((seq[m - 1] + seq[m]) / 2.0)


def quantile(values: Sequence[float], q: float) -> float:
    seq = sorted(float(v) for v in values)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    qq = min(1.0, max(0.0, float(q)))
    pos = qq * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(seq[lo])
    w = pos - float(lo)
    return float((1.0 - w) * seq[lo] + w * seq[hi])


def load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def binary_auc(risk_scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pos = sum(1 for y in labels if int(y) == 1)
    neg = sum(1 for y in labels if int(y) == 0)
    if pos == 0 or neg == 0:
        return None
    ranks = assign_avg_ranks(risk_scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if int(y) == 1)
    u = rank_sum_pos - float(pos * (pos + 1)) / 2.0
    return float(u / float(pos * neg))


def average_precision(risk_scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pos = sum(1 for y in labels if int(y) == 1)
    if pos == 0:
        return None
    order = sorted(range(len(risk_scores)), key=lambda i: float(risk_scores[i]), reverse=True)
    tp = 0
    ap = 0.0
    for rank, idx in enumerate(order, start=1):
        if int(labels[idx]) == 1:
            tp += 1
            ap += float(tp) / float(rank)
    return float(ap / float(pos))


def roc_curve_points(risk_scores: Sequence[float], labels: Sequence[int]) -> List[Dict[str, float]]:
    pos = sum(1 for y in labels if int(y) == 1)
    neg = sum(1 for y in labels if int(y) == 0)
    if pos == 0 or neg == 0:
        return []
    order = sorted(range(len(risk_scores)), key=lambda i: float(risk_scores[i]), reverse=True)
    tp = 0
    fp = 0
    out: List[Dict[str, float]] = [{"threshold": float("inf"), "tpr": 0.0, "fpr": 0.0}]
    prev_score: Optional[float] = None
    for idx in order:
        score = float(risk_scores[idx])
        if prev_score is not None and score != prev_score:
            out.append(
                {
                    "threshold": float(prev_score),
                    "tpr": float(tp / float(pos)),
                    "fpr": float(fp / float(neg)),
                }
            )
        if int(labels[idx]) == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    out.append(
        {
            "threshold": float(prev_score if prev_score is not None else 0.0),
            "tpr": float(tp / float(pos)),
            "fpr": float(fp / float(neg)),
        }
    )
    return out


def pr_curve_points(risk_scores: Sequence[float], labels: Sequence[int]) -> List[Dict[str, float]]:
    pos = sum(1 for y in labels if int(y) == 1)
    if pos == 0:
        return []
    order = sorted(range(len(risk_scores)), key=lambda i: float(risk_scores[i]), reverse=True)
    tp = 0
    fp = 0
    out: List[Dict[str, float]] = []
    prev_score: Optional[float] = None
    for idx in order:
        score = float(risk_scores[idx])
        if prev_score is not None and score != prev_score:
            recall = float(tp / float(pos))
            precision = float(tp / float(max(1, tp + fp)))
            out.append({"threshold": float(prev_score), "recall": recall, "precision": precision})
        if int(labels[idx]) == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    recall = float(tp / float(pos))
    precision = float(tp / float(max(1, tp + fp)))
    out.append({"threshold": float(prev_score if prev_score is not None else 0.0), "recall": recall, "precision": precision})
    return out


def build_group_stats(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups = sorted(set(str(row["case_type"]) for row in rows))
    for group in groups:
        sub = [row for row in rows if str(row["case_type"]) == group]
        scores = [float(row["stage_b_score"]) for row in sub]
        risks = [float(row["risk_score"]) for row in sub]
        out.append(
            {
                "case_type": group,
                "n": int(len(sub)),
                "stage_b_score_mean": mean_or_zero(scores),
                "stage_b_score_std": std_or_zero(scores),
                "stage_b_score_median": median(scores),
                "stage_b_score_q25": quantile(scores, 0.25),
                "stage_b_score_q75": quantile(scores, 0.75),
                "risk_score_mean": mean_or_zero(risks),
                "risk_score_std": std_or_zero(risks),
            }
        )
    return out


def build_pairwise_metrics(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, float]]], Dict[str, List[Dict[str, float]]]]:
    pairs = [
        ("regression_vs_non_regression", {"vga_regression"}, {"vga_improvement", "both_correct", "both_wrong"}),
        ("regression_vs_improvement", {"vga_regression"}, {"vga_improvement"}),
        ("regression_vs_both_correct", {"vga_regression"}, {"both_correct"}),
        ("regression_vs_both_wrong", {"vga_regression"}, {"both_wrong"}),
    ]
    summary_rows: List[Dict[str, Any]] = []
    roc_map: Dict[str, List[Dict[str, float]]] = {}
    pr_map: Dict[str, List[Dict[str, float]]] = {}

    for name, pos_set, neg_set in pairs:
        sub = [row for row in rows if row["case_type"] in pos_set or row["case_type"] in neg_set]
        labels = [1 if row["case_type"] in pos_set else 0 for row in sub]
        scores = [float(row["risk_score"]) for row in sub]
        auc = binary_auc(scores, labels)
        ap = average_precision(scores, labels)
        roc = roc_curve_points(scores, labels)
        pr = pr_curve_points(scores, labels)
        roc_map[name] = roc
        pr_map[name] = pr
        summary_rows.append(
            {
                "comparison": name,
                "n_total": int(len(sub)),
                "n_pos": int(sum(labels)),
                "n_neg": int(len(labels) - sum(labels)),
                "auroc": auc,
                "average_precision": ap,
            }
        )
    return summary_rows, roc_map, pr_map


def threshold_grid(values: Sequence[float]) -> List[float]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return [0.0]
    out = {finite[0] - 1e-6, finite[-1] + 1e-6}
    for q in [i / 100.0 for i in range(1, 100)]:
        out.add(quantile(finite, q))
    return sorted(out)


def operating_sweep(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [row for row in rows if row["case_type"] != "unknown"]
    regressions = [row for row in valid if row["case_type"] == "vga_regression"]
    thresholds = threshold_grid([float(row["stage_b_score"]) for row in valid])
    out: List[Dict[str, Any]] = []

    for tau in thresholds:
        flagged = [row for row in valid if float(row["stage_b_score"]) < float(tau)]
        kept = [row for row in valid if float(row["stage_b_score"]) >= float(tau)]
        n_flagged = len(flagged)
        n_reg_flagged = sum(1 for row in flagged if row["case_type"] == "vga_regression")
        n_reg_total = max(1, len(regressions))
        final_correct = 0
        n_eval = 0
        kept_int_correct = 0
        kept_n = 0
        for row in valid:
            int_correct = row["intervention_correct"]
            base_correct = row["baseline_correct"]
            if int_correct is None or base_correct is None:
                continue
            n_eval += 1
            if float(row["stage_b_score"]) < float(tau):
                final_correct += int(base_correct)
            else:
                final_correct += int(int_correct)
                kept_n += 1
                kept_int_correct += int(int_correct)
        out.append(
            {
                "tau_b": float(tau),
                "flagged_rate": float(n_flagged / max(1, len(valid))),
                "regression_precision": float(n_reg_flagged / max(1, n_flagged)),
                "regression_recall": float(n_reg_flagged / float(n_reg_total)),
                "kept_intervention_acc": float(kept_int_correct / max(1, kept_n)),
                "counterfactual_rescue_final_acc": (None if n_eval == 0 else float(final_correct / float(n_eval))),
                "n_flagged": int(n_flagged),
            }
        )
    return out


def budget_best_rows(sweep_rows: Sequence[Dict[str, Any]], budgets: Sequence[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for budget in budgets:
        feasible = [row for row in sweep_rows if float(row["flagged_rate"]) <= float(budget)]
        if not feasible:
            out.append({"flag_budget": float(budget), "found": False})
            continue
        best = max(
            feasible,
            key=lambda row: (
                -1.0 if row["counterfactual_rescue_final_acc"] is None else float(row["counterfactual_rescue_final_acc"]),
                -float(row["regression_recall"]),
                -float(row["flagged_rate"]),
            ),
        )
        out.append(
            {
                "flag_budget": float(budget),
                "found": True,
                **best,
            }
        )
    return out


def maybe_make_plots(
    out_dir: str,
    rows: Sequence[Dict[str, Any]],
    pr_map: Mapping[str, Sequence[Dict[str, float]]],
    roc_map: Mapping[str, Sequence[Dict[str, float]]],
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    saved: List[str] = []
    os.makedirs(out_dir, exist_ok=True)

    group_order = ["vga_regression", "vga_improvement", "both_correct", "both_wrong"]
    colors = {
        "vga_regression": "#d73027",
        "vga_improvement": "#1a9850",
        "both_correct": "#4575b4",
        "both_wrong": "#666666",
    }

    plt.figure(figsize=(8, 5))
    for group in group_order:
        sub = [float(row["stage_b_score"]) for row in rows if row["case_type"] == group]
        if not sub:
            continue
        plt.hist(sub, bins=40, alpha=0.45, density=True, label=group, color=colors.get(group))
    plt.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    plt.xlabel("Stage B score")
    plt.ylabel("Density")
    plt.title("Stage B Score by Case Type")
    plt.legend()
    hist_path = os.path.join(out_dir, "stage_b_score_hist.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=180)
    plt.close()
    saved.append(hist_path)

    key = "regression_vs_non_regression"
    roc = list(roc_map.get(key, []))
    if roc:
        plt.figure(figsize=(5, 5))
        plt.plot([row["fpr"] for row in roc], [row["tpr"] for row in roc], color="#d73027", linewidth=2.0)
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Stage B ROC: Regression vs Non-Regression")
        roc_path = os.path.join(out_dir, "stage_b_roc_regression_vs_non_regression.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=180)
        plt.close()
        saved.append(roc_path)

    pr = list(pr_map.get(key, []))
    if pr:
        plt.figure(figsize=(5, 5))
        plt.plot([row["recall"] for row in pr], [row["precision"] for row in pr], color="#1a9850", linewidth=2.0)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Stage B PR: Regression vs Non-Regression")
        pr_path = os.path.join(out_dir, "stage_b_pr_regression_vs_non_regression.png")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=180)
        plt.close()
        saved.append(pr_path)

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Stage B signal existence from clean-room score CSV.")
    ap.add_argument("--scores_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--make_plots", type=str, default="true")
    args = ap.parse_args()

    raw_rows = load_rows(args.scores_csv)
    valid_rows: List[Dict[str, Any]] = []
    score_error_count = 0
    for row in raw_rows:
        score = maybe_float(row.get("stage_b_score"))
        base_correct = maybe_int(row.get("baseline_correct"))
        int_correct = maybe_int(row.get("intervention_correct"))
        score_error = str(row.get("score_error", "")).strip()
        case_type = infer_case_type(base_correct=base_correct, int_correct=int_correct)
        if score_error:
            score_error_count += 1
        if score is None or case_type == "unknown":
            continue
        valid_rows.append(
            {
                "id": str(row.get("id", "")),
                "stage_b_score": float(score),
                "risk_score": float(-score),
                "baseline_correct": base_correct,
                "intervention_correct": int_correct,
                "case_type": case_type,
            }
        )

    group_stats = build_group_stats(valid_rows)
    pairwise_rows, roc_map, pr_map = build_pairwise_metrics(valid_rows)
    sweep_rows = operating_sweep(valid_rows)
    budget_rows = budget_best_rows(sweep_rows, budgets=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    plot_paths = maybe_make_plots(
        out_dir=os.path.join(args.out_dir, "plots"),
        rows=valid_rows,
        pr_map=pr_map,
        roc_map=roc_map,
    ) if str(args.make_plots).strip().lower() == "true" else []

    group_csv = os.path.join(args.out_dir, "group_stats.csv")
    pairwise_csv = os.path.join(args.out_dir, "pairwise_metrics.csv")
    sweep_csv = os.path.join(args.out_dir, "operating_sweep.csv")
    budget_csv = os.path.join(args.out_dir, "budget_best.csv")
    write_csv(group_csv, group_stats)
    write_csv(pairwise_csv, pairwise_rows)
    write_csv(sweep_csv, sweep_rows)
    write_csv(budget_csv, budget_rows)

    roc_dir = os.path.join(args.out_dir, "curve_csv")
    os.makedirs(roc_dir, exist_ok=True)
    for key, rows in roc_map.items():
        write_csv(os.path.join(roc_dir, f"{key}_roc.csv"), list(rows))
    for key, rows in pr_map.items():
        write_csv(os.path.join(roc_dir, f"{key}_pr.csv"), list(rows))

    main_pair = next((row for row in pairwise_rows if row["comparison"] == "regression_vs_non_regression"), None)
    summary = {
        "inputs": {
            "scores_csv": os.path.abspath(args.scores_csv),
        },
        "counts": {
            "n_raw_rows": int(len(raw_rows)),
            "n_valid_rows": int(len(valid_rows)),
            "n_score_error_rows": int(score_error_count),
            "n_regression": int(sum(1 for row in valid_rows if row["case_type"] == "vga_regression")),
            "n_improvement": int(sum(1 for row in valid_rows if row["case_type"] == "vga_improvement")),
            "n_both_correct": int(sum(1 for row in valid_rows if row["case_type"] == "both_correct")),
            "n_both_wrong": int(sum(1 for row in valid_rows if row["case_type"] == "both_wrong")),
        },
        "main_signal": main_pair,
        "outputs": {
            "group_csv": os.path.abspath(group_csv),
            "pairwise_csv": os.path.abspath(pairwise_csv),
            "sweep_csv": os.path.abspath(sweep_csv),
            "budget_csv": os.path.abspath(budget_csv),
            "plot_paths": [os.path.abspath(path) for path in plot_paths],
        },
    }
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_json(summary_json, summary)

    print("[saved]", group_csv)
    print("[saved]", pairwise_csv)
    print("[saved]", sweep_csv)
    print("[saved]", budget_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()

