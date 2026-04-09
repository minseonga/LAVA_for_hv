#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)

CASE_ORDER = ["help", "neutral", "harm"]
CASE_LABELS = {"help": "Improvement", "neutral": "Neutral", "harm": "Regression"}
CASE_COLORS = {"help": "#2E8B57", "neutral": "#9AA0A6", "harm": "#C0392B"}
FEATURE_SPECS: List[Tuple[str, str, str]] = [
    ("cheap_target_gap_content_min", "low", "Target Gap Min"),
    ("cheap_lp_content_min", "low", "Target LogProb Min"),
    ("cheap_lp_content_std", "high", "Target LogProb Std"),
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_boolish(value: object) -> int:
    s = str(value if value is not None else "").strip().lower()
    return 1 if s in {"1", "true", "yes"} else 0


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def classify_case(row: Dict[str, str]) -> str:
    base = parse_boolish(row.get("baseline_correct"))
    intr = parse_boolish(row.get("intervention_correct"))
    if intr == 1 and base == 0:
        return "help"
    if intr == 0 and base == 1:
        return "harm"
    return "neutral"


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and pairs[order[j]][0] == pairs[order[i]][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[i] for i, (_, y) in enumerate(pairs) if y == 1)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def merge_rows(scores_rows: Sequence[Dict[str, str]], feature_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    score_map = {str(row.get("id", "")).strip(): dict(row) for row in scores_rows if str(row.get("id", "")).strip()}
    merged: List[Dict[str, str]] = []
    for row in feature_rows:
        sid = str(row.get("id", "")).strip()
        if sid == "" or sid not in score_map:
            continue
        item = dict(score_map[sid])
        item.update(row)
        item["case_type"] = classify_case(item)
        merged.append(item)
    return merged


def sample_points(values: Sequence[float], max_n: int, seed: int = 0) -> List[float]:
    vals = [float(v) for v in values]
    if len(vals) <= int(max_n):
        return vals
    rng = random.Random(int(seed))
    picked = list(vals)
    rng.shuffle(picked)
    return picked[: int(max_n)]


def draw_case_boxplot(
    ax: plt.Axes,
    rows: Sequence[Dict[str, str]],
    feature: str,
    direction: str,
    label: str,
) -> Dict[str, Any]:
    case_values: Dict[str, List[float]] = {case: [] for case in CASE_ORDER}
    scores: List[float] = []
    labels_bin: List[int] = []

    for row in rows:
        value = maybe_float(row.get(feature))
        if value is None:
            continue
        case = str(row.get("case_type", "neutral"))
        case_values.setdefault(case, []).append(float(value))
        oriented = float(value) if direction == "high" else -float(value)
        scores.append(oriented)
        labels_bin.append(1 if case == "harm" else 0)

    positions = [1, 2, 3]
    data = [case_values[case] for case in CASE_ORDER]
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#777777", "linewidth": 1.0},
        capprops={"color": "#777777", "linewidth": 1.0},
    )
    for patch, case in zip(bp["boxes"], CASE_ORDER):
        patch.set_facecolor(CASE_COLORS[case])
        patch.set_edgecolor(CASE_COLORS[case])
        patch.set_alpha(0.35)
        patch.set_linewidth(1.4)

    for pos, case in zip(positions, CASE_ORDER):
        vals = sample_points(case_values[case], max_n=220, seed=17 + pos)
        rng = random.Random(100 + pos)
        xs = [float(pos) + rng.uniform(-0.13, 0.13) for _ in vals]
        ax.scatter(
            xs,
            vals,
            s=8,
            alpha=0.20,
            color=CASE_COLORS[case],
            edgecolors="none",
            zorder=3,
        )

    auc = binary_auroc(scores, labels_bin)
    direction_note = "Lower = riskier" if direction == "low" else "Higher = riskier"
    ax.set_title(f"{label}\nAUROC {auc:.3f}  |  {direction_note}" if auc is not None else label)
    ax.set_xticks(positions)
    ax.set_xticklabels([CASE_LABELS[case] for case in CASE_ORDER])
    ax.tick_params(axis="x", length=0)
    ax.grid(True, axis="y", linestyle=(0, (1, 2)), alpha=0.35)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return {
        "feature": feature,
        "label": label,
        "direction": direction,
        "auroc_harm_vs_rest": None if auc is None else float(auc),
        "n_help": int(len(case_values["help"])),
        "n_neutral": int(len(case_values["neutral"])),
        "n_harm": int(len(case_values["harm"])),
        "help_mean": float(sum(case_values["help"]) / len(case_values["help"])) if case_values["help"] else 0.0,
        "neutral_mean": float(sum(case_values["neutral"]) / len(case_values["neutral"])) if case_values["neutral"] else 0.0,
        "harm_mean": float(sum(case_values["harm"]) / len(case_values["harm"])) if case_values["harm"] else 0.0,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    scores_csv = repo_root / "experiments/paper_main_b_c_v1_full/discovery_stageb/sample_scores.csv"
    features_csv = repo_root / "experiments/paper_main_b_c_v1_full/discovery/cheap_online_features.csv"
    out_root = repo_root / "experiments/discriminative_mechanism_viz"
    out_png = out_root / "figure_disc_selected_c_features.png"
    out_pdf = out_png.with_suffix(".pdf")
    out_summary = out_root / "figure_disc_selected_c_features.summary.json"

    merged = merge_rows(read_csv(scores_csv), read_csv(features_csv))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    summaries: List[Dict[str, Any]] = []
    for ax, (feature, direction, label) in zip(axes, FEATURE_SPECS):
        summaries.append(draw_case_boxplot(ax, merged, feature, direction, label))
    axes[0].set_ylabel("Feature Value")
    fig.suptitle("Selected C-Side Features in Strong Meta Discovery", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    ensure_parent(out_png)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    counts = {"help": 0, "neutral": 0, "harm": 0}
    for row in merged:
        counts[str(row.get("case_type", "neutral"))] += 1
    write_json(
        out_summary,
        {
            "inputs": {
                "scores_csv": str(scores_csv),
                "features_csv": str(features_csv),
            },
            "counts": counts,
            "features": summaries,
            "outputs": {
                "png": str(out_png),
                "pdf": str(out_pdf),
            },
        },
    )
    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()
