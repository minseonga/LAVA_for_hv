#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = ("adversarial", "popular", "random")
CATEGORY_LABELS = {
    "adversarial": "Adversarial",
    "popular": "Popular",
    "random": "Random",
}
COLORS = {
    "baseline": "#7A869A",
    "method": "#D97706",
    "rapic": "#1D4ED8",
    "harm": "#C62828",
    "help": "#2E7D32",
    "line": "#94A3B8",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def safe_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def parse_yes_no(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip().lower() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    if s.lower().startswith("no"):
        return "no"
    if s.lower().startswith("yes"):
        return "yes"
    return "yes"


def load_gt(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv(path):
        qid = safe_id(row.get("id") or row.get("question_id"))
        answer = safe_id(row.get("answer") or row.get("label")).lower()
        category = safe_id(row.get("category")).lower()
        if qid and answer in {"yes", "no"}:
            out[qid] = {"answer": answer, "category": category or "overall"}
    return out


def load_pred(path: Path, text_key: str = "auto") -> Dict[str, str]:
    keys = [text_key] if text_key and text_key != "auto" else []
    keys.extend(["text", "output", "answer", "caption", "pred", "prediction"])
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = safe_id(row.get("question_id") or row.get("id"))
            if not qid:
                continue
            text = ""
            for key in keys:
                if key in row and str(row[key]).strip():
                    text = str(row[key])
                    break
            out[qid] = parse_yes_no(text)
    return out


def load_routes(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv(path):
        qid = safe_id(row.get("id") or row.get("question_id"))
        if not qid:
            continue
        out[qid] = {
            "route": safe_id(row.get("route")).lower() or "method",
            "harm": int(float(row.get("harm") or 0)),
            "help": int(float(row.get("help") or 0)),
        }
    return out


def dataset_gt_path(cal_root: Path, dataset: str) -> Path:
    if dataset == "mscoco":
        return cal_root / "experiments/pope_full_9000/pope_9000_gt.csv"
    if dataset == "aokvqa":
        return cal_root / "experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_gt.csv"
    if dataset == "gqa":
        return cal_root / "experiments/pope_hf_multidataset/gqa/pope_gqa_9000_gt.csv"
    raise ValueError(f"unsupported dataset={dataset!r}")


def baseline_pred_path(cal_root: Path, target: str, dataset: str) -> Path:
    if target == "vga_llava15" and dataset == "mscoco":
        return cal_root / "experiments/pope_full_9000/stage_b_signal_validation_vga/pred_baseline.jsonl"
    if target == "vga_llava15" and dataset == "aokvqa":
        return cal_root / "experiments/paper_raw/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/baseline_full9000/pred_baseline.jsonl"
    if target == "vga_llava15" and dataset == "gqa":
        return cal_root / "experiments/paper_raw/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/baseline_full9000/pred_baseline.jsonl"
    raise ValueError(f"unsupported baseline path target={target!r} dataset={dataset!r}")


def method_pred_path(cal_root: Path, target: str, dataset: str) -> Path:
    if target == "vga_llava15" and dataset == "mscoco":
        return Path(os.environ.get("VGA_LLAVA15_MSCOCO_PRED", "")) if os.environ.get("VGA_LLAVA15_MSCOCO_PRED") else cal_root / "experiments/pope_full_9000/all_models_full_strict/vga/pred_vga_9000.jsonl"
    if target == "vga_llava15" and dataset == "aokvqa":
        return cal_root / "experiments/paper_raw/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/vga_full9000/pred_vga.jsonl"
    if target == "vga_llava15" and dataset == "gqa":
        return cal_root / "experiments/paper_raw/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/vga_full9000/pred_vga.jsonl"
    raise ValueError(f"unsupported method path target={target!r} dataset={dataset!r}")


def route_rows_path(apply_root: Path, target: str, dataset: str) -> Path:
    return apply_root / target / dataset / "pcp_route_rows.csv"


def init_bucket() -> Dict[str, Any]:
    return {
        "n": 0,
        "baseline_correct": 0,
        "method_correct": 0,
        "rapic_correct": 0,
        "method_harm": 0,
        "method_help": 0,
        "selected_harm": 0,
        "selected_help": 0,
        "fallback": 0,
    }


def compute_rows(
    *,
    gt: Dict[str, Dict[str, str]],
    baseline: Dict[str, str],
    method: Dict[str, str],
    routes: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    buckets = {cat: init_bucket() for cat in CATEGORY_ORDER}
    buckets["overall"] = init_bucket()

    for qid, g in gt.items():
        cat = g["category"]
        if cat not in buckets:
            continue
        gold = g["answer"]
        b = baseline.get(qid, "")
        m = method.get(qid, "")
        route = routes.get(qid, {"route": "method", "harm": 0, "help": 0})
        final = b if route["route"] == "baseline" else m
        b_correct = int(b == gold)
        m_correct = int(m == gold)
        r_correct = int(final == gold)
        changed = int(b in {"yes", "no"} and m in {"yes", "no"} and b != m)
        harm = int(changed and b_correct and not m_correct)
        help_ = int(changed and (not b_correct) and m_correct)
        selected = int(route["route"] == "baseline")
        for key in (cat, "overall"):
            bucket = buckets[key]
            bucket["n"] += 1
            bucket["baseline_correct"] += b_correct
            bucket["method_correct"] += m_correct
            bucket["rapic_correct"] += r_correct
            bucket["method_harm"] += harm
            bucket["method_help"] += help_
            bucket["selected_harm"] += int(selected and harm)
            bucket["selected_help"] += int(selected and help_)
            bucket["fallback"] += selected

    rows: List[Dict[str, Any]] = []
    for cat in list(CATEGORY_ORDER) + ["overall"]:
        b = buckets[cat]
        n = max(1, int(b["n"]))
        final_harm = int(b["method_harm"] - b["selected_harm"])
        final_help = int(b["method_help"] - b["selected_help"])
        rows.append(
            {
                "category": cat,
                "n": int(b["n"]),
                "baseline_acc": b["baseline_correct"] / float(n),
                "method_acc": b["method_correct"] / float(n),
                "rapic_acc": b["rapic_correct"] / float(n),
                "method_harm": int(b["method_harm"]),
                "method_help": int(b["method_help"]),
                "selected_harm": int(b["selected_harm"]),
                "selected_help": int(b["selected_help"]),
                "final_harm": final_harm,
                "final_help": final_help,
                "fallback": int(b["fallback"]),
                "raw_gain_harm_ratio": b["method_help"] / float(max(1, b["method_harm"])),
                "rapic_gain_harm_ratio": final_help / float(max(1, final_harm)),
            }
        )
    return rows


def panel_accuracy(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> None:
    cats = list(CATEGORY_ORDER)
    row_map = {r["category"]: r for r in rows}
    x = np.arange(len(cats))
    width = 0.24
    series = [
        ("Base", "baseline_acc", COLORS["baseline"]),
        ("VGA", "method_acc", COLORS["method"]),
        ("RAPIC", "rapic_acc", COLORS["rapic"]),
    ]
    for idx, (label, key, color) in enumerate(series):
        vals = [100.0 * float(row_map[c][key]) for c in cats]
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, color=color, label=label)
        for rect, val in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.25, f"{val:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x, [CATEGORY_LABELS[c] for c in cats])
    vals_all = [100.0 * float(row_map[c][k]) for c in cats for k in ("baseline_acc", "method_acc", "rapic_acc")]
    ax.set_ylim(max(0.0, min(vals_all) - 3.0), min(100.0, max(vals_all) + 4.0))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Split-wise accuracy")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_gain_harm(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> None:
    cats = list(CATEGORY_ORDER)
    row_map = {r["category"]: r for r in rows}
    y = np.arange(len(cats))[::-1]
    raw_vals = [float(row_map[c]["raw_gain_harm_ratio"]) for c in cats]
    rapic_vals = [float(row_map[c]["rapic_gain_harm_ratio"]) for c in cats]
    for yy, cat, raw, rapic in zip(y, cats, raw_vals, rapic_vals):
        row = row_map[cat]
        ax.plot([raw, rapic], [yy, yy], color=COLORS["line"], linewidth=2.0, zorder=1)
        ax.scatter(raw, yy, color=COLORS["method"], s=56, zorder=2, label="VGA" if cat == cats[0] else None)
        ax.scatter(rapic, yy, color=COLORS["rapic"], s=56, zorder=3, label="RAPIC" if cat == cats[0] else None)
        ax.text(raw, yy + 0.12, f"{int(row['method_help'])}/{int(row['method_harm'])}", ha="center", va="bottom", fontsize=7.0, color=COLORS["method"])
        ax.text(rapic, yy - 0.15, f"{int(row['final_help'])}/{int(row['final_harm'])}", ha="center", va="top", fontsize=7.0, color=COLORS["rapic"])
    ax.set_yticks(y, [CATEGORY_LABELS[c] for c in cats])
    vals = raw_vals + rapic_vals
    ax.set_xlim(max(0.0, min(vals) - 0.2), max(vals) + 0.35)
    ax.set_xlabel("Helpful gains / harmful flips")
    ax.set_title("(b) Gain-to-harm ratio")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_selected_hg(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> None:
    cats = list(CATEGORY_ORDER)
    row_map = {r["category"]: r for r in rows}
    x = np.arange(len(cats))
    width = 0.34
    harm = [float(row_map[c]["selected_harm"]) for c in cats]
    help_ = [float(row_map[c]["selected_help"]) for c in cats]
    bars_h = ax.bar(x - width / 2, harm, width=width, color=COLORS["harm"], label="Harm caught")
    bars_g = ax.bar(x + width / 2, help_, width=width, color=COLORS["help"], label="Help lost")
    for bars in (bars_h, bars_g):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.0, f"{int(rect.get_height())}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x, [CATEGORY_LABELS[c] for c in cats])
    ax.set_ylabel("Selected fallbacks")
    ax.set_title("(c) Selected harm vs lost gain")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.24, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(rows: Sequence[Dict[str, Any]], out_path: Path, title: str) -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 9.1,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.9), gridspec_kw={"width_ratios": [1.12, 1.0, 1.0], "wspace": 0.36})
    panel_accuracy(axes[0], rows)
    panel_gain_harm(axes[1], rows)
    panel_selected_hg(axes[2], rows)
    fig.suptitle(title, y=1.05, fontsize=13)
    save_fig(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make the fixed-C3 RAPIC main case-study figure.")
    ap.add_argument("--cal_root", default="/home/kms/LLaVA_calibration")
    ap.add_argument("--apply_root", required=True, help="Root containing target/dataset/pcp_route_rows.csv")
    ap.add_argument("--target", default="vga_llava15")
    ap.add_argument("--dataset", default="mscoco", choices=["mscoco", "aokvqa", "gqa"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--baseline_pred_jsonl", default="", help="Optional override")
    ap.add_argument("--method_pred_jsonl", default="", help="Optional override")
    args = ap.parse_args()

    cal_root = Path(args.cal_root).resolve()
    apply_root = Path(args.apply_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    gt_path = dataset_gt_path(cal_root, args.dataset)
    baseline_path = Path(args.baseline_pred_jsonl).resolve() if args.baseline_pred_jsonl else baseline_pred_path(cal_root, args.target, args.dataset)
    method_path = Path(args.method_pred_jsonl).resolve() if args.method_pred_jsonl else method_pred_path(cal_root, args.target, args.dataset)
    route_path = route_rows_path(apply_root, args.target, args.dataset)
    for path in (gt_path, baseline_path, method_path, route_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = compute_rows(
        gt=load_gt(gt_path),
        baseline=load_pred(baseline_path),
        method=load_pred(method_path),
        routes=load_routes(route_path),
    )

    prefix = f"{args.target}_{args.dataset}_fixed_c3_case_study"
    metrics_csv = out_dir / f"{prefix}_metrics.csv"
    summary_json = out_dir / f"{prefix}_summary.json"
    fig_path = out_dir / f"{prefix}_figure.png"
    write_csv(metrics_csv, rows)
    make_figure(rows, fig_path, f"Fixed C3 RAPIC case study: VGA / LLaVA-1.5 / {args.dataset.upper()}")
    write_json(
        summary_json,
        {
            "inputs": {
                "gt_csv": str(gt_path),
                "baseline_pred_jsonl": str(baseline_path),
                "method_pred_jsonl": str(method_path),
                "route_rows_csv": str(route_path),
            },
            "rows": rows,
            "outputs": {
                "metrics_csv": str(metrics_csv),
                "png": str(fig_path),
                "pdf": str(fig_path.with_suffix(".pdf")),
            },
        },
    )
    print("[saved]", fig_path)
    print("[saved]", fig_path.with_suffix(".pdf"))
    print("[saved]", metrics_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
