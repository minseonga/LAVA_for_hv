#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Tuple

import pandas as pd


YES_SET = {"yes", "y", "1", "true"}
NO_SET = {"no", "n", "0", "false"}


def to_label(x: str) -> str:
    s = str(x).strip().lower()
    if s in YES_SET:
        return "yes"
    if s in NO_SET:
        return "no"
    raise ValueError(f"Unsupported label value: {x}")


def metrics_from_cols(df: pd.DataFrame, pred_col: str, gt_col: str = "gt") -> Dict[str, float]:
    y = df[gt_col].map(to_label)
    p = df[pred_col].map(to_label)
    tp = int(((p == "yes") & (y == "yes")).sum())
    fp = int(((p == "yes") & (y == "no")).sum())
    tn = int(((p == "no") & (y == "no")).sum())
    fn = int(((p == "no") & (y == "yes")).sum())
    n = int(len(df))
    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": n,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "yes_ratio": float((p == "yes").mean()) if n else 0.0,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def diff_vs_base(df: pd.DataFrame, base_col: str, new_col: str, gt_col: str = "gt") -> Dict[str, int]:
    y = df[gt_col].map(to_label)
    b = df[base_col].map(to_label)
    n = df[new_col].map(to_label)
    changed = (b != n)
    gain = ((b != y) & (n == y) & changed).sum()
    harm = ((b == y) & (n != y) & changed).sum()
    return {
        "changed_pred": int(changed.sum()),
        "gain": int(gain),
        "harm": int(harm),
        "net_gain": int(gain - harm),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build decision-level VGA oracles from baseline-vs-VGA per-case table.")
    ap.add_argument("--per_case_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--gt_col", type=str, default="gt")
    ap.add_argument("--baseline_col", type=str, default="pred_baseline")
    ap.add_argument("--vga_col", type=str, default="pred_vga")
    ap.add_argument("--case_col", type=str, default="case_type")
    ap.add_argument("--tie_preference", type=str, default="vga", choices=["baseline", "vga"])
    ap.add_argument("--regime_safe_action", type=str, default="vga", choices=["baseline", "vga"])
    ap.add_argument("--regime_hard_action", type=str, default="vga", choices=["baseline", "vga"])
    ap.add_argument("--regime_d1_action", type=str, default="vga", choices=["baseline", "vga"])
    ap.add_argument("--regime_d2_action", type=str, default="baseline", choices=["baseline", "vga"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.per_case_csv)

    keep_cols = [args.id_col, args.gt_col, args.baseline_col, args.vga_col, args.case_col]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    out.rename(
        columns={
            args.id_col: "id",
            args.gt_col: "gt",
            args.baseline_col: "pred_baseline",
            args.vga_col: "pred_vga",
            args.case_col: "case_type",
        },
        inplace=True,
    )
    out["gt"] = out["gt"].map(to_label)
    out["pred_baseline"] = out["pred_baseline"].map(to_label)
    out["pred_vga"] = out["pred_vga"].map(to_label)

    # Oracle 1: best-of-two switch oracle
    # Equivalent to choosing correct model by case-type labels.
    def pick_best(row: pd.Series) -> str:
        case = str(row["case_type"])
        if case == "vga_improvement":
            return row["pred_vga"]
        if case == "vga_regression":
            return row["pred_baseline"]
        return row["pred_vga"] if args.tie_preference == "vga" else row["pred_baseline"]

    out["oracle_best_of_two"] = out.apply(pick_best, axis=1)

    # Oracle 2: veto-only (turn off VGA only on D2 = vga_regression)
    out["oracle_veto_only"] = out.apply(
        lambda r: r["pred_baseline"] if str(r["case_type"]) == "vga_regression" else r["pred_vga"], axis=1
    )

    # Oracle 3: 4-way regime oracle with action map
    action_map = {
        "both_correct": args.regime_safe_action,
        "both_wrong": args.regime_hard_action,
        "vga_improvement": args.regime_d1_action,
        "vga_regression": args.regime_d2_action,
    }

    def pick_regime(row: pd.Series) -> str:
        action = action_map.get(str(row["case_type"]), args.tie_preference)
        return row["pred_vga"] if action == "vga" else row["pred_baseline"]

    out["oracle_regime_4way"] = out.apply(pick_regime, axis=1)

    # sanity: in binary setting, best-of-two and veto-only are identical by definition.
    out["best_equals_veto"] = (out["oracle_best_of_two"] == out["oracle_veto_only"]).astype(int)
    out["best_equals_regime"] = (out["oracle_best_of_two"] == out["oracle_regime_4way"]).astype(int)

    per_id_csv = os.path.join(args.out_dir, "oracle_per_id.csv")
    out.to_csv(per_id_csv, index=False)

    # metrics
    m_base = metrics_from_cols(out, "pred_baseline", "gt")
    m_vga = metrics_from_cols(out, "pred_vga", "gt")
    m_best = metrics_from_cols(out, "oracle_best_of_two", "gt")
    m_veto = metrics_from_cols(out, "oracle_veto_only", "gt")
    m_reg = metrics_from_cols(out, "oracle_regime_4way", "gt")

    summary = {
        "inputs": {
            "per_case_csv": os.path.abspath(args.per_case_csv),
            "tie_preference": args.tie_preference,
            "regime_action_map": action_map,
        },
        "counts": {
            "n_rows": int(len(out)),
            "case_type": out["case_type"].value_counts().to_dict(),
            "best_equals_veto_frac": float(out["best_equals_veto"].mean()),
            "best_equals_regime_frac": float(out["best_equals_regime"].mean()),
        },
        "metrics": {
            "baseline": m_base,
            "vga": m_vga,
            "oracle_best_of_two": m_best,
            "oracle_veto_only": m_veto,
            "oracle_regime_4way": m_reg,
        },
        "delta_vs_baseline": {
            "oracle_best_of_two": {**diff_vs_base(out, "pred_baseline", "oracle_best_of_two", "gt"), "delta_acc": m_best["acc"] - m_base["acc"], "delta_f1": m_best["f1"] - m_base["f1"]},
            "oracle_veto_only": {**diff_vs_base(out, "pred_baseline", "oracle_veto_only", "gt"), "delta_acc": m_veto["acc"] - m_base["acc"], "delta_f1": m_veto["f1"] - m_base["f1"]},
            "oracle_regime_4way": {**diff_vs_base(out, "pred_baseline", "oracle_regime_4way", "gt"), "delta_acc": m_reg["acc"] - m_base["acc"], "delta_f1": m_reg["f1"] - m_base["f1"]},
        },
        "delta_vs_vga": {
            "oracle_best_of_two": {**diff_vs_base(out, "pred_vga", "oracle_best_of_two", "gt"), "delta_acc": m_best["acc"] - m_vga["acc"], "delta_f1": m_best["f1"] - m_vga["f1"]},
            "oracle_veto_only": {**diff_vs_base(out, "pred_vga", "oracle_veto_only", "gt"), "delta_acc": m_veto["acc"] - m_vga["acc"], "delta_f1": m_veto["f1"] - m_vga["f1"]},
            "oracle_regime_4way": {**diff_vs_base(out, "pred_vga", "oracle_regime_4way", "gt"), "delta_acc": m_reg["acc"] - m_vga["acc"], "delta_f1": m_reg["f1"] - m_vga["f1"]},
        },
        "outputs": {"per_id_csv": per_id_csv},
    }

    summary_json = os.path.join(args.out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", per_id_csv)
    print("[saved]", summary_json)
    print("[summary]", json.dumps({
        "base_acc": m_base["acc"],
        "vga_acc": m_vga["acc"],
        "best2_acc": m_best["acc"],
        "veto_acc": m_veto["acc"],
        "regime_acc": m_reg["acc"],
        "delta_best2_vs_vga": m_best["acc"] - m_vga["acc"],
    }))


if __name__ == "__main__":
    main()

