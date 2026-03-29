#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


YES_SET = {"yes", "y", "1", "true"}
NO_SET = {"no", "n", "0", "false"}


def to_label(v) -> str:
    s = str(v).strip().lower()
    if s in YES_SET:
        return "yes"
    if s in NO_SET:
        return "no"
    raise ValueError(f"Unsupported label: {v}")


def metrics(y: pd.Series, p: pd.Series) -> Dict[str, float]:
    y = y.map(to_label)
    p = p.map(to_label)
    tp = int(((p == "yes") & (y == "yes")).sum())
    fp = int(((p == "yes") & (y == "no")).sum())
    tn = int(((p == "no") & (y == "no")).sum())
    fn = int(((p == "no") & (y == "yes")).sum())
    n = int(len(y))
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


def change_stats(y: pd.Series, base: pd.Series, new: pd.Series) -> Dict[str, int]:
    y = y.map(to_label)
    base = base.map(to_label)
    new = new.map(to_label)
    changed = (base != new)
    gain = ((base != y) & (new == y) & changed).sum()
    harm = ((base == y) & (new != y) & changed).sum()
    return {
        "changed_pred": int(changed.sum()),
        "gain": int(gain),
        "harm": int(harm),
        "net_gain": int(gain - harm),
    }


@dataclass
class TauResult:
    tau_c: float
    tau_e: float
    objective: float
    n_cal: int
    d1_wrong_veto: int
    d2_correct_veto: int
    veto_rate: float


def compute_veto_mask(
    c_vals: np.ndarray,
    e_vals: np.ndarray,
    tau_c: float,
    tau_e: float,
    use_c: bool = True,
    use_e: bool = True,
) -> np.ndarray:
    if use_c and use_e:
        return (c_vals >= tau_c) | (e_vals >= tau_e)
    if use_c:
        return c_vals >= tau_c
    if use_e:
        return e_vals >= tau_e
    return np.zeros_like(c_vals, dtype=bool)


def calibrate_thresholds(
    df_cal: pd.DataFrame,
    c_col: str,
    e_col: str,
    case_col: str,
    q_grid: List[float],
    lambda_d1: float,
    max_d1_wrong_rate: float,
    use_c: bool = True,
    use_e: bool = True,
) -> TauResult:
    c_arr = df_cal[c_col].to_numpy()
    e_arr = df_cal[e_col].to_numpy()
    d1 = (df_cal[case_col] == "vga_improvement").to_numpy()
    d2 = (df_cal[case_col] == "vga_regression").to_numpy()

    cand_c = sorted(set(float(np.quantile(c_arr, q)) for q in q_grid)) if use_c else [float("inf")]
    cand_e = sorted(set(float(np.quantile(e_arr, q)) for q in q_grid)) if use_e else [float("inf")]

    best: Optional[TauResult] = None
    for tc in cand_c:
        for te in cand_e:
            veto = compute_veto_mask(c_arr, e_arr, tc, te, use_c=use_c, use_e=use_e)
            d1_wrong = int((veto & d1).sum())
            d2_corr = int((veto & d2).sum())
            d1_total = int(d1.sum())
            d1_wrong_rate = (d1_wrong / d1_total) if d1_total > 0 else 0.0
            if d1_wrong_rate > max_d1_wrong_rate:
                continue
            obj = float(d2_corr - lambda_d1 * d1_wrong)
            cur = TauResult(
                tau_c=tc,
                tau_e=te,
                objective=obj,
                n_cal=len(df_cal),
                d1_wrong_veto=d1_wrong,
                d2_correct_veto=d2_corr,
                veto_rate=float(veto.mean()) if len(veto) else 0.0,
            )
            if best is None:
                best = cur
            else:
                # tie-break: higher objective, then lower d1_wrong, then lower veto rate
                if (cur.objective > best.objective) or (
                    np.isclose(cur.objective, best.objective) and (
                        cur.d1_wrong_veto < best.d1_wrong_veto or
                        (cur.d1_wrong_veto == best.d1_wrong_veto and cur.veto_rate < best.veto_rate)
                    )
                ):
                    best = cur

    if best is not None:
        return best

    # fallback when all candidates violate the D1 constraint: pick max objective regardless
    fallback: Optional[TauResult] = None
    for tc in cand_c:
        for te in cand_e:
            veto = compute_veto_mask(c_arr, e_arr, tc, te, use_c=use_c, use_e=use_e)
            d1_wrong = int((veto & d1).sum())
            d2_corr = int((veto & d2).sum())
            obj = float(d2_corr - lambda_d1 * d1_wrong)
            cur = TauResult(
                tau_c=tc,
                tau_e=te,
                objective=obj,
                n_cal=len(df_cal),
                d1_wrong_veto=d1_wrong,
                d2_correct_veto=d2_corr,
                veto_rate=float(veto.mean()) if len(veto) else 0.0,
            )
            if fallback is None or cur.objective > fallback.objective:
                fallback = cur
    assert fallback is not None
    return fallback


def merge_features(
    per_case: pd.DataFrame,
    features: pd.DataFrame,
    id_col_per_case: str,
    id_col_feat: str,
    c_col: str,
    e_col: str,
) -> pd.DataFrame:
    f = features[[id_col_feat, c_col, e_col]].copy()
    f = f.rename(columns={id_col_feat: "__id__", c_col: "__C__", e_col: "__E__"})
    m = per_case.copy()
    m["__id__"] = m[id_col_per_case]
    m = m.merge(f, on="__id__", how="left")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Run training-free hard veto controller on baseline/VGA outputs.")
    ap.add_argument("--per_case_csv", type=str, required=True)
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--id_col_per_case", type=str, default="id")
    ap.add_argument("--id_col_feature", type=str, default="id")
    ap.add_argument("--gt_col", type=str, default="gt")
    ap.add_argument("--baseline_col", type=str, default="pred_baseline")
    ap.add_argument("--vga_col", type=str, default="pred_vga")
    ap.add_argument("--case_col", type=str, default="case_type")
    ap.add_argument("--c_col", type=str, default="faithful_minus_global_attn")
    ap.add_argument("--e_col", type=str, default="guidance_mismatch_score")
    ap.add_argument("--fallback_when_missing_feature", type=str, default="vga", choices=["baseline", "vga"])
    ap.add_argument("--calib_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_d1", type=float, default=1.0)
    ap.add_argument("--max_d1_wrong_rate", type=float, default=0.35)
    ap.add_argument("--q_grid", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    ap.add_argument("--tau_c", type=float, default=None)
    ap.add_argument("--tau_e", type=float, default=None)
    ap.add_argument("--use_c", type=int, default=1, choices=[0, 1])
    ap.add_argument("--use_e", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    use_c = bool(args.use_c)
    use_e = bool(args.use_e)
    if not use_c and not use_e:
        raise ValueError("At least one of --use_c or --use_e must be 1.")

    os.makedirs(args.out_dir, exist_ok=True)
    per_case = pd.read_csv(args.per_case_csv)
    feats = pd.read_csv(args.features_csv)

    df = merge_features(
        per_case,
        feats,
        args.id_col_per_case,
        args.id_col_feature,
        args.c_col,
        args.e_col,
    )

    required = [args.gt_col, args.baseline_col, args.vga_col, args.case_col, "__C__", "__E__"]
    for c in [args.gt_col, args.baseline_col, args.vga_col, args.case_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column in per_case csv: {c}")

    # drop rows with missing base/vga/gt labels
    df = df.dropna(subset=[args.gt_col, args.baseline_col, args.vga_col]).copy()
    df["__gt__"] = df[args.gt_col].map(to_label)
    df["__base__"] = df[args.baseline_col].map(to_label)
    df["__vga__"] = df[args.vga_col].map(to_label)

    missing_feat = df["__C__"].isna() | df["__E__"].isna()
    n_missing_feat = int(missing_feat.sum())

    df_feat = df[~missing_feat].copy()
    if len(df_feat) == 0:
        raise RuntimeError("No rows with both C/E features.")

    # train/test split for threshold calibration
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df_feat))
    rng.shuffle(idx)
    n_cal = max(1, int(round(len(df_feat) * args.calib_ratio)))
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:] if n_cal < len(df_feat) else idx[:0]
    df_cal = df_feat.iloc[cal_idx].copy()
    df_test = df_feat.iloc[test_idx].copy()

    # thresholds
    if args.tau_c is not None and args.tau_e is not None:
        best_tau = TauResult(
            tau_c=float(args.tau_c),
            tau_e=float(args.tau_e),
            objective=float("nan"),
            n_cal=len(df_cal),
            d1_wrong_veto=-1,
            d2_correct_veto=-1,
            veto_rate=float("nan"),
        )
        tau_mode = "manual"
    else:
        q_grid = [float(x.strip()) for x in str(args.q_grid).split(",") if x.strip()]
        best_tau = calibrate_thresholds(
            df_cal=df_cal,
            c_col="__C__",
            e_col="__E__",
            case_col=args.case_col,
            q_grid=q_grid,
            lambda_d1=args.lambda_d1,
            max_d1_wrong_rate=args.max_d1_wrong_rate,
            use_c=use_c,
            use_e=use_e,
        )
        tau_mode = "calibrated"

    # apply veto to all rows
    df["veto"] = False
    has_feat = ~(missing_feat)
    df.loc[has_feat, "veto"] = compute_veto_mask(
        df.loc[has_feat, "__C__"].to_numpy(),
        df.loc[has_feat, "__E__"].to_numpy(),
        best_tau.tau_c,
        best_tau.tau_e,
        use_c=use_c,
        use_e=use_e,
    )
    if args.fallback_when_missing_feature == "baseline":
        df.loc[missing_feat, "veto"] = True
    else:
        df.loc[missing_feat, "veto"] = False

    df["pred_controller"] = np.where(df["veto"], df["__base__"], df["__vga__"])
    df["route"] = np.where(df["veto"], "baseline", "vga")

    # metrics
    m_base = metrics(df["__gt__"], df["__base__"])
    m_vga = metrics(df["__gt__"], df["__vga__"])
    m_ctl = metrics(df["__gt__"], df["pred_controller"])

    # split metrics
    def split_eval(sdf: pd.DataFrame) -> Dict[str, float]:
        if len(sdf) == 0:
            return {"n": 0}
        return {
            **metrics(sdf["__gt__"], sdf["pred_controller"]),
            "veto_rate": float(sdf["veto"].mean()),
            "base_acc": float(metrics(sdf["__gt__"], sdf["__base__"])["acc"]),
            "vga_acc": float(metrics(sdf["__gt__"], sdf["__vga__"])["acc"]),
        }

    cal_ids = set(df_cal["__id__"].tolist())
    test_ids = set(df_test["__id__"].tolist())
    eval_cal = split_eval(df[df["__id__"].isin(cal_ids)].copy())
    eval_test = split_eval(df[df["__id__"].isin(test_ids)].copy())

    # D1 / D2 stats
    case = df[args.case_col].astype(str)
    d1 = (case == "vga_improvement")
    d2 = (case == "vga_regression")
    d1_wrong_veto = int((d1 & df["veto"]).sum())
    d2_correct_veto = int((d2 & df["veto"]).sum())
    d1_total = int(d1.sum())
    d2_total = int(d2.sum())

    summary = {
        "inputs": {
            "per_case_csv": os.path.abspath(args.per_case_csv),
            "features_csv": os.path.abspath(args.features_csv),
            "c_col": args.c_col,
            "e_col": args.e_col,
            "use_c": use_c,
            "use_e": use_e,
            "fallback_when_missing_feature": args.fallback_when_missing_feature,
            "calib_ratio": args.calib_ratio,
            "seed": args.seed,
            "lambda_d1": args.lambda_d1,
            "max_d1_wrong_rate": args.max_d1_wrong_rate,
            "q_grid": args.q_grid,
            "tau_mode": tau_mode,
        },
        "thresholds": {
            "tau_c": best_tau.tau_c,
            "tau_e": best_tau.tau_e,
            "calib_objective": best_tau.objective,
            "calib_veto_rate": best_tau.veto_rate,
            "calib_d1_wrong_veto": best_tau.d1_wrong_veto,
            "calib_d2_correct_veto": best_tau.d2_correct_veto,
        },
        "counts": {
            "n_total": int(len(df)),
            "n_with_features": int((~missing_feat).sum()),
            "n_missing_features": n_missing_feat,
            "veto_count": int(df["veto"].sum()),
            "veto_rate": float(df["veto"].mean()),
            "D1_total": d1_total,
            "D2_total": d2_total,
            "D1_wrong_veto": d1_wrong_veto,
            "D2_correct_veto": d2_correct_veto,
            "D1_wrong_veto_rate": (d1_wrong_veto / d1_total) if d1_total else 0.0,
            "D2_correct_veto_rate": (d2_correct_veto / d2_total) if d2_total else 0.0,
        },
        "metrics": {
            "baseline": m_base,
            "vga": m_vga,
            "controller": m_ctl,
            "controller_delta_vs_baseline": {
                "delta_acc": m_ctl["acc"] - m_base["acc"],
                "delta_f1": m_ctl["f1"] - m_base["f1"],
                **change_stats(df["__gt__"], df["__base__"], df["pred_controller"]),
            },
            "controller_delta_vs_vga": {
                "delta_acc": m_ctl["acc"] - m_vga["acc"],
                "delta_f1": m_ctl["f1"] - m_vga["f1"],
                **change_stats(df["__gt__"], df["__vga__"], df["pred_controller"]),
            },
            "cal_split": eval_cal,
            "test_split": eval_test,
        },
    }

    per_id_cols = [
        args.id_col_per_case, args.gt_col, args.baseline_col, args.vga_col, args.case_col,
        "__C__", "__E__", "veto", "route", "pred_controller",
    ]
    per_id_cols = [c for c in per_id_cols if c in df.columns]
    per_id_csv = os.path.join(args.out_dir, "per_id_controller.csv")
    df[per_id_cols].to_csv(per_id_csv, index=False)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", per_id_csv)
    print("[saved]", summary_path)
    print(
        "[summary]",
        json.dumps(
            {
                "tau_c": best_tau.tau_c,
                "tau_e": best_tau.tau_e,
                "base_acc": m_base["acc"],
                "vga_acc": m_vga["acc"],
                "ctl_acc": m_ctl["acc"],
                "delta_vs_vga": m_ctl["acc"] - m_vga["acc"],
                "veto_rate": float(df["veto"].mean()),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
