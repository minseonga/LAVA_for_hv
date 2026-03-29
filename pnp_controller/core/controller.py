from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .schemas import CalibrationResult, HardVetoConfig, OfflineTableSchema


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


def change_stats(y: pd.Series, old: pd.Series, new: pd.Series) -> Dict[str, int]:
    y = y.map(to_label)
    old = old.map(to_label)
    new = new.map(to_label)
    changed = old != new
    gain = ((old != y) & (new == y) & changed).sum()
    harm = ((old == y) & (new != y) & changed).sum()
    return {
        "changed_pred": int(changed.sum()),
        "gain": int(gain),
        "harm": int(harm),
        "net_gain": int(gain - harm),
    }


def compute_veto_mask(frg_vals: np.ndarray, gmi_vals: np.ndarray, tau_frg: float, tau_gmi: float) -> np.ndarray:
    return (frg_vals >= tau_frg) | (gmi_vals >= tau_gmi)


def calibrate_thresholds(
    df_cal: pd.DataFrame,
    schema: OfflineTableSchema,
    controller_cfg: HardVetoConfig,
) -> CalibrationResult:
    frg_arr = df_cal["__FRG__"].to_numpy()
    gmi_arr = df_cal["__GMI__"].to_numpy()
    improvement = (df_cal[schema.case_col] == controller_cfg.improvement_case_value).to_numpy()
    regression = (df_cal[schema.case_col] == controller_cfg.regression_case_value).to_numpy()

    cand_frg = sorted(
        set(float(np.quantile(frg_arr, q)) for q in controller_cfg.calibration.q_grid)
    )
    cand_gmi = sorted(
        set(float(np.quantile(gmi_arr, q)) for q in controller_cfg.calibration.q_grid)
    )

    best: Optional[CalibrationResult] = None
    for tau_frg in cand_frg:
        for tau_gmi in cand_gmi:
            veto = compute_veto_mask(frg_arr, gmi_arr, tau_frg, tau_gmi)
            wrong_veto = int((veto & improvement).sum())
            correct_veto = int((veto & regression).sum())
            improvement_total = int(improvement.sum())
            wrong_rate = (wrong_veto / improvement_total) if improvement_total else 0.0
            if wrong_rate > controller_cfg.calibration.max_wrong_veto_rate:
                continue
            objective = float(correct_veto - controller_cfg.calibration.lambda_improvement * wrong_veto)
            cur = CalibrationResult(
                tau_frg=tau_frg,
                tau_gmi=tau_gmi,
                objective=objective,
                n_cal=int(len(df_cal)),
                wrong_veto_count=wrong_veto,
                correct_veto_count=correct_veto,
                veto_rate=float(veto.mean()) if len(veto) else 0.0,
                mode="calibrated",
            )
            if best is None:
                best = cur
                continue
            if (cur.objective > best.objective) or (
                np.isclose(cur.objective, best.objective) and (
                    cur.wrong_veto_count < best.wrong_veto_count or (
                        cur.wrong_veto_count == best.wrong_veto_count and cur.veto_rate < best.veto_rate
                    )
                )
            ):
                best = cur

    if best is not None:
        return best

    fallback: Optional[CalibrationResult] = None
    for tau_frg in cand_frg:
        for tau_gmi in cand_gmi:
            veto = compute_veto_mask(frg_arr, gmi_arr, tau_frg, tau_gmi)
            wrong_veto = int((veto & improvement).sum())
            correct_veto = int((veto & regression).sum())
            objective = float(correct_veto - controller_cfg.calibration.lambda_improvement * wrong_veto)
            cur = CalibrationResult(
                tau_frg=tau_frg,
                tau_gmi=tau_gmi,
                objective=objective,
                n_cal=int(len(df_cal)),
                wrong_veto_count=wrong_veto,
                correct_veto_count=correct_veto,
                veto_rate=float(veto.mean()) if len(veto) else 0.0,
                mode="fallback_unconstrained",
            )
            if fallback is None or cur.objective > fallback.objective:
                fallback = cur
    assert fallback is not None
    return fallback


def run_offline_hard_veto(
    merged_df: pd.DataFrame,
    schema: OfflineTableSchema,
    controller_cfg: HardVetoConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    for col in (schema.gt_col, schema.baseline_col, schema.method_col, schema.case_col):
        if col not in merged_df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = merged_df.copy()
    df = df.dropna(subset=[schema.gt_col, schema.baseline_col, schema.method_col]).copy()
    df["__gt__"] = df[schema.gt_col].map(to_label)
    df["__base__"] = df[schema.baseline_col].map(to_label)
    df["__method__"] = df[schema.method_col].map(to_label)

    missing_feat = df["__FRG__"].isna() | df["__GMI__"].isna()
    n_missing_feat = int(missing_feat.sum())
    df_feat = df[~missing_feat].copy()
    if len(df_feat) == 0:
        raise RuntimeError("No rows with both FRG/GMI features.")

    rng = np.random.default_rng(controller_cfg.calibration.seed)
    idx = np.arange(len(df_feat))
    rng.shuffle(idx)
    n_cal = max(1, int(round(len(df_feat) * controller_cfg.calibration.calib_ratio)))
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:] if n_cal < len(df_feat) else idx[:0]
    df_cal = df_feat.iloc[cal_idx].copy()
    df_test = df_feat.iloc[test_idx].copy()

    if controller_cfg.tau_frg is not None and controller_cfg.tau_gmi is not None:
        best_tau = CalibrationResult(
            tau_frg=float(controller_cfg.tau_frg),
            tau_gmi=float(controller_cfg.tau_gmi),
            objective=float("nan"),
            n_cal=int(len(df_cal)),
            wrong_veto_count=-1,
            correct_veto_count=-1,
            veto_rate=float("nan"),
            mode="manual",
        )
    else:
        best_tau = calibrate_thresholds(df_cal=df_cal, schema=schema, controller_cfg=controller_cfg)

    df["veto"] = False
    has_feat = ~missing_feat
    df.loc[has_feat, "veto"] = compute_veto_mask(
        df.loc[has_feat, "__FRG__"].to_numpy(),
        df.loc[has_feat, "__GMI__"].to_numpy(),
        best_tau.tau_frg,
        best_tau.tau_gmi,
    )
    if controller_cfg.fallback_when_missing_feature == "baseline":
        df.loc[missing_feat, "veto"] = True
    else:
        df.loc[missing_feat, "veto"] = False

    df["pred_controller"] = np.where(df["veto"], df["__base__"], df["__method__"])
    df["route"] = np.where(df["veto"], "baseline", "method")

    m_base = metrics(df["__gt__"], df["__base__"])
    m_method = metrics(df["__gt__"], df["__method__"])
    m_ctl = metrics(df["__gt__"], df["pred_controller"])

    def split_eval(sdf: pd.DataFrame) -> Dict[str, float]:
        if len(sdf) == 0:
            return {"n": 0}
        return {
            **metrics(sdf["__gt__"], sdf["pred_controller"]),
            "veto_rate": float(sdf["veto"].mean()),
            "base_acc": float(metrics(sdf["__gt__"], sdf["__base__"])["acc"]),
            "method_acc": float(metrics(sdf["__gt__"], sdf["__method__"])["acc"]),
        }

    cal_ids = set(df_cal["__id__"].tolist())
    test_ids = set(df_test["__id__"].tolist())
    eval_cal = split_eval(df[df["__id__"].isin(cal_ids)].copy())
    eval_test = split_eval(df[df["__id__"].isin(test_ids)].copy())

    case = df[schema.case_col].astype(str)
    improvement = case == controller_cfg.improvement_case_value
    regression = case == controller_cfg.regression_case_value
    wrong_veto = int((improvement & df["veto"]).sum())
    correct_veto = int((regression & df["veto"]).sum())
    improvement_total = int(improvement.sum())
    regression_total = int(regression.sum())

    summary: Dict[str, object] = {
        "inputs": {
            "frg_col": controller_cfg.frg_col,
            "gmi_col": controller_cfg.gmi_col,
            "fallback_when_missing_feature": controller_cfg.fallback_when_missing_feature,
            "calib_ratio": controller_cfg.calibration.calib_ratio,
            "seed": controller_cfg.calibration.seed,
            "lambda_improvement": controller_cfg.calibration.lambda_improvement,
            "max_wrong_veto_rate": controller_cfg.calibration.max_wrong_veto_rate,
            "q_grid": controller_cfg.calibration.q_grid,
            "threshold_mode": best_tau.mode,
            "improvement_case_value": controller_cfg.improvement_case_value,
            "regression_case_value": controller_cfg.regression_case_value,
        },
        "thresholds": {
            "tau_frg": best_tau.tau_frg,
            "tau_gmi": best_tau.tau_gmi,
            "calib_objective": best_tau.objective,
            "calib_veto_rate": best_tau.veto_rate,
            "calib_wrong_veto_count": best_tau.wrong_veto_count,
            "calib_correct_veto_count": best_tau.correct_veto_count,
        },
        "counts": {
            "n_total": int(len(df)),
            "n_with_features": int((~missing_feat).sum()),
            "n_missing_features": n_missing_feat,
            "veto_count": int(df["veto"].sum()),
            "veto_rate": float(df["veto"].mean()),
            "improvement_total": improvement_total,
            "regression_total": regression_total,
            "wrong_veto_count": wrong_veto,
            "correct_veto_count": correct_veto,
            "wrong_veto_rate": (wrong_veto / improvement_total) if improvement_total else 0.0,
            "correct_veto_rate": (correct_veto / regression_total) if regression_total else 0.0,
        },
        "metrics": {
            "baseline": m_base,
            "method": m_method,
            "controller": m_ctl,
            "controller_delta_vs_baseline": {
                "delta_acc": m_ctl["acc"] - m_base["acc"],
                "delta_f1": m_ctl["f1"] - m_base["f1"],
                **change_stats(df["__gt__"], df["__base__"], df["pred_controller"]),
            },
            "controller_delta_vs_method": {
                "delta_acc": m_ctl["acc"] - m_method["acc"],
                "delta_f1": m_ctl["f1"] - m_method["f1"],
                **change_stats(df["__gt__"], df["__method__"], df["pred_controller"]),
            },
            "cal_split": eval_cal,
            "test_split": eval_test,
        },
    }
    return df, summary
