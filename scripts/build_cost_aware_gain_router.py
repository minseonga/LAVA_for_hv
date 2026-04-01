#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor


BUDGETS_DEFAULT = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
REQUIRED_TAXONOMY_COLUMNS = [
    "id",
    "gt",
    "pred_baseline",
    "pred_vga",
    "baseline_ok",
    "vga_ok",
    "case_type",
]


def parse_anchor_yes(text: Any) -> int:
    s = str(text or "").strip().lower()
    return 1 if s == "yes" else 0


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def compute_metrics(pred: pd.Series, gt: pd.Series) -> Dict[str, Any]:
    pred = pred.astype(str).str.strip().str.lower()
    gt = gt.astype(str).str.strip().str.lower()
    tp = int(((pred == "yes") & (gt == "yes")).sum())
    fp = int(((pred == "yes") & (gt == "no")).sum())
    tn = int(((pred == "no") & (gt == "no")).sum())
    fn = int(((pred == "no") & (gt == "yes")).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": int(len(pred)),
        "acc": float((pred == gt).mean()),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "yes_ratio": float((pred == "yes").mean()),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def evaluate_topk_policy(
    df: pd.DataFrame,
    score_col: str,
    budgets: Sequence[float],
    method_col: str,
    baseline_col: str,
    gt_col: str,
    utility_col: str,
    id_col: str,
    frg_col: str,
) -> List[Dict[str, Any]]:
    sort_df = df[[id_col, score_col, frg_col, method_col, baseline_col, gt_col, utility_col]].copy()
    sort_df = sort_df.sort_values([score_col, frg_col, id_col], ascending=[False, False, True]).reset_index(drop=True)
    n = len(sort_df)
    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        k = int(round(float(budget) * n))
        selected = pd.Series(False, index=sort_df.index)
        if k > 0:
            selected.iloc[:k] = True
        pred = sort_df[baseline_col].where(~selected, sort_df[method_col])
        metrics = compute_metrics(pred, sort_df[gt_col])
        selected_util = sort_df.loc[selected, utility_col]
        rows.append(
            {
                "model": score_col,
                "budget": float(budget),
                "method_rate": float(selected.mean()),
                "avg_actual_utility_selected": float(selected_util.mean()) if len(selected_util) else 0.0,
                "selected_gain_rate": float(((selected) & (sort_df[utility_col] == 1)).mean()),
                "selected_harm_rate": float(((selected) & (sort_df[utility_col] == -1)).mean()),
                **metrics,
            }
        )
    return rows


def fit_oof_scores(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    utility_col: str,
    id_col: str,
    seed: int,
    n_splits: int,
) -> pd.DataFrame:
    out = pd.DataFrame({id_col: df[id_col], "utility_true": df[utility_col]})
    out["frg_rank"] = -df["frg_off"].astype(float)

    X = df[list(feature_cols)].astype(float)
    y = df[utility_col].astype(int)
    y_strat = y.astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    tree_oof = pd.Series(index=df.index, dtype=float)
    hgb_oof = pd.Series(index=df.index, dtype=float)

    for train_idx, test_idx in skf.split(X, y_strat):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=seed)
        tree.fit(X_train, y_train)
        tree_oof.iloc[test_idx] = tree.predict(X_test)

        hgb = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=30,
            random_state=seed,
        )
        hgb.fit(X_train, y_train)
        hgb_oof.iloc[test_idx] = hgb.predict(X_test)

    out["tree_utility"] = tree_oof
    out["hgb_utility"] = hgb_oof
    return out


def fit_full_model_scores(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    utility_col: str,
    seed: int,
) -> pd.DataFrame:
    X = df[list(feature_cols)].astype(float)
    y = df[utility_col].astype(int)

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=seed)
    tree.fit(X, y)

    hgb = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=30,
        random_state=seed,
    )
    hgb.fit(X, y)

    return pd.DataFrame(
        {
            "tree_utility_fullfit": tree.predict(X),
            "hgb_utility_fullfit": hgb.predict(X),
        }
    )


def fit_full_models(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    utility_col: str,
    seed: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    X = df[list(feature_cols)].astype(float)
    y = df[utility_col].astype(int)

    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=seed)
    tree.fit(X, y)

    hgb = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=30,
        random_state=seed,
    )
    hgb.fit(X, y)

    scores = pd.DataFrame(
        {
            "tree_utility_fullfit": tree.predict(X),
            "hgb_utility_fullfit": hgb.predict(X),
        }
    )
    return scores, {"tree_utility_fullfit": tree, "hgb_utility_fullfit": hgb}


def save_router_artifact(
    out_dir: str,
    model: Any,
    metadata: Dict[str, Any],
) -> Dict[str, str]:
    model_path = os.path.join(out_dir, "router_model.pkl")
    metadata_path = os.path.join(out_dir, "router_metadata.json")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return {"model_pkl": model_path, "metadata_json": metadata_path}


def deployment_score_bands(
    df: pd.DataFrame,
    score_col: str,
    utility_col: str,
    n_bands: int,
    id_col: str,
    frg_col: str,
    budget: float,
) -> tuple[list[Dict[str, Any]], float]:
    tmp = df[[id_col, score_col, utility_col, frg_col]].copy()
    tmp = tmp.sort_values([score_col, frg_col, id_col], ascending=[False, False, True]).reset_index(drop=True)
    n = len(tmp)
    k = int(round(float(budget) * n))
    cutoff = float(tmp.iloc[max(k - 1, 0)][score_col]) if k > 0 else float("inf")
    rows: List[Dict[str, Any]] = []
    for i in range(n_bands):
        lo = int(round(i * n / n_bands))
        hi = int(round((i + 1) * n / n_bands))
        band = tmp.iloc[lo:hi]
        if band.empty:
            continue
        band_name = f"{int(100*i/n_bands):02d}-{int(100*(i+1)/n_bands):02d}pct"
        rows.append(
            {
                "band": band_name,
                "count": int(len(band)),
                "share": float(len(band) / n),
                "score_min": float(band[score_col].min()),
                "score_max": float(band[score_col].max()),
                "mean_score": float(band[score_col].mean()),
                "gain_rate": float((band[utility_col] == 1).mean()),
                "harm_rate": float((band[utility_col] == -1).mean()),
                "neutral_rate": float((band[utility_col] == 0).mean()),
                "deployment_action": "method" if hi <= k else "baseline",
            }
        )
    return rows, cutoff


def plot_budget_and_pareto(
    budget_rows: List[Dict[str, Any]],
    reference_rows: List[Dict[str, Any]],
    out_budget_png: str,
    out_pareto_png: str,
    title_prefix: str,
) -> None:
    df = pd.DataFrame(budget_rows)
    ref = pd.DataFrame(reference_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in df.groupby("model"):
        ax.plot(sub["budget"], sub["acc"], marker="o", label=model)
    for _, row in ref.iterrows():
        ax.axhline(float(row["acc"]), linestyle="--", linewidth=1, alpha=0.6)
        ax.text(0.505, float(row["acc"]), row["policy"], va="center", fontsize=8)
    ax.set_xlabel("Method Budget")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title_prefix}: Budget Sweep")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_budget_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in df.groupby("model"):
        ax.plot(sub["method_rate"], sub["acc"], marker="o", label=model)
    ax.scatter(ref["method_rate"], ref["acc"], s=40, c="black", marker="x", label="reference_policies")
    for _, row in ref.iterrows():
        ax.text(float(row["method_rate"]), float(row["acc"]), row["policy"], fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Method Rate / Latency Proxy")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title_prefix}: Accuracy-Cost Trade-off")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pareto_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cost-aware gain router artifacts from cheap probe logs and branch correctness tables.")
    ap.add_argument("--probe_log_csv", type=str, required=True)
    ap.add_argument("--taxonomy_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--backend_name", type=str, default="vga")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--deployment_budget", type=float, default=0.30)
    ap.add_argument("--feature_variant", type=str, default="full", choices=["full", "no_abs"])
    ap.add_argument("--save_router_artifact", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    probe = pd.read_csv(args.probe_log_csv)
    tax = pd.read_csv(args.taxonomy_csv)
    probe["id"] = probe["id"].astype(str)

    missing_tax_cols = [c for c in REQUIRED_TAXONOMY_COLUMNS if c not in tax.columns]
    if missing_tax_cols:
        raise ValueError(
            "taxonomy_csv is not a VGA branch-correctness table. "
            f"Missing required columns: {missing_tax_cols}. "
            f"Present columns: {list(tax.columns)}. "
            "Expected a per_case_compare.csv produced by scripts/build_vga_failure_taxonomy.py."
        )
    tax["id"] = tax["id"].astype(str)

    df = probe[["id", "frg", "g_top5_mass", "probe_anchor"]].merge(
        tax[REQUIRED_TAXONOMY_COLUMNS],
        on="id",
        how="inner",
    )

    for c in ["baseline_ok", "vga_ok"]:
        df[c] = df[c].astype(int)
    for c in ["gt", "pred_baseline", "pred_vga", "case_type"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    df["utility_true"] = df["vga_ok"] - df["baseline_ok"]
    df["frg_off"] = df["frg"].astype(float)
    df["g_top5_mass"] = df["g_top5_mass"].astype(float)
    df["probe_anchor_yes"] = df["probe_anchor"].map(parse_anchor_yes).astype(int)
    df["abs_frg_to_tau"] = (df["frg_off"] - float(args.tau)).abs()
    df["frg_x_mass"] = df["frg_off"] * df["g_top5_mass"]

    feature_cols = ["frg_off", "g_top5_mass", "probe_anchor_yes", "frg_x_mass"]
    if args.feature_variant == "full":
        feature_cols.insert(3, "abs_frg_to_tau")

    oof = fit_oof_scores(
        df=df,
        feature_cols=feature_cols,
        utility_col="utility_true",
        id_col="id",
        seed=int(args.seed),
        n_splits=int(args.n_splits),
    )
    fullfit_scores, fullfit_models = fit_full_models(
        df=df,
        feature_cols=feature_cols,
        utility_col="utility_true",
        seed=int(args.seed),
    )

    joined = pd.concat([df.reset_index(drop=True), oof[["frg_rank", "tree_utility", "hgb_utility"]], fullfit_scores], axis=1)

    reference_rows = []
    for policy, pred in [
        ("baseline_only", joined["pred_baseline"]),
        (f"{args.backend_name}_only", joined["pred_vga"]),
        ("strict_gain_oracle", joined["pred_vga"].where(joined["utility_true"] == 1, joined["pred_baseline"])),
    ]:
        metrics = compute_metrics(pred, joined["gt"])
        if policy == "baseline_only":
            method_rate = 0.0
        elif policy == f"{args.backend_name}_only":
            method_rate = 1.0
        else:
            method_rate = float((joined["utility_true"] == 1).mean())
        reference_rows.append(
            {
                "policy": policy,
                "method_rate": method_rate,
                **metrics,
            }
        )

    budget_rows: List[Dict[str, Any]] = []
    for score_col in ["frg_rank", "tree_utility", "hgb_utility"]:
        rows = evaluate_topk_policy(
            df=joined,
            score_col=score_col,
            budgets=BUDGETS_DEFAULT,
            method_col="pred_vga",
            baseline_col="pred_baseline",
            gt_col="gt",
            utility_col="utility_true",
            id_col="id",
            frg_col="frg_off",
        )
        budget_rows.extend(rows)

    bands, cutoff = deployment_score_bands(
        df=joined,
        score_col="hgb_utility_fullfit",
        utility_col="utility_true",
        n_bands=10,
        id_col="id",
        frg_col="frg_off",
        budget=float(args.deployment_budget),
    )

    write_csv(os.path.join(args.out_dir, "reference_policies.csv"), reference_rows)
    write_csv(os.path.join(args.out_dir, "budget_sweep.csv"), budget_rows)
    oof[["id", "utility_true", "frg_rank", "tree_utility", "hgb_utility"]].to_csv(
        os.path.join(args.out_dir, "oof_scores.csv"), index=False
    )
    write_csv(os.path.join(args.out_dir, f"deployment_score_bands_{int(round(args.deployment_budget*100))}_budget.csv"), bands)
    joined[["id", "frg_off", "g_top5_mass", "probe_anchor_yes", "abs_frg_to_tau", "frg_x_mass", "utility_true"]].to_csv(
        os.path.join(args.out_dir, "training_table.csv"), index=False
    )

    budget_df = pd.DataFrame(budget_rows)
    hgb_budget_row = budget_df[(budget_df["model"] == "hgb_utility") & (budget_df["budget"] == float(args.deployment_budget))]
    hgb_budget_summary = hgb_budget_row.iloc[0].to_dict() if not hgb_budget_row.empty else {}

    summary = {
        "backend_name": args.backend_name,
        "feature_variant": args.feature_variant,
        "tau": float(args.tau),
        "feature_cols": list(feature_cols),
        "n_samples": int(len(joined)),
        "utility_counts": {
            "gain": int((joined["utility_true"] == 1).sum()),
            "harm": int((joined["utility_true"] == -1).sum()),
            "neutral": int((joined["utility_true"] == 0).sum()),
        },
        "deployment_budget": float(args.deployment_budget),
        "deployment_cutoff_hgb_fullfit": float(cutoff),
        "hgb_budget_row": hgb_budget_summary,
        "reference_policies": reference_rows,
    }

    artifact_paths: Dict[str, str] = {}
    if bool(args.save_router_artifact):
        artifact_metadata = {
            "router_type": "hgb_utility_fullfit",
            "backend_name": args.backend_name,
            "feature_variant": args.feature_variant,
            "feature_cols": list(feature_cols),
            "tau": float(args.tau),
            "deployment_budget": float(args.deployment_budget),
            "deployment_cutoff": float(cutoff),
            "n_samples": int(len(joined)),
            "utility_counts": summary["utility_counts"],
            "score_policy": "method if utility_score >= deployment_cutoff else baseline",
            "tie_break_note": "Threshold policy may yield method rate slightly above or below exact top-k because identical scores share the cutoff.",
            "training_sources": {
                "probe_log_csv": os.path.abspath(args.probe_log_csv),
                "taxonomy_csv": os.path.abspath(args.taxonomy_csv),
            },
        }
        artifact_paths = save_router_artifact(
            out_dir=args.out_dir,
            model=fullfit_models["hgb_utility_fullfit"],
            metadata=artifact_metadata,
        )
        summary["router_artifact"] = artifact_paths

    with open(os.path.join(args.out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "probe_log_csv": os.path.abspath(args.probe_log_csv),
                "taxonomy_csv": os.path.abspath(args.taxonomy_csv),
                "tau": float(args.tau),
                "seed": int(args.seed),
                "n_splits": int(args.n_splits),
                "deployment_budget": float(args.deployment_budget),
                "backend_name": args.backend_name,
                "feature_variant": args.feature_variant,
                "feature_cols": list(feature_cols),
                "n_samples": int(len(joined)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_budget_and_pareto(
        budget_rows=budget_rows,
        reference_rows=reference_rows,
        out_budget_png=os.path.join(args.out_dir, "budget_sweep_plot.png"),
        out_pareto_png=os.path.join(args.out_dir, "accuracy_cost_proxy_pareto.png"),
        title_prefix=f"{args.backend_name.upper()} / {args.feature_variant}",
    )

    print("[saved]", os.path.join(args.out_dir, "summary.json"))
    print("[saved]", os.path.join(args.out_dir, "reference_policies.csv"))
    print("[saved]", os.path.join(args.out_dir, "budget_sweep.csv"))
    print("[saved]", os.path.join(args.out_dir, "oof_scores.csv"))
    if artifact_paths:
        print("[saved]", artifact_paths["model_pkl"])
        print("[saved]", artifact_paths["metadata_json"])
    print(
        "[summary]",
        json.dumps(
            {
                "backend": args.backend_name,
                "feature_variant": args.feature_variant,
                "hgb_budget_acc": hgb_budget_summary.get("acc", None),
                "hgb_budget_method_rate": hgb_budget_summary.get("method_rate", None),
                "deployment_cutoff_hgb_fullfit": cutoff,
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
