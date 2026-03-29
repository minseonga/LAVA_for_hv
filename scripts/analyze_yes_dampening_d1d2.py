#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import pandas as pd


def auc_rank(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, 0
    r = pd.Series(s).rank(method="average").to_numpy()
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc), int(len(y))


def ks_stat(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    a = np.sort(s[y == 1])
    b = np.sort(s[y == 0])
    if len(a) == 0 or len(b) == 0:
        return np.nan
    grid = np.unique(np.concatenate([a, b]))
    cdfa = np.searchsorted(a, grid, side="right") / len(a)
    cdfb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdfa - cdfb)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze yes->no dampening split: D1 beneficial vs D2 harmful")
    ap.add_argument("--per_case_csv", type=str, required=True, help="per_case_compare.csv")
    ap.add_argument("--features_csv", type=str, default="", help="optional features_unified_table.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    c = pd.read_csv(args.per_case_csv)

    c["pred_baseline"] = c["pred_baseline"].astype(str).str.lower()
    c["pred_vga"] = c["pred_vga"].astype(str).str.lower()
    c["gt"] = c["gt"].astype(str).str.lower()

    c["pair"] = c["pred_baseline"] + "->" + c["pred_vga"]
    trans = c.groupby(["gt", "pair"]).size().reset_index(name="n")
    trans.to_csv(os.path.join(args.out_dir, "transition_by_gt.csv"), index=False)

    y2n = c[(c["pred_baseline"] == "yes") & (c["pred_vga"] == "no")].copy()
    y2n["d_regime"] = np.where(y2n["gt"] == "no", "D1_beneficial_yes2no", "D2_harmful_yes2no")
    y2n.to_csv(os.path.join(args.out_dir, "yes2no_cases.csv"), index=False)

    cnt = y2n["d_regime"].value_counts().rename_axis("regime").reset_index(name="n")
    if len(cnt) > 0:
        cnt["ratio_within_yes2no"] = cnt["n"] / cnt["n"].sum()
    else:
        cnt["ratio_within_yes2no"] = []
    cnt.to_csv(os.path.join(args.out_dir, "d1_d2_counts_within_yes2no.csv"), index=False)

    summary = {
        "counts": {
            "n_total": int(len(c)),
            "n_yes2no_changed": int(len(y2n)),
            "n_D1_beneficial_yes2no": int((y2n["d_regime"] == "D1_beneficial_yes2no").sum()),
            "n_D2_harmful_yes2no": int((y2n["d_regime"] == "D2_harmful_yes2no").sum()),
        },
        "outputs": {
            "transition_by_gt_csv": os.path.join(args.out_dir, "transition_by_gt.csv"),
            "yes2no_cases_csv": os.path.join(args.out_dir, "yes2no_cases.csv"),
            "d1_d2_counts_csv": os.path.join(args.out_dir, "d1_d2_counts_within_yes2no.csv"),
        },
    }

    if str(args.features_csv).strip() != "":
        f = pd.read_csv(args.features_csv)
        f["id"] = f["id"].astype(str)
        z = y2n[["id", "d_regime", "gt", "category", "question"]].copy()
        z["id"] = z["id"].astype(str)
        m = z.merge(f, on="id", how="left")

        if "harmful_minus_faithful" in m.columns:
            m["Y_overassert_proxy"] = m["harmful_minus_faithful"]

        cand = [
            "obj_token_prob_lse",
            "faithful_minus_global_attn",
            "guidance_mismatch_score",
            "Y_overassert_proxy",
            "faithful_on_nonG_mass",
            "harmful_inside_G",
            "supportive_outside_G",
            "harmful_on_G_mass",
            "faithful_head_attn_mean",
            "harmful_head_attn_mean",
            "late_uplift",
            "late_topk_persistence",
        ]
        cand = [c for c in cand if c in m.columns]

        yy = (m["d_regime"] == "D2_harmful_yes2no").astype(int).to_numpy()
        rows = []
        for col in cand:
            s = m[col].to_numpy(dtype=float)
            auc, n = auc_rank(yy, s)
            ks = ks_stat(yy, s)
            d2 = s[yy == 1]
            d1 = s[yy == 0]
            rows.append(
                {
                    "feature": col,
                    "n": n,
                    "mean_D1": float(np.nanmean(d1)) if np.isfinite(d1).any() else np.nan,
                    "mean_D2": float(np.nanmean(d2)) if np.isfinite(d2).any() else np.nan,
                    "delta_D2_minus_D1": float(np.nanmean(d2) - np.nanmean(d1)) if (np.isfinite(d1).any() and np.isfinite(d2).any()) else np.nan,
                    "auc_D2_high": auc,
                    "auc_best_dir": float(max(auc, 1 - auc)) if np.isfinite(auc) else np.nan,
                    "direction": ("higher_in_D2" if auc >= 0.5 else "lower_in_D2") if np.isfinite(auc) else "na",
                    "ks": ks,
                }
            )

        sep = pd.DataFrame(rows).sort_values(["auc_best_dir", "ks"], ascending=False)
        gm = m.groupby("d_regime")[cand].mean(numeric_only=True).reset_index()

        m.to_csv(os.path.join(args.out_dir, "d1_d2_joined_features.csv"), index=False)
        sep.to_csv(os.path.join(args.out_dir, "d1_d2_feature_separation.csv"), index=False)
        gm.to_csv(os.path.join(args.out_dir, "d1_d2_group_means.csv"), index=False)

        summary["feature_analysis"] = {
            "features_csv": os.path.abspath(args.features_csv),
            "n_joined": int(len(m)),
            "top_feature": None if sep.empty else sep.iloc[0].to_dict(),
            "outputs": {
                "joined_csv": os.path.join(args.out_dir, "d1_d2_joined_features.csv"),
                "separation_csv": os.path.join(args.out_dir, "d1_d2_feature_separation.csv"),
                "group_means_csv": os.path.join(args.out_dir, "d1_d2_group_means.csv"),
            },
        }

    s_path = os.path.join(args.out_dir, "summary.json")
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", os.path.join(args.out_dir, "transition_by_gt.csv"))
    print("[saved]", os.path.join(args.out_dir, "d1_d2_counts_within_yes2no.csv"))
    print("[saved]", s_path)
    print("[counts]", summary["counts"])
    if "feature_analysis" in summary and summary["feature_analysis"]["top_feature"] is not None:
        print("[top_feature]", summary["feature_analysis"]["top_feature"])


if __name__ == "__main__":
    main()
