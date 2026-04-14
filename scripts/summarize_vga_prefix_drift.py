#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import pandas as pd


GENERIC_TOKENS = {
    "<0x0A>",
    "▁shows",
    "▁dep",
    "icts",
    "▁depicts",
    "▁features",
    "▁image",
    "▁scene",
    "▁situated",
    "▁and",
    ",",
    ".",
}


def is_num(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize VGA free-run prefix drift diagnostics.")
    ap.add_argument("--steps-csv", required=True)
    ap.add_argument("--samples-json", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    steps = pd.read_csv(args.steps_csv)
    samples_obj = json.load(open(args.samples_json))
    samples = pd.DataFrame(samples_obj.get("samples", []))
    if "sample_group" not in samples.columns and "sample_group" in steps.columns:
        groups = steps[["id", "sample_group"]].drop_duplicates("id")
        samples = samples.merge(groups, on="id", how="left")
    if "sample_group" not in samples.columns:
        samples["sample_group"] = "unknown"

    for col in [
        "first_divergence_step",
        "first_vanilla_lost_step",
        "first_pvg_lost_step",
    ]:
        samples[col] = pd.to_numeric(samples.get(col), errors="coerce")
    samples["has_divergence"] = samples["first_divergence_step"].notna()
    samples["vanilla_mentions_lost"] = samples["first_vanilla_lost_step"].notna()
    samples["pvg_mentions_lost"] = samples["first_pvg_lost_step"].notna()
    samples["divergence_before_vanilla_lost"] = (
        samples["has_divergence"]
        & samples["vanilla_mentions_lost"]
        & (samples["first_divergence_step"] < samples["first_vanilla_lost_step"])
    )
    samples["pvg_misses_lost_when_vanilla_mentions"] = samples["vanilla_mentions_lost"] & ~samples["pvg_mentions_lost"]

    first = steps[steps.get("is_first_divergence_step", 0) == 1].copy()
    if not first.empty:
        first["pvg_generic_divergence"] = first["pvg_next_token_text"].astype(str).isin(GENERIC_TOKENS)
        first["vanilla_generic_divergence"] = first["vanilla_next_token_text"].astype(str).isin(GENERIC_TOKENS)
        for col in [
            "vanilla_minus_pvg_for_vanilla_best_lost",
            "pvg_guidance_vs_vanilla_best_lost_guidance_token_dist_cosine",
        ]:
            first[col] = pd.to_numeric(first.get(col), errors="coerce")
    else:
        first["pvg_generic_divergence"] = []
        first["vanilla_generic_divergence"] = []

    sample_summary = (
        samples.groupby("sample_group")
        .agg(
            n=("id", "count"),
            divergence_rate=("has_divergence", "mean"),
            first_divergence_median=("first_divergence_step", "median"),
            vanilla_lost_rate=("vanilla_mentions_lost", "mean"),
            pvg_lost_rate=("pvg_mentions_lost", "mean"),
            pvg_misses_lost_when_vanilla_mentions=("pvg_misses_lost_when_vanilla_mentions", "mean"),
            divergence_before_vanilla_lost=("divergence_before_vanilla_lost", "mean"),
        )
        .reset_index()
    )

    if not first.empty:
        first_summary = (
            first.groupby("sample_group")
            .agg(
                n=("id", "count"),
                first_step_median=("step", "median"),
                pvg_generic_divergence=("pvg_generic_divergence", "mean"),
                vanilla_generic_divergence=("vanilla_generic_divergence", "mean"),
                suppression_median=("vanilla_minus_pvg_for_vanilla_best_lost", "median"),
                guidance_cosine_median=("pvg_guidance_vs_vanilla_best_lost_guidance_token_dist_cosine", "median"),
            )
            .reset_index()
        )
    else:
        first_summary = pd.DataFrame()

    out = {
        "inputs": vars(args),
        "sample_summary": sample_summary.to_dict(orient="records"),
        "first_divergence_summary": first_summary.to_dict(orient="records"),
        "first_divergence_rows": first.to_dict(orient="records"),
        "samples": samples.to_dict(orient="records"),
    }
    write_json(args.out_json, out)

    print("== sample summary ==")
    print(sample_summary.to_string(index=False))
    if not first_summary.empty:
        print("\n== first divergence summary ==")
        print(first_summary.to_string(index=False))
        print("\n== first divergence rows ==")
        cols = [
            "id",
            "sample_group",
            "mode",
            "step",
            "vanilla_next_token_text",
            "pvg_next_token_text",
            "vanilla_lost_best_word",
            "vanilla_lost_best_rank",
            "pvg_rank_for_vanilla_best_lost",
            "vanilla_minus_pvg_for_vanilla_best_lost",
            "pvg_guidance_vs_vanilla_best_lost_guidance_token_dist_cosine",
        ]
        existing = [c for c in cols if c in first.columns]
        print(first[existing].to_string(index=False))
    print("[saved]", os.path.abspath(args.out_json))


if __name__ == "__main__":
    main()
