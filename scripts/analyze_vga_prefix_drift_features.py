#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


GENERIC_TOKEN_TEXTS = {
    "<0x0A>",
    ",",
    ".",
    ":",
    ";",
    "▁and",
    "▁dep",
    "icts",
    "▁depicts",
    "▁features",
    "▁image",
    "▁scene",
    "▁shows",
    "▁situated",
}

CONTENT_STOP = {
    "",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "dep",
    "depict",
    "depicts",
    "feature",
    "features",
    "for",
    "from",
    "has",
    "have",
    "having",
    "image",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "scene",
    "several",
    "show",
    "shows",
    "situated",
    "the",
    "there",
    "this",
    "to",
    "two",
    "with",
}


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                cols.append(key)
                seen.add(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and math.isfinite(float(value))
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y"}


def norm_token_text(text: Any) -> str:
    raw = str(text or "").replace("▁", "").replace("Ġ", "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "", raw)
    return raw


def is_word_start(text: Any) -> bool:
    raw = str(text or "")
    return raw.startswith("▁") or raw.startswith("Ġ") or raw.startswith(" ")


def is_generic(text: Any, norm: Any) -> bool:
    raw = str(text or "")
    tok = str(norm or "").strip().lower()
    return raw in GENERIC_TOKEN_TEXTS or tok in CONTENT_STOP


def is_content(text: Any, norm: Any) -> bool:
    tok = str(norm or "").strip().lower()
    if not tok:
        tok = norm_token_text(text)
    return is_word_start(text) and tok not in CONTENT_STOP and bool(re.fullmatch(r"[a-z][a-z0-9]{2,}", tok))


def first_true_step(sub: pd.DataFrame, mask: pd.Series) -> float:
    hit = sub.loc[mask.fillna(False), "step"]
    if hit.empty:
        return float("nan")
    return float(hit.min())


def nanmean(values: Iterable[Any]) -> float:
    vals = [to_float(x) for x in values]
    vals = [x for x in vals if math.isfinite(x)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def nanmedian(values: Iterable[Any]) -> float:
    vals = sorted(x for x in (to_float(v) for v in values) if math.isfinite(x))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def text_words(text: Any) -> List[str]:
    return [x.lower() for x in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*", str(text or ""))]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return float(len(aa & bb) / len(aa | bb))


def auc_high(y: Sequence[int], score: Sequence[float]) -> float:
    pairs = [(float(s), int(t)) for s, t in zip(score, y) if math.isfinite(float(s))]
    pos = [s for s, t in pairs if t == 1]
    neg = [s for s, t in pairs if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def average_precision(y: Sequence[int], score: Sequence[float]) -> float:
    pairs = sorted(
        [(float(s), int(t)) for s, t in zip(score, y) if math.isfinite(float(s))],
        key=lambda x: x[0],
        reverse=True,
    )
    n_pos = sum(t for _, t in pairs)
    if n_pos <= 0:
        return float("nan")
    hits = 0
    acc = 0.0
    for idx, (_, target) in enumerate(pairs, start=1):
        if target:
            hits += 1
            acc += hits / idx
    return float(acc / n_pos)


def precision_at_k(y: Sequence[int], score: Sequence[float], k: int) -> float:
    pairs = sorted(
        [(float(s), int(t)) for s, t in zip(score, y) if math.isfinite(float(s))],
        key=lambda x: x[0],
        reverse=True,
    )
    if not pairs:
        return float("nan")
    top = pairs[: min(int(k), len(pairs))]
    return float(sum(t for _, t in top) / len(top))


def feature_metrics(features: pd.DataFrame, *, comparison: str, groups: Sequence[str]) -> List[Dict[str, Any]]:
    sub = features[features["sample_group"].isin(groups)].copy().reset_index(drop=True)
    if sub.empty:
        return []
    y = (sub["sample_group"] == "target_recoverable").astype(int).tolist()
    skip = {"id", "sample_group", "mode", "lost_objects", "target"}
    out: List[Dict[str, Any]] = []
    for col in sub.columns:
        if col in skip:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce")
        valid = vals.notna()
        if valid.sum() < 4 or len(set(vals[valid].tolist())) < 2:
            continue
        yy = [int(v) for v in pd.Series(y, index=sub.index)[valid].tolist()]
        xx = vals[valid].astype(float).tolist()
        if not yy or sum(yy) == 0 or sum(yy) == len(yy):
            continue
        ah = auc_high(yy, xx)
        direction = "high"
        auc = ah
        score = xx
        if math.isfinite(ah) and ah < 0.5:
            direction = "low"
            auc = 1.0 - ah
            score = [-x for x in xx]
        out.append(
            {
                "comparison": comparison,
                "feature": col,
                "feature_family": "audit" if col.startswith("audit_") else col.split("_", 1)[0],
                "direction": direction,
                "auc": auc,
                "auc_high": ah,
                "ap": average_precision(yy, score),
                "p_at_10": precision_at_k(yy, score, 10),
                "p_at_25": precision_at_k(yy, score, 25),
                "n": int(len(yy)),
                "n_pos": int(sum(yy)),
                "n_neg": int(len(yy) - sum(yy)),
                "pos_mean": nanmean([x for x, t in zip(xx, yy) if t == 1]),
                "neg_mean": nanmean([x for x, t in zip(xx, yy) if t == 0]),
            }
        )
    return sorted(out, key=lambda r: (to_float(r["auc"], 0.0), to_float(r["ap"], 0.0)), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build sample-level feature metrics from VGA prefix-drift diagnostics.")
    ap.add_argument("--steps-csv", required=True)
    ap.add_argument("--samples-json", required=True)
    ap.add_argument("--manifest-csv", default="")
    ap.add_argument("--windows", default="5,10,15,30")
    ap.add_argument("--out-features-csv", required=True)
    ap.add_argument("--out-metrics-csv", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    steps = pd.read_csv(args.steps_csv)
    samples_obj = json.load(open(args.samples_json, encoding="utf-8"))
    samples = pd.DataFrame(samples_obj.get("samples", []))
    if "sample_group" not in samples.columns and "sample_group" in steps.columns:
        samples = samples.merge(steps[["id", "sample_group"]].drop_duplicates("id"), on="id", how="left")
    if "sample_group" not in samples.columns:
        samples["sample_group"] = "unknown"

    manifest = pd.DataFrame()
    if args.manifest_csv:
        manifest = pd.read_csv(args.manifest_csv)
        manifest["id"] = manifest["id"].astype(str)
        rename = {
            "group": "manifest_group",
            "base_n_gt_objects": "audit_base_n_gt_objects",
            "n_base_only_supported_unique": "audit_n_base_only_supported_unique",
            "delta_recall_base_minus_int": "audit_delta_recall_base_minus_int",
            "delta_f1_unique_base_minus_int": "audit_delta_f1_unique_base_minus_int",
            "delta_ci_unique_base_minus_int": "audit_delta_ci_unique_base_minus_int",
        }
        manifest = manifest.rename(columns={k: v for k, v in rename.items() if k in manifest.columns})

    samples["id"] = samples["id"].astype(str)
    steps["id"] = steps["id"].astype(str)
    windows = [int(x.strip()) for x in str(args.windows).split(",") if x.strip()]

    if not manifest.empty:
        keep = [
            c
            for c in [
                "id",
                "audit_base_n_gt_objects",
                "audit_n_base_only_supported_unique",
                "audit_delta_recall_base_minus_int",
                "audit_delta_f1_unique_base_minus_int",
                "audit_delta_ci_unique_base_minus_int",
            ]
            if c in manifest.columns
        ]
        samples = samples.merge(manifest[keep], on="id", how="left")

    rows: List[Dict[str, Any]] = []
    for _, sample in samples.iterrows():
        sid = str(sample.get("id", ""))
        sub = steps[steps["id"] == sid].copy()
        row: Dict[str, Any] = {
            "id": sid,
            "sample_group": sample.get("sample_group", ""),
            "mode": sample.get("mode", ""),
            "lost_objects": "|".join(sample.get("lost_objects", []))
            if isinstance(sample.get("lost_objects", []), list)
            else sample.get("lost_objects", ""),
            "target": int(sample.get("sample_group", "") == "target_recoverable"),
        }
        for col in [
            "audit_base_n_gt_objects",
            "audit_n_base_only_supported_unique",
            "audit_delta_recall_base_minus_int",
            "audit_delta_f1_unique_base_minus_int",
            "audit_delta_ci_unique_base_minus_int",
        ]:
            if col in sample:
                row[col] = sample.get(col)

        for col in ["first_divergence_step", "first_vanilla_lost_step", "first_pvg_lost_step"]:
            row[f"drift_{col}"] = to_float(sample.get(col))
        row["drift_neg_first_divergence_step"] = -row["drift_first_divergence_step"] if math.isfinite(row["drift_first_divergence_step"]) else float("nan")
        row["drift_vanilla_mentions_lost"] = int(math.isfinite(row["drift_first_vanilla_lost_step"]))
        row["drift_pvg_mentions_lost"] = int(math.isfinite(row["drift_first_pvg_lost_step"]))
        row["drift_pvg_misses_lost_when_vanilla_mentions"] = int(
            row["drift_vanilla_mentions_lost"] == 1 and row["drift_pvg_mentions_lost"] == 0
        )
        row["drift_divergence_before_vanilla_lost"] = int(
            math.isfinite(row["drift_first_divergence_step"])
            and math.isfinite(row["drift_first_vanilla_lost_step"])
            and row["drift_first_divergence_step"] < row["drift_first_vanilla_lost_step"]
        )

        v_words = text_words(sample.get("vanilla_generated_text", ""))
        p_words = text_words(sample.get("pvg_generated_text", ""))
        row["out_vanilla_word_count"] = len(v_words)
        row["out_pvg_word_count"] = len(p_words)
        row["out_delta_word_count_pvg_minus_vanilla"] = len(p_words) - len(v_words)
        row["out_caption_word_jaccard"] = jaccard(v_words, p_words)

        if sub.empty:
            rows.append(row)
            continue

        first = sub[sub.get("is_first_divergence_step", 0) == 1]
        if not first.empty:
            f = first.iloc[0]
            row["drift_pvg_generic_at_first_div"] = int(is_generic(f.get("pvg_next_token_text"), f.get("pvg_next_token_norm")))
            row["drift_vanilla_generic_at_first_div"] = int(
                is_generic(f.get("vanilla_next_token_text"), f.get("vanilla_next_token_norm"))
            )
            row["drift_delta_generic_at_first_div"] = (
                row["drift_pvg_generic_at_first_div"] - row["drift_vanilla_generic_at_first_div"]
            )
            row["drift_pvg_newline_at_first_div"] = int(str(f.get("pvg_next_token_text")) == "<0x0A>")
            row["cand_first_div_suppression"] = to_float(f.get("vanilla_minus_pvg_for_vanilla_best_lost"))
            row["cand_first_div_guidance_cosine"] = to_float(
                f.get("pvg_guidance_vs_vanilla_best_lost_guidance_token_dist_cosine")
            )

        for branch in ["vanilla", "pvg"]:
            sub[f"{branch}_generic"] = [
                is_generic(t, n)
                for t, n in zip(sub.get(f"{branch}_next_token_text", []), sub.get(f"{branch}_next_token_norm", []))
            ]
            sub[f"{branch}_content"] = [
                is_content(t, n)
                for t, n in zip(sub.get(f"{branch}_next_token_text", []), sub.get(f"{branch}_next_token_norm", []))
            ]
            sub[f"{branch}_newline"] = sub.get(f"{branch}_next_token_text", pd.Series(index=sub.index)).astype(str) == "<0x0A>"

        for w in windows:
            early = sub[sub["step"] < int(w)].copy()
            if early.empty:
                continue
            row[f"drift_has_divergence_w{w}"] = int(row["drift_first_divergence_step"] < w) if math.isfinite(row["drift_first_divergence_step"]) else 0
            for branch in ["vanilla", "pvg"]:
                row[f"drift_{branch}_generic_rate_w{w}"] = float(early[f"{branch}_generic"].mean())
                row[f"drift_{branch}_content_rate_w{w}"] = float(early[f"{branch}_content"].mean())
                row[f"drift_{branch}_newline_rate_w{w}"] = float(early[f"{branch}_newline"].mean())
                row[f"drift_{branch}_first_content_step_w{w}"] = first_true_step(early, early[f"{branch}_content"])
                row[f"drift_{branch}_entropy_mean_w{w}"] = nanmean(early.get(f"{branch}_entropy", []))
                row[f"drift_{branch}_top1_gap_mean_w{w}"] = nanmean(early.get(f"{branch}_top1_gap", []))
                row[f"drift_{branch}_top1_logprob_mean_w{w}"] = nanmean(early.get(f"{branch}_top1_logprob", []))
            row[f"drift_delta_generic_rate_w{w}"] = row[f"drift_pvg_generic_rate_w{w}"] - row[f"drift_vanilla_generic_rate_w{w}"]
            row[f"drift_delta_content_rate_w{w}"] = row[f"drift_pvg_content_rate_w{w}"] - row[f"drift_vanilla_content_rate_w{w}"]
            row[f"drift_delta_newline_rate_w{w}"] = row[f"drift_pvg_newline_rate_w{w}"] - row[f"drift_vanilla_newline_rate_w{w}"]
            row[f"drift_delta_first_content_step_w{w}"] = (
                row[f"drift_pvg_first_content_step_w{w}"] - row[f"drift_vanilla_first_content_step_w{w}"]
                if math.isfinite(row[f"drift_pvg_first_content_step_w{w}"])
                and math.isfinite(row[f"drift_vanilla_first_content_step_w{w}"])
                else float("nan")
            )
            row[f"drift_delta_entropy_mean_w{w}"] = row[f"drift_pvg_entropy_mean_w{w}"] - row[f"drift_vanilla_entropy_mean_w{w}"]
            row[f"drift_delta_top1_gap_mean_w{w}"] = row[f"drift_pvg_top1_gap_mean_w{w}"] - row[f"drift_vanilla_top1_gap_mean_w{w}"]
            row[f"drift_delta_top1_logprob_mean_w{w}"] = (
                row[f"drift_pvg_top1_logprob_mean_w{w}"] - row[f"drift_vanilla_top1_logprob_mean_w{w}"]
            )

        pre = sub[sub.get("prefix_equal_before_step", 0).map(safe_bool)].copy()
        if not pre.empty:
            suppress = pd.to_numeric(pre.get("vanilla_minus_pvg_for_vanilla_best_lost"), errors="coerce")
            cosine = pd.to_numeric(
                pre.get("pvg_guidance_vs_vanilla_best_lost_guidance_token_dist_cosine"), errors="coerce"
            )
            row["cand_pre_suppression_mean"] = nanmean(suppress.tolist())
            row["cand_pre_suppression_median"] = nanmedian(suppress.tolist())
            row["cand_pre_suppress_rate_gt_050"] = float((suppress > 0.5).mean()) if suppress.notna().any() else float("nan")
            row["cand_pre_pvg_lift_rate_gt_050"] = float((suppress < -0.5).mean()) if suppress.notna().any() else float("nan")
            row["cand_pre_guidance_cosine_median"] = nanmedian(cosine.tolist())
            row["cand_pre_low_guidance_cosine_rate_lt_020"] = float((cosine < 0.2).mean()) if cosine.notna().any() else float("nan")

        rows.append(row)

    metrics: List[Dict[str, Any]] = []
    comparisons = {
        "target_vs_all_controls": ["target_recoverable", "safe_control", "loss_nonrecoverable_control"],
        "target_vs_safe_control": ["target_recoverable", "safe_control"],
        "target_vs_loss_nonrecoverable_control": ["target_recoverable", "loss_nonrecoverable_control"],
    }
    features = pd.DataFrame(rows)
    for name, groups in comparisons.items():
        metrics.extend(feature_metrics(features, comparison=name, groups=groups))

    non_audit_metrics = [row for row in metrics if row.get("feature_family") != "audit"]
    audit_metrics = [row for row in metrics if row.get("feature_family") == "audit"]

    write_csv(args.out_features_csv, rows)
    write_csv(args.out_metrics_csv, metrics)
    summary = {
        "inputs": vars(args),
        "counts": {
            "n_samples": int(len(features)),
            "by_group": features.groupby("sample_group").size().to_dict() if "sample_group" in features else {},
            "n_metrics": len(metrics),
        },
        "top_non_audit_metrics": non_audit_metrics[:30],
        "top_audit_reference_metrics": audit_metrics[:20],
        "outputs": {
            "features_csv": os.path.abspath(args.out_features_csv),
            "metrics_csv": os.path.abspath(args.out_metrics_csv),
            "summary_json": os.path.abspath(args.out_summary_json),
        },
    }
    write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_features_csv))
    print("[saved]", os.path.abspath(args.out_metrics_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))
    print("== top non-audit metrics ==")
    for row in non_audit_metrics[:20]:
        print(
            row["comparison"],
            row["feature"],
            "dir=",
            row["direction"],
            "auc=",
            row["auc"],
            "ap=",
            row["ap"],
            "n=",
            row["n"],
        )
    print("\n== top audit reference metrics ==")
    for row in audit_metrics[:10]:
        print(
            row["comparison"],
            row["feature"],
            "dir=",
            row["direction"],
            "auc=",
            row["auc"],
            "ap=",
            row["ap"],
            "n=",
            row["n"],
        )


if __name__ == "__main__":
    main()
