#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from extract_generative_semantic_pairwise_features import content_tokens, read_prediction_map


NONE_RE = re.compile(r"^\s*(?:none|no visible|not visible|n/a|na)\s*[.!]?\s*$", flags=re.IGNORECASE)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("id") or row.get("image_id") or row.get("question_id") or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def target_flag(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def frac(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def norm_phrase(text: str) -> str:
    return " ".join(tok for tok, _ in content_tokens(text))


def split_response_items(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text or NONE_RE.match(text):
        return []
    text = re.sub(r"\b(?:visible|objects?|entities?|candidates?)\s*:\s*", "", text, flags=re.IGNORECASE)
    parts = re.split(r"[,;\n]+", text)
    out: List[str] = []
    for part in parts:
        part = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip(" .")
        if part and not NONE_RE.match(part):
            out.append(part)
    return out


def selected_candidates(response: str, candidates: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    items = split_response_items(response)
    norm_items = [norm_phrase(item) for item in items]
    norm_items = [item for item in norm_items if item]
    selected: List[str] = []
    selected_norms = set()
    for cand in candidates:
        cand_norm = norm_phrase(cand)
        if not cand_norm:
            continue
        cand_tokens = set(cand_norm.split())
        ok = False
        for item_norm in norm_items:
            item_tokens = set(item_norm.split())
            if cand_norm == item_norm:
                ok = True
            elif len(cand_tokens) == 1 and cand_tokens <= item_tokens:
                ok = True
            elif len(cand_tokens) > 1 and (cand_tokens <= item_tokens or item_tokens <= cand_tokens):
                ok = True
            if ok:
                break
        if ok and cand_norm not in selected_norms:
            selected_norms.add(cand_norm)
            selected.append(cand)
    added = [item for item in norm_items if item and item not in selected_norms]
    return selected, added, norm_items


def auc_high(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    good = 0.0
    total = 0
    for a in pos:
        for b in neg:
            total += 1
            if a > b:
                good += 1.0
            elif a == b:
                good += 0.5
    return good / float(total) if total else None


def average_precision(items: Sequence[Tuple[int, float]]) -> Optional[float]:
    ranked = sorted(items, key=lambda item: item[1], reverse=True)
    n_pos = sum(label for label, _ in ranked)
    if n_pos <= 0:
        return None
    hits = 0
    total = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label:
            hits += 1
            total += hits / float(rank)
    return total / float(n_pos)


def precision_at(items: Sequence[Tuple[int, float]], k: int) -> Optional[float]:
    top = sorted(items, key=lambda item: item[1], reverse=True)[:k]
    if not top:
        return None
    return sum(label for label, _ in top) / float(len(top))


def numeric_feature_metrics(rows: Sequence[Dict[str, Any]], target_col: str) -> List[Dict[str, Any]]:
    excluded = {
        "id",
        "image",
        "candidate_terms",
        "verifier_output",
        "selected_candidates",
        "added_response_items",
    }
    metrics: List[Dict[str, Any]] = []
    for feature in rows[0].keys() if rows else []:
        if feature in excluded or feature.endswith("_json"):
            continue
        pairs: List[Tuple[int, float]] = []
        for row in rows:
            value = safe_float(row.get(feature))
            if value is not None:
                pairs.append((target_flag(row.get(target_col)), value))
        if len(pairs) < max(10, int(0.8 * len(rows))):
            continue
        vals = {round(score, 12) for _, score in pairs}
        if len(vals) < 3:
            continue
        pos = [score for label, score in pairs if label]
        neg = [score for label, score in pairs if not label]
        auc = auc_high(pos, neg)
        if auc is None:
            continue
        direction = "high" if auc >= 0.5 else "low"
        oriented = [(label, score if direction == "high" else -score) for label, score in pairs]
        metrics.append(
            {
                "feature": feature,
                "direction": direction,
                "auroc": max(float(auc), float(1.0 - auc)),
                "auroc_high": float(auc),
                "ap": average_precision(oriented),
                "p_at_10": precision_at(oriented, 10),
                "p_at_25": precision_at(oriented, 25),
                "p_at_50": precision_at(oriented, 50),
                "n": len(pairs),
                "n_pos": sum(label for label, _ in pairs),
                "pos_mean": sum(pos) / float(len(pos)) if pos else "",
                "neg_mean": sum(neg) / float(len(neg)) if neg else "",
            }
        )
    metrics.sort(key=lambda row: (float(row.get("auroc") or 0.0), float(row.get("ap") or 0.0)), reverse=True)
    return metrics


def load_candidates(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = safe_id(row)
        try:
            candidates = json.loads(str(row.get("candidate_terms_json") or "[]"))
        except Exception:
            candidates = []
        out[sid] = {**row, "candidate_terms_list": [str(x) for x in candidates]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze candidate-conditioned inventory verifier outputs.")
    ap.add_argument("--candidate_question_jsonl", required=True)
    ap.add_argument("--verifier_pred_jsonl", required=True)
    ap.add_argument("--oracle_rows_csv", required=True)
    ap.add_argument("--target_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_feature_metrics_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    args = ap.parse_args()

    candidates_by_id = load_candidates(args.candidate_question_jsonl)
    verifier = read_prediction_map(args.verifier_pred_jsonl)
    oracle_by_id = {safe_id(row): row for row in read_csv_rows(args.oracle_rows_csv)}
    ids = sorted(
        set(candidates_by_id) & set(verifier) & set(oracle_by_id),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )

    rows: List[Dict[str, Any]] = []
    for sid in ids:
        candidates = candidates_by_id[sid]["candidate_terms_list"]
        output = verifier[sid]["text"]
        selected, added, norm_items = selected_candidates(output, candidates)
        selected_tokens = {tok for cand in selected for tok, _ in content_tokens(cand)}
        candidate_tokens = {tok for cand in candidates for tok, _ in content_tokens(cand)}
        row = {
            "id": sid,
            "image": candidates_by_id[sid].get("image", ""),
            args.target_col: target_flag(oracle_by_id[sid].get(args.target_col)),
            "candidate_terms": "|".join(candidates),
            "verifier_output": output,
            "selected_candidates": "|".join(selected),
            "added_response_items": "|".join(added),
            "cand_candidate_count": len(candidates),
            "cand_candidate_token_count": len(candidate_tokens),
            "cand_selected_count": len(selected),
            "cand_selected_token_count": len(selected_tokens),
            "cand_selected_rate": frac(len(selected), len(candidates)),
            "cand_selected_token_rate": frac(len(selected_tokens), len(candidate_tokens)),
            "cand_unselected_count": len(candidates) - len(selected),
            "cand_response_item_count": len(norm_items),
            "cand_added_response_item_count": len(added),
            "cand_added_response_item_rate": frac(len(added), len(norm_items)),
            "cand_selected_minus_added": len(selected) - len(added),
            "cand_selected_x_candidate_count": len(selected) * len(candidates),
        }
        rows.append(row)

    metrics = numeric_feature_metrics(rows, args.target_col)
    target_count = sum(target_flag(row.get(args.target_col)) for row in rows)
    write_csv(args.out_csv, rows)
    write_csv(args.out_feature_metrics_csv, metrics)
    write_json(
        args.out_summary_json,
        {
            "inputs": {
                "candidate_question_jsonl": os.path.abspath(args.candidate_question_jsonl),
                "verifier_pred_jsonl": os.path.abspath(args.verifier_pred_jsonl),
                "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv),
                "target_col": args.target_col,
            },
            "counts": {
                "n_rows": len(rows),
                "n_target": int(target_count),
                "target_rate": frac(target_count, len(rows)),
                "n_feature_metrics": len(metrics),
            },
            "top_feature_metrics": metrics[:30],
            "outputs": {
                "features_csv": os.path.abspath(args.out_csv),
                "feature_metrics_csv": os.path.abspath(args.out_feature_metrics_csv),
                "summary_json": os.path.abspath(args.out_summary_json),
            },
        },
    )
    print("[saved]", os.path.abspath(args.out_csv))
    print("[saved]", os.path.abspath(args.out_feature_metrics_csv))
    print("[saved]", os.path.abspath(args.out_summary_json))
    for metric in metrics[:10]:
        print(
            "[metric]",
            metric["feature"],
            "dir=",
            metric["direction"],
            "auc=",
            metric["auroc"],
            "ap=",
            metric["ap"],
        )


if __name__ == "__main__":
    main()
