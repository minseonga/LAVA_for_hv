#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TEXT_KEYS = ("output", "text", "caption", "prediction", "answer")
ID_KEYS = ("image_id", "question_id", "id", "qid")


# Generic English function words and caption boilerplate. This intentionally does
# not use CHAIR/COCO object lists or mscoco_generated_words.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "for",
    "from",
    "has",
    "have",
    "having",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "under",
    "was",
    "we",
    "were",
    "which",
    "while",
    "with",
    "you",
    "your",
    # Caption boilerplate / weak semantic carriers.
    "background",
    "foreground",
    "front",
    "back",
    "behind",
    "beside",
    "near",
    "next",
    "around",
    "across",
    "area",
    "areas",
    "appears",
    "appear",
    "appearing",
    "depict",
    "depicts",
    "depicted",
    "image",
    "picture",
    "photo",
    "photograph",
    "scene",
    "show",
    "shows",
    "shown",
    "visible",
    "overall",
    "quite",
    "very",
    "some",
    "many",
    "much",
    "several",
    "various",
    "large",
    "small",
    "big",
    "little",
    "different",
    "color",
    "colors",
    "colored",
    "white",
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "gray",
    "grey",
}


LIGHT_ALIASES = {
    "people": "person",
    "persons": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
}


GENERIC_FILLERS = {
    "image",
    "picture",
    "scene",
    "photo",
    "depicts",
    "shows",
    "appears",
    "background",
    "foreground",
    "overall",
    "various",
    "several",
    "many",
    "some",
    "large",
    "small",
    "different",
    "busy",
    "clear",
    "likely",
    "possibly",
}


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']*")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(obj: Dict[str, Any]) -> str:
    for key in ID_KEYS:
        value = str(obj.get(key, "")).strip()
        if value:
            try:
                return str(int(value))
            except Exception:
                return value
    return ""


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def read_prediction_map(path: str, text_key: str = "auto") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = safe_id(obj)
            if not sid:
                continue
            if text_key and text_key != "auto":
                text = obj.get(text_key, "")
            else:
                text = ""
                for key in TEXT_KEYS:
                    if obj.get(key) is not None:
                        text = obj.get(key)
                        break
            out[sid] = {
                "id": sid,
                "image": obj.get("image", ""),
                "text": clean_text(text),
            }
    return out


def load_trace_rows(path: str) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    rows = read_csv_rows(os.path.abspath(path))
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sid = str(row.get("id") or row.get("image_id") or row.get("question_id") or "").strip()
        if sid:
            try:
                sid = str(int(sid))
            except Exception:
                pass
            out[sid] = row
    return out


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def normalize_token(token: str) -> str:
    tok = token.lower().strip("'")
    tok = LIGHT_ALIASES.get(tok, tok)
    if len(tok) > 4 and tok.endswith("ies"):
        tok = tok[:-3] + "y"
    elif len(tok) > 5 and tok.endswith("ves"):
        tok = tok[:-3] + "f"
    elif len(tok) > 4 and tok.endswith("es") and not tok.endswith(("ses", "xes")):
        tok = tok[:-2]
    elif len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us")):
        tok = tok[:-1]
    return tok


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def content_tokens(text: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for pos, token in enumerate(tokenize(text)):
        tok = normalize_token(token)
        if len(tok) <= 1:
            continue
        if tok in STOPWORDS:
            continue
        if tok.isdigit():
            continue
        out.append((tok, pos))
    return out


def semantic_units(tokens: List[Tuple[str, int]], max_gap: int = 3) -> Tuple[List[str], List[str]]:
    unigrams = [tok for tok, _ in tokens]
    phrases: List[str] = []
    for idx in range(len(tokens) - 1):
        left, left_pos = tokens[idx]
        right, right_pos = tokens[idx + 1]
        if right_pos - left_pos <= max_gap and left != right:
            phrases.append(f"{left}_{right}")
    return unigrams, phrases


def repetition_rate(items: Sequence[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    repeated = sum(max(0, c - 1) for c in counts.values())
    return repeated / float(len(items))


def generic_ratio(tokens_with_pos: Sequence[Tuple[str, int]]) -> float:
    if not tokens_with_pos:
        return 0.0
    return sum(1 for tok, _ in tokens_with_pos if tok in GENERIC_FILLERS) / float(len(tokens_with_pos))


def unit_summary(text: str) -> Dict[str, Any]:
    toks = content_tokens(text)
    token_units, phrase_units = semantic_units(toks)
    all_units = token_units + phrase_units
    seen = set()
    first_positions: Dict[str, int] = {}
    new_flags: List[int] = []
    for tok, pos in toks:
        is_new = tok not in seen
        new_flags.append(int(is_new))
        if is_new:
            first_positions[tok] = pos
            seen.add(tok)
    n_raw = len(tokenize(text))
    tail_start = max(0, int(len(toks) * 0.67))
    tail_flags = new_flags[tail_start:] if toks else []
    last_new_pos = max(first_positions.values()) if first_positions else -1
    last_new_pos_frac = (last_new_pos + 1) / float(max(1, n_raw)) if last_new_pos >= 0 else 0.0
    return {
        "text": text,
        "n_words": n_raw,
        "n_content_tokens": len(toks),
        "content_tokens": token_units,
        "phrase_units": phrase_units,
        "all_units": all_units,
        "unique_token_units": set(token_units),
        "unique_phrase_units": set(phrase_units),
        "unique_all_units": set(all_units),
        "content_diversity": len(set(token_units)) / float(max(1, len(token_units))),
        "phrase_diversity": len(set(phrase_units)) / float(max(1, len(phrase_units))),
        "repetition_rate": repetition_rate(token_units),
        "phrase_repetition_rate": repetition_rate(phrase_units),
        "last_new_unit_pos_frac": last_new_pos_frac,
        "tail_new_unit_rate": sum(tail_flags) / float(max(1, len(tail_flags))),
        "generic_phrase_ratio": generic_ratio(toks),
        "tail_content_token_count": len(toks) - tail_start if toks else 0,
    }


def sorted_preview(values: Iterable[str], limit: int) -> str:
    vals = sorted({str(v) for v in values if str(v).strip()})
    if limit > 0:
        vals = vals[:limit]
    return "|".join(vals)


def add_trace_copy(out: Dict[str, Any], trace: Dict[str, str], prefix: str, cols: Sequence[str]) -> None:
    for col in cols:
        value = safe_float(trace.get(col))
        if value is not None:
            out[f"{prefix}_{col}"] = value


def parse_trace_word_rows(trace: Dict[str, str]) -> List[Dict[str, Any]]:
    raw = str(trace.get("probe_content_word_trace_json", "")).strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    return [row for row in data if isinstance(row, dict)]


def add_v53_aliases(out: Dict[str, Any], baseline_trace: Dict[str, str], token_base_only: Iterable[str]) -> None:
    """Expose the v53 benefit-cost names without changing the raw v52 columns."""
    out["sem_benefit_delta_unique_content_units"] = out["sem_delta_unique_unit_count"]
    out["sem_benefit_base_only_sem_unit_count"] = out["sem_base_only_unit_count"]
    out["sem_benefit_delta_last_new_content_pos_frac"] = out["sem_delta_last_new_content_pos_frac"]
    out["sem_benefit_delta_tail_new_content_rate"] = out["sem_delta_tail_new_content_rate"]
    out["sem_benefit_delta_content_diversity"] = out["sem_delta_content_diversity"]
    out["sem_cost_base_tail_repetition_rate"] = out["sem_base_repetition_rate"]
    out["sem_cost_base_generic_phrase_ratio"] = out["sem_base_generic_phrase_ratio"]

    # Baseline trace cost branch. These are generic content-token trace columns,
    # not CHAIR object parser outputs.
    trace_aliases = {
        "sem_trace_base_probe_lp_content_min_real": "sem_cost_base_new_content_lp_min",
        "sem_trace_base_probe_target_gap_content_min_real": "sem_cost_base_new_content_gap_min",
        "sem_trace_base_probe_entropy_content_max_real": "sem_cost_base_new_content_entropy_max",
        "sem_trace_base_probe_stop_eos_margin_real": "sem_cost_base_stop_eos_margin",
        "sem_trace_base_probe_stop_eos_logprob_real": "sem_cost_base_stop_eos_logprob",
        "sem_trace_base_probe_lp_tail_minus_head_real": "sem_cost_base_lp_tail_minus_head",
        "sem_trace_base_probe_gap_tail_minus_head_real": "sem_cost_base_gap_tail_minus_head",
    }
    for src, dst in trace_aliases.items():
        if src in out:
            out[dst] = out[src]

    lp = safe_float(out.get("sem_cost_base_new_content_lp_min"))
    gap = safe_float(out.get("sem_cost_base_new_content_gap_min"))
    ent = safe_float(out.get("sem_cost_base_new_content_entropy_max"))
    base_only = safe_float(out.get("sem_benefit_base_only_sem_unit_count")) or 0.0
    if lp is not None:
        out["sem_cost_base_low_lp_x_base_only"] = max(0.0, -lp) * base_only
    if gap is not None:
        out["sem_cost_base_low_gap_x_base_only"] = max(0.0, -gap) * base_only
    if ent is not None:
        out["sem_cost_base_high_entropy_x_base_only"] = ent * base_only

    base_only_tokens = {normalize_token(str(tok)) for tok in token_base_only if str(tok).strip()}
    new_rows = [
        row
        for row in parse_trace_word_rows(baseline_trace)
        if normalize_token(str(row.get("word", ""))) in base_only_tokens
    ]
    out["sem_cost_base_new_content_word_count"] = len(new_rows)
    if new_rows:
        lp_vals = [safe_float(row.get("lp_min")) for row in new_rows]
        gap_vals = [safe_float(row.get("gap_min")) for row in new_rows]
        ent_vals = [safe_float(row.get("ent_max")) for row in new_rows]
        lp_vals = [v for v in lp_vals if v is not None]
        gap_vals = [v for v in gap_vals if v is not None]
        ent_vals = [v for v in ent_vals if v is not None]
        out["sem_cost_base_new_content_lp_min"] = min(lp_vals) if lp_vals else 0.0
        out["sem_cost_base_new_content_gap_min"] = min(gap_vals) if gap_vals else 0.0
        out["sem_cost_base_new_content_entropy_max"] = max(ent_vals) if ent_vals else 0.0
        out["sem_cost_base_new_content_low_lp_count_le_m2"] = sum(1 for v in lp_vals if v <= -2.0)
        out["sem_cost_base_new_content_low_lp_count_le_m3"] = sum(1 for v in lp_vals if v <= -3.0)
        out["sem_cost_base_new_content_low_gap_count_le_000"] = sum(1 for v in gap_vals if v <= 0.0)
        out["sem_cost_base_new_content_low_gap_count_le_025"] = sum(1 for v in gap_vals if v <= 0.25)
        out["sem_cost_base_new_content_high_entropy_count_ge_300"] = sum(1 for v in ent_vals if v >= 3.0)
        out["sem_cost_base_new_content_high_entropy_count_ge_350"] = sum(1 for v in ent_vals if v >= 3.5)
    else:
        out["sem_cost_base_new_content_low_lp_count_le_m2"] = 0
        out["sem_cost_base_new_content_low_lp_count_le_m3"] = 0
        out["sem_cost_base_new_content_low_gap_count_le_000"] = 0
        out["sem_cost_base_new_content_low_gap_count_le_025"] = 0
        out["sem_cost_base_new_content_high_entropy_count_ge_300"] = 0
        out["sem_cost_base_new_content_high_entropy_count_ge_350"] = 0


def add_v54_visual_aliases(out: Dict[str, Any]) -> None:
    """Pairwise visual-token trace aliases for post-hoc VGA/PAI-style routing."""

    def get(name: str) -> Optional[float]:
        return safe_float(out.get(name))

    pairs = {
        "content_mean": "probe_vis_attn_content_mean_real",
        "content_min": "probe_vis_attn_content_min_real",
        "tail_mean": "probe_vis_attn_tail_mean_real",
        "last4_mean": "probe_vis_attn_last4_mean_real",
        "tail_minus_head": "probe_vis_attn_tail_minus_head_real",
        "topk_content_mean": "probe_vis_attn_topk_content_mean_real",
        "topk_tail_minus_head": "probe_vis_attn_topk_tail_minus_head_real",
        "entropy_content_mean": "probe_vis_entropy_content_mean_real",
        "entropy_content_max": "probe_vis_entropy_content_max_real",
        "entropy_tail_minus_head": "probe_vis_entropy_tail_minus_head_real",
        "top1_mass_content_mean": "probe_vis_top1_mass_content_mean_real",
        "ess_content_mean": "probe_vis_ess_content_mean_real",
    }
    for alias, col in pairs.items():
        b = get(f"sem_trace_base_{col}")
        i = get(f"sem_trace_int_{col}")
        if b is None or i is None:
            continue
        out[f"sem_visual_base_{alias}"] = b
        out[f"sem_visual_int_{alias}"] = i
        # Positive value means the intervention caption is less visually grounded
        # than the baseline replay at the same generic caption-token slice.
        out[f"sem_visual_suppression_{alias}"] = b - i
        out[f"sem_visual_uplift_{alias}"] = i - b

    benefit = safe_float(out.get("sem_benefit_base_only_sem_unit_count")) or 0.0
    for alias in ("content_mean", "tail_mean", "last4_mean", "tail_minus_head"):
        suppression = safe_float(out.get(f"sem_visual_suppression_{alias}"))
        if suppression is not None:
            out[f"sem_visual_suppression_x_base_only_{alias}"] = max(0.0, suppression) * benefit


def build_row(
    sid: str,
    base: Dict[str, Any],
    intervention: Dict[str, Any],
    *,
    baseline_trace: Dict[str, str],
    intervention_trace: Dict[str, str],
    trace_cols: Sequence[str],
    preview_limit: int,
) -> Dict[str, Any]:
    b = unit_summary(base.get("text", ""))
    i = unit_summary(intervention.get("text", ""))
    b_units = b["unique_all_units"]
    i_units = i["unique_all_units"]
    b_token_units = b["unique_token_units"]
    i_token_units = i["unique_token_units"]
    b_phrase_units = b["unique_phrase_units"]
    i_phrase_units = i["unique_phrase_units"]
    base_only = b_units - i_units
    int_only = i_units - b_units
    shared = b_units & i_units
    token_base_only = b_token_units - i_token_units
    phrase_base_only = b_phrase_units - i_phrase_units
    union = b_units | i_units
    row: Dict[str, Any] = {
        "id": sid,
        "image": intervention.get("image") or base.get("image") or "",
        "sem_base_n_words": b["n_words"],
        "sem_int_n_words": i["n_words"],
        "sem_delta_n_words": b["n_words"] - i["n_words"],
        "sem_base_n_content_tokens": b["n_content_tokens"],
        "sem_int_n_content_tokens": i["n_content_tokens"],
        "sem_delta_n_content_tokens": b["n_content_tokens"] - i["n_content_tokens"],
        "sem_base_unique_unit_count": len(b_units),
        "sem_int_unique_unit_count": len(i_units),
        "sem_delta_unique_unit_count": len(b_units) - len(i_units),
        "sem_base_unique_token_unit_count": len(b_token_units),
        "sem_int_unique_token_unit_count": len(i_token_units),
        "sem_delta_unique_token_unit_count": len(b_token_units) - len(i_token_units),
        "sem_base_unique_phrase_unit_count": len(b_phrase_units),
        "sem_int_unique_phrase_unit_count": len(i_phrase_units),
        "sem_delta_unique_phrase_unit_count": len(b_phrase_units) - len(i_phrase_units),
        "sem_base_only_unit_count": len(base_only),
        "sem_int_only_unit_count": len(int_only),
        "sem_shared_unit_count": len(shared),
        "sem_unit_jaccard": len(shared) / float(max(1, len(union))),
        "sem_base_only_unit_rate": len(base_only) / float(max(1, len(b_units))),
        "sem_int_only_unit_rate": len(int_only) / float(max(1, len(i_units))),
        "sem_token_base_only_count": len(token_base_only),
        "sem_phrase_base_only_count": len(phrase_base_only),
        "sem_base_content_diversity": b["content_diversity"],
        "sem_int_content_diversity": i["content_diversity"],
        "sem_delta_content_diversity": b["content_diversity"] - i["content_diversity"],
        "sem_base_repetition_rate": b["repetition_rate"],
        "sem_int_repetition_rate": i["repetition_rate"],
        "sem_repetition_delta_int_minus_base": i["repetition_rate"] - b["repetition_rate"],
        "sem_base_last_new_content_pos_frac": b["last_new_unit_pos_frac"],
        "sem_int_last_new_content_pos_frac": i["last_new_unit_pos_frac"],
        "sem_delta_last_new_content_pos_frac": b["last_new_unit_pos_frac"] - i["last_new_unit_pos_frac"],
        "sem_base_tail_new_content_rate": b["tail_new_unit_rate"],
        "sem_int_tail_new_content_rate": i["tail_new_unit_rate"],
        "sem_delta_tail_new_content_rate": b["tail_new_unit_rate"] - i["tail_new_unit_rate"],
        "sem_base_tail_content_token_count": b["tail_content_token_count"],
        "sem_int_tail_content_token_count": i["tail_content_token_count"],
        "sem_delta_tail_content_token_count": b["tail_content_token_count"] - i["tail_content_token_count"],
        "sem_base_generic_phrase_ratio": b["generic_phrase_ratio"],
        "sem_int_generic_phrase_ratio": i["generic_phrase_ratio"],
        "sem_delta_generic_phrase_ratio_base_minus_int": b["generic_phrase_ratio"] - i["generic_phrase_ratio"],
        "sem_base_text": base.get("text", ""),
        "sem_int_text": intervention.get("text", ""),
        "sem_base_only_units": sorted_preview(base_only, preview_limit),
        "sem_int_only_units": sorted_preview(int_only, preview_limit),
        "sem_token_base_only_units": sorted_preview(token_base_only, preview_limit),
        "sem_phrase_base_only_units": sorted_preview(phrase_base_only, preview_limit),
    }
    add_trace_copy(row, baseline_trace, "sem_trace_base", trace_cols)
    add_trace_copy(row, intervention_trace, "sem_trace_int", trace_cols)
    for col in trace_cols:
        b_val = safe_float(baseline_trace.get(col))
        i_val = safe_float(intervention_trace.get(col))
        if b_val is not None and i_val is not None:
            row[f"sem_trace_delta_{col}_base_minus_int"] = b_val - i_val
    add_v53_aliases(row, baseline_trace, token_base_only)
    add_v54_visual_aliases(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CHAIR-free semantic caption-pair features for generative routing."
    )
    parser.add_argument("--baseline_pred_jsonl", required=True)
    parser.add_argument("--intervention_pred_jsonl", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary_json", required=True)
    parser.add_argument("--baseline_pred_text_key", default="auto")
    parser.add_argument("--intervention_pred_text_key", default="auto")
    parser.add_argument("--baseline_trace_csv", default="")
    parser.add_argument("--intervention_trace_csv", default="")
    parser.add_argument("--trace_col", action="append", default=[])
    parser.add_argument("--preview_limit", type=int, default=80)
    args = parser.parse_args()

    baseline = read_prediction_map(os.path.abspath(args.baseline_pred_jsonl), args.baseline_pred_text_key)
    intervention = read_prediction_map(os.path.abspath(args.intervention_pred_jsonl), args.intervention_pred_text_key)
    baseline_trace = load_trace_rows(args.baseline_trace_csv)
    intervention_trace = load_trace_rows(args.intervention_trace_csv)
    trace_cols = args.trace_col or [
        "probe_stop_eos_margin_real",
        "probe_stop_eos_logprob_real",
        "probe_lp_content_min_real",
        "probe_lp_content_mean_real",
        "probe_target_gap_content_min_real",
        "probe_entropy_content_max_real",
        "probe_tail_after_last_object_entropy_max_real",
        "probe_tail_after_last_object_gap_min_real",
        "probe_tail_after_last_object_lp_min_real",
        "probe_lp_tail_minus_head_real",
        "probe_gap_tail_minus_head_real",
        "probe_vis_attn_content_mean_real",
        "probe_vis_attn_content_min_real",
        "probe_vis_attn_tail_mean_real",
        "probe_vis_attn_last4_mean_real",
        "probe_vis_attn_tail_minus_head_real",
        "probe_vis_attn_topk_content_mean_real",
        "probe_vis_attn_topk_tail_minus_head_real",
        "probe_vis_entropy_content_mean_real",
        "probe_vis_entropy_content_max_real",
        "probe_vis_entropy_tail_minus_head_real",
        "probe_vis_top1_mass_content_mean_real",
        "probe_vis_ess_content_mean_real",
    ]

    ids = sorted(set(baseline) & set(intervention), key=lambda x: int(x) if x.isdigit() else x)
    rows = [
        build_row(
            sid,
            baseline[sid],
            intervention[sid],
            baseline_trace=baseline_trace.get(sid, {}),
            intervention_trace=intervention_trace.get(sid, {}),
            trace_cols=trace_cols,
            preview_limit=int(args.preview_limit),
        )
        for sid in ids
    ]

    write_csv(os.path.abspath(args.out_csv), rows)
    summary = {
        "inputs": {
            "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "baseline_trace_csv": os.path.abspath(args.baseline_trace_csv) if args.baseline_trace_csv else "",
            "intervention_trace_csv": os.path.abspath(args.intervention_trace_csv) if args.intervention_trace_csv else "",
            "uses_chair_parser": False,
            "uses_coco_ontology": False,
            "uses_mscoco_generated_words": False,
        },
        "counts": {
            "n_baseline": len(baseline),
            "n_intervention": len(intervention),
            "n_rows": len(rows),
            "n_baseline_trace": len(baseline_trace),
            "n_intervention_trace": len(intervention_trace),
        },
        "outputs": {
            "features_csv": os.path.abspath(args.out_csv),
            "summary_json": os.path.abspath(args.out_summary_json),
        },
    }
    write_json(os.path.abspath(args.out_summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
