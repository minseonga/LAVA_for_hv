#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from extract_vga_generative_mention_features import (
    build_feature_payload,
    max_or_zero,
    mean_or_zero,
    min_or_zero,
    is_object_word,
    pick,
)
from frgavr_cleanroom.runtime import (
    CleanroomLlavaRuntime,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    safe_id,
    select_content_indices,
    write_csv,
    write_json,
)


WORD_RE = re.compile(r"[A-Za-z0-9]+")
LEADING_DETERMINERS = {"a", "an", "the"}
CLAIM_TYPES_ALL = ("object", "relation", "count")


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out


def tokenize_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(str(text or ""))]


def mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def percentile_scores(values: Sequence[float], *, higher_better: bool) -> List[float]:
    seq = [float(v) for v in values]
    n = len(seq)
    if n <= 1:
        return [1.0 for _ in seq]
    indexed = list(enumerate(seq))
    indexed.sort(key=lambda x: x[1], reverse=bool(higher_better))
    out = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_pos = (float(i) + float(j - 1)) / 2.0
        score = 1.0 - float(avg_pos / float(max(1, n - 1)))
        for k in range(i, j):
            out[indexed[k][0]] = float(score)
        i = j
    return out


def normalize_claim_phrase(text: str) -> str:
    toks = tokenize_words(text)
    while toks and toks[0] in LEADING_DETERMINERS:
        toks = toks[1:]
    return " ".join(str(tok) for tok in toks)


def claim_type_for_mention(mention_row: Dict[str, Any]) -> str:
    kinds = {str(x).strip() for x in str(mention_row.get("kinds", "")).split("|") if str(x).strip()}
    if {"object_mention", "noun_phrase"} & kinds:
        return "object"
    if "relation_phrase" in kinds:
        return "relation"
    if "count_phrase" in kinds:
        return "count"
    return ""


def claim_key_for_mention(
    mention_row: Dict[str, Any],
    *,
    allowed_types: Optional[Sequence[str]] = None,
) -> str:
    claim_type = claim_type_for_mention(mention_row)
    if not claim_type:
        return ""
    if allowed_types is not None and claim_type not in {str(x) for x in allowed_types}:
        return ""
    phrase = normalize_claim_phrase(str(mention_row.get("text", "")))
    if not phrase:
        return ""
    if claim_type == "object":
        toks = tokenize_words(phrase)
        if not toks or not any(is_object_word(tok) for tok in toks):
            return ""
    return f"{claim_type}::{phrase}"


def zero_replay_metrics() -> Dict[str, float]:
    return {
        "replay_lp_mean": 0.0,
        "replay_lp_min": 0.0,
        "replay_gap_mean": 0.0,
        "replay_gap_min": 0.0,
        "replay_ent_mean": 0.0,
        "replay_ent_max": 0.0,
        "replay_argmax_mean": 0.0,
        "replay_n_content_tokens": 0.0,
        "replay_error": 1.0,
    }


def replay_claim_metrics(
    runtime: CleanroomLlavaRuntime,
    image: Any,
    question: str,
    claim_text: str,
) -> Dict[str, float]:
    text = str(claim_text or "").strip()
    if not text:
        return zero_replay_metrics()

    pack = runtime.teacher_force_candidate(
        image=image,
        question=question,
        candidate_text=text,
        output_attentions=False,
    )
    content_indices = select_content_indices(runtime.tokenizer, pack.cont_ids)

    logits = pack.logits.to(torch.float32)
    decision_positions = pack.decision_positions.long()
    target_ids = pack.labels_exp[pack.cont_label_positions.long()].long()

    token_logits = logits[decision_positions]
    log_probs = F.log_softmax(token_logits, dim=-1)
    probs = torch.softmax(token_logits, dim=-1)
    token_ent = -(probs * log_probs).sum(dim=-1)
    target_lp = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

    top2_vals, top2_idx = torch.topk(token_logits, k=2, dim=-1)
    top1_logit = top2_vals[:, 0]
    top2_logit = top2_vals[:, 1]
    top1_id = top2_idx[:, 0]
    target_logit = token_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    best_other_logit = torch.where(top1_id == target_ids, top2_logit, top1_logit)
    target_gap = target_logit - best_other_logit
    target_is_argmax = (top1_id == target_ids).to(torch.float32)

    all_indices = list(range(int(target_ids.numel())))
    pick_indices = [int(x) for x in content_indices] if content_indices else all_indices
    lp_vals = pick([float(x.item()) for x in target_lp], pick_indices)
    gap_vals = pick([float(x.item()) for x in target_gap], pick_indices)
    ent_vals = pick([float(x.item()) for x in token_ent], pick_indices)
    argmax_vals = pick([float(x.item()) for x in target_is_argmax], pick_indices)

    return {
        "replay_lp_mean": float(mean_or_zero(lp_vals)),
        "replay_lp_min": float(min_or_zero(lp_vals)),
        "replay_gap_mean": float(mean_or_zero(gap_vals)),
        "replay_gap_min": float(min_or_zero(gap_vals)),
        "replay_ent_mean": float(mean_or_zero(ent_vals)),
        "replay_ent_max": float(max_or_zero(ent_vals)),
        "replay_argmax_mean": float(mean_or_zero(argmax_vals)),
        "replay_n_content_tokens": float(len(pick_indices)),
        "replay_error": 0.0,
    }


def attach_replay_support_scores(
    claims: Dict[str, Dict[str, Any]],
    runtime: CleanroomLlavaRuntime,
    image: Any,
    question: str,
    support_cache: Dict[str, Dict[str, float]],
) -> None:
    if not claims:
        return

    keys = list(claims.keys())
    raw_metrics: List[Dict[str, float]] = []
    for key in keys:
        replay_text = str(claims[key].get("replay_text", "")).strip()
        metrics = support_cache.get(replay_text)
        if metrics is None:
            try:
                metrics = replay_claim_metrics(runtime, image=image, question=question, claim_text=replay_text)
            except Exception:
                metrics = zero_replay_metrics()
            support_cache[replay_text] = dict(metrics)
        raw_metrics.append(dict(metrics))

    lp_mean_pct = percentile_scores([float(m["replay_lp_mean"]) for m in raw_metrics], higher_better=True)
    lp_min_pct = percentile_scores([float(m["replay_lp_min"]) for m in raw_metrics], higher_better=True)
    gap_mean_pct = percentile_scores([float(m["replay_gap_mean"]) for m in raw_metrics], higher_better=True)
    gap_min_pct = percentile_scores([float(m["replay_gap_min"]) for m in raw_metrics], higher_better=True)
    ent_mean_pct = percentile_scores([float(m["replay_ent_mean"]) for m in raw_metrics], higher_better=False)
    ent_max_pct = percentile_scores([float(m["replay_ent_max"]) for m in raw_metrics], higher_better=False)
    argmax_pct = percentile_scores([float(m["replay_argmax_mean"]) for m in raw_metrics], higher_better=True)

    for idx, key in enumerate(keys):
        item = claims[key]
        metrics = raw_metrics[idx]
        item.update(metrics)
        item["support_lp_mean_pct"] = float(lp_mean_pct[idx])
        item["support_lp_min_pct"] = float(lp_min_pct[idx])
        item["support_gap_mean_pct"] = float(gap_mean_pct[idx])
        item["support_gap_min_pct"] = float(gap_min_pct[idx])
        item["support_ent_mean_pct"] = float(ent_mean_pct[idx])
        item["support_ent_max_pct"] = float(ent_max_pct[idx])
        item["support_argmax_pct"] = float(argmax_pct[idx])
        item["support_score"] = float(
            mean(
                [
                    lp_mean_pct[idx],
                    lp_min_pct[idx],
                    gap_mean_pct[idx],
                    gap_min_pct[idx],
                    ent_mean_pct[idx],
                    ent_max_pct[idx],
                    argmax_pct[idx],
                ]
            )
        )


def claim_map_from_payload(
    payload: Dict[str, Any],
    *,
    runtime: CleanroomLlavaRuntime,
    image: Any,
    question: str,
    support_cache: Dict[str, Dict[str, float]],
    allowed_types: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    mention_rows = [dict(row) for row in payload.get("mention_rows", [])]
    max_content_idx = int(payload.get("max_content_idx", 0))
    out: Dict[str, Dict[str, Any]] = {}
    for row in mention_rows:
        key = claim_key_for_mention(row, allowed_types=allowed_types)
        if not key:
            continue
        claim_type, replay_text = key.split("::", 1)
        replay_text = str(replay_text).strip()
        text = normalize_claim_phrase(str(row.get("text", ""))) or replay_text
        item = {
            "key": key,
            "claim_type": claim_type,
            "text": text,
            "replay_text": replay_text,
            "first_idx": int(row.get("first_idx", 0)),
            "last_idx": int(row.get("last_idx", 0)),
            "last_pos_frac": float(safe_div(float(int(row.get("last_idx", 0))), float(max(1, max_content_idx)))),
            "n_occurrences": 1,
        }
        prev = out.get(key)
        if prev is None:
            out[key] = item
            continue
        prev["first_idx"] = int(min(int(prev["first_idx"]), int(item["first_idx"])))
        prev["last_idx"] = int(max(int(prev["last_idx"]), int(item["last_idx"])))
        prev["last_pos_frac"] = float(max(float(prev["last_pos_frac"]), float(item["last_pos_frac"])))
        prev["n_occurrences"] = int(prev.get("n_occurrences", 1)) + 1
        if len(str(item["text"])) > len(str(prev.get("text", ""))):
            prev["text"] = str(item["text"])
            prev["replay_text"] = str(item["replay_text"])

    attach_replay_support_scores(out, runtime=runtime, image=image, question=question, support_cache=support_cache)
    return out


def sum_vals(items: Sequence[float]) -> float:
    return float(sum(float(x) for x in items))


def max_or_zero(items: Sequence[float]) -> float:
    if not items:
        return 0.0
    return float(max(float(x) for x in items))


def count_ge(items: Sequence[float], threshold: float) -> int:
    return int(sum(1 for x in items if float(x) >= float(threshold)))


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def low_support_items(claims: Sequence[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    return [c for c in claims if float(c.get("support_score", 0.0)) <= float(threshold)]


def late_items(claims: Sequence[Dict[str, Any]], min_pos_frac: float) -> List[Dict[str, Any]]:
    return [c for c in claims if float(c.get("last_pos_frac", 0.0)) >= float(min_pos_frac)]


def repetition_density(claims: Sequence[Dict[str, Any]]) -> float:
    heads = []
    for claim in claims:
        toks = tokenize_words(str(claim.get("text", "")))
        if toks:
            heads.append(str(toks[-1]))
    if not heads:
        return 0.0
    uniq = len(set(heads))
    return float(safe_div(float(len(heads) - uniq), float(max(1, len(heads)))))


def filter_claims_by_type(claims: Dict[str, Dict[str, Any]], claim_type: str) -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in claims.items() if str(v.get("claim_type", "")) == str(claim_type)}


def support_mass(claims: Dict[str, Dict[str, Any]]) -> float:
    return float(sum(float(item.get("support_score", 0.0)) for item in claims.values()))


def support_weighted_recall(
    base_claims: Dict[str, Dict[str, Any]],
    int_claims: Dict[str, Dict[str, Any]],
) -> float:
    base_mass = support_mass(base_claims)
    if base_mass <= 0.0:
        return 1.0
    shared_mass = 0.0
    for key, base_item in base_claims.items():
        int_item = int_claims.get(key)
        if int_item is None:
            continue
        shared_mass += min(float(base_item.get("support_score", 0.0)), float(int_item.get("support_score", 0.0)))
    return clamp01(shared_mass / base_mass)


def strong_retention_rate(
    base_claims: Dict[str, Dict[str, Any]],
    int_claims: Dict[str, Dict[str, Any]],
    *,
    base_threshold: float = 0.75,
    retain_threshold: float = 0.5,
) -> float:
    strong_keys = [k for k, item in base_claims.items() if float(item.get("support_score", 0.0)) >= float(base_threshold)]
    if not strong_keys:
        return 1.0
    kept = 0
    for key in strong_keys:
        int_item = int_claims.get(key)
        if int_item is None:
            continue
        if float(int_item.get("support_score", 0.0)) >= float(retain_threshold):
            kept += 1
    return clamp01(safe_div(float(kept), float(len(strong_keys))))


def dropped_strong_support_count(
    base_claims: Dict[str, Dict[str, Any]],
    int_claims: Dict[str, Dict[str, Any]],
    *,
    base_threshold: float = 0.75,
) -> int:
    return int(
        sum(
            1
            for key, item in base_claims.items()
            if float(item.get("support_score", 0.0)) >= float(base_threshold) and key not in int_claims
        )
    )


def unsupported_add_summary(
    int_claims: Dict[str, Dict[str, Any]],
    base_claims: Dict[str, Dict[str, Any]],
    *,
    unsupported_threshold: float = 0.5,
) -> Dict[str, float]:
    added = [item for key, item in int_claims.items() if key not in base_claims]
    unsupported_mass = [float(1.0 - float(item.get("support_score", 0.0))) for item in added]
    unsupported_count = int(sum(1 for item in added if float(item.get("support_score", 0.0)) <= float(unsupported_threshold)))
    denom_claims = float(max(1, len(int_claims)))
    denom_added = float(max(1, len(added)))
    return {
        "count": int(len(added)),
        "unsupported_count": int(unsupported_count),
        "unsupported_count_rate": float(safe_div(float(unsupported_count), denom_claims)),
        "unsupported_add_rate": float(safe_div(float(len(added)), denom_claims)),
        "unsupported_mass": float(sum_vals(unsupported_mass)),
        "unsupported_mass_rate": float(safe_div(sum_vals(unsupported_mass), denom_claims)),
        "unsupported_mean": float(mean(unsupported_mass)),
        "unsupported_share_among_added": float(safe_div(float(unsupported_count), denom_added)),
    }


def degeneration_summary(
    base_payload: Dict[str, Any],
    int_payload: Dict[str, Any],
    base_all_claims: Dict[str, Dict[str, Any]],
    int_all_claims: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    claims = list(int_all_claims.values())
    low_claims = low_support_items(claims, 0.5)
    late_low_claims = low_support_items(late_items(claims, 0.5), 0.5)
    strong_claims = [c for c in claims if float(c.get("support_score", 0.0)) >= 0.75]
    last_strong_pos = max((float(c.get("last_pos_frac", 0.0)) for c in strong_claims), default=0.0)

    base_content = float(base_payload.get("probe_n_content_tokens", 0.0))
    int_content = float(int_payload.get("probe_n_content_tokens", 0.0))
    content_ratio = float(safe_div(int_content, float(max(1.0, base_content))))
    base_claim_count = float(max(1, len(base_all_claims)))
    int_claim_count = float(len(int_all_claims))
    claim_ratio = float(safe_div(int_claim_count, base_claim_count))

    tail_after_last_strong = float(max(0.0, 1.0 - last_strong_pos))
    late_low_rate = float(safe_div(float(len(late_low_claims)), float(max(1, len(claims)))))
    compression_deficit_content = float(max(0.0, 1.0 - min(1.0, content_ratio)))
    compression_deficit_claims = float(max(0.0, 1.0 - min(1.0, claim_ratio)))
    low_repetition = float(repetition_density(low_claims))

    s_degenerate = float(
        mean(
            [
                tail_after_last_strong,
                late_low_rate,
                compression_deficit_content,
                compression_deficit_claims,
                low_repetition,
            ]
        )
    )
    return {
        "content_ratio_int_vs_base": float(content_ratio),
        "claim_count_ratio_int_vs_base": float(claim_ratio),
        "compression_deficit_content": float(compression_deficit_content),
        "compression_deficit_claims": float(compression_deficit_claims),
        "tail_after_last_strong_support_claim_frac": float(tail_after_last_strong),
        "late_low_support_claim_rate": float(late_low_rate),
        "low_support_repetition_density": float(low_repetition),
        "s_degenerate": float(s_degenerate),
    }


def claim_delta_features(
    base_payload: Dict[str, Any],
    int_payload: Dict[str, Any],
    *,
    runtime: CleanroomLlavaRuntime,
    image: Any,
    question: str,
) -> Dict[str, Any]:
    support_cache: Dict[str, Dict[str, float]] = {}
    base_claims = claim_map_from_payload(
        base_payload,
        runtime=runtime,
        image=image,
        question=question,
        support_cache=support_cache,
        allowed_types=("object",),
    )
    int_claims = claim_map_from_payload(
        int_payload,
        runtime=runtime,
        image=image,
        question=question,
        support_cache=support_cache,
        allowed_types=("object",),
    )
    base_all_claims = claim_map_from_payload(
        base_payload,
        runtime=runtime,
        image=image,
        question=question,
        support_cache=support_cache,
        allowed_types=CLAIM_TYPES_ALL,
    )
    int_all_claims = claim_map_from_payload(
        int_payload,
        runtime=runtime,
        image=image,
        question=question,
        support_cache=support_cache,
        allowed_types=CLAIM_TYPES_ALL,
    )
    base_keys = set(base_claims.keys())
    int_keys = set(int_claims.keys())
    dropped = sorted(base_keys - int_keys)
    added = sorted(int_keys - base_keys)
    shared = sorted(base_keys & int_keys)

    dropped_supports = [float(base_claims[key]["support_score"]) for key in dropped]
    added_unsupported = [float(1.0 - float(int_claims[key]["support_score"])) for key in added]
    shared_weaken = [
        float(max(0.0, float(base_claims[key]["support_score"]) - float(int_claims[key]["support_score"])))
        for key in shared
    ]

    strong_dropped = [key for key in dropped if float(base_claims[key]["support_score"]) >= 0.5]
    strong_last_pos = max((float(base_claims[key]["last_pos_frac"]) for key in strong_dropped), default=0.0)
    any_last_pos = max((float(base_claims[key]["last_pos_frac"]) for key in dropped), default=0.0)

    preserve_recall = float(support_weighted_recall(base_all_claims, int_all_claims))
    strong_preserve = float(strong_retention_rate(base_all_claims, int_all_claims))
    dropped_strong_count = int(dropped_strong_support_count(base_all_claims, int_all_claims))
    base_all_mass = float(support_mass(base_all_claims))
    shared_preserved_mass = float(preserve_recall * base_all_mass)
    support_mass_drop = float(max(0.0, base_all_mass - shared_preserved_mass))
    dropped_strong_rate = float(safe_div(float(dropped_strong_count), float(max(1, sum(1 for item in base_all_claims.values() if float(item.get("support_score", 0.0)) >= 0.75)))))

    type_retention_rates: List[float] = []
    preserve_aux: Dict[str, Any] = {}
    add_aux: Dict[str, Any] = {}
    for claim_type in CLAIM_TYPES_ALL:
        base_t = filter_claims_by_type(base_all_claims, claim_type)
        int_t = filter_claims_by_type(int_all_claims, claim_type)
        retention_t = float(support_weighted_recall(base_t, int_t))
        type_retention_rates.append(retention_t)
        preserve_aux[f"pair_claimdelta_preserve_{claim_type}_support_recall"] = retention_t
        add_t = unsupported_add_summary(int_t, base_t)
        add_aux[f"pair_claimdelta_add_{claim_type}_unsupported_rate"] = float(add_t["unsupported_count_rate"])
        add_aux[f"pair_claimdelta_add_{claim_type}_unsupported_mass_rate"] = float(add_t["unsupported_mass_rate"])

    preserve_type_mean = float(mean(type_retention_rates)) if type_retention_rates else 1.0
    s_preserve = float(
        mean(
            [
                1.0 - preserve_recall,
                1.0 - strong_preserve,
                dropped_strong_rate,
                1.0 - preserve_type_mean,
            ]
        )
    )

    add_all = unsupported_add_summary(int_all_claims, base_all_claims)
    s_add = float(
        mean(
            [
                float(add_all["unsupported_count_rate"]),
                float(add_all["unsupported_mass_rate"]),
                float(add_aux.get("pair_claimdelta_add_relation_unsupported_rate", 0.0)),
                float(add_aux.get("pair_claimdelta_add_count_unsupported_rate", 0.0)),
            ]
        )
    )

    degenerate = degeneration_summary(base_payload, int_payload, base_all_claims, int_all_claims)

    out = {
        "pair_claimdelta_n_base_object_claims": int(len(base_keys)),
        "pair_claimdelta_n_int_object_claims": int(len(int_keys)),
        "pair_claimdelta_n_shared_object_claims": int(len(shared)),
        "pair_claimdelta_n_dropped_object_claims": int(len(dropped)),
        "pair_claimdelta_n_added_object_claims": int(len(added)),
        "pair_claimdelta_supported_drop_sum": float(sum_vals(dropped_supports)),
        "pair_claimdelta_supported_drop_mean": float(mean(dropped_supports)),
        "pair_claimdelta_supported_drop_max": float(max_or_zero(dropped_supports)),
        "pair_claimdelta_supported_drop_rate": float(safe_div(sum_vals(dropped_supports), float(max(1, len(base_keys))))),
        "pair_claimdelta_supported_drop_count_ge_050": int(count_ge(dropped_supports, 0.5)),
        "pair_claimdelta_supported_drop_count_ge_075": int(count_ge(dropped_supports, 0.75)),
        "pair_claimdelta_supported_drop_frac_ge_050": float(
            safe_div(float(count_ge(dropped_supports, 0.5)), float(max(1, len(base_keys))))
        ),
        "pair_claimdelta_supported_drop_frac_ge_075": float(
            safe_div(float(count_ge(dropped_supports, 0.75)), float(max(1, len(base_keys))))
        ),
        "pair_claimdelta_unsupported_add_sum": float(sum_vals(added_unsupported)),
        "pair_claimdelta_unsupported_add_mean": float(mean(added_unsupported)),
        "pair_claimdelta_unsupported_add_max": float(max_or_zero(added_unsupported)),
        "pair_claimdelta_unsupported_add_rate": float(safe_div(sum_vals(added_unsupported), float(max(1, len(int_keys))))),
        "pair_claimdelta_unsupported_add_count_ge_050": int(count_ge(added_unsupported, 0.5)),
        "pair_claimdelta_shared_weaken_sum": float(sum_vals(shared_weaken)),
        "pair_claimdelta_shared_weaken_mean": float(mean(shared_weaken)),
        "pair_claimdelta_shared_weaken_max": float(max_or_zero(shared_weaken)),
        "pair_claimdelta_shared_weaken_count_ge_025": int(count_ge(shared_weaken, 0.25)),
        "pair_claimdelta_shared_weaken_count_ge_050": int(count_ge(shared_weaken, 0.5)),
        "pair_claimdelta_last_supported_drop_pos_frac": float(strong_last_pos),
        "pair_claimdelta_tail_after_last_supported_drop_frac": float(max(0.0, 1.0 - strong_last_pos)),
        "pair_claimdelta_last_any_drop_pos_frac": float(any_last_pos),
        "pair_claimdelta_tail_after_last_any_drop_frac": float(max(0.0, 1.0 - any_last_pos)),
        "pair_claimdelta_preserve_support_mass_base": float(base_all_mass),
        "pair_claimdelta_preserve_support_mass_shared": float(shared_preserved_mass),
        "pair_claimdelta_preserve_support_mass_drop": float(support_mass_drop),
        "pair_claimdelta_preserve_support_mass_drop_rate": float(safe_div(support_mass_drop, float(max(1.0, base_all_mass)))),
        "pair_claimdelta_preserve_support_weighted_claim_recall": float(preserve_recall),
        "pair_claimdelta_preserve_strong_support_claim_retention_rate": float(strong_preserve),
        "pair_claimdelta_preserve_dropped_strong_support_claim_count": int(dropped_strong_count),
        "pair_claimdelta_preserve_dropped_strong_support_claim_rate": float(dropped_strong_rate),
        "pair_claimdelta_preserve_type_support_recall_mean": float(preserve_type_mean),
        "pair_claimdelta_add_unsupported_added_claim_count": int(add_all["unsupported_count"]),
        "pair_claimdelta_add_unsupported_added_claim_rate": float(add_all["unsupported_count_rate"]),
        "pair_claimdelta_add_unsupported_added_support_mass": float(add_all["unsupported_mass"]),
        "pair_claimdelta_add_unsupported_added_support_mass_rate": float(add_all["unsupported_mass_rate"]),
        "pair_claimdelta_add_unsupported_added_claim_share": float(add_all["unsupported_share_among_added"]),
        "pair_claimdelta_s_preserve": float(s_preserve),
        "pair_claimdelta_s_add": float(s_add),
        "pair_claimdelta_s_degenerate": float(degenerate["s_degenerate"]),
        "pair_claimdelta_dropped_claim_heads": " || ".join(str(base_claims[key]["text"]) for key in dropped[:8]),
        "pair_claimdelta_added_claim_heads": " || ".join(str(int_claims[key]["text"]) for key in added[:8]),
        "pair_claimdelta_support_cache_size": int(len(support_cache)),
        "pair_claimdelta_support_replay_errors": int(
            sum(int(float(item.get("replay_error", 0.0)) > 0.0) for item in base_all_claims.values())
            + sum(int(float(item.get("replay_error", 0.0)) > 0.0) for item in int_all_claims.values())
        ),
    }
    out.update(preserve_aux)
    out.update(add_aux)
    out.update(
        {
            "pair_claimdelta_degenerate_content_token_ratio_int_vs_base": float(degenerate["content_ratio_int_vs_base"]),
            "pair_claimdelta_degenerate_claim_count_ratio_int_vs_base": float(degenerate["claim_count_ratio_int_vs_base"]),
            "pair_claimdelta_degenerate_content_compression_deficit": float(degenerate["compression_deficit_content"]),
            "pair_claimdelta_degenerate_claim_compression_deficit": float(degenerate["compression_deficit_claims"]),
            "pair_claimdelta_degenerate_tail_after_last_strong_support_claim_frac": float(
                degenerate["tail_after_last_strong_support_claim_frac"]
            ),
            "pair_claimdelta_degenerate_late_low_support_claim_rate": float(degenerate["late_low_support_claim_rate"]),
            "pair_claimdelta_degenerate_low_support_repetition_density": float(
                degenerate["low_support_repetition_density"]
            ),
        }
    )
    return out


def load_feature_map(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not str(path or "").strip():
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = safe_id(row.get("id"))
            if sid:
                rows[sid] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract support-weighted claim delta features for generative fallback control.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--base_features_csv", type=str, default="")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_pred_text_key", type=str, default="auto")
    ap.add_argument("--intervention_pred_text_key", type=str, default="auto")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--max_mentions", type=int, default=12)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    baseline_map = load_prediction_text_map(args.baseline_pred_jsonl, text_key=args.baseline_pred_text_key)
    intervention_map = load_prediction_text_map(args.intervention_pred_jsonl, text_key=args.intervention_pred_text_key)
    base_feature_map = load_feature_map(args.base_features_csv)

    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(args.model_base or None),
        conv_mode=args.conv_mode,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_base_feature = 0
    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        baseline_text = str(baseline_map.get(sample_id, "")).strip()
        intervention_text = str(intervention_map.get(sample_id, "")).strip()
        image_path = os.path.join(args.image_folder, image_name)

        row: Dict[str, Any] = dict(base_feature_map.get(sample_id, {}))
        if not row and base_feature_map:
            n_missing_base_feature += 1
            row = {"id": sample_id, "image": image_name, "question": question}
        row["pair_claimdelta_error"] = ""

        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not question:
                raise ValueError("Missing question text.")
            if not baseline_text:
                raise ValueError("Missing baseline prediction text.")
            if not intervention_text:
                raise ValueError("Missing intervention prediction text.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = runtime.load_image(image_path)
            base_payload = build_feature_payload(
                runtime=runtime,
                image_path=image_path,
                question=question,
                candidate_text=baseline_text,
                sample_id=sample_id,
                image_name=image_name,
                image=image,
                max_mentions=int(args.max_mentions),
            )
            int_payload = build_feature_payload(
                runtime=runtime,
                image_path=image_path,
                question=question,
                candidate_text=intervention_text,
                sample_id=sample_id,
                image_name=image_name,
                image=image,
                max_mentions=int(args.max_mentions),
            )
            row.update(
                claim_delta_features(
                    base_payload,
                    int_payload,
                    runtime=runtime,
                    image=image,
                    question=question,
                )
            )
        except Exception as exc:
            n_errors += 1
            row["pair_claimdelta_error"] = str(exc)

        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[claim-delta] {idx + 1}/{len(question_rows)}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")

    if str(args.out_summary_json or "").strip():
        feature_keys = [k for k in rows[0].keys() if k.startswith("pair_claimdelta_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "base_features_csv": os.path.abspath(args.base_features_csv) if str(args.base_features_csv).strip() else "",
                    "model_path": args.model_path,
                    "model_base": args.model_base,
                    "conv_mode": args.conv_mode,
                    "device": args.device,
                },
                "counts": {
                    "n_questions": int(len(question_rows)),
                    "n_rows": int(len(rows)),
                    "n_claimdelta_features": int(len(feature_keys)),
                    "n_errors": int(n_errors),
                    "n_missing_base_feature_rows": int(n_missing_base_feature),
                },
                "settings": {
                    "claimdelta_mode": "isolated_claim_replay",
                },
                "outputs": {
                    "feature_csv": os.path.abspath(args.out_csv),
                },
            },
        )
        print(f"[saved] {os.path.abspath(args.out_summary_json)}")


if __name__ == "__main__":
    main()
