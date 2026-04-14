#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from extract_generative_claim_support_delta_features import replay_claim_metrics, zero_replay_metrics
from extract_generative_semantic_pairwise_features import (
    GENERIC_FILLERS,
    STOPWORDS,
    content_tokens,
    normalize_token,
    read_prediction_map,
    semantic_units,
    unit_summary,
)
from frgavr_cleanroom.runtime import CleanroomLlavaRuntime, load_question_rows, parse_bool, safe_id


BAD_SINGLE_UNITS = {
    "left",
    "right",
    "center",
    "middle",
    "side",
    "top",
    "bottom",
    "front",
    "back",
    "nearby",
    "possible",
    "possibly",
    "likely",
    "item",
    "items",
    "object",
    "objects",
    "thing",
    "things",
    "view",
    "way",
    "group",
    "groups",
    "number",
    "numbers",
    "lot",
    "lots",
    "pair",
    "pairs",
    "set",
    "sets",
}

ABSTRACT_OR_EVAL_IRRELEVANT_UNITS = {
    "action",
    "activity",
    "addition",
    "appearance",
    "area",
    "atmosphere",
    "background",
    "body",
    "break",
    "color",
    "comfort",
    "company",
    "direction",
    "edge",
    "environment",
    "expertise",
    "fixture",
    "focus",
    "foreground",
    "game",
    "journey",
    "landscape",
    "location",
    "moment",
    "other",
    "part",
    "piece",
    "portion",
    "position",
    "process",
    "progress",
    "project",
    "scene",
    "setting",
    "size",
    "skill",
    "space",
    "standing",
    "style",
    "talent",
    "task",
    "time",
    "total",
    "touch",
    "unity",
    "use",
    "variety",
    "warmth",
}

OBJECT_CONTEXT_TERMS = {
    "also",
    "addition",
    "behind",
    "beside",
    "background",
    "foreground",
    "front",
    "located",
    "near",
    "nearby",
    "next",
    "present",
    "right",
    "left",
    "seen",
    "side",
    "visible",
    "with",
}

PERSON_ALIASES = {
    "adult",
    "bicyclist",
    "boy",
    "child",
    "chef",
    "girl",
    "kid",
    "man",
    "men",
    "passenger",
    "pedestrian",
    "people",
    "person",
    "player",
    "rider",
    "skier",
    "snowboarder",
    "surfer",
    "teammate",
    "woman",
    "women",
    "worker",
}

SPACY_NLP: Any = None


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


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def norm_id(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        return str(int(raw))
    except Exception:
        return raw


def load_feature_map(path: str) -> Dict[str, Dict[str, str]]:
    if not str(path or "").strip():
        return {}
    rows = read_csv_rows(os.path.abspath(path))
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sid = norm_id(row.get("id") or row.get("image_id") or row.get("question_id"))
        if sid:
            out[sid] = dict(row)
    return out


def sigmoid(value: float) -> float:
    x = max(-50.0, min(50.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-x)))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def min_or_zero(values: Sequence[float]) -> float:
    return float(min(values)) if values else 0.0


def max_or_zero(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def topk_sum(values: Sequence[float], k: int) -> float:
    return float(sum(sorted((float(x) for x in values), reverse=True)[: max(0, int(k))]))


def unit_display(unit: str) -> str:
    return str(unit).replace("_", " ").strip()


def keep_unit(unit: str) -> bool:
    parts = [part for part in str(unit).split("_") if part]
    if not parts:
        return False
    if len(parts) == 1:
        part = parts[0]
        if part in BAD_SINGLE_UNITS or part in GENERIC_FILLERS:
            return False
        return len(part) >= 3
    return any(part not in BAD_SINGLE_UNITS and part not in GENERIC_FILLERS and len(part) >= 3 for part in parts)


def candidate_head(unit: str) -> str:
    parts = [part for part in str(unit).split("_") if part]
    return parts[-1] if parts else ""


def normalize_candidate_alias(unit: str) -> str:
    parts = [part for part in str(unit).split("_") if part]
    if not parts:
        return ""
    if parts[-1] in PERSON_ALIASES:
        return "person"
    if str(unit) in {"sport_ball", "sports_ball", "tenni_ball"}:
        return "sports_ball"
    if str(unit) in {"cell_phone", "phone"}:
        return "cell_phone"
    if str(unit) in {"remote_controller", "controller"}:
        return "remote"
    return str(unit)


def keep_object_priority_unit(unit: str) -> bool:
    if not keep_unit(unit):
        return False
    parts = [part for part in str(unit).split("_") if part]
    if not parts:
        return False
    head = parts[-1]
    if head in ABSTRACT_OR_EVAL_IRRELEVANT_UNITS:
        return False
    if all(part in ABSTRACT_OR_EVAL_IRRELEVANT_UNITS or part in GENERIC_FILLERS for part in parts):
        return False
    return True


def get_spacy_nlp(model_name: str) -> Any:
    global SPACY_NLP
    if SPACY_NLP is None:
        import spacy

        SPACY_NLP = spacy.load(str(model_name))
    return SPACY_NLP


def spacy_token_unit(token: Any) -> str:
    raw = str(getattr(token, "lemma_", "") or getattr(token, "text", "") or "").strip().lower()
    if raw == "-pron-":
        raw = str(getattr(token, "text", "") or "").strip().lower()
    raw = re.sub(r"[^a-z0-9']+", "", raw)
    return normalize_token(raw) if raw else ""


def keep_spacy_content_token(token: Any) -> bool:
    unit = spacy_token_unit(token)
    if not keep_unit(unit):
        return False
    if unit in STOPWORDS or unit in BAD_SINGLE_UNITS or unit in GENERIC_FILLERS:
        return False
    if bool(getattr(token, "is_stop", False)) or bool(getattr(token, "is_punct", False)):
        return False
    pos = str(getattr(token, "pos_", ""))
    dep = str(getattr(token, "dep_", ""))
    return pos in {"NOUN", "PROPN", "ADJ"} or dep in {"compound", "amod"}


def spacy_caption_units(text: str, *, spacy_model: str) -> List[Tuple[int, int, str]]:
    """Return object-like noun phrase/head candidates as (position, priority, unit)."""
    nlp = get_spacy_nlp(spacy_model)
    doc = nlp(str(text or ""))
    candidates: List[Tuple[int, int, str]] = []
    covered_token_idxs: set[int] = set()

    for chunk in doc.noun_chunks:
        content = [tok for tok in chunk if keep_spacy_content_token(tok)]
        covered_token_idxs.update(int(tok.i) for tok in chunk)
        words = [spacy_token_unit(tok) for tok in content]
        words = [word for word in words if keep_unit(word)]
        if len(words) >= 2:
            phrase = "_".join(words)
            if keep_unit(phrase):
                candidates.append((int(chunk.start), 0, phrase))

        root = getattr(chunk, "root", None)
        if root is not None and str(getattr(root, "pos_", "")) in {"NOUN", "PROPN"}:
            head = spacy_token_unit(root)
            if keep_unit(head):
                candidates.append((int(root.i), 1, head))

        for tok in content:
            if str(getattr(tok, "pos_", "")) in {"NOUN", "PROPN"}:
                unit = spacy_token_unit(tok)
                if keep_unit(unit):
                    candidates.append((int(tok.i), 2, unit))

    for tok in doc:
        if int(tok.i) in covered_token_idxs:
            continue
        if str(getattr(tok, "pos_", "")) in {"NOUN", "PROPN"}:
            unit = spacy_token_unit(tok)
            if keep_unit(unit):
                candidates.append((int(tok.i), 3, unit))

    return candidates


def object_priority_window_score(doc: Any, token_index: int) -> float:
    start = max(0, int(token_index) - 10)
    end = min(len(doc), int(token_index) + 4)
    words = {spacy_token_unit(tok) for tok in doc[start:end]}
    score = 0.0
    if words & OBJECT_CONTEXT_TERMS:
        score += 2.0
    if {"there", "are"} <= words or {"there", "is"} <= words:
        score += 2.0
    if {"in", "addition"} <= words:
        score += 2.0
    return score


def spacy_object_priority_candidates(text: str, *, spacy_model: str) -> List[Tuple[float, int, int, str]]:
    nlp = get_spacy_nlp(spacy_model)
    doc = nlp(str(text or ""))
    candidates: List[Tuple[float, int, int, str]] = []
    covered_token_idxs: set[int] = set()
    denom = max(1, len(doc) - 1)

    def add_candidate(unit: str, pos: int, priority: int, root: Any) -> None:
        unit = normalize_candidate_alias(unit)
        if not keep_object_priority_unit(unit):
            return
        head = candidate_head(unit)
        score = 0.0
        score += object_priority_window_score(doc, pos)
        score += 1.5 * float(pos) / float(denom)
        score += 0.6 if "_" in unit else 0.0
        score += 0.5 if head and head not in ABSTRACT_OR_EVAL_IRRELEVANT_UNITS else 0.0
        dep = str(getattr(root, "dep_", ""))
        if dep in {"pobj", "dobj", "nsubj", "conj", "attr", "appos"}:
            score += 0.4
        candidates.append((score, int(pos), int(priority), unit))

    for chunk in doc.noun_chunks:
        content = [tok for tok in chunk if keep_spacy_content_token(tok)]
        covered_token_idxs.update(int(tok.i) for tok in chunk)
        root = getattr(chunk, "root", None)
        if root is None:
            continue

        root_unit = normalize_candidate_alias(spacy_token_unit(root))
        words = [spacy_token_unit(tok) for tok in content]
        words = [word for word in words if keep_object_priority_unit(normalize_candidate_alias(word))]
        if len(words) >= 2:
            phrase = normalize_candidate_alias("_".join(words))
            add_candidate(phrase, int(chunk.start), 0, root)

        if str(getattr(root, "pos_", "")) in {"NOUN", "PROPN"}:
            add_candidate(root_unit, int(root.i), 1, root)

        for tok in content:
            if str(getattr(tok, "pos_", "")) in {"NOUN", "PROPN"}:
                add_candidate(spacy_token_unit(tok), int(tok.i), 2, tok)

    for tok in doc:
        if int(tok.i) in covered_token_idxs:
            continue
        if str(getattr(tok, "pos_", "")) in {"NOUN", "PROPN"}:
            add_candidate(spacy_token_unit(tok), int(tok.i), 3, tok)

    return candidates


def intervention_unit_blocklist(units: Sequence[Tuple[int, int, str]]) -> set[str]:
    out: set[str] = set()
    for _, _, unit in units:
        parts = [part for part in str(unit).split("_") if part]
        out.add(str(unit))
        if parts:
            out.add(parts[-1])
        for part in parts:
            if keep_unit(part):
                out.add(part)
    return out


def candidate_is_base_only(unit: str, intervention_blocklist: set[str]) -> bool:
    parts = [part for part in str(unit).split("_") if part]
    if str(unit) in intervention_blocklist:
        return False
    if len(parts) > 1 and parts[-1] in intervention_blocklist:
        return False
    if len(parts) == 1 and parts[0] in intervention_blocklist:
        return False
    return True


def candidate_is_base_only_object_priority(unit: str, intervention_blocklist: set[str]) -> bool:
    if not candidate_is_base_only(unit, intervention_blocklist):
        return False
    head = candidate_head(unit)
    if head and head in intervention_blocklist:
        return False
    return True


def ordered_base_only_units(
    baseline_text: str,
    intervention_text: str,
    *,
    max_units: int,
    include_phrases: bool,
    include_tokens: bool,
) -> List[str]:
    int_units = unit_summary(intervention_text)["unique_all_units"]
    toks = content_tokens(baseline_text)
    candidates: List[Tuple[int, int, str]] = []

    if include_phrases:
        for idx in range(len(toks) - 1):
            left, left_pos = toks[idx]
            right, right_pos = toks[idx + 1]
            if right_pos - left_pos <= 3 and left != right:
                unit = f"{left}_{right}"
                if unit not in int_units and keep_unit(unit):
                    # Phrases are queried before same-position unigrams.
                    candidates.append((left_pos, 0, unit))

    if include_tokens:
        token_units, _ = semantic_units(toks)
        for (tok, pos), unit in zip(toks, token_units):
            if unit not in int_units and keep_unit(unit):
                candidates.append((pos, 1, unit))

    out: List[str] = []
    seen = set()
    for _, _, unit in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
        if unit in seen:
            continue
        seen.add(unit)
        out.append(unit)
        if len(out) >= int(max_units):
            break
    return out


def ordered_base_only_spacy_units(
    baseline_text: str,
    intervention_text: str,
    *,
    max_units: int,
    spacy_model: str,
) -> List[str]:
    int_block = intervention_unit_blocklist(spacy_caption_units(intervention_text, spacy_model=spacy_model))
    candidates = spacy_caption_units(baseline_text, spacy_model=spacy_model)
    out: List[str] = []
    seen = set()
    for _, _, unit in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
        if unit in seen or not candidate_is_base_only(unit, int_block):
            continue
        seen.add(unit)
        out.append(unit)
        if len(out) >= int(max_units):
            break
    return out


def ordered_base_only_spacy_object_priority_units(
    baseline_text: str,
    intervention_text: str,
    *,
    max_units: int,
    spacy_model: str,
) -> List[str]:
    int_units = [(pos, priority, unit) for _, pos, priority, unit in spacy_object_priority_candidates(intervention_text, spacy_model=spacy_model)]
    int_block = intervention_unit_blocklist(int_units)
    candidates = spacy_object_priority_candidates(baseline_text, spacy_model=spacy_model)
    out: List[str] = []
    seen = set()
    for score, pos, priority, unit in sorted(candidates, key=lambda item: (-item[0], -item[1], item[2], item[3])):
        if unit in seen or not candidate_is_base_only_object_priority(unit, int_block):
            continue
        seen.add(unit)
        out.append(unit)
        if len(out) >= int(max_units):
            break
    return out


def yesno_metrics(
    *,
    runtime: CleanroomLlavaRuntime,
    image: Any,
    unit: str,
    question_template: str,
    yes_text: str,
    no_text: str,
    score_mode: str,
    cache: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    phrase = unit_display(unit)
    question = str(question_template).replace("{unit}", phrase).replace("{object}", phrase)
    cache_key = f"{score_mode}\t{question}\t{yes_text}\t{no_text}"
    if cache_key in cache:
        return dict(cache[cache_key])

    try:
        yes = replay_claim_metrics(runtime, image=image, question=question, claim_text=str(yes_text))
    except Exception:
        yes = zero_replay_metrics()

    if str(score_mode) == "yes_only":
        no = zero_replay_metrics()
    else:
        try:
            no = replay_claim_metrics(runtime, image=image, question=question, claim_text=str(no_text))
        except Exception:
            no = zero_replay_metrics()

    yes_lp = float(yes.get("replay_lp_mean", 0.0))
    no_lp = float(no.get("replay_lp_mean", 0.0))
    yes_gap = float(yes.get("replay_gap_mean", 0.0))
    no_gap = float(no.get("replay_gap_mean", 0.0))
    yes_ent = float(yes.get("replay_ent_mean", 0.0))
    no_ent = float(no.get("replay_ent_mean", 0.0))
    margin = float(yes_lp - no_lp)
    out = {
        "yesno_yes_lp": yes_lp,
        "yesno_no_lp": no_lp,
        "yesno_lp_margin": margin,
        "yesno_yes_gap": yes_gap,
        "yesno_no_gap": no_gap,
        "yesno_gap_margin": float(yes_gap - no_gap),
        "yesno_yes_ent": yes_ent,
        "yesno_no_ent": no_ent,
        "yesno_ent_margin": float(no_ent - yes_ent),
        "yesno_prob": sigmoid(margin),
        "yesno_risk": float(1.0 - sigmoid(margin)),
        "yesno_error": float(max(float(yes.get("replay_error", 0.0)), float(no.get("replay_error", 0.0)))),
    }
    cache[cache_key] = dict(out)
    return out


def add_verify_stats(out: Dict[str, Any], items: Sequence[Dict[str, Any]]) -> None:
    probs = [float(item.get("yesno_prob", 0.0)) for item in items]
    risks = [float(item.get("yesno_risk", 0.0)) for item in items]
    margins = [float(item.get("yesno_lp_margin", 0.0)) for item in items]
    gaps = [float(item.get("yesno_gap_margin", 0.0)) for item in items]
    yes_lps = [float(item.get("yesno_yes_lp", 0.0)) for item in items]
    ents = [float(item.get("yesno_yes_ent", 0.0)) for item in items]
    n = max(1, len(items))

    out.update(
        {
            "sem_verify_base_only_unit_count": int(len(items)),
            "sem_verify_base_only_unit_names": " | ".join(str(item.get("unit", "")) for item in items),
            "sem_verify_base_only_unit_details_json": json.dumps(list(items), ensure_ascii=False, sort_keys=True),
            "sem_benefit_verify_base_only_yes_prob_sum": float(sum(probs)),
            "sem_benefit_verify_base_only_yes_prob_mean": mean(probs),
            "sem_benefit_verify_base_only_yes_prob_min": min_or_zero(probs),
            "sem_benefit_verify_base_only_yes_prob_max": max_or_zero(probs),
            "sem_benefit_verify_base_only_yes_prob_top1": topk_sum(probs, 1),
            "sem_benefit_verify_base_only_yes_prob_top3": topk_sum(probs, 3),
            "sem_benefit_verify_base_only_yes_prob_gt050_count": int(sum(1 for p in probs if p > 0.50)),
            "sem_benefit_verify_base_only_yes_prob_gt060_count": int(sum(1 for p in probs if p > 0.60)),
            "sem_benefit_verify_base_only_yes_prob_gt070_count": int(sum(1 for p in probs if p > 0.70)),
            "sem_benefit_verify_base_only_yes_precision_gt050": float(sum(1 for p in probs if p > 0.50) / n),
            "sem_benefit_verify_base_only_yes_precision_gt060": float(sum(1 for p in probs if p > 0.60) / n),
            "sem_benefit_verify_base_only_margin_sum": float(sum(margins)),
            "sem_benefit_verify_base_only_margin_mean": mean(margins),
            "sem_benefit_verify_base_only_gap_margin_mean": mean(gaps),
            "sem_benefit_verify_base_only_yes_lp_mean": mean(yes_lps),
            "sem_benefit_verify_base_only_support_minus_risk": float(sum(probs) - sum(risks)),
            "sem_cost_verify_base_only_no_risk_sum": float(sum(risks)),
            "sem_cost_verify_base_only_no_risk_mean": mean(risks),
            "sem_cost_verify_base_only_no_risk_max": max_or_zero(risks),
            "sem_cost_verify_base_only_yes_prob_lt050_count": int(sum(1 for p in probs if p < 0.50)),
            "sem_cost_verify_base_only_yes_prob_lt040_count": int(sum(1 for p in probs if p < 0.40)),
            "sem_cost_verify_base_only_yes_prob_lt030_count": int(sum(1 for p in probs if p < 0.30)),
            "sem_cost_verify_base_only_yes_prob_lt050_rate": float(sum(1 for p in probs if p < 0.50) / n),
            "sem_cost_verify_base_only_yes_prob_lt040_rate": float(sum(1 for p in probs if p < 0.40) / n),
            "sem_cost_verify_base_only_yes_entropy_mean": mean(ents),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify baseline-only generic semantic units with an image-grounded yes/no replay."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--baseline_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--base_feature_csv", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_units", type=int, default=6)
    ap.add_argument(
        "--candidate_mode",
        choices=["semantic_units", "spacy_noun_chunks", "spacy_object_priority"],
        default="semantic_units",
    )
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--include_phrases", type=parse_bool, default=True)
    ap.add_argument("--include_tokens", type=parse_bool, default=True)
    ap.add_argument("--baseline_pred_text_key", default="auto")
    ap.add_argument("--intervention_pred_text_key", default="auto")
    ap.add_argument(
        "--question_template",
        default="Is the following visual detail supported by the image: {unit}? Answer yes or no.",
    )
    ap.add_argument("--yes_text", default="Yes")
    ap.add_argument("--no_text", default="No")
    ap.add_argument("--score_mode", choices=["yesno", "yes_only"], default="yesno")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    questions = load_question_rows(os.path.abspath(args.question_file), limit=int(args.limit))
    baseline = read_prediction_map(os.path.abspath(args.baseline_pred_jsonl), str(args.baseline_pred_text_key))
    intervention = read_prediction_map(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_text_key))
    base_features = load_feature_map(str(args.base_feature_csv))
    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=(str(args.model_base) or None),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_unit_probes = 0
    for idx, q in enumerate(questions):
        sid = norm_id(q.get("question_id") or q.get("id") or q.get("image_id"))
        image_name = str(q.get("image", "")).strip()
        row: Dict[str, Any] = dict(base_features.get(sid, {}))
        row.update({"id": sid, "image_id": sid, "image": image_name, "sem_verify_error": ""})
        try:
            if not sid or sid not in baseline or sid not in intervention:
                raise ValueError("Missing baseline/intervention prediction row.")
            image_path = os.path.join(str(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            b_text = str(baseline[sid].get("text", ""))
            i_text = str(intervention[sid].get("text", ""))
            if str(args.candidate_mode) == "spacy_object_priority":
                units = ordered_base_only_spacy_object_priority_units(
                    b_text,
                    i_text,
                    max_units=int(args.max_units),
                    spacy_model=str(args.spacy_model),
                )
            elif str(args.candidate_mode) == "spacy_noun_chunks":
                units = ordered_base_only_spacy_units(
                    b_text,
                    i_text,
                    max_units=int(args.max_units),
                    spacy_model=str(args.spacy_model),
                )
            else:
                units = ordered_base_only_units(
                    b_text,
                    i_text,
                    max_units=int(args.max_units),
                    include_phrases=bool(args.include_phrases),
                    include_tokens=bool(args.include_tokens),
                )
            image = runtime.load_image(image_path)
            cache: Dict[str, Dict[str, float]] = {}
            scored: List[Dict[str, Any]] = []
            for unit in units:
                metrics = yesno_metrics(
                    runtime=runtime,
                    image=image,
                    unit=unit,
                    question_template=str(args.question_template),
                    yes_text=str(args.yes_text),
                    no_text=str(args.no_text),
                    score_mode=str(args.score_mode),
                    cache=cache,
                )
                scored.append({"unit": unit_display(unit), "unit_key": unit, **metrics})
            n_unit_probes += len(scored)
            add_verify_stats(row, scored)
        except Exception as exc:
            n_errors += 1
            row["sem_verify_error"] = str(exc)
            add_verify_stats(row, [])
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[semantic-unit-yesno] {idx + 1}/{len(questions)} unit_probes={n_unit_probes}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")
    if str(args.out_summary_json or "").strip():
        feature_keys = [key for key in rows[0] if key.startswith("sem_verify_") or key.startswith("sem_benefit_verify_") or key.startswith("sem_cost_verify_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "baseline_pred_jsonl": os.path.abspath(args.baseline_pred_jsonl),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "base_feature_csv": os.path.abspath(args.base_feature_csv) if str(args.base_feature_csv or "").strip() else "",
                    "model_path": str(args.model_path),
                    "model_base": str(args.model_base),
                    "conv_mode": str(args.conv_mode),
                    "device": str(args.device),
                    "limit": int(args.limit),
                    "max_units": int(args.max_units),
                    "candidate_mode": str(args.candidate_mode),
                    "spacy_model": str(args.spacy_model),
                    "question_template": str(args.question_template),
                    "score_mode": str(args.score_mode),
                },
                "counts": {
                    "n_rows": len(rows),
                    "n_errors": n_errors,
                    "n_unit_probes": n_unit_probes,
                    "n_forward_passes_est": n_unit_probes * (1 if str(args.score_mode) == "yes_only" else 2),
                    "n_features": len(feature_keys),
                },
                "feature_keys": feature_keys,
                "outputs": {"out_csv": os.path.abspath(args.out_csv)},
            },
        )


if __name__ == "__main__":
    main()
