#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from analyze_caption_conditioned_object_extraction_proxy import object_matches, split_object_list, split_pipe_objects
from extract_generative_semantic_pairwise_features import read_prediction_map, write_csv, write_json
from extract_intervention_object_inventory_yesno_features import extract_mentioned_objects, parse_object_vocab
from frgavr_cleanroom.runtime import CleanroomLlavaRuntime, load_question_rows, parse_bool


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def norm_id(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def sigmoid(value: float, temperature: float = 1.0) -> float:
    z = max(-50.0, min(50.0, float(value) / max(float(temperature), 1e-6)))
    return float(1.0 / (1.0 + math.exp(-z)))


def ordered_unique(values: Iterable[str], *, max_items: int = 0) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if int(max_items) > 0 and len(out) >= int(max_items):
            break
    return out


def object_candidates_with_surfaces(
    raw_objects: Sequence[str],
    vocab: Sequence[str],
    *,
    filter_to_vocab: bool,
    max_items: int,
) -> List[Dict[str, Any]]:
    by_obj: Dict[str, Dict[str, Any]] = {}
    for raw in raw_objects:
        raw_text = str(raw or "").strip()
        if not raw_text:
            continue
        objects = extract_mentioned_objects(raw_text, vocab) if bool(filter_to_vocab) else [raw_text]
        for obj in objects:
            obj_text = str(obj or "").strip()
            if not obj_text:
                continue
            key = obj_text.lower()
            if key not in by_obj:
                by_obj[key] = {"object": obj_text, "surfaces": []}
            for surface in (raw_text, obj_text):
                if surface and surface not in by_obj[key]["surfaces"]:
                    by_obj[key]["surfaces"].append(surface)
            if int(max_items) > 0 and len(by_obj) >= int(max_items):
                return list(by_obj.values())
    return list(by_obj.values())


def token_variants(text: str) -> List[str]:
    base = str(text or "").strip()
    if not base:
        return []
    variants = [base, " " + base, base.capitalize(), " " + base.capitalize()]
    if not base.endswith("s"):
        variants += [base + "s", " " + base + "s"]
    if base.endswith("y") and len(base) > 1:
        variants += [base[:-1] + "ies", " " + base[:-1] + "ies"]
    out: List[str] = []
    seen: set[str] = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def content_token_ids(tokenizer: Any, text: str) -> List[int]:
    toks = tokenizer(str(text), add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    out: List[int] = []
    for token_id in toks:
        tid = int(token_id)
        if tid < 0 or tid in special:
            continue
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=True)
        except Exception:
            decoded = ""
        if str(decoded).strip():
            out.append(tid)
    return out


def find_subseq(seq: Sequence[int], pat: Sequence[int]) -> List[Tuple[int, int]]:
    if not seq or not pat or len(pat) > len(seq):
        return []
    out: List[Tuple[int, int]] = []
    n = len(pat)
    for idx in range(0, len(seq) - n + 1):
        if list(seq[idx : idx + n]) == list(pat):
            out.append((idx, idx + n))
    return out


def object_spans(tokenizer: Any, cont_ids: Sequence[int], obj: str, surfaces: Sequence[str]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    seen: set[Tuple[int, int]] = set()
    terms = ordered_unique([*surfaces, obj], max_items=0)
    for term in terms:
        for variant in token_variants(term):
            pat = content_token_ids(tokenizer, variant)
            for start, end in find_subseq(cont_ids, pat):
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                spans.append((start, end, term))
    return sorted(spans, key=lambda item: (item[0], item[1]))


def token_metrics(pack: Any, rel_idx: int) -> Dict[str, float]:
    target_id = int(pack.cont_ids[int(rel_idx)].item())
    decision_pos = int(pack.decision_positions[int(rel_idx)].item())
    vec = pack.logits[decision_pos].to(torch.float32)
    log_probs = F.log_softmax(vec, dim=-1)
    probs = torch.softmax(vec, dim=-1)
    entropy = float(-(probs * log_probs).sum().item())
    target_lp = float(log_probs[target_id].item())
    target_logit = float(vec[target_id].item())
    top2_vals, top2_idx = torch.topk(vec, k=2, dim=-1)
    top1_id = int(top2_idx[0].item())
    best_other = float(top2_vals[1].item() if top1_id == target_id else top2_vals[0].item())
    gap = float(target_logit - best_other)
    rank = int((vec > float(target_logit)).sum().item() + 1)
    return {
        "lp": target_lp,
        "gap": gap,
        "entropy": entropy,
        "rank": float(rank),
        "top1": float(rank == 1),
    }


def summarize_object(
    obj: str,
    surfaces: Sequence[str],
    spans: Sequence[Tuple[int, int, str]],
    pack: Any,
) -> Dict[str, Any]:
    metrics: List[Dict[str, float]] = []
    matched_terms: List[str] = []
    matched_span_text: List[str] = []
    for start, end, term in spans:
        matched_terms.append(str(term))
        try:
            span_text = pack.candidate_text
            decoded = ""
        except Exception:
            decoded = ""
        matched_span_text.append(decoded)
        for rel_idx in range(int(start), int(end)):
            metrics.append(token_metrics(pack, rel_idx))

    if not metrics:
        return {
            "object": obj,
            "surfaces": " | ".join(surfaces),
            "matched_terms": "",
            "n_spans": 0,
            "n_tokens": 0,
            "obj_lp_min": 0.0,
            "obj_lp_mean": 0.0,
            "obj_gap_min": 0.0,
            "obj_gap_mean": 0.0,
            "obj_entropy_mean": 0.0,
            "obj_entropy_max": 0.0,
            "obj_rank_max": 0.0,
            "obj_top1_rate": 0.0,
            "obj_support_gap_prob": 1.0,
            "obj_support_lp_prob": 1.0,
        }

    lps = [float(m["lp"]) for m in metrics]
    gaps = [float(m["gap"]) for m in metrics]
    ents = [float(m["entropy"]) for m in metrics]
    ranks = [float(m["rank"]) for m in metrics]
    top1s = [float(m["top1"]) for m in metrics]
    lp_min = min(lps)
    gap_min = min(gaps)
    ent_mean = sum(ents) / len(ents)
    return {
        "object": obj,
        "surfaces": " | ".join(surfaces),
        "matched_terms": " | ".join(ordered_unique(matched_terms, max_items=0)),
        "n_spans": int(len(spans)),
        "n_tokens": int(len(metrics)),
        "obj_lp_min": float(lp_min),
        "obj_lp_mean": float(sum(lps) / len(lps)),
        "obj_gap_min": float(gap_min),
        "obj_gap_mean": float(sum(gaps) / len(gaps)),
        "obj_entropy_mean": float(ent_mean),
        "obj_entropy_max": float(max(ents)),
        "obj_rank_max": float(max(ranks)),
        "obj_top1_rate": float(sum(top1s) / len(top1s)),
        "obj_support_gap_prob": sigmoid(gap_min),
        "obj_support_lp_prob": float(math.exp(max(-50.0, min(0.0, lp_min)))),
    }


def support_score(item: Dict[str, Any], mode: str) -> float:
    if mode == "gap_prob":
        return safe_float(item.get("obj_support_gap_prob"), 1.0)
    if mode == "lp_prob":
        return safe_float(item.get("obj_support_lp_prob"), 1.0)
    if mode == "top1_rate":
        return safe_float(item.get("obj_top1_rate"), 1.0)
    if mode == "gap_entropy":
        gap = safe_float(item.get("obj_gap_min"), 0.0)
        ent = safe_float(item.get("obj_entropy_mean"), 0.0)
        return sigmoid(gap - 0.25 * ent)
    raise ValueError(f"unsupported risk_score_mode={mode!r}")


def add_risk_features(row: Dict[str, Any], scored: Sequence[Dict[str, Any]], *, mode: str) -> List[Dict[str, Any]]:
    valid = [dict(item) for item in scored if int(safe_float(item.get("n_tokens"), 0.0)) > 0]
    for item in valid:
        item["risk_support_score"] = support_score(item, mode)
        item["risk_score_mode"] = str(mode)
    ranked = sorted(valid, key=lambda item: safe_float(item.get("risk_support_score"), 1.0))
    top = ranked[0] if ranked else {}
    second = ranked[1] if len(ranked) > 1 else {}
    top_score = safe_float(top.get("risk_support_score"), 1.0) if top else 1.0
    second_score = safe_float(second.get("risk_support_score"), 1.0) if second else 1.0
    row.update(
        {
            "risk_object_count": int(len(ranked)),
            "risk_top_object": str(top.get("object", "")) if top else "",
            "risk_top_yes_prob": float(top_score),
            "risk_top_no_risk": float(1.0 - top_score),
            "risk_top_lp_margin": safe_float(top.get("obj_gap_min"), 0.0) if top else 0.0,
            "risk_top_yes_lp": safe_float(top.get("obj_lp_min"), 0.0) if top else 0.0,
            "risk_top_entropy": safe_float(top.get("obj_entropy_mean"), 0.0) if top else 0.0,
            "risk_top_rank_max": safe_float(top.get("obj_rank_max"), 0.0) if top else 0.0,
            "risk_second_object": str(second.get("object", "")) if second else "",
            "risk_second_yes_prob": float(second_score),
            "risk_second_no_risk": float(1.0 - second_score),
            "risk_second_lp_margin": safe_float(second.get("obj_gap_min"), 0.0) if second else 0.0,
            "risk_second_minus_top_yes_prob": float(second_score - top_score),
            "risk_ranked_objects": " | ".join(str(item.get("object", "")) for item in ranked),
        }
    )
    return ranked


def oracle_topk_hit(candidates: Sequence[str], gold: Sequence[str], k: int) -> int:
    top = list(candidates)[: max(0, int(k))]
    return int(any(object_matches(g, c) for g in gold for c in top))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score intervention-caption object mentions by teacher-forced caption replay and emit v82-compatible risk CSV."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--intervention_object_pred_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--oracle_rows_csv", default="")
    ap.add_argument("--oracle_object_col", default="int_hallucinated_unique")
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_objects", type=int, default=8)
    ap.add_argument("--object_vocab", default="coco80")
    ap.add_argument("--filter_to_vocab", type=parse_bool, default=True)
    ap.add_argument("--risk_score_mode", choices=["gap_prob", "lp_prob", "top1_rate", "gap_entropy"], default="gap_prob")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(os.path.abspath(args.out_csv)):
        print(f"[reuse] {args.out_csv}")
        return

    questions = load_question_rows(os.path.abspath(args.question_file), limit=int(args.limit))
    int_caps = read_prediction_map(os.path.abspath(args.intervention_pred_jsonl))
    int_objects_pred = read_prediction_map(os.path.abspath(args.intervention_object_pred_jsonl))
    object_vocab = parse_object_vocab(str(args.object_vocab))
    oracle_by_id: Dict[str, Dict[str, str]] = {}
    if str(args.oracle_rows_csv or "").strip() and os.path.isfile(os.path.abspath(args.oracle_rows_csv)):
        oracle_by_id = {norm_id(row.get("id") or row.get("image_id") or row.get("question_id")): row for row in read_csv_rows(args.oracle_rows_csv)}

    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=(str(args.model_base) or None),
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_replay_forwards = 0
    n_objects = 0
    n_matched_objects = 0
    for idx, sample in enumerate(questions):
        sid = norm_id(sample.get("question_id", sample.get("id", sample.get("image_id"))))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("question", sample.get("text", ""))).strip()
        row: Dict[str, Any] = {
            "id": sid,
            "image_id": sid,
            "image": image_name,
            "risk_error": "",
            "risk_score_mode": str(args.risk_score_mode),
        }
        try:
            if not sid or sid not in int_caps:
                raise ValueError("Missing intervention caption prediction row.")
            if sid not in int_objects_pred:
                raise ValueError("Missing intervention object prediction row.")
            caption = str(int_caps[sid].get("text", "")).strip()
            if not caption:
                raise ValueError("Empty intervention caption.")
            image_path = os.path.join(str(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            raw_objects = ordered_unique(split_object_list(str(int_objects_pred[sid].get("text", ""))), max_items=int(args.max_objects))
            candidates = object_candidates_with_surfaces(
                raw_objects,
                object_vocab,
                filter_to_vocab=bool(args.filter_to_vocab),
                max_items=int(args.max_objects),
            )
            image = runtime.load_image(image_path)
            pack = runtime.teacher_force_candidate(
                image=image,
                question=question,
                candidate_text=caption,
                output_attentions=False,
            )
            n_replay_forwards += 1
            cont_ids = [int(x) for x in pack.cont_ids.tolist()]
            scored: List[Dict[str, Any]] = []
            for cand in candidates:
                obj = str(cand["object"])
                surfaces = [str(x) for x in cand.get("surfaces", [])]
                spans = object_spans(runtime.tokenizer, cont_ids, obj, surfaces)
                scored.append(summarize_object(obj, surfaces, spans, pack))
            n_objects += len(candidates)
            n_matched_objects += sum(1 for item in scored if int(safe_float(item.get("n_tokens"), 0.0)) > 0)
            ranked = add_risk_features(row, scored, mode=str(args.risk_score_mode))
            row.update(
                {
                    "caption": caption,
                    "int_raw_object_names": " | ".join(raw_objects),
                    "int_raw_object_count": int(len(raw_objects)),
                    "int_object_names": " | ".join(str(c["object"]) for c in candidates),
                    "int_object_count": int(len(candidates)),
                    "risk_filter_to_vocab": int(bool(args.filter_to_vocab)),
                    "risk_details_json": json.dumps(ranked, ensure_ascii=False, sort_keys=True),
                }
            )
            oracle = oracle_by_id.get(sid, {})
            gold = split_pipe_objects(oracle.get(str(args.oracle_object_col), "")) if oracle else []
            ranked_names = [str(item.get("object", "")) for item in ranked]
            row.update(
                {
                    "oracle_object_col": str(args.oracle_object_col),
                    "oracle_hallucinated_objects": " | ".join(gold),
                    "oracle_hallucinated_object_count": int(len(gold)),
                    "oracle_any_hallucinated_object": int(bool(gold)),
                    "oracle_risk_top1_hit": oracle_topk_hit(ranked_names, gold, 1),
                    "oracle_risk_top2_hit": oracle_topk_hit(ranked_names, gold, 2),
                    "oracle_risk_top3_hit": oracle_topk_hit(ranked_names, gold, 3),
                }
            )
        except Exception as exc:
            n_errors += 1
            row["risk_error"] = str(exc)
            add_risk_features(row, [], mode=str(args.risk_score_mode))
            row["risk_details_json"] = "[]"
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(
                f"[intervention-object-replay-risk] {idx + 1}/{len(questions)} "
                f"objects={n_objects} matched={n_matched_objects} replay_forwards={n_replay_forwards}",
                flush=True,
            )

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")
    if str(args.out_summary_json or "").strip():
        selected_like = [row for row in rows if safe_float(row.get("risk_top_yes_prob"), 1.0) < 0.5]
        top1_hits = sum(int(row.get("oracle_risk_top1_hit", 0) or 0) for row in rows)
        any_gold = sum(int(row.get("oracle_any_hallucinated_object", 0) or 0) for row in rows)
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "intervention_object_pred_jsonl": os.path.abspath(args.intervention_object_pred_jsonl),
                    "oracle_rows_csv": os.path.abspath(args.oracle_rows_csv) if args.oracle_rows_csv else "",
                    "model_path": str(args.model_path),
                    "conv_mode": str(args.conv_mode),
                    "limit": int(args.limit),
                    "max_objects": int(args.max_objects),
                    "object_vocab": str(args.object_vocab),
                    "filter_to_vocab": bool(args.filter_to_vocab),
                    "risk_score_mode": str(args.risk_score_mode),
                },
                "counts": {
                    "n_rows": int(len(rows)),
                    "n_errors": int(n_errors),
                    "n_objects": int(n_objects),
                    "n_matched_objects": int(n_matched_objects),
                    "n_replay_forwards": int(n_replay_forwards),
                    "n_forward_passes_est": int(n_replay_forwards),
                    "n_oracle_any_hallucinated_object": int(any_gold),
                    "oracle_top1_hit_rate_over_all": float(top1_hits / max(1, len(rows))),
                    "oracle_top1_hit_rate_over_gold": float(top1_hits / max(1, any_gold)),
                    "n_risk_top_yes_prob_lt_050": int(len(selected_like)),
                },
                "outputs": {"out_csv": os.path.abspath(args.out_csv)},
            },
        )
        print(f"[saved] {args.out_summary_json}")


if __name__ == "__main__":
    main()
