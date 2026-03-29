#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Two-stage adaptive ARTrap evaluator (minimal, method-only).

Kept method:
1) Stage-1 routing gate on greedy champion:
   - vpmi_only: champ_vpmi < tau_vpmi
   - and:       champ_vpmi < tau_vpmi AND champ_s_full < tau_sfull
2) Stage-2 decision on expanded Beam-6 pool:
   - fallback prediction = expanded-run champion
   - selector = agree_vminpm_wmin_dfull_le:-0.05
   - trigger  = P3

To keep behavior exactly aligned with prior reported numbers,
this file reuses canonical offline logic from eval_selector_tradeoff:
- load_samples
- select_candidate
- switch_cond
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from eval_selector_tradeoff import load_samples, select_candidate, switch_cond
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def read_questions_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rows: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        for qid, meta in obj.items():
            if not isinstance(meta, dict):
                continue
            row = dict(meta)
            row["id"] = str(qid)
            rows.append(row)
    elif isinstance(obj, list):
        for i, meta in enumerate(obj):
            if not isinstance(meta, dict):
                continue
            row = dict(meta)
            row.setdefault("id", str(i))
            rows.append(row)
    return rows


def write_questions_subset(path: str, rows: Sequence[Dict[str, Any]], keep_ids: Set[str]) -> int:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = str(r.get("id", ""))
        if sid == "" or sid not in keep_ids:
            continue
        rr = dict(r)
        rr.pop("id", None)
        out[sid] = rr
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return int(len(out))


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    if os.path.getsize(path) <= 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_or_none(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if len(xs) == 0:
        return None
    return float(sum(xs) / len(xs))


def load_subset_ids(path: Optional[str]) -> Optional[Set[str]]:
    if path is None or str(path).strip() == "":
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    out: Set[str] = set()
    if isinstance(obj, dict):
        for k in obj.keys():
            out.add(str(k))
        for v in obj.values():
            if isinstance(v, dict) and "id" in v:
                out.add(str(v.get("id")))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, dict):
                out.add(str(v.get("id", i)))
            else:
                out.add(str(v))

    if len(out) == 0:
        raise RuntimeError(f"No IDs parsed from subset json: {path}")
    return out


@dataclass
class Gate:
    name: str
    mode: str  # vpmi_only | and | never
    tau_vpmi: Optional[float]
    tau_sfull: Optional[float]


@dataclass
class EvalArtifacts:
    table_csv: str
    per_sample_csv: str
    summary_json: str
    n_greedy_samples: int
    n_expand_samples: int
    n_common_eval_ids: int


def parse_gates(raw: str) -> List[Gate]:
    txt = str(raw or "").strip()
    if txt == "":
        txt = (
            "gate_and_vpmi_m5.5_and_sfull_m6|and|-5.5|-6;"
            "gate_vpmi_only_m5.0|vpmi_only|-5.0|none"
        )

    rows = [x.strip() for x in txt.split(";") if x.strip() != ""]
    out: List[Gate] = []
    for rr in rows:
        parts = [x.strip() for x in rr.split("|")]
        if len(parts) != 4:
            raise ValueError(
                "Each gate must be 'name|mode|tau_vpmi|tau_sfull'. "
                f"Invalid gate: {rr}"
            )
        name, mode, tv, tf = parts
        mode_l = mode.lower()
        if mode_l not in {"vpmi_only", "and", "never"}:
            raise ValueError(f"Unsupported gate mode for this minimal file: {mode}")

        tau_v = None if tv.lower() in {"", "none", "na"} else safe_float(tv)
        tau_f = None if tf.lower() in {"", "none", "na"} else safe_float(tf)
        out.append(Gate(name=name, mode=mode_l, tau_vpmi=tau_v, tau_sfull=tau_f))
    return out


def should_expand(gate: Gate, champ_vpmi: Optional[float], champ_sfull: Optional[float]) -> bool:
    if gate.mode == "never":
        return False

    cond_v = False
    cond_f = False
    if gate.tau_vpmi is not None and champ_vpmi is not None:
        cond_v = bool(float(champ_vpmi) < float(gate.tau_vpmi))
    if gate.tau_sfull is not None and champ_sfull is not None:
        cond_f = bool(float(champ_sfull) < float(gate.tau_sfull))

    if gate.mode == "vpmi_only":
        return cond_v
    if gate.mode == "and":
        return bool(cond_v and cond_f)
    return False


def evaluate_gate(
    gate: Gate,
    ids: Sequence[str],
    greedy_by_id: Dict[str, Any],
    expand_by_id: Dict[str, Any],
    policy: str,
    trigger: str,
    extra_candidates_cost: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = int(len(ids))
    base_correct = 0
    final_correct = 0
    gain = 0
    harm = 0
    same = 0
    n_expand = 0
    n_switch = 0

    detail_rows: List[Dict[str, Any]] = []
    for sid in ids:
        gs = greedy_by_id[sid]
        es = None

        base_ok = bool(gs.base_ok)
        pred_ok = bool(base_ok)
        if base_ok:
            base_correct += 1

        expanded = should_expand(gate=gate, champ_vpmi=gs.champ.vpmi, champ_sfull=gs.champ.s_full)
        switched = False
        safe_idx = None

        if expanded:
            n_expand += 1
            es = expand_by_id.get(sid)
            if es is None:
                raise RuntimeError(f"Expanded sample missing in expand pool for id={sid}")
            # Expanded-sample fallback: beam champion.
            pred_ok = bool(es.base_ok)

            safe = select_candidate(str(policy), es)
            if safe is not None and switch_cond(str(trigger), es, safe):
                switched = True
                n_switch += 1
                safe_idx = int(safe.idx)
                pred_ok = bool(es.safe_ok_by_idx.get(int(safe.idx), False))

        if pred_ok:
            final_correct += 1
        if pred_ok and (not base_ok):
            gain += 1
        elif (not pred_ok) and base_ok:
            harm += 1
        else:
            same += 1

        detail_rows.append(
            {
                "id": sid,
                "gate": gate.name,
                "base_ok": bool(base_ok),
                "pred_ok": bool(pred_ok),
                "expanded": bool(expanded),
                "switched": bool(switched),
                "safe_idx": safe_idx,
                "greedy_champ_vpmi": gs.champ.vpmi,
                "greedy_champ_s_full": gs.champ.s_full,
                "expand_champ_vpmi": (None if es is None else es.champ.vpmi),
                "expand_champ_s_full": (None if es is None else es.champ.s_full),
            }
        )

    base_acc = float(base_correct / n) if n > 0 else None
    final_acc = float(final_correct / n) if n > 0 else None
    delta_acc = None if base_acc is None or final_acc is None else float(final_acc - base_acc)

    expand_rate = float(n_expand / n) if n > 0 else None
    switch_rate = float(n_switch / n) if n > 0 else None
    precision_gain = None if (gain + harm) == 0 else float(gain / (gain + harm))

    avg_cost_rel = None
    speedup_vs_fixed6 = None
    if expand_rate is not None:
        avg_cost_rel = float(1.0 + float(extra_candidates_cost) * float(expand_rate))
        if avg_cost_rel > 0.0:
            speedup_vs_fixed6 = float(6.0 / avg_cost_rel)

    summary_row = {
        "gate": gate.name,
        "gate_mode": gate.mode,
        "tau_vpmi": gate.tau_vpmi,
        "tau_sfull": gate.tau_sfull,
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": delta_acc,
        "expand_rate": expand_rate,
        "switch_rate": switch_rate,
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "precision_gain": precision_gain,
        "avg_cost_rel": avg_cost_rel,
        "speedup_vs_fixed6": speedup_vs_fixed6,
    }
    return summary_row, detail_rows


def import_pairwise_module(path: str):
    mod_name = f"pairwise_fragility_module_{abs(hash(os.path.abspath(path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pairwise module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves type metadata via sys.modules[cls.__module__]
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


def init_pairwise_runtime(
    *,
    pf,
    model_path: str,
    model_base: Optional[str],
    conv_mode_override: Optional[str],
    attn_impl: str,
    use_flash_attn: bool,
):
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"PyTorch required: {e}")

    from PIL import Image
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.utils import disable_torch_init

    pf.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
    pf.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
    pf.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
    pf.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
    pf.conv_templates = conv_templates
    pf.process_images = process_images
    pf.tokenizer_image_token = tokenizer_image_token
    pf.disable_torch_init = disable_torch_init
    pf.Image = Image

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_4bit=False,
        load_8bit=False,
        use_flash_attn=bool(use_flash_attn),
        device_map="auto",
    )
    model.eval()

    if str(attn_impl) != "auto":
        try:
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = str(attn_impl)
        except Exception:
            pass
        try:
            mm = model.get_model()
            if hasattr(mm.config, "attn_implementation"):
                mm.config.attn_implementation = str(attn_impl)
        except Exception:
            pass

    conv_mode = pf.resolve_conv_mode(model_name, conv_mode_override)
    device = model.get_model().embed_tokens.weight.device
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None or int(pad_id) < 0:
        pad_id = eos_id if eos_id is not None else 0
    phrase_id_map = pf.build_phrase_id_map(tokenizer, pf.FORMAT_PHRASES)
    return {
        "torch": torch,
        "Image": Image,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "conv_mode": conv_mode,
        "device": device,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "phrase_id_map": phrase_id_map,
    }


def score_beam_once(
    *,
    pf,
    tokenizer,
    model,
    question: str,
    answer: str,
    input_ids_img,
    input_ids_q,
    images_tensor,
    image_sizes,
    eos_id: Optional[int],
    pad_id: int,
    phrase_id_map,
    num_beams: int,
    num_return_sequences: int,
    max_new_tokens: int,
    answer_span_max_tokens: int,
    eval_mode: str,
    vpmi_min_sq_logp_max: float = -0.1,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[float], Optional[float], bool]:
    cand_rows = pf.generate_candidates(
        model=model,
        tokenizer=tokenizer,
        input_ids_img=input_ids_img,
        images_tensor=images_tensor,
        image_sizes=image_sizes,
        question=question,
        phrase_id_map=phrase_id_map,
        candidate_gen_mode="beam",
        num_beams=int(num_beams),
        num_beam_groups=1,
        diversity_penalty=0.0,
        num_return_sequences=int(num_return_sequences),
        anchor_depth=2,
        anchor_width=4,
        anchor_prefix_cap=6,
        anchor_completion_budget=6,
        num_extra_samples=0,
        extra_sample_temperature=1.0,
        extra_sample_top_p=0.9,
        extra_sample_top_k=0,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=eos_id,
    )
    if len(cand_rows) == 0:
        return {"error": "no_candidates"}, [], None, None, False

    cands: List[Any] = []
    core_ids_for_sq: List[List[int]] = []
    core_img_logps_for_sq: List[List[float]] = []
    core_idxs_for_sq: List[List[int]] = []
    for ci, gen_c in enumerate(cand_rows):
        cand_ids = [int(x) for x in gen_c.token_ids]
        token_logps_img = gen_c.token_logps_img
        if token_logps_img is None or len(token_logps_img) != len(cand_ids):
            token_logps_img = pf.sequence_token_logps(
                model=model,
                prefix_ids=input_ids_img,
                cont_ids=cand_ids,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
            )
        if token_logps_img is None or len(token_logps_img) == 0:
            continue

        text = tokenizer.decode(cand_ids, skip_special_tokens=True).strip()
        short_answer = pf.extract_core_answer_text(question=question, text=text, max_words=6)
        core_start_aligned, core_match_ids = pf.locate_core_span_from_text(
            tokenizer=tokenizer,
            cand_ids=[int(x) for x in cand_ids],
            core_text=short_answer,
        )
        if core_start_aligned is not None and len(core_match_ids) > 0:
            core_idxs = list(range(int(core_start_aligned), int(core_start_aligned) + int(len(core_match_ids))))
            format_len = int(max(0, int(core_start_aligned)))
            core_start_idx = int(core_start_aligned)
            core_span_source = "aligned"
        else:
            format_len_phrase, _ = pf.detect_format_len(tokenizer, cand_ids, phrase_id_map)
            format_len_prefix = pf.infer_prefix_format_len_from_tokens(tokenizer, cand_ids, question)
            format_len = int(max(int(format_len_phrase), int(format_len_prefix)))
            core_start = int(min(format_len, max(0, len(cand_ids) - 1)))
            span_len = int(answer_span_max_tokens) if int(answer_span_max_tokens) > 0 else 1
            span_len = max(1, span_len)
            core_idxs = list(range(core_start, min(len(cand_ids), core_start + span_len)))
            core_start_idx = int(core_start)
            core_span_source = "fallback"

        if int(answer_span_max_tokens) > 0 and len(core_idxs) > int(answer_span_max_tokens):
            core_idxs = core_idxs[: int(answer_span_max_tokens)]
        if len(core_idxs) == 0 and len(cand_ids) > 0:
            core_idxs = [0]
        core_ids = [int(cand_ids[j]) for j in core_idxs if 0 <= int(j) < len(cand_ids)]
        if len(core_ids) == 0 and len(cand_ids) > 0:
            core_ids = [int(cand_ids[0])]
            core_idxs = [0]
        core_img_logps = [float(token_logps_img[j]) for j in core_idxs if 0 <= int(j) < len(token_logps_img)]

        s_format_img = (
            pf.mean_selected(token_logps_img, list(range(min(int(format_len), len(token_logps_img)))))
            if int(format_len) > 0
            else None
        )
        s_core_img = pf.mean_selected(token_logps_img, core_idxs)
        s_full = pf.mean_selected(token_logps_img, list(range(len(token_logps_img))))

        cands.append(
            pf.Candidate(
                cand_idx=int(ci),
                token_ids=[int(x) for x in cand_ids],
                text=text,
                short_answer=short_answer,
                s_full=s_full,
                s_format_img=s_format_img,
                s_core_img=s_core_img,
                s_ans_q=None,
                s_core_img_min=None,
                s_ans_q_min=None,
                vpmi_core_mean=None,
                vpmi_core_min=None,
                vpmi_core_min_raw=None,
                vpmi_core_min_prior_masked=None,
                vpmi_word_min=None,
                vpmi_core_tail_min=None,
                vpmi_core_min_pos_norm=None,
                vpmi_core_min_mean_gap=None,
                vpmi_core_sign_flip_count=None,
                margin_core_img_min=None,
                margin_core_img_mean=None,
                core_img_toks=None,
                core_q_toks=None,
                core_vpmi_toks=None,
                format_len=int(format_len),
                core_len=int(len(core_ids)),
                core_start=int(core_start_idx),
                core_span_source=str(core_span_source),
            )
        )
        core_ids_for_sq.append([int(x) for x in core_ids])
        core_img_logps_for_sq.append([float(x) for x in core_img_logps])
        core_idxs_for_sq.append([int(x) for x in core_idxs])

    if len(cands) == 0:
        return {"error": "no_scored_candidates"}, [], None, None, False

    token_logps_q_rows = pf.sequence_token_logps_batched(
        model=model,
        prefix_ids=input_ids_q,
        cont_ids_list=core_ids_for_sq,
        pad_token_id=pad_id,
        images_tensor=None,
        image_sizes=None,
    )
    for ci, c in enumerate(cands):
        tlogps_q = token_logps_q_rows[ci] if ci < len(token_logps_q_rows) else None
        c.s_ans_q = pf.mean_selected(tlogps_q, list(range(len(tlogps_q or []))))
        core_img_toks = core_img_logps_for_sq[ci] if ci < len(core_img_logps_for_sq) else []
        core_q_toks = (tlogps_q or [])
        c.s_core_img_min = (None if len(core_img_toks) == 0 else float(min(core_img_toks)))
        c.s_ans_q_min = (None if len(core_q_toks) == 0 else float(min(core_q_toks)))
        m = int(min(len(core_img_toks), len(core_q_toks)))
        if m > 0:
            vpmi_toks = [float(core_img_toks[k]) - float(core_q_toks[k]) for k in range(m)]
            c.vpmi_core_mean = float(sum(vpmi_toks) / len(vpmi_toks))
            vpmi_min_raw = float(min(vpmi_toks))
            c.vpmi_core_min_raw = vpmi_min_raw
            keep_idxs = [int(k) for k, sq in enumerate(core_q_toks[:m]) if float(sq) <= float(vpmi_min_sq_logp_max)]
            c.vpmi_core_min_prior_masked = (
                None if len(keep_idxs) == 0 else float(min(vpmi_toks[k] for k in keep_idxs))
            )
            core_ids_aligned = [int(x) for x in core_ids_for_sq[ci][:m]]
            word_groups = pf.group_token_positions_by_word(tokenizer, core_ids_aligned)
            word_scores = [
                float(sum(vpmi_toks[j] for j in g) / len(g))
                for g in word_groups
                if len(g) > 0
            ]
            c.vpmi_word_min = (None if len(word_scores) == 0 else float(min(word_scores)))
            c.vpmi_core_min = vpmi_min_raw

    cands = [c for c in cands if c.s_full is not None]
    if len(cands) == 0:
        return {"error": "no_full_score"}, [], None, None, False

    cands_sorted = sorted(cands, key=lambda x: float(x.s_full), reverse=True)
    champ = cands_sorted[0]
    champ_vpmi = (None if champ.s_core_img is None or champ.s_ans_q is None else float(champ.s_core_img - champ.s_ans_q))
    champ_sfull = champ.s_full
    prior_vals = [float(c.s_ans_q) for c in cands if c.s_ans_q is not None]
    p_q_map: Dict[int, Optional[float]] = {}
    for c in cands:
        if c.s_ans_q is None:
            p_q_map[c.cand_idx] = None
        else:
            p_q_map[c.cand_idx] = pf.percentile_rank(prior_vals, float(c.s_ans_q))

    pred_answer = pf.first_clause(champ.short_answer if champ.short_answer else champ.text)
    is_success_strict = bool(pf.norm_text(pred_answer) == pf.norm_text(answer))
    is_success_heur = pf.is_success_heuristic(
        question=question,
        answer=answer,
        champ_text=champ.text,
        champ_short=champ.short_answer,
    )
    use_heur = bool(str(eval_mode) != "strict")
    is_success = bool(is_success_heur if use_heur else is_success_strict)
    label = "success" if is_success else "failure"

    per_sample_row: Dict[str, Any] = {
        "id": None,
        "question": question,
        "answer": answer,
        "label": label,
        "is_success": bool(is_success),
        "is_success_strict": bool(is_success_strict),
        "is_success_heuristic": bool(is_success_heur),
        "eval_match_mode": ("heuristic" if use_heur else "strict"),
        "pred_answer_eval": pred_answer,
        "n_candidates": int(len(cands)),
        "champ_idx": int(champ.cand_idx),
        "champ_text": champ.text,
        "champ_short_answer": champ.short_answer,
        "champ_s_full": champ.s_full,
        "champ_s_core_img": champ.s_core_img,
        "champ_s_ans_q": champ.s_ans_q,
    }

    per_candidate_rows: List[Dict[str, Any]] = []
    for c in cands:
        per_candidate_rows.append(
            {
                "id": None,
                "label": label,
                "cand_idx": int(c.cand_idx),
                "is_champion": bool(c.cand_idx == champ.cand_idx),
                "text": c.text,
                "short_answer": c.short_answer,
                "s_full": c.s_full,
                "s_core_img": c.s_core_img,
                "s_ans_q": c.s_ans_q,
                "s_core_img_min": c.s_core_img_min,
                "s_ans_q_min": c.s_ans_q_min,
                "vpmi_core_mean": c.vpmi_core_mean,
                "vpmi_core_min": c.vpmi_core_min,
                "vpmi_core_min_raw": c.vpmi_core_min_raw,
                "vpmi_core_min_prior_masked": c.vpmi_core_min_prior_masked,
                "vpmi_word_min": c.vpmi_word_min,
                "core_len": int(c.core_len),
                "p_q": p_q_map.get(c.cand_idx),
                "is_safe": False,
            }
        )
    return per_sample_row, per_candidate_rows, champ_vpmi, champ_sfull, True


def run_stage_with_metrics(
    *,
    fn,
    torch_mod,
    use_cuda: bool,
    profile_flops: bool,
) -> Tuple[Any, Dict[str, Optional[float]]]:
    if bool(use_cuda):
        try:
            torch_mod.cuda.synchronize()
        except Exception:
            pass
        ev_start = torch_mod.cuda.Event(enable_timing=True)
        ev_end = torch_mod.cuda.Event(enable_timing=True)
        ev_start.record()
    else:
        ev_start = None
        ev_end = None

    t0 = time.time()
    flops = None
    if bool(profile_flops):
        activities = [torch_mod.profiler.ProfilerActivity.CPU]
        if bool(use_cuda):
            activities.append(torch_mod.profiler.ProfilerActivity.CUDA)
        with torch_mod.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            out = fn()
        try:
            flops = float(sum(float(getattr(evt, "flops", 0.0) or 0.0) for evt in prof.key_averages()))
        except Exception:
            flops = None
    else:
        out = fn()
    wall_ms = float((time.time() - t0) * 1000.0)

    gpu_ms: Optional[float] = None
    if bool(use_cuda) and ev_start is not None and ev_end is not None:
        try:
            ev_end.record()
            torch_mod.cuda.synchronize()
            gpu_ms = float(ev_start.elapsed_time(ev_end))
        except Exception:
            gpu_ms = None
    return out, {"wall_ms": wall_ms, "gpu_ms": gpu_ms, "flops": flops}


def run_pairwise_one_shot(
    *,
    out_dir: str,
    pairwise_script: str,
    questions_json: str,
    image_root: str,
    model_path: str,
    model_base: Optional[str],
    conv_mode: Optional[str],
    attn_impl: str,
    use_flash_attn: bool,
    num_samples: int,
    seed: int,
    max_new_tokens: int,
    answer_span_max_tokens: int,
    expand_num_beams: int,
    expand_num_return_sequences: int,
    eval_mode: str,
    gates_raw: str,
    subset_json: Optional[str],
    pairwise_tmp_dir: Optional[str],
    pairwise_extra_args: str,
    enable_tqdm: bool,
    profile_flops: bool,
    flops_profile_samples: int,
) -> Tuple[str, str, str]:
    if str(pairwise_extra_args).strip() != "":
        raise ValueError("--pairwise_extra_args is not supported in in-process mode.")
    ts = time.strftime("%Y%m%d_%H%M%S")
    root_dir = (
        os.path.abspath(pairwise_tmp_dir)
        if pairwise_tmp_dir is not None and str(pairwise_tmp_dir).strip() != ""
        else os.path.join(os.path.abspath(out_dir), f"_pairwise_runs_{ts}")
    )
    os.makedirs(root_dir, exist_ok=True)

    greedy_dir = os.path.join(root_dir, "greedy_b1")
    expand_dir = os.path.join(root_dir, "beam6_b6")
    os.makedirs(greedy_dir, exist_ok=True)
    os.makedirs(expand_dir, exist_ok=True)

    pf = import_pairwise_module(str(pairwise_script))
    rt = init_pairwise_runtime(
        pf=pf,
        model_path=str(model_path),
        model_base=model_base,
        conv_mode_override=conv_mode,
        attn_impl=str(attn_impl),
        use_flash_attn=bool(use_flash_attn),
    )
    torch = rt["torch"]
    tokenizer = rt["tokenizer"]
    model = rt["model"]
    image_processor = rt["image_processor"]
    conv_mode_resolved = rt["conv_mode"]
    device = rt["device"]
    eos_id = rt["eos_id"]
    pad_id = int(rt["pad_id"])
    phrase_id_map = rt["phrase_id_map"]
    process_images = rt["process_images"]
    tokenizer_image_token = rt["tokenizer_image_token"]
    Image = rt["Image"]
    use_cuda = bool(getattr(torch, "cuda", None) is not None and torch.cuda.is_available())

    if int(expand_num_beams) < 1:
        raise ValueError("--expand_num_beams must be >= 1")
    if int(expand_num_return_sequences) < 1:
        raise ValueError("--expand_num_return_sequences must be >= 1")
    if int(expand_num_return_sequences) > int(expand_num_beams):
        raise ValueError("--expand_num_return_sequences must be <= --expand_num_beams")

    rows = read_questions_rows(str(questions_json))
    if int(num_samples) > 0:
        rows = rows[: int(num_samples)]
    subset_ids = load_subset_ids(subset_json if str(subset_json or "").strip() != "" else None)
    if subset_ids is not None:
        rows = [r for r in rows if str(r.get("id", "")) in subset_ids]
    gates = parse_gates(gates_raw)

    greedy_per_sample: List[Dict[str, Any]] = []
    greedy_per_candidate: List[Dict[str, Any]] = []
    expand_per_sample: List[Dict[str, Any]] = []
    expand_per_candidate: List[Dict[str, Any]] = []
    expand_count = 0
    progress_rows: List[Dict[str, Any]] = []
    greedy_wall_ms_all: List[Optional[float]] = []
    greedy_gpu_ms_all: List[Optional[float]] = []
    greedy_flops_measured: List[Optional[float]] = []
    expand_wall_ms_all: List[Optional[float]] = []
    expand_gpu_ms_all: List[Optional[float]] = []
    expand_flops_measured: List[Optional[float]] = []
    greedy_calls = 0
    expand_calls = 0
    flops_profile_budget_g = int(max(0, flops_profile_samples))
    flops_profile_budget_e = int(max(0, flops_profile_samples))
    one_shot_t0 = time.time()

    total_n = int(len(rows))
    pbar = tqdm(
        rows,
        total=total_n,
        desc="adaptive-onepass",
        dynamic_ncols=True,
        disable=(not bool(enable_tqdm)),
    )
    for i, r in enumerate(pbar):
        qid = str(r.get("id", ""))
        question = str(r.get("question", ""))
        answer = str(r.get("answer", ""))
        image_id = str(r.get("imageId", ""))
        image_path = os.path.join(str(image_root), f"{image_id}.jpg")
        if not os.path.isfile(image_path):
            greedy_per_sample.append({"id": qid, "error": f"missing_image:{image_path}"})
            continue
        try:
            img_prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode_resolved,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            q_prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode_resolved,
                with_image_token=False,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            input_ids_img = tokenizer_image_token(
                img_prompt, tokenizer, pf.IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            input_ids_q = tokenizer(q_prompt, return_tensors="pt").input_ids.to(device)
            image = Image.open(image_path).convert("RGB")
            image_sizes = [image.size]
            images_tensor = process_images([image], image_processor, model.config).to(device=model.device, dtype=torch.float16)
        except Exception as e:
            greedy_per_sample.append({"id": qid, "error": f"build_input:{e}"})
            continue

        greedy_calls += 1
        do_profile_g = bool(profile_flops and flops_profile_budget_g > 0)
        if do_profile_g:
            flops_profile_budget_g -= 1
        (g_row, g_cands, g_vpmi, g_sfull, g_ok), g_metrics = run_stage_with_metrics(
            fn=lambda: score_beam_once(
                pf=pf,
                tokenizer=tokenizer,
                model=model,
                question=question,
                answer=answer,
                input_ids_img=input_ids_img,
                input_ids_q=input_ids_q,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
                eos_id=eos_id,
                pad_id=pad_id,
                phrase_id_map=phrase_id_map,
                num_beams=1,
                num_return_sequences=1,
                max_new_tokens=int(max_new_tokens),
                answer_span_max_tokens=int(answer_span_max_tokens),
                eval_mode=str(eval_mode),
            ),
            torch_mod=torch,
            use_cuda=use_cuda,
            profile_flops=do_profile_g,
        )
        g_row["id"] = qid
        if "error" not in g_row:
            g_row["image_id"] = image_id
            for cc in g_cands:
                cc["id"] = qid
        greedy_per_sample.append(g_row)
        greedy_per_candidate.extend(g_cands)
        greedy_wall_ms_all.append(g_metrics.get("wall_ms"))
        greedy_gpu_ms_all.append(g_metrics.get("gpu_ms"))
        greedy_flops_measured.append(g_metrics.get("flops"))

        do_expand = bool(
            g_ok and any(should_expand(gate=g, champ_vpmi=g_vpmi, champ_sfull=g_sfull) for g in gates)
        )
        expand_wall_this = None
        expand_gpu_this = None
        expand_flops_this = None
        if do_expand:
            expand_calls += 1
            do_profile_e = bool(profile_flops and flops_profile_budget_e > 0)
            if do_profile_e:
                flops_profile_budget_e -= 1
            (e_row, e_cands, _, _, e_ok), e_metrics = run_stage_with_metrics(
                fn=lambda: score_beam_once(
                    pf=pf,
                    tokenizer=tokenizer,
                    model=model,
                    question=question,
                    answer=answer,
                    input_ids_img=input_ids_img,
                    input_ids_q=input_ids_q,
                    images_tensor=images_tensor,
                    image_sizes=image_sizes,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    phrase_id_map=phrase_id_map,
                    num_beams=int(expand_num_beams),
                    num_return_sequences=int(expand_num_return_sequences),
                    max_new_tokens=int(max_new_tokens),
                    answer_span_max_tokens=int(answer_span_max_tokens),
                    eval_mode=str(eval_mode),
                ),
                torch_mod=torch,
                use_cuda=use_cuda,
                profile_flops=do_profile_e,
            )
            expand_wall_this = e_metrics.get("wall_ms")
            expand_gpu_this = e_metrics.get("gpu_ms")
            expand_flops_this = e_metrics.get("flops")
            expand_wall_ms_all.append(expand_wall_this)
            expand_gpu_ms_all.append(expand_gpu_this)
            expand_flops_measured.append(expand_flops_this)
            if e_ok:
                e_row["id"] = qid
                e_row["image_id"] = image_id
                for cc in e_cands:
                    cc["id"] = qid
                expand_per_sample.append(e_row)
                expand_per_candidate.extend(e_cands)
            else:
                # Fallback: if beam run failed, keep greedy data for this expanded sample.
                grow = dict(g_row)
                grow.pop("error", None)
                expand_per_sample.append(grow)
                for cc in g_cands:
                    expand_per_candidate.append(dict(cc))
            expand_count += 1

        progress_rows.append(
            {
                "step": int(i + 1),
                "n_total": int(total_n),
                "id": qid,
                "expanded": bool(do_expand),
                "expanded_count": int(expand_count),
                "greedy_wall_ms": g_metrics.get("wall_ms"),
                "greedy_gpu_ms": g_metrics.get("gpu_ms"),
                "greedy_flops": g_metrics.get("flops"),
                "expand_wall_ms": expand_wall_this,
                "expand_gpu_ms": expand_gpu_this,
                "expand_flops": expand_flops_this,
            }
        )
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(expanded=int(expand_count))

    write_csv(os.path.join(greedy_dir, "per_sample.csv"), greedy_per_sample)
    write_csv(os.path.join(greedy_dir, "per_candidate.csv"), greedy_per_candidate)
    write_csv(os.path.join(expand_dir, "per_sample.csv"), expand_per_sample)
    write_csv(os.path.join(expand_dir, "per_candidate.csv"), expand_per_candidate)
    progress_csv = os.path.join(root_dir, "onepass_progress.csv")
    write_csv(progress_csv, progress_rows)

    greedy_flops_vals = [float(v) for v in greedy_flops_measured if v is not None and math.isfinite(float(v))]
    expand_flops_vals = [float(v) for v in expand_flops_measured if v is not None and math.isfinite(float(v))]
    mean_g_flops = (None if len(greedy_flops_vals) == 0 else float(sum(greedy_flops_vals) / len(greedy_flops_vals)))
    mean_e_flops = (None if len(expand_flops_vals) == 0 else float(sum(expand_flops_vals) / len(expand_flops_vals)))
    est_total_flops = None
    if mean_g_flops is not None or mean_e_flops is not None:
        est_total_flops = float(
            (0.0 if mean_g_flops is None else mean_g_flops * float(greedy_calls))
            + (0.0 if mean_e_flops is None else mean_e_flops * float(expand_calls))
        )
    runtime_stats = {
        "counts": {
            "n_total_rows": int(total_n),
            "n_greedy_calls": int(greedy_calls),
            "n_expand_calls": int(expand_calls),
            "n_expand_success_rows": int(len(expand_per_sample)),
        },
        "latency_ms": {
            "wall_total_ms": float((time.time() - one_shot_t0) * 1000.0),
            "greedy_wall_mean_ms": mean_or_none(greedy_wall_ms_all),
            "greedy_gpu_mean_ms": mean_or_none(greedy_gpu_ms_all),
            "expand_wall_mean_ms": mean_or_none(expand_wall_ms_all),
            "expand_gpu_mean_ms": mean_or_none(expand_gpu_ms_all),
            "greedy_wall_sum_ms": sum(float(x) for x in greedy_wall_ms_all if x is not None),
            "expand_wall_sum_ms": sum(float(x) for x in expand_wall_ms_all if x is not None),
        },
        "flops": {
            "profile_enabled": bool(profile_flops),
            "profile_samples_per_stage": int(flops_profile_samples),
            "greedy_profiled_calls": int(len(greedy_flops_vals)),
            "expand_profiled_calls": int(len(expand_flops_vals)),
            "greedy_measured_flops_mean": mean_g_flops,
            "expand_measured_flops_mean": mean_e_flops,
            "greedy_measured_flops_sum": (None if len(greedy_flops_vals) == 0 else float(sum(greedy_flops_vals))),
            "expand_measured_flops_sum": (None if len(expand_flops_vals) == 0 else float(sum(expand_flops_vals))),
            "total_flops_estimated": est_total_flops,
        },
        "outputs": {
            "progress_csv": os.path.abspath(progress_csv),
            "greedy_dir": os.path.abspath(greedy_dir),
            "expand_dir": os.path.abspath(expand_dir),
        },
    }
    runtime_stats_json = os.path.join(root_dir, "onepass_runtime_stats.json")
    with open(runtime_stats_json, "w", encoding="utf-8") as f:
        json.dump(runtime_stats, f, indent=2, ensure_ascii=False)
    print("[saved]", runtime_stats_json)

    return root_dir, greedy_dir, expand_dir


def evaluate_from_dirs(
    *,
    greedy_dir: str,
    expand_dir: str,
    out_dir: str,
    subset_json: Optional[str],
    eval_mode: str,
    gates_raw: str,
    policy: str,
    trigger: str,
    extra_candidates_cost: int,
    base_ids_source: str = "intersection",
) -> EvalArtifacts:
    os.makedirs(out_dir, exist_ok=True)

    greedy_samples = load_samples(greedy_dir, eval_mode=str(eval_mode))
    expand_samples = load_samples(expand_dir, eval_mode=str(eval_mode))
    greedy_by_id = {str(s.sid): s for s in greedy_samples}
    expand_by_id = {str(s.sid): s for s in expand_samples}

    mode = str(base_ids_source).strip().lower()
    if mode == "intersection":
        ids = sorted(set(greedy_by_id.keys()) & set(expand_by_id.keys()))
    elif mode == "greedy":
        ids = sorted(greedy_by_id.keys())
    else:
        raise ValueError(f"Unsupported base_ids_source: {base_ids_source}")
    subset_ids = load_subset_ids(subset_json if str(subset_json or "").strip() != "" else None)
    if subset_ids is not None:
        ids = [sid for sid in ids if sid in subset_ids]
    if len(ids) == 0:
        raise RuntimeError("No common ids to evaluate after subset filtering.")

    gates = parse_gates(gates_raw)

    rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    baseline_gate = Gate(name="greedy_only", mode="never", tau_vpmi=None, tau_sfull=None)
    for g in [baseline_gate] + gates:
        row, details = evaluate_gate(
            gate=g,
            ids=ids,
            greedy_by_id=greedy_by_id,
            expand_by_id=expand_by_id,
            policy=str(policy),
            trigger=str(trigger),
            extra_candidates_cost=int(extra_candidates_cost),
        )
        rows.append(row)
        detail_rows.extend(details)

    rows_sorted = sorted(rows, key=lambda x: (safe_float(x.get("delta_acc")) or -1e9), reverse=True)

    table_csv = os.path.join(out_dir, "adaptive_two_stage_table.csv")
    detail_csv = os.path.join(out_dir, "adaptive_two_stage_per_sample.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    write_csv(table_csv, rows_sorted)
    write_csv(detail_csv, detail_rows)

    summary = {
        "inputs": {
            "greedy_dir": os.path.abspath(greedy_dir),
            "expand_dir": os.path.abspath(expand_dir),
            "subset_json": (None if str(subset_json or "").strip() == "" else os.path.abspath(str(subset_json))),
            "eval_mode": str(eval_mode),
            "extra_candidates_cost": int(extra_candidates_cost),
            "gates": [g.__dict__ for g in gates],
            "policy": str(policy),
            "trigger": str(trigger),
            "base_ids_source": mode,
        },
        "counts": {
            "n_greedy_samples": int(len(greedy_samples)),
            "n_expand_samples": int(len(expand_samples)),
            "n_common_eval_ids": int(len(ids)),
        },
        "outputs": {
            "table_csv": os.path.abspath(table_csv),
            "per_sample_csv": os.path.abspath(detail_csv),
            "summary_json": os.path.abspath(summary_json),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", table_csv)
    print("[saved]", detail_csv)
    print("[saved]", summary_json)
    return EvalArtifacts(
        table_csv=os.path.abspath(table_csv),
        per_sample_csv=os.path.abspath(detail_csv),
        summary_json=os.path.abspath(summary_json),
        n_greedy_samples=int(len(greedy_samples)),
        n_expand_samples=int(len(expand_samples)),
        n_common_eval_ids=int(len(ids)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal two-stage ARTrap evaluator (gate + selector + P3)")
    ap.add_argument("--greedy_dir", type=str, default="")
    ap.add_argument("--expand_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--subset_json", type=str, default="")
    ap.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict", "heuristic"])
    ap.add_argument(
        "--gates",
        type=str,
        default="",
        help=(
            "Semicolon list. Each gate: name|mode|tau_vpmi|tau_sfull. "
            "mode in {vpmi_only,and,never}."
        ),
    )
    ap.add_argument("--policy", type=str, default="agree_vminpm_wmin_dfull_le:-0.05")
    ap.add_argument("--trigger", type=str, default="P3")
    ap.add_argument("--extra_candidates_cost", type=int, default=5, help="Relative cost of expanded sample vs greedy.")
    ap.add_argument("--questions_json", type=str, default="", help="If set, run one-shot generation + evaluation.")
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="")
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--answer_span_max_tokens", type=int, default=4)
    ap.add_argument("--expand_num_beams", type=int, default=6, help="Beam size used when gate triggers expansion.")
    ap.add_argument(
        "--expand_num_return_sequences",
        type=int,
        default=6,
        help="Return sequences used in expansion stage (<= expand_num_beams).",
    )
    ap.add_argument(
        "--pairwise_script",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_artrap_pairwise_fragility.py"),
    )
    ap.add_argument("--pairwise_tmp_dir", type=str, default="", help="Intermediate dir for one-shot b1/b6 runs.")
    ap.add_argument("--pairwise_extra_args", type=str, default="", help="Extra args appended to pairwise runs.")
    ap.add_argument("--enable_tqdm", action="store_true", help="Show tqdm progress bar in one-shot mode.")
    ap.add_argument("--profile_flops", action="store_true", help="Profile FLOPs with torch.profiler (one-shot).")
    ap.add_argument(
        "--flops_profile_samples",
        type=int,
        default=20,
        help="How many calls per stage to profile FLOPs for (one-shot).",
    )
    ap.add_argument("--keep_pairwise_dirs", action="store_true", help="Keep intermediate b1/b6 pairwise directories.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    use_one_shot = bool(str(args.questions_json).strip() != "")
    use_offline_dirs = bool(str(args.greedy_dir).strip() != "" and str(args.expand_dir).strip() != "")

    if not use_one_shot and not use_offline_dirs:
        raise ValueError(
            "Provide either (--greedy_dir and --expand_dir) for offline mode, "
            "or --questions_json for one-shot mode."
        )

    cleanup_dir: Optional[str] = None
    greedy_dir = str(args.greedy_dir)
    expand_dir = str(args.expand_dir)
    if use_one_shot:
        if not os.path.isfile(str(args.pairwise_script)):
            raise FileNotFoundError(f"pairwise script not found: {args.pairwise_script}")
        if not os.path.isfile(str(args.questions_json)):
            raise FileNotFoundError(f"questions_json not found: {args.questions_json}")
        cleanup_dir, greedy_dir, expand_dir = run_pairwise_one_shot(
            out_dir=str(args.out_dir),
            pairwise_script=str(args.pairwise_script),
            questions_json=str(args.questions_json),
            image_root=str(args.image_root),
            model_path=str(args.model_path),
            model_base=(None if str(args.model_base).strip() == "" else str(args.model_base)),
            conv_mode=(None if str(args.conv_mode).strip() == "" else str(args.conv_mode)),
            attn_impl=str(args.attn_impl),
            use_flash_attn=bool(args.use_flash_attn),
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
            answer_span_max_tokens=int(args.answer_span_max_tokens),
            expand_num_beams=int(args.expand_num_beams),
            expand_num_return_sequences=int(args.expand_num_return_sequences),
            eval_mode=str(args.eval_mode),
            gates_raw=str(args.gates),
            subset_json=(None if str(args.subset_json).strip() == "" else str(args.subset_json)),
            pairwise_tmp_dir=(None if str(args.pairwise_tmp_dir).strip() == "" else str(args.pairwise_tmp_dir)),
            pairwise_extra_args=str(args.pairwise_extra_args),
            enable_tqdm=bool(args.enable_tqdm),
            profile_flops=bool(args.profile_flops),
            flops_profile_samples=int(args.flops_profile_samples),
        )

    try:
        evaluate_from_dirs(
            greedy_dir=str(greedy_dir),
            expand_dir=str(expand_dir),
            out_dir=str(args.out_dir),
            subset_json=(None if str(args.subset_json).strip() == "" else str(args.subset_json)),
            eval_mode=str(args.eval_mode),
            gates_raw=str(args.gates),
            policy=str(args.policy),
            trigger=str(args.trigger),
            extra_candidates_cost=int(args.extra_candidates_cost),
            base_ids_source=("greedy" if use_one_shot else "intersection"),
        )
    finally:
        if use_one_shot and cleanup_dir is not None and not bool(args.keep_pairwise_dirs):
            shutil.rmtree(cleanup_dir, ignore_errors=True)
            print("[cleaned]", os.path.abspath(cleanup_dir))


if __name__ == "__main__":
    main()
