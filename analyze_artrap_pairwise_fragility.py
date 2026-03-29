#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
except Exception:
    torch = None
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable

IMAGE_TOKEN_INDEX = None
DEFAULT_IMAGE_TOKEN = None
DEFAULT_IM_START_TOKEN = None
DEFAULT_IM_END_TOKEN = None
conv_templates = None
load_pretrained_model = None
get_model_name_from_path = None
process_images = None
tokenizer_image_token = None
disable_torch_init = None
Image = None


FORMAT_PHRASES = [
    "the answer is",
    "answer is",
    "it is",
    "it's",
    "there is",
    "there are",
]

YESNO_Q_PREFIXES = (
    "is ",
    "are ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
    "has ",
    "have ",
    "had ",
    "was ",
    "were ",
)

COLOR_WORDS = {
    "black", "white", "red", "green", "blue", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "gold", "silver", "beige", "tan",
}
SIDE_WORDS = {"left", "right", "middle", "center", "top", "bottom"}
PERSON_WORDS = {"man", "woman", "boy", "girl", "person", "people", "child", "guy", "lady"}
FORMAT_PREFIX_WORDS = {
    "the", "a", "an", "there", "is", "are", "it", "this", "that", "these", "those",
    "in", "on", "at", "of", "to", "for", "with", "from", "by", "as", "and", "or",
    "yes", "no", "based", "according",
}


def no_grad_fn(fn):
    if torch is None:
        return fn
    return torch.no_grad()(fn)


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def mean_or_none(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return None
    qq = min(1.0, max(0.0, float(q)))
    if len(xs) == 1:
        return float(xs[0])
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def iqr(vals: Sequence[float]) -> Optional[float]:
    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)
    if q1 is None or q3 is None:
        return None
    return float(q3 - q1)


def norm_text(x: Any) -> str:
    s = str("" if x is None else x).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_questions(path: str) -> List[Dict[str, Any]]:
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


def resolve_conv_mode(model_name: str, override: Optional[str]) -> str:
    if override:
        return str(override)
    m = str(model_name).lower()
    if "llama-2" in m:
        return "llava_llama_2"
    if "mistral" in m:
        return "mistral_instruct"
    if "v1.6-34b" in m:
        return "chatml_direct"
    if "v1" in m:
        return "llava_v1"
    if "mpt" in m:
        return "mpt"
    return "llava_v0"


def build_prompt(question: str, conv_mode: str, with_image_token: bool, mm_use_im_start_end: bool) -> str:
    q = str(question or "").strip()
    if with_image_token:
        if mm_use_im_start_end:
            head = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            head = DEFAULT_IMAGE_TOKEN
        user = head + "\n" + q
    else:
        user = q
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def is_yesno_question(question: str) -> bool:
    q = norm_text(question)
    return any(q.startswith(p) for p in YESNO_Q_PREFIXES)


def extract_core_answer_text(question: str, text: str, max_words: int = 6) -> str:
    clause = first_clause(text)
    clause_n = norm_text(clause)
    q = norm_text(question)
    if clause_n == "":
        return ""

    # question-type targeted extraction
    if is_yesno_question(question):
        # yes/no may appear after discourse wrappers (e.g., "The answer is yes").
        m = re.search(r"\b(yes|no)\b", clause_n)
        if m:
            return m.group(1)

    if "which side" in q or "on which side" in q or "left or right" in q:
        for w in ("left", "right", "middle", "center", "top", "bottom"):
            if contains_whole(w, clause_n):
                return w

    if "what color" in q or "which color" in q:
        for c in sorted(COLOR_WORDS):
            if contains_whole(c, clause_n):
                return c

    if "what gender" in q:
        for w in ("male", "female", "man", "woman", "boy", "girl"):
            if contains_whole(w, clause_n):
                return w

    if q.startswith("who "):
        for w in ("man", "woman", "boy", "girl", "person", "people", "child"):
            if contains_whole(w, clause_n):
                return w

    # generic cleanup: remove discourse wrappers then keep the first content chunk
    s = str(clause).strip()
    s = re.sub(r"^(yes|no)\b[\s,:\-]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^there\s+(is|are)\b[\s,:\-]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^it\s+is\b[\s,:\-]*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(the|a|an)\b\s+", "", s, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9']+", s)
    if not words:
        words = re.findall(r"[A-Za-z0-9']+", clause)
    if not words:
        return ""
    out: List[str] = []
    for w in words:
        wl = w.lower()
        if len(out) == 0 and wl in FORMAT_PREFIX_WORDS:
            continue
        out.append(w)
        if len(out) >= int(max_words):
            break
    if len(out) == 0:
        out = words[: int(max_words)]
    return " ".join(out).strip()


def first_clause(text: str) -> str:
    s = str(text or "").strip()
    s = re.split(r"[\n\.!?;]", s)[0].strip()
    return s


def build_phrase_id_map(tokenizer, phrases: Sequence[str]) -> List[Tuple[str, List[int]]]:
    rows: List[Tuple[str, List[int]]] = []
    seen = set()
    for p in phrases:
        base = str(p).strip()
        if base == "":
            continue
        vars_set = {
            base,
            base.lower(),
            base.capitalize(),
            base.title(),
            " " + base,
            " " + base.lower(),
            " " + base.capitalize(),
            " " + base.title(),
        }
        for vv in vars_set:
            ids = tokenizer(vv, add_special_tokens=False).input_ids
            key = tuple(int(x) for x in ids)
            if len(key) == 0 or key in seen:
                continue
            seen.add(key)
            rows.append((base, [int(x) for x in key]))
    # Prefer longer phrase matches when multiple patterns fit.
    rows.sort(key=lambda x: len(x[1]), reverse=True)
    return rows


def contains_whole(needle: str, hay: str) -> bool:
    if needle == "":
        return False
    pat = rf"(^|\s){re.escape(needle)}(\s|$)"
    return re.search(pat, hay) is not None


def singularize_word(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def map_gender_tokens(s: str) -> str:
    toks = s.split()
    out: List[str] = []
    for t in toks:
        if t in {"man", "male", "boy", "gentleman", "guy"}:
            out.append("male")
        elif t in {"woman", "female", "girl", "lady"}:
            out.append("female")
        else:
            out.append(t)
    return " ".join(out)


def first_polarity(s: str) -> Optional[str]:
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    return None


def is_success_heuristic(question: str, answer: str, champ_text: str, champ_short: str) -> bool:
    q = norm_text(question)
    gt = norm_text(answer)
    pt = norm_text(champ_text)
    ps = norm_text(champ_short)
    if gt == "":
        return False

    if gt == ps or gt == pt:
        return True

    if gt in {"yes", "no"}:
        pol = first_polarity(pt) or first_polarity(ps)
        if pol is not None:
            return pol == gt
        return contains_whole(gt, pt)

    gt_gender = map_gender_tokens(gt)
    pt_gender = map_gender_tokens(pt)
    ps_gender = map_gender_tokens(ps)
    if gt_gender in {"male", "female"}:
        if contains_whole(gt_gender, pt_gender) or contains_whole(gt_gender, ps_gender):
            return True

    if len(gt.split()) >= 2:
        if contains_whole(gt, pt) or contains_whole(gt, ps):
            return True

    if len(gt.split()) == 1:
        g = singularize_word(gt)
        toks_t = [singularize_word(t) for t in pt.split()]
        toks_s = [singularize_word(t) for t in ps.split()]
        if g in toks_t or g in toks_s:
            return True

    if contains_whole(gt, pt):
        return True

    if "what color" in q or "which color" in q:
        return contains_whole(gt, pt)
    if "which side" in q or "left or right" in q:
        return contains_whole(gt, pt)
    return False


def percentile_rank(vals: Sequence[float], v: float) -> Optional[float]:
    xs = [float(x) for x in vals if math.isfinite(float(x))]
    if not xs:
        return None
    return float(sum(1 for x in xs if x <= float(v)) / len(xs))


def stats_of(vals: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "iqr": None, "min": None, "max": None}
    return {
        "n": int(len(xs)),
        "mean": float(sum(xs) / len(xs)),
        "median": quantile(xs, 0.5),
        "iqr": iqr(xs),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


@dataclass
class Candidate:
    cand_idx: int
    token_ids: List[int]
    text: str
    short_answer: str
    s_full: Optional[float]
    s_format_img: Optional[float]
    s_core_img: Optional[float]
    s_ans_q: Optional[float]
    s_core_img_min: Optional[float]
    s_ans_q_min: Optional[float]
    vpmi_core_mean: Optional[float]
    vpmi_core_min: Optional[float]
    vpmi_core_min_raw: Optional[float]
    vpmi_core_min_prior_masked: Optional[float]
    vpmi_word_min: Optional[float]
    vpmi_core_tail_min: Optional[float]
    vpmi_core_min_pos_norm: Optional[float]
    vpmi_core_min_mean_gap: Optional[float]
    vpmi_core_sign_flip_count: Optional[float]
    margin_core_img_min: Optional[float]
    margin_core_img_mean: Optional[float]
    core_img_toks: Optional[List[float]]
    core_q_toks: Optional[List[float]]
    core_vpmi_toks: Optional[List[float]]
    format_len: int
    core_len: int
    core_start: int
    core_span_source: str


@dataclass
class GeneratedCandidate:
    token_ids: List[int]
    token_logps_img: Optional[List[float]]


@dataclass
class PrefixPath:
    token_ids: List[int]
    token_logps: List[float]
    score_sum: float


def detect_format_len(tokenizer, cand_ids: List[int], phrase_id_map: List[Tuple[str, List[int]]]) -> Tuple[int, Optional[str]]:
    best_len = 0
    best_phrase: Optional[str] = None
    for phrase, pids in phrase_id_map:
        if not pids:
            continue
        plen = int(len(pids))
        if plen <= best_len:
            continue
        if len(cand_ids) >= plen and cand_ids[:plen] == pids:
            best_len = plen
            best_phrase = phrase
    return int(best_len), best_phrase


def build_core_token_variants(tokenizer, core_text: str) -> List[List[int]]:
    raw = str(core_text or "").strip()
    base = norm_text(core_text)
    if raw == "" and base == "":
        return []
    vals = set()
    for s in (raw, base):
        ss = str(s).strip()
        if ss == "":
            continue
        vals.update({
            ss,
            ss.lower(),
            ss.capitalize(),
            ss.title(),
        })
    # Singular variants help with one-word noun answers.
    for s in list(vals):
        if len(str(s).split()) == 1:
            vals.add(singularize_word(str(s).lower()))
    out: List[List[int]] = []
    seen = set()
    for v in vals:
        vv = str(v).strip()
        if vv == "":
            continue
        for s in (vv, " " + vv):
            ids = tokenizer(s, add_special_tokens=False).input_ids
            key = tuple(int(x) for x in ids)
            if len(key) == 0 or key in seen:
                continue
            seen.add(key)
            out.append([int(x) for x in key])
    out.sort(key=len, reverse=True)
    return out


def find_subsequence(seq: List[int], sub: List[int]) -> Optional[int]:
    if len(sub) == 0 or len(seq) < len(sub):
        return None
    m = len(sub)
    for i in range(0, len(seq) - m + 1):
        if seq[i:i + m] == sub:
            return int(i)
    return None


def locate_core_span_from_text(tokenizer, cand_ids: List[int], core_text: str) -> Tuple[Optional[int], List[int]]:
    variants = build_core_token_variants(tokenizer, core_text)
    best_start: Optional[int] = None
    best_ids: List[int] = []
    for vids in variants:
        pos = find_subsequence(cand_ids, vids)
        if pos is None:
            continue
        if best_start is None:
            best_start = int(pos)
            best_ids = [int(x) for x in vids]
            continue
        # prefer earlier, then longer.
        if int(pos) < int(best_start) or (int(pos) == int(best_start) and len(vids) > len(best_ids)):
            best_start = int(pos)
            best_ids = [int(x) for x in vids]
    return best_start, best_ids


def group_token_positions_by_word(tokenizer, token_ids: List[int]) -> List[List[int]]:
    """
    Group token positions into approximate word units via tokenizer boundary markers.
    - SentencePiece: token starts with '▁'
    - GPT/BPE: token starts with 'Ġ'
    """
    groups: List[List[int]] = []
    cur: List[int] = []
    for i, tid in enumerate(token_ids):
        tok = str(tokenizer.convert_ids_to_tokens(int(tid)))
        is_start = bool(i == 0 or tok.startswith("▁") or tok.startswith("Ġ"))
        if is_start:
            if len(cur) > 0:
                groups.append(cur)
            cur = [int(i)]
        else:
            cur.append(int(i))
    if len(cur) > 0:
        groups.append(cur)
    return groups


def infer_prefix_format_len_from_tokens(tokenizer, cand_ids: List[int], question: str) -> int:
    if len(cand_ids) == 0:
        return 0
    yesno_q = is_yesno_question(question)
    n = 0
    for i, tid in enumerate(cand_ids):
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        w = str(tok).replace("▁", "").replace("Ġ", "").lower()
        w = re.sub(r"[^a-z0-9']+", "", w)
        if w == "":
            n += 1
            continue
        if i == 0 and yesno_q and w in {"yes", "no"}:
            return 0
        if w in FORMAT_PREFIX_WORDS:
            n += 1
            continue
        break
    return int(n)


@no_grad_fn
def sequence_token_logps(
    model,
    prefix_ids: torch.Tensor,
    cont_ids: List[int],
    images_tensor: Optional[torch.Tensor],
    image_sizes: Optional[List[Tuple[int, int]]],
) -> Optional[List[float]]:
    if len(cont_ids) == 0:
        return None
    device = prefix_ids.device
    cont = torch.tensor(cont_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = torch.cat([prefix_ids, cont], dim=1)
    try:
        out = model(
            input_ids=input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    except Exception:
        return None
    logits = out.logits
    p = int(prefix_ids.size(1))
    t = int(cont.size(1))
    if p <= 0:
        return None
    seg = logits[:, p - 1: p - 1 + t, :]
    logp = torch.log_softmax(seg.float(), dim=-1)
    gather = logp.gather(dim=-1, index=cont.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gather[0].tolist()]


@no_grad_fn
def sequence_token_logps_batched(
    model,
    prefix_ids: torch.Tensor,
    cont_ids_list: List[List[int]],
    pad_token_id: Optional[int],
    images_tensor: Optional[torch.Tensor],
    image_sizes: Optional[List[Tuple[int, int]]],
) -> List[Optional[List[float]]]:
    out_rows: List[Optional[List[float]]] = [None] * int(len(cont_ids_list))
    valid: List[Tuple[int, List[int]]] = [(i, ids) for i, ids in enumerate(cont_ids_list) if len(ids) > 0]
    if len(valid) == 0:
        return out_rows

    device = prefix_ids.device
    p = int(prefix_ids.size(1))
    if p <= 0:
        return out_rows

    bsz = int(len(valid))
    max_t = int(max(len(ids) for _, ids in valid))
    if pad_token_id is None or int(pad_token_id) < 0:
        pad_token_id = 0

    prefix = prefix_ids.expand(bsz, -1)
    cont = torch.full((bsz, max_t), int(pad_token_id), dtype=torch.long, device=device)
    lens: List[int] = []
    for bi, (_, ids) in enumerate(valid):
        t = int(len(ids))
        lens.append(t)
        cont[bi, :t] = torch.tensor(ids, dtype=torch.long, device=device)

    input_ids = torch.cat([prefix, cont], dim=1)
    try:
        out = model(
            input_ids=input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    except Exception:
        return out_rows

    logits = out.logits
    seg = logits[:, p - 1: p - 1 + max_t, :]
    logp = torch.log_softmax(seg.float(), dim=-1)
    gather = logp.gather(dim=-1, index=cont.unsqueeze(-1)).squeeze(-1)
    for bi, (orig_idx, _) in enumerate(valid):
        t = int(lens[bi])
        out_rows[orig_idx] = [float(x) for x in gather[bi, :t].tolist()]
    return out_rows


@no_grad_fn
def sequence_token_top2_margin_batched(
    model,
    prefix_ids: torch.Tensor,
    cont_ids_list: List[List[int]],
    pad_token_id: Optional[int],
    images_tensor: Optional[torch.Tensor],
    image_sizes: Optional[List[Tuple[int, int]]],
) -> List[Optional[List[float]]]:
    out_rows: List[Optional[List[float]]] = [None] * int(len(cont_ids_list))
    valid: List[Tuple[int, List[int]]] = [(i, ids) for i, ids in enumerate(cont_ids_list) if len(ids) > 0]
    if len(valid) == 0:
        return out_rows

    device = prefix_ids.device
    p = int(prefix_ids.size(1))
    if p <= 0:
        return out_rows

    bsz = int(len(valid))
    max_t = int(max(len(ids) for _, ids in valid))
    if pad_token_id is None or int(pad_token_id) < 0:
        pad_token_id = 0

    prefix = prefix_ids.expand(bsz, -1)
    cont = torch.full((bsz, max_t), int(pad_token_id), dtype=torch.long, device=device)
    lens: List[int] = []
    for bi, (_, ids) in enumerate(valid):
        t = int(len(ids))
        lens.append(t)
        cont[bi, :t] = torch.tensor(ids, dtype=torch.long, device=device)

    input_ids = torch.cat([prefix, cont], dim=1)
    try:
        out = model(
            input_ids=input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    except Exception:
        return out_rows

    logits = out.logits
    seg = logits[:, p - 1: p - 1 + max_t, :]
    logp = torch.log_softmax(seg.float(), dim=-1)
    top2_vals = torch.topk(logp, k=2, dim=-1).values
    margin = top2_vals[:, :, 0] - top2_vals[:, :, 1]
    for bi, (orig_idx, _) in enumerate(valid):
        t = int(lens[bi])
        out_rows[orig_idx] = [float(x) for x in margin[bi, :t].tolist()]
    return out_rows


def mean_selected(xs: Optional[List[float]], idxs: List[int]) -> Optional[float]:
    if xs is None:
        return None
    vals = [float(xs[i]) for i in idxs if 0 <= i < len(xs)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


@no_grad_fn
def diagnose_correct_answer_img(
    model,
    tokenizer,
    prefix_ids_img: torch.Tensor,
    images_tensor: Optional[torch.Tensor],
    image_sizes: Optional[List[Tuple[int, int]]],
    answer_text: str,
    max_tokens: int,
) -> Dict[str, Any]:
    out = {
        "correct_variant_text": None,
        "correct_variant_token_ids_json": None,
        "correct_token_len": None,
        "correct_logp_img_mean": None,
        "correct_logp_img_min": None,
        "correct_prefix3_logp_img_mean": None,
        "correct_first_token_logp_img": None,
        "correct_first_token_rank_img": None,
        "correct_first_token_in_top6_img": None,
        "correct_first_token_in_top20_img": None,
    }
    if torch is None:
        return out
    raw = first_clause(answer_text)
    variants = build_core_token_variants(tokenizer, raw)
    if len(variants) == 0:
        return out
    max_t = int(max(1, max_tokens))

    best_ids: Optional[List[int]] = None
    best_logps: Optional[List[float]] = None
    best_mean: Optional[float] = None
    seen = set()
    for vids in variants:
        ids = [int(x) for x in vids]
        key = tuple(ids)
        if len(ids) == 0 or key in seen:
            continue
        seen.add(key)
        if len(ids) > max_t:
            continue
        lps = sequence_token_logps(
            model=model,
            prefix_ids=prefix_ids_img,
            cont_ids=ids,
            images_tensor=images_tensor,
            image_sizes=image_sizes,
        )
        if lps is None or len(lps) == 0:
            continue
        mm = float(sum(float(x) for x in lps) / len(lps))
        if best_mean is None or mm > float(best_mean):
            best_mean = mm
            best_ids = ids
            best_logps = [float(x) for x in lps]
    if best_ids is None or best_logps is None or len(best_ids) == 0:
        return out

    out["correct_variant_text"] = tokenizer.decode(best_ids, skip_special_tokens=True).strip()
    out["correct_variant_token_ids_json"] = json.dumps([int(x) for x in best_ids], ensure_ascii=False)
    out["correct_token_len"] = int(len(best_ids))
    out["correct_logp_img_mean"] = float(sum(best_logps) / len(best_logps))
    out["correct_logp_img_min"] = float(min(best_logps))
    out["correct_prefix3_logp_img_mean"] = float(sum(best_logps[: min(3, len(best_logps))]) / min(3, len(best_logps)))
    out["correct_first_token_logp_img"] = float(best_logps[0])

    try:
        ff = model(
            input_ids=prefix_ids_img,
            images=images_tensor,
            image_sizes=image_sizes,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        lp = torch.log_softmax(ff.logits[:, -1, :].float(), dim=-1)[0]
        tid0 = int(best_ids[0])
        lp0 = float(lp[tid0].item())
        rank = int((lp > lp[tid0]).sum().item() + 1)
        out["correct_first_token_logp_img"] = lp0
        out["correct_first_token_rank_img"] = rank
        out["correct_first_token_in_top6_img"] = bool(rank <= 6)
        out["correct_first_token_in_top20_img"] = bool(rank <= 20)
    except Exception:
        pass
    return out


@no_grad_fn
def generate_candidates(
    model,
    tokenizer,
    input_ids_img: torch.Tensor,
    images_tensor: torch.Tensor,
    image_sizes: List[Tuple[int, int]],
    question: str,
    phrase_id_map: List[Tuple[str, List[int]]],
    candidate_gen_mode: str,
    num_beams: int,
    num_beam_groups: int,
    diversity_penalty: float,
    num_return_sequences: int,
    anchor_depth: int,
    anchor_width: int,
    anchor_prefix_cap: int,
    anchor_completion_budget: int,
    num_extra_samples: int,
    extra_sample_temperature: float,
    extra_sample_top_p: float,
    extra_sample_top_k: int,
    max_new_tokens: int,
    eos_token_id: Optional[int],
) -> List[GeneratedCandidate]:
    prompt_ids = [int(x) for x in input_ids_img[0].tolist()]
    prompt_len = int(len(prompt_ids))

    def extract_new_ids_from_prompt(seq_ids: List[int]) -> List[int]:
        # LLaVA/HF may return either [prompt + new] or [new only] depending on generation path.
        if len(seq_ids) >= prompt_len and prompt_len > 0 and seq_ids[:prompt_len] == prompt_ids:
            ids = seq_ids[prompt_len:]
        else:
            ids = seq_ids
        # Defensive trim: keep only a reasonable tail when a full prompt leaked in with mismatch.
        if len(ids) > int(max_new_tokens):
            ids = ids[-int(max_new_tokens):]
        return [int(x) for x in ids]

    def compute_transition_logps(gen_out, sequences) -> Optional[List[List[float]]]:
        scores = getattr(gen_out, "scores", None)
        if scores is None or len(scores) == 0:
            return None
        if not hasattr(model, "compute_transition_scores"):
            return None
        beam_indices = getattr(gen_out, "beam_indices", None)
        try:
            ts = model.compute_transition_scores(
                sequences,
                scores,
                beam_indices=beam_indices,
                normalize_logits=True,
            )
        except Exception:
            try:
                ts = model.compute_transition_scores(
                    sequences,
                    scores,
                    normalize_logits=True,
                )
            except Exception:
                return None
        rows: List[List[float]] = []
        for r in ts:
            rows.append([float(x) for x in r.tolist()])
        return rows

    def extract_new_logps(raw_logps: Optional[List[float]], new_len: int) -> Optional[List[float]]:
        if raw_logps is None:
            return None
        if new_len <= 0:
            return []
        if len(raw_logps) < int(new_len):
            return None
        return [float(x) for x in raw_logps[: int(new_len)]]

    def run_generate(
        beam_n: int,
        ret_n: int,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        base_input_ids: Optional[torch.Tensor] = None,
        base_attention_mask: Optional[torch.Tensor] = None,
        base_images: Optional[torch.Tensor] = None,
        base_image_sizes: Optional[List[Tuple[int, int]]] = None,
        max_new_tokens_override: Optional[int] = None,
    ):
        src_ids = input_ids_img if base_input_ids is None else base_input_ids
        src_images = images_tensor if base_images is None else base_images
        src_image_sizes = image_sizes if base_image_sizes is None else base_image_sizes
        new_tok = int(max_new_tokens if max_new_tokens_override is None else max_new_tokens_override)
        gen_kwargs: Dict[str, Any] = {
            "images": src_images,
            "image_sizes": src_image_sizes,
            "do_sample": bool(do_sample),
            "temperature": float(max(0.0, temperature)),
            "num_beams": int(max(1, beam_n)),
            "num_return_sequences": int(max(1, ret_n)),
            "max_new_tokens": int(max(0, new_tok)),
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if bool(do_sample):
            gen_kwargs["temperature"] = float(max(1e-5, temperature))
            gen_kwargs["top_p"] = float(min(1.0, max(0.0, top_p)))
            if int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)
        if base_attention_mask is not None:
            gen_kwargs["attention_mask"] = base_attention_mask
        if int(max(1, num_beam_groups)) > 1 and int(max(1, beam_n)) > 1:
            gen_kwargs["num_beam_groups"] = int(max(1, num_beam_groups))
            gen_kwargs["diversity_penalty"] = float(diversity_penalty)
        return model.generate(src_ids, **gen_kwargs)

    def run_generate_from_embeds(
        *,
        base_inputs_embeds: torch.Tensor,
        base_attention_mask: Optional[torch.Tensor],
        beam_n: int,
        ret_n: int,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        max_new_tokens_override: Optional[int] = None,
    ):
        new_tok = int(max_new_tokens if max_new_tokens_override is None else max_new_tokens_override)
        gen_kwargs: Dict[str, Any] = {
            "do_sample": bool(do_sample),
            "temperature": float(max(0.0, temperature)),
            "num_beams": int(max(1, beam_n)),
            "num_return_sequences": int(max(1, ret_n)),
            "max_new_tokens": int(max(0, new_tok)),
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if bool(do_sample):
            gen_kwargs["temperature"] = float(max(1e-5, temperature))
            gen_kwargs["top_p"] = float(min(1.0, max(0.0, top_p)))
            if int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)
        if base_attention_mask is not None:
            gen_kwargs["attention_mask"] = base_attention_mask
        if int(max(1, num_beam_groups)) > 1 and int(max(1, beam_n)) > 1:
            gen_kwargs["num_beam_groups"] = int(max(1, num_beam_groups))
            gen_kwargs["diversity_penalty"] = float(diversity_penalty)

        # LLaVA wrapper blocks inputs_embeds in generate(); call parent generate directly.
        parent_generate = type(model).mro()[1].generate
        return parent_generate(model, inputs_embeds=base_inputs_embeds, **gen_kwargs)

    rows: List[GeneratedCandidate] = []
    seen = set()

    def append_rows_from_gen_out(
        gen_out,
        *,
        base_ids: List[int],
        prefix_ids: Optional[List[int]] = None,
        prefix_logps: Optional[List[float]] = None,
    ) -> None:
        pref_ids = [int(x) for x in (prefix_ids or [])]
        pref_lps = [float(x) for x in (prefix_logps or [])]
        sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
        trans_rows = compute_transition_logps(gen_out, sequences)
        for seq_i, seq in enumerate(sequences):
            seq_ids = [int(x) for x in seq.tolist()]
            tail_ids: List[int]
            if len(seq_ids) >= len(base_ids) and len(base_ids) > 0 and seq_ids[: len(base_ids)] == base_ids:
                tail_ids = [int(x) for x in seq_ids[len(base_ids):]]
                ids = [int(x) for x in pref_ids + tail_ids]
                tail_logps = extract_new_logps(
                    None if trans_rows is None or seq_i >= len(trans_rows) else trans_rows[seq_i],
                    new_len=len(tail_ids),
                )
                if (
                    tail_logps is not None
                    and len(pref_lps) == len(pref_ids)
                ):
                    logps = [float(x) for x in pref_lps + tail_logps]
                else:
                    logps = None
            elif len(seq_ids) >= prompt_len and prompt_len > 0 and seq_ids[:prompt_len] == prompt_ids:
                ids = [int(x) for x in seq_ids[prompt_len:]]
                logps = extract_new_logps(
                    None if trans_rows is None or seq_i >= len(trans_rows) else trans_rows[seq_i],
                    new_len=len(ids),
                )
            else:
                # generated-only sequence path (e.g., generate from cached inputs_embeds)
                ids_tail = [int(x) for x in seq_ids]
                if len(ids_tail) == 0:
                    ids_tail = extract_new_ids_from_prompt(seq_ids)
                tail_logps = extract_new_logps(
                    None if trans_rows is None or seq_i >= len(trans_rows) else trans_rows[seq_i],
                    new_len=len(ids_tail),
                )
                if len(pref_ids) > 0:
                    ids = [int(x) for x in pref_ids + ids_tail]
                    if tail_logps is not None and len(pref_lps) == len(pref_ids):
                        logps = [float(x) for x in pref_lps + tail_logps]
                    else:
                        logps = None
                else:
                    ids = [int(x) for x in ids_tail]
                    logps = tail_logps

            if len(ids) > int(max_new_tokens):
                ids = ids[: int(max_new_tokens)]
                if logps is not None:
                    logps = logps[: int(max_new_tokens)]

            if eos_token_id is not None and eos_token_id in ids:
                eos_pos = int(ids.index(int(eos_token_id)))
                ids = ids[:eos_pos]
                if logps is not None:
                    logps = logps[:eos_pos]
            if logps is not None and len(logps) == len(ids):
                pairs = [(int(tid), float(lp)) for tid, lp in zip(ids, logps) if int(tid) >= 0]
                ids = [int(tid) for tid, _ in pairs]
                logps = [float(lp) for _, lp in pairs]
            else:
                ids = [int(x) for x in ids if int(x) >= 0]
                logps = None
            key = tuple(ids)
            if len(key) == 0 or key in seen:
                continue
            seen.add(key)
            rows.append(GeneratedCandidate(token_ids=list(key), token_logps_img=logps))

    def append_rows_from_gen_out_multi(
        gen_out,
        *,
        base_ids_list: List[List[int]],
        prefix_ids_list: Optional[List[List[int]]] = None,
        prefix_logps_list: Optional[List[List[float]]] = None,
        ret_n: int = 1,
    ) -> None:
        if len(base_ids_list) == 0:
            return
        pref_ids_list = prefix_ids_list or [[] for _ in base_ids_list]
        pref_lps_list = prefix_logps_list or [[] for _ in base_ids_list]
        sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
        trans_rows = compute_transition_logps(gen_out, sequences)
        for seq_i, seq in enumerate(sequences):
            src_i = int(seq_i // max(1, int(ret_n)))
            if src_i >= len(base_ids_list):
                continue
            base_ids = [int(x) for x in base_ids_list[src_i]]
            pref_ids = [int(x) for x in pref_ids_list[src_i]]
            pref_lps = [float(x) for x in pref_lps_list[src_i]]
            seq_ids = [int(x) for x in seq.tolist()]
            tail_ids: List[int]
            if len(seq_ids) >= len(base_ids) and len(base_ids) > 0 and seq_ids[: len(base_ids)] == base_ids:
                tail_ids = [int(x) for x in seq_ids[len(base_ids):]]
                ids = [int(x) for x in pref_ids + tail_ids]
                tail_logps = extract_new_logps(
                    None if trans_rows is None or seq_i >= len(trans_rows) else trans_rows[seq_i],
                    new_len=len(tail_ids),
                )
                if tail_logps is not None and len(pref_lps) == len(pref_ids):
                    logps = [float(x) for x in pref_lps + tail_logps]
                else:
                    logps = None
            elif len(seq_ids) >= prompt_len and prompt_len > 0 and seq_ids[:prompt_len] == prompt_ids:
                ids = [int(x) for x in seq_ids[prompt_len:]]
                logps = extract_new_logps(
                    None if trans_rows is None or seq_i >= len(trans_rows) else trans_rows[seq_i],
                    new_len=len(ids),
                )
            else:
                # generated-only sequence path (e.g., generate from cached inputs_embeds)
                ids_tail = [int(x) for x in seq_ids]
                if len(ids_tail) == 0:
                    ids_tail = extract_new_ids_from_prompt(seq_ids)
                tail_logps = extract_new_logps(
                    None if trans_rows is None or seq_i >= len(trans_rows) else trans_rows[seq_i],
                    new_len=len(ids_tail),
                )
                if len(pref_ids) > 0:
                    ids = [int(x) for x in pref_ids + ids_tail]
                    if tail_logps is not None and len(pref_lps) == len(pref_ids):
                        logps = [float(x) for x in pref_lps + tail_logps]
                    else:
                        logps = None
                else:
                    ids = [int(x) for x in ids_tail]
                    logps = tail_logps
            if len(ids) > int(max_new_tokens):
                ids = ids[: int(max_new_tokens)]
                if logps is not None:
                    logps = logps[: int(max_new_tokens)]
            if eos_token_id is not None and eos_token_id in ids:
                eos_pos = int(ids.index(int(eos_token_id)))
                ids = ids[:eos_pos]
                if logps is not None:
                    logps = logps[:eos_pos]
            if logps is not None and len(logps) == len(ids):
                pairs = [(int(tid), float(lp)) for tid, lp in zip(ids, logps) if int(tid) >= 0]
                ids = [int(tid) for tid, _ in pairs]
                logps = [float(lp) for _, lp in pairs]
            else:
                ids = [int(x) for x in ids if int(x) >= 0]
                logps = None
            key = tuple(ids)
            if len(key) == 0 or key in seen:
                continue
            seen.add(key)
            rows.append(GeneratedCandidate(token_ids=list(key), token_logps_img=logps))

    def build_padded_id_batch(seqs: List[List[int]], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if len(seqs) == 0:
            return (
                torch.empty((0, 0), dtype=torch.long, device=input_ids_img.device),
                torch.empty((0, 0), dtype=torch.long, device=input_ids_img.device),
                [],
            )
        lens = [int(len(s)) for s in seqs]
        max_len = int(max(lens))
        ids = torch.full((len(seqs), max_len), int(pad_token_id), dtype=torch.long, device=input_ids_img.device)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=input_ids_img.device)
        for bi, s in enumerate(seqs):
            t = int(len(s))
            if t <= 0:
                continue
            ids[bi, :t] = torch.tensor([int(x) for x in s], dtype=torch.long, device=input_ids_img.device)
            attn[bi, :t] = 1
        return ids, attn, lens

    def repeat_images_for_batch(img: torch.Tensor, bsz: int) -> torch.Tensor:
        if int(img.size(0)) == int(bsz):
            return img
        if int(img.size(0)) == 1:
            return img.expand(int(bsz), *img.shape[1:]).contiguous()
        rep = int(math.ceil(float(bsz) / float(img.size(0))))
        tiled = img.repeat((rep,) + (1,) * (img.ndim - 1))
        return tiled[: int(bsz)].contiguous()

    mode = str(candidate_gen_mode).strip().lower()
    if mode == "core_anchor":
        width = int(max(1, anchor_width))
        completion_budget = int(max(1, anchor_completion_budget))
        pad_tok = int(getattr(tokenizer, "pad_token_id", -1))
        if pad_tok < 0:
            pad_tok = int(getattr(tokenizer, "eos_token_id", 0) or 0)
        prompt_mm_embeds: Optional[torch.Tensor] = None
        prompt_mm_len: int = 0

        # Step 0) Get greedy anchor response once.
        greedy_out = run_generate(beam_n=1, ret_n=1, do_sample=False, temperature=0.0)
        append_rows_from_gen_out(greedy_out, base_ids=prompt_ids)

        greedy_seq = None
        if hasattr(greedy_out, "sequences"):
            seqs = greedy_out.sequences
            if seqs is not None and len(seqs) > 0:
                greedy_seq = [int(x) for x in seqs[0].tolist()]
        if greedy_seq is None:
            return rows
        if len(greedy_seq) >= prompt_len and prompt_len > 0 and greedy_seq[:prompt_len] == prompt_ids:
            greedy_ids = [int(x) for x in greedy_seq[prompt_len:]]
        else:
            greedy_ids = extract_new_ids_from_prompt(greedy_seq)
        if len(greedy_ids) > int(max_new_tokens):
            greedy_ids = greedy_ids[: int(max_new_tokens)]
        if eos_token_id is not None and int(eos_token_id) in greedy_ids:
            greedy_ids = greedy_ids[: int(greedy_ids.index(int(eos_token_id)))]
        if len(greedy_ids) == 0:
            return rows

        # Step 1) Fast-forward to estimated core start from greedy formatting prefix.
        format_len_phrase, _ = detect_format_len(tokenizer, greedy_ids, phrase_id_map)
        format_len_prefix = infer_prefix_format_len_from_tokens(tokenizer, greedy_ids, question)
        core_start = int(max(int(format_len_phrase), int(format_len_prefix)))
        core_start = int(min(max(0, core_start), max(0, len(greedy_ids) - 1)))
        prefix_ids = [int(x) for x in greedy_ids[:core_start]]

        # Optional prefix logps for exact stitched candidate logprobs.
        prefix_logps: List[float] = []
        greedy_logps = sequence_token_logps(
            model=model,
            prefix_ids=input_ids_img,
            cont_ids=[int(x) for x in greedy_ids],
            images_tensor=images_tensor,
            image_sizes=image_sizes,
        )
        if greedy_logps is not None and len(greedy_logps) >= int(core_start):
            prefix_logps = [float(x) for x in greedy_logps[: int(core_start)]]

        # Prepare multimodal prompt embeds once for no-image-recompute path.
        try:
            prompt_attn = torch.ones_like(input_ids_img, dtype=torch.long, device=input_ids_img.device)
            _, _, _, _, mm_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                input_ids_img,
                None,
                prompt_attn,
                None,
                None,
                images_tensor,
                image_sizes,
            )
            if mm_embeds is not None and int(mm_embeds.shape[0]) == 1:
                prompt_mm_embeds = mm_embeds[0].detach()
                prompt_mm_len = int(prompt_mm_embeds.shape[0])
        except Exception:
            prompt_mm_embeds = None
            prompt_mm_len = 0

        # Step 2) Core-anchor top-k branching.
        base_ids = [int(x) for x in (prompt_ids + prefix_ids)]
        if prompt_mm_embeds is not None and int(prompt_mm_len) > 0:
            try:
                base_len = int(len(prefix_ids))
                total_len = int(prompt_mm_len + base_len)
                hidden = int(prompt_mm_embeds.shape[-1])
                base_embeds = prompt_mm_embeds.new_zeros((1, total_len, hidden))
                base_mask = torch.zeros((1, total_len), dtype=torch.long, device=input_ids_img.device)
                base_embeds[0, :prompt_mm_len, :] = prompt_mm_embeds
                if base_len > 0:
                    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=input_ids_img.device)
                    prefix_embeds = model.get_model().embed_tokens(prefix_tensor)
                    base_embeds[0, prompt_mm_len:prompt_mm_len + base_len, :] = prefix_embeds[0, :base_len, :]
                base_mask[0, :total_len] = 1
                out = model(
                    inputs_embeds=base_embeds,
                    attention_mask=base_mask,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
                lp = torch.log_softmax(out.logits[:, total_len - 1, :].float(), dim=-1)[0]
            except Exception:
                return rows
        else:
            base = torch.tensor(base_ids, dtype=torch.long, device=input_ids_img.device).unsqueeze(0)
            try:
                out = model(
                    input_ids=base,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
                lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0]
            except Exception:
                return rows

        kk = int(min(width, int(lp.numel())))
        if kk <= 0:
            return rows
        vals, idxs = torch.topk(lp, k=kk, dim=-1)

        paths: List[PrefixPath] = []
        for v, tid in zip(vals.tolist(), idxs.tolist()):
            tid_i = int(tid)
            if eos_token_id is not None and tid_i == int(eos_token_id):
                continue
            if len(prefix_logps) == int(core_start):
                path_logps = [float(x) for x in (prefix_logps + [float(v)])]
            else:
                path_logps = []
            paths.append(
                PrefixPath(
                    token_ids=[int(x) for x in (prefix_ids + [tid_i])],
                    token_logps=path_logps,
                    score_sum=float(v),
                )
            )
        if len(paths) == 0:
            return rows

        paths = sorted(paths, key=lambda x: float(x.score_sum), reverse=True)[:completion_budget]

        # Step 3) Batched greedy completion from branched prefixes.
        grouped: Dict[int, List[PrefixPath]] = {}
        for p in paths:
            rem = int(max_new_tokens) - int(len(p.token_ids))
            grouped.setdefault(int(rem), []).append(p)
        for rem, group in grouped.items():
            if rem <= 0:
                for p in group:
                    key = tuple(int(x) for x in p.token_ids[: int(max_new_tokens)])
                    if len(key) > 0 and key not in seen:
                        seen.add(key)
                        rows.append(
                            GeneratedCandidate(
                                token_ids=list(key),
                                token_logps_img=[float(x) for x in p.token_logps[: len(key)]]
                                if len(p.token_logps) >= len(key)
                                else None,
                            )
                        )
                continue
            base_ids_list = [[int(x) for x in (prompt_ids + p.token_ids)] for p in group]
            gen_out = None

            # Preferred path: generate from cached multimodal embeds (no ViT re-encode).
            if prompt_mm_embeds is not None and int(prompt_mm_len) > 0:
                try:
                    bsz = int(len(group))
                    max_plen = int(max(len(p.token_ids) for p in group))
                    hidden = int(prompt_mm_embeds.shape[-1])
                    total_len = int(prompt_mm_len + max_plen)
                    batch_embeds = prompt_mm_embeds.new_zeros((bsz, total_len, hidden))
                    batch_mask = torch.zeros((bsz, total_len), dtype=torch.long, device=input_ids_img.device)
                    path_ids = torch.full((bsz, max_plen), int(pad_tok), dtype=torch.long, device=input_ids_img.device)
                    for bi, p in enumerate(group):
                        t = int(len(p.token_ids))
                        if t > 0:
                            path_ids[bi, :t] = torch.tensor([int(x) for x in p.token_ids], dtype=torch.long, device=input_ids_img.device)
                    path_embeds = model.get_model().embed_tokens(path_ids) if max_plen > 0 else None
                    for bi, p in enumerate(group):
                        t = int(len(p.token_ids))
                        cur_len = int(prompt_mm_len + t)
                        batch_embeds[bi, :prompt_mm_len, :] = prompt_mm_embeds
                        if t > 0 and path_embeds is not None:
                            batch_embeds[bi, prompt_mm_len:prompt_mm_len + t, :] = path_embeds[bi, :t, :]
                        batch_mask[bi, :cur_len] = 1
                    gen_out = run_generate_from_embeds(
                        base_inputs_embeds=batch_embeds,
                        base_attention_mask=batch_mask,
                        beam_n=1,
                        ret_n=1,
                        do_sample=False,
                        temperature=0.0,
                        max_new_tokens_override=int(rem),
                    )
                except Exception:
                    gen_out = None

            # Fallback path.
            if gen_out is None:
                base_batch, base_mask, _ = build_padded_id_batch(base_ids_list, pad_tok)
                bsz = int(base_batch.size(0))
                images_batch = repeat_images_for_batch(images_tensor, bsz)
                img_size = tuple(image_sizes[0]) if len(image_sizes) > 0 else None
                image_sizes_batch = ([img_size] * bsz) if img_size is not None else image_sizes
                gen_out = run_generate(
                    beam_n=1,
                    ret_n=1,
                    do_sample=False,
                    temperature=0.0,
                    base_input_ids=base_batch,
                    base_attention_mask=base_mask,
                    base_images=images_batch,
                    base_image_sizes=image_sizes_batch,
                    max_new_tokens_override=int(rem),
                )
            append_rows_from_gen_out_multi(
                gen_out,
                base_ids_list=base_ids_list,
                prefix_ids_list=[[int(x) for x in p.token_ids] for p in group],
                prefix_logps_list=[
                    [float(x) for x in p.token_logps] if len(p.token_logps) > 0 else []
                    for p in group
                ],
                ret_n=1,
            )
    elif mode == "anchor_branch":
        depth = int(max(1, anchor_depth))
        width = int(max(1, anchor_width))
        prefix_cap = int(max(1, anchor_prefix_cap))
        completion_budget = int(max(1, anchor_completion_budget))
        pad_tok = int(getattr(tokenizer, "pad_token_id", -1))
        if pad_tok < 0:
            pad_tok = int(getattr(tokenizer, "eos_token_id", 0) or 0)

        prompt_mm_embeds: Optional[torch.Tensor] = None
        prompt_mm_len: int = 0
        try:
            prompt_attn = torch.ones_like(input_ids_img, dtype=torch.long, device=input_ids_img.device)
            _, _, _, _, mm_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                input_ids_img,
                None,
                prompt_attn,
                None,
                None,
                images_tensor,
                image_sizes,
            )
            if mm_embeds is not None and int(mm_embeds.shape[0]) == 1:
                prompt_mm_embeds = mm_embeds[0].detach()
                prompt_mm_len = int(prompt_mm_embeds.shape[0])
        except Exception:
            prompt_mm_embeds = None
            prompt_mm_len = 0

        paths: List[PrefixPath] = [PrefixPath(token_ids=[], token_logps=[], score_sum=0.0)]
        for _ in range(depth):
            next_paths: List[PrefixPath] = []
            if len(paths) == 0:
                break
            if prompt_mm_embeds is not None and int(prompt_mm_len) > 0:
                try:
                    # Build batched multimodal embeds once and run a single forward.
                    bsz = int(len(paths))
                    max_plen = int(max(len(p.token_ids) for p in paths))
                    hidden = int(prompt_mm_embeds.shape[-1])
                    batch_embeds = prompt_mm_embeds.new_zeros((bsz, prompt_mm_len + max_plen, hidden))
                    batch_mask = torch.zeros((bsz, prompt_mm_len + max_plen), dtype=torch.long, device=input_ids_img.device)
                    if max_plen > 0:
                        path_ids = torch.full((bsz, max_plen), int(pad_tok), dtype=torch.long, device=input_ids_img.device)
                        for bi, p in enumerate(paths):
                            t = int(len(p.token_ids))
                            if t > 0:
                                path_ids[bi, :t] = torch.tensor([int(x) for x in p.token_ids], dtype=torch.long, device=input_ids_img.device)
                        path_embeds = model.get_model().embed_tokens(path_ids)
                    else:
                        path_embeds = None
                    for bi, p in enumerate(paths):
                        t = int(len(p.token_ids))
                        cur_len = int(prompt_mm_len + t)
                        batch_embeds[bi, :prompt_mm_len, :] = prompt_mm_embeds
                        if t > 0 and path_embeds is not None:
                            batch_embeds[bi, prompt_mm_len:prompt_mm_len + t, :] = path_embeds[bi, :t, :]
                        batch_mask[bi, :cur_len] = 1
                    out = model(
                        inputs_embeds=batch_embeds,
                        attention_mask=batch_mask,
                        use_cache=False,
                        output_attentions=False,
                        return_dict=True,
                    )
                    logits = out.logits
                    for bi, path in enumerate(paths):
                        cur_len = int(prompt_mm_len + len(path.token_ids))
                        if cur_len <= 0:
                            continue
                        lp = torch.log_softmax(logits[bi, cur_len - 1, :].float(), dim=-1)
                        kk = int(min(width, int(lp.numel())))
                        if kk <= 0:
                            continue
                        vals, idxs = torch.topk(lp, k=kk, dim=-1)
                        for v, tid in zip(vals.tolist(), idxs.tolist()):
                            tid_i = int(tid)
                            if eos_token_id is not None and tid_i == int(eos_token_id):
                                continue
                            next_paths.append(
                                PrefixPath(
                                    token_ids=[int(x) for x in (path.token_ids + [tid_i])],
                                    token_logps=[float(x) for x in (path.token_logps + [float(v)])],
                                    score_sum=float(path.score_sum + float(v)),
                                )
                            )
                except Exception:
                    next_paths = []
            if len(next_paths) == 0:
                # Fallback path when cached multimodal embed route fails: batched input_ids + repeated image.
                try:
                    base_ids_list = [[int(x) for x in (prompt_ids + p.token_ids)] for p in paths]
                    base_batch, base_mask, base_lens = build_padded_id_batch(base_ids_list, pad_tok)
                    bsz = int(base_batch.size(0))
                    images_batch = repeat_images_for_batch(images_tensor, bsz)
                    img_size = tuple(image_sizes[0]) if len(image_sizes) > 0 else None
                    image_sizes_batch = ([img_size] * bsz) if img_size is not None else image_sizes
                    out = model(
                        input_ids=base_batch,
                        attention_mask=base_mask,
                        images=images_batch,
                        image_sizes=image_sizes_batch,
                        use_cache=False,
                        output_attentions=False,
                        return_dict=True,
                    )
                    logits = out.logits
                    for bi, path in enumerate(paths):
                        cur_len = int(base_lens[bi])
                        if cur_len <= 0:
                            continue
                        lp = torch.log_softmax(logits[bi, cur_len - 1, :].float(), dim=-1)
                        kk = int(min(width, int(lp.numel())))
                        if kk <= 0:
                            continue
                        vals, idxs = torch.topk(lp, k=kk, dim=-1)
                        for v, tid in zip(vals.tolist(), idxs.tolist()):
                            tid_i = int(tid)
                            if eos_token_id is not None and tid_i == int(eos_token_id):
                                continue
                            next_paths.append(
                                PrefixPath(
                                    token_ids=[int(x) for x in (path.token_ids + [tid_i])],
                                    token_logps=[float(x) for x in (path.token_logps + [float(v)])],
                                    score_sum=float(path.score_sum + float(v)),
                                )
                            )
                except Exception:
                    next_paths = []
            if len(next_paths) == 0:
                break

            best_by_key: Dict[Tuple[int, ...], PrefixPath] = {}
            for p in next_paths:
                k = tuple(int(x) for x in p.token_ids)
                cur = best_by_key.get(k)
                if cur is None or float(p.score_sum) > float(cur.score_sum):
                    best_by_key[k] = p
            paths = sorted(
                list(best_by_key.values()),
                key=lambda x: float(x.score_sum),
                reverse=True,
            )[:prefix_cap]

        if len(paths) == 0:
            gen_out = run_generate(beam_n=1, ret_n=1, do_sample=False, temperature=0.0)
            append_rows_from_gen_out(gen_out, base_ids=prompt_ids)
        else:
            grouped: Dict[int, List[PrefixPath]] = {}
            for p in paths[:completion_budget]:
                rem = int(max_new_tokens) - int(len(p.token_ids))
                grouped.setdefault(int(rem), []).append(p)
            for rem, group in grouped.items():
                if rem <= 0:
                    for p in group:
                        key = tuple(int(x) for x in p.token_ids[: int(max_new_tokens)])
                        if len(key) > 0 and key not in seen:
                            seen.add(key)
                            rows.append(
                                GeneratedCandidate(
                                    token_ids=list(key),
                                    token_logps_img=[float(x) for x in p.token_logps[: len(key)]],
                                )
                            )
                    continue
                base_ids_list = [[int(x) for x in (prompt_ids + p.token_ids)] for p in group]
                base_batch, base_mask, _ = build_padded_id_batch(base_ids_list, pad_tok)
                bsz = int(base_batch.size(0))
                images_batch = repeat_images_for_batch(images_tensor, bsz)
                img_size = tuple(image_sizes[0]) if len(image_sizes) > 0 else None
                image_sizes_batch = ([img_size] * bsz) if img_size is not None else image_sizes
                gen_out = run_generate(
                    beam_n=1,
                    ret_n=1,
                    do_sample=False,
                    temperature=0.0,
                    base_input_ids=base_batch,
                    base_attention_mask=base_mask,
                    base_images=images_batch,
                    base_image_sizes=image_sizes_batch,
                    max_new_tokens_override=int(rem),
                )
                append_rows_from_gen_out_multi(
                    gen_out,
                    base_ids_list=base_ids_list,
                    prefix_ids_list=[[int(x) for x in p.token_ids] for p in group],
                    prefix_logps_list=[[float(x) for x in p.token_logps] for p in group],
                    ret_n=1,
                )
    else:
        gen_out = run_generate(
            beam_n=int(max(1, num_beams)),
            ret_n=int(max(1, min(num_beams, num_return_sequences))),
            do_sample=False,
            temperature=0.0,
        )
        append_rows_from_gen_out(gen_out, base_ids=prompt_ids)

    # Optional additional sampled candidates to improve coverage with compact beam.
    if int(num_extra_samples) > 0:
        try:
            gen_out_sample = run_generate(
                beam_n=1,
                ret_n=int(max(1, num_extra_samples)),
                do_sample=True,
                temperature=float(extra_sample_temperature),
                top_p=float(extra_sample_top_p),
                top_k=int(extra_sample_top_k),
            )
            append_rows_from_gen_out(gen_out_sample, base_ids=prompt_ids)
        except Exception:
            pass

    # Fallback: keep at least one candidate so downstream analysis doesn't collapse.
    if len(rows) == 0:
        gen_out = run_generate(beam_n=1, ret_n=1, do_sample=False, temperature=0.0)
        append_rows_from_gen_out(gen_out, base_ids=prompt_ids)
    return rows


def bin_index(v: Optional[float], edges: List[float]) -> Optional[int]:
    if v is None or not math.isfinite(float(v)):
        return None
    x = float(v)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if (x >= lo and x < hi) or (i == len(edges) - 2 and x == hi):
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Vanilla LLaVA AR-Trap + Pairwise Fragility analyzer")
    ap.add_argument("--questions_json", type=str, required=True, help="GQA-like json (dict or list)")
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--num_beams", type=int, default=6)
    ap.add_argument("--num_beam_groups", type=int, default=1)
    ap.add_argument("--diversity_penalty", type=float, default=0.0)
    ap.add_argument("--num_return_sequences", type=int, default=6)
    ap.add_argument(
        "--candidate_gen_mode",
        type=str,
        default="beam",
        choices=["beam", "anchor_branch", "core_anchor"],
        help="Candidate generation mode: standard beam, limited anchor branching, or core-anchored batched greedy.",
    )
    ap.add_argument("--anchor_depth", type=int, default=2, help="Anchor branching depth L.")
    ap.add_argument("--anchor_width", type=int, default=4, help="Top-k width per anchor step.")
    ap.add_argument("--anchor_prefix_cap", type=int, default=6, help="Max prefixes retained after each anchor step.")
    ap.add_argument("--anchor_completion_budget", type=int, default=6, help="Number of anchor prefixes to greedily complete.")
    ap.add_argument("--num_extra_samples", type=int, default=0, help="Additional sampled candidates to append.")
    ap.add_argument("--extra_sample_temperature", type=float, default=1.0)
    ap.add_argument("--extra_sample_top_p", type=float, default=0.9)
    ap.add_argument("--extra_sample_top_k", type=int, default=0, help="0 disables top-k filtering.")
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--answer_span_max_tokens", type=int, default=4)
    ap.add_argument(
        "--vpmi_min_mode",
        type=str,
        default="raw",
        choices=["raw", "prior_masked", "word_min"],
        help="How vpmi_core_min is computed from core-token VPMI.",
    )
    ap.add_argument(
        "--vpmi_min_sq_logp_max",
        type=float,
        default=-0.1,
        help="For prior_masked mode: keep tokens with S_q <= threshold (exclude deterministic high-prior suffixes).",
    )
    ap.add_argument(
        "--vpmi_min_fallback",
        type=str,
        default="raw",
        choices=["raw", "none"],
        help="Fallback when selected vpmi_min_mode has no valid token/word.",
    )
    ap.add_argument(
        "--save_top2_margin",
        action="store_true",
        help="Save true token-level top1-top2 logprob margin stats for core span.",
    )
    ap.add_argument(
        "--save_token_ids_json",
        action="store_true",
        help="Save full candidate token ids as JSON in per_candidate.csv.",
    )
    ap.add_argument(
        "--save_core_tokenwise_vpmi",
        action="store_true",
        help="Save core tokenwise image/q/vpmi arrays as JSON in per_candidate.csv.",
    )
    ap.add_argument(
        "--diag_correct_answer",
        action="store_true",
        help="Compute diagnostic logprobs/ranks for gold answer tokens per sample.",
    )
    ap.add_argument(
        "--diag_correct_max_tokens",
        type=int,
        default=8,
        help="Max token length of gold-answer variant used for diagnostics.",
    )

    ap.add_argument("--beta_q", type=float, default=0.8, help="safe challenger prior upper bound")
    ap.add_argument("--tau_gap", type=float, default=0.65, help="fragile margin threshold")
    ap.add_argument("--eval_match_mode", type=str, default="heuristic", choices=["strict", "heuristic"])

    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    random.seed(int(args.seed))
    if torch is not None:
        torch.manual_seed(int(args.seed))

    if int(args.num_beam_groups) < 1:
        raise ValueError("--num_beam_groups must be >= 1")
    if int(args.num_beams) < int(args.num_beam_groups):
        raise ValueError("--num_beams must be >= --num_beam_groups")
    if int(args.num_beams) % int(args.num_beam_groups) != 0:
        raise ValueError("--num_beams must be divisible by --num_beam_groups for diverse beam search")
    if int(args.num_beam_groups) > 1 and float(args.diversity_penalty) <= 0.0:
        raise ValueError("--diversity_penalty must be > 0 when --num_beam_groups > 1")

    qpath = os.path.abspath(args.questions_json)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = read_questions(qpath)
    if int(args.num_samples) > 0:
        rows = rows[: int(args.num_samples)]

    if args.dry_run:
        dry = {
            "inputs": {
                "questions_json": qpath,
                "image_root": os.path.abspath(args.image_root),
                "model_path": str(args.model_path),
                "num_samples": int(len(rows)),
                "eval_match_mode": str(args.eval_match_mode),
            },
            "counts": {
                "n_rows": int(len(rows)),
                "n_missing_image": int(sum(1 for r in rows if not os.path.isfile(os.path.join(args.image_root, f"{r.get('imageId')}.jpg")))),
            },
        }
        with open(os.path.join(out_dir, "dry_run.json"), "w", encoding="utf-8") as f:
            json.dump(dry, f, indent=2, ensure_ascii=False)
        print("[saved]", os.path.join(out_dir, "dry_run.json"))
        return

    if torch is None:
        raise RuntimeError("PyTorch is required for full run. Use --dry_run to validate I/O only.")

    # Lazy import so that --dry_run works without full LLaVA runtime deps.
    global IMAGE_TOKEN_INDEX
    global DEFAULT_IMAGE_TOKEN
    global DEFAULT_IM_START_TOKEN
    global DEFAULT_IM_END_TOKEN
    global conv_templates
    global load_pretrained_model
    global get_model_name_from_path
    global process_images
    global tokenizer_image_token
    global disable_torch_init
    global Image
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
    from PIL import Image

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_4bit=False,
        load_8bit=False,
        use_flash_attn=bool(args.use_flash_attn),
        device_map="auto",
    )
    model.eval()

    if str(args.attn_impl) != "auto":
        try:
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = str(args.attn_impl)
        except Exception:
            pass
        try:
            mm = model.get_model()
            if hasattr(mm.config, "attn_implementation"):
                mm.config.attn_implementation = str(args.attn_impl)
        except Exception:
            pass

    conv_mode = resolve_conv_mode(model_name, args.conv_mode)
    device = model.get_model().embed_tokens.weight.device

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None or int(pad_id) < 0:
        pad_id = eos_id if eos_id is not None else 0
    phrase_id_map: List[Tuple[str, List[int]]] = build_phrase_id_map(tokenizer, FORMAT_PHRASES)

    per_sample: List[Dict[str, Any]] = []
    per_candidate: List[Dict[str, Any]] = []
    scatter_rows: List[Dict[str, Any]] = []

    t0 = time.time()
    pbar = tqdm(rows, total=len(rows), desc="artrap-fragility", dynamic_ncols=True)
    for i, r in enumerate(pbar):
        qid = str(r.get("id", ""))
        question = str(r.get("question", ""))
        answer = str(r.get("answer", ""))
        image_id = str(r.get("imageId", ""))
        image_path = os.path.join(args.image_root, f"{image_id}.jpg")
        if not os.path.isfile(image_path):
            per_sample.append({"id": qid, "error": f"missing_image:{image_path}"})
            continue

        try:
            img_prompt = build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            q_prompt = build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=False,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )

            input_ids_img = tokenizer_image_token(img_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            input_ids_q = tokenizer(q_prompt, return_tensors="pt").input_ids.to(device)

            image = Image.open(image_path).convert("RGB")
            image_sizes = [image.size]
            images_tensor = process_images([image], image_processor, model.config).to(device=model.device, dtype=torch.float16)
        except Exception as e:
            per_sample.append({"id": qid, "error": f"build_input:{e}"})
            continue

        correct_diag = {
            "correct_variant_text": None,
            "correct_variant_token_ids_json": None,
            "correct_token_len": None,
            "correct_logp_img_mean": None,
            "correct_logp_img_min": None,
            "correct_prefix3_logp_img_mean": None,
            "correct_first_token_logp_img": None,
            "correct_first_token_rank_img": None,
            "correct_first_token_in_top6_img": None,
            "correct_first_token_in_top20_img": None,
        }
        if bool(args.diag_correct_answer):
            try:
                correct_diag = diagnose_correct_answer_img(
                    model=model,
                    tokenizer=tokenizer,
                    prefix_ids_img=input_ids_img,
                    images_tensor=images_tensor,
                    image_sizes=image_sizes,
                    answer_text=answer,
                    max_tokens=int(args.diag_correct_max_tokens),
                )
            except Exception:
                pass

        try:
            cand_rows = generate_candidates(
                model=model,
                tokenizer=tokenizer,
                input_ids_img=input_ids_img,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
                question=question,
                phrase_id_map=phrase_id_map,
                candidate_gen_mode=str(args.candidate_gen_mode),
                num_beams=int(args.num_beams),
                num_beam_groups=int(args.num_beam_groups),
                diversity_penalty=float(args.diversity_penalty),
                num_return_sequences=int(args.num_return_sequences),
                anchor_depth=int(args.anchor_depth),
                anchor_width=int(args.anchor_width),
                anchor_prefix_cap=int(args.anchor_prefix_cap),
                anchor_completion_budget=int(args.anchor_completion_budget),
                num_extra_samples=int(args.num_extra_samples),
                extra_sample_temperature=float(args.extra_sample_temperature),
                extra_sample_top_p=float(args.extra_sample_top_p),
                extra_sample_top_k=int(args.extra_sample_top_k),
                max_new_tokens=int(args.max_new_tokens),
                eos_token_id=eos_id,
            )
        except Exception as e:
            per_sample.append({"id": qid, "error": f"generate:{e}"})
            continue

        if len(cand_rows) == 0:
            per_sample.append({"id": qid, "error": "no_candidates"})
            continue

        cands: List[Candidate] = []
        core_ids_for_sq: List[List[int]] = []
        core_img_logps_for_sq: List[List[float]] = []
        core_idxs_for_sq: List[List[int]] = []
        for ci, gen_c in enumerate(cand_rows):
            cand_ids = [int(x) for x in gen_c.token_ids]
            token_logps_img = gen_c.token_logps_img
            if token_logps_img is None or len(token_logps_img) != len(cand_ids):
                token_logps_img = sequence_token_logps(
                    model=model,
                    prefix_ids=input_ids_img,
                    cont_ids=cand_ids,
                    images_tensor=images_tensor,
                    image_sizes=image_sizes,
                )
            if token_logps_img is None or len(token_logps_img) == 0:
                continue

            text = tokenizer.decode(cand_ids, skip_special_tokens=True).strip()
            short_answer = extract_core_answer_text(question=question, text=text, max_words=6)

            # Prefer exact core-span alignment in candidate token ids.
            core_start_aligned, core_match_ids = locate_core_span_from_text(
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
                # Fallback: phrase-based + function-word prefix based format estimation.
                format_len_phrase, _ = detect_format_len(tokenizer, cand_ids, phrase_id_map)
                format_len_prefix = infer_prefix_format_len_from_tokens(tokenizer, cand_ids, question)
                format_len = int(max(int(format_len_phrase), int(format_len_prefix)))
                core_start = int(min(format_len, max(0, len(cand_ids) - 1)))
                span_len = int(args.answer_span_max_tokens) if int(args.answer_span_max_tokens) > 0 else 1
                span_len = max(1, span_len)
                core_idxs = list(range(core_start, min(len(cand_ids), core_start + span_len)))
                core_start_idx = int(core_start)
                core_span_source = "fallback"

            if int(args.answer_span_max_tokens) > 0 and len(core_idxs) > int(args.answer_span_max_tokens):
                core_idxs = core_idxs[: int(args.answer_span_max_tokens)]
            if len(core_idxs) == 0 and len(cand_ids) > 0:
                core_idxs = [0]
            core_ids = [int(cand_ids[j]) for j in core_idxs if 0 <= int(j) < len(cand_ids)]
            if len(core_ids) == 0 and len(cand_ids) > 0:
                core_ids = [int(cand_ids[0])]
                core_idxs = [0]
            core_img_logps = [float(token_logps_img[j]) for j in core_idxs if 0 <= int(j) < len(token_logps_img)]

            s_format_img = (
                mean_selected(token_logps_img, list(range(min(int(format_len), len(token_logps_img)))))
                if int(format_len) > 0
                else None
            )
            s_core_img = mean_selected(token_logps_img, core_idxs)
            s_full = mean_selected(token_logps_img, list(range(len(token_logps_img))))

            cands.append(
                Candidate(
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
            per_sample.append({"id": qid, "error": "no_scored_candidates"})
            continue

        token_logps_q_rows = sequence_token_logps_batched(
            model=model,
            prefix_ids=input_ids_q,
            cont_ids_list=core_ids_for_sq,
            pad_token_id=pad_id,
            images_tensor=None,
            image_sizes=None,
        )
        token_margin_img_rows: List[Optional[List[float]]] = [None] * int(len(cands))
        if bool(args.save_top2_margin):
            token_margin_img_rows = sequence_token_top2_margin_batched(
                model=model,
                prefix_ids=input_ids_img,
                cont_ids_list=[[int(x) for x in c.token_ids] for c in cands],
                pad_token_id=pad_id,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
            )
        for ci, c in enumerate(cands):
            tlogps_q = token_logps_q_rows[ci] if ci < len(token_logps_q_rows) else None
            c.s_ans_q = mean_selected(tlogps_q, list(range(len(tlogps_q or []))))
            core_img_toks = core_img_logps_for_sq[ci] if ci < len(core_img_logps_for_sq) else []
            core_q_toks = (tlogps_q or [])
            c.s_core_img_min = (None if len(core_img_toks) == 0 else float(min(core_img_toks)))
            c.s_ans_q_min = (None if len(core_q_toks) == 0 else float(min(core_q_toks)))
            m = int(min(len(core_img_toks), len(core_q_toks)))
            if m > 0:
                c.core_img_toks = [float(x) for x in core_img_toks[:m]]
                c.core_q_toks = [float(x) for x in core_q_toks[:m]]
                vpmi_toks = [float(core_img_toks[k]) - float(core_q_toks[k]) for k in range(m)]
                c.core_vpmi_toks = [float(x) for x in vpmi_toks]
                c.vpmi_core_mean = float(sum(vpmi_toks) / len(vpmi_toks))
                vpmi_min_raw = float(min(vpmi_toks))
                c.vpmi_core_min_raw = vpmi_min_raw

                # Method A: prior-masked min
                keep_idxs = [
                    int(k)
                    for k, sq in enumerate(core_q_toks[:m])
                    if float(sq) <= float(args.vpmi_min_sq_logp_max)
                ]
                vpmi_min_prior_masked = (
                    None if len(keep_idxs) == 0 else float(min(vpmi_toks[k] for k in keep_idxs))
                )
                c.vpmi_core_min_prior_masked = vpmi_min_prior_masked

                # Method B: word-level min to reduce subword tail artifacts
                core_ids_aligned = [int(x) for x in core_ids_for_sq[ci][:m]]
                word_groups = group_token_positions_by_word(tokenizer, core_ids_aligned)
                word_scores = [
                    float(sum(vpmi_toks[j] for j in g) / len(g))
                    for g in word_groups
                    if len(g) > 0
                ]
                vpmi_word_min = (None if len(word_scores) == 0 else float(min(word_scores)))
                c.vpmi_word_min = vpmi_word_min

                # Extra token-level diagnostics:
                # - tail min over last 50% core tokens
                # - position of minimum token VPMI (normalized)
                # - mean-minus-min gap (instability)
                # - non-zero sign-flip count along core VPMI sequence
                tail_start = int(math.floor(0.5 * m))
                tail_vals = vpmi_toks[tail_start:] if tail_start < m else [vpmi_toks[-1]]
                c.vpmi_core_tail_min = (None if len(tail_vals) == 0 else float(min(tail_vals)))

                min_k = int(min(range(m), key=lambda k: float(vpmi_toks[k])))
                c.vpmi_core_min_pos_norm = float(min_k / max(1, m - 1))

                vpmi_mean = float(sum(vpmi_toks) / len(vpmi_toks))
                c.vpmi_core_min_mean_gap = float(vpmi_mean - vpmi_min_raw)

                prev_sign = 0
                flips = 0
                for vv in vpmi_toks:
                    sgn = (1 if float(vv) > 1e-8 else (-1 if float(vv) < -1e-8 else 0))
                    if sgn == 0:
                        continue
                    if prev_sign != 0 and sgn != prev_sign:
                        flips += 1
                    prev_sign = sgn
                c.vpmi_core_sign_flip_count = float(flips)

                # Exported vpmi_core_min follows configured mode
                mode = str(args.vpmi_min_mode)
                selected: Optional[float] = None
                if mode == "raw":
                    selected = vpmi_min_raw
                elif mode == "prior_masked":
                    selected = vpmi_min_prior_masked
                elif mode == "word_min":
                    selected = vpmi_word_min

                if selected is None and str(args.vpmi_min_fallback) == "raw":
                    selected = vpmi_min_raw
                c.vpmi_core_min = selected

            # True top1-top2 margin on image-conditioned logits for core span.
            mimg = token_margin_img_rows[ci] if ci < len(token_margin_img_rows) else None
            if mimg is not None:
                cidxs = core_idxs_for_sq[ci] if ci < len(core_idxs_for_sq) else []
                margins = [float(mimg[k]) for k in cidxs if 0 <= int(k) < len(mimg)]
                if len(margins) > 0:
                    c.margin_core_img_min = float(min(margins))
                    c.margin_core_img_mean = float(sum(margins) / len(margins))

        cands = [c for c in cands if c.s_full is not None]
        if len(cands) == 0:
            per_sample.append({"id": qid, "error": "no_full_score"})
            continue

        cands_sorted = sorted(cands, key=lambda x: float(x.s_full), reverse=True)
        champ = cands_sorted[0]
        rank_sfull_map = {int(c.cand_idx): int(i + 1) for i, c in enumerate(cands_sorted)}

        correct_eval_by_idx: Dict[int, bool] = {}
        correct_strict_by_idx: Dict[int, bool] = {}
        correct_heur_by_idx: Dict[int, bool] = {}
        for c in cands:
            pred_c = first_clause(c.short_answer if c.short_answer else c.text)
            c_ok_strict = bool(norm_text(pred_c) == norm_text(answer))
            c_ok_heur = bool(
                is_success_heuristic(
                    question=question,
                    answer=answer,
                    champ_text=c.text,
                    champ_short=c.short_answer,
                )
            )
            c_ok_eval = bool(c_ok_heur if str(args.eval_match_mode) == "heuristic" else c_ok_strict)
            correct_strict_by_idx[int(c.cand_idx)] = bool(c_ok_strict)
            correct_heur_by_idx[int(c.cand_idx)] = bool(c_ok_heur)
            correct_eval_by_idx[int(c.cand_idx)] = bool(c_ok_eval)

        correct_idxs = [int(c.cand_idx) for c in cands if bool(correct_eval_by_idx.get(int(c.cand_idx), False))]
        correct_candidate_count = int(len(correct_idxs))
        correct_best_rank_sfull = (
            None
            if len(correct_idxs) == 0
            else int(min(int(rank_sfull_map.get(ii, 10**9)) for ii in correct_idxs))
        )

        prior_vals = [float(c.s_ans_q) for c in cands if c.s_ans_q is not None]
        p_q_map: Dict[int, Optional[float]] = {}
        for c in cands:
            if c.s_ans_q is None:
                p_q_map[c.cand_idx] = None
            else:
                p_q_map[c.cand_idx] = percentile_rank(prior_vals, float(c.s_ans_q))

        champ_p_q = p_q_map.get(champ.cand_idx)

        safe_pool = [
            c for c in cands
            if c.cand_idx != champ.cand_idx
            and p_q_map.get(c.cand_idx) is not None
            and float(p_q_map[c.cand_idx]) <= float(args.beta_q)
            and c.s_full is not None
        ]
        safe_challenger = None if len(safe_pool) == 0 else max(safe_pool, key=lambda x: float(x.s_full))
        delta_safe = None
        if safe_challenger is not None and champ.s_full is not None and safe_challenger.s_full is not None:
            delta_safe = float(float(champ.s_full) - float(safe_challenger.s_full))

        pred_answer = first_clause(champ.short_answer if champ.short_answer else champ.text)
        is_success_strict = bool(norm_text(pred_answer) == norm_text(answer))
        is_success_heur = is_success_heuristic(
            question=question,
            answer=answer,
            champ_text=champ.text,
            champ_short=champ.short_answer,
        )
        is_success = bool(is_success_heur if str(args.eval_match_mode) == "heuristic" else is_success_strict)
        label = "success" if is_success else "failure"

        # AR-trap indicators on champion
        illusion_gap = None
        if champ.s_format_img is not None and champ.s_core_img is not None:
            illusion_gap = float(champ.s_format_img - champ.s_core_img)

        per_sample.append({
            "id": qid,
            "image_id": image_id,
            "question": question,
            "answer": answer,
            "label": label,
            "is_success": bool(is_success),
            "is_success_strict": bool(is_success_strict),
            "is_success_heuristic": bool(is_success_heur),
            "eval_match_mode": str(args.eval_match_mode),
            "pred_answer_eval": pred_answer,
            "n_candidates": int(len(cands)),
            "correct_candidate_count_eval": int(correct_candidate_count),
            "correct_best_rank_sfull_eval": correct_best_rank_sfull,
            "correct_in_top6_eval": (
                None if correct_best_rank_sfull is None else bool(int(correct_best_rank_sfull) <= 6)
            ),
            "correct_in_top20_eval": (
                None if correct_best_rank_sfull is None else bool(int(correct_best_rank_sfull) <= 20)
            ),
            "champ_idx": int(champ.cand_idx),
            "champ_text": champ.text,
            "champ_short_answer": champ.short_answer,
            "champ_s_full": champ.s_full,
            "champ_s_format_img": champ.s_format_img,
            "champ_s_core_img": champ.s_core_img,
            "champ_s_ans_q": champ.s_ans_q,
            "champ_s_core_img_min": champ.s_core_img_min,
            "champ_s_ans_q_min": champ.s_ans_q_min,
            "champ_vpmi_core_mean": champ.vpmi_core_mean,
            "champ_vpmi_core_min": champ.vpmi_core_min,
            "champ_vpmi_core_min_raw": champ.vpmi_core_min_raw,
            "champ_vpmi_core_min_prior_masked": champ.vpmi_core_min_prior_masked,
            "champ_vpmi_word_min": champ.vpmi_word_min,
            "champ_vpmi_core_tail_min": champ.vpmi_core_tail_min,
            "champ_vpmi_core_min_pos_norm": champ.vpmi_core_min_pos_norm,
            "champ_vpmi_core_min_mean_gap": champ.vpmi_core_min_mean_gap,
            "champ_vpmi_core_sign_flip_count": champ.vpmi_core_sign_flip_count,
            "champ_margin_core_img_min": champ.margin_core_img_min,
            "champ_margin_core_img_mean": champ.margin_core_img_mean,
            "champ_p_q": champ_p_q,
            "champ_format_len": int(champ.format_len),
            "champ_core_len": int(champ.core_len),
            "champ_core_start": int(champ.core_start),
            "champ_core_span_source": str(champ.core_span_source),
            "illusion_gap_format_minus_core": illusion_gap,
            "safe_exists": bool(safe_challenger is not None),
            "safe_idx": (None if safe_challenger is None else int(safe_challenger.cand_idx)),
            "safe_text": (None if safe_challenger is None else safe_challenger.text),
            "safe_s_full": (None if safe_challenger is None else safe_challenger.s_full),
            "safe_p_q": (None if safe_challenger is None else p_q_map.get(safe_challenger.cand_idx)),
            "safe_s_core_img_min": (None if safe_challenger is None else safe_challenger.s_core_img_min),
            "safe_s_ans_q_min": (None if safe_challenger is None else safe_challenger.s_ans_q_min),
            "safe_vpmi_core_mean": (None if safe_challenger is None else safe_challenger.vpmi_core_mean),
            "safe_vpmi_core_min": (None if safe_challenger is None else safe_challenger.vpmi_core_min),
            "safe_vpmi_core_min_raw": (None if safe_challenger is None else safe_challenger.vpmi_core_min_raw),
            "safe_vpmi_core_min_prior_masked": (None if safe_challenger is None else safe_challenger.vpmi_core_min_prior_masked),
            "safe_vpmi_word_min": (None if safe_challenger is None else safe_challenger.vpmi_word_min),
            "safe_vpmi_core_tail_min": (None if safe_challenger is None else safe_challenger.vpmi_core_tail_min),
            "safe_vpmi_core_min_pos_norm": (None if safe_challenger is None else safe_challenger.vpmi_core_min_pos_norm),
            "safe_vpmi_core_min_mean_gap": (None if safe_challenger is None else safe_challenger.vpmi_core_min_mean_gap),
            "safe_vpmi_core_sign_flip_count": (None if safe_challenger is None else safe_challenger.vpmi_core_sign_flip_count),
            "safe_margin_core_img_min": (None if safe_challenger is None else safe_challenger.margin_core_img_min),
            "safe_margin_core_img_mean": (None if safe_challenger is None else safe_challenger.margin_core_img_mean),
            "correct_variant_text": correct_diag.get("correct_variant_text"),
            "correct_variant_token_ids_json": correct_diag.get("correct_variant_token_ids_json"),
            "correct_token_len": correct_diag.get("correct_token_len"),
            "correct_logp_img_mean": correct_diag.get("correct_logp_img_mean"),
            "correct_logp_img_min": correct_diag.get("correct_logp_img_min"),
            "correct_prefix3_logp_img_mean": correct_diag.get("correct_prefix3_logp_img_mean"),
            "correct_first_token_logp_img": correct_diag.get("correct_first_token_logp_img"),
            "correct_first_token_rank_img": correct_diag.get("correct_first_token_rank_img"),
            "correct_first_token_in_top6_img": correct_diag.get("correct_first_token_in_top6_img"),
            "correct_first_token_in_top20_img": correct_diag.get("correct_first_token_in_top20_img"),
            "delta_safe": delta_safe,
            "is_fragile": (None if delta_safe is None else bool(float(delta_safe) <= float(args.tau_gap))),
        })

        scatter_rows.append({
            "id": qid,
            "label": label,
            "champ_p_q": champ_p_q,
            "delta_safe": delta_safe,
            "safe_exists": bool(safe_challenger is not None),
            "is_fragile": (None if delta_safe is None else bool(float(delta_safe) <= float(args.tau_gap))),
            "is_high_prior": (None if champ_p_q is None else bool(float(champ_p_q) >= 0.8)),
            "champ_s_full": champ.s_full,
            "champ_s_core_img": champ.s_core_img,
            "champ_s_format_img": champ.s_format_img,
        })

        for c in cands:
            per_candidate.append({
                "id": qid,
                "label": label,
                "cand_idx": int(c.cand_idx),
                "is_champion": bool(c.cand_idx == champ.cand_idx),
                "text": c.text,
                "short_answer": c.short_answer,
                "s_full": c.s_full,
                "s_format_img": c.s_format_img,
                "s_core_img": c.s_core_img,
                "s_ans_q": c.s_ans_q,
                "s_core_img_min": c.s_core_img_min,
                "s_ans_q_min": c.s_ans_q_min,
                "vpmi_core_mean": c.vpmi_core_mean,
                "vpmi_core_min": c.vpmi_core_min,
                "vpmi_core_min_raw": c.vpmi_core_min_raw,
                "vpmi_core_min_prior_masked": c.vpmi_core_min_prior_masked,
                "vpmi_word_min": c.vpmi_word_min,
                "vpmi_core_tail_min": c.vpmi_core_tail_min,
                "vpmi_core_min_pos_norm": c.vpmi_core_min_pos_norm,
                "vpmi_core_min_mean_gap": c.vpmi_core_min_mean_gap,
                "vpmi_core_sign_flip_count": c.vpmi_core_sign_flip_count,
                "margin_core_img_min": c.margin_core_img_min,
                "margin_core_img_mean": c.margin_core_img_mean,
                "is_correct_eval": bool(correct_eval_by_idx.get(int(c.cand_idx), False)),
                "is_correct_heuristic": bool(correct_heur_by_idx.get(int(c.cand_idx), False)),
                "is_correct_strict": bool(correct_strict_by_idx.get(int(c.cand_idx), False)),
                "p_q": p_q_map.get(c.cand_idx),
                "format_len": int(c.format_len),
                "core_len": int(c.core_len),
                "core_start": int(c.core_start),
                "core_span_source": str(c.core_span_source),
                "token_ids_json": (
                    json.dumps([int(x) for x in c.token_ids], ensure_ascii=False)
                    if bool(args.save_token_ids_json)
                    else None
                ),
                "core_img_toks_json": (
                    json.dumps([float(x) for x in (c.core_img_toks or [])], ensure_ascii=False)
                    if bool(args.save_core_tokenwise_vpmi)
                    else None
                ),
                "core_q_toks_json": (
                    json.dumps([float(x) for x in (c.core_q_toks or [])], ensure_ascii=False)
                    if bool(args.save_core_tokenwise_vpmi)
                    else None
                ),
                "core_vpmi_toks_json": (
                    json.dumps([float(x) for x in (c.core_vpmi_toks or [])], ensure_ascii=False)
                    if bool(args.save_core_tokenwise_vpmi)
                    else None
                ),
                "is_safe": bool(safe_challenger is not None and c.cand_idx == safe_challenger.cand_idx),
            })

        elapsed = time.time() - t0
        eta = elapsed / max(1, i + 1) * max(0, len(rows) - i - 1)
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(elapsed=f"{elapsed:.1f}s", eta=f"{eta:.1f}s")

    valid = [r for r in per_sample if r.get("error") is None]
    success = [r for r in valid if bool(r.get("is_success", False))]
    failure = [r for r in valid if not bool(r.get("is_success", False))]

    high_prior = [r for r in valid if safe_float(r.get("champ_p_q")) is not None and float(r["champ_p_q"]) >= 0.8]
    fragile = [r for r in valid if safe_float(r.get("delta_safe")) is not None and float(r["delta_safe"]) <= float(args.tau_gap)]

    # 2D binning for p_q vs delta_safe
    pq_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
    d_edges = [0.0, 0.25, 0.5, 1.0, 2.0, 1e9]
    bin_rows: List[Dict[str, Any]] = []
    for pi in range(len(pq_edges) - 1):
        for di in range(len(d_edges) - 1):
            cell = []
            for r in valid:
                p = safe_float(r.get("champ_p_q"))
                d = safe_float(r.get("delta_safe"))
                if p is None or d is None:
                    continue
                if bin_index(p, pq_edges) == pi and bin_index(d, d_edges) == di:
                    cell.append(r)
            if len(cell) == 0:
                continue
            n_fail = int(sum(1 for r in cell if not bool(r.get("is_success", False))))
            n_succ = int(len(cell) - n_fail)
            bin_rows.append({
                "pq_bin": f"[{pq_edges[pi]:.2f},{pq_edges[pi+1]:.2f})",
                "delta_bin": f"[{d_edges[di]:.2f},{d_edges[di+1]:.2f})",
                "n": int(len(cell)),
                "n_failure": int(n_fail),
                "n_success": int(n_succ),
                "failure_rate": float(n_fail / len(cell)),
            })

    summary = {
        "inputs": {
            "questions_json": qpath,
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "num_beams": int(args.num_beams),
            "num_beam_groups": int(args.num_beam_groups),
            "diversity_penalty": float(args.diversity_penalty),
            "num_return_sequences": int(args.num_return_sequences),
            "candidate_gen_mode": str(args.candidate_gen_mode),
            "anchor_depth": int(args.anchor_depth),
            "anchor_width": int(args.anchor_width),
            "anchor_prefix_cap": int(args.anchor_prefix_cap),
            "anchor_completion_budget": int(args.anchor_completion_budget),
            "num_extra_samples": int(args.num_extra_samples),
            "extra_sample_temperature": float(args.extra_sample_temperature),
            "extra_sample_top_p": float(args.extra_sample_top_p),
            "extra_sample_top_k": int(args.extra_sample_top_k),
            "max_new_tokens": int(args.max_new_tokens),
            "answer_span_max_tokens": int(args.answer_span_max_tokens),
            "vpmi_min_mode": str(args.vpmi_min_mode),
            "vpmi_min_sq_logp_max": float(args.vpmi_min_sq_logp_max),
            "vpmi_min_fallback": str(args.vpmi_min_fallback),
            "save_top2_margin": bool(args.save_top2_margin),
            "save_token_ids_json": bool(args.save_token_ids_json),
            "save_core_tokenwise_vpmi": bool(args.save_core_tokenwise_vpmi),
            "diag_correct_answer": bool(args.diag_correct_answer),
            "diag_correct_max_tokens": int(args.diag_correct_max_tokens),
            "beta_q": float(args.beta_q),
            "tau_gap": float(args.tau_gap),
            "eval_match_mode": str(args.eval_match_mode),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "attn_impl": str(args.attn_impl),
            "use_flash_attn": bool(args.use_flash_attn),
        },
        "counts": {
            "n_total": int(len(rows)),
            "n_valid": int(len(valid)),
            "n_success": int(len(success)),
            "n_failure": int(len(failure)),
            "accuracy": mean_or_none([1.0 if bool(r.get("is_success", False)) else 0.0 for r in valid]),
            "accuracy_strict": mean_or_none([1.0 if bool(r.get("is_success_strict", False)) else 0.0 for r in valid]),
            "accuracy_heuristic": mean_or_none([1.0 if bool(r.get("is_success_heuristic", False)) else 0.0 for r in valid]),
        },
        "proof1_artrap": {
            "champ_s_full_success": stats_of([safe_float(r.get("champ_s_full")) for r in success]),
            "champ_s_full_failure": stats_of([safe_float(r.get("champ_s_full")) for r in failure]),
            "champ_s_format_img_success": stats_of([safe_float(r.get("champ_s_format_img")) for r in success]),
            "champ_s_format_img_failure": stats_of([safe_float(r.get("champ_s_format_img")) for r in failure]),
            "champ_s_core_img_success": stats_of([safe_float(r.get("champ_s_core_img")) for r in success]),
            "champ_s_core_img_failure": stats_of([safe_float(r.get("champ_s_core_img")) for r in failure]),
            "champ_illusion_gap_success": stats_of([safe_float(r.get("illusion_gap_format_minus_core")) for r in success]),
            "champ_illusion_gap_failure": stats_of([safe_float(r.get("illusion_gap_format_minus_core")) for r in failure]),
        },
        "proof2_pairwise_fragility": {
            "champ_p_q_success": stats_of([safe_float(r.get("champ_p_q")) for r in success]),
            "champ_p_q_failure": stats_of([safe_float(r.get("champ_p_q")) for r in failure]),
            "delta_safe_success": stats_of([safe_float(r.get("delta_safe")) for r in success]),
            "delta_safe_failure": stats_of([safe_float(r.get("delta_safe")) for r in failure]),
            "high_prior_region": {
                "threshold": 0.8,
                "n": int(len(high_prior)),
                "n_failure": int(sum(1 for r in high_prior if not bool(r.get("is_success", False)))),
                "failure_rate": mean_or_none([0.0 if bool(r.get("is_success", False)) else 1.0 for r in high_prior]),
            },
            "fragile_region": {
                "tau_gap": float(args.tau_gap),
                "n": int(len(fragile)),
                "n_failure": int(sum(1 for r in fragile if not bool(r.get("is_success", False)))),
                "failure_rate": mean_or_none([0.0 if bool(r.get("is_success", False)) else 1.0 for r in fragile]),
            },
        },
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample.csv"),
            "per_candidate_csv": os.path.join(out_dir, "per_candidate.csv"),
            "scatter_csv": os.path.join(out_dir, "scatter_pq_vs_delta_safe.csv"),
            "binning_csv": os.path.join(out_dir, "binning_pq_delta.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_sample.csv"), per_sample)
    write_csv(os.path.join(out_dir, "per_candidate.csv"), per_candidate)
    write_csv(os.path.join(out_dir, "scatter_pq_vs_delta_safe.csv"), scatter_rows)
    write_csv(os.path.join(out_dir, "binning_pq_delta.csv"), bin_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample.csv"))
    print("[saved]", os.path.join(out_dir, "per_candidate.csv"))
    print("[saved]", os.path.join(out_dir, "scatter_pq_vs_delta_safe.csv"))
    print("[saved]", os.path.join(out_dir, "binning_pq_delta.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
