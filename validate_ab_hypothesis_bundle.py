#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable

import analyze_artrap_pairwise_fragility as pf


TRUE_SET = {"1", "true", "t", "yes", "y"}


def as_bool(x: Any) -> bool:
    return str("" if x is None else x).strip().lower() in TRUE_SET


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


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


def parse_json_float_list(x: Any) -> List[float]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: List[float] = []
    for v in obj:
        vv = safe_float(v)
        if vv is not None:
            out.append(float(vv))
    return out


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if len(xs) == 0:
        return None
    qq = max(0.0, min(1.0, float(q)))
    if len(xs) == 1:
        return float(xs[0])
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def token_is_prefix(tok: str) -> bool:
    t = str(tok)
    if t.startswith("<0x") and t.endswith(">"):
        return False
    return t.startswith("\u2581") or t.startswith("Ġ")


@dataclass
class Cand:
    idx: int
    is_champion: bool
    is_correct_eval: bool
    vpmi: Optional[float]
    margin_min: Optional[float]
    core_vpmi_toks: List[float]
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    score_ab: Optional[float] = None
    rank_a: Optional[float] = None
    rank_b: Optional[float] = None
    prefix_max_k: Optional[float] = None
    prefix_mean_k: Optional[float] = None
    suffix_min_k: Optional[float] = None


def rank_desc_pct(vals_by_idx: Dict[int, float]) -> Dict[int, float]:
    pairs = sorted(vals_by_idx.items(), key=lambda x: float(x[1]), reverse=True)
    n = len(pairs)
    if n == 0:
        return {}
    if n == 1:
        return {int(pairs[0][0]): 1.0}
    out: Dict[int, float] = {}
    for r, (idx, _) in enumerate(pairs):
        out[int(idx)] = float((n - 1 - r) / (n - 1))
    return out


def load_pool(beam_dir: str, k_prefix: int, k_suffix: int, b_prefix_penalty: float) -> Dict[str, Any]:
    ps = read_csv(os.path.join(beam_dir, "per_sample.csv"))
    pc = read_csv(os.path.join(beam_dir, "per_candidate.csv"))
    sample_map: Dict[str, Dict[str, Any]] = {}
    for r in ps:
        sid = str(r.get("id", ""))
        if sid == "":
            continue
        if str(r.get("error", "")).strip() != "":
            continue
        sample_map[sid] = {
            "id": sid,
            "question": str(r.get("question", "")),
            "answer": str(r.get("answer", "")),
            "is_success": as_bool(r.get("is_success", "")),
            "champ_idx_ps": safe_float(r.get("champ_idx")),
            "cand_list": [],
        }

    for r in pc:
        sid = str(r.get("id", ""))
        s = sample_map.get(sid)
        if s is None:
            continue
        idx = safe_float(r.get("cand_idx"))
        if idx is None:
            continue
        toks = parse_json_float_list(r.get("core_vpmi_toks_json"))
        kp = int(max(1, k_prefix))
        ks = int(max(1, k_suffix))
        pref = toks[: min(kp, len(toks))]
        suf = toks[max(0, len(toks) - ks):]
        prefix_max = None if len(pref) == 0 else float(max(pref))
        prefix_mean = None if len(pref) == 0 else float(sum(pref) / len(pref))
        suffix_min = None if len(suf) == 0 else float(min(suf))
        c = Cand(
            idx=int(idx),
            is_champion=as_bool(r.get("is_champion", "")),
            is_correct_eval=as_bool(r.get("is_correct_eval", "")),
            vpmi=safe_float(r.get("vpmi_core_mean")),
            margin_min=safe_float(r.get("margin_core_img_min")),
            core_vpmi_toks=toks,
            prefix_max_k=prefix_max,
            prefix_mean_k=prefix_mean,
            suffix_min_k=suffix_min,
        )
        s["cand_list"].append(c)

    # build A/B/AB scores
    for sid, s in list(sample_map.items()):
        cands: List[Cand] = s["cand_list"]
        if len(cands) == 0:
            del sample_map[sid]
            continue
        vp = {c.idx: float(c.vpmi) for c in cands if c.vpmi is not None}
        rk_a = rank_desc_pct(vp)
        b_raw: Dict[int, float] = {}
        for c in cands:
            if c.suffix_min_k is None:
                continue
            pmax = float(c.prefix_max_k if c.prefix_max_k is not None else 0.0)
            b_raw[c.idx] = float(c.suffix_min_k - float(b_prefix_penalty) * max(0.0, pmax))
        rk_b = rank_desc_pct(b_raw)

        for c in cands:
            c.rank_a = rk_a.get(c.idx)
            c.rank_b = rk_b.get(c.idx)
            c.score_a = c.rank_a
            c.score_b = b_raw.get(c.idx)
            if c.rank_a is not None and c.rank_b is not None:
                c.score_ab = float(c.rank_a + c.rank_b)
            elif c.rank_a is not None:
                c.score_ab = float(c.rank_a)
            elif c.rank_b is not None:
                c.score_ab = float(c.rank_b)
            else:
                c.score_ab = None
    return sample_map


def choose_best(cands: List[Cand], mode: str) -> Cand:
    key_name = {
        "A": "score_a",
        "B": "score_b",
        "AB": "score_ab",
    }[str(mode)]
    finite = [c for c in cands if getattr(c, key_name) is not None]
    if len(finite) == 0:
        # fallback to champion
        champ = next((c for c in cands if c.is_champion), None)
        return cands[0] if champ is None else champ
    return max(finite, key=lambda x: float(getattr(x, key_name)))


def eval_switch(sample_map: Dict[str, Any], mode: str, constrained: bool, params: Dict[str, float]) -> Dict[str, Any]:
    n = 0
    base_correct = 0
    new_correct = 0
    switch = 0
    gain = 0
    harm = 0
    rows: List[Dict[str, Any]] = []
    for sid, s in sample_map.items():
        cands: List[Cand] = s["cand_list"]
        champ = next((c for c in cands if c.is_champion), None)
        if champ is None:
            continue
        chosen = choose_best(cands, mode=mode)
        do_switch = bool(chosen.idx != champ.idx)
        reason = "unconstrained"
        if constrained:
            # shared low-margin gate
            cm = float(champ.margin_min if champ.margin_min is not None else 1e9)
            cond_margin = bool(cm <= float(params.get("tau_margin", 1e9)))
            if mode == "A":
                rg = (
                    float((chosen.rank_a or 0.0) - (champ.rank_a or 0.0))
                    if chosen.rank_a is not None and champ.rank_a is not None
                    else -1e9
                )
                cond_rank = bool(rg >= float(params.get("tau_rank_gap", -1e9)))
                do_switch = bool(do_switch and cond_margin and cond_rank)
                reason = f"margin={cond_margin},rankgap={cond_rank}"
            elif mode == "B":
                sg = (
                    float((chosen.suffix_min_k or -1e9) - (champ.suffix_min_k or -1e9))
                    if chosen.suffix_min_k is not None and champ.suffix_min_k is not None
                    else -1e9
                )
                csc = float(champ.suffix_min_k if champ.suffix_min_k is not None else 1e9)
                cond_sg = bool(sg >= float(params.get("tau_suffix_gap", -1e9)))
                cond_collapse = bool(csc <= float(params.get("tau_champ_suffix", 1e9)))
                do_switch = bool(do_switch and cond_margin and cond_sg and cond_collapse)
                reason = f"margin={cond_margin},suffixgap={cond_sg},collapse={cond_collapse}"
            else:
                rg = (
                    float((chosen.rank_a or 0.0) - (champ.rank_a or 0.0))
                    if chosen.rank_a is not None and champ.rank_a is not None
                    else -1e9
                )
                sg = (
                    float((chosen.suffix_min_k or -1e9) - (champ.suffix_min_k or -1e9))
                    if chosen.suffix_min_k is not None and champ.suffix_min_k is not None
                    else -1e9
                )
                csc = float(champ.suffix_min_k if champ.suffix_min_k is not None else 1e9)
                cond_rank = bool(rg >= float(params.get("tau_rank_gap", -1e9)))
                cond_sg = bool(sg >= float(params.get("tau_suffix_gap", -1e9)))
                cond_collapse = bool(csc <= float(params.get("tau_champ_suffix", 1e9)))
                do_switch = bool(do_switch and cond_margin and cond_rank and cond_sg and cond_collapse)
                reason = f"margin={cond_margin},rankgap={cond_rank},suffixgap={cond_sg},collapse={cond_collapse}"

        final = chosen if do_switch else champ
        base_ok = bool(champ.is_correct_eval)
        new_ok = bool(final.is_correct_eval)
        n += 1
        base_correct += int(base_ok)
        new_correct += int(new_ok)
        if do_switch:
            switch += 1
        if (not base_ok) and new_ok:
            gain += 1
        if base_ok and (not new_ok):
            harm += 1
        rows.append(
            {
                "id": sid,
                "mode": mode,
                "constrained": bool(constrained),
                "champ_idx": int(champ.idx),
                "chosen_idx": int(chosen.idx),
                "final_idx": int(final.idx),
                "do_switch": bool(do_switch),
                "base_ok": bool(base_ok),
                "new_ok": bool(new_ok),
                "reason": reason,
            }
        )
    acc_base = None if n == 0 else float(base_correct / n)
    acc_new = None if n == 0 else float(new_correct / n)
    return {
        "n": int(n),
        "mode": str(mode),
        "constrained": bool(constrained),
        "accuracy_base": acc_base,
        "accuracy_new": acc_new,
        "delta_acc": (None if acc_base is None or acc_new is None else float(acc_new - acc_base)),
        "switch": int(switch),
        "switch_rate": (None if n == 0 else float(switch / n)),
        "gain": int(gain),
        "harm": int(harm),
        "net": int(gain - harm),
        "precision_gain": (None if switch == 0 else float(gain / switch)),
        "rows": rows,
    }


def search_constrained(sample_map: Dict[str, Any], mode: str) -> Dict[str, Any]:
    # collect feature pools
    champ_margin = []
    rank_gap = []
    suffix_gap = []
    champ_suffix = []
    for s in sample_map.values():
        cands = s["cand_list"]
        champ = next((c for c in cands if c.is_champion), None)
        if champ is None:
            continue
        cm = safe_float(champ.margin_min)
        if cm is not None:
            champ_margin.append(float(cm))
        best = choose_best(cands, mode=mode)
        rg = None
        if best.rank_a is not None and champ.rank_a is not None:
            rg = float(best.rank_a - champ.rank_a)
            rank_gap.append(float(rg))
        if best.suffix_min_k is not None and champ.suffix_min_k is not None:
            sg = float(best.suffix_min_k - champ.suffix_min_k)
            suffix_gap.append(sg)
            champ_suffix.append(float(champ.suffix_min_k))
    # candidate grids
    q_grid = [0.2, 0.35, 0.5, 0.65, 0.8]
    margin_grid = [quantile(champ_margin, q) for q in q_grid]
    margin_grid = [x for x in margin_grid if x is not None]
    rank_grid = [quantile(rank_gap, q) for q in [0.4, 0.5, 0.6, 0.7, 0.8]]
    rank_grid = [x for x in rank_grid if x is not None]
    suffix_grid = [quantile(suffix_gap, q) for q in [0.4, 0.5, 0.6, 0.7, 0.8]]
    suffix_grid = [x for x in suffix_grid if x is not None]
    champ_suf_grid = [quantile(champ_suffix, q) for q in [0.2, 0.35, 0.5, 0.65]]
    champ_suf_grid = [x for x in champ_suf_grid if x is not None]

    if len(margin_grid) == 0:
        margin_grid = [1e9]
    if len(rank_grid) == 0:
        rank_grid = [-1e9]
    if len(suffix_grid) == 0:
        suffix_grid = [-1e9]
    if len(champ_suf_grid) == 0:
        champ_suf_grid = [1e9]

    best_res = None
    tried = 0
    if mode == "A":
        for tm in margin_grid:
            for tr in rank_grid:
                tried += 1
                cur = eval_switch(
                    sample_map, mode=mode, constrained=True,
                    params={"tau_margin": float(tm), "tau_rank_gap": float(tr)},
                )
                key = (int(cur["net"]), float(cur["delta_acc"] or -1e9), int(cur["gain"]), -int(cur["harm"]))
                if best_res is None or key > best_res[0]:
                    best_res = (key, cur, {"tau_margin": float(tm), "tau_rank_gap": float(tr)})
    elif mode == "B":
        for tm in margin_grid:
            for ts in suffix_grid:
                for tcs in champ_suf_grid:
                    tried += 1
                    cur = eval_switch(
                        sample_map, mode=mode, constrained=True,
                        params={
                            "tau_margin": float(tm),
                            "tau_suffix_gap": float(ts),
                            "tau_champ_suffix": float(tcs),
                        },
                    )
                    key = (int(cur["net"]), float(cur["delta_acc"] or -1e9), int(cur["gain"]), -int(cur["harm"]))
                    if best_res is None or key > best_res[0]:
                        best_res = (key, cur, {"tau_margin": float(tm), "tau_suffix_gap": float(ts), "tau_champ_suffix": float(tcs)})
    else:
        for tm in margin_grid:
            for tr in rank_grid:
                for ts in suffix_grid:
                    for tcs in champ_suf_grid:
                        tried += 1
                        cur = eval_switch(
                            sample_map, mode=mode, constrained=True,
                            params={
                                "tau_margin": float(tm),
                                "tau_rank_gap": float(tr),
                                "tau_suffix_gap": float(ts),
                                "tau_champ_suffix": float(tcs),
                            },
                        )
                        key = (int(cur["net"]), float(cur["delta_acc"] or -1e9), int(cur["gain"]), -int(cur["harm"]))
                        if best_res is None or key > best_res[0]:
                            best_res = (key, cur, {
                                "tau_margin": float(tm),
                                "tau_rank_gap": float(tr),
                                "tau_suffix_gap": float(ts),
                                "tau_champ_suffix": float(tcs),
                            })
    assert best_res is not None
    out = dict(best_res[1])
    out["best_params"] = best_res[2]
    out["n_grid"] = int(tried)
    return out


@torch.no_grad()
def replay_token_counterfactual(
    *,
    sample_rows: List[Dict[str, Any]],
    image_root: str,
    model_path: str,
    model_base: Optional[str],
    conv_mode_override: Optional[str],
    max_steps: int,
    top_k_rank: int,
    alpha_rank: float,
    state_top_k: int,
    prefix_spike_z: float,
    beta_prefix: float,
    suffix_vpmi_floor: float,
    beta_suffix: float,
    # constrained conditions
    tau_low_margin: float,
    tau_rank_gap: float,
    b_core_max_steps: int,
    tau_suffix_collapse: float,
    use_flash_attn: bool,
    attn_impl: str,
) -> Dict[str, Any]:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for replay stage.")

    # lazy imports
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

    pf.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
    pf.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
    pf.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
    pf.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
    pf.conv_templates = conv_templates

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

    stats = {
        "A": {"changed": 0, "beneficial": 0, "total_steps": 0, "gold_rank_delta_sum": 0.0, "gold_rank_n": 0, "margin_delta_sum": 0.0, "margin_delta_n": 0},
        "B": {"changed": 0, "beneficial": 0, "total_steps": 0, "gold_rank_delta_sum": 0.0, "gold_rank_n": 0, "margin_delta_sum": 0.0, "margin_delta_n": 0},
        "AB": {"changed": 0, "beneficial": 0, "total_steps": 0, "gold_rank_delta_sum": 0.0, "gold_rank_n": 0, "margin_delta_sum": 0.0, "margin_delta_n": 0},
    }

    def top1_top2_margin(logits_vec: torch.Tensor) -> float:
        v = torch.topk(logits_vec, k=2, dim=-1).values
        return float((v[0] - v[1]).item())

    def apply_A(logits_img: torch.Tensor, vpmi: torch.Tensor, active: bool) -> torch.Tensor:
        out = logits_img.clone()
        if not active:
            return out
        kk = int(min(max(1, top_k_rank), int(out.numel())))
        top_ids = torch.topk(logits_img, k=kk, dim=-1).indices
        vp = vpmi[top_ids]
        order = torch.argsort(vp, descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(kk, device=order.device)
        rank_pct = (float(kk - 1) - rank.float()) / float(max(1, kk - 1))
        bias = float(alpha_rank) * (rank_pct - 0.5)
        out[top_ids] = out[top_ids] + bias
        return out

    def apply_B(logits_img: torch.Tensor, vpmi: torch.Tensor, active: bool) -> torch.Tensor:
        out = logits_img.clone()
        if not active:
            return out
        kk = int(min(max(1, state_top_k), int(out.numel())))
        top_ids = torch.topk(out, k=kk, dim=-1).indices
        top_v = vpmi[top_ids]
        mu = float(torch.mean(top_v).item())
        sd = float(torch.std(top_v, unbiased=False).item())
        thr = float(mu + float(prefix_spike_z) * sd)
        tok_strs = tokenizer.convert_ids_to_tokens(top_ids.tolist())
        for tid, tok in zip(top_ids.tolist(), tok_strs):
            v = float(vpmi[int(tid)].item())
            if token_is_prefix(tok):
                if v > thr:
                    out[int(tid)] -= float(beta_prefix) * float(v - thr)
            else:
                if v >= float(suffix_vpmi_floor):
                    boost = float(math.tanh((v - float(suffix_vpmi_floor)) / 2.0))
                    out[int(tid)] += float(beta_suffix) * boost
        return out

    def gold_ids_from_answer(ans: str) -> List[int]:
        core = str(pf.first_clause(ans)).strip()
        if core == "":
            return []
        vars_ = pf.build_core_token_variants(tokenizer, core)
        if len(vars_) > 0:
            return [int(x) for x in vars_[0]]
        ids = tokenizer(" " + core, add_special_tokens=False).input_ids
        return [int(x) for x in ids]

    for r in tqdm(sample_rows, total=len(sample_rows), desc="replay", dynamic_ncols=True):
        q = str(r.get("question", ""))
        ans = str(r.get("answer", ""))
        img_id = str(r.get("imageId", ""))
        p = os.path.join(image_root, f"{img_id}.jpg")
        if not os.path.isfile(p):
            continue
        try:
            img_prompt = pf.build_prompt(
                question=q,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            q_prompt = pf.build_prompt(
                question=q,
                conv_mode=conv_mode,
                with_image_token=False,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            input_ids_img = tokenizer_image_token(img_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            input_ids_q = tokenizer(q_prompt, return_tensors="pt").input_ids.to(device)
            image = Image.open(p).convert("RGB")
            image_sizes = [image.size]
            images_tensor = process_images([image], image_processor, model.config).to(device=model.device, dtype=torch.float16)
        except Exception:
            continue

        try:
            out_img = model(
                input_ids=input_ids_img,
                images=images_tensor,
                image_sizes=image_sizes,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            out_q = model(
                input_ids=input_ids_q,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
        except Exception:
            continue
        past_img = out_img.past_key_values
        past_q = out_q.past_key_values
        logits_img = out_img.logits[:, -1, :].float()
        logits_q = out_q.logits[:, -1, :].float()

        gold_ids = gold_ids_from_answer(ans)
        for t in range(int(max_steps)):
            lv = logits_img[0]
            lq = logits_q[0]
            lp_v = F.log_softmax(lv, dim=-1)
            lp_q = F.log_softmax(lq, dim=-1)
            vpmi = lp_v - lp_q
            base_top1 = int(torch.argmax(lv, dim=-1).item())
            base_margin = top1_top2_margin(lv)

            # A activation: low-margin + enough vpmi gap
            kk = int(min(max(2, top_k_rank), int(vpmi.numel())))
            ids_v = torch.topk(vpmi, k=kk, dim=-1).indices
            vals_v = vpmi[ids_v]
            rank_gap = float((vals_v[0] - vals_v[1]).item())
            a_active = bool((base_margin <= float(tau_low_margin)) and (rank_gap >= float(tau_rank_gap)))

            # B activation: core window + suffix collapse at current top1 token
            tok_top = str(tokenizer.convert_ids_to_tokens(int(base_top1)))
            is_suffix = not token_is_prefix(tok_top)
            cur_v = float(vpmi[int(base_top1)].item())
            b_active = bool((t < int(max(1, b_core_max_steps))) and is_suffix and (cur_v <= float(tau_suffix_collapse)))

            adj_a = apply_A(lv, vpmi, a_active)
            adj_b = apply_B(lv, vpmi, b_active)
            adj_ab = apply_B(apply_A(lv, vpmi, a_active), vpmi, b_active)
            by_mode = {"A": adj_a, "B": adj_b, "AB": adj_ab}

            for mode, adj in by_mode.items():
                st = stats[mode]
                st["total_steps"] += 1
                new_top1 = int(torch.argmax(adj, dim=-1).item())
                changed = bool(new_top1 != base_top1)
                if changed:
                    st["changed"] += 1
                st["margin_delta_sum"] += float(top1_top2_margin(adj) - base_margin)
                st["margin_delta_n"] += 1

                if t < len(gold_ids):
                    g = int(gold_ids[t])
                    # rank = #tokens strictly greater + 1
                    rb = int((lv > lv[g]).sum().item() + 1)
                    ra = int((adj > adj[g]).sum().item() + 1)
                    st["gold_rank_delta_sum"] += float(rb - ra)  # positive is good
                    st["gold_rank_n"] += 1
                    if changed and ra < rb:
                        st["beneficial"] += 1

            # baseline path replay: advance with base top1 token only.
            if eos_id is not None and int(base_top1) == int(eos_id):
                break
            step = torch.tensor([[int(base_top1)]], dtype=torch.long, device=device)
            try:
                out_img = model(
                    input_ids=step,
                    past_key_values=past_img,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
                out_q = model(
                    input_ids=step,
                    past_key_values=past_q,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
            except Exception:
                break
            past_img = out_img.past_key_values
            past_q = out_q.past_key_values
            logits_img = out_img.logits[:, -1, :].float()
            logits_q = out_q.logits[:, -1, :].float()

    out: Dict[str, Any] = {}
    for mode, st in stats.items():
        changed = int(st["changed"])
        beneficial = int(st["beneficial"])
        total_steps = int(st["total_steps"])
        out[mode] = {
            "changed_steps": changed,
            "total_steps": total_steps,
            "top1_change_rate": (None if total_steps == 0 else float(changed / total_steps)),
            "changed_step_precision": (None if changed == 0 else float(beneficial / changed)),
            "gold_rank_delta_mean": (
                None if int(st["gold_rank_n"]) == 0 else float(st["gold_rank_delta_sum"] / int(st["gold_rank_n"]))
            ),
            "margin_delta_mean": (
                None if int(st["margin_delta_n"]) == 0 else float(st["margin_delta_sum"] / int(st["margin_delta_n"]))
            ),
        }
    return out


def eval_e2e_run(run_root: str, baseline_per_sample: str) -> List[Dict[str, Any]]:
    methods = ["a_only", "b_only", "a_plus_b"]
    base_rows = read_csv(baseline_per_sample)
    base_map = {str(r.get("id", "")): r for r in base_rows if str(r.get("error", "")).strip() == ""}
    out: List[Dict[str, Any]] = []
    for m in methods:
        p = os.path.join(run_root, m, "per_sample.csv")
        if not os.path.isfile(p):
            continue
        rows = read_csv(p)
        cur_map = {str(r.get("id", "")): r for r in rows if str(r.get("error", "")).strip() == ""}
        ids = sorted(set(base_map.keys()) & set(cur_map.keys()))
        if len(ids) == 0:
            continue
        gain = harm = switch = same = 0
        succ = 0
        for sid in ids:
            b_ok = as_bool(base_map[sid].get("is_success", ""))
            c_ok = as_bool(cur_map[sid].get("is_success", ""))
            succ += int(c_ok)
            if b_ok != c_ok:
                switch += 1
                if (not b_ok) and c_ok:
                    gain += 1
                if b_ok and (not c_ok):
                    harm += 1
            else:
                same += 1
        out.append(
            {
                "run_root": run_root,
                "method": m,
                "n": int(len(ids)),
                "accuracy": float(succ / len(ids)),
                "switch": int(switch),
                "switch_rate": float(switch / len(ids)),
                "gain": int(gain),
                "harm": int(harm),
                "net": int(gain - harm),
                "precision_gain": (None if switch == 0 else float(gain / switch)),
                "same": int(same),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="One-bundle validation for A/B hypothesis.")
    ap.add_argument("--beam_dir", type=str, required=True, help="beam6 directory with per_sample/per_candidate")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--baseline_per_sample", type=str, required=True, help="greedy baseline per_sample.csv for e2e deltas")
    ap.add_argument("--run200_root", type=str, default="", help="dualstream run root (contains a_only,b_only,a_plus_b)")
    ap.add_argument("--run1000_root", type=str, default="", help="dualstream run root (contains a_only,b_only,a_plus_b)")

    # Oracle feature params
    ap.add_argument("--k_prefix", type=int, default=2)
    ap.add_argument("--k_suffix", type=int, default=2)
    ap.add_argument("--b_prefix_penalty", type=float, default=0.5)

    # Replay options
    ap.add_argument("--run_replay", action="store_true")
    ap.add_argument("--questions_json", type=str, default="")
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--replay_num_samples", type=int, default=200)
    ap.add_argument("--replay_max_steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_flash_attn", action="store_true")
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])

    # token-level A/B params (same names as dualstream)
    ap.add_argument("--top_k_rank", type=int, default=50)
    ap.add_argument("--alpha_rank", type=float, default=1.2)
    ap.add_argument("--state_top_k", type=int, default=30)
    ap.add_argument("--prefix_spike_z", type=float, default=1.8)
    ap.add_argument("--beta_prefix", type=float, default=0.25)
    ap.add_argument("--suffix_vpmi_floor", type=float, default=0.0)
    ap.add_argument("--beta_suffix", type=float, default=0.2)

    # constrained activation params for replay
    ap.add_argument("--tau_low_margin", type=float, default=1.0)
    ap.add_argument("--tau_rank_gap", type=float, default=0.15)
    ap.add_argument("--b_core_max_steps", type=int, default=4)
    ap.add_argument("--tau_suffix_collapse", type=float, default=-0.3)

    args = ap.parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch is not None:
        torch.manual_seed(int(args.seed))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Stage 1: Offline Oracle + constrained search.
    sample_map = load_pool(
        beam_dir=os.path.abspath(args.beam_dir),
        k_prefix=int(args.k_prefix),
        k_suffix=int(args.k_suffix),
        b_prefix_penalty=float(args.b_prefix_penalty),
    )
    stage1_rows: List[Dict[str, Any]] = []
    stage1_detail_rows: List[Dict[str, Any]] = []
    for mode in ["A", "B", "AB"]:
        uncon = eval_switch(sample_map, mode=mode, constrained=False, params={})
        best = search_constrained(sample_map, mode=mode)
        stage1_rows.append(
            {
                "stage": "oracle_unconstrained",
                "mode": mode,
                "n": uncon["n"],
                "accuracy_base": uncon["accuracy_base"],
                "accuracy_new": uncon["accuracy_new"],
                "delta_acc": uncon["delta_acc"],
                "switch": uncon["switch"],
                "switch_rate": uncon["switch_rate"],
                "gain": uncon["gain"],
                "harm": uncon["harm"],
                "net": uncon["net"],
                "precision_gain": uncon["precision_gain"],
                "params_json": "",
            }
        )
        stage1_rows.append(
            {
                "stage": "oracle_constrained_best",
                "mode": mode,
                "n": best["n"],
                "accuracy_base": best["accuracy_base"],
                "accuracy_new": best["accuracy_new"],
                "delta_acc": best["delta_acc"],
                "switch": best["switch"],
                "switch_rate": best["switch_rate"],
                "gain": best["gain"],
                "harm": best["harm"],
                "net": best["net"],
                "precision_gain": best["precision_gain"],
                "params_json": json.dumps(best.get("best_params", {}), ensure_ascii=False),
            }
        )
        for rr in best.get("rows", []):
            stage1_detail_rows.append(rr)

    write_csv(os.path.join(out_dir, "stage1_oracle_summary.csv"), stage1_rows)
    write_csv(os.path.join(out_dir, "stage1_oracle_best_switch_rows.csv"), stage1_detail_rows)

    # Stage 2: Replay
    replay_metrics = None
    if bool(args.run_replay):
        if torch is None:
            raise RuntimeError("PyTorch not available for replay stage.")
        qpath = os.path.abspath(args.questions_json)
        if not os.path.isfile(qpath):
            raise FileNotFoundError(f"missing --questions_json: {qpath}")
        qrows = pf.read_questions(qpath)
        # align with beam ids to keep distribution consistent.
        beam_ids = sorted(sample_map.keys())
        qmap = {str(r.get("id", "")): r for r in qrows}
        selected = [qmap[sid] for sid in beam_ids if sid in qmap]
        selected = selected[: int(max(1, args.replay_num_samples))]
        replay_metrics = replay_token_counterfactual(
            sample_rows=selected,
            image_root=os.path.abspath(args.image_root),
            model_path=str(args.model_path),
            model_base=args.model_base,
            conv_mode_override=args.conv_mode,
            max_steps=int(args.replay_max_steps),
            top_k_rank=int(args.top_k_rank),
            alpha_rank=float(args.alpha_rank),
            state_top_k=int(args.state_top_k),
            prefix_spike_z=float(args.prefix_spike_z),
            beta_prefix=float(args.beta_prefix),
            suffix_vpmi_floor=float(args.suffix_vpmi_floor),
            beta_suffix=float(args.beta_suffix),
            tau_low_margin=float(args.tau_low_margin),
            tau_rank_gap=float(args.tau_rank_gap),
            b_core_max_steps=int(args.b_core_max_steps),
            tau_suffix_collapse=float(args.tau_suffix_collapse),
            use_flash_attn=bool(args.use_flash_attn),
            attn_impl=str(args.attn_impl),
        )

    # Stage 3/4: E2E from run roots (200 / 1000)
    e2e_rows: List[Dict[str, Any]] = []
    if str(args.run200_root).strip() != "":
        e2e_rows.extend(eval_e2e_run(os.path.abspath(args.run200_root), os.path.abspath(args.baseline_per_sample)))
    if str(args.run1000_root).strip() != "":
        e2e_rows.extend(eval_e2e_run(os.path.abspath(args.run1000_root), os.path.abspath(args.baseline_per_sample)))
    write_csv(os.path.join(out_dir, "stage3_4_e2e_summary.csv"), e2e_rows)

    # Stage 5: pre-defined criteria checks
    checks: List[Dict[str, Any]] = []
    # C1: changed_step_precision > 0.55
    if replay_metrics is not None:
        for mode in ["A", "B", "AB"]:
            csp = safe_float(replay_metrics.get(mode, {}).get("changed_step_precision"))
            checks.append(
                {
                    "criterion": "changed_step_precision>0.55",
                    "mode": mode,
                    "value": csp,
                    "pass": (None if csp is None else bool(csp > 0.55)),
                }
            )
    # C2: gain-harm > 0 on 200 and 1000 for AB
    for rr in e2e_rows:
        if str(rr.get("method")) != "a_plus_b":
            continue
        checks.append(
            {
                "criterion": "AB_net>0",
                "mode": str(rr.get("run_root")),
                "value": int(rr.get("net", 0)),
                "pass": bool(int(rr.get("net", 0)) > 0),
            }
        )
    # C3: AB better than A and B on each run root.
    roots = sorted(set(str(r.get("run_root")) for r in e2e_rows))
    for root in roots:
        sub = [r for r in e2e_rows if str(r.get("run_root")) == root]
        mp = {str(r.get("method")): safe_float(r.get("accuracy")) for r in sub}
        if "a_plus_b" in mp and "a_only" in mp and "b_only" in mp:
            ab = mp["a_plus_b"]
            ao = mp["a_only"]
            bo = mp["b_only"]
            ok = None
            if ab is not None and ao is not None and bo is not None:
                ok = bool(ab > ao and ab > bo)
            checks.append(
                {
                    "criterion": "AB>max(A,B)",
                    "mode": root,
                    "value": ab,
                    "pass": ok,
                }
            )
    write_csv(os.path.join(out_dir, "stage5_criteria_checks.csv"), checks)

    summary = {
        "inputs": {
            "beam_dir": os.path.abspath(args.beam_dir),
            "baseline_per_sample": os.path.abspath(args.baseline_per_sample),
            "run200_root": (None if str(args.run200_root).strip() == "" else os.path.abspath(args.run200_root)),
            "run1000_root": (None if str(args.run1000_root).strip() == "" else os.path.abspath(args.run1000_root)),
            "run_replay": bool(args.run_replay),
            "questions_json": (None if str(args.questions_json).strip() == "" else os.path.abspath(args.questions_json)),
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "k_prefix": int(args.k_prefix),
            "k_suffix": int(args.k_suffix),
            "b_prefix_penalty": float(args.b_prefix_penalty),
            "replay_num_samples": int(args.replay_num_samples),
            "replay_max_steps": int(args.replay_max_steps),
            "top_k_rank": int(args.top_k_rank),
            "alpha_rank": float(args.alpha_rank),
            "state_top_k": int(args.state_top_k),
            "prefix_spike_z": float(args.prefix_spike_z),
            "beta_prefix": float(args.beta_prefix),
            "suffix_vpmi_floor": float(args.suffix_vpmi_floor),
            "beta_suffix": float(args.beta_suffix),
            "tau_low_margin": float(args.tau_low_margin),
            "tau_rank_gap": float(args.tau_rank_gap),
            "b_core_max_steps": int(args.b_core_max_steps),
            "tau_suffix_collapse": float(args.tau_suffix_collapse),
            "seed": int(args.seed),
        },
        "stage1_rows": int(len(stage1_rows)),
        "stage3_4_rows": int(len(e2e_rows)),
        "stage5_rows": int(len(checks)),
        "replay_metrics": replay_metrics,
        "outputs": {
            "stage1_oracle_summary_csv": os.path.join(out_dir, "stage1_oracle_summary.csv"),
            "stage1_oracle_best_switch_rows_csv": os.path.join(out_dir, "stage1_oracle_best_switch_rows.csv"),
            "stage3_4_e2e_summary_csv": os.path.join(out_dir, "stage3_4_e2e_summary.csv"),
            "stage5_criteria_checks_csv": os.path.join(out_dir, "stage5_criteria_checks.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "stage1_oracle_summary.csv"))
    print("[saved]", os.path.join(out_dir, "stage1_oracle_best_switch_rows.csv"))
    print("[saved]", os.path.join(out_dir, "stage3_4_e2e_summary.csv"))
    print("[saved]", os.path.join(out_dir, "stage5_criteria_checks.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()

