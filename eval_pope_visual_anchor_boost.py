#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable

import analyze_artrap_pairwise_fragility as pf


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def normalize_yesno(x: Any) -> Optional[str]:
    s = str("" if x is None else x).strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    m = re.search(r"\b(yes|no)\b", s)
    if m:
        return str(m.group(1))
    return None


def collect_single_token_ids(tokenizer, word: str) -> List[int]:
    variants = [
        word,
        " " + word,
        word.capitalize(),
        " " + word.capitalize(),
        word.upper(),
        " " + word.upper(),
    ]
    ids: Set[int] = set()
    for s in variants:
        try:
            toks = tokenizer(s, add_special_tokens=False).input_ids
        except Exception:
            continue
        if isinstance(toks, list) and len(toks) == 1:
            ids.add(int(toks[0]))
    return sorted(ids)


def entropy_topk(scores_row: torch.Tensor, k: int = 5) -> float:
    if scores_row.ndim != 1 or int(scores_row.numel()) <= 1:
        return float("nan")
    kk = int(min(max(1, int(k)), int(scores_row.numel())))
    vals = torch.topk(scores_row, k=kk, dim=-1).values
    p = torch.softmax(vals.float(), dim=-1)
    h = -(p * torch.log(torch.clamp(p, min=1e-12))).sum()
    return float(h.item())


def resolve_image_path(image_root: str, image_id: str) -> Optional[str]:
    iid = str(image_id or "").strip()
    if iid == "":
        return None
    cands = [iid]
    if not iid.lower().endswith(".jpg"):
        cands.append(iid + ".jpg")
    for c in cands:
        p = os.path.join(image_root, c)
        if os.path.isfile(p):
            return p
    return None


def extract_new_ids(output_ids: torch.Tensor, prompt_ids: torch.Tensor, eos_token_id: Optional[int]) -> List[int]:
    seq = [int(x) for x in output_ids[0].tolist()]
    pref = [int(x) for x in prompt_ids[0].tolist()]
    if len(seq) >= len(pref) and seq[: len(pref)] == pref:
        gen = seq[len(pref) :]
    else:
        gen = seq
    if eos_token_id is not None and int(eos_token_id) in gen:
        gen = gen[: gen.index(int(eos_token_id))]
    return [int(x) for x in gen]


class VisualAnchorBoostProcessor(LogitsProcessor):
    """
    Base:
      Logit_new = Logit_orig + alpha * max_p cos(E_cand, V_p), applied on current top-k tokens only.
    Optional risk-gate:
      - Build per-step risk score from groundedness/yes-no gap/entropy.
      - Reduce boost by (1 - risk * risk_reduce_boost).
      - Optionally apply additional penalty to yes-token logits.
    """

    def __init__(
        self,
        token_embed_weight: torch.Tensor,
        vis_tokens: torch.Tensor,
        yes_token_ids: Sequence[int],
        no_token_ids: Sequence[int],
        alpha: float,
        topk_tokens: int,
        apply_after_step: int = 0,
        patch_topk: int = 1,
        prompt_len: int = 0,
        use_risk_gate: bool = False,
        risk_tau_ground_gap: float = 0.0,
        risk_tau_yesno_gap: float = 0.0,
        risk_entropy_thr: float = 0.8,
        risk_w_ground: float = 0.5,
        risk_w_yesgap: float = 0.3,
        risk_w_conf: float = 0.2,
        risk_lambda: float = 0.8,
        risk_max_penalty: float = 1.5,
        risk_reduce_boost: float = 1.0,
        risk_penalty_mode: str = "yes_ids",
        risk_min_step: int = 0,
        risk_only_first_step: bool = False,
    ) -> None:
        super().__init__()
        self.embed_weight = token_embed_weight
        self.vis_n = F.normalize(vis_tokens.float(), dim=-1)
        self.alpha = float(alpha)
        self.topk_tokens = int(max(1, topk_tokens))
        self.apply_after_step = int(max(0, apply_after_step))
        self.patch_topk = int(max(1, patch_topk))
        self.prompt_len = int(max(0, prompt_len))
        self.use_risk_gate = bool(use_risk_gate)
        self.risk_tau_ground_gap = float(risk_tau_ground_gap)
        self.risk_tau_yesno_gap = float(risk_tau_yesno_gap)
        self.risk_entropy_thr = float(risk_entropy_thr)
        self.risk_w_ground = float(risk_w_ground)
        self.risk_w_yesgap = float(risk_w_yesgap)
        self.risk_w_conf = float(risk_w_conf)
        self.risk_lambda = float(max(0.0, risk_lambda))
        self.risk_max_penalty = float(max(0.0, risk_max_penalty))
        self.risk_reduce_boost = float(min(1.0, max(0.0, risk_reduce_boost)))
        self.risk_penalty_mode = str(risk_penalty_mode)
        self.risk_min_step = int(max(0, risk_min_step))
        self.risk_only_first_step = bool(risk_only_first_step)
        self.yes_ids = [int(x) for x in yes_token_ids]
        self.no_ids = [int(x) for x in no_token_ids]
        self.trace: List[Dict[str, Any]] = []

        dev = self.embed_weight.device
        self._yes_ids_t = torch.tensor(self.yes_ids, dtype=torch.long, device=dev) if len(self.yes_ids) > 0 else None
        self._no_ids_t = torch.tensor(self.no_ids, dtype=torch.long, device=dev) if len(self.no_ids) > 0 else None
        self._yes_emb_n = None
        self._no_emb_n = None
        if self._yes_ids_t is not None:
            self._yes_emb_n = F.normalize(self.embed_weight[self._yes_ids_t].float(), dim=-1)
        if self._no_ids_t is not None:
            self._no_emb_n = F.normalize(self.embed_weight[self._no_ids_t].float(), dim=-1)

        self.sample_id: str = ""
        self.answer_gt: str = ""
        self.question: str = ""
        self._step_counter: int = 0

    def set_context(self, sample_id: str, answer_gt: str, question: str) -> None:
        self.sample_id = str(sample_id)
        self.answer_gt = str(answer_gt)
        self.question = str(question)
        self.trace = []
        self._step_counter = 0

    def _yn_gap(self, row_scores: torch.Tensor) -> Optional[float]:
        if self._yes_ids_t is None or self._no_ids_t is None:
            return None
        if int(self._yes_ids_t.numel()) == 0 or int(self._no_ids_t.numel()) == 0:
            return None
        ys = torch.logsumexp(row_scores[self._yes_ids_t], dim=0)
        ns = torch.logsumexp(row_scores[self._no_ids_t], dim=0)
        return float((ys - ns).item())

    def _yn_groundedness(self) -> Dict[str, Optional[float]]:
        out = {"g_yes": None, "g_no": None}
        if self._yes_emb_n is not None:
            sim_y = torch.matmul(self._yes_emb_n, self.vis_n.t())
            out["g_yes"] = float(sim_y.max().item())
        if self._no_emb_n is not None:
            sim_n = torch.matmul(self._no_emb_n, self.vis_n.t())
            out["g_no"] = float(sim_n.max().item())
        return out

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [B, V]
        if self.alpha == 0.0:
            return scores
        if scores.ndim != 2:
            return scores
        bsz, vocab = int(scores.size(0)), int(scores.size(1))
        if bsz <= 0 or vocab <= 1:
            return scores
        # Generation may pass only last token when KV-cache is enabled.
        # Track step with internal counter instead of input length.
        gen_steps = int(self._step_counter)
        row0 = scores[0]
        yn_gap_pre = self._yn_gap(row0)
        h5_pre = entropy_topk(row0, k=5)
        pre_top1_val, pre_top1_idx = torch.max(row0, dim=-1)
        yg = self._yn_groundedness()
        ggap = (
            None
            if (yg["g_yes"] is None or yg["g_no"] is None)
            else float(yg["g_yes"] - yg["g_no"])
        )

        applied = bool(gen_steps >= self.apply_after_step)
        if not applied:
            self.trace.append(
                {
                    "id": self.sample_id,
                    "answer_gt": self.answer_gt,
                    "question": self.question,
                    "step": int(gen_steps),
                    "applied": False,
                    "yes_no_gap_pre": yn_gap_pre,
                    "yes_no_gap_post": yn_gap_pre,
                    "entropy_top5_pre": h5_pre,
                    "entropy_top5_post": h5_pre,
                    "top1_pre_id": int(pre_top1_idx.item()),
                    "top1_post_id": int(pre_top1_idx.item()),
                    "top1_pre_logit": float(pre_top1_val.item()),
                    "top1_post_logit": float(pre_top1_val.item()),
                    "groundedness_yes": yg["g_yes"],
                    "groundedness_no": yg["g_no"],
                    "groundedness_gap_yes_minus_no": (
                        ggap
                    ),
                    "max_boost_topk": 0.0,
                    "mean_boost_topk": 0.0,
                    "risk_score": 0.0,
                    "effective_alpha": 0.0,
                    "risk_penalty_applied": 0.0,
                }
            )
            self._step_counter += 1
            return scores

        k_tok = int(min(self.topk_tokens, vocab))
        _, top_idx = torch.topk(scores, k=k_tok, dim=-1)  # [B, K]

        tok_emb = self.embed_weight[top_idx]  # [B, K, D]
        tok_emb = F.normalize(tok_emb.float(), dim=-1)
        sim = torch.matmul(tok_emb, self.vis_n.t())  # [B, K, P]
        if self.patch_topk <= 1:
            boost = sim.max(dim=-1).values  # [B, K]
        else:
            kk = int(min(self.patch_topk, int(sim.size(-1))))
            boost = torch.topk(sim, k=kk, dim=-1).values.mean(dim=-1)

        # Per-step risk score in [0, 1]
        risk_score = 0.0
        risk_step_ok = bool(gen_steps >= self.risk_min_step)
        if self.risk_only_first_step and gen_steps > 0:
            risk_step_ok = False
        if self.use_risk_gate and risk_step_ok:
            c_ground = 0.0
            c_yesgap = 0.0
            c_conf = 0.0
            if ggap is not None:
                # lower groundedness gap can indicate risk
                c_ground = max(0.0, float(self.risk_tau_ground_gap) - float(ggap))
            if yn_gap_pre is not None:
                # lower yes-no gap can indicate risk
                c_yesgap = max(0.0, float(self.risk_tau_yesno_gap) - float(yn_gap_pre))
            if h5_pre is not None and math.isfinite(float(h5_pre)):
                # higher entropy can indicate risk
                c_conf = max(0.0, float(h5_pre) - float(self.risk_entropy_thr))
            risk_raw = (
                float(self.risk_w_ground) * c_ground
                + float(self.risk_w_yesgap) * c_yesgap
                + float(self.risk_w_conf) * c_conf
            )
            risk_score = float(min(1.0, max(0.0, risk_raw)))

        eff_alpha = float(self.alpha) * (1.0 - float(self.risk_reduce_boost) * float(risk_score))

        out = scores.clone()
        out.scatter_add_(1, top_idx, float(eff_alpha) * boost.to(out.dtype))
        risk_penalty_applied = 0.0
        if self.use_risk_gate and risk_score > 0.0 and self.risk_lambda > 0.0 and risk_step_ok:
            # Keep penalty in logit units (not scaled by max_logit) to avoid collapse.
            pen = float(self.risk_lambda) * float(risk_score)
            pen = float(min(float(self.risk_max_penalty), max(0.0, pen)))
            if (
                self.risk_penalty_mode == "yes_ids"
                and self._yes_ids_t is not None
                and int(self._yes_ids_t.numel()) > 0
                and yn_gap_pre is not None
                and float(yn_gap_pre) > float(self.risk_tau_yesno_gap)
            ):
                out[:, self._yes_ids_t] = out[:, self._yes_ids_t] - pen
                risk_penalty_applied = float(pen)
            elif self.risk_penalty_mode == "top1":
                out[0, int(pre_top1_idx.item())] = out[0, int(pre_top1_idx.item())] - float(pen)
                risk_penalty_applied = float(pen)
            elif self.risk_penalty_mode == "none":
                risk_penalty_applied = 0.0

        row1 = out[0]
        yn_gap_post = self._yn_gap(row1)
        h5_post = entropy_topk(row1, k=5)
        post_top1_val, post_top1_idx = torch.max(row1, dim=-1)
        b0 = boost[0]
        self.trace.append(
            {
                "id": self.sample_id,
                "answer_gt": self.answer_gt,
                "question": self.question,
                "step": int(gen_steps),
                "applied": True,
                "yes_no_gap_pre": yn_gap_pre,
                "yes_no_gap_post": yn_gap_post,
                "entropy_top5_pre": h5_pre,
                "entropy_top5_post": h5_post,
                "top1_pre_id": int(pre_top1_idx.item()),
                "top1_post_id": int(post_top1_idx.item()),
                "top1_pre_logit": float(pre_top1_val.item()),
                "top1_post_logit": float(post_top1_val.item()),
                "groundedness_yes": yg["g_yes"],
                "groundedness_no": yg["g_no"],
                "groundedness_gap_yes_minus_no": (
                    ggap
                ),
                "max_boost_topk": float(torch.max(b0).item()),
                "mean_boost_topk": float(torch.mean(b0).item()),
                "risk_score": float(risk_score),
                "effective_alpha": float(eff_alpha),
                "risk_penalty_applied": float(risk_penalty_applied),
            }
        )
        self._step_counter += 1
        return out


def compute_confusion_counts(rows: List[Dict[str, Any]], pred_key: str) -> Dict[str, int]:
    tp_yes = 0
    tn_no = 0
    fp_no_to_yes = 0
    fn_yes_to_no = 0
    for r in rows:
        gt = normalize_yesno(r.get("answer_gt"))
        pd = normalize_yesno(r.get(pred_key))
        if gt not in {"yes", "no"} or pd not in {"yes", "no"}:
            continue
        if gt == "yes" and pd == "yes":
            tp_yes += 1
        elif gt == "no" and pd == "no":
            tn_no += 1
        elif gt == "no" and pd == "yes":
            fp_no_to_yes += 1
        elif gt == "yes" and pd == "no":
            fn_yes_to_no += 1
    return {
        "tp_yes": int(tp_yes),
        "tn_no": int(tn_no),
        "fp_no_to_yes": int(fp_no_to_yes),
        "fn_yes_to_no": int(fn_yes_to_no),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="POPE zero-shot intervention with visual-anchor boost logits processor.")
    ap.add_argument("--samples_csv", type=str, required=True, help="POPE baseline per_sample.csv (must include question/answer/image_id)")
    ap.add_argument("--image_root", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--alpha", type=float, default=1.5, help="Boost scale alpha.")
    ap.add_argument("--topk_tokens", type=int, default=10, help="Apply boost only to current top-k token candidates.")
    ap.add_argument("--patch_topk", type=int, default=1, help="Aggregate top-k visual patch sims per candidate.")
    ap.add_argument("--apply_after_step", type=int, default=0, help="Start boost after this generated length.")
    ap.add_argument("--use_risk_gate", action="store_true", help="Enable risk-gated intervention.")
    ap.add_argument("--risk_tau_ground_gap", type=float, default=0.00, help="Risk increases when (g_yes - g_no) is below this.")
    ap.add_argument("--risk_tau_yesno_gap", type=float, default=0.00, help="Risk increases when yes-no gap is below this.")
    ap.add_argument("--risk_entropy_thr", type=float, default=0.80, help="Risk increases when top5 entropy is above this.")
    ap.add_argument("--risk_w_ground", type=float, default=0.50, help="Weight for groundedness gap term.")
    ap.add_argument("--risk_w_yesgap", type=float, default=0.30, help="Weight for yes-no gap term.")
    ap.add_argument("--risk_w_conf", type=float, default=0.20, help="Weight for confidence (low entropy) term.")
    ap.add_argument("--risk_lambda", type=float, default=0.80, help="Penalty scale under risk gate.")
    ap.add_argument("--risk_max_penalty", type=float, default=1.50, help="Max penalty in logit units.")
    ap.add_argument("--risk_reduce_boost", type=float, default=1.00, help="How much to reduce boost by risk (0~1).")
    ap.add_argument("--risk_penalty_mode", type=str, default="yes_ids", choices=["yes_ids", "top1", "none"])
    ap.add_argument("--risk_min_step", type=int, default=0, help="Apply risk gate after this generated step.")
    ap.add_argument("--risk_only_first_step", action="store_true", help="Apply risk gate only on first generated token.")

    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_token_trace", action="store_true", help="Save per-step decoding traces.")
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows_in = read_csv(os.path.abspath(args.samples_csv))
    if int(args.num_samples) > 0:
        rows_in = rows_in[: int(args.num_samples)]
    if len(rows_in) == 0:
        raise RuntimeError("No rows in samples_csv.")

    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from PIL import Image

    pf.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
    pf.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
    pf.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
    pf.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
    pf.conv_templates = conv_templates

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

    conv_mode = pf.resolve_conv_mode(model_name, args.conv_mode)
    device = model.get_model().embed_tokens.weight.device
    eos_id = getattr(tokenizer, "eos_token_id", None)

    yes_token_ids = collect_single_token_ids(tokenizer, "yes")
    no_token_ids = collect_single_token_ids(tokenizer, "no")
    if len(yes_token_ids) == 0 or len(no_token_ids) == 0:
        raise RuntimeError("Failed to resolve single-token yes/no ids from tokenizer.")

    per_rows: List[Dict[str, Any]] = []
    token_trace_rows: List[Dict[str, Any]] = []
    skipped = 0

    pbar = tqdm(rows_in, total=len(rows_in), desc="pope-visual-anchor-boost", dynamic_ncols=True)
    for rr in pbar:
        sid = str(rr.get("id") or "")
        question = str(rr.get("question") or "")
        answer = str(rr.get("answer") or "")
        image_id = str(rr.get("image_id") or rr.get("imageId") or "")
        base_pred_text = str(rr.get("pred_answer_eval") or rr.get("pred_text") or rr.get("champ_text") or "").strip()
        if sid == "" or question == "" or answer == "" or image_id == "":
            skipped += 1
            continue
        gt = normalize_yesno(answer)
        if gt not in {"yes", "no"}:
            skipped += 1
            continue
        base_pred = normalize_yesno(base_pred_text)

        image_path = resolve_image_path(args.image_root, image_id)
        if image_path is None:
            skipped += 1
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            images_tensor = process_images([img], image_processor, model.config).to(
                device=model.device,
                dtype=torch.float16,
            )
            image_sizes = [img.size]

            prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            with torch.no_grad():
                vis = model.encode_images(images_tensor)
                if isinstance(vis, (list, tuple)):
                    vis = vis[0]
                if vis.ndim == 3:
                    vis = vis[0]
                if vis.ndim != 2 or int(vis.size(0)) < 2:
                    raise RuntimeError("bad visual token shape")

                token_embed_weight = model.get_model().embed_tokens.weight
                processor = VisualAnchorBoostProcessor(
                    token_embed_weight=token_embed_weight,
                    vis_tokens=vis.to(token_embed_weight.device),
                    yes_token_ids=yes_token_ids,
                    no_token_ids=no_token_ids,
                    alpha=float(args.alpha),
                    topk_tokens=int(args.topk_tokens),
                    apply_after_step=int(args.apply_after_step),
                    patch_topk=int(args.patch_topk),
                    prompt_len=int(prompt_ids.size(1)),
                    use_risk_gate=bool(args.use_risk_gate),
                    risk_tau_ground_gap=float(args.risk_tau_ground_gap),
                    risk_tau_yesno_gap=float(args.risk_tau_yesno_gap),
                    risk_entropy_thr=float(args.risk_entropy_thr),
                    risk_w_ground=float(args.risk_w_ground),
                    risk_w_yesgap=float(args.risk_w_yesgap),
                    risk_w_conf=float(args.risk_w_conf),
                    risk_lambda=float(args.risk_lambda),
                    risk_max_penalty=float(args.risk_max_penalty),
                    risk_reduce_boost=float(args.risk_reduce_boost),
                    risk_penalty_mode=str(args.risk_penalty_mode),
                    risk_min_step=int(args.risk_min_step),
                    risk_only_first_step=bool(args.risk_only_first_step),
                )
                processor.set_context(sample_id=sid, answer_gt=gt, question=question)
                lp = LogitsProcessorList([processor])

                do_sample = bool(float(args.temperature) > 0.0)
                gen_out = model.generate(
                    prompt_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=do_sample,
                    temperature=(float(max(1e-5, args.temperature)) if do_sample else 0.0),
                    top_p=float(min(1.0, max(0.0, args.top_p))),
                    num_beams=int(max(1, args.num_beams)),
                    max_new_tokens=int(max(1, args.max_new_tokens)),
                    use_cache=True,
                    logits_processor=lp,
                )

            gen_ids = extract_new_ids(gen_out, prompt_ids, eos_id)
            boost_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            boost_pred = normalize_yesno(boost_text)
            boost_ok = (boost_pred == gt) if boost_pred in {"yes", "no"} else False
            base_ok = (base_pred == gt) if base_pred in {"yes", "no"} else False

            if bool(args.save_token_trace):
                for i, tr in enumerate(processor.trace):
                    rr_t = dict(tr)
                    rr_t["step_index"] = int(i)
                    if i < len(gen_ids):
                        tid = int(gen_ids[i])
                        rr_t["gen_token_id"] = tid
                        try:
                            rr_t["gen_token_str"] = str(tokenizer.convert_ids_to_tokens(tid))
                        except Exception:
                            rr_t["gen_token_str"] = None
                    else:
                        rr_t["gen_token_id"] = None
                        rr_t["gen_token_str"] = None
                    rr_t["base_pred"] = base_pred
                    rr_t["boost_pred"] = boost_pred
                    rr_t["base_ok"] = bool(base_ok)
                    rr_t["boost_ok"] = bool(boost_ok)
                    token_trace_rows.append(rr_t)

            per_rows.append(
                {
                    "id": sid,
                    "image_id": image_id,
                    "question": question,
                    "answer_gt": gt,
                    "base_pred": base_pred,
                    "base_pred_text": base_pred_text,
                    "boost_pred": boost_pred,
                    "boost_text": boost_text,
                    "base_ok": bool(base_ok),
                    "boost_ok": bool(boost_ok),
                    "changed_pred": bool((base_pred in {"yes", "no"}) and (boost_pred in {"yes", "no"}) and (base_pred != boost_pred)),
                    "gain": bool((not base_ok) and boost_ok),
                    "harm": bool(base_ok and (not boost_ok)),
                }
            )
        except Exception as e:
            skipped += 1
            per_rows.append(
                {
                    "id": sid,
                    "image_id": image_id,
                    "question": question,
                    "answer_gt": gt,
                    "base_pred": base_pred,
                    "base_pred_text": base_pred_text,
                    "boost_pred": None,
                    "boost_text": None,
                    "base_ok": (None if base_pred not in {"yes", "no"} else bool(base_pred == gt)),
                    "boost_ok": None,
                    "changed_pred": None,
                    "gain": None,
                    "harm": None,
                    "error": str(e),
                }
            )

    valid = [r for r in per_rows if r.get("boost_ok") is not None]
    if len(valid) == 0:
        raise RuntimeError("No valid rows produced.")

    n = len(valid)
    base_acc = float(sum(1 for r in valid if bool(r.get("base_ok"))) / max(1, n))
    boost_acc = float(sum(1 for r in valid if bool(r.get("boost_ok"))) / max(1, n))
    gain = int(sum(1 for r in valid if bool(r.get("gain"))))
    harm = int(sum(1 for r in valid if bool(r.get("harm"))))
    changed = int(sum(1 for r in valid if bool(r.get("changed_pred"))))
    net_gain = int(gain - harm)

    base_conf = compute_confusion_counts(valid, pred_key="base_pred")
    boost_conf = compute_confusion_counts(valid, pred_key="boost_pred")

    summary = {
        "inputs": {
            "samples_csv": os.path.abspath(args.samples_csv),
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "alpha": float(args.alpha),
            "topk_tokens": int(args.topk_tokens),
            "patch_topk": int(args.patch_topk),
            "apply_after_step": int(args.apply_after_step),
            "use_risk_gate": bool(args.use_risk_gate),
            "risk_tau_ground_gap": float(args.risk_tau_ground_gap),
            "risk_tau_yesno_gap": float(args.risk_tau_yesno_gap),
            "risk_entropy_thr": float(args.risk_entropy_thr),
            "risk_w_ground": float(args.risk_w_ground),
            "risk_w_yesgap": float(args.risk_w_yesgap),
            "risk_w_conf": float(args.risk_w_conf),
            "risk_lambda": float(args.risk_lambda),
            "risk_max_penalty": float(args.risk_max_penalty),
            "risk_reduce_boost": float(args.risk_reduce_boost),
            "risk_penalty_mode": str(args.risk_penalty_mode),
            "risk_min_step": int(args.risk_min_step),
            "risk_only_first_step": bool(args.risk_only_first_step),
            "max_new_tokens": int(args.max_new_tokens),
            "num_beams": int(args.num_beams),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "save_token_trace": bool(args.save_token_trace),
        },
        "counts": {
            "n_input": int(len(rows_in)),
            "n_output_rows": int(len(per_rows)),
            "n_valid": int(len(valid)),
            "n_skipped_or_error": int(skipped),
            "n_changed_pred": int(changed),
            "gain": int(gain),
            "harm": int(harm),
            "net_gain": int(net_gain),
        },
        "metrics": {
            "base_acc": float(base_acc),
            "boost_acc": float(boost_acc),
            "delta_acc": float(boost_acc - base_acc),
        },
        "confusion_base": base_conf,
        "confusion_boost": boost_conf,
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample_intervention.csv"),
            "per_token_trace_csv": (os.path.join(out_dir, "per_token_trace.csv") if bool(args.save_token_trace) else None),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_sample_intervention.csv"), per_rows)
    if bool(args.save_token_trace):
        write_csv(os.path.join(out_dir, "per_token_trace.csv"), token_trace_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample_intervention.csv"))
    if bool(args.save_token_trace):
        print("[saved]", os.path.join(out_dir, "per_token_trace.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
