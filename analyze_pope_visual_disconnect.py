#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable

import analyze_artrap_pairwise_fragility as pf


def parse_bool(x: Any) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if len(xs) == 0:
        return None
    if len(xs) == 1:
        return float(xs[0])
    qq = min(1.0, max(0.0, float(q)))
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = float(pos - lo)
    return float((1.0 - w) * xs[lo] + w * xs[hi])


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


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = [(int(labels[i]), float(scores[i])) for i in range(len(labels))]
    n_pos = int(sum(1 for y, _ in pairs if y == 1))
    n_neg = int(sum(1 for y, _ in pairs if y == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    idxs = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(idxs):
        j = i + 1
        while j < len(idxs) and pairs[idxs[j]][1] == pairs[idxs[i]][1]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[idxs[k]] = float(avg_rank)
        i = j

    sum_pos = float(sum(ranks[i] for i in range(len(pairs)) if pairs[i][0] == 1))
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def ks_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pos = sorted(float(scores[i]) for i in range(len(scores)) if int(labels[i]) == 1)
    neg = sorted(float(scores[i]) for i in range(len(scores)) if int(labels[i]) == 0)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    # Tie-safe KS on union support: D = sup_x |F_pos(x) - F_neg(x)|
    support = sorted(set(pos + neg))
    i = 0
    j = 0
    dmax = 0.0
    for v in support:
        while i < n_pos and pos[i] <= v:
            i += 1
        while j < n_neg and neg[j] <= v:
            j += 1
        f_pos = float(i / n_pos)
        f_neg = float(j / n_neg)
        dmax = max(dmax, abs(f_pos - f_neg))
    return float(dmax)


def bootstrap_ci(
    metric_fn,
    labels: Sequence[int],
    scores: Sequence[float],
    n_boot: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float]]:
    if int(n_boot) <= 0:
        return None, None
    n = int(len(labels))
    if n <= 1:
        return None, None
    rng = random.Random(int(seed))
    vals: List[float] = []
    for _ in range(int(n_boot)):
        ids = [rng.randrange(n) for _ in range(n)]
        lb = [int(labels[i]) for i in ids]
        sc = [float(scores[i]) for i in ids]
        m = metric_fn(lb, sc)
        if m is not None and math.isfinite(float(m)):
            vals.append(float(m))
    if len(vals) == 0:
        return None, None
    return quantile(vals, 0.025), quantile(vals, 0.975)


def choose_cont_ids(tokenizer, text: str) -> List[int]:
    raw = str(text or "")
    if raw.strip() == "":
        return []
    cands = [raw, " " + raw]
    best: List[int] = []
    best_score = -1.0
    target = pf.norm_text(raw)
    for s in cands:
        ids = [int(x) for x in tokenizer(s, add_special_tokens=False).input_ids]
        dec = tokenizer.decode(ids, skip_special_tokens=True)
        score = difflib.SequenceMatcher(None, target, pf.norm_text(dec)).ratio()
        if score > best_score:
            best = ids
            best_score = float(score)
    return best


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


def extract_pope_object(question: str) -> str:
    q = str(question or "").strip().lower()
    # Canonical POPE form: "Is there a/an <object> in the image?"
    m = re.search(r"^is\s+there\s+an?\s+(.+?)\s+in\s+the\s+image\??$", q)
    if m:
        return str(m.group(1)).strip()
    # fallback: content between "there" and "in the image"
    m = re.search(r"there\s+(.+?)\s+in\s+the\s+image", q)
    if m:
        s = str(m.group(1)).strip()
        s = re.sub(r"^an?\s+", "", s)
        return s.strip()
    return ""


def pick_gqa_anchor_phrase(rr: Dict[str, Any], preferred_field: str, pred_text: str, answer: str) -> str:
    cand_fields = [str(preferred_field or "").strip(), "pred_answer_eval", "champ_short_answer", "answer", "pred_text", "champ_text"]
    seen = set()
    for k in cand_fields:
        if k == "" or k in seen:
            continue
        seen.add(k)
        v = str(rr.get(k) or "").strip()
        if v != "":
            return v
    fb = str(pred_text or "").strip()
    if fb != "":
        return fb
    return str(answer or "").strip()


def locate_phrase_start(tokenizer, cont_ids: List[int], phrase: str) -> Optional[int]:
    p = str(phrase or "").strip()
    if p == "":
        return None
    pos, vids = pf.locate_core_span_from_text(tokenizer, cont_ids, p)
    if pos is None or len(vids) == 0:
        return None
    return int(pos)


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


def shuffle_image_blocks(img, grid: int, seed: int):
    from PIL import Image

    grid = int(max(2, grid))
    w, h = img.size
    xs = [int(round(i * w / grid)) for i in range(grid + 1)]
    ys = [int(round(i * h / grid)) for i in range(grid + 1)]
    blocks = []
    for gy in range(grid):
        for gx in range(grid):
            l, r = xs[gx], xs[gx + 1]
            t, b = ys[gy], ys[gy + 1]
            if r <= l or b <= t:
                continue
            blocks.append((gx, gy, img.crop((l, t, r, b))))
    if len(blocks) <= 1:
        return img.copy()

    rng = random.Random(int(seed))
    order = list(range(len(blocks)))
    rng.shuffle(order)
    out = Image.new(img.mode, img.size)
    for dst_i, src_i in enumerate(order):
        gx, gy, patch = blocks[src_i]
        dx, dy, _ = blocks[dst_i]
        l, r = xs[dx], xs[dx + 1]
        t, b = ys[dy], ys[dy + 1]
        out.paste(patch.resize((max(1, r - l), max(1, b - t))), (l, t))
    return out


def make_control_image(img, mode: str, shuffle_grid: int, blur_radius: float, seed: int):
    mode = str(mode).strip().lower()
    if mode == "shuffle":
        return shuffle_image_blocks(img, grid=shuffle_grid, seed=seed)
    if mode == "blur":
        from PIL import ImageFilter

        return img.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    raise ValueError(f"Unknown control mode: {mode}")


def compute_token_sims_stats(
    h: torch.Tensor,
    vis: torch.Tensor,
    topk_local: int,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    # h: [T, D], vis: [P, D]
    h_n = F.normalize(h.float(), dim=-1)
    vis_n = F.normalize(vis.float(), dim=-1)
    sim_patch = torch.matmul(h_n, vis_n.t())  # [T, P]
    kk = int(min(max(1, topk_local), sim_patch.size(1)))
    sim_local_topk = torch.topk(sim_patch, k=kk, dim=-1).values.mean(dim=-1)
    sim_local_max = torch.max(sim_patch, dim=-1).values
    sim_local_mean = sim_patch.mean(dim=-1)
    sim_local_std = sim_patch.std(dim=-1, unbiased=False)
    denom = torch.clamp(sim_local_std, min=float(eps))
    z_local_max = (sim_local_max - sim_local_mean) / denom
    z_local_topk = (sim_local_topk - sim_local_mean) / denom
    if int(sim_patch.size(1)) >= 2:
        top2 = torch.topk(sim_patch, k=2, dim=-1).values
        gap_local = top2[:, 0] - top2[:, 1]
    else:
        gap_local = torch.zeros_like(sim_local_max)
    z_local_gap = gap_local / denom
    return {
        "sim_local_topk": sim_local_topk,
        "sim_local_max": sim_local_max,
        "sim_local_mean": sim_local_mean,
        "sim_local_std": sim_local_std,
        "sim_local_gap": gap_local,
        "z_local_topk": z_local_topk,
        "z_local_max": z_local_max,
        "z_local_gap": z_local_gap,
    }


def compute_token_sims(h: torch.Tensor, vis: torch.Tensor, topk_local: int) -> Tuple[torch.Tensor, torch.Tensor]:
    stats = compute_token_sims_stats(h=h, vis=vis, topk_local=topk_local)
    return stats["sim_local_topk"], stats["sim_local_max"]


def encode_text_mean_embedding(model, tokenizer, text: str) -> Optional[torch.Tensor]:
    s = str(text or "").strip()
    if s == "":
        return None
    ids = choose_cont_ids(tokenizer, s)
    if len(ids) == 0:
        ids = [int(x) for x in tokenizer(s, add_special_tokens=False).input_ids]
    if len(ids) == 0:
        return None
    dev = model.get_model().embed_tokens.weight.device
    ids_t = torch.tensor(ids, dtype=torch.long, device=dev)
    with torch.no_grad():
        emb = model.get_model().embed_tokens(ids_t).float()  # [T, D]
    if emb.ndim != 2 or emb.size(0) == 0:
        return None
    return emb.mean(dim=0)


def select_object_patch_indices(
    vis: torch.Tensor,
    obj_text_emb: Optional[torch.Tensor],
    object_patch_topk: int,
) -> Tuple[torch.Tensor, Dict[str, Optional[float]]]:
    # vis: [P, D]
    p = int(vis.size(0))
    if p <= 0:
        return torch.empty((0,), dtype=torch.long), {"objpatch_rel_mean": None, "objpatch_rel_median": None, "objpatch_rel_max": None}
    if obj_text_emb is None or int(object_patch_topk) <= 0:
        idx = torch.arange(p, dtype=torch.long)
        return idx, {"objpatch_rel_mean": None, "objpatch_rel_median": None, "objpatch_rel_max": None}

    vis_n = F.normalize(vis.float(), dim=-1)  # [P, D]
    obj_n = F.normalize(obj_text_emb.float().to(vis_n.device), dim=-1)  # [D]
    rel = torch.matmul(vis_n, obj_n)  # [P]
    k = int(min(max(1, int(object_patch_topk)), p))
    vals, idx = torch.topk(rel, k=k, dim=-1)
    stats = {
        "objpatch_rel_mean": float(vals.mean().item()),
        "objpatch_rel_median": float(vals.median().item()),
        "objpatch_rel_max": float(vals.max().item()),
    }
    return idx.long().cpu(), stats


def find_cont_label_positions(labels_expanded: torch.Tensor, cont_ids: List[int], ignore_index: int) -> Optional[torch.Tensor]:
    # labels_expanded: [L]
    if labels_expanded.ndim != 1:
        return None
    tlen = int(len(cont_ids))
    if tlen <= 0:
        return None
    valid_pos = torch.where(labels_expanded != int(ignore_index))[0]
    if int(valid_pos.numel()) < tlen:
        return None

    # Fast path: continuation is always sequence tail in this setup.
    tail_pos = valid_pos[-tlen:]
    tail_ids = labels_expanded[tail_pos].tolist()
    if [int(x) for x in tail_ids] == [int(x) for x in cont_ids]:
        return tail_pos

    # Fallback: exact subsequence match over non-ignore labels.
    valid_ids = labels_expanded[valid_pos].tolist()
    target = [int(x) for x in cont_ids]
    for s in range(0, int(len(valid_ids) - tlen + 1)):
        if [int(x) for x in valid_ids[s : s + tlen]] == target:
            return valid_pos[s : s + tlen]
    return None


def fit_joint_pca_basis(vis: torch.Tensor, h: torch.Tensor, pca_dim: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    # vis: [P, D], h: [T, D]
    if vis.ndim != 2 or h.ndim != 2:
        return None, None, 0
    if vis.size(1) != h.size(1):
        return None, None, 0
    x = torch.cat([vis.float(), h.float()], dim=0)
    if x.size(0) < 2:
        return None, None, 0
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    try:
        # xc = U S Vh ; principal directions are rows of Vh.
        _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    except Exception:
        return None, None, 0
    k = int(min(max(1, int(pca_dim)), int(vh.size(0)), int(xc.size(1))))
    basis = vh[:k, :].t().contiguous()  # [D, k]
    return mu, basis, k


def compute_projected_token_sims(
    h: torch.Tensor,
    vis: torch.Tensor,
    topk_local: int,
    mu: torch.Tensor,
    basis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Project onto shared PCA subspace then compute token-vs-patch cosine.
    hp = torch.matmul(h.float() - mu, basis)
    vp = torch.matmul(vis.float() - mu, basis)
    return compute_token_sims(h=hp, vis=vp, topk_local=topk_local)


def compute_attention_vision_probes(
    attentions: Sequence[torch.Tensor],
    layer_idx: int,
    decision_positions: torch.Tensor,
    vision_positions: torch.Tensor,
    text_positions: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Returns per-token (vision-sum, vision-ratio) averaged over heads.
    if attentions is None or len(attentions) == 0:
        return None, None
    if not (-len(attentions) <= int(layer_idx) < len(attentions)):
        return None, None
    att = attentions[int(layer_idx)]
    if att is None or att.ndim != 4 or int(att.size(0)) != 1:
        return None, None
    # [H, Lq, Lk]
    att = att[0].float()
    if att.ndim != 3:
        return None, None
    if decision_positions.numel() == 0 or vision_positions.numel() == 0:
        return None, None

    vis_vals: List[float] = []
    ratio_vals: List[float] = []
    eps = 1e-12
    for dp in decision_positions.tolist():
        d = int(dp)
        if d < 0 or d >= int(att.size(1)):
            vis_vals.append(float("nan"))
            ratio_vals.append(float("nan"))
            continue
        row = att[:, d, :]  # [H, Lk]
        vis_sum_h = row[:, vision_positions].sum(dim=-1)  # [H]
        if text_positions.numel() > 0:
            txt_sum_h = row[:, text_positions].sum(dim=-1)
        else:
            txt_sum_h = torch.zeros_like(vis_sum_h)
        den = vis_sum_h + txt_sum_h + eps
        vis_vals.append(float(vis_sum_h.mean().item()))
        ratio_vals.append(float((vis_sum_h / den).mean().item()))

    return torch.tensor(vis_vals, dtype=torch.float32), torch.tensor(ratio_vals, dtype=torch.float32)


def compute_attention_head_probes(
    att_l: torch.Tensor,
    decision_pos: int,
    vision_positions: torch.Tensor,
    text_positions: torch.Tensor,
    eps: float = 1e-12,
) -> Optional[Dict[str, torch.Tensor]]:
    # att_l: [1, H, Lq, Lk] or [H, Lq, Lk]
    if att_l is None:
        return None
    if att_l.ndim == 4:
        if int(att_l.size(0)) != 1:
            return None
        att = att_l[0].float()
    elif att_l.ndim == 3:
        att = att_l.float()
    else:
        return None
    if decision_pos < 0 or decision_pos >= int(att.size(1)):
        return None
    if vision_positions.numel() == 0:
        return None

    row = att[:, int(decision_pos), :]  # [H, Lk]
    vis_block = row[:, vision_positions]  # [H, Pv]
    vis_sum_h = vis_block.sum(dim=-1)
    if text_positions.numel() > 0:
        txt_sum_h = row[:, text_positions].sum(dim=-1)
    else:
        txt_sum_h = torch.zeros_like(vis_sum_h)
    den = vis_sum_h + txt_sum_h + eps
    vis_ratio_h = vis_sum_h / den
    vis_peak_h = torch.max(vis_block, dim=-1).values
    vis_probs = vis_block / torch.clamp(vis_sum_h.unsqueeze(-1), min=eps)
    vis_entropy_h = -(vis_probs * torch.log(torch.clamp(vis_probs, min=eps))).sum(dim=-1)
    if int(vis_block.size(-1)) > 1:
        vis_entropy_h = vis_entropy_h / math.log(float(int(vis_block.size(-1))))
    else:
        vis_entropy_h = torch.zeros_like(vis_entropy_h)
    return {
        "head_attn_vis_sum": vis_sum_h.detach().cpu(),
        "head_attn_vis_ratio": vis_ratio_h.detach().cpu(),
        "head_attn_vis_peak": vis_peak_h.detach().cpu(),
        "head_attn_vis_entropy": vis_entropy_h.detach().cpu(),
    }


def summarize_metric(
    name: str,
    labels: List[int],
    scores: List[float],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    auc = auc_from_scores(labels, scores)
    ks = ks_from_scores(labels, scores)
    auc_lo, auc_hi = bootstrap_ci(auc_from_scores, labels, scores, int(bootstrap), int(seed) + 11)
    ks_lo, ks_hi = bootstrap_ci(ks_from_scores, labels, scores, int(bootstrap), int(seed) + 29)
    return {
        "metric": name,
        "n": int(len(scores)),
        "auc_hall_high": auc,
        "auc_best_dir": (None if auc is None else float(max(float(auc), 1.0 - float(auc)))),
        "direction": (None if auc is None else ("higher_in_hallucination" if float(auc) >= 0.5 else "lower_in_hallucination")),
        "auc_ci95_lo": auc_lo,
        "auc_ci95_hi": auc_hi,
        "ks_hall_high": ks,
        "ks_ci95_lo": ks_lo,
        "ks_ci95_hi": ks_hi,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Visual disconnect probe at answer-anchor token (POPE/GQA).")
    ap.add_argument("--samples_csv", type=str, required=True, help="per_sample.csv input (POPE or GQA-like).")
    ap.add_argument("--image_root", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dataset_mode", type=str, default="pope", choices=["pope", "gqa"])
    ap.add_argument("--gqa_success_col", type=str, default="is_success", help="Used only when --dataset_mode gqa.")
    ap.add_argument(
        "--gqa_anchor_field",
        type=str,
        default="pred_answer_eval",
        help="Preferred field for anchor phrase in GQA mode (fallbacks: pred_answer_eval/champ_short_answer/answer/pred_text).",
    )

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--topk_local", type=int, default=16)
    ap.add_argument(
        "--object_patch_topk",
        type=int,
        default=64,
        help="Top-K object-relevant visual patches selected from real image (0 disables object-conditioned patch probe).",
    )
    ap.add_argument("--pca_dim", type=int, default=64, help="PCA dimension for projected similarity probe.")
    ap.add_argument("--disable_pca_probe", action="store_true", help="Disable PCA-projected similarity probe.")
    ap.add_argument("--disable_attention_probe", action="store_true", help="Disable attention vision-sum/ratio probe.")
    ap.add_argument(
        "--hidden_layer_idx",
        type=int,
        default=-1,
        help="Hidden layer index from output_hidden_states (e.g., -1 final, -2 penultimate).",
    )
    ap.add_argument(
        "--attn_layer_idx",
        type=int,
        default=-1,
        help="Attention layer index from output_attentions (e.g., -1 final attention layer).",
    )
    ap.add_argument(
        "--save_layer_trace",
        action="store_true",
        help="Save per-layer yes-token trace (raw cosine + attention vision probes) across all transformer blocks.",
    )
    ap.add_argument(
        "--save_head_trace",
        action="store_true",
        help="Save per-head attention trace for yes-token decision state across selected layers.",
    )
    ap.add_argument("--head_layer_start", type=int, default=10, help="Start block layer index for head trace export.")
    ap.add_argument("--head_layer_end", type=int, default=24, help="End block layer index for head trace export.")
    ap.add_argument("--control_modes", type=str, default="blur,shuffle", help="comma-separated: blur,shuffle (or empty/none for real-only)")
    ap.add_argument("--shuffle_grid", type=int, default=4)
    ap.add_argument("--blur_radius", type=float, default=12.0)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--min_pred_tokens", type=int, default=1)
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    modes_raw = [m.strip().lower() for m in str(args.control_modes).split(",") if m.strip()]
    if any(m in {"none", "off", "real_only", "real"} for m in modes_raw):
        modes = []
    else:
        modes = [m for m in modes_raw if m in {"blur", "shuffle"}]
        if len(modes_raw) > 0 and len(modes) == 0:
            raise RuntimeError("No valid control modes. Use blur and/or shuffle (or none).")

    rows_in: List[Dict[str, Any]] = []
    with open(os.path.abspath(args.samples_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows_in.append(r)
    if int(args.num_samples) > 0:
        rows_in = rows_in[: int(args.num_samples)]
    if len(rows_in) == 0:
        raise RuntimeError("No rows in samples CSV.")

    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IGNORE_INDEX,
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

    per_sample_rows: List[Dict[str, Any]] = []
    per_layer_rows: List[Dict[str, Any]] = []
    per_head_rows: List[Dict[str, Any]] = []
    skipped = 0
    logged_sample_errors = 0

    pbar = tqdm(rows_in, total=len(rows_in), desc=f"{str(args.dataset_mode)}-visual-disconnect", dynamic_ncols=True)
    for rr in pbar:
        sid = str(rr.get("id") or "")
        question = str(rr.get("question") or "")
        answer = str(rr.get("answer") or "")
        image_id = str(rr.get("image_id") or rr.get("imageId") or "")
        pred_answer_eval = str(rr.get("pred_answer_eval") or "").strip()
        pred_text = str(rr.get("pred_text") or rr.get("champ_text") or pred_answer_eval).strip()
        if sid == "" or question == "" or answer == "" or image_id == "" or pred_text == "":
            skipped += 1
            continue

        image_path = resolve_image_path(args.image_root, image_id)
        if image_path is None:
            skipped += 1
            continue

        anchor_phrase = ""
        if str(args.dataset_mode) == "pope":
            gt = normalize_yesno(answer)
            pred = normalize_yesno(pred_answer_eval if pred_answer_eval != "" else pred_text)
            if gt not in {"yes", "no"} or pred not in {"yes", "no"}:
                skipped += 1
                continue
            is_fp_hall = bool(pred == "yes" and gt == "no")
            is_tp_yes = bool(pred == "yes" and gt == "yes")
            is_tn_no = bool(pred == "no" and gt == "no")
            is_fn_miss = bool(pred == "no" and gt == "yes")
            is_correct = bool(pred == gt)
            anchor_phrase = str(pred or "").strip()
            obj_phrase = extract_pope_object(question)
        else:
            is_correct = bool(parse_bool(rr.get(args.gqa_success_col)))
            is_fp_hall = bool(not is_correct)  # alias: failure
            is_tp_yes = bool(is_correct)       # alias: success
            is_tn_no = False
            is_fn_miss = False
            gt = str(answer or "").strip().lower()
            pred = str(pred_answer_eval if pred_answer_eval != "" else pred_text).strip().lower()
            anchor_phrase = pick_gqa_anchor_phrase(rr, preferred_field=str(args.gqa_anchor_field), pred_text=pred_text, answer=answer)
            obj_phrase = str(anchor_phrase or "").strip()

        cont_ids = choose_cont_ids(tokenizer, pred_text)
        tlen = int(len(cont_ids))
        if tlen < int(args.min_pred_tokens):
            skipped += 1
            continue

        yesno_idx = locate_phrase_start(tokenizer, cont_ids, anchor_phrase)
        if yesno_idx is None:
            yesno_idx = 0
        yesno_idx = int(max(0, min(tlen - 1, int(yesno_idx))))
        obj_idx = locate_phrase_start(tokenizer, cont_ids, obj_phrase)

        try:
            img_prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            prompt_ids = tokenizer_image_token(
                img_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            cont_t = torch.tensor([cont_ids], dtype=torch.long, device=device)
            full_ids = torch.cat([prompt_ids, cont_t], dim=1)

            img = Image.open(image_path).convert("RGB")

            mode_pack: Dict[str, Dict[str, torch.Tensor]] = {}
            real_trace_cache: Dict[str, Any] = {}
            use_attn_probe = not bool(args.disable_attention_probe)
            for mode in ["real"] + modes:
                if mode == "real":
                    im = img
                else:
                    im = make_control_image(
                        img=img,
                        mode=mode,
                        shuffle_grid=int(args.shuffle_grid),
                        blur_radius=float(args.blur_radius),
                        seed=(int(args.seed) + abs(hash((sid, mode))) % 100000),
                    )
                images_tensor = process_images([im], image_processor, model.config).to(
                    device=model.device,
                    dtype=torch.float16,
                )
                image_sizes = [im.size]
                with torch.no_grad():
                    base_attn = torch.ones_like(full_ids, dtype=torch.long, device=device)
                    _, pos_ids_e, attn_mask_e, _, mm_embeds_e, labels_e = model.prepare_inputs_labels_for_multimodal(
                        full_ids,
                        None,
                        base_attn,
                        None,
                        full_ids,
                        images_tensor,
                        image_sizes,
                    )
                    if mm_embeds_e is None or labels_e is None:
                        raise RuntimeError("failed to build multimodal expanded sequence")

                    labels_exp = labels_e[0]
                    cont_label_pos = find_cont_label_positions(labels_exp, cont_ids, int(IGNORE_INDEX))
                    if cont_label_pos is None or int(cont_label_pos.numel()) != tlen:
                        raise RuntimeError("failed to align continuation labels on expanded sequence")
                    dec_pos = cont_label_pos - 1
                    if int(dec_pos.min().item()) < 0:
                        raise RuntimeError("invalid decision positions")
                    vision_pos = torch.where(labels_exp == int(IGNORE_INDEX))[0]
                    text_pos = torch.where(labels_exp != int(IGNORE_INDEX))[0]
                    if int(vision_pos.numel()) < 2:
                        raise RuntimeError("no visual positions found")

                    out = model(
                        inputs_embeds=mm_embeds_e,
                        attention_mask=attn_mask_e,
                        position_ids=pos_ids_e,
                        use_cache=False,
                        output_attentions=bool(use_attn_probe),
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hs_all = out.hidden_states
                    layer_idx = int(args.hidden_layer_idx)
                    if not (-len(hs_all) <= layer_idx < len(hs_all)):
                        raise RuntimeError(f"hidden_layer_idx out of range: {layer_idx} (n={len(hs_all)})")
                    hs = hs_all[layer_idx][0]
                    h_dec = hs[dec_pos, :].float()
                    vis = mm_embeds_e[0, vision_pos, :].float()
                    if h_dec.size(0) != tlen or vis.ndim != 2 or vis.size(0) < 2:
                        raise RuntimeError("bad hidden/visual shapes")
                    sim_stats = compute_token_sims_stats(
                        h=h_dec, vis=vis, topk_local=int(args.topk_local)
                    )
                    attn_vis_sum = None
                    attn_vis_ratio = None
                    if use_attn_probe:
                        attn_vis_sum, attn_vis_ratio = compute_attention_vision_probes(
                            attentions=out.attentions,
                            layer_idx=int(args.attn_layer_idx),
                            decision_positions=dec_pos,
                            vision_positions=vision_pos,
                            text_positions=text_pos,
                        )
                mode_pack[mode] = {
                    "sim_local_topk": sim_stats["sim_local_topk"].cpu(),
                    "sim_local_max": sim_stats["sim_local_max"].cpu(),
                    "sim_local_mean": sim_stats["sim_local_mean"].cpu(),
                    "sim_local_std": sim_stats["sim_local_std"].cpu(),
                    "sim_local_gap": sim_stats["sim_local_gap"].cpu(),
                    "z_local_topk": sim_stats["z_local_topk"].cpu(),
                    "z_local_max": sim_stats["z_local_max"].cpu(),
                    "z_local_gap": sim_stats["z_local_gap"].cpu(),
                    "h_dec": h_dec.cpu(),
                    "vis": vis.cpu(),
                    "attn_vis_sum": (None if attn_vis_sum is None else attn_vis_sum.cpu()),
                    "attn_vis_ratio": (None if attn_vis_ratio is None else attn_vis_ratio.cpu()),
                }
                if mode == "real":
                    real_trace_cache = {
                        "hs_all": hs_all,
                        "attentions": out.attentions,
                        "dec_pos": dec_pos.detach().clone(),
                        "vision_pos": vision_pos.detach().clone(),
                        "text_pos": text_pos.detach().clone(),
                    }

            # Object-conditioned patch subset from REAL image only.
            obj_emb = encode_text_mean_embedding(model=model, tokenizer=tokenizer, text=obj_phrase)
            obj_patch_idx, obj_patch_stats = select_object_patch_indices(
                vis=mode_pack["real"]["vis"],
                obj_text_emb=obj_emb,
                object_patch_topk=int(args.object_patch_topk),
            )
            if int(obj_patch_idx.numel()) <= 0:
                obj_patch_idx = torch.arange(int(mode_pack["real"]["vis"].size(0)), dtype=torch.long)

            for mode in ["real"] + modes:
                vis_mode = mode_pack[mode]["vis"]
                p_idx = obj_patch_idx
                if int(p_idx.numel()) > int(vis_mode.size(0)):
                    p_idx = torch.arange(int(vis_mode.size(0)), dtype=torch.long)
                vis_obj = vis_mode[p_idx, :]
                sim_obj_topk, sim_obj_max = compute_token_sims(
                    h=mode_pack[mode]["h_dec"],
                    vis=vis_obj,
                    topk_local=int(args.topk_local),
                )
                mode_pack[mode]["sim_objpatch_topk"] = sim_obj_topk.cpu()
                mode_pack[mode]["sim_objpatch_max"] = sim_obj_max.cpu()

            if bool(args.save_layer_trace):
                hs_all = real_trace_cache.get("hs_all")
                atts = real_trace_cache.get("attentions")
                dec_pos = real_trace_cache.get("dec_pos")
                vision_pos = real_trace_cache.get("vision_pos")
                text_pos = real_trace_cache.get("text_pos")
                if hs_all is not None and dec_pos is not None and vision_pos is not None and text_pos is not None:
                    # hidden_states includes embedding output at index 0, so block l -> hidden_states[l+1].
                    n_blocks_hidden = max(0, int(len(hs_all) - 1))
                    vis_f = mode_pack["real"]["vis"].float()
                    vis_n = F.normalize(vis_f, dim=-1)
                    vis_obj_f = vis_f[obj_patch_idx, :]
                    vis_obj_n = F.normalize(vis_obj_f, dim=-1)
                    kk_layer = int(min(max(1, int(args.topk_local)), int(vis_n.size(0))))
                    kk_obj_layer = int(min(max(1, int(args.topk_local)), int(vis_obj_n.size(0))))
                    yes_dec_pos = int(dec_pos[yesno_idx].item())
                    eps = 1e-12

                    for block_idx in range(n_blocks_hidden):
                        hs_block = hs_all[block_idx + 1][0]  # [L, D]
                        if yes_dec_pos < 0 or yes_dec_pos >= int(hs_block.size(0)):
                            continue

                        y = hs_block[yes_dec_pos, :].float()  # [D]
                        y_n = F.normalize(y.to(vis_n.device), dim=-1)
                        sim_patch = torch.matmul(vis_n, y_n)  # [P]
                        topk_pack = torch.topk(sim_patch, k=kk_layer, dim=-1)
                        topk_vals = topk_pack.values
                        topk_idx = topk_pack.indices.long()
                        yes_sim_local_topk = float(topk_vals.mean().item())
                        yes_sim_local_max = float(torch.max(sim_patch).item())
                        yes_sim_local_argmax_idx = int(topk_idx[0].item())
                        yes_sim_local_topk_idx_json = json.dumps(
                            [int(v) for v in topk_idx.tolist()],
                            ensure_ascii=False,
                        )
                        yes_sim_local_topk_weight_json = json.dumps(
                            [float(v) for v in topk_vals.tolist()],
                            ensure_ascii=False,
                        )
                        yes_sim_local_mean = float(sim_patch.mean().item())
                        yes_sim_local_std = float(sim_patch.std(unbiased=False).item())
                        den_local = max(1e-6, yes_sim_local_std)
                        yes_z_local_topk = float((yes_sim_local_topk - yes_sim_local_mean) / den_local)
                        yes_z_local_max = float((yes_sim_local_max - yes_sim_local_mean) / den_local)
                        if int(sim_patch.numel()) >= 2:
                            top2_patch = torch.topk(sim_patch, k=2, dim=-1).values
                            yes_sim_local_gap = float((top2_patch[0] - top2_patch[1]).item())
                        else:
                            yes_sim_local_gap = 0.0
                        yes_z_local_gap = float(yes_sim_local_gap / den_local)

                        sim_obj_patch = torch.matmul(vis_obj_n, y_n)  # [Kobj]
                        topk_obj_pack = torch.topk(sim_obj_patch, k=kk_obj_layer, dim=-1)
                        topk_obj_vals = topk_obj_pack.values
                        topk_obj_idx_local = topk_obj_pack.indices.long()
                        topk_obj_idx_global = obj_patch_idx[topk_obj_idx_local].long()
                        yes_sim_objpatch_topk = float(topk_obj_vals.mean().item())
                        yes_sim_objpatch_max = float(torch.max(sim_obj_patch).item())
                        yes_sim_objpatch_argmax_idx_global = int(topk_obj_idx_global[0].item())
                        yes_sim_objpatch_topk_idx_global_json = json.dumps(
                            [int(v) for v in topk_obj_idx_global.tolist()],
                            ensure_ascii=False,
                        )
                        yes_sim_objpatch_topk_weight_json = json.dumps(
                            [float(v) for v in topk_obj_vals.tolist()],
                            ensure_ascii=False,
                        )

                        yes_attn_vis_sum = None
                        yes_attn_vis_ratio = None
                        yes_attn_vis_topk_idx_json = None
                        yes_attn_vis_argmax_idx = None
                        if use_attn_probe and atts is not None and block_idx < len(atts):
                            att_l = atts[block_idx]
                            if att_l is not None and att_l.ndim == 4 and int(att_l.size(0)) == 1:
                                row_att = att_l[0, :, yes_dec_pos, :].float()  # [H, Lk]
                                vis_sum_h = row_att[:, vision_pos].sum(dim=-1)  # [H]
                                if text_pos.numel() > 0:
                                    txt_sum_h = row_att[:, text_pos].sum(dim=-1)
                                else:
                                    txt_sum_h = torch.zeros_like(vis_sum_h)
                                den = vis_sum_h + txt_sum_h + eps
                                yes_attn_vis_sum = float(vis_sum_h.mean().item())
                                yes_attn_vis_ratio = float((vis_sum_h / den).mean().item())
                                # Mean over heads -> patch-level visual attention profile.
                                vis_attn_patch = row_att[:, vision_pos].mean(dim=0)  # [P]
                                kk_att = int(min(max(1, int(args.topk_local)), int(vis_attn_patch.numel())))
                                att_topk_pack = torch.topk(vis_attn_patch, k=kk_att, dim=-1)
                                att_topk_idx = att_topk_pack.indices.long()
                                yes_attn_vis_argmax_idx = int(att_topk_idx[0].item())
                                yes_attn_vis_topk_idx_json = json.dumps(
                                    [int(v) for v in att_topk_idx.tolist()],
                                    ensure_ascii=False,
                                )
                                yes_attn_vis_topk_weight_json = json.dumps(
                                    [float(v) for v in att_topk_pack.values.tolist()],
                                    ensure_ascii=False,
                                )
                            else:
                                yes_attn_vis_topk_weight_json = None
                        else:
                            yes_attn_vis_topk_weight_json = None

                        per_layer_rows.append(
                            {
                                "id": sid,
                                "image_id": image_id,
                                "question": question,
                                "answer_gt": gt,
                                "answer_pred": pred,
                                "pred_text": pred_text,
                                "dataset_mode": str(args.dataset_mode),
                                "anchor_phrase": str(anchor_phrase),
                                "object_phrase": obj_phrase,
                                "is_correct": bool(is_correct),
                                "is_fp_hallucination": bool(is_fp_hall),
                                "is_tp_yes": bool(is_tp_yes),
                                "is_tn_no": bool(is_tn_no),
                                "is_fn_miss": bool(is_fn_miss),
                                "yesno_token_idx": int(yesno_idx),
                                "yesno_token_str": str(tokenizer.convert_ids_to_tokens(int(cont_ids[yesno_idx]))),
                                "n_tokens": int(tlen),
                                "n_visual_tokens": int(vis_f.size(0)),
                                "n_object_patches": int(obj_patch_idx.numel()),
                                "hidden_state_idx": int(block_idx + 1),
                                "block_layer_idx": int(block_idx),
                                "yes_sim_local_topk": yes_sim_local_topk,
                                "yes_sim_local_max": yes_sim_local_max,
                                "yes_sim_local_mean": yes_sim_local_mean,
                                "yes_sim_local_std": yes_sim_local_std,
                                "yes_sim_local_gap": yes_sim_local_gap,
                                "yes_z_local_topk": yes_z_local_topk,
                                "yes_z_local_max": yes_z_local_max,
                                "yes_z_local_gap": yes_z_local_gap,
                                "yes_sim_local_argmax_idx": int(yes_sim_local_argmax_idx),
                                "yes_sim_local_topk_idx_json": yes_sim_local_topk_idx_json,
                                "yes_sim_local_topk_weight_json": yes_sim_local_topk_weight_json,
                                "yes_sim_objpatch_topk": yes_sim_objpatch_topk,
                                "yes_sim_objpatch_max": yes_sim_objpatch_max,
                                "yes_sim_objpatch_argmax_idx_global": int(yes_sim_objpatch_argmax_idx_global),
                                "yes_sim_objpatch_topk_idx_global_json": yes_sim_objpatch_topk_idx_global_json,
                                "yes_sim_objpatch_topk_weight_json": yes_sim_objpatch_topk_weight_json,
                                "yes_attn_vis_sum": yes_attn_vis_sum,
                                "yes_attn_vis_ratio": yes_attn_vis_ratio,
                                "yes_attn_vis_argmax_idx": yes_attn_vis_argmax_idx,
                                "yes_attn_vis_topk_idx_json": yes_attn_vis_topk_idx_json,
                                "yes_attn_vis_topk_weight_json": yes_attn_vis_topk_weight_json,
                            }
                        )

            if bool(args.save_head_trace):
                atts = real_trace_cache.get("attentions")
                dec_pos = real_trace_cache.get("dec_pos")
                vision_pos = real_trace_cache.get("vision_pos")
                text_pos = real_trace_cache.get("text_pos")
                if atts is not None and dec_pos is not None and vision_pos is not None and text_pos is not None:
                    yes_dec_pos = int(dec_pos[yesno_idx].item())
                    head_l0 = int(min(args.head_layer_start, args.head_layer_end))
                    head_l1 = int(max(args.head_layer_start, args.head_layer_end))
                    for block_idx in range(head_l0, min(head_l1 + 1, len(atts))):
                        att_l = atts[block_idx]
                        probes = compute_attention_head_probes(
                            att_l=att_l,
                            decision_pos=yes_dec_pos,
                            vision_positions=vision_pos,
                            text_positions=text_pos,
                        )
                        if probes is None:
                            continue
                        n_heads = int(probes["head_attn_vis_sum"].numel())
                        for head_idx in range(n_heads):
                            per_head_rows.append(
                                {
                                    "id": sid,
                                    "image_id": image_id,
                                    "question": question,
                                    "answer_gt": gt,
                                    "answer_pred": pred,
                                    "pred_text": pred_text,
                                    "dataset_mode": str(args.dataset_mode),
                                    "anchor_phrase": str(anchor_phrase),
                                    "object_phrase": obj_phrase,
                                    "is_correct": bool(is_correct),
                                    "is_fp_hallucination": bool(is_fp_hall),
                                    "is_tp_yes": bool(is_tp_yes),
                                    "is_tn_no": bool(is_tn_no),
                                    "is_fn_miss": bool(is_fn_miss),
                                    "yesno_token_idx": int(yesno_idx),
                                    "yesno_token_str": str(tokenizer.convert_ids_to_tokens(int(cont_ids[yesno_idx]))),
                                    "block_layer_idx": int(block_idx),
                                    "head_idx": int(head_idx),
                                    "head_attn_vis_sum": float(probes["head_attn_vis_sum"][head_idx].item()),
                                    "head_attn_vis_ratio": float(probes["head_attn_vis_ratio"][head_idx].item()),
                                    "head_attn_vis_peak": float(probes["head_attn_vis_peak"][head_idx].item()),
                                    "head_attn_vis_entropy": float(probes["head_attn_vis_entropy"][head_idx].item()),
                                }
                            )

            use_pca_probe = not bool(args.disable_pca_probe)
            pca_k = 0
            if use_pca_probe:
                mu, basis, pca_k = fit_joint_pca_basis(
                    vis=mode_pack["real"]["vis"],
                    h=mode_pack["real"]["h_dec"],
                    pca_dim=int(args.pca_dim),
                )
                if mu is not None and basis is not None and int(pca_k) > 0:
                    for mode in ["real"] + modes:
                        sim_topk_pca, sim_max_pca = compute_projected_token_sims(
                            h=mode_pack[mode]["h_dec"],
                            vis=mode_pack[mode]["vis"],
                            topk_local=int(args.topk_local),
                            mu=mu,
                            basis=basis,
                        )
                        mode_pack[mode]["sim_local_topk_pca"] = sim_topk_pca.cpu()
                        mode_pack[mode]["sim_local_max_pca"] = sim_max_pca.cpu()

            row: Dict[str, Any] = {
                "id": sid,
                "image_id": image_id,
                "question": question,
                "answer_gt": gt,
                "answer_pred": pred,
                "pred_text": pred_text,
                "dataset_mode": str(args.dataset_mode),
                "anchor_phrase": str(anchor_phrase),
                "is_correct": bool(is_correct),
                "is_fp_hallucination": bool(is_fp_hall),
                "is_tp_yes": bool(is_tp_yes),
                "is_tn_no": bool(is_tn_no),
                "is_fn_miss": bool(is_fn_miss),
                "n_tokens": int(tlen),
                "object_phrase": obj_phrase,
                "yesno_token_idx": int(yesno_idx),
                "yesno_token_str": str(tokenizer.convert_ids_to_tokens(int(cont_ids[yesno_idx]))),
                "object_token_idx": (None if obj_idx is None else int(obj_idx)),
                "object_token_str": (
                    None if obj_idx is None else str(tokenizer.convert_ids_to_tokens(int(cont_ids[int(obj_idx)])))
                ),
                "controls": ",".join(modes),
                "n_visual_tokens": int(mode_pack["real"]["vis"].size(0)),
                "n_object_patches": int(obj_patch_idx.numel()),
                "objpatch_rel_mean": obj_patch_stats.get("objpatch_rel_mean"),
                "objpatch_rel_median": obj_patch_stats.get("objpatch_rel_median"),
                "objpatch_rel_max": obj_patch_stats.get("objpatch_rel_max"),
                "pca_dim_effective": int(pca_k) if int(pca_k) > 0 else None,
            }

            # Real anchors
            row["yes_sim_local_topk_real"] = float(mode_pack["real"]["sim_local_topk"][yesno_idx].item())
            row["yes_sim_local_max_real"] = float(mode_pack["real"]["sim_local_max"][yesno_idx].item())
            row["yes_sim_local_mean_real"] = float(mode_pack["real"]["sim_local_mean"][yesno_idx].item())
            row["yes_sim_local_std_real"] = float(mode_pack["real"]["sim_local_std"][yesno_idx].item())
            row["yes_sim_local_gap_real"] = float(mode_pack["real"]["sim_local_gap"][yesno_idx].item())
            row["yes_z_local_topk_real"] = float(mode_pack["real"]["z_local_topk"][yesno_idx].item())
            row["yes_z_local_max_real"] = float(mode_pack["real"]["z_local_max"][yesno_idx].item())
            row["yes_z_local_gap_real"] = float(mode_pack["real"]["z_local_gap"][yesno_idx].item())
            row["yes_sim_objpatch_topk_real"] = float(mode_pack["real"]["sim_objpatch_topk"][yesno_idx].item())
            row["yes_sim_objpatch_max_real"] = float(mode_pack["real"]["sim_objpatch_max"][yesno_idx].item())
            if "sim_local_topk_pca" in mode_pack["real"]:
                row["yes_sim_local_topk_pca_real"] = float(mode_pack["real"]["sim_local_topk_pca"][yesno_idx].item())
                row["yes_sim_local_max_pca_real"] = float(mode_pack["real"]["sim_local_max_pca"][yesno_idx].item())
            else:
                row["yes_sim_local_topk_pca_real"] = None
                row["yes_sim_local_max_pca_real"] = None
            if mode_pack["real"]["attn_vis_sum"] is not None:
                row["yes_attn_vis_sum_real"] = float(mode_pack["real"]["attn_vis_sum"][yesno_idx].item())
                row["yes_attn_vis_ratio_real"] = float(mode_pack["real"]["attn_vis_ratio"][yesno_idx].item())
            else:
                row["yes_attn_vis_sum_real"] = None
                row["yes_attn_vis_ratio_real"] = None
            if obj_idx is not None and 0 <= int(obj_idx) < tlen:
                oi = int(obj_idx)
                row["obj_sim_local_topk_real"] = float(mode_pack["real"]["sim_local_topk"][oi].item())
                row["obj_sim_local_max_real"] = float(mode_pack["real"]["sim_local_max"][oi].item())
                row["obj_sim_local_mean_real"] = float(mode_pack["real"]["sim_local_mean"][oi].item())
                row["obj_sim_local_std_real"] = float(mode_pack["real"]["sim_local_std"][oi].item())
                row["obj_sim_local_gap_real"] = float(mode_pack["real"]["sim_local_gap"][oi].item())
                row["obj_z_local_topk_real"] = float(mode_pack["real"]["z_local_topk"][oi].item())
                row["obj_z_local_max_real"] = float(mode_pack["real"]["z_local_max"][oi].item())
                row["obj_z_local_gap_real"] = float(mode_pack["real"]["z_local_gap"][oi].item())
                if "sim_local_topk_pca" in mode_pack["real"]:
                    row["obj_sim_local_topk_pca_real"] = float(mode_pack["real"]["sim_local_topk_pca"][oi].item())
                    row["obj_sim_local_max_pca_real"] = float(mode_pack["real"]["sim_local_max_pca"][oi].item())
                else:
                    row["obj_sim_local_topk_pca_real"] = None
                    row["obj_sim_local_max_pca_real"] = None
                if mode_pack["real"]["attn_vis_sum"] is not None:
                    row["obj_attn_vis_sum_real"] = float(mode_pack["real"]["attn_vis_sum"][oi].item())
                    row["obj_attn_vis_ratio_real"] = float(mode_pack["real"]["attn_vis_ratio"][oi].item())
                else:
                    row["obj_attn_vis_sum_real"] = None
                    row["obj_attn_vis_ratio_real"] = None
            else:
                row["obj_sim_local_topk_real"] = None
                row["obj_sim_local_max_real"] = None
                row["obj_sim_local_mean_real"] = None
                row["obj_sim_local_std_real"] = None
                row["obj_sim_local_gap_real"] = None
                row["obj_z_local_topk_real"] = None
                row["obj_z_local_max_real"] = None
                row["obj_z_local_gap_real"] = None
                row["obj_sim_local_topk_pca_real"] = None
                row["obj_sim_local_max_pca_real"] = None
                row["obj_attn_vis_sum_real"] = None
                row["obj_attn_vis_ratio_real"] = None

            # Control deltas
            for mode in modes:
                row[f"yes_sim_local_topk_{mode}"] = float(mode_pack[mode]["sim_local_topk"][yesno_idx].item())
                row[f"yes_sim_local_max_{mode}"] = float(mode_pack[mode]["sim_local_max"][yesno_idx].item())
                row[f"yes_sim_local_mean_{mode}"] = float(mode_pack[mode]["sim_local_mean"][yesno_idx].item())
                row[f"yes_sim_local_std_{mode}"] = float(mode_pack[mode]["sim_local_std"][yesno_idx].item())
                row[f"yes_sim_local_gap_{mode}"] = float(mode_pack[mode]["sim_local_gap"][yesno_idx].item())
                row[f"yes_z_local_topk_{mode}"] = float(mode_pack[mode]["z_local_topk"][yesno_idx].item())
                row[f"yes_z_local_max_{mode}"] = float(mode_pack[mode]["z_local_max"][yesno_idx].item())
                row[f"yes_z_local_gap_{mode}"] = float(mode_pack[mode]["z_local_gap"][yesno_idx].item())
                row[f"yes_sim_objpatch_topk_{mode}"] = float(mode_pack[mode]["sim_objpatch_topk"][yesno_idx].item())
                row[f"yes_sim_objpatch_max_{mode}"] = float(mode_pack[mode]["sim_objpatch_max"][yesno_idx].item())
                row[f"yes_delta_topk_real_minus_{mode}"] = float(
                    row["yes_sim_local_topk_real"] - row[f"yes_sim_local_topk_{mode}"]
                )
                row[f"yes_delta_max_real_minus_{mode}"] = float(
                    row["yes_sim_local_max_real"] - row[f"yes_sim_local_max_{mode}"]
                )
                row[f"yes_delta_z_topk_real_minus_{mode}"] = float(
                    row["yes_z_local_topk_real"] - row[f"yes_z_local_topk_{mode}"]
                )
                row[f"yes_delta_z_max_real_minus_{mode}"] = float(
                    row["yes_z_local_max_real"] - row[f"yes_z_local_max_{mode}"]
                )
                row[f"yes_delta_z_gap_real_minus_{mode}"] = float(
                    row["yes_z_local_gap_real"] - row[f"yes_z_local_gap_{mode}"]
                )
                row[f"yes_delta_objpatch_topk_real_minus_{mode}"] = float(
                    row["yes_sim_objpatch_topk_real"] - row[f"yes_sim_objpatch_topk_{mode}"]
                )
                row[f"yes_delta_objpatch_max_real_minus_{mode}"] = float(
                    row["yes_sim_objpatch_max_real"] - row[f"yes_sim_objpatch_max_{mode}"]
                )
                if "sim_local_topk_pca" in mode_pack[mode] and row["yes_sim_local_topk_pca_real"] is not None:
                    row[f"yes_sim_local_topk_pca_{mode}"] = float(mode_pack[mode]["sim_local_topk_pca"][yesno_idx].item())
                    row[f"yes_sim_local_max_pca_{mode}"] = float(mode_pack[mode]["sim_local_max_pca"][yesno_idx].item())
                    row[f"yes_delta_topk_pca_real_minus_{mode}"] = float(
                        row["yes_sim_local_topk_pca_real"] - row[f"yes_sim_local_topk_pca_{mode}"]
                    )
                    row[f"yes_delta_max_pca_real_minus_{mode}"] = float(
                        row["yes_sim_local_max_pca_real"] - row[f"yes_sim_local_max_pca_{mode}"]
                    )
                else:
                    row[f"yes_sim_local_topk_pca_{mode}"] = None
                    row[f"yes_sim_local_max_pca_{mode}"] = None
                    row[f"yes_delta_topk_pca_real_minus_{mode}"] = None
                    row[f"yes_delta_max_pca_real_minus_{mode}"] = None
                if mode_pack[mode]["attn_vis_sum"] is not None and row["yes_attn_vis_sum_real"] is not None:
                    row[f"yes_attn_vis_sum_{mode}"] = float(mode_pack[mode]["attn_vis_sum"][yesno_idx].item())
                    row[f"yes_attn_vis_ratio_{mode}"] = float(mode_pack[mode]["attn_vis_ratio"][yesno_idx].item())
                    row[f"yes_delta_attn_vis_sum_real_minus_{mode}"] = float(
                        row["yes_attn_vis_sum_real"] - row[f"yes_attn_vis_sum_{mode}"]
                    )
                    row[f"yes_delta_attn_vis_ratio_real_minus_{mode}"] = float(
                        row["yes_attn_vis_ratio_real"] - row[f"yes_attn_vis_ratio_{mode}"]
                    )
                else:
                    row[f"yes_attn_vis_sum_{mode}"] = None
                    row[f"yes_attn_vis_ratio_{mode}"] = None
                    row[f"yes_delta_attn_vis_sum_real_minus_{mode}"] = None
                    row[f"yes_delta_attn_vis_ratio_real_minus_{mode}"] = None

                if obj_idx is not None and 0 <= int(obj_idx) < tlen:
                    oi = int(obj_idx)
                    obj_topk_mode = float(mode_pack[mode]["sim_local_topk"][oi].item())
                    obj_max_mode = float(mode_pack[mode]["sim_local_max"][oi].item())
                    obj_mean_mode = float(mode_pack[mode]["sim_local_mean"][oi].item())
                    obj_std_mode = float(mode_pack[mode]["sim_local_std"][oi].item())
                    obj_gap_mode = float(mode_pack[mode]["sim_local_gap"][oi].item())
                    obj_z_topk_mode = float(mode_pack[mode]["z_local_topk"][oi].item())
                    obj_z_max_mode = float(mode_pack[mode]["z_local_max"][oi].item())
                    obj_z_gap_mode = float(mode_pack[mode]["z_local_gap"][oi].item())
                    row[f"obj_sim_local_topk_{mode}"] = obj_topk_mode
                    row[f"obj_sim_local_max_{mode}"] = obj_max_mode
                    row[f"obj_sim_local_mean_{mode}"] = obj_mean_mode
                    row[f"obj_sim_local_std_{mode}"] = obj_std_mode
                    row[f"obj_sim_local_gap_{mode}"] = obj_gap_mode
                    row[f"obj_z_local_topk_{mode}"] = obj_z_topk_mode
                    row[f"obj_z_local_max_{mode}"] = obj_z_max_mode
                    row[f"obj_z_local_gap_{mode}"] = obj_z_gap_mode
                    row[f"obj_delta_topk_real_minus_{mode}"] = float(
                        row["obj_sim_local_topk_real"] - obj_topk_mode
                    )
                    row[f"obj_delta_max_real_minus_{mode}"] = float(
                        row["obj_sim_local_max_real"] - obj_max_mode
                    )
                    row[f"obj_delta_z_topk_real_minus_{mode}"] = float(
                        row["obj_z_local_topk_real"] - obj_z_topk_mode
                    )
                    row[f"obj_delta_z_max_real_minus_{mode}"] = float(
                        row["obj_z_local_max_real"] - obj_z_max_mode
                    )
                    row[f"obj_delta_z_gap_real_minus_{mode}"] = float(
                        row["obj_z_local_gap_real"] - obj_z_gap_mode
                    )
                    if "sim_local_topk_pca" in mode_pack[mode] and row["obj_sim_local_topk_pca_real"] is not None:
                        obj_topk_pca_mode = float(mode_pack[mode]["sim_local_topk_pca"][oi].item())
                        obj_max_pca_mode = float(mode_pack[mode]["sim_local_max_pca"][oi].item())
                        row[f"obj_sim_local_topk_pca_{mode}"] = obj_topk_pca_mode
                        row[f"obj_sim_local_max_pca_{mode}"] = obj_max_pca_mode
                        row[f"obj_delta_topk_pca_real_minus_{mode}"] = float(
                            row["obj_sim_local_topk_pca_real"] - obj_topk_pca_mode
                        )
                        row[f"obj_delta_max_pca_real_minus_{mode}"] = float(
                            row["obj_sim_local_max_pca_real"] - obj_max_pca_mode
                        )
                    else:
                        row[f"obj_sim_local_topk_pca_{mode}"] = None
                        row[f"obj_sim_local_max_pca_{mode}"] = None
                        row[f"obj_delta_topk_pca_real_minus_{mode}"] = None
                        row[f"obj_delta_max_pca_real_minus_{mode}"] = None

                    if mode_pack[mode]["attn_vis_sum"] is not None and row["obj_attn_vis_sum_real"] is not None:
                        obj_attn_sum_mode = float(mode_pack[mode]["attn_vis_sum"][oi].item())
                        obj_attn_ratio_mode = float(mode_pack[mode]["attn_vis_ratio"][oi].item())
                        row[f"obj_attn_vis_sum_{mode}"] = obj_attn_sum_mode
                        row[f"obj_attn_vis_ratio_{mode}"] = obj_attn_ratio_mode
                        row[f"obj_delta_attn_vis_sum_real_minus_{mode}"] = float(
                            row["obj_attn_vis_sum_real"] - obj_attn_sum_mode
                        )
                        row[f"obj_delta_attn_vis_ratio_real_minus_{mode}"] = float(
                            row["obj_attn_vis_ratio_real"] - obj_attn_ratio_mode
                        )
                    else:
                        row[f"obj_attn_vis_sum_{mode}"] = None
                        row[f"obj_attn_vis_ratio_{mode}"] = None
                        row[f"obj_delta_attn_vis_sum_real_minus_{mode}"] = None
                        row[f"obj_delta_attn_vis_ratio_real_minus_{mode}"] = None
                else:
                    row[f"obj_sim_local_topk_{mode}"] = None
                    row[f"obj_sim_local_max_{mode}"] = None
                    row[f"obj_sim_local_mean_{mode}"] = None
                    row[f"obj_sim_local_std_{mode}"] = None
                    row[f"obj_sim_local_gap_{mode}"] = None
                    row[f"obj_z_local_topk_{mode}"] = None
                    row[f"obj_z_local_max_{mode}"] = None
                    row[f"obj_z_local_gap_{mode}"] = None
                    row[f"obj_delta_topk_real_minus_{mode}"] = None
                    row[f"obj_delta_max_real_minus_{mode}"] = None
                    row[f"obj_delta_z_topk_real_minus_{mode}"] = None
                    row[f"obj_delta_z_max_real_minus_{mode}"] = None
                    row[f"obj_delta_z_gap_real_minus_{mode}"] = None
                    row[f"obj_sim_local_topk_pca_{mode}"] = None
                    row[f"obj_sim_local_max_pca_{mode}"] = None
                    row[f"obj_delta_topk_pca_real_minus_{mode}"] = None
                    row[f"obj_delta_max_pca_real_minus_{mode}"] = None
                    row[f"obj_attn_vis_sum_{mode}"] = None
                    row[f"obj_attn_vis_ratio_{mode}"] = None
                    row[f"obj_delta_attn_vis_sum_real_minus_{mode}"] = None
                    row[f"obj_delta_attn_vis_ratio_real_minus_{mode}"] = None

            per_sample_rows.append(row)

        except Exception as e:
            skipped += 1
            if logged_sample_errors < 5:
                print(f"[warn] skip sample id={sid} image_id={image_id}: {type(e).__name__}: {e}")
                logged_sample_errors += 1
            continue

    if len(per_sample_rows) == 0:
        raise RuntimeError("No extracted rows.")

    # Checkpoint raw extracted rows before expensive evaluation loops.
    ckpt_dir = os.path.join(out_dir, "_checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    write_csv(os.path.join(ckpt_dir, "per_sample_anchor.csv"), per_sample_rows)
    if bool(args.save_layer_trace):
        write_csv(os.path.join(ckpt_dir, "per_layer_yes_trace.csv"), per_layer_rows)
    if bool(args.save_head_trace):
        write_csv(os.path.join(ckpt_dir, "per_head_yes_trace.csv"), per_head_rows)
    print("[checkpoint]", os.path.join(ckpt_dir, "per_sample_anchor.csv"))
    if bool(args.save_layer_trace):
        print("[checkpoint]", os.path.join(ckpt_dir, "per_layer_yes_trace.csv"))
    if bool(args.save_head_trace):
        print("[checkpoint]", os.path.join(ckpt_dir, "per_head_yes_trace.csv"))

    # Main evaluation: fp_hallucination (positive) vs tp_yes (negative)
    eval_rows: List[Dict[str, Any]] = []
    for mode in modes:
        metric_candidates = sorted(
            [
                k
                for k in per_sample_rows[0].keys()
                if (k.startswith("yes_delta_") or k.startswith("obj_delta_"))
                and k.endswith(f"_real_minus_{mode}")
            ]
        )
        for metric in metric_candidates:
            labels: List[int] = []
            scores: List[float] = []
            for r in per_sample_rows:
                if not (bool(r.get("is_fp_hallucination")) or bool(r.get("is_tp_yes"))):
                    continue
                v = safe_float(r.get(metric))
                if v is None:
                    continue
                labels.append(1 if bool(r.get("is_fp_hallucination")) else 0)
                scores.append(float(v))
            if len(scores) < 20:
                continue
            rr = summarize_metric(metric, labels, scores, int(args.bootstrap), int(args.seed))
            rr["comparison"] = "fp_hall_vs_tp_yes"
            eval_rows.append(rr)

    eval_rows = sorted(
        eval_rows,
        key=lambda x: float(safe_float(x.get("auc_best_dir")) or -1.0),
        reverse=True,
    )

    layer_eval_rows: List[Dict[str, Any]] = []
    layer_curve_rows: List[Dict[str, Any]] = []
    head_eval_rows: List[Dict[str, Any]] = []
    head_curve_rows: List[Dict[str, Any]] = []
    if bool(args.save_layer_trace) and len(per_layer_rows) > 0:
        layer_ids = sorted(
            {
                int(r.get("block_layer_idx"))
                for r in per_layer_rows
                if safe_float(r.get("block_layer_idx")) is not None
            }
        )
        layer_metrics = [
            "yes_sim_local_topk",
            "yes_sim_local_max",
            "yes_sim_local_mean",
            "yes_sim_local_std",
            "yes_sim_local_gap",
            "yes_z_local_topk",
            "yes_z_local_max",
            "yes_z_local_gap",
            "yes_sim_objpatch_topk",
            "yes_sim_objpatch_max",
            "yes_attn_vis_sum",
            "yes_attn_vis_ratio",
        ]

        # Per-layer fp_hall vs tp_yes separation
        for layer in layer_ids:
            for metric in layer_metrics:
                labels_l: List[int] = []
                scores_l: List[float] = []
                for r in per_layer_rows:
                    if int(r.get("block_layer_idx")) != int(layer):
                        continue
                    if not (bool(r.get("is_fp_hallucination")) or bool(r.get("is_tp_yes"))):
                        continue
                    v = safe_float(r.get(metric))
                    if v is None:
                        continue
                    labels_l.append(1 if bool(r.get("is_fp_hallucination")) else 0)
                    scores_l.append(float(v))
                if len(scores_l) < 20:
                    continue
                rr = summarize_metric(
                    name=f"{metric}__layer_{int(layer)}",
                    labels=labels_l,
                    scores=scores_l,
                    bootstrap=int(args.bootstrap),
                    seed=int(args.seed) + 101 + int(layer),
                )
                rr["comparison"] = "fp_hall_vs_tp_yes"
                rr["metric_base"] = metric
                rr["block_layer_idx"] = int(layer)
                layer_eval_rows.append(rr)

        layer_eval_rows = sorted(
            layer_eval_rows,
            key=lambda x: float(safe_float(x.get("auc_best_dir")) or -1.0),
            reverse=True,
        )

        # Simple layer curves by group mean/median
        group_defs_layer = [
            ("fp_hall", lambda r: bool(r.get("is_fp_hallucination"))),
            ("tp_yes", lambda r: bool(r.get("is_tp_yes"))),
            ("tn_no", lambda r: bool(r.get("is_tn_no"))),
            ("fn_miss", lambda r: bool(r.get("is_fn_miss"))),
        ]
        for layer in layer_ids:
            for gname, gfn in group_defs_layer:
                grp = [
                    r
                    for r in per_layer_rows
                    if int(r.get("block_layer_idx")) == int(layer) and gfn(r)
                ]
                if len(grp) == 0:
                    continue
                outc: Dict[str, Any] = {
                    "block_layer_idx": int(layer),
                    "group": gname,
                    "n": int(len(grp)),
                }
                for metric in layer_metrics:
                    vals = [safe_float(r.get(metric)) for r in grp]
                    vals = [float(v) for v in vals if v is not None]
                    outc[f"{metric}__mean"] = (None if len(vals) == 0 else float(sum(vals) / len(vals)))
                    outc[f"{metric}__median"] = (None if len(vals) == 0 else quantile(vals, 0.5))
                layer_curve_rows.append(outc)

    if bool(args.save_head_trace) and len(per_head_rows) > 0:
        head_metrics = [
            "head_attn_vis_sum",
            "head_attn_vis_ratio",
            "head_attn_vis_peak",
            "head_attn_vis_entropy",
        ]

        # Build per-(layer,head) groups once to avoid repeatedly scanning all rows.
        rows_by_lh: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for r in per_head_rows:
            l = safe_float(r.get("block_layer_idx"))
            h = safe_float(r.get("head_idx"))
            if l is None or h is None:
                continue
            key = (int(l), int(h))
            rows_by_lh.setdefault(key, []).append(r)

        layer_head_pairs = sorted(rows_by_lh.keys())
        for layer, head_idx in layer_head_pairs:
            rows_lh = rows_by_lh.get((int(layer), int(head_idx)), [])
            rows_fptp = [
                r for r in rows_lh
                if bool(r.get("is_fp_hallucination")) or bool(r.get("is_tp_yes"))
            ]
            for metric in head_metrics:
                labels_h: List[int] = []
                scores_h: List[float] = []
                for r in rows_fptp:
                    v = safe_float(r.get(metric))
                    if v is None:
                        continue
                    labels_h.append(1 if bool(r.get("is_fp_hallucination")) else 0)
                    scores_h.append(float(v))
                if len(scores_h) < 20:
                    continue
                rr = summarize_metric(
                    name=f"{metric}__layer_{int(layer)}__head_{int(head_idx)}",
                    labels=labels_h,
                    scores=scores_h,
                    bootstrap=int(args.bootstrap),
                    seed=int(args.seed) + 5000 + int(layer) * 97 + int(head_idx),
                )
                rr["comparison"] = "fp_hall_vs_tp_yes"
                rr["metric_base"] = metric
                rr["block_layer_idx"] = int(layer)
                rr["head_idx"] = int(head_idx)
                head_eval_rows.append(rr)

        head_eval_rows = sorted(
            head_eval_rows,
            key=lambda x: float(safe_float(x.get("auc_best_dir")) or -1.0),
            reverse=True,
        )

        group_defs_head = [
            ("fp_hall", lambda r: bool(r.get("is_fp_hallucination"))),
            ("tp_yes", lambda r: bool(r.get("is_tp_yes"))),
            ("tn_no", lambda r: bool(r.get("is_tn_no"))),
            ("fn_miss", lambda r: bool(r.get("is_fn_miss"))),
        ]
        for layer, head_idx in layer_head_pairs:
            rows_lh = rows_by_lh.get((int(layer), int(head_idx)), [])
            for gname, gfn in group_defs_head:
                grp = [r for r in rows_lh if gfn(r)]
                if len(grp) == 0:
                    continue
                outc: Dict[str, Any] = {
                    "block_layer_idx": int(layer),
                    "head_idx": int(head_idx),
                    "group": gname,
                    "n": int(len(grp)),
                }
                for metric in head_metrics:
                    vals = [safe_float(r.get(metric)) for r in grp]
                    vals = [float(v) for v in vals if v is not None]
                    outc[f"{metric}__mean"] = (None if len(vals) == 0 else float(sum(vals) / len(vals)))
                    outc[f"{metric}__median"] = (None if len(vals) == 0 else quantile(vals, 0.5))
                head_curve_rows.append(outc)

    # Group means for quick sanity
    group_stats: List[Dict[str, Any]] = []
    group_defs = [
        ("fp_hall", lambda r: bool(r.get("is_fp_hallucination"))),
        ("tp_yes", lambda r: bool(r.get("is_tp_yes"))),
        ("tn_no", lambda r: bool(r.get("is_tn_no"))),
        ("fn_miss", lambda r: bool(r.get("is_fn_miss"))),
    ]
    stat_fields = [k for k in (per_sample_rows[0].keys() if len(per_sample_rows) > 0 else []) if k.endswith("_real") or "_delta_" in k]
    for gname, gfn in group_defs:
        grp = [r for r in per_sample_rows if gfn(r)]
        if len(grp) == 0:
            continue
        out = {"group": gname, "n": int(len(grp))}
        for sf in stat_fields:
            vals = [safe_float(r.get(sf)) for r in grp]
            vals = [float(v) for v in vals if v is not None]
            out[f"{sf}__mean"] = (None if len(vals) == 0 else float(sum(vals) / len(vals)))
            out[f"{sf}__median"] = quantile(vals, 0.5) if len(vals) > 0 else None
        group_stats.append(out)

    n_fp = int(sum(1 for r in per_sample_rows if bool(r.get("is_fp_hallucination"))))
    n_tp_yes = int(sum(1 for r in per_sample_rows if bool(r.get("is_tp_yes"))))
    n_tn = int(sum(1 for r in per_sample_rows if bool(r.get("is_tn_no"))))
    n_fn = int(sum(1 for r in per_sample_rows if bool(r.get("is_fn_miss"))))

    summary = {
        "inputs": {
            "samples_csv": os.path.abspath(args.samples_csv),
            "image_root": os.path.abspath(args.image_root),
            "dataset_mode": str(args.dataset_mode),
            "gqa_success_col": str(args.gqa_success_col),
            "gqa_anchor_field": str(args.gqa_anchor_field),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "topk_local": int(args.topk_local),
            "object_patch_topk": int(args.object_patch_topk),
            "pca_dim": int(args.pca_dim),
            "pca_probe_enabled": (not bool(args.disable_pca_probe)),
            "attention_probe_enabled": (not bool(args.disable_attention_probe)),
            "hidden_layer_idx": int(args.hidden_layer_idx),
            "attn_layer_idx": int(args.attn_layer_idx),
            "save_layer_trace": bool(args.save_layer_trace),
            "save_head_trace": bool(args.save_head_trace),
            "head_layer_start": int(args.head_layer_start),
            "head_layer_end": int(args.head_layer_end),
            "control_modes": modes,
            "shuffle_grid": int(args.shuffle_grid),
            "blur_radius": float(args.blur_radius),
            "bootstrap": int(args.bootstrap),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
        },
        "counts": {
            "n_rows_input": int(len(rows_in)),
            "n_rows_extracted": int(len(per_sample_rows)),
            "n_fp_hallucination": int(n_fp),
            "n_tp_yes": int(n_tp_yes),
            "n_tn_no": int(n_tn),
            "n_fn_miss": int(n_fn),
            "n_skipped": int(skipped),
        },
        "best_eval": (None if len(eval_rows) == 0 else eval_rows[0]),
        "best_layer_eval": (None if len(layer_eval_rows) == 0 else layer_eval_rows[0]),
        "best_head_eval": (None if len(head_eval_rows) == 0 else head_eval_rows[0]),
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample_anchor.csv"),
            "eval_csv": os.path.join(out_dir, "eval_table.csv"),
            "group_stats_csv": os.path.join(out_dir, "group_stats.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    if str(args.dataset_mode) == "gqa":
        summary["counts"]["n_failure"] = int(n_fp)
        summary["counts"]["n_success"] = int(n_tp_yes)
    if bool(args.save_layer_trace):
        summary["counts"]["n_layer_rows"] = int(len(per_layer_rows))
        summary["outputs"]["per_layer_trace_csv"] = os.path.join(out_dir, "per_layer_yes_trace.csv")
        summary["outputs"]["layer_eval_csv"] = os.path.join(out_dir, "layer_eval_fp_vs_tp_yes.csv")
        summary["outputs"]["layer_curve_csv"] = os.path.join(out_dir, "layer_curve_yes_by_group.csv")
    if bool(args.save_head_trace):
        summary["counts"]["n_head_rows"] = int(len(per_head_rows))
        summary["outputs"]["per_head_trace_csv"] = os.path.join(out_dir, "per_head_yes_trace.csv")
        summary["outputs"]["head_eval_csv"] = os.path.join(out_dir, "head_eval_fp_vs_tp_yes.csv")
        summary["outputs"]["head_curve_csv"] = os.path.join(out_dir, "head_curve_yes_by_group.csv")

    write_csv(os.path.join(out_dir, "per_sample_anchor.csv"), per_sample_rows)
    write_csv(os.path.join(out_dir, "eval_table.csv"), eval_rows)
    write_csv(os.path.join(out_dir, "group_stats.csv"), group_stats)
    if bool(args.save_layer_trace):
        write_csv(os.path.join(out_dir, "per_layer_yes_trace.csv"), per_layer_rows)
        write_csv(os.path.join(out_dir, "layer_eval_fp_vs_tp_yes.csv"), layer_eval_rows)
        write_csv(os.path.join(out_dir, "layer_curve_yes_by_group.csv"), layer_curve_rows)
    if bool(args.save_head_trace):
        write_csv(os.path.join(out_dir, "per_head_yes_trace.csv"), per_head_rows)
        write_csv(os.path.join(out_dir, "head_eval_fp_vs_tp_yes.csv"), head_eval_rows)
        write_csv(os.path.join(out_dir, "head_curve_yes_by_group.csv"), head_curve_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample_anchor.csv"))
    print("[saved]", os.path.join(out_dir, "eval_table.csv"))
    print("[saved]", os.path.join(out_dir, "group_stats.csv"))
    if bool(args.save_layer_trace):
        print("[saved]", os.path.join(out_dir, "per_layer_yes_trace.csv"))
        print("[saved]", os.path.join(out_dir, "layer_eval_fp_vs_tp_yes.csv"))
        print("[saved]", os.path.join(out_dir, "layer_curve_yes_by_group.csv"))
    if bool(args.save_head_trace):
        print("[saved]", os.path.join(out_dir, "per_head_yes_trace.csv"))
        print("[saved]", os.path.join(out_dir, "head_eval_fp_vs_tp_yes.csv"))
        print("[saved]", os.path.join(out_dir, "head_curve_yes_by_group.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
