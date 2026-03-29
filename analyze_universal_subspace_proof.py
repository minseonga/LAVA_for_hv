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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

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


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = [float(v) for v in xs if math.isfinite(float(v))]
    y = [float(v) for v in ys if math.isfinite(float(v))]
    if len(x) != len(xs) or len(y) != len(ys):
        return None
    mx = float(sum(x) / len(x))
    my = float(sum(y) / len(y))
    vx = float(sum((v - mx) ** 2 for v in x))
    vy = float(sum((v - my) ** 2 for v in y))
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = float(sum((x[i] - mx) * (y[i] - my) for i in range(len(x))))
    return float(cov / math.sqrt(vx * vy))


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = [(int(labels[i]), float(scores[i])) for i in range(len(labels))]
    pos = [p for p in pairs if p[0] == 1]
    neg = [p for p in pairs if p[0] == 0]
    n_pos = len(pos)
    n_neg = len(neg)
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
    i = 0
    j = 0
    dmax = 0.0
    while i < n_pos or j < n_neg:
        if j >= n_neg or (i < n_pos and pos[i] <= neg[j]):
            v = pos[i]
            while i < n_pos and pos[i] == v:
                i += 1
        else:
            v = neg[j]
            while j < n_neg and neg[j] == v:
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
    lo = quantile(vals, 0.025)
    hi = quantile(vals, 0.975)
    return lo, hi


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
            best_score = float(score)
            best = ids
    return best


@dataclass
class SampleHidden:
    sid: str
    image_id: str
    question: str
    answer: str
    pred_text: str
    is_success: bool
    hidden: torch.Tensor  # [T, D] on CPU float32


def fit_subspace_from_success(
    success_hidden: torch.Tensor,
    var_threshold: float,
    max_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    if success_hidden.ndim != 2 or success_hidden.size(0) < 2:
        raise RuntimeError("Need at least two success hidden vectors for PCA.")
    x = success_hidden.float()
    mu = torch.mean(x, dim=0, keepdim=True)
    xc = x - mu

    q = int(min(max(2, int(max_rank)), xc.size(0) - 1, xc.size(1)))
    if q < 2:
        q = int(min(xc.size(0), xc.size(1)))
    u, s, v = torch.pca_lowrank(xc, q=q, center=False)
    var = s ** 2
    denom = float(torch.sum(var).item())
    if denom <= 0.0:
        raise RuntimeError("Degenerate PCA variance.")
    ratio = var / denom
    csum = torch.cumsum(ratio, dim=0)
    kk = int((csum >= float(var_threshold)).nonzero(as_tuple=False)[0].item() + 1)
    kk = int(max(1, min(kk, v.size(1))))
    explained = float(csum[kk - 1].item())
    basis = v[:, :kk].contiguous()  # [D, k]
    return mu.squeeze(0), basis, kk, explained


def residual_ratio(x: torch.Tensor, mu: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    # x: [T, D], mu: [D], basis: [D, k]
    xc = x - mu.unsqueeze(0)
    proj = (xc @ basis) @ basis.t()
    res = xc - proj
    res_n = torch.norm(res, p=2, dim=1)
    tot_n = torch.norm(xc, p=2, dim=1) + 1e-8
    return (res_n / tot_n).float()


def main() -> None:
    ap = argparse.ArgumentParser(description="Universal subspace proof (PCA residual) on existing baseline run.")
    ap.add_argument("--baseline_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--var_threshold", type=float, default=0.9)
    ap.add_argument("--max_rank", type=int, default=256)
    ap.add_argument("--min_pred_tokens", type=int, default=1)
    ap.add_argument("--curve_bins", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows_in: List[Dict[str, Any]] = []
    with open(os.path.abspath(args.baseline_csv), "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if str(r.get("error") or "").strip() != "":
                continue
            if str(r.get("pred_text") or "").strip() == "":
                continue
            rows_in.append(r)
    if int(args.num_samples) > 0:
        rows_in = rows_in[: int(args.num_samples)]
    if len(rows_in) == 0:
        raise RuntimeError("No valid rows from baseline_csv.")

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

    samples: List[SampleHidden] = []
    success_chunks: List[torch.Tensor] = []
    skipped = 0

    pbar = tqdm(rows_in, total=len(rows_in), desc="subspace-extract", dynamic_ncols=True)
    for rr in pbar:
        sid = str(rr.get("id") or "")
        image_id = str(rr.get("image_id") or "")
        question = str(rr.get("question") or "")
        answer = str(rr.get("answer") or "")
        pred_text = str(rr.get("pred_text") or "")
        is_success = bool(parse_bool(rr.get("is_success")))

        image_path = os.path.join(args.image_root, f"{image_id}.jpg")
        if sid == "" or image_id == "" or not os.path.isfile(image_path):
            skipped += 1
            continue

        cont_ids = choose_cont_ids(tokenizer, pred_text)
        if len(cont_ids) < int(args.min_pred_tokens):
            skipped += 1
            continue

        try:
            img_prompt = pf.build_prompt(
                question=question,
                conv_mode=conv_mode,
                with_image_token=True,
                mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
            )
            prompt_ids = tokenizer_image_token(
                img_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)
            cont_t = torch.tensor([cont_ids], dtype=torch.long, device=device)
            full_ids = torch.cat([prompt_ids, cont_t], dim=1)
            prompt_len = int(prompt_ids.size(1))

            image = Image.open(image_path).convert("RGB")
            image_sizes = [image.size]
            images_tensor = process_images([image], image_processor, model.config).to(
                device=model.device,
                dtype=torch.float16,
            )

            with torch.no_grad():
                out = model(
                    input_ids=full_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hs = out.hidden_states[-1][0]  # [seq, dim]
            # Decision-state alignment:
            # token t in continuation is predicted by hidden at (prompt_len - 1 + t).
            dec_start = int(max(0, prompt_len - 1))
            dec_end = int(dec_start + len(cont_ids))
            h_cont = hs[dec_start: dec_end, :].detach().float().cpu()
            if h_cont.size(0) == 0:
                skipped += 1
                continue

            item = SampleHidden(
                sid=sid,
                image_id=image_id,
                question=question,
                answer=answer,
                pred_text=pred_text,
                is_success=is_success,
                hidden=h_cont,
            )
            samples.append(item)
            if bool(is_success):
                success_chunks.append(h_cont)
        except Exception:
            skipped += 1
            continue

    if len(samples) == 0:
        raise RuntimeError("No extracted hidden trajectories.")
    if len(success_chunks) == 0:
        raise RuntimeError("No success trajectories available for universal subspace.")

    x_succ = torch.cat(success_chunks, dim=0)
    mu, basis, k_dim, explained = fit_subspace_from_success(
        success_hidden=x_succ,
        var_threshold=float(args.var_threshold),
        max_rank=int(args.max_rank),
    )

    per_sample: List[Dict[str, Any]] = []
    curve_acc: Dict[Tuple[str, int], List[float]] = {}

    for s in tqdm(samples, total=len(samples), desc="subspace-residual", dynamic_ncols=True):
        rr = residual_ratio(s.hidden, mu, basis).cpu().tolist()
        t = len(rr)
        if t == 0:
            continue
        rr_vals = [float(v) for v in rr]
        tail_start = int(math.floor(0.7 * t))
        tail = rr_vals[tail_start:] if tail_start < t else [rr_vals[-1]]
        med = quantile(rr_vals, 0.5)
        peak_idx = int(max(range(t), key=lambda i: rr_vals[i]))
        slope = None
        if t >= 2:
            xs = [float(i) for i in range(t)]
            mx = float(sum(xs) / t)
            my = float(sum(rr_vals) / t)
            denom = float(sum((x - mx) ** 2 for x in xs))
            if denom > 0.0:
                slope = float(sum((xs[i] - mx) * (rr_vals[i] - my) for i in range(t)) / denom)

        row = {
            "id": s.sid,
            "image_id": s.image_id,
            "is_success": bool(s.is_success),
            "is_failure": bool(not s.is_success),
            "n_tokens": int(t),
            "residual_mean": float(sum(rr_vals) / t),
            "residual_max": float(max(rr_vals)),
            "residual_tail_mean": float(sum(tail) / len(tail)),
            "residual_tail_max": float(max(tail)),
            "residual_last": float(rr_vals[-1]),
            "residual_spike": (None if med is None else float(max(rr_vals) - float(med))),
            "residual_peak_pos_norm": float(peak_idx / max(1, t - 1)),
            "residual_slope": slope,
        }
        per_sample.append(row)

        bins = int(max(2, args.curve_bins))
        for i, v in enumerate(rr_vals):
            pos = (0.0 if t <= 1 else float(i / (t - 1)))
            bi = int(min(bins - 1, max(0, math.floor(pos * bins))))
            key = ("success" if s.is_success else "failure", bi)
            if key not in curve_acc:
                curve_acc[key] = []
            curve_acc[key].append(float(v))

    feature_names = [
        "residual_mean",
        "residual_max",
        "residual_tail_mean",
        "residual_tail_max",
        "residual_last",
        "residual_spike",
        "residual_peak_pos_norm",
        "residual_slope",
    ]
    rank_rows: List[Dict[str, Any]] = []
    for fn in feature_names:
        labels: List[int] = []
        scores: List[float] = []
        lens: List[float] = []
        for r in per_sample:
            v = safe_float(r.get(fn))
            ln = safe_float(r.get("n_tokens"))
            if v is None or ln is None:
                continue
            labels.append(1 if bool(r.get("is_failure")) else 0)
            scores.append(float(v))
            lens.append(float(ln))
        if len(scores) < 10:
            continue
        auc = auc_from_scores(labels, scores)
        ks = ks_from_scores(labels, scores)
        auc_lo, auc_hi = bootstrap_ci(auc_from_scores, labels, scores, int(args.bootstrap), int(args.seed) + 13)
        ks_lo, ks_hi = bootstrap_ci(ks_from_scores, labels, scores, int(args.bootstrap), int(args.seed) + 29)
        corr_len = pearson_corr(scores, lens)
        rank_rows.append(
            {
                "feature": fn,
                "n": int(len(scores)),
                "auc_failure_high": auc,
                "auc_ci95_lo": auc_lo,
                "auc_ci95_hi": auc_hi,
                "ks_failure_high": ks,
                "ks_ci95_lo": ks_lo,
                "ks_ci95_hi": ks_hi,
                "corr_with_length": corr_len,
            }
        )

    rank_rows = sorted(
        rank_rows,
        key=lambda r: float(safe_float(r.get("ks_failure_high")) or -1.0),
        reverse=True,
    )

    curve_rows: List[Dict[str, Any]] = []
    bins = int(max(2, args.curve_bins))
    for cls in ("success", "failure"):
        for bi in range(bins):
            vals = curve_acc.get((cls, bi), [])
            if len(vals) == 0:
                continue
            curve_rows.append(
                {
                    "class": cls,
                    "bin_idx": int(bi),
                    "pos_left": float(bi / bins),
                    "pos_right": float((bi + 1) / bins),
                    "n_tokens": int(len(vals)),
                    "residual_mean": float(sum(vals) / len(vals)),
                    "residual_median": quantile(vals, 0.5),
                    "residual_q25": quantile(vals, 0.25),
                    "residual_q75": quantile(vals, 0.75),
                }
            )

    n_s = int(sum(1 for r in per_sample if bool(r.get("is_success"))))
    n_f = int(sum(1 for r in per_sample if bool(r.get("is_failure"))))
    summary = {
        "inputs": {
            "baseline_csv": os.path.abspath(args.baseline_csv),
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "var_threshold": float(args.var_threshold),
            "max_rank": int(args.max_rank),
            "min_pred_tokens": int(args.min_pred_tokens),
            "curve_bins": int(args.curve_bins),
            "bootstrap": int(args.bootstrap),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
        },
        "counts": {
            "n_rows_input": int(len(rows_in)),
            "n_samples_extracted": int(len(per_sample)),
            "n_success": int(n_s),
            "n_failure": int(n_f),
            "n_skipped": int(skipped),
            "subspace_k": int(k_dim),
            "subspace_explained_variance": float(explained),
        },
        "best_by_ks": (None if len(rank_rows) == 0 else rank_rows[0]),
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample_residual_features.csv"),
            "feature_ranking_csv": os.path.join(out_dir, "feature_split_ranking.csv"),
            "curve_csv": os.path.join(out_dir, "curve_residual_by_pos.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_sample_residual_features.csv"), per_sample)
    write_csv(os.path.join(out_dir, "feature_split_ranking.csv"), rank_rows)
    write_csv(os.path.join(out_dir, "curve_residual_by_pos.csv"), curve_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample_residual_features.csv"))
    print("[saved]", os.path.join(out_dir, "feature_split_ranking.csv"))
    print("[saved]", os.path.join(out_dir, "curve_residual_by_pos.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
