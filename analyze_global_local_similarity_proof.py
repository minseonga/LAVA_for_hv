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


def mean_or_none(xs: Sequence[float]) -> Optional[float]:
    ys = [float(v) for v in xs if math.isfinite(float(v))]
    if len(ys) == 0:
        return None
    return float(sum(ys) / len(ys))


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline proof for global/local visual grounding similarity.")
    ap.add_argument("--baseline_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--topk_local", type=int, default=16)
    ap.add_argument("--prefix_frac", type=float, default=0.4)
    ap.add_argument("--suffix_frac", type=float, default=0.6)
    ap.add_argument("--curve_bins", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--save_token_csv", action="store_true")

    ap.add_argument("--min_pred_tokens", type=int, default=1)
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
        raise RuntimeError("No valid rows in baseline CSV.")

    # Runtime imports
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
    topk_local = int(max(1, args.topk_local))
    pfrac = float(min(0.9, max(0.1, args.prefix_frac)))
    sfrac = float(min(0.95, max(0.05, args.suffix_frac)))

    per_sample: List[Dict[str, Any]] = []
    per_token: List[Dict[str, Any]] = []
    curve_acc: Dict[Tuple[str, str, int], List[float]] = {}
    skipped = 0

    pbar = tqdm(rows_in, total=len(rows_in), desc="global-local-sim", dynamic_ncols=True)
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
                img_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            cont_t = torch.tensor([cont_ids], dtype=torch.long, device=device)
            full_ids = torch.cat([prompt_ids, cont_t], dim=1)
            prompt_len = int(prompt_ids.size(1))

            image = Image.open(image_path).convert("RGB")
            images_tensor = process_images([image], image_processor, model.config).to(
                device=model.device,
                dtype=torch.float16,
            )
            image_sizes = [image.size]

            with torch.no_grad():
                # Projected visual tokens in language space [1, P, D]
                vis = model.encode_images(images_tensor)
                if isinstance(vis, (list, tuple)):
                    vis = vis[0]
                if vis.ndim == 3:
                    vis = vis[0]
                vis = vis.float()
                if vis.ndim != 2 or vis.size(0) < 2:
                    skipped += 1
                    continue

                out = model(
                    input_ids=full_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hs = out.hidden_states[-1][0]  # [seq, D]
                # Decision-state alignment:
                # token t in continuation is predicted by hidden at (prompt_len - 1 + t).
                dec_start = int(max(0, prompt_len - 1))
                dec_end = int(dec_start + len(cont_ids))
                h = hs[dec_start: dec_end, :].float()
                if h.size(0) == 0:
                    skipped += 1
                    continue

            # Cosine sims
            vis_n = F.normalize(vis, dim=-1)  # [P, D]
            g = torch.mean(vis, dim=0, keepdim=True)
            g_n = F.normalize(g, dim=-1)[0]  # [D]
            h_n = F.normalize(h, dim=-1)  # [T, D]

            sim_global = torch.matmul(h_n, g_n)  # [T]
            sim_patch = torch.matmul(h_n, vis_n.t())  # [T, P]
            kk = int(min(max(1, topk_local), sim_patch.size(1)))
            sim_local_topk = torch.topk(sim_patch, k=kk, dim=-1).values.mean(dim=-1)  # [T]
            sim_local_max = torch.max(sim_patch, dim=-1).values  # [T]
            sim_local_minus_global = sim_local_topk - sim_global

            sg = [float(x) for x in sim_global.cpu().tolist()]
            sl = [float(x) for x in sim_local_topk.cpu().tolist()]
            sm = [float(x) for x in sim_local_max.cpu().tolist()]
            dlg = [float(x) for x in sim_local_minus_global.cpu().tolist()]
            tlen = int(len(sg))
            if tlen == 0:
                skipped += 1
                continue

            pos = [0.0 if tlen <= 1 else float(i / (tlen - 1)) for i in range(tlen)]
            pref_idx = [i for i, p in enumerate(pos) if p <= pfrac]
            suf_idx = [i for i, p in enumerate(pos) if p >= sfrac]
            if len(pref_idx) == 0:
                pref_idx = list(range(min(2, tlen)))
            if len(suf_idx) == 0:
                suf_idx = list(range(max(0, tlen - 2), tlen))

            def pick(arr: List[float], idxs: List[int]) -> List[float]:
                return [float(arr[i]) for i in idxs if 0 <= i < len(arr)]

            sg_pref = pick(sg, pref_idx)
            sg_suf = pick(sg, suf_idx)
            sl_pref = pick(sl, pref_idx)
            sl_suf = pick(sl, suf_idx)
            dlg_pref = pick(dlg, pref_idx)
            dlg_suf = pick(dlg, suf_idx)

            peak_i = int(max(range(tlen), key=lambda i: sl[i]))
            post = [sl[i] for i in range(peak_i + 1, min(tlen, peak_i + 4))]
            post_mean = (float(sum(post) / len(post)) if len(post) > 0 else float(sl[peak_i]))

            row = {
                "id": sid,
                "image_id": image_id,
                "question": question,
                "answer": answer,
                "pred_text": pred_text,
                "is_success": bool(is_success),
                "is_failure": bool(not is_success),
                "n_tokens": int(tlen),
                # Global alignment
                "sim_global_mean": mean_or_none(sg),
                "sim_global_tail_mean": mean_or_none(sg_suf),
                "sim_global_min": (None if len(sg) == 0 else float(min(sg))),
                # Local alignment
                "sim_local_topk_mean": mean_or_none(sl),
                "sim_local_topk_tail_mean": mean_or_none(sl_suf),
                "sim_local_topk_min": (None if len(sl) == 0 else float(min(sl))),
                "sim_local_max_mean": mean_or_none(sm),
                # Local vs global contrast
                "sim_local_minus_global_mean": mean_or_none(dlg),
                "sim_local_minus_global_tail_mean": mean_or_none(dlg_suf),
                # Collapse-style morphology
                "sim_local_collapse_gap": (
                    None if len(sl_pref) == 0 or len(sl_suf) == 0
                    else float((sum(sl_pref) / len(sl_pref)) - (sum(sl_suf) / len(sl_suf)))
                ),
                "sim_local_peak_pos_norm": float(peak_i / max(1, tlen - 1)),
                "sim_local_postpeak_drop3": float(post_mean - float(sl[peak_i])),
            }
            per_sample.append(row)

            bins = int(max(2, args.curve_bins))
            cls = ("success" if is_success else "failure")
            for i in range(tlen):
                bi = int(min(bins - 1, max(0, math.floor(pos[i] * bins))))
                for key, val in (
                    ("sim_global", sg[i]),
                    ("sim_local_topk", sl[i]),
                    ("sim_local_minus_global", dlg[i]),
                ):
                    ckey = (cls, key, bi)
                    if ckey not in curve_acc:
                        curve_acc[ckey] = []
                    curve_acc[ckey].append(float(val))

                if bool(args.save_token_csv):
                    tok = tokenizer.convert_ids_to_tokens(int(cont_ids[i]))
                    per_token.append(
                        {
                            "id": sid,
                            "image_id": image_id,
                            "is_success": bool(is_success),
                            "tok_idx": int(i),
                            "tok_pos_norm": float(pos[i]),
                            "token_id": int(cont_ids[i]),
                            "token_str": str(tok),
                            "sim_global": float(sg[i]),
                            "sim_local_topk": float(sl[i]),
                            "sim_local_max": float(sm[i]),
                            "sim_local_minus_global": float(dlg[i]),
                        }
                    )

        except Exception:
            skipped += 1
            continue

    if len(per_sample) == 0:
        raise RuntimeError("No sample features extracted.")

    # Split ranking
    feature_names = [
        "sim_global_mean",
        "sim_global_tail_mean",
        "sim_global_min",
        "sim_local_topk_mean",
        "sim_local_topk_tail_mean",
        "sim_local_topk_min",
        "sim_local_max_mean",
        "sim_local_minus_global_mean",
        "sim_local_minus_global_tail_mean",
        "sim_local_collapse_gap",
        "sim_local_peak_pos_norm",
        "sim_local_postpeak_drop3",
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
        auc_lo, auc_hi = bootstrap_ci(auc_from_scores, labels, scores, int(args.bootstrap), int(args.seed) + 11)
        ks_lo, ks_hi = bootstrap_ci(ks_from_scores, labels, scores, int(args.bootstrap), int(args.seed) + 23)

        auc_best = None if auc is None else float(max(float(auc), 1.0 - float(auc)))
        direction = (
            None
            if auc is None
            else ("higher_in_failure" if float(auc) >= 0.5 else "lower_in_failure")
        )
        rank_rows.append(
            {
                "feature": fn,
                "n": int(len(scores)),
                "auc_failure_high": auc,
                "auc_best_dir": auc_best,
                "direction": direction,
                "auc_ci95_lo": auc_lo,
                "auc_ci95_hi": auc_hi,
                "ks_failure_high": ks,
                "ks_ci95_lo": ks_lo,
                "ks_ci95_hi": ks_hi,
                "corr_with_length": pearson_corr(scores, lens),
            }
        )

    rank_rows = sorted(
        rank_rows,
        key=lambda r: float(safe_float(r.get("ks_failure_high")) or -1.0),
        reverse=True,
    )

    # Curves
    curve_rows: List[Dict[str, Any]] = []
    bins = int(max(2, args.curve_bins))
    for cls in ("success", "failure"):
        for metric in ("sim_global", "sim_local_topk", "sim_local_minus_global"):
            for bi in range(bins):
                vals = curve_acc.get((cls, metric, bi), [])
                if len(vals) == 0:
                    continue
                curve_rows.append(
                    {
                        "class": cls,
                        "metric": metric,
                        "bin_idx": int(bi),
                        "pos_left": float(bi / bins),
                        "pos_right": float((bi + 1) / bins),
                        "n_tokens": int(len(vals)),
                        "mean": float(sum(vals) / len(vals)),
                        "median": quantile(vals, 0.5),
                        "q25": quantile(vals, 0.25),
                        "q75": quantile(vals, 0.75),
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
            "topk_local": int(topk_local),
            "prefix_frac": float(pfrac),
            "suffix_frac": float(sfrac),
            "curve_bins": int(args.curve_bins),
            "bootstrap": int(args.bootstrap),
            "save_token_csv": bool(args.save_token_csv),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
        },
        "counts": {
            "n_rows_input": int(len(rows_in)),
            "n_samples_extracted": int(len(per_sample)),
            "n_success": int(n_s),
            "n_failure": int(n_f),
            "n_token_rows": int(len(per_token)),
            "n_skipped": int(skipped),
        },
        "best_by_ks": (None if len(rank_rows) == 0 else rank_rows[0]),
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample_similarity_features.csv"),
            "feature_ranking_csv": os.path.join(out_dir, "feature_split_ranking.csv"),
            "curve_csv": os.path.join(out_dir, "curve_similarity_by_pos.csv"),
            "per_token_csv": (
                os.path.join(out_dir, "per_token_similarity.csv")
                if bool(args.save_token_csv)
                else None
            ),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_sample_similarity_features.csv"), per_sample)
    write_csv(os.path.join(out_dir, "feature_split_ranking.csv"), rank_rows)
    write_csv(os.path.join(out_dir, "curve_similarity_by_pos.csv"), curve_rows)
    if bool(args.save_token_csv):
        write_csv(os.path.join(out_dir, "per_token_similarity.csv"), per_token)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample_similarity_features.csv"))
    print("[saved]", os.path.join(out_dir, "feature_split_ranking.csv"))
    print("[saved]", os.path.join(out_dir, "curve_similarity_by_pos.csv"))
    if bool(args.save_token_csv):
        print("[saved]", os.path.join(out_dir, "per_token_similarity.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
