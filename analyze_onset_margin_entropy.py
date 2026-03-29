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


def lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = int(min(len(a), len(b)))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return int(i)


def detect_failure_onset_idx(
    tokenizer,
    question: str,
    answer: str,
    pred_text: str,
    cont_ids: List[int],
    strict_onset: bool,
) -> Tuple[Optional[int], str, Dict[str, Any]]:
    tlen = int(len(cont_ids))
    info: Dict[str, Any] = {
        "gold_core_text": "",
        "pred_core_text": "",
        "pred_core_start": None,
        "pred_core_token_len": None,
        "lcp_gold_pred_core": None,
    }
    if tlen <= 0:
        return None, "empty", info

    gold_core = str(pf.extract_core_answer_text(question, answer) or "").strip()
    pred_core = str(pf.extract_core_answer_text(question, pred_text) or "").strip()
    info["gold_core_text"] = gold_core
    info["pred_core_text"] = pred_core

    pred_start, pred_core_ids = pf.locate_core_span_from_text(tokenizer, cont_ids, pred_core)
    info["pred_core_start"] = (None if pred_start is None else int(pred_start))
    info["pred_core_token_len"] = int(len(pred_core_ids))

    if pred_start is None or len(pred_core_ids) == 0:
        if bool(strict_onset):
            return None, "skip_no_core_span", info
        fstart = int(pf.infer_prefix_format_len_from_tokens(tokenizer, cont_ids, question))
        fstart = int(max(0, min(tlen - 1, fstart)))
        return int(fstart), "fallback_prefix_format", info

    gold_variants = pf.build_core_token_variants(tokenizer, gold_core)
    best_lcp = -1
    best_glen = 0
    for g in gold_variants:
        cur = lcp_len(pred_core_ids, g)
        if cur > best_lcp or (cur == best_lcp and len(g) > best_glen):
            best_lcp = int(cur)
            best_glen = int(len(g))
    if best_lcp < 0:
        best_lcp = 0
    info["lcp_gold_pred_core"] = int(best_lcp)

    if int(best_lcp) < int(len(pred_core_ids)):
        onset = int(pred_start + best_lcp)
        onset = int(max(0, min(tlen - 1, onset)))
        return onset, "core_divergence", info

    if bool(strict_onset):
        return None, "skip_no_divergence", info

    onset = int(max(0, min(tlen - 1, int(pred_start))))
    return onset, "fallback_core_start", info


def eval_binary_feature(
    labels: List[int],
    scores: List[float],
    bootstrap_n: int,
    seed: int,
) -> Dict[str, Any]:
    auc = auc_from_scores(labels, scores)
    ks = ks_from_scores(labels, scores)
    auc_lo, auc_hi = bootstrap_ci(auc_from_scores, labels, scores, int(bootstrap_n), int(seed) + 17)
    ks_lo, ks_hi = bootstrap_ci(ks_from_scores, labels, scores, int(bootstrap_n), int(seed) + 31)
    return {
        "n": int(len(scores)),
        "auc_onset_high": auc,
        "auc_best_dir": (None if auc is None else float(max(float(auc), 1.0 - float(auc)))),
        "direction": (
            None if auc is None else ("higher_in_onset" if float(auc) >= 0.5 else "lower_in_onset")
        ),
        "auc_ci95_lo": auc_lo,
        "auc_ci95_hi": auc_hi,
        "ks_onset_high": ks,
        "ks_ci95_lo": ks_lo,
        "ks_ci95_hi": ks_hi,
    }


def threshold_sweep(
    labels: List[int],
    scores: List[float],
    feature_name: str,
    steps: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    vals = [float(v) for v in scores if math.isfinite(float(v))]
    if len(vals) == 0:
        return [], {}, {}
    uniq = sorted(set(vals))
    ths: List[float]
    if len(uniq) <= int(max(2, steps)):
        ths = [float(v) for v in uniq]
    else:
        ths = []
        for i in range(int(max(2, steps))):
            q = float(i / (max(2, steps) - 1))
            t = quantile(vals, q)
            if t is not None:
                ths.append(float(t))
        ths = sorted(set(ths))

    rows: List[Dict[str, Any]] = []
    best_low: Optional[Dict[str, Any]] = None
    best_high: Optional[Dict[str, Any]] = None
    n = int(len(labels))
    n_pos = int(sum(1 for y in labels if int(y) == 1))
    n_neg = int(n - n_pos)
    if n_pos <= 0 or n_neg <= 0:
        return [], {}, {}

    for t in ths:
        for mode in ("onset_when_low", "onset_when_high"):
            if mode == "onset_when_low":
                pred = [1 if float(scores[i]) <= float(t) else 0 for i in range(n)]
            else:
                pred = [1 if float(scores[i]) >= float(t) else 0 for i in range(n)]

            tp = int(sum(1 for i in range(n) if labels[i] == 1 and pred[i] == 1))
            fp = int(sum(1 for i in range(n) if labels[i] == 0 and pred[i] == 1))
            tn = int(sum(1 for i in range(n) if labels[i] == 0 and pred[i] == 0))
            fn = int(sum(1 for i in range(n) if labels[i] == 1 and pred[i] == 0))

            tpr = float(tp / n_pos)
            fpr = float(fp / n_neg)
            tnr = float(tn / n_neg)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tpr)
            f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            youden = float(tpr + tnr - 1.0)
            row = {
                "feature": feature_name,
                "mode": mode,
                "threshold": float(t),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "tpr_recall": float(tpr),
                "fpr": float(fpr),
                "tnr_specificity": float(tnr),
                "precision": float(prec),
                "f1": float(f1),
                "youden_j": float(youden),
            }
            rows.append(row)
            if mode == "onset_when_low":
                if best_low is None or float(row["youden_j"]) > float(best_low["youden_j"]):
                    best_low = row
            else:
                if best_high is None or float(row["youden_j"]) > float(best_high["youden_j"]):
                    best_high = row

    return rows, (best_low or {}), (best_high or {})


def main() -> None:
    ap = argparse.ArgumentParser(description="Failure onset vs normal token separability by margin/entropy.")
    ap.add_argument("--baseline_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default="/home/kms/data/gqa/images")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default=None)
    ap.add_argument("--conv_mode", type=str, default=None)
    ap.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "eager"])
    ap.add_argument("--use_flash_attn", action="store_true")

    ap.add_argument("--topk_entropy", type=int, default=5)
    ap.add_argument("--threshold_steps", type=int, default=101)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--min_pred_tokens", type=int, default=1)
    ap.add_argument("--strict_onset", action="store_true")

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
    topk_entropy = int(max(2, args.topk_entropy))

    token_rows: List[Dict[str, Any]] = []
    skipped = 0
    n_success = 0
    n_failure = 0
    n_failure_onset_detected = 0
    n_failure_onset_skipped = 0
    onset_source_counts: Dict[str, int] = {}

    pbar = tqdm(rows_in, total=len(rows_in), desc="onset-margin-entropy", dynamic_ncols=True)
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
        tlen = int(len(cont_ids))
        if tlen < int(args.min_pred_tokens):
            skipped += 1
            continue

        onset_idx: Optional[int] = None
        onset_source = "success_all_normal"
        onset_info: Dict[str, Any] = {}
        if is_success:
            n_success += 1
        else:
            n_failure += 1
            onset_idx, onset_source, onset_info = detect_failure_onset_idx(
                tokenizer=tokenizer,
                question=question,
                answer=answer,
                pred_text=pred_text,
                cont_ids=cont_ids,
                strict_onset=bool(args.strict_onset),
            )
            if onset_idx is None:
                n_failure_onset_skipped += 1
                skipped += 1
                continue
            n_failure_onset_detected += 1
            onset_source_counts[onset_source] = int(onset_source_counts.get(onset_source, 0) + 1)

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
                out = model(
                    input_ids=full_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
                logits = out.logits[0, prompt_len - 1: prompt_len - 1 + tlen, :].float()
                probs = torch.softmax(logits, dim=-1)

                k = int(min(topk_entropy, probs.size(-1)))
                topk_vals, topk_ids = torch.topk(probs, k=k, dim=-1)
                p1 = topk_vals[:, 0]
                p2 = topk_vals[:, 1]
                margin = p1 - p2
                entropy_topk = -(topk_vals * torch.log(topk_vals + 1e-12)).sum(dim=-1)
                entropy_topk_norm = entropy_topk / math.log(float(k))
                topk_mass = topk_vals.sum(dim=-1)

            for i in range(tlen):
                token_role = "post_onset"
                label: Optional[int] = None
                if is_success:
                    token_role = "normal"
                    label = 0
                else:
                    if int(i) < int(onset_idx):
                        token_role = "normal"
                        label = 0
                    elif int(i) == int(onset_idx):
                        token_role = "onset"
                        label = 1

                row = {
                    "id": sid,
                    "image_id": image_id,
                    "question": question,
                    "answer": answer,
                    "pred_text": pred_text,
                    "is_success": bool(is_success),
                    "tok_idx": int(i),
                    "tok_pos_norm": (0.0 if tlen <= 1 else float(i / (tlen - 1))),
                    "token_id": int(cont_ids[i]),
                    "token_str": str(tokenizer.convert_ids_to_tokens(int(cont_ids[i]))),
                    "label_onset_vs_normal": label,
                    "token_role": token_role,
                    "onset_idx": (None if onset_idx is None else int(onset_idx)),
                    "onset_source": str(onset_source),
                    "gold_core_text": str(onset_info.get("gold_core_text", "")),
                    "pred_core_text": str(onset_info.get("pred_core_text", "")),
                    "pred_core_start": onset_info.get("pred_core_start", None),
                    "pred_core_token_len": onset_info.get("pred_core_token_len", None),
                    "lcp_gold_pred_core": onset_info.get("lcp_gold_pred_core", None),
                    "top1_prob": float(p1[i].item()),
                    "top2_prob": float(p2[i].item()),
                    "margin_top1_minus_top2": float(margin[i].item()),
                    "entropy_topk": float(entropy_topk[i].item()),
                    "entropy_topk_norm": float(entropy_topk_norm[i].item()),
                    "topk_mass": float(topk_mass[i].item()),
                    "topk": int(k),
                    "top1_token_id": int(topk_ids[i, 0].item()),
                    "top2_token_id": int(topk_ids[i, 1].item()),
                }
                token_rows.append(row)

        except Exception:
            skipped += 1
            continue

    labeled = [r for r in token_rows if r.get("label_onset_vs_normal") in (0, 1)]
    if len(labeled) == 0:
        raise RuntimeError("No labeled tokens for onset-vs-normal analysis.")

    labels = [int(r["label_onset_vs_normal"]) for r in labeled]
    scores_margin = [float(r["margin_top1_minus_top2"]) for r in labeled]
    scores_entropy = [float(r["entropy_topk"]) for r in labeled]
    scores_entropy_norm = [float(r["entropy_topk_norm"]) for r in labeled]

    metric_rows: List[Dict[str, Any]] = []
    margin_eval = eval_binary_feature(labels, scores_margin, int(args.bootstrap), int(args.seed))
    margin_eval["feature"] = "margin_top1_minus_top2"
    metric_rows.append(margin_eval)

    ent_eval = eval_binary_feature(labels, scores_entropy, int(args.bootstrap), int(args.seed) + 101)
    ent_eval["feature"] = f"entropy_top{int(topk_entropy)}"
    metric_rows.append(ent_eval)

    entn_eval = eval_binary_feature(labels, scores_entropy_norm, int(args.bootstrap), int(args.seed) + 211)
    entn_eval["feature"] = f"entropy_top{int(topk_entropy)}_norm"
    metric_rows.append(entn_eval)

    sweep_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    for fname, vals in (
        ("margin_top1_minus_top2", scores_margin),
        (f"entropy_top{int(topk_entropy)}", scores_entropy),
        (f"entropy_top{int(topk_entropy)}_norm", scores_entropy_norm),
    ):
        rows_s, best_low, best_high = threshold_sweep(
            labels=labels,
            scores=vals,
            feature_name=fname,
            steps=int(args.threshold_steps),
        )
        sweep_rows.extend(rows_s)
        if len(best_low) > 0:
            best_rows.append(dict(best_low))
        if len(best_high) > 0:
            best_rows.append(dict(best_high))

    best_rows = sorted(best_rows, key=lambda r: float(r.get("youden_j") or -1.0), reverse=True)

    n_normal = int(sum(1 for y in labels if y == 0))
    n_onset = int(sum(1 for y in labels if y == 1))
    summary = {
        "inputs": {
            "baseline_csv": os.path.abspath(args.baseline_csv),
            "image_root": os.path.abspath(args.image_root),
            "model_path": str(args.model_path),
            "conv_mode": str(conv_mode),
            "topk_entropy": int(topk_entropy),
            "threshold_steps": int(args.threshold_steps),
            "bootstrap": int(args.bootstrap),
            "strict_onset": bool(args.strict_onset),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
        },
        "counts": {
            "n_rows_input": int(len(rows_in)),
            "n_samples_success": int(n_success),
            "n_samples_failure": int(n_failure),
            "n_failure_onset_detected": int(n_failure_onset_detected),
            "n_failure_onset_skipped": int(n_failure_onset_skipped),
            "n_token_rows_total": int(len(token_rows)),
            "n_token_rows_labeled": int(len(labeled)),
            "n_normal_tokens": int(n_normal),
            "n_onset_tokens": int(n_onset),
            "n_skipped": int(skipped),
            "onset_source_counts": dict(sorted(onset_source_counts.items())),
        },
        "feature_metrics": metric_rows,
        "best_threshold_overall": (None if len(best_rows) == 0 else best_rows[0]),
        "targets": {
            "target_auc_gt": 0.65,
            "target_ks_gt": 0.30,
        },
        "outputs": {
            "per_token_csv": os.path.join(out_dir, "per_token_onset_labels.csv"),
            "feature_metrics_csv": os.path.join(out_dir, "feature_metrics.csv"),
            "threshold_sweep_csv": os.path.join(out_dir, "threshold_sweep.csv"),
            "threshold_best_csv": os.path.join(out_dir, "threshold_best.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_token_onset_labels.csv"), token_rows)
    write_csv(os.path.join(out_dir, "feature_metrics.csv"), metric_rows)
    write_csv(os.path.join(out_dir, "threshold_sweep.csv"), sweep_rows)
    write_csv(os.path.join(out_dir, "threshold_best.csv"), best_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_token_onset_labels.csv"))
    print("[saved]", os.path.join(out_dir, "feature_metrics.csv"))
    print("[saved]", os.path.join(out_dir, "threshold_sweep.csv"))
    print("[saved]", os.path.join(out_dir, "threshold_best.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()

