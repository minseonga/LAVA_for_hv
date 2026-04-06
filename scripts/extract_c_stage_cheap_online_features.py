#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import (
    CleanroomLlavaRuntime,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    safe_id,
    select_content_indices,
    write_csv,
    write_json,
)


def std_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0
    mu = float(sum(seq) / float(len(seq)))
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(math.sqrt(max(0.0, var)))


def mean_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def min_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(min(seq))


def max_or_zero(values: Sequence[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(max(seq))


def quantile_or_zero(values: Sequence[float], q: float) -> float:
    seq = sorted(float(v) for v in values)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    qq = float(min(max(q, 0.0), 1.0))
    pos = qq * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(seq[lo])
    w = pos - float(lo)
    return float((1.0 - w) * seq[lo] + w * seq[hi])


def summarize(values: Sequence[float], prefix: str) -> Dict[str, float]:
    seq = [float(v) for v in values]
    return {
        f"{prefix}_mean": mean_or_zero(seq),
        f"{prefix}_std": std_or_zero(seq),
        f"{prefix}_min": min_or_zero(seq),
        f"{prefix}_max": max_or_zero(seq),
    }


def build_feature_row(
    runtime: CleanroomLlavaRuntime,
    image_path: str,
    question: str,
    candidate_text: str,
    sample_id: str,
    image_name: str,
    *,
    lp_tail_quantile: float = 0.10,
    lp_tail_eps: float = 1e-6,
    lp_len_corr_alpha: float = 0.35,
) -> Dict[str, Any]:
    image = runtime.load_image(image_path)
    pack = runtime.teacher_force_candidate(
        image=image,
        question=question,
        candidate_text=candidate_text,
        output_attentions=False,
    )

    content_indices = select_content_indices(runtime.tokenizer, pack.cont_ids)
    logits = pack.logits.to(torch.float32)
    decision_positions = pack.decision_positions.long()
    target_ids = pack.labels_exp[pack.cont_label_positions.long()].long()

    token_logits = logits[decision_positions]
    log_probs = F.log_softmax(token_logits, dim=-1)
    probs = torch.softmax(token_logits, dim=-1)
    token_ent = -(probs * log_probs).sum(dim=-1)
    target_lp = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)

    top2_vals, top2_idx = torch.topk(token_logits, k=2, dim=-1)
    top1_logit = top2_vals[:, 0]
    top2_logit = top2_vals[:, 1]
    top1_id = top2_idx[:, 0]
    top1_margin = top1_logit - top2_logit
    target_logit = token_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    best_other_logit = torch.where(top1_id == target_ids, top2_logit, top1_logit)
    target_gap = target_logit - best_other_logit
    target_is_argmax = (top1_id == target_ids).to(torch.float32)

    cont_idx = list(range(int(target_ids.numel())))
    pick = content_indices if content_indices else cont_idx

    def pick_values(t: torch.Tensor, idxs: Sequence[int]) -> List[float]:
        return [float(t[int(i)].item()) for i in idxs]

    lp_all = pick_values(target_lp, cont_idx)
    lp_content = pick_values(target_lp, pick)
    ent_all = pick_values(token_ent, cont_idx)
    ent_content = pick_values(token_ent, pick)
    margin_all = pick_values(top1_margin, cont_idx)
    margin_content = pick_values(top1_margin, pick)
    gap_all = pick_values(target_gap, cont_idx)
    gap_content = pick_values(target_gap, pick)
    argmax_all = pick_values(target_is_argmax, cont_idx)
    argmax_content = pick_values(target_is_argmax, pick)

    lp_content_mean = mean_or_zero(lp_content)
    lp_content_std = std_or_zero(lp_content)
    lp_content_min = min_or_zero(lp_content)
    lp_content_tail_gap = float(lp_content_min - lp_content_mean)
    lp_content_tail_z = float(lp_content_tail_gap / float(max(lp_content_std, float(lp_tail_eps))))
    lp_content_q10 = quantile_or_zero(lp_content, float(lp_tail_quantile))
    lp_content_min_len_corr = float(
        lp_content_min + float(lp_len_corr_alpha) * math.log(max(1, len(pick)))
    )

    row: Dict[str, Any] = {
        "id": sample_id,
        "image": image_name,
        "question": question,
        "n_cont_tokens": int(len(cont_idx)),
        "n_content_tokens": int(len(pick)),
        "cheap_content_fraction": float(len(pick) / max(1, len(cont_idx))),
        "cheap_conflict_lp_minus_entropy": float(mean_or_zero(lp_content) - mean_or_zero(ent_content)),
        "cheap_conflict_gap_minus_entropy": float(mean_or_zero(gap_content) - mean_or_zero(ent_content)),
        "cheap_lp_content_tail_gap": lp_content_tail_gap,
        "cheap_lp_content_tail_z": lp_content_tail_z,
        "cheap_lp_content_q10": lp_content_q10,
        "cheap_lp_content_min_len_corr": lp_content_min_len_corr,
    }
    row.update(summarize(lp_all, "cheap_lp_all"))
    row.update(summarize(lp_content, "cheap_lp_content"))
    row.update(summarize(ent_all, "cheap_entropy_all"))
    row.update(summarize(ent_content, "cheap_entropy_content"))
    row.update(summarize(margin_all, "cheap_margin_all"))
    row.update(summarize(margin_content, "cheap_margin_content"))
    row.update(summarize(gap_all, "cheap_target_gap_all"))
    row.update(summarize(gap_content, "cheap_target_gap_content"))
    row.update(summarize(argmax_all, "cheap_target_argmax_all"))
    row.update(summarize(argmax_content, "cheap_target_argmax_content"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract cheap online C-stage proxy features from intervention predictions.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pred_text_key", type=str, default="auto")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--progress_style", type=str, default="tqdm", choices=["tqdm", "log"])
    ap.add_argument("--lp_tail_quantile", type=float, default=0.10)
    ap.add_argument("--lp_tail_eps", type=float, default=1e-6)
    ap.add_argument("--lp_len_corr_alpha", type=float, default=0.35)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}", flush=True)
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    pred_map = load_prediction_text_map(args.intervention_pred_jsonl, text_key=args.pred_text_key)

    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(args.model_base or None),
        conv_mode=args.conv_mode,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    use_tqdm = args.progress_style == "tqdm" and tqdm is not None
    progress = (
        tqdm(
            total=len(question_rows),
            desc="cheap-proxy",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
        )
        if use_tqdm
        else None
    )
    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        intervention_text = str(pred_map.get(sample_id, "")).strip()
        image_path = os.path.join(args.image_folder, image_name)

        row: Dict[str, Any] = {
            "id": sample_id,
            "image": image_name,
            "question": question,
            "score_error": "",
        }
        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not question:
                raise ValueError("Missing question text.")
            if not intervention_text:
                raise ValueError("Missing intervention prediction text.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            row.update(
                build_feature_row(
                    runtime=runtime,
                    image_path=image_path,
                    question=question,
                    candidate_text=intervention_text,
                    sample_id=sample_id,
                    image_name=image_name,
                    lp_tail_quantile=float(args.lp_tail_quantile),
                    lp_tail_eps=float(args.lp_tail_eps),
                    lp_len_corr_alpha=float(args.lp_len_corr_alpha),
                )
            )
        except Exception as exc:
            n_errors += 1
            row["score_error"] = str(exc)
        rows.append(row)
        if progress is not None:
            progress.update(1)
        elif (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[cheap-proxy] {idx + 1}/{len(question_rows)}", flush=True)

    if progress is not None:
        progress.close()

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}", flush=True)

    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "model_path": args.model_path,
                    "model_base": args.model_base,
                    "conv_mode": args.conv_mode,
                    "device": args.device,
                    "lp_tail_quantile": float(args.lp_tail_quantile),
                    "lp_tail_eps": float(args.lp_tail_eps),
                    "lp_len_corr_alpha": float(args.lp_len_corr_alpha),
                },
                "counts": {
                    "n_questions": int(len(question_rows)),
                    "n_rows": int(len(rows)),
                    "n_errors": int(n_errors),
                },
                "outputs": {
                    "out_csv": os.path.abspath(args.out_csv),
                },
            },
        )


if __name__ == "__main__":
    main()
