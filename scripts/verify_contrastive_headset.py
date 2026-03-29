#!/usr/bin/env python
"""
Verify per-sample contrastive null-image probing for dynamic faithful head selection.

Two candidate FRG metrics are compared against static headset FRG:
  Candidate 1 (FRG_A):     mean(vis_ratio_A of dyn faithful) - mean(vis_ratio_A of all late)
  Candidate 2 (FRG_delta): TopKMean(delta)  — direct image-dependent faithful response

Note: dynamic harmful heads are NOT computed (low image dependence ≠ harmful).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# ── helpers ─────────────────────────────────────────────────────────
def _mean(vals):
    vals = list(vals)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _extract_all_head_vis_ratios(
    attentions,
    image_start: int,
    image_end: int,
    late_start: int,
    late_end: int,
    eps: float = 1e-6,
) -> Dict[Tuple[int, int], float]:
    """Return {(layer, head): vis_ratio} for every head in the late window."""
    out: Dict[Tuple[int, int], float] = {}
    for layer_idx, attn in enumerate(attentions):
        if layer_idx < late_start or layer_idx > late_end:
            continue
        if attn is None:
            continue
        row = attn[0, :, -1, :].to(torch.float32)           # [H, K]
        vis_sum = row[:, image_start:image_end].sum(dim=-1)  # [H]
        txt_left = row[:, :image_start].sum(dim=-1)
        txt_right = row[:, image_end:].sum(dim=-1)
        txt_sum = txt_left + txt_right
        vis_ratio = vis_sum / torch.clamp(vis_sum + txt_sum, min=eps)  # [H]
        for h_idx in range(vis_ratio.size(0)):
            out[(layer_idx, h_idx)] = float(vis_ratio[h_idx].item())
    return out


def _jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    inter = a & b
    union = a | b
    return float(len(inter) / len(union)) if union else 0.0


# ── main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vga_root", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--image_folder", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--question_file", type=str, default="/home/kms/VISTA/pope_coco/coco_pope_random.json")
    ap.add_argument("--headset_json", type=str, default="/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--topk", type=int, default=16, help="Top-K heads to select as faithful")
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--out_dir", type=str, default="/home/kms/LLaVA_calibration/experiments/contrastive_headset_verify")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── bootstrap VGA imports ───────────────────────────────────
    vga_root = os.path.abspath(args.vga_root)
    if vga_root not in sys.path:
        sys.path.insert(0, vga_root)

    # Monkey-patch: newer transformers removed several internal helpers from
    # bloom and opt modules.  VGA_origin's MPT code imports them but we never
    # use MPT/OPT, so we shim them all as no-ops.
    _noop = lambda *a, **kw: a[0] if a else None
    import transformers.models.bloom.modeling_bloom as _bloom_mod
    for _fn in ("_expand_mask", "_make_causal_mask"):
        if not hasattr(_bloom_mod, _fn):
            setattr(_bloom_mod, _fn, _noop)
    import transformers.models.opt.modeling_opt as _opt_mod
    for _fn in ("_expand_mask", "_make_causal_mask"):
        if not hasattr(_opt_mod, _fn):
            setattr(_opt_mod, _fn, _noop)
    print("[DEBUG] Patched bloom/opt shims for MPT compatibility")

    from transformers import set_seed
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from vcd_utils.greedy_sample import evolve_greedy_sampling

    evolve_greedy_sampling()
    set_seed(42)

    # ── load model ──────────────────────────────────────────────
    print("[DEBUG] Loading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        os.path.expanduser(args.model_path), None, model_name, device=args.device,
    )
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head
    device = torch.device(args.device)
    print(f"[DEBUG] Model loaded: {model_name}")

    # ── load static headset ─────────────────────────────────────
    hs = json.loads(Path(args.headset_json).read_text())
    static_faithful = set()
    for x in hs.get("faithful_heads", []):
        static_faithful.add((int(x["layer"]), int(x["head"])))
    static_harmful = set()
    for x in hs.get("harmful_heads", []):
        static_harmful.add((int(x["layer"]), int(x["head"])))
    print(f"[DEBUG] Static headset: {len(static_faithful)} faithful, {len(static_harmful)} harmful")

    static_faithful_by_layer = {}
    for l, h in static_faithful:
        static_faithful_by_layer.setdefault(l, []).append(h)
    static_harmful_by_layer = {}
    for l, h in static_harmful:
        static_harmful_by_layer.setdefault(l, []).append(h)

    # ── load questions ──────────────────────────────────────────
    questions = []
    with open(args.question_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    questions = questions[:args.n_samples]
    print(f"[DEBUG] Loaded {len(questions)} questions")

    # ── prompt builder ──────────────────────────────────────────
    def build_prompt(question_text):
        qs = question_text.strip()
        if getattr(model.config, "mm_use_im_start_end", False):
            qs_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs_prompt = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs_prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    # ── image span helper ───────────────────────────────────────
    def get_image_span(input_ids, n_image_tokens):
        pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        if pos.numel() == 0:
            raise ValueError("No image token found")
        start = int(pos[0].item())
        return start, start + n_image_tokens

    # ── probe function (returns per-head vis_ratio) ─────────────
    def probe_vis_ratios(input_ids, image_tensor):
        with torch.inference_mode():
            prefill = model(
                input_ids[:, :-1],
                images=image_tensor.unsqueeze(0),
                use_cache=True,
                return_dict=True,
            )
            n_image_tokens = prefill.logits.size(1) - input_ids[:, :-1].size(1) + 576
            # Use the standard 576 image tokens for LLaVA-1.5
            n_image_tokens = 576

            probe_last = model(
                input_ids[:, -1:],
                attention_mask=torch.ones((1, 1), dtype=torch.long, device=device),
                past_key_values=prefill.past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        image_start, image_end = get_image_span(input_ids, n_image_tokens)
        vis_ratios = _extract_all_head_vis_ratios(
            attentions=probe_last.attentions,
            image_start=image_start,
            image_end=image_end,
            late_start=args.late_start,
            late_end=args.late_end,
        )
        return vis_ratios

    # ── helpers for TopKMean ──────────────────────────────────────
    def _topkmean(vals, k):
        vals = sorted(vals, reverse=True)
        k = max(1, min(k, len(vals)))
        return float(sum(vals[:k]) / k) if vals else 0.0

    # ── main loop ───────────────────────────────────────────────
    csv_rows = []
    all_head_overlap_faithful = []
    all_static_frg = []
    all_dyn_frg_A = []      # Candidate 1: mu_dyn - mu_late
    all_dyn_frg_delta = []   # Candidate 2: TopKMean(delta)

    t0 = time.time()
    for idx, q in enumerate(questions):
        question_text = q.get("text", q.get("question", ""))
        image_file = q.get("image", "")
        label = q.get("label", "")
        qid = q.get("question_id", idx)

        # Build prompt
        prompt = build_prompt(question_text)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        # Load real image
        img_path = os.path.join(args.image_folder, image_file)
        image = Image.open(img_path).convert("RGB")
        real_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(device)
        if device.type == "cuda":
            real_tensor = real_tensor.half()

        # Create black image (null baseline)
        black_tensor = torch.zeros_like(real_tensor)

        # Pass A: real image
        vis_ratio_A = probe_vis_ratios(input_ids, real_tensor)

        # Pass B: black image
        vis_ratio_B = probe_vis_ratios(input_ids, black_tensor)

        # Compute delta = vis_ratio(real) - vis_ratio(black)
        all_heads = sorted(set(vis_ratio_A.keys()) & set(vis_ratio_B.keys()))
        deltas = {}
        for key in all_heads:
            deltas[key] = vis_ratio_A[key] - vis_ratio_B[key]

        # Dynamic faithful = Top-K by delta (no harmful — low image dep ≠ harmful)
        sorted_by_delta = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
        dyn_faithful = set(k for k, _ in sorted_by_delta[:args.topk])

        # ── Candidate 1:  FRG_A = mu_dyn_faithful - mu_late ──
        all_vis_A = [vis_ratio_A[k] for k in all_heads]
        faithful_vis_A = [vis_ratio_A[k] for k in dyn_faithful]
        dyn_frg_A = _mean(faithful_vis_A) - _mean(all_vis_A)

        # ── Candidate 2:  FRG_delta = TopKMean(delta) ──
        delta_vals = [deltas[k] for k in all_heads]
        dyn_frg_delta = _topkmean(delta_vals, args.topk)

        # ── Static FRG (ground truth baseline) ──
        static_f_vis = [vis_ratio_A[k] for k in all_heads if k in static_faithful]
        static_frg = _mean(static_f_vis) - _mean(all_vis_A) if static_f_vis else 0.0

        # Head overlap (faithful only)
        jacc_f = _jaccard(dyn_faithful, static_faithful)
        all_head_overlap_faithful.append(jacc_f)
        all_static_frg.append(static_frg)
        all_dyn_frg_A.append(dyn_frg_A)
        all_dyn_frg_delta.append(dyn_frg_delta)

        csv_rows.append({
            "question_id": qid,
            "image": image_file,
            "label": label,
            "static_frg": f"{static_frg:.6f}",
            "dyn_frg_A": f"{dyn_frg_A:.6f}",
            "dyn_frg_delta": f"{dyn_frg_delta:.6f}",
            "jaccard_faithful": f"{jacc_f:.4f}",
            "dyn_faithful_heads": str(sorted(dyn_faithful)),
        })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[DEBUG] {idx+1}/{len(questions)}  elapsed={elapsed:.1f}s  "
                  f"jacc_f={jacc_f:.3f}  static={static_frg:.4f}  "
                  f"dyn_A={dyn_frg_A:.4f}  dyn_delta={dyn_frg_delta:.4f}")

    # ── write CSV ───────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "contrastive_vs_static.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[DEBUG] Saved CSV: {csv_path}")

    # ── compute summary stats ───────────────────────────────────
    import numpy as np
    from scipy.stats import spearmanr

    sf = np.array(all_static_frg)
    da = np.array(all_dyn_frg_A)
    dd = np.array(all_dyn_frg_delta)

    pearson_A = float(np.corrcoef(sf, da)[0, 1]) if len(sf) > 1 else 0.0
    pearson_delta = float(np.corrcoef(sf, dd)[0, 1]) if len(sf) > 1 else 0.0
    spearman_A_r, spearman_A_p = spearmanr(sf, da)
    spearman_delta_r, spearman_delta_p = spearmanr(sf, dd)

    mean_jacc_f = _mean(all_head_overlap_faithful)

    summary = {
        "n_samples": len(questions),
        "topk": args.topk,
        "late_window": f"{args.late_start}-{args.late_end}",
        "candidate_1_FRG_A": {
            "desc": "mu_dyn_faithful(vis_ratio_A) - mu_late(vis_ratio_A)",
            "pearson_vs_static": float(pearson_A),
            "spearman_vs_static": float(spearman_A_r),
            "spearman_pvalue": float(spearman_A_p),
            "mean": float(da.mean()),
            "std": float(da.std()),
        },
        "candidate_2_FRG_delta": {
            "desc": "TopKMean(delta = vis_ratio_real - vis_ratio_black)",
            "pearson_vs_static": float(pearson_delta),
            "spearman_vs_static": float(spearman_delta_r),
            "spearman_pvalue": float(spearman_delta_p),
            "mean": float(dd.mean()),
            "std": float(dd.std()),
        },
        "static_frg": {
            "mean": float(sf.mean()),
            "std": float(sf.std()),
        },
        "mean_jaccard_faithful": float(mean_jacc_f),
        "headset_json": os.path.abspath(args.headset_json),
        "question_file": os.path.abspath(args.question_file),
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] Saved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("CONTRASTIVE HEADSET VERIFICATION — TWO CANDIDATES")
    print("=" * 70)
    print(f"  Samples: {summary['n_samples']}   Top-K: {summary['topk']}")
    print(f"  Mean Jaccard (faithful):  {mean_jacc_f:.4f}")
    print(f"")
    print(f"  Static FRG (GT baseline):  mean={sf.mean():.4f}  std={sf.std():.4f}")
    print(f"")
    print(f"  Candidate 1 — FRG_A (mu_dyn - mu_late):")
    print(f"    mean={da.mean():.4f}  std={da.std():.4f}")
    print(f"    Pearson  vs static: {pearson_A:.4f}")
    print(f"    Spearman vs static: {spearman_A_r:.4f} (p={spearman_A_p:.2e})")
    print(f"")
    print(f"  Candidate 2 — FRG_delta (TopKMean of delta):")
    print(f"    mean={dd.mean():.4f}  std={dd.std():.4f}")
    print(f"    Pearson  vs static: {pearson_delta:.4f}")
    print(f"    Spearman vs static: {spearman_delta_r:.4f} (p={spearman_delta_p:.2e})")
    print("=" * 70)


if __name__ == "__main__":
    main()
