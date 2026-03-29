#!/usr/bin/env python
"""
Profile ALL late-window heads across POPE samples to find unsupervised
features that distinguish static faithful heads from the rest.

Only requires 1 forward pass per sample (real image only).
Outputs per-head statistics: mean, std, CV, entropy of vis_ratio.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


def _extract_per_head_features(
    attentions,
    image_start: int,
    image_end: int,
    late_start: int,
    late_end: int,
    eps: float = 1e-6,
) -> Dict[Tuple[int, int], dict]:
    """Return {(layer, head): {vis_ratio, entropy, peak_ratio}} for all late heads."""
    out = {}
    for layer_idx, attn in enumerate(attentions):
        if layer_idx < late_start or layer_idx > late_end:
            continue
        if attn is None:
            continue
        row = attn[0, :, -1, :].to(torch.float32)  # [H, K]
        n_heads = row.size(0)

        # vis_ratio
        vis_sum = row[:, image_start:image_end].sum(dim=-1)
        txt_left = row[:, :image_start].sum(dim=-1)
        txt_right = row[:, image_end:].sum(dim=-1)
        txt_sum = txt_left + txt_right
        vis_ratio = vis_sum / torch.clamp(vis_sum + txt_sum, min=eps)

        # attention distribution over image patches (normalized)
        img_attn = row[:, image_start:image_end]  # [H, N_img]
        img_attn_norm = img_attn / torch.clamp(img_attn.sum(dim=-1, keepdim=True), min=eps)

        # entropy of attention over image patches
        log_p = torch.log(torch.clamp(img_attn_norm, min=eps))
        entropy = -(img_attn_norm * log_p).sum(dim=-1)  # [H]
        max_entropy = float(np.log(max(1, image_end - image_start)))

        # peak ratio: max attention / mean attention (concentration)
        peak = img_attn_norm.max(dim=-1).values  # [H]

        for h_idx in range(n_heads):
            out[(layer_idx, h_idx)] = {
                "vis_ratio": float(vis_ratio[h_idx].item()),
                "entropy": float(entropy[h_idx].item()),
                "entropy_norm": float(entropy[h_idx].item() / max_entropy) if max_entropy > 0 else 0.0,
                "peak_ratio": float(peak[h_idx].item()),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vga_root", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--image_folder", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--question_file", type=str, default="/home/kms/VISTA/pope_coco/coco_pope_random.json")
    ap.add_argument("--headset_json", type=str, default="/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--out_dir", type=str, default="/home/kms/LLaVA_calibration/experiments/head_profile")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── bootstrap VGA imports ───────────────────────────────────
    vga_root = os.path.abspath(args.vga_root)
    if vga_root not in sys.path:
        sys.path.insert(0, vga_root)

    _noop = lambda *a, **kw: a[0] if a else None
    import transformers.models.bloom.modeling_bloom as _bloom_mod
    for _fn in ("_expand_mask", "_make_causal_mask"):
        if not hasattr(_bloom_mod, _fn):
            setattr(_bloom_mod, _fn, _noop)
    import transformers.models.opt.modeling_opt as _opt_mod
    for _fn in ("_expand_mask", "_make_causal_mask"):
        if not hasattr(_opt_mod, _fn):
            setattr(_opt_mod, _fn, _noop)
    print("[DEBUG] Patched bloom/opt shims")

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

    # ── collect per-head data ───────────────────────────────────
    # Accumulate: head_data[(l,h)] = {"vis_ratio": [...], "entropy": [...], ...}
    head_data = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    for idx, q in enumerate(questions):
        question_text = q.get("text", q.get("question", ""))
        image_file = q.get("image", "")

        prompt = build_prompt(question_text)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        img_path = os.path.join(args.image_folder, image_file)
        image = Image.open(img_path).convert("RGB")
        real_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(device)
        if device.type == "cuda":
            real_tensor = real_tensor.half()

        with torch.inference_mode():
            prefill = model(
                input_ids[:, :-1],
                images=real_tensor.unsqueeze(0),
                use_cache=True,
                return_dict=True,
            )
            probe_last = model(
                input_ids[:, -1:],
                attention_mask=torch.ones((1, 1), dtype=torch.long, device=device),
                past_key_values=prefill.past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        image_start = int(pos[0].item())
        image_end = image_start + 576

        features = _extract_per_head_features(
            attentions=probe_last.attentions,
            image_start=image_start,
            image_end=image_end,
            late_start=args.late_start,
            late_end=args.late_end,
        )

        for key, feat in features.items():
            for feat_name, feat_val in feat.items():
                head_data[key][feat_name].append(feat_val)

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"[DEBUG] {idx+1}/{len(questions)}  elapsed={elapsed:.1f}s")

    # ── compute per-head statistics ─────────────────────────────
    print(f"\n[DEBUG] Computing per-head statistics over {len(questions)} samples...")

    rows = []
    for (layer, head), feat_dict in sorted(head_data.items()):
        role = "faithful" if (layer, head) in static_faithful else \
               "harmful" if (layer, head) in static_harmful else "other"

        vis_arr = np.array(feat_dict["vis_ratio"])
        ent_arr = np.array(feat_dict["entropy_norm"])
        peak_arr = np.array(feat_dict["peak_ratio"])

        rows.append({
            "layer": layer,
            "head": head,
            "role": role,
            "vis_ratio_mean": float(vis_arr.mean()),
            "vis_ratio_std": float(vis_arr.std()),
            "vis_ratio_cv": float(vis_arr.std() / max(vis_arr.mean(), 1e-8)),
            "vis_ratio_median": float(np.median(vis_arr)),
            "vis_ratio_q25": float(np.percentile(vis_arr, 25)),
            "vis_ratio_q75": float(np.percentile(vis_arr, 75)),
            "entropy_norm_mean": float(ent_arr.mean()),
            "entropy_norm_std": float(ent_arr.std()),
            "peak_ratio_mean": float(peak_arr.mean()),
            "peak_ratio_std": float(peak_arr.std()),
        })

    # ── print comparison table ──────────────────────────────────
    print("\n" + "=" * 90)
    print("HEAD PROFILE: FAITHFUL vs HARMFUL vs OTHER (unsupervised features)")
    print("=" * 90)

    for role_name in ["faithful", "harmful", "other"]:
        role_rows = [r for r in rows if r["role"] == role_name]
        if not role_rows:
            continue
        n = len(role_rows)
        vr_mean = np.mean([r["vis_ratio_mean"] for r in role_rows])
        vr_std = np.mean([r["vis_ratio_std"] for r in role_rows])
        vr_cv = np.mean([r["vis_ratio_cv"] for r in role_rows])
        ent_mean = np.mean([r["entropy_norm_mean"] for r in role_rows])
        peak_mean = np.mean([r["peak_ratio_mean"] for r in role_rows])

        print(f"\n  [{role_name.upper()}] ({n} heads)")
        print(f"    vis_ratio:     mean={vr_mean:.4f}  std={vr_std:.4f}  CV={vr_cv:.4f}")
        print(f"    entropy_norm:  mean={ent_mean:.4f}")
        print(f"    peak_ratio:    mean={peak_mean:.4f}")

    # ── per-head detail for faithful ────────────────────────────
    print("\n" + "-" * 90)
    print("DETAILED: Faithful heads (sorted by vis_ratio_mean)")
    print(f"{'layer':>5} {'head':>4} {'vis_mean':>9} {'vis_std':>8} {'vis_CV':>7} {'ent_norm':>9} {'peak':>7}")
    for r in sorted([r for r in rows if r["role"] == "faithful"],
                    key=lambda x: -x["vis_ratio_mean"]):
        print(f"  {r['layer']:>3}   {r['head']:>3}   {r['vis_ratio_mean']:.4f}   "
              f"{r['vis_ratio_std']:.4f}   {r['vis_ratio_cv']:.3f}   "
              f"{r['entropy_norm_mean']:.4f}   {r['peak_ratio_mean']:.4f}")

    # ── statistical tests ───────────────────────────────────────
    from scipy.stats import mannwhitneyu

    faithful_rows = [r for r in rows if r["role"] == "faithful"]
    other_rows = [r for r in rows if r["role"] == "other"]

    print("\n" + "-" * 90)
    print("MANN-WHITNEY U TESTS: Faithful vs Other")
    for feat in ["vis_ratio_mean", "vis_ratio_std", "vis_ratio_cv",
                  "entropy_norm_mean", "peak_ratio_mean"]:
        f_vals = [r[feat] for r in faithful_rows]
        o_vals = [r[feat] for r in other_rows]
        stat, p = mannwhitneyu(f_vals, o_vals, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        f_med = np.median(f_vals)
        o_med = np.median(o_vals)
        print(f"  {feat:<22}  faithful_med={f_med:.4f}  other_med={o_med:.4f}  "
              f"U={stat:.0f}  p={p:.4e}  {sig}")

    # ── save results ────────────────────────────────────────────
    import csv
    csv_path = os.path.join(args.out_dir, "head_profile.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DEBUG] Saved CSV: {csv_path}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"n_samples": len(questions), "n_heads": len(rows),
                    "n_faithful": len(faithful_rows), "n_harmful": len([r for r in rows if r["role"] == "harmful"]),
                    "n_other": len(other_rows)}, f, indent=2)
    print(f"[DEBUG] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
