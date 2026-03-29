#!/usr/bin/env python
"""
discover_headset_from_pope_style.py

Runs LLaVA-1.5-7b inference on a POPE-style JSONL discovery set built
from COCO train2014 images (strictly disjoint from POPE val2014 eval images).

For each sample, collects head_attn_vis_ratio for all late-layer heads.
Treats GT yes/no labels as the signal. Computes ROC-AUC for each head.

"Faithful" heads: high AUC, direction = lower_in_hallucination 
  (vis_ratio is LOW when model says "yes" but GT says "no")
"Harmful" heads: high AUC, direction = higher_in_hallucination
  (vis_ratio is HIGH when model says "yes" but GT says "no")

Then saves an output headset JSON compatible with VGAOnlineAdapter static_headset mode.

Usage (step 2 of the pipeline):
    python scripts/discover_headset_from_pope_style.py \
        --question_file /home/kms/LLaVA_calibration/experiments/pope_discovery/discovery_adversarial.jsonl \
        --image_folder /home/kms/data/images/mscoco/images/train2014 \
        --out_dir /home/kms/LLaVA_calibration/experiments/pope_discovery \
        --n_samples 1000
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score


def _mean(vals):
    vals = list(vals)
    return float(sum(vals) / len(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vga_root", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--question_file", type=str,
                    default="/home/kms/LLaVA_calibration/experiments/pope_discovery/discovery_adversarial.jsonl")
    ap.add_argument("--image_folder", type=str,
                    default="/home/kms/data/images/mscoco/images/train2014")
    ap.add_argument("--out_dir", type=str,
                    default="/home/kms/LLaVA_calibration/experiments/pope_discovery")
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--auc_min", type=float, default=0.60,
                    help="Minimum AUC to be considered as a candidate head")
    ap.add_argument("--n_samples", type=int, default=1000,
                    help="Max number of questions to use from the discovery file")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Bootstrap VGA ────────────────────────────────────────────────
    vga_root = os.path.abspath(args.vga_root)
    if vga_root not in sys.path:
        sys.path.insert(0, vga_root)

    _noop = lambda *a, **kw: a[0] if a else None
    import transformers.models.bloom.modeling_bloom as _b
    for _fn in ("_expand_mask", "_make_causal_mask"):
        if not hasattr(_b, _fn): setattr(_b, _fn, _noop)
    import transformers.models.opt.modeling_opt as _o
    for _fn in ("_expand_mask", "_make_causal_mask"):
        if not hasattr(_o, _fn): setattr(_o, _fn, _noop)

    from transformers import set_seed
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from vcd_utils.greedy_sample import evolve_greedy_sampling

    evolve_greedy_sampling()
    set_seed(42)

    print("[DEBUG] Loading model ...")
    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        os.path.expanduser(args.model_path), None, model_name, device=args.device)
    tokenizer.padding_side = "right"
    device = torch.device(args.device)

    # ── Load Questions ────────────────────────────────────────────────
    questions = []
    with open(args.question_file) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    questions = questions[:args.n_samples]
    print(f"[DEBUG] Loaded {len(questions)} questions from {args.question_file}")

    # ── Per-sample Inference ──────────────────────────────────────────
    # head_vis_ratios[(l,h)] = list of vis_ratio values per sample
    head_vis_ratios = defaultdict(list)
    # model_answer_is_yes[i] = bool : model said "yes" for this sample
    sample_meta = []  # {gt_label, model_yes}

    t0 = time.time()
    for idx, q in enumerate(questions):
        text = q["text"]
        image_file = q["image"]
        gt_label = q["label"].strip().lower()  # "yes" or "no"

        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        try:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            img_t = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(device).half()
        except Exception as e:
            print(f"[WARN] Image load error {image_file}: {e}")
            continue

        with torch.inference_mode():
            pf = model(input_ids[:, :-1], images=img_t.unsqueeze(0),
                       use_cache=True, return_dict=True)
            out = model(input_ids[:, -1:],
                        attention_mask=torch.ones((1, 1), dtype=torch.long, device=device),
                        past_key_values=pf.past_key_values,
                        use_cache=True, output_attentions=True,
                        return_dict=True)
            # Generate single token to get model's answer
            gen_ids = model.generate(
                input_ids,
                images=img_t.unsqueeze(0),
                max_new_tokens=8,
                do_sample=False,
            )
        gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        model_yes = gen_text.startswith("yes")

        pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        img_start = int(pos[0].item())
        img_end = img_start + 576

        # Collect vis_ratio per head
        for li, attn in enumerate(out.attentions):
            if li < args.late_start or li > args.late_end or attn is None:
                continue
            row = attn[0, :, -1, :].float()  # [H, Seq]
            vis = row[:, img_start:img_end].sum(-1)
            txt = row[:, :img_start].sum(-1) + row[:, img_end:].sum(-1)
            vr = vis / torch.clamp(vis + txt, min=1e-6)
            for hi in range(vr.size(0)):
                head_vis_ratios[(li, hi)].append(float(vr[hi].item()))

        sample_meta.append({"gt": gt_label, "model_yes": model_yes, "idx": idx})

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"[DEBUG] {idx+1}/{len(questions)} elapsed={elapsed:.1f}s")

    print(f"\n[DEBUG] Completed inference on {len(sample_meta)} samples")

    # ── AUC Computation ───────────────────────────────────────────────
    # Hallucination events: model_yes=True AND gt_label="no"  → FP (hallucination)
    # Correct events: model_yes=True AND gt_label="yes"       → TP (faithful)
    # We only score on samples where model said "yes"
    yes_mask = [m["model_yes"] for m in sample_meta]
    yes_indices = [i for i, m in enumerate(sample_meta) if m["model_yes"]]

    # label for AUC: 1 = TP (gt=yes), 0 = FP (gt=no) among model_yes samples
    labels_binary = [1 if sample_meta[i]["gt"] == "yes" else 0 for i in yes_indices]

    n_yes = len(yes_indices)
    n_tp = sum(labels_binary)
    n_fp = n_yes - n_tp
    print(f"[DEBUG] Model said 'yes' on {n_yes} samples: TP={n_tp}, FP={n_fp}")

    if n_tp == 0 or n_fp == 0:
        print("[ERROR] Cannot compute AUC: need both TP and FP samples. Try larger n_samples.")
        return

    auc_rows = []
    for (li, hi), vis_list in head_vis_ratios.items():
        # Subset to model_yes samples
        yes_vis = [vis_list[i] for i in yes_indices if i < len(vis_list)]
        if len(yes_vis) != n_yes:
            continue

        auc_val = roc_auc_score(labels_binary, yes_vis)
        pb_r, pb_p = pointbiserialr(labels_binary, yes_vis)
        mean_tp = _mean([yes_vis[i] for i, l in enumerate(labels_binary) if l == 1])
        mean_fp = _mean([yes_vis[i] for i, l in enumerate(labels_binary) if l == 0])

        # Direction: does vis_ratio go DOWN during hallucination (FP)?
        # AUC > 0.5 means vis_ratio is HIGHER in TP → lower in hallucination
        if auc_val > 0.5:
            direction = "lower_in_hallucination"
            auc_best = auc_val
        else:
            direction = "higher_in_hallucination"
            auc_best = 1.0 - auc_val

        auc_rows.append({
            "layer": li, "head": hi,
            "raw_auc": auc_val,
            "auc_best_dir": auc_best,
            "direction": direction,
            "pb_r": pb_r,
            "mean_tp": mean_tp,
            "mean_fp": mean_fp,
            "diff_tp_fp": mean_tp - mean_fp,
        })

    auc_rows.sort(key=lambda r: -r["auc_best_dir"])

    # Save head AUC CSV
    csv_path = os.path.join(args.out_dir, "discovery_head_auc.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer", "head", "raw_auc", "auc_best_dir", "direction", "pb_r", "mean_tp", "mean_fp", "diff_tp_fp"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(auc_rows)
    print(f"[DEBUG] Head AUC CSV saved to {csv_path}")

    # ── Select Faithful and Harmful Heads ─────────────────────────────
    faithful_heads = []
    seen = set()
    for r in auc_rows:
        if r["direction"] != "lower_in_hallucination":
            continue
        if r["auc_best_dir"] < args.auc_min:
            continue
        key = (r["layer"], r["head"])
        if key in seen:
            continue
        seen.add(key)
        faithful_heads.append(r)
        if len(faithful_heads) >= args.top_k:
            break

    harmful_heads = []
    seen = set()
    for r in auc_rows:
        if r["direction"] != "higher_in_hallucination":
            continue
        if r["auc_best_dir"] < args.auc_min:
            continue
        key = (r["layer"], r["head"])
        if key in seen:
            continue
        seen.add(key)
        harmful_heads.append(r)
        if len(harmful_heads) >= args.top_k:
            break

    # ── Console Report ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"DISCOVERY HEADSET  (source: {os.path.basename(args.question_file)})")
    print(f"{'='*70}")
    print(f"  FAITHFUL HEADS ({len(faithful_heads)}):")
    for i, r in enumerate(faithful_heads):
        print(f"    {i+1:>2}. L{r['layer']:>2} H{r['head']:>2}  AUC={r['auc_best_dir']:.4f}  mean_TP={r['mean_tp']:.3f}  mean_FP={r['mean_fp']:.3f}")
    print(f"\n  HARMFUL HEADS ({len(harmful_heads)}):")
    for i, r in enumerate(harmful_heads):
        print(f"    {i+1:>2}. L{r['layer']:>2} H{r['head']:>2}  AUC={r['auc_best_dir']:.4f}  mean_TP={r['mean_tp']:.3f}  mean_FP={r['mean_fp']:.3f}")

    # ── Compare with Original Static Headset ─────────────────────────
    try:
        static_path = "/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json"
        with open(static_path) as f:
            st = json.load(f)
        st_f = set((int(x["layer"]), int(x["head"])) for x in st.get("faithful_heads", []))
        st_h = set((int(x["layer"]), int(x["head"])) for x in st.get("harmful_heads", []))
        disc_f = set((r["layer"], r["head"]) for r in faithful_heads)
        disc_h = set((r["layer"], r["head"]) for r in harmful_heads)
        ov_f = len(st_f & disc_f)
        ov_h = len(st_h & disc_h)
        jac_f = ov_f / len(st_f | disc_f) if (st_f | disc_f) else 0.0
        jac_h = ov_h / len(st_h | disc_h) if (st_h | disc_h) else 0.0
        print(f"\n  Overlap with static headset:")
        print(f"    Faithful  overlap={ov_f}/{len(st_f)}  Jaccard={jac_f:.3f}")
        print(f"    Harmful   overlap={ov_h}/{len(st_h)}  Jaccard={jac_h:.3f}")
    except Exception:
        pass

    # ── Save Headset JSON ─────────────────────────────────────────────
    headset_payload = {
        "source": "coco_train2014_pope_style_discovery",
        "discovery_file": args.question_file,
        "faithful_heads": [{"layer": r["layer"], "head": r["head"]} for r in faithful_heads],
        "harmful_heads": [{"layer": r["layer"], "head": r["head"]} for r in harmful_heads],
        "faithful_head_specs": [f"{r['layer']}:{r['head']}" for r in faithful_heads],
        "harmful_head_specs": [f"{r['layer']}:{r['head']}" for r in harmful_heads],
        "faithful_heads_by_layer": {},
        "harmful_heads_by_layer": {},
    }
    for r in faithful_heads:
        headset_payload["faithful_heads_by_layer"].setdefault(str(r["layer"]), []).append(r["head"])
    for r in harmful_heads:
        headset_payload["harmful_heads_by_layer"].setdefault(str(r["layer"]), []).append(r["head"])

    out_json = os.path.join(args.out_dir, "discovery_headset.json")
    with open(out_json, "w") as f:
        json.dump(headset_payload, f, indent=2)
    print(f"\n[DEBUG] Discovery headset saved to {out_json}")
    print("[DEBUG] Done.")


if __name__ == "__main__":
    main()
