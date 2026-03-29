#!/usr/bin/env python
"""
Extract a 100% GT-Free, Unsupervised "Faithful" Headset using OOD images.
Dataset: COCO train2014 (Disjoint from POPE evaluation images).
Prompt: "Describe this image in detail." (No QA formatting).

We extract 3 structural features for each head in layers 16-24 over 500 natural images:
1. CV (Coefficient of Variation) of vis_ratio -> High is better
2. Spatial Consensus (Alignment with mean global salience) -> High is better
3. Text Target Ratio (Attention given to User Prompt vs System Prompt) -> Low is better

We assign a robust rank (1 to N) for each metric, sum the ranks, and
pick the top-K heads (lowest rank sum) as the 'Unsupervised Faithful Headset'.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

def _mean(vals):
    vals = list(vals)
    return float(sum(vals)/len(vals)) if vals else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vga_root", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--image_folder", type=str, default="/home/kms/data/images/mscoco/images/train2014")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--out_dir", type=str, default="/home/kms/LLaVA_calibration/experiments/headsets_unsupervised")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # ── bootstrap VGA ───────────────────────────────────────────
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

    evolve_greedy_sampling(); set_seed(42)

    print("[DEBUG] Loading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        os.path.expanduser(args.model_path), None, model_name, device=args.device)
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head
    device = torch.device(args.device)

    # ── 1. Select Random OOD Images ─────────────────────────────
    all_images = [f for f in os.listdir(args.image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_images.sort()
    selected_images = random.sample(all_images, min(args.n_samples, len(all_images)))
    print(f"[DEBUG] Selected {len(selected_images)} random images from {args.image_folder}")

    # The single generic extraction prompt
    generic_question = "Describe this image in detail."
    conv_mode = "llava_v1"
    
    # Store per-sample metrics to compute CV later
    # Format: stats[(l,h)][param_name] = [list of sample values]
    head_stats = defaultdict(lambda: defaultdict(list))
    
    # Pre-compute system prompt length
    conv = conv_templates[conv_mode].copy()
    sys_len = len(tokenizer(conv.system).input_ids)

    print("[DEBUG] Extracting unsupervised features over samples...")
    t0 = time.time()
    
    valid_samples = 0
    for idx, img_file in enumerate(selected_images):
        qs = DEFAULT_IMAGE_TOKEN + "\n" + generic_question
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
        pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        if pos.numel() == 0: continue
        img_start = int(pos[0].item())
        img_end = img_start + 576
        
        try:
            image = Image.open(os.path.join(args.image_folder, img_file)).convert("RGB")
            img_t = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(device).half()
        except Exception as e:
            continue
        
        with torch.inference_mode():
            prefill = model(input_ids[:, :-1], images=img_t.unsqueeze(0), use_cache=True, return_dict=True)
            outputs = model(input_ids[:, -1:],
                            attention_mask=torch.ones((1, 1), dtype=torch.long, device=device),
                            past_key_values=prefill.past_key_values,
                            use_cache=True, output_attentions=True, return_dict=True)
        
        attns = outputs.attentions
        
        # 1. Compute global salient spatial map (mean of all late-layer heads)
        all_spatial = []
        for l in range(args.late_start, args.late_end+1):
            if attns[l] is None: continue
            sp = attns[l][0, :, -1, img_start:img_end].float() # [H, 576]
            all_spatial.append(sp)
        if not all_spatial: continue
        global_spatial = torch.cat(all_spatial, dim=0).mean(dim=0) # [576]
        global_spatial = global_spatial / (global_spatial.sum() + 1e-6)
        
        # 2. Extract per-head metrics
        for l in range(args.late_start, args.late_end+1):
            if attns[l] is None: continue
            row = attns[l][0, :, -1, :].float() # [H, Seq]
            for h in range(row.size(0)):
                h_row = row[h]
                
                # Metric 1: vis_ratio (to compute CV later)
                h_img_sum = h_row[img_start:img_end].sum().item()
                h_txt_sum = h_row[:img_start].sum().item() + h_row[img_end:].sum().item()
                vis_ratio = h_img_sum / max(h_img_sum + h_txt_sum, 1e-6)
                head_stats[(l,h)]["vis_ratio"].append(vis_ratio)
                
                # Metric 2: Spatial Consensus
                h_spatial = h_row[img_start:img_end]
                if h_spatial.sum() > 1e-6:
                    h_spatial_norm = h_spatial / h_spatial.sum()
                    pearson = torch.corrcoef(torch.stack([h_spatial_norm, global_spatial]))[0, 1].item()
                    if not np.isnan(pearson):
                        head_stats[(l,h)]["spatial_consensus"].append(pearson)
                
                # Metric 3: Text Q Ratio (Attention to User Prompt vs SYS)
                txt_sys = h_row[:img_start][:sys_len].sum().item()
                txt_q = h_row[:img_start][sys_len:].sum().item() + h_row[img_end:].sum().item()
                total_txt = txt_sys + txt_q
                if total_txt > 1e-6:
                    q_ratio = txt_q / total_txt
                    head_stats[(l,h)]["text_q_ratio"].append(q_ratio)
        
        valid_samples += 1
        if valid_samples % 50 == 0:
            print(f"[DEBUG] {valid_samples}/{args.n_samples} elapsed={time.time()-t0:.1f}s")
            
    # ── 3. Aggregate Features and Compute Ensemble Rank ────────────────
    print(f"\\n[DEBUG] Aggregating features and ranking...")
    
    rows = []
    for lh, stats in head_stats.items():
        vr_array = np.array(stats["vis_ratio"])
        vr_mean = np.mean(vr_array)
        vr_std = np.std(vr_array)
        vr_cv = (vr_std / vr_mean) if vr_mean > 1e-6 else 0.0
        
        sp_cons = _mean(stats["spatial_consensus"])
        txt_q_ratio = _mean(stats["text_q_ratio"])
        
        rows.append({
            "layer": lh[0], "head": lh[1],
            "cv": vr_cv,
            "sp_cons": sp_cons,
            "txt_q_ratio": txt_q_ratio
        })
        
    # Rank them
    # cv -> higher is better (descending)
    rows.sort(key=lambda r: -r["cv"])
    for i, r in enumerate(rows): r["rank_cv"] = i + 1

    # sp_cons -> higher is better (descending)
    rows.sort(key=lambda r: -r["sp_cons"])
    for i, r in enumerate(rows): r["rank_sp_cons"] = i + 1

    # txt_q_ratio -> lower is better (ascending)
    rows.sort(key=lambda r: r["txt_q_ratio"])
    for i, r in enumerate(rows): r["rank_txt_ratio"] = i + 1
    
    # Ensemble Rank Score (Sum of Ranks) -> lower is better
    for r in rows:
        r["rank_sum"] = r["rank_cv"] + r["rank_sp_cons"] + r["rank_txt_ratio"]
        
    # Final Sorting
    rows.sort(key=lambda r: r["rank_sum"])
    
    # ── 4. Print and Save the Target Headset ──────────────────────────
    print(f"\\n{'='*90}")
    print(f"UNSUPERVISED GT-FREE HEADS (Extracted from COCO train2014, Prompt='Describe this image')")
    print(f"{'='*90}")
    print(f"{'Rank':>4} | {'L':>3} {'H':>3} | {'RankSum':>7} | {'Rank_CV':>7} | {'Rank_Sp':>7} | {'Rank_Tx':>7} || {'CV':>6} | {'Sp_Cons':>7} | {'Txt_Q_Rat':>10}")
    
    faithful_heads = []
    for i, r in enumerate(rows[:args.top_k]):
        print(f"{i+1:>4} | {r['layer']:>3} {r['head']:>3} | {r['rank_sum']:>7} | {r['rank_cv']:>7} | {r['rank_sp_cons']:>7} | {r['rank_txt_ratio']:>7} || {r['cv']:6.3f} | {r['sp_cons']:7.4f} | {r['txt_q_ratio']:10.4f}")
        faithful_heads.append({"layer": r["layer"], "head": r["head"]})
        
    out_json = os.path.join(args.out_dir, "unsupervised_headset.json")
    out_csv = os.path.join(args.out_dir, "head_ranks.csv")
    
    with open(out_json, "w") as f:
        json.dump({"faithful_heads": faithful_heads, "harmful_heads": [], "source": "coco_train2014_unsupervised"}, f, indent=4)
        
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        
    print(f"\\n[DEBUG] Headset saved to {out_json}")
    
    # Quick sanity check with previous static headset
    # See how many overlap
    try:
        static_path = "/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json"
        with open(static_path) as f:
            st = json.load(f)
            st_f = set((int(x["layer"]), int(x["head"])) for x in st.get("faithful_heads", []))
            un_f = set((int(x["layer"]), int(x["head"])) for x in faithful_heads)
            overlap = len(st_f & un_f)
            print(f"[DEBUG] Overlap with GT-based static faithful heads: {overlap}/{len(st_f)} (Jaccard: {overlap/len(st_f | un_f):.3f})")
    except Exception:
        pass

if __name__ == "__main__":
    main()
