#!/usr/bin/env python
"""
Quantitatively profile the attention behavior of heads across POPE samples.
We calculate:
1. Spatial Consensus: How much a head's 2d spatial attention correlates
   with the 'mean late-layer attention' (proxy for salient object).
2. Semantic Target: When looking at text, what % of attention goes to
   the Question vs the System Prompt?
3. FP vs TP Shift: How these metrics change when predicting correctly (TP)
   vs hallucinating (FP).
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

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
    ap.add_argument("--image_folder", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--question_file", type=str, default="/home/kms/VISTA/pope_coco/coco_pope_random.json")
    ap.add_argument("--headset_json", type=str, default="/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--out_dir", type=str, default="/home/kms/LLaVA_calibration/experiments/attention_quant")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

    # ── load static headset ─────────────────────────────────────
    hs = json.loads(Path(args.headset_json).read_text())
    st_f = set((int(x["layer"]), int(x["head"])) for x in hs.get("faithful_heads", []))
    st_h = set((int(x["layer"]), int(x["head"])) for x in hs.get("harmful_heads", []))
    
    # ── load questions ──────────────────────────────────────────
    questions = []
    with open(args.question_file, "r") as f:
        for line in f:
            if line.strip(): questions.append(json.loads(line.strip()))
    questions = questions[:args.n_samples]

    conv_mode = "llava_v1"
    
    head_stats = defaultdict(lambda: defaultdict(list))
    
    t0 = time.time()
    for idx, q in enumerate(questions):
        text = q.get("text", "")
        img_file = q.get("image", "")
        gt_label = q.get("label", "").lower()
        
        # Exact prompt construction to find system vs question tokens
        conv = conv_templates[conv_mode].copy()
        sys_len = len(tokenizer(conv.system).input_ids) # approx sys prompt len
        
        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
        pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        if pos.numel() == 0: continue
        img_start = int(pos[0].item())
        img_end = img_start + 576
        
        image = Image.open(os.path.join(args.image_folder, img_file)).convert("RGB")
        img_t = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(device).half()
        
        with torch.inference_mode():
            prefill = model(input_ids[:, :-1], images=img_t.unsqueeze(0), use_cache=True, return_dict=True)
            outputs = model(input_ids[:, -1:],
                            attention_mask=torch.ones((1, 1), dtype=torch.long, device=device),
                            past_key_values=prefill.past_key_values,
                            use_cache=True, output_attentions=True, return_dict=True)
            pred_id = outputs.logits[0, -1].argmax().item()
            pred_text = tokenizer.decode([pred_id]).strip().lower()
            
            # Determine TP / FP / TN / FN
            is_yes = pred_text.startswith("yes")
            is_no = pred_text.startswith("no")
            if is_yes and gt_label == "yes": res_type = "TP"
            elif is_yes and gt_label == "no": res_type = "FP"  # Hallucination!
            elif is_no and gt_label == "no": res_type = "TN"
            elif is_no and gt_label == "yes": res_type = "FN"
            else: res_type = "OTHER"

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
                
                # Spatial Consensus: correlation with global salient map
                h_spatial = h_row[img_start:img_end]
                if h_spatial.sum() > 1e-6:
                    h_spatial_norm = h_spatial / h_spatial.sum()
                    pearson = torch.corrcoef(torch.stack([h_spatial_norm, global_spatial]))[0, 1].item()
                    if not np.isnan(pearson):
                        head_stats[(l,h)][f"spatial_consensus_{res_type}"].append(pearson)
                        head_stats[(l,h)]["spatial_consensus_ALL"].append(pearson)
                
                # Semantic Targets
                txt_sys = h_row[:img_start][:sys_len].sum().item()
                txt_q = h_row[:img_start][sys_len:].sum().item() + h_row[img_end:].sum().item()
                total_txt = txt_sys + txt_q
                
                if total_txt > 1e-6:
                    q_ratio = txt_q / total_txt
                    head_stats[(l,h)][f"text_q_ratio_{res_type}"].append(q_ratio)
                    head_stats[(l,h)]["text_q_ratio_ALL"].append(q_ratio)
        
        if (idx+1) % 50 == 0:
            print(f"[DEBUG] {idx+1}/{args.n_samples} elapsed={time.time()-t0:.1f}s")
            
    # ── Aggregate and Print ─────────────────────────────────────
    rows = []
    for lh, stats in head_stats.items():
        role = "FAITHFUL" if lh in st_f else "HARMFUL" if lh in st_h else "other"
        r = {"layer": lh[0], "head": lh[1], "role": role}
        
        r["sp_cons_ALL"] = _mean(stats["spatial_consensus_ALL"])
        r["sp_cons_TP"] = _mean(stats["spatial_consensus_TP"])
        r["sp_cons_FP"] = _mean(stats["spatial_consensus_FP"])
        r["sp_cons_drop"] = r["sp_cons_TP"] - r["sp_cons_FP"] # Positive means consensus drops when hallucinating
        
        r["txt_q_ratio_ALL"] = _mean(stats["text_q_ratio_ALL"])
        r["txt_q_ratio_TP"] = _mean(stats["text_q_ratio_TP"])
        r["txt_q_ratio_FP"] = _mean(stats["text_q_ratio_FP"])
        r["txt_q_ratio_diff"] = r["txt_q_ratio_TP"] - r["txt_q_ratio_FP"]
        
        rows.append(r)
        
    print("\n" + "="*90)
    print("QUANTITATIVE PROFILING: FAITHFUL vs HARMFUL vs OTHER")
    print("="*90)
    
    for r_name in ["FAITHFUL", "HARMFUL", "other"]:
        sub = [r for r in rows if r["role"] == r_name]
        if not sub: continue
        print(f"\n[{r_name}] (n={len(sub)} heads)")
        
        sp_all = np.mean([x["sp_cons_ALL"] for x in sub])
        sp_drop = np.mean([x["sp_cons_drop"] for x in sub])
        print(f"  Spatial Consensus (Alignment w/ Global Salience): {sp_all:.4f}")
        print(f"  Spatial Consensus Drop (TP -> FP)               : {sp_drop:.4f}  (>0 means loses focus during hallucination)")
        
        txt_all = np.mean([x["txt_q_ratio_ALL"] for x in sub])
        txt_diff = np.mean([x["txt_q_ratio_diff"] for x in sub])
        print(f"  Text Target Ratio (% attention to Question)     : {txt_all:.4f}")
        print(f"  Text Target Diff (TP - FP)                      : {txt_diff:.4f}")

    # Output detailed faithful 
    print("\n" + "-"*90)
    print("DETAILED FAITHFUL HEADS")
    print(f"{'L':>3} {'H':>3} | {'Sp_Cons':>7} | {'Sp_Drop_FP':>10} | {'Txt_Q_Ratio':>11} | {'Txt_Q_Diff_FP':>13}")
    for r in sorted([x for x in rows if x["role"] == "FAITHFUL"], key=lambda x: -x["sp_cons_drop"]):
        print(f"{r['layer']:>3} {r['head']:>3} | {r['sp_cons_ALL']:7.4f} | {r['sp_cons_drop']:10.4f} | {r['txt_q_ratio_ALL']:11.4f} | {r['txt_q_ratio_diff']:13.4f}")

    csv_path = os.path.join(args.out_dir, "quant_profile.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[DEBUG] Saved to {csv_path}")

if __name__ == "__main__":
    main()
