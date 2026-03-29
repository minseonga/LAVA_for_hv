#!/usr/bin/env python
"""
Visualize the spatial attention maps (24x24 grid over 576 image tokens)
for static faithful vs harmful heads across a few selected POPE samples.

We will extract the attention weights of the LAST token (the predicted token)
over the 576 image tokens, reshape them to 24x24, and overlay them on the
original image.
"""

import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path


def _overlay_attention_map(image: Image.Image, attn_map: np.ndarray, alpha=0.5):
    """
    Overlay a 24x24 attention map onto the original PIL image.
    1. Resize attn_map to match image size using bicubic interpolation.
    2. Apply a colormap (e.g., jet).
    3. Blend with the original image.
    """
    import cv2
    img_np = np.array(image.convert("RGB"))
    h, w = img_np.shape[:2]

    # Normalize attention map to [0, 1]
    attn_min, attn_max = attn_map.min(), attn_map.max()
    if attn_max > attn_min:
        attn_map = (attn_map - attn_min) / (attn_max - attn_min)
    else:
        attn_map = np.zeros_like(attn_map)

    # Resize attention map
    attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply colormap (Jet)
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return blended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vga_root", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--image_folder", type=str, default="/home/kms/data/pope/val2014")
    ap.add_argument("--question_file", type=str, default="/home/kms/VISTA/pope_coco/coco_pope_random.json")
    ap.add_argument("--headset_json", type=str, default="/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json")
    ap.add_argument("--n_samples", type=int, default=10, help="Number of samples to visualize")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--out_dir", type=str, default="/home/kms/LLaVA_calibration/experiments/attention_viz")
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
    static_faithful = [(int(x["layer"]), int(x["head"])) for x in hs.get("faithful_heads", [])]
    static_harmful = [(int(x["layer"]), int(x["head"])) for x in hs.get("harmful_heads", [])]
    print(f"[DEBUG] Static headset: {len(static_faithful)} faithful, {len(static_harmful)} harmful")

    # Select top 5 of each for visualization to keep the plot manageable
    top_f = static_faithful[:4]
    top_h = static_harmful[:4]

    # ── load questions ──────────────────────────────────────────
    questions = []
    with open(args.question_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    # Select a mix of yes/no
    selected_qs = []
    yes_seen = 0
    no_seen = 0
    for q in questions:
        lbl = q.get("label", "").lower()
        if lbl == "yes" and yes_seen < args.n_samples // 2:
            selected_qs.append(q)
            yes_seen += 1
        elif lbl == "no" and no_seen < args.n_samples // 2:
            selected_qs.append(q)
            no_seen += 1
        if len(selected_qs) >= args.n_samples:
            break
    
    print(f"[DEBUG] Selected {len(selected_qs)} questions for visualization")

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

    # ── inference loop ──────────────────────────────────────────
    for idx, q in enumerate(selected_qs):
        question_text = q.get("text", q.get("question", ""))
        image_file = q.get("image", "")
        label = q.get("label", "")
        qid = q.get("question_id", idx)

        prompt = build_prompt(question_text)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        img_path = os.path.join(args.image_folder, image_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Could not load image {img_path}: {e}")
            continue
            
        real_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(device)
        if device.type == "cuda":
            real_tensor = real_tensor.half()

        # Find image token position
        pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
        img_start = int(pos[0].item())
        img_end = img_start + 576

        with torch.inference_mode():
            prefill = model(
                input_ids[:, :-1],
                images=real_tensor.unsqueeze(0),
                use_cache=True,
                return_dict=True,
            )
            # Generation pass for the first token
            outputs = model(
                input_ids[:, -1:],
                attention_mask=torch.ones((1, 1), dtype=torch.long, device=device),
                past_key_values=prefill.past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            pred_id = outputs.logits[0, -1].argmax().item()
            pred_token = tokenizer.decode([pred_id])

        attentions = outputs.attentions  # Tuple of [1, 32, 1, seq_len]

        # ── Plotting ─────────────────────────────────────────────
        # We will plot: [Original Image] | [Mean Faithful] | [F1] | [F2] | [F3] | [F4]
        #               [Original Image] | [Mean Harmful]  | [H1] | [H2] | [H3] | [H4]
        
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        fig.suptitle(f"Q: {question_text} | GT: {label} | Pred: {pred_token.strip()}", fontsize=16)

        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image (Faithful)")
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(image)
        axes[1, 0].set_title("Original Image (Harmful)")
        axes[1, 0].axis('off')

        # Helper to extract and reshape 24x24 attention
        def get_attn_24x24(layer, head):
            attn = attentions[layer][0, head, -1, img_start:img_end].float().cpu().numpy()
            return attn.reshape(24, 24)

        # Plot Faithful
        f_maps = []
        for i, (l, h) in enumerate(top_f):
            amap = get_attn_24x24(l, h)
            f_maps.append(amap)
            blended = _overlay_attention_map(image, amap)
            axes[0, i+2].imshow(blended)
            axes[0, i+2].set_title(f"Faithful L{l}H{h}")
            axes[0, i+2].axis('off')
            
        # Mean Faithful
        mean_f_map = np.mean(f_maps, axis=0)
        axes[0, 1].imshow(_overlay_attention_map(image, mean_f_map))
        axes[0, 1].set_title("Mean Top-4 Faithful")
        axes[0, 1].axis('off')

        # Plot Harmful
        h_maps = []
        for i, (l, h) in enumerate(top_h):
            amap = get_attn_24x24(l, h)
            h_maps.append(amap)
            blended = _overlay_attention_map(image, amap)
            axes[1, i+2].imshow(blended)
            axes[1, i+2].set_title(f"Harmful L{l}H{h}")
            axes[1, i+2].axis('off')
            
        # Mean Harmful
        mean_h_map = np.mean(h_maps, axis=0)
        axes[1, 1].imshow(_overlay_attention_map(image, mean_h_map))
        axes[1, 1].set_title("Mean Top-4 Harmful")
        axes[1, 1].axis('off')

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"q{qid}_{label}_{pred_token.strip()}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        
        print(f"[DEBUG] Saved {out_path}")

    print(f"\n[DEBUG] Visualization complete. Check {args.out_dir}")

if __name__ == "__main__":
    main()
