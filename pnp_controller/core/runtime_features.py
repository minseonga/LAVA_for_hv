from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


def normalize_head_map(raw: object) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if not isinstance(raw, Mapping):
        return out
    for k, v in raw.items():
        try:
            layer = int(k)
        except Exception:
            continue
        heads: List[int] = []
        if isinstance(v, str):
            parts = [x.strip() for x in v.split(",") if x.strip() != ""]
            for x in parts:
                try:
                    heads.append(int(x))
                except Exception:
                    continue
        elif isinstance(v, Sequence):
            for x in v:
                try:
                    heads.append(int(x))
                except Exception:
                    continue
        if heads:
            out[layer] = sorted(set(heads))
    return out


def image_span_from_prompt_input_ids(input_ids: torch.Tensor, image_token_index: int, image_token_count: int) -> Tuple[int, int]:
    if input_ids.dim() != 2 or int(input_ids.size(0)) != 1:
        raise ValueError(f"Expected input_ids shape [1, T], got {tuple(input_ids.shape)}")
    pos = (input_ids[0] == int(image_token_index)).nonzero(as_tuple=False)
    if int(pos.numel()) == 0:
        raise ValueError("No image token placeholder found in prompt input_ids.")
    if int(pos.size(0)) != 1:
        raise ValueError("Multiple image token placeholders are not supported in the current online adapter.")
    image_start = int(pos[0].item())
    image_end = image_start + int(image_token_count)
    return image_start, image_end


def topk_mass(values: torch.Tensor, k: int) -> float:
    if values.numel() == 0:
        return 0.0
    kk = max(1, min(int(k), int(values.numel())))
    return float(torch.topk(values, kk).values.sum().item())


def _mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _topk_mean(vals: Iterable[float], k_ratio: float = 0.2) -> float:
    vals = list(vals)
    if not vals:
        return 0.0
    kk = max(1, min(len(vals), int(round(len(vals) * float(k_ratio)))))
    vals = sorted(vals, reverse=True)
    return float(sum(vals[:kk]) / float(kk))


def compute_head_attn_vis_ratio_last_row(
    attentions: Sequence[torch.Tensor],
    image_start: int,
    image_end: int,
    late_start: int,
    late_end: int,
    faithful_heads_by_layer: Mapping[int, Sequence[int]],
    harmful_heads_by_layer: Mapping[int, Sequence[int]],
    eps: float = 1e-6,
) -> Dict[str, float]:
    faithful_heads_by_layer = normalize_head_map(faithful_heads_by_layer)
    harmful_heads_by_layer = normalize_head_map(harmful_heads_by_layer)

    global_vals: List[float] = []
    faithful_vals: List[float] = []
    harmful_vals: List[float] = []

    per_layer: Dict[str, Dict[str, float]] = {}
    for layer_idx, attn in enumerate(attentions):
        if layer_idx < int(late_start) or layer_idx > int(late_end):
            continue
        if attn is None:
            continue
        if attn.dim() != 4 or int(attn.size(0)) != 1:
            raise ValueError(f"Expected attention tensor [1,H,Q,K], got {tuple(attn.shape)} at layer {layer_idx}")
        row = attn[0, :, -1, :].to(torch.float32)
        if image_end > int(row.size(-1)):
            raise ValueError(
                f"Image span [{image_start}, {image_end}) exceeds attention width {int(row.size(-1))} at layer {layer_idx}"
            )
        vis_sum = row[:, image_start:image_end].sum(dim=-1)
        txt_left = row[:, :image_start].sum(dim=-1)
        txt_right = row[:, image_end:].sum(dim=-1)
        txt_sum = txt_left + txt_right
        vis_ratio = vis_sum / torch.clamp(vis_sum + txt_sum, min=float(eps))
        vis_ratio_list = [float(x.item()) for x in vis_ratio]
        global_vals.extend(vis_ratio_list)

        faithful_heads = faithful_heads_by_layer.get(int(layer_idx), [])
        harmful_heads = harmful_heads_by_layer.get(int(layer_idx), [])
        for h in faithful_heads:
            if 0 <= int(h) < len(vis_ratio_list):
                faithful_vals.append(vis_ratio_list[int(h)])
        for h in harmful_heads:
            if 0 <= int(h) < len(vis_ratio_list):
                harmful_vals.append(vis_ratio_list[int(h)])

        per_layer[str(layer_idx)] = {
            "global_mean": _mean(vis_ratio_list),
            "faithful_mean": _mean(vis_ratio_list[int(h)] for h in faithful_heads if 0 <= int(h) < len(vis_ratio_list)),
            "harmful_mean": _mean(vis_ratio_list[int(h)] for h in harmful_heads if 0 <= int(h) < len(vis_ratio_list)),
        }

    faithful_mean = _mean(faithful_vals)
    harmful_mean = _mean(harmful_vals)
    global_mean = _mean(global_vals)
    return {
        "faithful_head_attn_mean": faithful_mean,
        "harmful_head_attn_mean": harmful_mean,
        "global_late_head_attn_mean": global_mean,
        "faithful_minus_global_attn": float(faithful_mean - global_mean),
        "guidance_mismatch_score": float((harmful_mean - faithful_mean)),
        "n_global_points": float(len(global_vals)),
        "n_faithful_points": float(len(faithful_vals)),
        "n_harmful_points": float(len(harmful_vals)),
        "per_layer": per_layer,
    }


def combine_gmi_with_guidance_mass(
    faithful_head_attn_mean: float,
    harmful_head_attn_mean: float,
    g_top5_mass: float,
) -> float:
    faithful_on_g = float(faithful_head_attn_mean) * float(g_top5_mass)
    harmful_on_g = float(harmful_head_attn_mean) * float(g_top5_mass)
    return float(harmful_on_g - faithful_on_g)


def compute_mean_image_attention_distribution_last_row(
    attentions: Sequence[torch.Tensor],
    image_start: int,
    image_end: int,
    late_start: int,
    late_end: int,
    eps: float = 1e-6,
) -> Dict[str, object]:
    alpha_rows: List[torch.Tensor] = []
    vis_ratio_vals: List[float] = []
    per_layer: Dict[str, Dict[str, float]] = {}

    for layer_idx, attn in enumerate(attentions):
        if layer_idx < int(late_start) or layer_idx > int(late_end):
            continue
        if attn is None:
            continue
        if attn.dim() != 4 or int(attn.size(0)) != 1:
            raise ValueError(f"Expected attention tensor [1,H,Q,K], got {tuple(attn.shape)} at layer {layer_idx}")

        row = attn[0, :, -1, :].to(torch.float32)
        if image_end > int(row.size(-1)):
            raise ValueError(
                f"Image span [{image_start}, {image_end}) exceeds attention width {int(row.size(-1))} at layer {layer_idx}"
            )
        img = row[:, image_start:image_end]
        img_mass = img.sum(dim=-1)
        total_mass = row.sum(dim=-1)
        vis_ratio = img_mass / torch.clamp(total_mass, min=float(eps))
        vis_ratio_list = [float(x.item()) for x in vis_ratio]
        vis_ratio_vals.extend(vis_ratio_list)

        valid = img_mass > float(eps)
        n_valid = int(valid.sum().item())
        if n_valid > 0:
            alpha_img = img[valid] / torch.clamp(img_mass[valid].unsqueeze(-1), min=float(eps))
            alpha_rows.append(alpha_img)
        per_layer[str(layer_idx)] = {
            "global_mean_vis_ratio": _mean(vis_ratio_list),
            "valid_head_fraction": float(n_valid / max(1, int(img.size(0)))),
            "n_valid_heads": float(n_valid),
        }

    k_img = max(0, int(image_end) - int(image_start))
    if alpha_rows:
        alpha_mean = torch.cat(alpha_rows, dim=0).mean(dim=0)
    elif k_img > 0:
        alpha_mean = torch.full((k_img,), 1.0 / float(k_img), dtype=torch.float32)
    else:
        alpha_mean = torch.zeros((0,), dtype=torch.float32)

    if alpha_mean.numel() > 0:
        alpha_mean = alpha_mean / torch.clamp(alpha_mean.sum(), min=float(eps))

    return {
        "alpha_img_mean": alpha_mean,
        "late_head_vis_ratio_mean": _mean(vis_ratio_vals),
        "late_head_vis_ratio_topkmean": _topk_mean(vis_ratio_vals, k_ratio=0.2),
        "late_head_count": float(len(vis_ratio_vals)),
        "per_layer": per_layer,
    }


def cosine_mismatch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    p = p.to(torch.float32).flatten()
    q = q.to(torch.float32).flatten()
    if p.numel() != q.numel():
        raise ValueError(f"cosine_mismatch requires same shape, got {tuple(p.shape)} vs {tuple(q.shape)}")
    if p.numel() == 0:
        return 0.0
    p = p / torch.clamp(p.sum(), min=float(eps))
    q = q / torch.clamp(q.sum(), min=float(eps))
    cos = F.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0), dim=-1, eps=float(eps))
    return float(1.0 - float(cos.item()))


def inner_product_risk(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    p = p.to(torch.float32).flatten()
    q = q.to(torch.float32).flatten()
    if p.numel() != q.numel():
        raise ValueError(f"inner_product_risk requires same shape, got {tuple(p.shape)} vs {tuple(q.shape)}")
    if p.numel() == 0:
        return 0.0
    p = p / torch.clamp(p.sum(), min=float(eps))
    q = q / torch.clamp(q.sum(), min=float(eps))
    return float(1.0 - float(torch.dot(p, q).item()))


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    p = p.to(torch.float32).flatten()
    q = q.to(torch.float32).flatten()
    if p.numel() != q.numel():
        raise ValueError(f"js_divergence requires same shape, got {tuple(p.shape)} vs {tuple(q.shape)}")
    if p.numel() == 0:
        return 0.0
    p = p / torch.clamp(p.sum(), min=float(eps))
    q = q / torch.clamp(q.sum(), min=float(eps))
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(torch.clamp(p, min=float(eps))) - torch.log(torch.clamp(m, min=float(eps)))))
    kl_qm = torch.sum(q * (torch.log(torch.clamp(q, min=float(eps))) - torch.log(torch.clamp(m, min=float(eps)))))
    return float(0.5 * (kl_pm + kl_qm).item())


def topk_guidance_coverage(alpha_img_mean: torch.Tensor, guidance: torch.Tensor, k: int = 5) -> float:
    alpha_img_mean = alpha_img_mean.to(torch.float32).flatten()
    guidance = guidance.to(torch.float32).flatten()
    if alpha_img_mean.numel() != guidance.numel():
        raise ValueError(
            f"topk_guidance_coverage requires same shape, got {tuple(alpha_img_mean.shape)} vs {tuple(guidance.shape)}"
        )
    if alpha_img_mean.numel() == 0:
        return 0.0
    alpha_img_mean = alpha_img_mean / torch.clamp(alpha_img_mean.sum(), min=1e-8)
    guidance = guidance / torch.clamp(guidance.sum(), min=1e-8)
    kk = max(1, min(int(k), int(guidance.numel())))
    idx = torch.topk(guidance, kk).indices
    return float(alpha_img_mean[idx].sum().item())


def compute_aggregate_probe_scores(
    attentions: Sequence[torch.Tensor],
    image_start: int,
    image_end: int,
    late_start: int,
    late_end: int,
    guidance: torch.Tensor,
    topk: int = 5,
    mismatch_lambda: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, object]:
    agg = compute_mean_image_attention_distribution_last_row(
        attentions=attentions,
        image_start=image_start,
        image_end=image_end,
        late_start=late_start,
        late_end=late_end,
        eps=eps,
    )
    alpha_img_mean = agg["alpha_img_mean"]
    guidance = guidance.to(torch.float32).flatten()
    if alpha_img_mean.numel() != guidance.numel():
        raise ValueError(
            f"Guidance/image attention size mismatch: {int(alpha_img_mean.numel())} vs {int(guidance.numel())}"
        )
    if guidance.numel() > 0:
        guidance = guidance / torch.clamp(guidance.sum(), min=float(eps))

    c_agg_cos = cosine_mismatch(alpha_img_mean, guidance, eps=eps)
    c_agg_ip = inner_product_risk(alpha_img_mean, guidance, eps=eps)
    topk_cov = topk_guidance_coverage(alpha_img_mean, guidance, k=topk)
    e_agg_js = js_divergence(alpha_img_mean, guidance, eps=eps)
    e_agg_combo = float(c_agg_cos + float(mismatch_lambda) * (1.0 - topk_cov))
    frg_shared_mean = float(1.0 - float(agg["late_head_vis_ratio_mean"]))
    frg_shared_topk = float(1.0 - float(agg["late_head_vis_ratio_topkmean"]))

    return {
        **agg,
        "frg_shared_mean": float(frg_shared_mean),
        "frg_shared_topk": float(frg_shared_topk),
        "c_agg_cos": float(c_agg_cos),
        "c_agg_ip": float(c_agg_ip),
        "e_agg_js": float(e_agg_js),
        "e_agg_combo": float(e_agg_combo),
        "topk_guidance_coverage": float(topk_cov),
    }
