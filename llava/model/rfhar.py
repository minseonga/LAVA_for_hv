from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RFHARConfig:
    gamma: float = 0.3
    r_percent: float = 0.2
    lambda_penalty: float = 0.5
    eps: float = 1e-6


class RFHAR(nn.Module):
    """
    RF-guided Head-Aware Reweighting (RF-HAR).

    Operates on last-query attention logits [B,H,K_full] and modifies only image-token
    columns. Returns modified logits (not softmaxed probabilities).
    """

    def __init__(
        self,
        gamma: float = 0.3,
        r_percent: float = 0.2,
        lambda_penalty: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.config = RFHARConfig(
            gamma=float(gamma),
            r_percent=float(r_percent),
            lambda_penalty=float(lambda_penalty),
            eps=float(eps),
        )

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / (std + float(eps))

    @staticmethod
    def _require_2d(x: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be a torch.Tensor")
        if x.dim() != 2:
            raise ValueError(f"{name} must be rank-2 [B,K_img], got shape={tuple(x.shape)}")
        return x

    def compute_rf(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        feats keys: C/A/D/B, each [B,K_img].
        """
        required = ("C", "A", "D", "B")
        for k in required:
            if k not in feats:
                raise KeyError(f"Missing RF-HAR feature: {k}")
        C = self._require_2d(feats["C"], "C").float()
        A = self._require_2d(feats["A"], "A").float()
        D = self._require_2d(feats["D"], "D").float()
        B = self._require_2d(feats["B"], "B").float()
        if not (C.shape == A.shape == D.shape == B.shape):
            raise ValueError(
                "RF-HAR features must share shape [B,K_img], got "
                f"C={tuple(C.shape)} A={tuple(A.shape)} D={tuple(D.shape)} B={tuple(B.shape)}"
            )

        eps = float(self.config.eps)
        C_tilde = torch.relu(self._zscore(C, eps=eps))
        A_tilde = torch.sigmoid(self._zscore(A, eps=eps))
        D_tilde = torch.sigmoid(self._zscore(D, eps=eps))
        B_tilde = torch.sigmoid(self._zscore(B, eps=eps))

        denom = 1.0 + float(self.config.lambda_penalty) * (D_tilde + B_tilde)
        rf = C_tilde * A_tilde / torch.clamp(denom, min=eps)
        rf = torch.clamp(rf, min=0.0)  # keep non-negative latent utility
        return rf

    @staticmethod
    def _topk_binary(scores: torch.Tensor, k: int) -> torch.Tensor:
        """
        scores: [B,H]
        returns binary mask [B,H]
        """
        bsz, n_heads = int(scores.size(0)), int(scores.size(1))
        if n_heads <= 0 or k <= 0:
            return torch.zeros_like(scores)
        k_eff = int(max(1, min(k, n_heads)))
        vals, idx = torch.topk(scores, k=k_eff, dim=-1)
        out = torch.zeros_like(scores)
        out.scatter_(1, idx, (vals > 0).to(dtype=scores.dtype))
        return out

    def dynamic_head_roles(
        self,
        attn_img_norm: torch.Tensor,  # [B,H,K_img]
        rf: torch.Tensor,  # [B,K_img]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if attn_img_norm.dim() != 3:
            raise ValueError(f"attn_img_norm must be [B,H,K_img], got {tuple(attn_img_norm.shape)}")
        if rf.dim() != 2:
            raise ValueError(f"rf must be [B,K_img], got {tuple(rf.shape)}")
        if int(attn_img_norm.size(0)) != int(rf.size(0)) or int(attn_img_norm.size(2)) != int(rf.size(1)):
            raise ValueError(
                f"attn_img_norm/rf shape mismatch: attn={tuple(attn_img_norm.shape)} rf={tuple(rf.shape)}"
            )
        B, H, _ = int(attn_img_norm.size(0)), int(attn_img_norm.size(1)), int(attn_img_norm.size(2))
        if H <= 0:
            z = torch.zeros((B, H), dtype=attn_img_norm.dtype, device=attn_img_norm.device)
            return z, z, {"k_heads": 0.0, "overlap_frac": 0.0}

        low_rf = torch.clamp(1.0 - rf, min=0.0)
        s_pos = (attn_img_norm * rf[:, None, :]).sum(dim=-1)  # [B,H]
        s_neg = (attn_img_norm * low_rf[:, None, :]).sum(dim=-1)  # [B,H]

        k_heads = int(max(1, math.ceil(float(self.config.r_percent) * float(H))))
        k_heads = int(min(k_heads, H))
        m_pos = self._topk_binary(s_pos, k=k_heads)

        # Enforce disjointness: remove positive heads from negative candidate pool.
        neg_scores = s_neg.masked_fill(m_pos > 0, float("-inf"))
        remaining = torch.clamp((m_pos == 0).to(torch.int64).sum(dim=-1), min=0)
        k_neg = int(min(k_heads, int(remaining.max().item()) if B > 0 else 0))
        if k_neg <= 0:
            m_neg = torch.zeros_like(m_pos)
        else:
            m_neg = self._topk_binary(neg_scores, k=k_neg)
            m_neg = m_neg * (m_pos == 0).to(dtype=m_neg.dtype)

        overlap = ((m_pos > 0) & (m_neg > 0)).to(torch.float32).mean().item()
        dbg = {"k_heads": float(k_heads), "overlap_frac": float(overlap)}
        return m_pos, m_neg, dbg

    def _align_rf_to_full(
        self,
        rf: torch.Tensor,  # [B,K_img]
        image_mask: torch.Tensor,  # [B,K_full] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          rf_full: [B,K_full]
          img_counts: [B]
        """
        if image_mask.dim() != 2:
            raise ValueError(f"image_mask must be [B,K_full], got {tuple(image_mask.shape)}")
        if int(image_mask.size(0)) != int(rf.size(0)):
            raise ValueError(
                f"batch mismatch in RF alignment: rf={tuple(rf.shape)} image_mask={tuple(image_mask.shape)}"
            )
        B, K_full = int(image_mask.size(0)), int(image_mask.size(1))
        K_img = int(rf.size(1))
        rf_full = torch.zeros((B, K_full), dtype=rf.dtype, device=rf.device)
        img_counts = image_mask.sum(dim=-1).to(torch.int64)
        for b in range(B):
            idx = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            n_img = int(idx.numel())
            if n_img != K_img:
                raise ValueError(
                    f"RF/image alignment mismatch at batch {b}: mask image count={n_img}, feat K_img={K_img}"
                )
            rf_full[b, idx] = rf[b]
        return rf_full, img_counts

    def forward(
        self,
        attn_logits_last: torch.Tensor,  # [B,H,K_full]
        image_mask: torch.Tensor,  # [B,K_full] bool
        feats: Dict[str, torch.Tensor],  # C/A/D/B: [B,K_img]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if attn_logits_last.dim() != 3:
            raise ValueError(f"attn_logits_last must be [B,H,K_full], got {tuple(attn_logits_last.shape)}")
        if image_mask.dim() != 2:
            raise ValueError(f"image_mask must be [B,K_full], got {tuple(image_mask.shape)}")
        B, H, K_full = (
            int(attn_logits_last.size(0)),
            int(attn_logits_last.size(1)),
            int(attn_logits_last.size(2)),
        )
        if int(image_mask.size(0)) != B or int(image_mask.size(1)) != K_full:
            raise ValueError(
                f"attn/image mask shape mismatch: attn={tuple(attn_logits_last.shape)} mask={tuple(image_mask.shape)}"
            )

        if float(self.config.gamma) == 0.0:
            return attn_logits_last, {
                "rf_mean": 0.0,
                "m_pos_mean": 0.0,
                "m_neg_mean": 0.0,
                "delta_abs_mean": 0.0,
                "disjoint_ok": 1.0,
            }

        rf = self.compute_rf(feats=feats)  # [B,K_img]
        rf_full, img_counts = self._align_rf_to_full(rf=rf.to(device=attn_logits_last.device), image_mask=image_mask)
        img_counts_f = torch.clamp(img_counts.to(dtype=attn_logits_last.dtype), min=1.0)

        # Full softmax first, then image-only renormalization.
        probs_full = torch.softmax(attn_logits_last.float(), dim=-1).to(dtype=attn_logits_last.dtype)  # [B,H,K]
        img_mask_f = image_mask[:, None, :].to(dtype=attn_logits_last.dtype)
        probs_img = probs_full * img_mask_f
        probs_img_norm = probs_img / torch.clamp(probs_img.sum(dim=-1, keepdim=True), min=float(self.config.eps))

        # Gather image-only attention per batch in runtime image-column order.
        K_img = int(rf.size(1))
        attn_img_norm = torch.zeros((B, H, K_img), dtype=attn_logits_last.dtype, device=attn_logits_last.device)
        for b in range(B):
            idx = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            attn_img_norm[b] = probs_img_norm[b, :, idx]

        m_pos, m_neg, role_dbg = self.dynamic_head_roles(attn_img_norm=attn_img_norm, rf=rf.to(attn_logits_last.device))
        disjoint_ok = float((((m_pos > 0) & (m_neg > 0)).sum().item()) == 0)

        low_rf = torch.clamp(1.0 - rf.to(attn_logits_last.device), min=0.0)
        delta_img = (
            float(self.config.gamma) * m_pos[:, :, None] * rf[:, None, :].to(attn_logits_last.device)
            - float(self.config.gamma) * m_neg[:, :, None] * low_rf[:, None, :]
        )  # [B,H,K_img]
        delta_img = delta_img.to(dtype=attn_logits_last.dtype)

        delta_full = torch.zeros((B, H, K_full), dtype=attn_logits_last.dtype, device=attn_logits_last.device)
        for b in range(B):
            idx = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            delta_full[b, :, idx] = delta_img[b]
        delta_full = delta_full * img_mask_f  # hard safety: text columns untouched

        out_logits = attn_logits_last + delta_full
        dbg = {
            "rf_mean": float(rf.mean().item()),
            "rf_max": float(rf.max().item()),
            "rf_min": float(rf.min().item()),
            "m_pos_mean": float((m_pos > 0).to(torch.float32).mean().item()),
            "m_neg_mean": float((m_neg > 0).to(torch.float32).mean().item()),
            "delta_abs_mean": float(delta_full.abs().sum(dim=-1).mean().item() / float(H) / float(torch.clamp(img_counts_f, min=1.0).mean().item())),
            "disjoint_ok": disjoint_ok,
            "k_heads": float(role_dbg.get("k_heads", 0.0)),
            "overlap_frac": float(role_dbg.get("overlap_frac", 0.0)),
        }
        return out_logits, dbg
