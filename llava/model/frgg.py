from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class FRGGConfig:
    gamma: float = 0.2
    tau_c: float = 0.5
    tau_e: float = 0.5
    k_c: float = 8.0
    k_e: float = 8.0
    topk_ratio: float = 0.2
    eps: float = 1e-6


class FRGG(nn.Module):
    """
    Faithful-Routing-Gated Guidance (FRGG).

    - Builds token prior from A/C: S_i = ReLU(z(C_i)) * sigmoid(z(A_i))
    - Builds sample gate from C/E top-k means
    - Applies additive bias only on image-token columns, faithful-head rows, last query logits.

    Input logits are [B,H,K_full] for last query row only.
    Returns modified logits (NOT probabilities).
    """

    def __init__(
        self,
        gamma: float = 0.2,
        tau_c: float = 0.5,
        tau_e: float = 0.5,
        k_c: float = 8.0,
        k_e: float = 8.0,
        topk_ratio: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.config = FRGGConfig(
            gamma=float(gamma),
            tau_c=float(tau_c),
            tau_e=float(tau_e),
            k_c=float(k_c),
            k_e=float(k_e),
            topk_ratio=float(topk_ratio),
            eps=float(eps),
        )

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / (sd + float(eps))

    @staticmethod
    def _require_2d(x: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be torch.Tensor")
        if x.dim() != 2:
            raise ValueError(f"{name} must be [B,K_img], got {tuple(x.shape)}")
        return x

    def _topk_mean(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,K]
        B, K = int(x.size(0)), int(x.size(1))
        if K <= 0:
            return torch.zeros((B,), dtype=x.dtype, device=x.device)
        k = int(max(1, math.ceil(float(self.config.topk_ratio) * float(K))))
        k = int(min(k, K))
        vals = torch.topk(x, k=k, dim=-1).values
        return vals.mean(dim=-1)

    def compute_prior(self, A: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        A = self._require_2d(A, "A").float()
        C = self._require_2d(C, "C").float()
        if A.shape != C.shape:
            raise ValueError(f"A/C shape mismatch: A={tuple(A.shape)} C={tuple(C.shape)}")
        eps = float(self.config.eps)
        S = torch.relu(self._zscore(C, eps=eps)) * torch.sigmoid(self._zscore(A, eps=eps))
        P = S / (S.sum(dim=-1, keepdim=True) + eps)
        return P

    def compute_gate(self, C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        C = self._require_2d(C, "C").float()
        E = self._require_2d(E, "E").float()
        if C.shape != E.shape:
            raise ValueError(f"C/E shape mismatch: C={tuple(C.shape)} E={tuple(E.shape)}")
        Cb = self._topk_mean(C)
        Eb = self._topk_mean(E)
        g_c = torch.sigmoid(float(self.config.k_c) * (float(self.config.tau_c) - Cb))
        g_e = torch.sigmoid(float(self.config.k_e) * (float(self.config.tau_e) - Eb))
        return g_c * g_e

    def _align_prior_to_full(self, P: torch.Tensor, image_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # P: [B,K_img], image_mask: [B,K_full]
        B, K_full = int(image_mask.size(0)), int(image_mask.size(1))
        K_img = int(P.size(1))
        P_full = torch.zeros((B, K_full), dtype=P.dtype, device=P.device)
        img_counts = image_mask.sum(dim=-1).to(torch.int64)
        for b in range(B):
            idx = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            n_img = int(idx.numel())
            if n_img != K_img:
                raise ValueError(
                    f"FRGG prior/image alignment mismatch at batch {b}: mask image count={n_img}, prior K_img={K_img}"
                )
            P_full[b, idx] = P[b]
        return P_full, img_counts

    def _resolve_head_mask(
        self,
        faithful_head_mask: Optional[torch.Tensor],
        B: int,
        H: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if faithful_head_mask is None:
            return None
        hm = faithful_head_mask
        if not torch.is_tensor(hm):
            hm = torch.as_tensor(hm)
        hm = hm.to(device=device, dtype=dtype)
        if hm.dim() == 1:
            if int(hm.numel()) != int(H):
                raise ValueError(f"faithful_head_mask [H] expected H={H}, got {tuple(hm.shape)}")
            hm = hm[None, :].expand(B, H)
        elif hm.dim() == 2:
            if int(hm.size(0)) == 1 and B > 1:
                hm = hm.expand(B, int(hm.size(1)))
            if int(hm.size(0)) != B or int(hm.size(1)) != H:
                raise ValueError(f"faithful_head_mask [B,H] mismatch: expected ({B},{H}), got {tuple(hm.shape)}")
        else:
            raise ValueError(f"faithful_head_mask must be [H] or [B,H], got {tuple(hm.shape)}")
        return hm

    def forward(
        self,
        attn_logits_last: torch.Tensor,  # [B,H,K_full]
        image_mask: torch.Tensor,  # [B,K_full] bool
        feats: Dict[str, torch.Tensor],  # A,C,E -> [B,K_img]
        faithful_head_mask: Optional[torch.Tensor] = None,  # [H] or [B,H]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if attn_logits_last.dim() != 3:
            raise ValueError(f"attn_logits_last must be [B,H,K_full], got {tuple(attn_logits_last.shape)}")
        if image_mask.dim() != 2:
            raise ValueError(f"image_mask must be [B,K_full], got {tuple(image_mask.shape)}")
        B, H, K_full = int(attn_logits_last.size(0)), int(attn_logits_last.size(1)), int(attn_logits_last.size(2))
        if int(image_mask.size(0)) != B or int(image_mask.size(1)) != K_full:
            raise ValueError(
                f"attn/image mask shape mismatch: attn={tuple(attn_logits_last.shape)} mask={tuple(image_mask.shape)}"
            )

        if float(self.config.gamma) == 0.0:
            return attn_logits_last, {
                "frgg_gate_mean": 0.0,
                "frgg_prior_entropy": 0.0,
                "frgg_delta_abs_mean": 0.0,
                "frgg_active_frac": 0.0,
            }

        required = ("A", "C", "E")
        for k in required:
            if k not in feats:
                raise KeyError(f"Missing FRGG feature: {k}")
        A = self._require_2d(feats["A"], "A").to(device=attn_logits_last.device)
        C = self._require_2d(feats["C"], "C").to(device=attn_logits_last.device)
        E = self._require_2d(feats["E"], "E").to(device=attn_logits_last.device)
        if not (A.shape == C.shape == E.shape):
            raise ValueError(f"FRGG feature shapes must match, got A={tuple(A.shape)} C={tuple(C.shape)} E={tuple(E.shape)}")

        hm = self._resolve_head_mask(
            faithful_head_mask=faithful_head_mask,
            B=B,
            H=H,
            device=attn_logits_last.device,
            dtype=attn_logits_last.dtype,
        )
        if hm is None or float((hm > 0).to(torch.float32).sum().item()) == 0.0:
            return attn_logits_last, {
                "frgg_gate_mean": 0.0,
                "frgg_prior_entropy": 0.0,
                "frgg_delta_abs_mean": 0.0,
                "frgg_active_frac": 0.0,
            }

        P_img = self.compute_prior(A=A, C=C).to(device=attn_logits_last.device, dtype=attn_logits_last.dtype)
        g = self.compute_gate(C=C, E=E).to(device=attn_logits_last.device, dtype=attn_logits_last.dtype)  # [B]
        P_full, img_counts = self._align_prior_to_full(P=P_img, image_mask=image_mask)

        img_mask_f = image_mask[:, None, :].to(dtype=attn_logits_last.dtype)
        delta = float(self.config.gamma) * g[:, None, None] * hm[:, :, None] * P_full[:, None, :]
        delta = delta * img_mask_f  # text columns unchanged

        out_logits = attn_logits_last + delta

        eps = float(max(self.config.eps, 1e-12))
        p_safe = torch.clamp(P_img.float(), min=eps)
        ent = float((-(p_safe * p_safe.log()).sum(dim=-1)).mean().item())
        img_cnt = torch.clamp(img_counts.to(dtype=attn_logits_last.dtype), min=1.0)
        delta_abs_mean = float((delta.abs().sum(dim=-1) / (img_cnt[:, None] + eps)).mean().item())
        active_frac = float((delta.abs().sum(dim=(-1, -2)) > 0).to(torch.float32).mean().item())

        dbg = {
            "frgg_gate_mean": float(g.mean().item()),
            "frgg_gate_min": float(g.min().item()),
            "frgg_gate_max": float(g.max().item()),
            "frgg_prior_entropy": ent,
            "frgg_faithful_head_cov": float((hm > 0).to(torch.float32).mean().item()),
            "frgg_delta_abs_mean": delta_abs_mean,
            "frgg_active_frac": active_frac,
        }
        return out_logits, dbg
