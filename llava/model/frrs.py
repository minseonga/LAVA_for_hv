from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class FRRSConfig:
    alpha: float = 0.5
    beta: float = 0.5
    tau_c: float = 0.0
    tau_e: float = 0.0
    k_c: float = 8.0
    k_e: float = 8.0
    topk_ratio: float = 0.2
    eps: float = 1e-6
    arm: str = "supportive"
    head_mode: str = "dynamic"   # static | dynamic | hybrid
    r_percent: float = 0.2


class FRRS(nn.Module):
    """
    Faithful Routing Residual Steering (FRRS).

    - Token scores (recomputed on every forward call):
      S_i = ReLU(z(C_i)) * sigmoid(z(A_i))
      M_i = sigmoid(z(D_i)) * sigmoid(z(A_i)) / (1 + ReLU(z(C_i)))
      g   = sigmoid(k_c*(tau_c-topkmean(z(C)))) * sigmoid(k_e*(tau_e-topkmean(z(E))))

    - Visual summaries:
      u_pos = sum_i Sbar_i * v_i
      u_neg = sum_i Mbar_i * v_i

    - Head routing mode:
      static:  use provided faithful/harmful head masks
      dynamic: select heads from image-only normalized attention (top-r%)
      hybrid:  union(static, dynamic), then enforce disjointness

    - Steering on last query row head output only:
      supportive: o' = o + g * alpha * I[h in H+] * u_pos
      bipolar:    o' = o + g * (alpha * I[h in H+] * u_pos - beta * I[h in H-] * u_neg)

    Inputs:
      attn_output: [B,H,Q,D]
      value_states: [B,H,K,D]
      image_mask: [B,K] bool
      feats: dict with A/C/E [B,K_img], optional D [B,K_img]
      faithful_head_mask / harmful_head_mask: optional [H] or [B,H]
      attn_weights_last: optional [B,H,K] (used for dynamic head selection)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        tau_c: float = 0.0,
        tau_e: float = 0.0,
        k_c: float = 8.0,
        k_e: float = 8.0,
        topk_ratio: float = 0.2,
        eps: float = 1e-6,
        arm: str = "supportive",
        head_mode: str = "dynamic",
        r_percent: float = 0.2,
    ) -> None:
        super().__init__()
        self.config = FRRSConfig(
            alpha=float(alpha),
            beta=float(beta),
            tau_c=float(tau_c),
            tau_e=float(tau_e),
            k_c=float(k_c),
            k_e=float(k_e),
            topk_ratio=float(topk_ratio),
            eps=float(eps),
            arm=str(arm or "supportive").strip().lower(),
            head_mode=str(head_mode or "dynamic").strip().lower(),
            r_percent=float(r_percent),
        )

    @staticmethod
    def _require_2d(x: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be torch.Tensor")
        if x.dim() != 2:
            raise ValueError(f"{name} must be [B,K_img], got {tuple(x.shape)}")
        return x

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / (sd + float(eps))

    def _topk_mean(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,K]
        bsz, k_total = int(x.size(0)), int(x.size(1))
        if k_total <= 0:
            return torch.zeros((bsz,), dtype=x.dtype, device=x.device)
        k = int(max(1, math.ceil(float(self.config.topk_ratio) * float(k_total))))
        k = int(min(k, k_total))
        vals = torch.topk(x, k=k, dim=-1).values
        return vals.mean(dim=-1)

    def _align_to_full(
        self,
        x_img: torch.Tensor,  # [B,K_img]
        image_mask: torch.Tensor,  # [B,K_full] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, k_full = int(image_mask.size(0)), int(image_mask.size(1))
        k_img = int(x_img.size(1))
        x_full = torch.zeros((bsz, k_full), dtype=x_img.dtype, device=x_img.device)
        img_counts = image_mask.sum(dim=-1).to(torch.int64)
        for b in range(bsz):
            idx = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            n_img = int(idx.numel())
            if n_img != k_img:
                raise ValueError(
                    f"FRRS image alignment mismatch at batch {b}: mask image count={n_img}, feat K_img={k_img}"
                )
            x_full[b, idx] = x_img[b]
        return x_full, img_counts

    def _resolve_head_mask(
        self,
        head_mask: Optional[torch.Tensor],  # [H] or [B,H]
        bsz: int,
        n_heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if head_mask is None:
            return None
        hm = head_mask
        if not torch.is_tensor(hm):
            hm = torch.as_tensor(hm)
        hm = hm.to(device=device, dtype=dtype)
        if hm.dim() == 1:
            if int(hm.numel()) != int(n_heads):
                raise ValueError(f"head_mask [H] mismatch: expected H={n_heads}, got {tuple(hm.shape)}")
            hm = hm[None, :].expand(bsz, n_heads)
        elif hm.dim() == 2:
            if int(hm.size(0)) == 1 and bsz > 1:
                hm = hm.expand(bsz, int(hm.size(1)))
            if int(hm.size(0)) != bsz or int(hm.size(1)) != n_heads:
                raise ValueError(
                    f"head_mask [B,H] mismatch: expected ({bsz},{n_heads}), got {tuple(hm.shape)}"
                )
        else:
            raise ValueError(f"head_mask must be [H] or [B,H], got {tuple(hm.shape)}")
        return hm

    @staticmethod
    def _topk_binary(scores: torch.Tensor, k: int) -> torch.Tensor:
        # scores: [B,H]
        bsz, n_heads = int(scores.size(0)), int(scores.size(1))
        if n_heads <= 0 or k <= 0:
            return torch.zeros_like(scores)
        k_eff = int(max(1, min(k, n_heads)))
        vals, idx = torch.topk(scores, k=k_eff, dim=-1)
        out = torch.zeros_like(scores)
        out.scatter_(1, idx, (vals > 0).to(dtype=scores.dtype))
        return out

    def _dynamic_head_roles(
        self,
        attn_img_norm: torch.Tensor,  # [B,H,K_full] (non-image already zeroed, renormed)
        s_full: torch.Tensor,         # [B,K_full]
        m_full: torch.Tensor,         # [B,K_full]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if attn_img_norm.dim() != 3:
            raise ValueError(f"attn_img_norm must be [B,H,K], got {tuple(attn_img_norm.shape)}")
        if s_full.dim() != 2 or m_full.dim() != 2:
            raise ValueError(f"s_full/m_full must be [B,K], got {tuple(s_full.shape)} {tuple(m_full.shape)}")
        if int(attn_img_norm.size(0)) != int(s_full.size(0)) or int(attn_img_norm.size(2)) != int(s_full.size(1)):
            raise ValueError(
                f"shape mismatch: attn={tuple(attn_img_norm.shape)} s={tuple(s_full.shape)} m={tuple(m_full.shape)}"
            )
        B, H, _ = int(attn_img_norm.size(0)), int(attn_img_norm.size(1)), int(attn_img_norm.size(2))
        if H <= 0:
            z = torch.zeros((B, H), dtype=attn_img_norm.dtype, device=attn_img_norm.device)
            return z, z, {"k_heads": 0.0, "overlap_frac": 0.0, "s_pos_mean": 0.0, "s_neg_mean": 0.0}

        s_pos = (attn_img_norm * s_full[:, None, :]).sum(dim=-1)  # [B,H]
        s_neg = (attn_img_norm * m_full[:, None, :]).sum(dim=-1)  # [B,H]

        k_heads = int(max(1, math.ceil(float(self.config.r_percent) * float(H))))
        k_heads = int(min(k_heads, H))

        m_pos = self._topk_binary(s_pos, k=k_heads)
        neg_scores = s_neg.masked_fill(m_pos > 0, float("-inf"))
        m_neg = self._topk_binary(neg_scores, k=k_heads)
        m_neg = m_neg * (m_pos == 0).to(dtype=m_neg.dtype)

        overlap = ((m_pos > 0) & (m_neg > 0)).to(torch.float32).mean().item()
        dbg = {
            "k_heads": float(k_heads),
            "overlap_frac": float(overlap),
            "s_pos_mean": float(s_pos.mean().item()),
            "s_neg_mean": float(s_neg.mean().item()),
        }
        return m_pos, m_neg, dbg

    def _compute_token_scores(
        self,
        feats: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        required = ("A", "C", "E")
        for k in required:
            if k not in feats:
                raise KeyError(f"Missing FRRS feature: {k}")
        A = self._require_2d(feats["A"], "A").to(device=device).float()
        C = self._require_2d(feats["C"], "C").to(device=device).float()
        E = self._require_2d(feats["E"], "E").to(device=device).float()
        if not (A.shape == C.shape == E.shape):
            raise ValueError(
                f"FRRS A/C/E shapes must match, got A={tuple(A.shape)} C={tuple(C.shape)} E={tuple(E.shape)}"
            )
        if "D" in feats and feats["D"] is not None:
            D = self._require_2d(feats["D"], "D").to(device=device).float()
            if D.shape != A.shape:
                raise ValueError(f"FRRS D shape mismatch: D={tuple(D.shape)} vs A={tuple(A.shape)}")
        else:
            D = torch.zeros_like(A)

        eps = float(self.config.eps)
        zA = self._zscore(A, eps=eps)
        zC = self._zscore(C, eps=eps)
        zD = self._zscore(D, eps=eps)
        zE = self._zscore(E, eps=eps)

        S = torch.relu(zC) * torch.sigmoid(zA)
        M = torch.sigmoid(zD) * torch.sigmoid(zA) / (1.0 + torch.relu(zC))

        S = S / (S.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-1, keepdim=True) + eps)

        c_bar = self._topk_mean(zC)
        e_bar = self._topk_mean(zE)
        g_c = torch.sigmoid(float(self.config.k_c) * (float(self.config.tau_c) - c_bar))
        g_e = torch.sigmoid(float(self.config.k_e) * (float(self.config.tau_e) - e_bar))
        gate = g_c * g_e
        return S, M, gate, c_bar, e_bar

    def forward(
        self,
        attn_output: torch.Tensor,  # [B,H,Q,D]
        value_states: torch.Tensor,  # [B,H,K,D]
        image_mask: torch.Tensor,  # [B,K] bool
        feats: Dict[str, torch.Tensor],  # A/C/E required, D optional
        faithful_head_mask: Optional[torch.Tensor] = None,  # [H] or [B,H]
        harmful_head_mask: Optional[torch.Tensor] = None,  # [H] or [B,H]
        attn_weights_last: Optional[torch.Tensor] = None,  # [B,H,K], optional
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if attn_output.dim() != 4:
            raise ValueError(f"attn_output must be [B,H,Q,D], got {tuple(attn_output.shape)}")
        if value_states.dim() != 4:
            raise ValueError(f"value_states must be [B,H,K,D], got {tuple(value_states.shape)}")
        if image_mask.dim() != 2:
            raise ValueError(f"image_mask must be [B,K], got {tuple(image_mask.shape)}")

        bsz, n_heads, _q_len, head_dim = (
            int(attn_output.size(0)),
            int(attn_output.size(1)),
            int(attn_output.size(2)),
            int(attn_output.size(3)),
        )
        if int(value_states.size(0)) != bsz or int(value_states.size(1)) != n_heads or int(value_states.size(3)) != head_dim:
            raise ValueError(
                f"attn_output/value_states shape mismatch: out={tuple(attn_output.shape)} v={tuple(value_states.shape)}"
            )
        if int(value_states.size(2)) != int(image_mask.size(1)) or int(image_mask.size(0)) != bsz:
            raise ValueError(
                f"value/image_mask mismatch: value={tuple(value_states.shape)} mask={tuple(image_mask.shape)}"
            )

        if float(self.config.alpha) == 0.0 and float(self.config.beta) == 0.0:
            return attn_output, {
                "frrs_gate_mean": 0.0,
                "frrs_delta_abs_mean": 0.0,
                "frrs_active_frac": 0.0,
                "frrs_supportive_head_cov": 0.0,
                "frrs_harmful_head_cov": 0.0,
                "frrs_dynamic_used": 0.0,
            }

        arm = str(self.config.arm or "supportive").strip().lower()
        if arm not in {"supportive", "bipolar"}:
            arm = "supportive"

        hm_pos_static = self._resolve_head_mask(
            head_mask=faithful_head_mask,
            bsz=bsz,
            n_heads=n_heads,
            device=attn_output.device,
            dtype=attn_output.dtype,
        )
        hm_neg_static = self._resolve_head_mask(
            head_mask=harmful_head_mask,
            bsz=bsz,
            n_heads=n_heads,
            device=attn_output.device,
            dtype=attn_output.dtype,
        )
        if hm_pos_static is None:
            hm_pos_static = torch.zeros((bsz, n_heads), device=attn_output.device, dtype=attn_output.dtype)
        if hm_neg_static is None:
            hm_neg_static = torch.zeros((bsz, n_heads), device=attn_output.device, dtype=attn_output.dtype)

        S_img, M_img, gate, c_bar, e_bar = self._compute_token_scores(feats=feats, device=attn_output.device)
        S_full, img_counts = self._align_to_full(S_img.to(device=attn_output.device), image_mask=image_mask)
        M_full, _ = self._align_to_full(M_img.to(device=attn_output.device), image_mask=image_mask)

        # Dynamic head role selection from image-only normalized attention over keys.
        hm_pos_dyn = torch.zeros((bsz, n_heads), device=attn_output.device, dtype=attn_output.dtype)
        hm_neg_dyn = torch.zeros((bsz, n_heads), device=attn_output.device, dtype=attn_output.dtype)
        dyn_dbg: Dict[str, float] = {"k_heads": 0.0, "overlap_frac": 0.0, "s_pos_mean": 0.0, "s_neg_mean": 0.0}
        dyn_used = 0.0
        head_mode = str(self.config.head_mode or "dynamic").strip().lower()
        if head_mode not in {"static", "dynamic", "hybrid"}:
            head_mode = "dynamic"

        if head_mode in {"dynamic", "hybrid"} and attn_weights_last is not None:
            if attn_weights_last.dim() != 3:
                raise ValueError(f"attn_weights_last must be [B,H,K], got {tuple(attn_weights_last.shape)}")
            if int(attn_weights_last.size(0)) != bsz or int(attn_weights_last.size(1)) != n_heads or int(attn_weights_last.size(2)) != int(image_mask.size(1)):
                raise ValueError(
                    f"attn_weights_last mismatch: got {tuple(attn_weights_last.shape)} expected ({bsz},{n_heads},{int(image_mask.size(1))})"
                )
            img_mask_f = image_mask[:, None, :].to(dtype=attn_weights_last.dtype)
            attn_img = attn_weights_last * img_mask_f
            attn_img_norm = attn_img / torch.clamp(attn_img.sum(dim=-1, keepdim=True), min=float(self.config.eps))
            hm_pos_dyn, hm_neg_dyn, dyn_dbg = self._dynamic_head_roles(
                attn_img_norm=attn_img_norm.to(dtype=attn_output.dtype),
                s_full=S_full.to(dtype=attn_output.dtype),
                m_full=M_full.to(dtype=attn_output.dtype),
            )
            dyn_used = 1.0

        if head_mode == "static":
            hm_pos = hm_pos_static
            hm_neg = hm_neg_static
        elif head_mode == "hybrid":
            hm_pos = torch.clamp(hm_pos_static + hm_pos_dyn, min=0.0, max=1.0)
            hm_neg = torch.clamp(hm_neg_static + hm_neg_dyn, min=0.0, max=1.0)
        else:
            hm_pos = hm_pos_dyn
            hm_neg = hm_neg_dyn

        # Enforce disjointness (positive priority).
        hm_neg = hm_neg * (hm_pos == 0).to(dtype=hm_neg.dtype)

        if arm != "bipolar":
            hm_neg = torch.zeros_like(hm_neg)

        if float(hm_pos.sum().item()) == 0.0 and float(hm_neg.sum().item()) == 0.0:
            return attn_output, {
                "frrs_gate_mean": float(gate.mean().item()),
                "frrs_delta_abs_mean": 0.0,
                "frrs_active_frac": 0.0,
                "frrs_supportive_head_cov": float((hm_pos > 0).to(torch.float32).mean().item()),
                "frrs_harmful_head_cov": float((hm_neg > 0).to(torch.float32).mean().item()),
                "frrs_dynamic_used": float(dyn_used),
                "frrs_head_mode": head_mode,
                "frrs_dyn_k_heads": float(dyn_dbg.get("k_heads", 0.0)),
                "frrs_dyn_overlap_frac": float(dyn_dbg.get("overlap_frac", 0.0)),
            }

        v = value_states.float()
        s = S_full.float()
        m = M_full.float()

        # [B,H,D]
        u_pos = torch.einsum("bk,bhkd->bhd", s, v)
        u_neg = torch.einsum("bk,bhkd->bhd", m, v)

        neg_term = float(self.config.beta) * hm_neg.float()[:, :, None] * u_neg if arm == "bipolar" else 0.0
        delta_last = (
            gate[:, None, None].float()
            * (
                float(self.config.alpha) * hm_pos.float()[:, :, None] * u_pos
                - neg_term
            )
        )
        delta_last = delta_last.to(dtype=attn_output.dtype)

        out = attn_output.clone()
        out[:, :, -1, :] = out[:, :, -1, :] + delta_last

        img_cnt_f = torch.clamp(img_counts.to(dtype=attn_output.dtype), min=1.0)
        delta_abs_mean = float(delta_last.abs().mean().item())
        active_frac = float((delta_last.abs().sum(dim=(-1, -2)) > 0).to(torch.float32).mean().item())
        dbg = {
            "frrs_gate_mean": float(gate.mean().item()),
            "frrs_gate_min": float(gate.min().item()),
            "frrs_gate_max": float(gate.max().item()),
            "frrs_cbar_mean": float(c_bar.mean().item()),
            "frrs_ebar_mean": float(e_bar.mean().item()),
            "frrs_delta_abs_mean": delta_abs_mean,
            "frrs_active_frac": active_frac,
            "frrs_supportive_head_cov": float((hm_pos > 0).to(torch.float32).mean().item()),
            "frrs_harmful_head_cov": float((hm_neg > 0).to(torch.float32).mean().item()),
            "frrs_img_cols_mean": float(img_cnt_f.mean().item()),
            "frrs_arm_bipolar": 1.0 if arm == "bipolar" else 0.0,
            "frrs_dynamic_used": float(dyn_used),
            "frrs_head_mode": head_mode,
            "frrs_dyn_k_heads": float(dyn_dbg.get("k_heads", 0.0)),
            "frrs_dyn_overlap_frac": float(dyn_dbg.get("overlap_frac", 0.0)),
            "frrs_dyn_s_pos_mean": float(dyn_dbg.get("s_pos_mean", 0.0)),
            "frrs_dyn_s_neg_mean": float(dyn_dbg.get("s_neg_mean", 0.0)),
        }
        return out, dbg
