from __future__ import annotations

import csv
import inspect
import json
import math
import os
import types
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from .rfhar import RFHAR
from .frgg import FRGG
from .frrs import FRRS


@dataclass
class AISGatingConfig:
    """Config for AIS-aware soft late-head gating."""

    enable_ais_gating: bool = False
    ais_early_start: int = 0
    ais_early_end: int = 15
    ais_late_start: int = 16
    ais_late_end: int = 31
    ais_topk: int = 8
    # Conservative defaults to avoid always-on triggering.
    ais_tau: float = 2.2
    ais_gamma: float = 0.2
    ais_eps: float = 1e-6
    ais_debug_log: bool = False
    # Experimental arm mode:
    # - legacy: existing behavior (dynamic omega-based suppression)
    # - harmful_only: suppress configured harmful heads only
    # - faithful_only: preserve/reinforce configured faithful heads only
    # - bipolar: harmful suppression + faithful preservation
    ais_arm: str = "legacy"
    # Optional head set inputs.
    # Format for string specs: "18:7,18:24,19:22"
    ais_harmful_heads: str = ""
    ais_faithful_heads: str = ""
    # Optional json path containing harmful/faithful head sets.
    ais_headset_json: str = ""
    # Relative strength for faithful reinforcement in non-legacy modes.
    ais_faithful_boost: float = 1.0
    # Whether harmful branch is additionally weighted by dynamic omega in non-legacy modes.
    ais_use_dynamic_omega: bool = True
    # Budget-centered intervention mode:
    # - when enabled, bypass tau trigger and allocate a fixed intervention mass per late layer.
    # - harmful top-r% heads are suppressed; faithful top-q% heads are preserved.
    ais_use_budget_routing: bool = False
    # Total per-step intervention mass across all late layers.
    ais_budget_total: float = 0.0
    # Top ratios in [0,1] for dynamic per-layer head selection.
    ais_harmful_top_ratio: float = 0.2
    ais_faithful_top_ratio: float = 0.2
    # In bipolar mode, fraction of per-layer budget assigned to harmful suppression.
    ais_bipolar_harmful_ratio: float = 0.5
    # Concentrate budget mass on top-k patch columns (0 or negative means use all patches).
    ais_budget_patch_topk: int = 16
    # If true and headset maps are provided, intervention is applied only on
    # layers explicitly listed in those maps.
    ais_strict_headset_layers: bool = False
    # Budget operator family:
    # - soft: scaled sparse bias (current default)
    # - semi_hard: fixed sparse dose on selected cells
    ais_operator: str = "soft"
    ais_semihard_penalty: float = 0.0
    # Path-probe diagnostic (first-token dominance test):
    # - none: disabled
    # - drop_img: weaken image->output path by penalizing image columns
    # - drop_text: weaken text->text path by penalizing non-image text columns
    # - drop_both: weaken both paths
    path_probe_mode: str = "none"
    path_probe_penalty: float = 0.0
    path_probe_first_step_only: bool = True
    # Oracle head-aware context reweighting:
    # bias on image columns only:
    # z' = z + λ_pos * I[h∈H+]I[p∈P+] - λ_neg * I[h∈H-]I[p∈P-]
    # and attention code subtracts bias, so supportive boost uses negative bias.
    ais_use_oracle_roles: bool = False
    ais_oracle_role_csv: str = ""
    ais_oracle_supportive_topk: int = 5
    ais_oracle_assertive_topk: int = 5
    ais_oracle_lambda_pos: float = 0.25
    ais_oracle_lambda_neg: float = 0.25
    ais_oracle_bias_clip: float = 2.0
    # RF-HAR (training-free, late-layer image-column logit reweighting)
    enable_rfhar: bool = False
    rfhar_early_start: int = 0
    rfhar_early_end: int = 15
    rfhar_late_start: int = 16
    rfhar_late_end: int = 31
    rfhar_r_percent: float = 0.2
    rfhar_gamma: float = 0.3
    rfhar_lambda_penalty: float = 0.5
    rfhar_eps: float = 1e-6
    rfhar_debug_log: bool = False
    # FRGG (Faithful-Routing-Gated Guidance)
    enable_frgg: bool = False
    frgg_late_start: int = 16
    frgg_late_end: int = 30
    frgg_gamma: float = 0.3
    frgg_tau_c: float = 0.0
    frgg_tau_e: float = 0.0
    frgg_k_c: float = 8.0
    frgg_k_e: float = 8.0
    frgg_topk_ratio: float = 0.2
    frgg_eps: float = 1e-6
    frgg_debug_log: bool = False
    # FRRS (Faithful Routing Residual Steering)
    enable_frrs: bool = False
    frrs_late_start: int = 18
    frrs_late_end: int = 21
    frrs_alpha: float = 0.5
    frrs_beta: float = 0.5
    frrs_tau_c: float = 0.0
    frrs_tau_e: float = 0.0
    frrs_k_c: float = 8.0
    frrs_k_e: float = 8.0
    frrs_topk_ratio: float = 0.2
    frrs_eps: float = 1e-6
    frrs_arm: str = "supportive"
    frrs_head_mode: str = "dynamic"
    frrs_r_percent: float = 0.2
    frrs_online_recompute_feats: bool = False
    frrs_online_blend: float = 1.0
    frrs_debug_log: bool = False
    # Decode-time proxy trace: collect lightweight late-head scalar summaries
    # during generation without any extra replay pass.
    enable_proxy_trace: bool = False
    proxy_late_start: int = 18
    proxy_late_end: int = 24
    proxy_eps: float = 1e-6


def _call_orig_forward(
    self_attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_value: Optional[Any],
    output_attentions: bool,
    use_cache: bool,
    kwargs: Dict[str, Any],
):
    orig = getattr(self_attn, "_ais_orig_forward", None)
    if orig is None:
        raise RuntimeError("Original forward is missing on patched attention module.")
    accepts_kwargs = bool(getattr(self_attn, "_ais_orig_accepts_kwargs", False))
    if accepts_kwargs:
        return orig(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
    return orig(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )


class AISGatingRuntime:
    """Stateful runtime used by patched attention layers during generation."""

    def __init__(self, config: Optional[AISGatingConfig] = None):
        self.config = config or AISGatingConfig()
        self._base_image_mask: Optional[torch.Tensor] = None  # CPU bool [B, L_prompt]
        self._active: bool = False
        self._generation_index: int = -1
        self._step_index: int = -1
        self._step_kv_seq_len: Optional[int] = None
        self._step_batch_size: Optional[int] = None
        self._early_sum: Optional[torch.Tensor] = None
        self._early_count: int = 0
        self._late_seen_in_step: int = 0
        self._late_triggered_in_step: int = 0
        self._debug_rows: List[Dict[str, Any]] = []
        self._harmful_heads_map: Dict[int, set] = {}
        self._faithful_heads_map: Dict[int, set] = {}
        self._oracle_supportive_map: Dict[str, List[int]] = {}
        self._oracle_assertive_map: Dict[str, List[int]] = {}
        self._oracle_loaded_sig: Optional[Tuple[str, int, int]] = None
        self._current_sample_ids: List[str] = []
        self._rfhar_feats_base: Optional[Dict[str, torch.Tensor]] = None
        self._rfhar_module: Optional[RFHAR] = None
        self._rfhar_module_sig: Optional[Tuple[float, float, float, float]] = None
        self._frgg_feats_base: Optional[Dict[str, torch.Tensor]] = None
        self._frgg_module: Optional[FRGG] = None
        self._frgg_module_sig: Optional[Tuple[float, float, float, float, float, float, float, float]] = None
        self._frrs_feats_base: Optional[Dict[str, torch.Tensor]] = None
        self._frrs_module: Optional[FRRS] = None
        self._frrs_module_sig: Optional[Tuple[float, float, float, float, float, float, float, float, float, str]] = None
        self._proxy_trace_rows: List[Dict[str, Any]] = []
        self._proxy_step_faithful_sum: Optional[torch.Tensor] = None
        self._proxy_step_harmful_sum: Optional[torch.Tensor] = None
        self._proxy_step_faithful_count: Optional[torch.Tensor] = None
        self._proxy_step_harmful_count: Optional[torch.Tensor] = None
        self._proxy_step_bsz: int = 0

    @property
    def active(self) -> bool:
        return bool(self._active)

    def is_effective_enabled(self) -> bool:
        ais_on = bool(
            self.config.enable_ais_gating
            and (
                float(self.config.ais_gamma) > 0.0
                or bool(self.config.ais_use_oracle_roles)
            )
        )
        rfhar_on = bool(self.config.enable_rfhar and float(self.config.rfhar_gamma) > 0.0)
        frgg_on = bool(self.config.enable_frgg and float(self.config.frgg_gamma) > 0.0)
        frrs_on = bool(self.config.enable_frrs and (float(self.config.frrs_alpha) > 0.0 or float(self.config.frrs_beta) > 0.0))
        proxy_on = bool(self.config.enable_proxy_trace)
        mode = str(self.config.path_probe_mode or "none").strip().lower()
        probe_on = bool(mode != "none" and float(self.config.path_probe_penalty) > 0.0)
        return bool(ais_on or probe_on or rfhar_on or frgg_on or frrs_on or proxy_on)

    def configure(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if v is None or not hasattr(self.config, k):
                continue
            setattr(self.config, k, v)
        self._refresh_head_sets()
        self._refresh_oracle_roles()

    def clear_debug_rows(self) -> None:
        self._debug_rows = []

    def get_debug_rows(self, reset: bool = False) -> List[Dict[str, Any]]:
        rows = list(self._debug_rows)
        if reset:
            self._debug_rows = []
        return rows

    def clear_proxy_trace_rows(self) -> None:
        self._proxy_trace_rows = []

    def get_proxy_trace_rows(self, reset: bool = False) -> List[Dict[str, Any]]:
        rows = list(self._proxy_trace_rows)
        if reset:
            self._proxy_trace_rows = []
        return rows

    def dump_debug_csv(self, path: str) -> None:
        rows = self.get_debug_rows(reset=False)
        if len(rows) == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return
        keys = sorted(set().union(*[set(r.keys()) for r in rows]))
        with open(path, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k, None) for k in keys})

    def begin_generation(
        self,
        image_token_mask: Optional[torch.Tensor],
        sample_ids: Optional[Sequence[Any]] = None,
        rfhar_feats: Optional[Dict[str, Any]] = None,
        frgg_feats: Optional[Dict[str, Any]] = None,
        frrs_feats: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Store base image-token mask on CPU; it indexes key columns in the prompt part.
        self._base_image_mask = None
        if image_token_mask is not None:
            self._base_image_mask = image_token_mask.detach().to(device="cpu", dtype=torch.bool)
        self._active = bool(self.is_effective_enabled() and self._base_image_mask is not None)
        self._current_sample_ids = []
        if sample_ids is not None:
            if isinstance(sample_ids, (list, tuple)):
                self._current_sample_ids = [str(x).strip() for x in sample_ids if str(x).strip() != ""]
            else:
                s = str(sample_ids).strip()
                if s != "":
                    self._current_sample_ids = [s]
        self._generation_index += 1
        self._step_index = -1
        self._step_kv_seq_len = None
        self._step_batch_size = None
        self._early_sum = None
        self._early_count = 0
        self._late_seen_in_step = 0
        self._late_triggered_in_step = 0
        self._proxy_step_bsz = 0
        self._proxy_step_faithful_sum = None
        self._proxy_step_harmful_sum = None
        self._proxy_step_faithful_count = None
        self._proxy_step_harmful_count = None
        self._rfhar_feats_base = None
        if rfhar_feats is not None:
            self.set_rfhar_feats(rfhar_feats)
        self._frgg_feats_base = None
        if frgg_feats is not None:
            self.set_frgg_feats(frgg_feats)
        self._frrs_feats_base = None
        if frrs_feats is not None:
            self.set_frrs_feats(frrs_feats)
        self._refresh_head_sets()
        self._refresh_oracle_roles()

    def end_generation(self) -> None:
        self._finalize_proxy_step()
        self._active = False
        self._step_kv_seq_len = None
        self._step_batch_size = None
        self._early_sum = None
        self._early_count = 0
        self._late_seen_in_step = 0
        self._late_triggered_in_step = 0
        self._current_sample_ids = []
        self._rfhar_feats_base = None
        self._frgg_feats_base = None
        self._frrs_feats_base = None
        self._proxy_step_bsz = 0
        self._proxy_step_faithful_sum = None
        self._proxy_step_harmful_sum = None
        self._proxy_step_faithful_count = None
        self._proxy_step_harmful_count = None

    def _in_early(self, layer_idx: int) -> bool:
        return int(self.config.ais_early_start) <= int(layer_idx) <= int(self.config.ais_early_end)

    def _in_late(self, layer_idx: int) -> bool:
        return int(self.config.ais_late_start) <= int(layer_idx) <= int(self.config.ais_late_end)

    def _in_rfhar_late(self, layer_idx: int) -> bool:
        return int(self.config.rfhar_late_start) <= int(layer_idx) <= int(self.config.rfhar_late_end)

    def _in_frgg_late(self, layer_idx: int) -> bool:
        return int(self.config.frgg_late_start) <= int(layer_idx) <= int(self.config.frgg_late_end)

    def _in_frrs_late(self, layer_idx: int) -> bool:
        return int(self.config.frrs_late_start) <= int(layer_idx) <= int(self.config.frrs_late_end)

    def _in_proxy_late(self, layer_idx: int) -> bool:
        return int(self.config.proxy_late_start) <= int(layer_idx) <= int(self.config.proxy_late_end)

    def _n_late_layers(self) -> int:
        lo = int(min(self.config.ais_late_start, self.config.ais_late_end))
        hi = int(max(self.config.ais_late_start, self.config.ais_late_end))
        return int(max(1, hi - lo + 1))

    def _rfhar_enabled(self) -> bool:
        return bool(self.config.enable_rfhar and float(self.config.rfhar_gamma) > 0.0)

    def _frgg_enabled(self) -> bool:
        return bool(self.config.enable_frgg and float(self.config.frgg_gamma) > 0.0)

    def _frrs_enabled(self) -> bool:
        return bool(self.config.enable_frrs and (float(self.config.frrs_alpha) > 0.0 or float(self.config.frrs_beta) > 0.0))

    def _proxy_trace_enabled(self) -> bool:
        return bool(self.config.enable_proxy_trace)

    def _init_proxy_step_state(self, bsz: int) -> None:
        self._proxy_step_bsz = int(bsz)
        self._proxy_step_faithful_sum = torch.zeros(int(bsz), dtype=torch.float32)
        self._proxy_step_harmful_sum = torch.zeros(int(bsz), dtype=torch.float32)
        self._proxy_step_faithful_count = torch.zeros(int(bsz), dtype=torch.float32)
        self._proxy_step_harmful_count = torch.zeros(int(bsz), dtype=torch.float32)

    def _finalize_proxy_step(self) -> None:
        if not self._proxy_trace_enabled():
            return
        if self._proxy_step_bsz <= 0:
            return
        if self._proxy_step_faithful_sum is None or self._proxy_step_harmful_sum is None:
            return
        if self._step_index < 0:
            return
        sample_ids = self._sample_ids_for_batch(self._proxy_step_bsz)
        faithful_sum = self._proxy_step_faithful_sum
        harmful_sum = self._proxy_step_harmful_sum
        faithful_count = self._proxy_step_faithful_count
        harmful_count = self._proxy_step_harmful_count
        for b in range(int(self._proxy_step_bsz)):
            f_cnt = float(faithful_count[b].item()) if faithful_count is not None else 0.0
            h_cnt = float(harmful_count[b].item()) if harmful_count is not None else 0.0
            faithful_mean = float(faithful_sum[b].item() / max(1.0, f_cnt))
            harmful_mean = float(harmful_sum[b].item() / max(1.0, h_cnt))
            self._proxy_trace_rows.append(
                {
                    "generation_idx": int(self._generation_index),
                    "step_idx": int(self._step_index),
                    "sample_id": str(sample_ids[b]).strip(),
                    "proxy_faithful_mean": faithful_mean,
                    "proxy_harmful_mean": harmful_mean,
                    "proxy_gap_mean": float(faithful_mean - harmful_mean),
                    "proxy_faithful_layer_count": int(round(f_cnt)),
                    "proxy_harmful_layer_count": int(round(h_cnt)),
                }
            )
        self._proxy_step_bsz = 0
        self._proxy_step_faithful_sum = None
        self._proxy_step_harmful_sum = None
        self._proxy_step_faithful_count = None
        self._proxy_step_harmful_count = None

    def _to_feat_2d_cpu(self, x: Any, name: str) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"RF-HAR feature {name} must be rank-2 [B,K_img], got {tuple(x.shape)}")
        return x.detach().to(device="cpu", dtype=torch.float32)

    def set_rfhar_feats(self, feats: Dict[str, Any]) -> None:
        if feats is None:
            self._rfhar_feats_base = None
            return
        required = ("C", "A", "D", "B")
        out: Dict[str, torch.Tensor] = {}
        for k in required:
            if k not in feats:
                raise KeyError(f"Missing RF-HAR feature key: {k}")
            out[k] = self._to_feat_2d_cpu(feats[k], k)
        shape = out["C"].shape
        for k in required:
            if out[k].shape != shape:
                raise ValueError(
                    f"RF-HAR feature shapes must match, got C={tuple(out['C'].shape)} "
                    f"A={tuple(out['A'].shape)} D={tuple(out['D'].shape)} B={tuple(out['B'].shape)}"
                )
        self._rfhar_feats_base = out

    def _get_rfhar_module(self) -> RFHAR:
        sig = (
            float(self.config.rfhar_gamma),
            float(self.config.rfhar_r_percent),
            float(self.config.rfhar_lambda_penalty),
            float(self.config.rfhar_eps),
        )
        if self._rfhar_module is None or self._rfhar_module_sig != sig:
            self._rfhar_module = RFHAR(
                gamma=float(self.config.rfhar_gamma),
                r_percent=float(self.config.rfhar_r_percent),
                lambda_penalty=float(self.config.rfhar_lambda_penalty),
                eps=float(self.config.rfhar_eps),
            )
            self._rfhar_module_sig = sig
        return self._rfhar_module

    def _resolve_rfhar_feats_for_batch(
        self,
        bsz: int,
        image_mask: torch.Tensor,  # [B,K] bool
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        base = self._rfhar_feats_base
        if base is None:
            return None
        required = ("C", "A", "D", "B")
        if any(k not in base for k in required):
            return None
        feat_b = int(base["C"].size(0))
        feat_k = int(base["C"].size(1))
        out: Dict[str, torch.Tensor] = {}
        for k in required:
            cur = base[k]
            if int(cur.size(0)) != feat_b or int(cur.size(1)) != feat_k:
                return None
            if feat_b != int(bsz):
                if feat_b > 0 and int(bsz) % feat_b == 0:
                    rep = int(bsz) // feat_b
                    cur = cur.repeat_interleave(rep, dim=0)
                else:
                    cur = cur[:1].expand(int(bsz), -1)
            out[k] = cur.to(device=device, dtype=dtype)

        img_counts = image_mask.sum(dim=-1).to(torch.int64)
        if int(img_counts.numel()) == 0:
            return None
        if not bool(torch.all(img_counts == int(feat_k)).item()):
            return None
        return out

    def set_frgg_feats(self, feats: Dict[str, Any]) -> None:
        if feats is None:
            self._frgg_feats_base = None
            return
        required = ("A", "C", "E")
        out: Dict[str, torch.Tensor] = {}
        for k in required:
            if k not in feats:
                raise KeyError(f"Missing FRGG feature key: {k}")
            out[k] = self._to_feat_2d_cpu(feats[k], k)
        shape = out["A"].shape
        for k in required:
            if out[k].shape != shape:
                raise ValueError(
                    f"FRGG feature shapes must match, got A={tuple(out['A'].shape)} "
                    f"C={tuple(out['C'].shape)} E={tuple(out['E'].shape)}"
                )
        self._frgg_feats_base = out

    def _get_frgg_module(self) -> FRGG:
        sig = (
            float(self.config.frgg_gamma),
            float(self.config.frgg_tau_c),
            float(self.config.frgg_tau_e),
            float(self.config.frgg_k_c),
            float(self.config.frgg_k_e),
            float(self.config.frgg_topk_ratio),
            float(self.config.frgg_eps),
            float(self.config.ais_eps),
        )
        if self._frgg_module is None or self._frgg_module_sig != sig:
            self._frgg_module = FRGG(
                gamma=float(self.config.frgg_gamma),
                tau_c=float(self.config.frgg_tau_c),
                tau_e=float(self.config.frgg_tau_e),
                k_c=float(self.config.frgg_k_c),
                k_e=float(self.config.frgg_k_e),
                topk_ratio=float(self.config.frgg_topk_ratio),
                eps=float(self.config.frgg_eps),
            )
            self._frgg_module_sig = sig
        return self._frgg_module

    def _resolve_frgg_feats_for_batch(
        self,
        bsz: int,
        image_mask: torch.Tensor,  # [B,K] bool
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        base = self._frgg_feats_base
        if base is None:
            return None
        required = ("A", "C", "E")
        if any(k not in base for k in required):
            return None
        feat_b = int(base["A"].size(0))
        feat_k = int(base["A"].size(1))
        out: Dict[str, torch.Tensor] = {}
        for k in required:
            cur = base[k]
            if int(cur.size(0)) != feat_b or int(cur.size(1)) != feat_k:
                return None
            if feat_b != int(bsz):
                if feat_b > 0 and int(bsz) % feat_b == 0:
                    rep = int(bsz) // feat_b
                    cur = cur.repeat_interleave(rep, dim=0)
                else:
                    cur = cur[:1].expand(int(bsz), -1)
            out[k] = cur.to(device=device, dtype=dtype)

        img_counts = image_mask.sum(dim=-1).to(torch.int64)
        if int(img_counts.numel()) == 0:
            return None
        if not bool(torch.all(img_counts == int(feat_k)).item()):
            return None
        return out

    def set_frrs_feats(self, feats: Dict[str, Any]) -> None:
        if feats is None:
            self._frrs_feats_base = None
            return
        required = ("A", "C", "E")
        out: Dict[str, torch.Tensor] = {}
        for k in required:
            if k not in feats:
                raise KeyError(f"Missing FRRS feature key: {k}")
            out[k] = self._to_feat_2d_cpu(feats[k], k)
        if "D" in feats and feats["D"] is not None:
            out["D"] = self._to_feat_2d_cpu(feats["D"], "D")
        shape = out["A"].shape
        for k in required:
            if out[k].shape != shape:
                raise ValueError(
                    f"FRRS feature shapes must match, got A={tuple(out['A'].shape)} "
                    f"C={tuple(out['C'].shape)} E={tuple(out['E'].shape)}"
                )
        if "D" in out and out["D"].shape != shape:
            raise ValueError(
                f"FRRS D shape mismatch, expected {tuple(shape)}, got {tuple(out['D'].shape)}"
            )
        self._frrs_feats_base = out

    def _get_frrs_module(self) -> FRRS:
        sig = (
            float(self.config.frrs_alpha),
            float(self.config.frrs_beta),
            float(self.config.frrs_tau_c),
            float(self.config.frrs_tau_e),
            float(self.config.frrs_k_c),
            float(self.config.frrs_k_e),
            float(self.config.frrs_topk_ratio),
            float(self.config.frrs_eps),
            float(self.config.ais_eps),
            str(self.config.frrs_arm),
            str(self.config.frrs_head_mode),
            float(self.config.frrs_r_percent),
        )
        if self._frrs_module is None or self._frrs_module_sig != sig:
            self._frrs_module = FRRS(
                alpha=float(self.config.frrs_alpha),
                beta=float(self.config.frrs_beta),
                tau_c=float(self.config.frrs_tau_c),
                tau_e=float(self.config.frrs_tau_e),
                k_c=float(self.config.frrs_k_c),
                k_e=float(self.config.frrs_k_e),
                topk_ratio=float(self.config.frrs_topk_ratio),
                eps=float(self.config.frrs_eps),
                arm=str(self.config.frrs_arm),
                head_mode=str(self.config.frrs_head_mode),
                r_percent=float(self.config.frrs_r_percent),
            )
            self._frrs_module_sig = sig
        return self._frrs_module

    def _resolve_frrs_feats_for_batch(
        self,
        bsz: int,
        image_mask: torch.Tensor,  # [B,K] bool
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        base = self._frrs_feats_base
        if base is None:
            return None
        required = ("A", "C", "E")
        if any(k not in base for k in required):
            return None
        feat_b = int(base["A"].size(0))
        feat_k = int(base["A"].size(1))
        out: Dict[str, torch.Tensor] = {}
        keys = ["A", "C", "E"] + (["D"] if "D" in base else [])
        for k in keys:
            cur = base[k]
            if int(cur.size(0)) != feat_b or int(cur.size(1)) != feat_k:
                return None
            if feat_b != int(bsz):
                if feat_b > 0 and int(bsz) % feat_b == 0:
                    rep = int(bsz) // feat_b
                    cur = cur.repeat_interleave(rep, dim=0)
                else:
                    cur = cur[:1].expand(int(bsz), -1)
            out[k] = cur.to(device=device, dtype=dtype)

        img_counts = image_mask.sum(dim=-1).to(torch.int64)
        if int(img_counts.numel()) == 0:
            return None
        if not bool(torch.all(img_counts == int(feat_k)).item()):
            return None
        return out

    def _recompute_frrs_feats_online(
        self,
        feats: Dict[str, torch.Tensor],
        attn_weights_last: torch.Tensor,  # [B,H,K]
        image_mask: torch.Tensor,  # [B,K] bool
        faithful_head_mask: Optional[torch.Tensor],  # [H] or [B,H]
        harmful_head_mask: Optional[torch.Tensor],  # [H] or [B,H]
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        eps = float(max(self.config.frrs_eps, 1e-9))
        bsz, n_heads, k_full = int(attn_weights_last.size(0)), int(attn_weights_last.size(1)), int(attn_weights_last.size(2))
        if int(image_mask.size(0)) != bsz or int(image_mask.size(1)) != k_full:
            return feats, {"frrs_online_feat_used": 0.0}

        k_img = int(feats["A"].size(1))
        img_counts = image_mask.sum(dim=-1).to(torch.int64)
        if not bool(torch.all(img_counts == int(k_img)).item()):
            return feats, {"frrs_online_feat_used": 0.0}

        attn = attn_weights_last.to(device=device, dtype=torch.float32)
        img_mask_f = image_mask[:, None, :].to(device=device, dtype=torch.float32)
        attn_img = attn * img_mask_f
        attn_img = attn_img / torch.clamp(attn_img.sum(dim=-1, keepdim=True), min=eps)

        idx_list = []
        for b in range(bsz):
            idx = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            if int(idx.numel()) != k_img:
                return feats, {"frrs_online_feat_used": 0.0}
            idx_list.append(idx)
        idx_tensor = torch.stack(idx_list, dim=0).to(device=device)  # [B,K_img]
        gather_idx = idx_tensor[:, None, :].expand(bsz, n_heads, k_img)
        attn_img_k = torch.gather(attn_img, dim=-1, index=gather_idx)  # [B,H,K_img]

        def _to_bh(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            m = mask.to(device=device, dtype=torch.float32)
            if m.dim() == 1:
                if int(m.numel()) != n_heads:
                    return None
                m = m[None, :].expand(bsz, n_heads)
            elif m.dim() == 2:
                if int(m.size(0)) == 1 and bsz > 1:
                    m = m.expand(bsz, int(m.size(1)))
                if int(m.size(0)) != bsz or int(m.size(1)) != n_heads:
                    return None
            else:
                return None
            return m

        hm_pos = _to_bh(faithful_head_mask)
        hm_neg = _to_bh(harmful_head_mask)

        head_mean = attn_img_k.mean(dim=1)  # [B,K_img]

        def _masked_head_mean(hm: Optional[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
            if hm is None:
                return fallback
            w = (hm > 0).to(torch.float32)
            den = w.sum(dim=1, keepdim=True)
            use = den > 0
            den = torch.clamp(den, min=1.0)
            v = (attn_img_k * w[:, :, None]).sum(dim=1) / den
            return torch.where(use, v, fallback)

        c_online = _masked_head_mean(hm_pos, head_mean)
        d_online = _masked_head_mean(hm_neg, torch.relu(head_mean - c_online))
        a_online = head_mean
        e_online = torch.relu(d_online - c_online)

        blend = float(max(0.0, min(1.0, self.config.frrs_online_blend)))
        out: Dict[str, torch.Tensor] = {}
        for k, v_online in (("A", a_online), ("C", c_online), ("D", d_online), ("E", e_online)):
            v_on = v_online.to(device=device, dtype=torch.float32)
            if k in feats:
                v_base = feats[k].to(device=device, dtype=torch.float32)
                if tuple(v_base.shape) != tuple(v_on.shape):
                    return feats, {"frrs_online_feat_used": 0.0}
                out[k] = ((1.0 - blend) * v_base + blend * v_on).to(dtype=dtype)
            else:
                out[k] = v_on.to(dtype=dtype)

        dbg = {
            "frrs_online_feat_used": 1.0,
            "frrs_online_blend": float(blend),
            "frrs_online_a_mean": float(a_online.mean().item()),
            "frrs_online_c_mean": float(c_online.mean().item()),
            "frrs_online_d_mean": float(d_online.mean().item()),
            "frrs_online_e_mean": float(e_online.mean().item()),
        }
        return out, dbg

    def _parse_head_spec_string(self, spec: str) -> Dict[int, set]:
        out: Dict[int, set] = {}
        s = str(spec or "").strip()
        if not s:
            return out
        for tok in s.split(","):
            t = tok.strip()
            if not t or ":" not in t:
                continue
            ls, hs = t.split(":", 1)
            try:
                li = int(ls.strip())
                hi = int(hs.strip())
            except Exception:
                continue
            out.setdefault(li, set()).add(hi)
        return out

    def _merge_head_maps(self, base: Dict[int, set], add: Dict[int, set]) -> Dict[int, set]:
        out: Dict[int, set] = {}
        for li, heads in base.items():
            out[int(li)] = set(int(h) for h in heads)
        for li, heads in add.items():
            out.setdefault(int(li), set()).update(int(h) for h in heads)
        return out

    def _parse_head_list_json_field(self, val: Any) -> Dict[int, set]:
        out: Dict[int, set] = {}
        if val is None:
            return out
        if isinstance(val, dict):
            for k, v in val.items():
                try:
                    li = int(k)
                except Exception:
                    continue
                if isinstance(v, list):
                    for h in v:
                        try:
                            out.setdefault(li, set()).add(int(h))
                        except Exception:
                            continue
            return out
        if isinstance(val, list):
            for x in val:
                if isinstance(x, str) and ":" in x:
                    m = self._parse_head_spec_string(x)
                    out = self._merge_head_maps(out, m)
                elif isinstance(x, dict):
                    li = x.get("layer")
                    hi = x.get("head")
                    if li is None or hi is None:
                        continue
                    try:
                        out.setdefault(int(li), set()).add(int(hi))
                    except Exception:
                        continue
            return out
        if isinstance(val, str):
            return self._parse_head_spec_string(val)
        return out

    def _load_headset_json(self, path: str) -> Tuple[Dict[int, set], Dict[int, set]]:
        if not path:
            return {}, {}
        p = str(path).strip()
        if not p:
            return {}, {}
        if not os.path.exists(p):
            return {}, {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            return {}, {}
        harmful = {}
        faithful = {}
        if isinstance(obj, dict):
            harmful = self._parse_head_list_json_field(obj.get("harmful_heads"))
            faithful = self._parse_head_list_json_field(obj.get("faithful_heads"))
        return harmful, faithful

    def _refresh_head_sets(self) -> None:
        harmful = self._parse_head_spec_string(self.config.ais_harmful_heads)
        faithful = self._parse_head_spec_string(self.config.ais_faithful_heads)
        js_harm, js_faith = self._load_headset_json(self.config.ais_headset_json)
        harmful = self._merge_head_maps(harmful, js_harm)
        faithful = self._merge_head_maps(faithful, js_faith)
        self._harmful_heads_map = harmful
        self._faithful_heads_map = faithful

    def _safe_int(self, x: Any) -> Optional[int]:
        try:
            if x is None or x == "":
                return None
            return int(x)
        except Exception:
            return None

    def _refresh_oracle_roles(self) -> None:
        if not bool(self.config.ais_use_oracle_roles):
            self._oracle_supportive_map = {}
            self._oracle_assertive_map = {}
            self._oracle_loaded_sig = None
            return
        p = str(self.config.ais_oracle_role_csv or "").strip()
        sup_k = int(self.config.ais_oracle_supportive_topk)
        ass_k = int(self.config.ais_oracle_assertive_topk)
        sig = (p, sup_k, ass_k)
        if self._oracle_loaded_sig == sig:
            return
        self._oracle_loaded_sig = sig
        self._oracle_supportive_map = {}
        self._oracle_assertive_map = {}
        if p == "" or (not os.path.isfile(p)):
            return

        sup_tmp: Dict[str, List[Tuple[int, int]]] = {}
        ass_tmp: Dict[str, List[Tuple[int, int]]] = {}
        try:
            with open(p, "r", encoding="utf-8", newline="") as f:
                rd = csv.DictReader(f)
                for row in rd:
                    sid = str(row.get("id") or "").strip()
                    if sid == "":
                        continue
                    pi = self._safe_int(row.get("candidate_patch_idx"))
                    if pi is None:
                        continue
                    rank = self._safe_int(row.get("candidate_rank"))
                    if rank is None:
                        rank = 10**9
                    role = str(row.get("role_label") or "").strip().lower()
                    if role == "supportive":
                        sup_tmp.setdefault(sid, []).append((int(rank), int(pi)))
                    elif role in {"harmful", "assertive"}:
                        ass_tmp.setdefault(sid, []).append((int(rank), int(pi)))
        except Exception:
            self._oracle_supportive_map = {}
            self._oracle_assertive_map = {}
            return

        def _finalize(mp: Dict[str, List[Tuple[int, int]]], topk: int) -> Dict[str, List[int]]:
            out: Dict[str, List[int]] = {}
            for sid, arr in mp.items():
                arr2 = sorted(arr, key=lambda x: (int(x[0]), int(x[1])))
                seen = set()
                keep: List[int] = []
                for _, pi in arr2:
                    if int(pi) in seen:
                        continue
                    keep.append(int(pi))
                    seen.add(int(pi))
                    if int(topk) > 0 and len(keep) >= int(topk):
                        break
                out[sid] = keep
            return out

        self._oracle_supportive_map = _finalize(sup_tmp, sup_k)
        self._oracle_assertive_map = _finalize(ass_tmp, ass_k)

    def _sample_ids_for_batch(self, bsz: int) -> List[str]:
        if len(self._current_sample_ids) == int(bsz):
            return list(self._current_sample_ids)
        if len(self._current_sample_ids) == 1:
            return [self._current_sample_ids[0] for _ in range(int(bsz))]
        if len(self._current_sample_ids) > int(bsz):
            return list(self._current_sample_ids[: int(bsz)])
        return ["" for _ in range(int(bsz))]

    def _resolve_oracle_patch_mask_for_kv(
        self,
        kind: str,
        bsz: int,
        kv_seq_len: int,
        image_mask: torch.Tensor,  # [B,K] bool over sequence columns
    ) -> torch.Tensor:
        out = torch.zeros((int(bsz), int(kv_seq_len)), dtype=torch.bool, device=image_mask.device)
        if kind == "supportive":
            mp = self._oracle_supportive_map
        else:
            mp = self._oracle_assertive_map
        sample_ids = self._sample_ids_for_batch(bsz)
        for b in range(int(bsz)):
            sid = str(sample_ids[b]).strip()
            if sid == "":
                continue
            patch_list = mp.get(sid, None)
            if not patch_list:
                continue
            img_pos = torch.nonzero(image_mask[b], as_tuple=False).flatten()
            nimg = int(img_pos.numel())
            for pi in patch_list:
                p = int(pi)
                if 0 <= p < nimg:
                    seq_col = int(img_pos[p].item())
                    out[b, seq_col] = True
                elif 0 <= p < int(kv_seq_len):
                    # Fallback if indices are already sequence columns.
                    if bool(image_mask[b, p].item()):
                        out[b, p] = True
        return out

    def _head_mask_for_layer(
        self,
        layer_idx: int,
        n_heads: int,
        kind: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if kind == "harmful":
            mp = self._harmful_heads_map
        else:
            mp = self._faithful_heads_map
        hs = mp.get(int(layer_idx), None)
        if hs is None or len(hs) == 0:
            return None
        v = torch.zeros((int(n_heads),), dtype=dtype, device=device)
        for h in hs:
            hi = int(h)
            if 0 <= hi < int(n_heads):
                v[hi] = 1.0
        return v

    def _top_ratio_mask(
        self,
        scores: torch.Tensor,  # [B,H]
        ratio: float,
    ) -> torch.Tensor:
        bsz, n_heads = int(scores.size(0)), int(scores.size(1))
        r = float(max(0.0, min(1.0, ratio)))
        if r <= 0.0 or n_heads <= 0:
            return torch.zeros_like(scores)
        k = int(max(1, round(r * n_heads)))
        k = int(min(k, n_heads))
        vals, idx = torch.topk(scores, k=k, dim=-1)
        out = torch.zeros_like(scores)
        # Keep only strictly-positive heads among top-k to avoid zero-tie expansion.
        out.scatter_(1, idx, (vals > 0).to(scores.dtype))
        return out

    def _normalize_last_dim(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        s = x.sum(dim=-1, keepdim=True)
        return x / (s + float(eps))

    def _topk_patch_dist(
        self,
        scores: torch.Tensor,  # [B,K]
        image_mask: torch.Tensor,  # [B,K] bool
        topk: int,
        eps: float,
    ) -> torch.Tensor:
        bsz, kv = int(scores.size(0)), int(scores.size(1))
        if kv <= 0:
            return torch.zeros_like(scores)
        k = int(topk)
        if k <= 0 or k >= kv:
            s = scores * image_mask.to(dtype=scores.dtype)
            return self._normalize_last_dim(s, eps=eps)
        neg = torch.finfo(scores.dtype).min
        masked_scores = torch.where(image_mask, scores, torch.full_like(scores, neg))
        topk_idx = torch.topk(masked_scores, k=k, dim=-1).indices  # [B,k]
        sel = torch.zeros_like(scores)
        sel.scatter_(1, topk_idx, 1.0)
        s = scores * sel * image_mask.to(dtype=scores.dtype)
        return self._normalize_last_dim(s, eps=eps)

    def _topk_patch_binary_mask_3d(
        self,
        scores: torch.Tensor,  # [B,H,K]
        image_mask: torch.Tensor,  # [B,K] bool
        topk: int,
    ) -> torch.Tensor:
        bsz, n_heads, kv = int(scores.size(0)), int(scores.size(1)), int(scores.size(2))
        if kv <= 0:
            return torch.zeros_like(scores)
        img = image_mask[:, None, :].to(dtype=scores.dtype)
        k = int(topk)
        if k <= 0 or k >= kv:
            return img.expand(bsz, n_heads, kv)

        neg = torch.finfo(scores.dtype).min
        masked_scores = torch.where(img > 0, scores, torch.full_like(scores, neg))
        topk_idx = torch.topk(masked_scores, k=k, dim=-1).indices  # [B,H,k]
        sel = torch.zeros_like(scores)
        sel.scatter_(-1, topk_idx, 1.0)
        return sel * img

    def should_intercept_layer(self, layer_idx: Optional[int]) -> bool:
        if not self.active:
            return False
        if layer_idx is None:
            return False
        li = int(layer_idx)
        ais_needed = bool(self.config.enable_ais_gating and (self._in_early(li) or self._in_late(li)))
        rfhar_needed = bool(self._rfhar_enabled() and self._in_rfhar_late(li))
        frgg_needed = bool(self._frgg_enabled() and self._in_frgg_late(li))
        frrs_needed = bool(self._frrs_enabled() and self._in_frrs_late(li))
        proxy_needed = bool(self._proxy_trace_enabled() and self._in_proxy_late(li))
        probe_needed = bool(
            str(self.config.path_probe_mode or "none").strip().lower() != "none"
            and float(self.config.path_probe_penalty) > 0.0
            and (self._in_early(li) or self._in_late(li))
        )
        return bool(ais_needed or rfhar_needed or frgg_needed or frrs_needed or proxy_needed or probe_needed)

    def _begin_step_if_needed(self, kv_seq_len: int, bsz: int) -> None:
        if self._step_kv_seq_len == int(kv_seq_len) and self._step_batch_size == int(bsz):
            return
        self._finalize_proxy_step()
        self._step_index += 1
        self._step_kv_seq_len = int(kv_seq_len)
        self._step_batch_size = int(bsz)
        self._early_sum = None
        self._early_count = 0
        self._late_seen_in_step = 0
        self._late_triggered_in_step = 0
        if self._proxy_trace_enabled():
            self._init_proxy_step_state(int(bsz))

    def _resolve_image_mask_for_kv(
        self,
        bsz: int,
        kv_seq_len: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self._base_image_mask is None:
            return None

        base = self._base_image_mask.to(device=device, dtype=torch.bool)
        if int(base.size(0)) != int(bsz):
            if int(base.size(0)) > 0 and int(bsz) % int(base.size(0)) == 0:
                rep = int(bsz) // int(base.size(0))
                base = base.repeat_interleave(rep, dim=0)
            else:
                base = base[:1].expand(int(bsz), -1)

        base_len = int(base.size(1))
        if base_len < int(kv_seq_len):
            pad = torch.zeros((int(bsz), int(kv_seq_len) - base_len), dtype=torch.bool, device=device)
            base = torch.cat([base, pad], dim=1)
        elif base_len > int(kv_seq_len):
            base = base[:, : int(kv_seq_len)]

        if attention_mask is not None:
            # Causal 4D mask where valid entries are finite.
            valid = torch.isfinite(attention_mask[:, 0, -1, :]).to(torch.bool)
            base = base & valid

        return base

    def _topk_mean_over_image(
        self,
        x: torch.Tensor,  # [B, K]
        image_mask: torch.Tensor,  # [B, K] bool
        k: int,
    ) -> torch.Tensor:
        bsz, kv = int(x.size(0)), int(x.size(1))
        k_eff = int(max(1, min(int(k), kv)))
        neg = torch.finfo(x.dtype).min
        x_masked = torch.where(image_mask, x, torch.full_like(x, neg))
        topk_vals = torch.topk(x_masked, k=k_eff, dim=-1).values
        finite = torch.isfinite(topk_vals)
        denom = torch.clamp(image_mask.sum(dim=-1), min=1, max=k_eff).to(x.dtype)
        num = torch.where(finite, topk_vals, torch.zeros_like(topk_vals)).sum(dim=-1)
        return num / (denom + float(self.config.ais_eps))

    def record_proxy_attn(
        self,
        layer_idx: int,
        attn_weights_last: Optional[torch.Tensor],  # [B,H,K]
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        if not bool(self._proxy_trace_enabled() and self._in_proxy_late(int(layer_idx))):
            return
        if attn_weights_last is None or attn_weights_last.dim() != 3:
            return
        bsz, _n_heads, kv_seq_len = (
            int(attn_weights_last.size(0)),
            int(attn_weights_last.size(1)),
            int(attn_weights_last.size(2)),
        )
        self._begin_step_if_needed(kv_seq_len=kv_seq_len, bsz=bsz)
        if self._proxy_step_faithful_sum is None or self._proxy_step_bsz != int(bsz):
            self._init_proxy_step_state(int(bsz))

        image_mask = self._resolve_image_mask_for_kv(
            bsz=bsz,
            kv_seq_len=kv_seq_len,
            attention_mask=attention_mask,
            device=attn_weights_last.device,
        )
        if image_mask is None or int(image_mask.sum().item()) == 0:
            return

        if attention_mask is not None:
            valid_mask = torch.isfinite(attention_mask[:, 0, -1, :]).to(torch.bool)
        else:
            valid_mask = torch.ones_like(image_mask, dtype=torch.bool)
        text_mask = valid_mask & (~image_mask)

        attn = attn_weights_last.to(torch.float32)
        img = image_mask[:, None, :].to(dtype=attn.dtype)
        txt = text_mask[:, None, :].to(dtype=attn.dtype)
        vis_sum = (attn * img).sum(dim=-1)
        txt_sum = (attn * txt).sum(dim=-1)
        ratio = vis_sum / torch.clamp(vis_sum + txt_sum, min=float(max(self.config.proxy_eps, 1e-9)))

        faithful_heads = sorted(self._faithful_heads_map.get(int(layer_idx), set()))
        harmful_heads = sorted(self._harmful_heads_map.get(int(layer_idx), set()))

        if faithful_heads:
            idx = torch.as_tensor(faithful_heads, dtype=torch.long, device=ratio.device)
            idx = idx[(idx >= 0) & (idx < int(ratio.size(1)))]
            if int(idx.numel()) > 0:
                faithful_mean = ratio.index_select(1, idx).mean(dim=1).detach().cpu()
                self._proxy_step_faithful_sum += faithful_mean
                self._proxy_step_faithful_count += 1.0

        if harmful_heads:
            idx = torch.as_tensor(harmful_heads, dtype=torch.long, device=ratio.device)
            idx = idx[(idx >= 0) & (idx < int(ratio.size(1)))]
            if int(idx.numel()) > 0:
                harmful_mean = ratio.index_select(1, idx).mean(dim=1).detach().cpu()
                self._proxy_step_harmful_sum += harmful_mean
                self._proxy_step_harmful_count += 1.0

    def compute_bias(
        self,
        layer_idx: int,
        attn_logits_masked: torch.Tensor,  # [B,H,Q,K], includes attn mask
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Returns additive penalty tensor (same shape as attn_logits_masked) to subtract from logits.
        Only image columns in late layers get non-zero values, only at last query position.
        """
        li = int(layer_idx)
        bsz, n_heads, q_len, kv_seq_len = (
            int(attn_logits_masked.size(0)),
            int(attn_logits_masked.size(1)),
            int(attn_logits_masked.size(2)),
            int(attn_logits_masked.size(3)),
        )
        self._begin_step_if_needed(kv_seq_len=kv_seq_len, bsz=bsz)
        is_ais_region = bool(self._in_early(li) or self._in_late(li))
        is_rfhar_region = bool(self._rfhar_enabled() and self._in_rfhar_late(li))
        is_frgg_region = bool(self._frgg_enabled() and self._in_frgg_late(li))
        if not is_ais_region and not is_rfhar_region and not is_frgg_region:
            return None

        image_mask = self._resolve_image_mask_for_kv(
            bsz=bsz,
            kv_seq_len=kv_seq_len,
            attention_mask=attention_mask,
            device=attn_logits_masked.device,
        )
        if image_mask is None or int(image_mask.sum().item()) == 0:
            return None

        # FRGG branch: late-layer, image-columns-only, faithful-head guidance.
        if is_frgg_region:
            feats = self._resolve_frgg_feats_for_batch(
                bsz=bsz,
                image_mask=image_mask,
                device=attn_logits_masked.device,
                dtype=attn_logits_masked.dtype,
            )
            if feats is None:
                return None
            faithful_head_mask = self._head_mask_for_layer(
                layer_idx=li,
                n_heads=n_heads,
                kind="faithful",
                device=attn_logits_masked.device,
                dtype=attn_logits_masked.dtype,
            )
            if faithful_head_mask is None:
                return None
            frgg = self._get_frgg_module()
            z_last = attn_logits_masked[:, :, -1, :]
            z_mod, dbg = frgg(
                attn_logits_last=z_last,
                image_mask=image_mask,
                feats=feats,
                faithful_head_mask=faithful_head_mask,
            )
            bias_last = z_last - z_mod
            if float(bias_last.abs().sum().item()) == 0.0:
                return None
            bias = torch.zeros_like(attn_logits_masked)
            bias[:, :, -1, :] = bias_last.to(dtype=attn_logits_masked.dtype)
            self._late_seen_in_step += 1
            self._late_triggered_in_step += int((bias_last.abs().sum(dim=(-1, -2)) > 0).any().item())

            if bool(self.config.frgg_debug_log or self.config.ais_debug_log):
                img_cnt = torch.clamp(image_mask.sum(dim=-1).to(dtype=bias_last.dtype), min=1.0)
                row = {
                    "generation_idx": int(self._generation_index),
                    "step_idx": int(self._step_index),
                    "layer_idx": int(li),
                    "frgg_mode": True,
                    "frgg_gamma": float(self.config.frgg_gamma),
                    "frgg_tau_c": float(self.config.frgg_tau_c),
                    "frgg_tau_e": float(self.config.frgg_tau_e),
                    "frgg_topk_ratio": float(self.config.frgg_topk_ratio),
                    "frgg_gate_mean": float(dbg.get("frgg_gate_mean", 0.0)),
                    "frgg_gate_min": float(dbg.get("frgg_gate_min", 0.0)),
                    "frgg_gate_max": float(dbg.get("frgg_gate_max", 0.0)),
                    "frgg_prior_entropy": float(dbg.get("frgg_prior_entropy", 0.0)),
                    "frgg_faithful_head_cov": float(dbg.get("frgg_faithful_head_cov", 0.0)),
                    "frgg_delta_abs_mean": float(dbg.get("frgg_delta_abs_mean", 0.0)),
                    "frgg_active_frac": float(dbg.get("frgg_active_frac", 0.0)),
                    "trigger_frac_batch": float((bias_last.abs().sum(dim=(-1, -2)) > 0).to(torch.float32).mean().item()),
                    "late_layers_seen_in_step": int(self._late_seen_in_step),
                    "late_layers_triggered_in_step": int(self._late_triggered_in_step),
                    "late_trigger_fraction_step": float(
                        float(self._late_triggered_in_step) / float(max(1, self._late_seen_in_step))
                    ),
                    "image_cols_per_batch_mean": float(img_cnt.mean().item()),
                    "frgg_bias_abs_mean": float((bias_last.abs().sum(dim=-1) / img_cnt[:, None]).mean().item()),
                }
                self._debug_rows.append(row)
            return bias

        # RF-HAR branch: last-query, late-layer, image-columns-only logit editing.
        if is_rfhar_region:
            feats = self._resolve_rfhar_feats_for_batch(
                bsz=bsz,
                image_mask=image_mask,
                device=attn_logits_masked.device,
                dtype=attn_logits_masked.dtype,
            )
            if feats is None:
                return None
            rfhar = self._get_rfhar_module()
            z_last = attn_logits_masked[:, :, -1, :]
            z_mod, dbg = rfhar(
                attn_logits_last=z_last,
                image_mask=image_mask,
                feats=feats,
            )
            # Runtime expects a subtractive bias: z_new = z - bias.
            bias_last = z_last - z_mod
            if float(bias_last.abs().sum().item()) == 0.0:
                return None
            bias = torch.zeros_like(attn_logits_masked)
            bias[:, :, -1, :] = bias_last.to(dtype=attn_logits_masked.dtype)
            self._late_seen_in_step += 1
            self._late_triggered_in_step += int((bias_last.abs().sum(dim=(-1, -2)) > 0).any().item())

            if bool(self.config.rfhar_debug_log or self.config.ais_debug_log):
                img_cnt = torch.clamp(image_mask.sum(dim=-1).to(dtype=bias_last.dtype), min=1.0)
                row = {
                    "generation_idx": int(self._generation_index),
                    "step_idx": int(self._step_index),
                    "layer_idx": int(li),
                    "rfhar_mode": True,
                    "rfhar_gamma": float(self.config.rfhar_gamma),
                    "rfhar_r_percent": float(self.config.rfhar_r_percent),
                    "rfhar_lambda_penalty": float(self.config.rfhar_lambda_penalty),
                    "rfhar_rf_mean": float(dbg.get("rf_mean", 0.0)),
                    "rfhar_rf_max": float(dbg.get("rf_max", 0.0)),
                    "rfhar_rf_min": float(dbg.get("rf_min", 0.0)),
                    "rfhar_m_pos_mean": float(dbg.get("m_pos_mean", 0.0)),
                    "rfhar_m_neg_mean": float(dbg.get("m_neg_mean", 0.0)),
                    "rfhar_disjoint_ok": float(dbg.get("disjoint_ok", 0.0)),
                    "rfhar_delta_abs_mean": float(dbg.get("delta_abs_mean", 0.0)),
                    "trigger_frac_batch": float((bias_last.abs().sum(dim=(-1, -2)) > 0).to(torch.float32).mean().item()),
                    "late_layers_seen_in_step": int(self._late_seen_in_step),
                    "late_layers_triggered_in_step": int(self._late_triggered_in_step),
                    "late_trigger_fraction_step": float(
                        float(self._late_triggered_in_step) / float(max(1, self._late_seen_in_step))
                    ),
                    "image_cols_per_batch_mean": float(img_cnt.mean().item()),
                    "rfhar_bias_abs_mean": float((bias_last.abs().sum(dim=-1) / img_cnt[:, None]).mean().item()),
                }
                self._debug_rows.append(row)
            return bias

        oracle_mode = bool(self.config.ais_use_oracle_roles and self.config.enable_ais_gating)
        if oracle_mode:
            # Oracle mode: late layers only, image columns only, training-free fixed role masks.
            if not self._in_late(li):
                return None
            eps = float(max(float(self.config.ais_eps), 1e-12))
            img_mask_f = image_mask[:, None, :].to(dtype=attn_logits_masked.dtype)
            supportive_cols = self._resolve_oracle_patch_mask_for_kv(
                kind="supportive",
                bsz=bsz,
                kv_seq_len=kv_seq_len,
                image_mask=image_mask,
            )
            assertive_cols = self._resolve_oracle_patch_mask_for_kv(
                kind="assertive",
                bsz=bsz,
                kv_seq_len=kv_seq_len,
                image_mask=image_mask,
            )
            sup_cols_f = supportive_cols[:, None, :].to(dtype=attn_logits_masked.dtype)
            ass_cols_f = assertive_cols[:, None, :].to(dtype=attn_logits_masked.dtype)

            harmful_head_mask = self._head_mask_for_layer(
                layer_idx=li,
                n_heads=n_heads,
                kind="harmful",
                device=attn_logits_masked.device,
                dtype=attn_logits_masked.dtype,
            )
            faithful_head_mask = self._head_mask_for_layer(
                layer_idx=li,
                n_heads=n_heads,
                kind="faithful",
                device=attn_logits_masked.device,
                dtype=attn_logits_masked.dtype,
            )
            harmful_h = (
                torch.zeros((bsz, n_heads), dtype=attn_logits_masked.dtype, device=attn_logits_masked.device)
                if harmful_head_mask is None
                else harmful_head_mask[None, :].expand(bsz, n_heads)
            )
            faithful_h = (
                torch.zeros((bsz, n_heads), dtype=attn_logits_masked.dtype, device=attn_logits_masked.device)
                if faithful_head_mask is None
                else faithful_head_mask[None, :].expand(bsz, n_heads)
            )

            lam_pos = float(max(0.0, self.config.ais_oracle_lambda_pos))
            lam_neg = float(max(0.0, self.config.ais_oracle_lambda_neg))
            plus = float(lam_pos) * faithful_h[:, :, None] * sup_cols_f
            minus = float(lam_neg) * harmful_h[:, :, None] * ass_cols_f

            arm = str(self.config.ais_arm or "bipolar").strip().lower()
            if arm == "harmful_only":
                bias_last = minus
            elif arm == "faithful_only":
                bias_last = -plus
            else:
                # bipolar + legacy fallback
                bias_last = minus - plus

            clip = float(max(0.0, self.config.ais_oracle_bias_clip))
            if clip > 0.0:
                bias_last = torch.clamp(bias_last, min=-clip, max=clip)
            bias_last = bias_last * img_mask_f

            # Optional path-probe diagnostic bias (late layers only).
            path_probe_mode = str(self.config.path_probe_mode or "none").strip().lower()
            path_probe_penalty = float(max(0.0, self.config.path_probe_penalty))
            path_probe_applied = False
            probe_img_bias = torch.zeros_like(bias_last)
            probe_text_bias = torch.zeros_like(bias_last)
            if (
                path_probe_mode in {"drop_img", "drop_text", "drop_both"}
                and path_probe_penalty > 0.0
                and self._in_late(li)
            ):
                if (not bool(self.config.path_probe_first_step_only)) or int(self._step_index) == 0:
                    path_probe_applied = True
                    if attention_mask is not None:
                        valid_cols = torch.isfinite(attention_mask[:, 0, -1, :]).to(torch.bool)
                    else:
                        valid_cols = torch.ones((bsz, kv_seq_len), dtype=torch.bool, device=bias_last.device)
                    text_mask = valid_cols & (~image_mask)
                    if path_probe_mode in {"drop_img", "drop_both"}:
                        probe_img_bias = float(path_probe_penalty) * img_mask_f
                    if path_probe_mode in {"drop_text", "drop_both"}:
                        probe_text_bias = float(path_probe_penalty) * text_mask[:, None, :].to(dtype=bias_last.dtype)
                    bias_last = bias_last + probe_img_bias + probe_text_bias

            bias = torch.zeros_like(attn_logits_masked)
            bias[:, :, -1, :] = bias_last.to(dtype=attn_logits_masked.dtype)

            has_signal = ((sup_cols_f.sum(dim=-1) + ass_cols_f.sum(dim=-1)) > 0).to(attn_logits_masked.dtype)
            trigger = has_signal.max(dim=-1).values  # [B]
            self._late_seen_in_step += 1
            self._late_triggered_in_step += int((trigger > 0).any().item())

            if bool(self.config.ais_debug_log):
                img_cnt = torch.clamp(image_mask.sum(dim=-1).to(dtype=bias_last.dtype), min=1.0)
                if attention_mask is not None:
                    valid_cols_dbg = torch.isfinite(attention_mask[:, 0, -1, :]).to(torch.bool)
                else:
                    valid_cols_dbg = torch.ones((bsz, kv_seq_len), dtype=torch.bool, device=bias_last.device)
                text_mask_dbg = valid_cols_dbg & (~image_mask)
                text_cnt = torch.clamp(text_mask_dbg.sum(dim=-1).to(dtype=bias_last.dtype), min=1.0)
                harmful_penalty = minus * img_mask_f
                faithful_boost = plus * img_mask_f
                per_b_head = bias_last.sum(dim=-1) / img_cnt[:, None]
                per_b_head_abs = bias_last.abs().sum(dim=-1) / img_cnt[:, None]
                penalty_img_mean = float(per_b_head.mean().item())
                penalty_img_abs_mean = float(per_b_head_abs.mean().item())
                harmful_mean = float((harmful_penalty.sum(dim=-1) / img_cnt[:, None]).mean().item())
                faithful_mean = float((faithful_boost.sum(dim=-1) / img_cnt[:, None]).mean().item())
                probe_img_mean = float((probe_img_bias.sum(dim=-1) / img_cnt[:, None]).mean().item())
                probe_text_mean = float(((probe_text_bias.sum(dim=-1) / text_cnt[:, None]).mean()).item())

                harmful_selected_dbg = (harmful_h > 0).to(dtype=bias_last.dtype)
                harmful_patch_mask_dbg = ass_cols_f.expand(bsz, n_heads, kv_seq_len)
                selected_cells = (harmful_selected_dbg[:, :, None] > 0).to(dtype=harmful_patch_mask_dbg.dtype) * (
                    harmful_patch_mask_dbg > 0
                ).to(dtype=harmful_patch_mask_dbg.dtype)
                selected_heads_cnt = torch.clamp(
                    (harmful_selected_dbg > 0).to(torch.float32).sum(dim=-1),
                    min=1.0,
                )
                selected_patch_per_head = (
                    selected_cells.to(torch.float32).sum(dim=-1).sum(dim=-1) / selected_heads_cnt
                )
                selected_patch_any = selected_cells.to(torch.float32).amax(dim=1)
                selected_patch_cov = selected_patch_any.sum(dim=-1) / img_cnt
                sel_pos = harmful_penalty[harmful_penalty > 0]
                harmful_per_cell_dose = float(sel_pos.mean().item()) if int(sel_pos.numel()) > 0 else 0.0
                head_idx_b0 = (
                    torch.nonzero((harmful_h[0] > 0), as_tuple=False).flatten().tolist() if bsz > 0 else []
                )
                patch_idx_b0 = (
                    torch.nonzero((ass_cols_f[0, 0] > 0), as_tuple=False).flatten().tolist() if bsz > 0 else []
                )
                sup_patch_idx_b0 = (
                    torch.nonzero((sup_cols_f[0, 0] > 0), as_tuple=False).flatten().tolist() if bsz > 0 else []
                )
                row = {
                    "generation_idx": int(self._generation_index),
                    "step_idx": int(self._step_index),
                    "layer_idx": int(li),
                    "ais_arm": arm,
                    "oracle_mode": True,
                    "trigger_frac_batch": float((trigger > 0).to(torch.float32).mean().item()),
                    "late_layers_seen_in_step": int(self._late_seen_in_step),
                    "late_layers_triggered_in_step": int(self._late_triggered_in_step),
                    "late_trigger_fraction_step": float(
                        float(self._late_triggered_in_step) / float(max(1, self._late_seen_in_step))
                    ),
                    "penalty_img_mean": penalty_img_mean,
                    "penalty_img_abs_mean": penalty_img_abs_mean,
                    "harmful_penalty_img_mean": harmful_mean,
                    "faithful_boost_img_mean": faithful_mean,
                    "image_cols_per_batch_mean": float(image_mask.to(torch.float32).sum(dim=-1).mean().item()),
                    "early_count": int(self._early_count),
                    "n_harmful_heads_layer": int(len(self._harmful_heads_map.get(li, set()))),
                    "n_faithful_heads_layer": int(len(self._faithful_heads_map.get(li, set()))),
                    "oracle_supportive_cols_mean": float(supportive_cols.to(torch.float32).sum(dim=-1).mean().item()),
                    "oracle_assertive_cols_mean": float(assertive_cols.to(torch.float32).sum(dim=-1).mean().item()),
                    "oracle_lambda_pos": float(lam_pos),
                    "oracle_lambda_neg": float(lam_neg),
                    "oracle_bias_clip": float(clip),
                    "budget_mode": False,
                    "ais_operator": "oracle",
                    "ais_semihard_penalty": 0.0,
                    "budget_layer": 0.0,
                    "harmful_selected_heads_mean": float((harmful_h > 0).to(torch.float32).sum(dim=-1).mean().item()),
                    "harmful_selected_patch_per_head_mean": float(selected_patch_per_head.mean().item()),
                    "harmful_selected_patch_coverage_mean": float(selected_patch_cov.mean().item()),
                    "harmful_per_cell_dose_mean": harmful_per_cell_dose,
                    "applied_harmful_head_idx_b0": "|".join(str(int(x)) for x in head_idx_b0),
                    "applied_harmful_patch_idx_b0": "|".join(str(int(x)) for x in patch_idx_b0),
                    "applied_supportive_patch_idx_b0": "|".join(str(int(x)) for x in sup_patch_idx_b0),
                    "faithful_selected_heads_mean": float(
                        (((faithful_h > 0).to(torch.float32).sum(dim=-1)).mean().item())
                    ),
                    "budget_patch_topk": 0,
                    "path_probe_mode": path_probe_mode,
                    "path_probe_penalty": float(path_probe_penalty),
                    "path_probe_first_step_only": bool(self.config.path_probe_first_step_only),
                    "path_probe_applied": bool(path_probe_applied),
                    "path_probe_img_penalty_mean": probe_img_mean,
                    "path_probe_text_penalty_mean": probe_text_mean,
                    "oracle_sample_ids": "|".join(self._sample_ids_for_batch(min(bsz, 4))),
                }
                self._debug_rows.append(row)

            return bias

        eps = float(max(float(self.config.ais_eps), 1e-12))
        # alpha_raw: before AIS gating, after causal/padding masking.
        alpha_raw = torch.softmax(attn_logits_masked.float(), dim=-1).to(attn_logits_masked.dtype)
        alpha_last = alpha_raw[:, :, -1, :]  # [B,H,K]
        img_mask_f = image_mask[:, None, :].to(dtype=alpha_last.dtype)

        # Image-only normalization.
        alpha_img = alpha_last * img_mask_f
        alpha_img_norm = alpha_img / (alpha_img.sum(dim=-1, keepdim=True) + eps)
        support_now = alpha_img_norm.mean(dim=1)  # [B,K]

        if self._in_early(li):
            # Model may be sharded across multiple GPUs (device_map=auto). Keep cross-layer
            # accumulation on CPU to avoid device-mismatch when summing early supports.
            support_cpu = support_now.detach().to(device="cpu", dtype=torch.float32)
            if self._early_sum is None:
                self._early_sum = support_cpu.clone()
            else:
                self._early_sum = self._early_sum + support_cpu
            self._early_count += 1

        if not self._in_late(li):
            return None

        if self._early_sum is not None and int(self._early_count) > 0:
            early_support = (self._early_sum / float(self._early_count)).to(
                device=attn_logits_masked.device,
                dtype=support_now.dtype,
            )
        else:
            # Fallback when early layers are not available in the current config.
            early_support = support_now.detach()

        late_support = support_now
        a = torch.log((late_support + eps) / (early_support + eps))
        a = a * image_mask.to(dtype=a.dtype)
        relu_a = F.relu(a)

        ais = self._topk_mean_over_image(
            x=a,
            image_mask=image_mask,
            k=int(max(1, int(self.config.ais_topk))),
        )  # [B]
        trigger = (ais > float(self.config.ais_tau)).to(dtype=a.dtype)  # [B]

        # u_{l,h,t}: dynamic harmfulness proxy
        u = (alpha_img_norm * relu_a[:, None, :]).sum(dim=-1)  # [B,H]
        u_centered = F.relu(u - u.mean(dim=-1, keepdim=True))
        omega = u_centered / (u_centered.amax(dim=-1, keepdim=True) + eps)  # [B,H]

        # faithful score proxy per head:
        # high when head attends to early-supported but currently under-used patches.
        preserve = F.relu(early_support - late_support) * image_mask.to(dtype=late_support.dtype)  # [B,K]
        preserve_norm = self._normalize_last_dim(preserve, eps=eps)
        v = (alpha_img_norm * preserve_norm[:, None, :]).sum(dim=-1)  # [B,H]

        arm = str(self.config.ais_arm or "legacy").strip().lower()
        harmful_head_mask = self._head_mask_for_layer(
            layer_idx=li,
            n_heads=n_heads,
            kind="harmful",
            device=attn_logits_masked.device,
            dtype=omega.dtype,
        )
        faithful_head_mask = self._head_mask_for_layer(
            layer_idx=li,
            n_heads=n_heads,
            kind="faithful",
            device=attn_logits_masked.device,
            dtype=omega.dtype,
        )

        use_budget = bool(self.config.ais_use_budget_routing) and float(self.config.ais_budget_total) > 0.0
        op = str(self.config.ais_operator or "soft").strip().lower()
        harmful_selected_dbg = torch.zeros_like(omega)
        harmful_patch_mask_dbg = torch.zeros(
            (bsz, n_heads, kv_seq_len),
            dtype=alpha_last.dtype,
            device=alpha_last.device,
        )

        if use_budget:
            strict_layers = bool(self.config.ais_strict_headset_layers)
            has_harmful_map = len(self._harmful_heads_map) > 0
            has_faithful_map = len(self._faithful_heads_map) > 0
            # Candidate pools: if headset exists, constrain ranking within it.
            # In strict mode, layers missing from the map get zero pool (no intervention).
            # Otherwise, missing layers fall back to all heads.
            if harmful_head_mask is None:
                harmful_pool = (
                    torch.zeros_like(omega)
                    if (strict_layers and has_harmful_map)
                    else torch.ones_like(omega)
                )
            else:
                harmful_pool = harmful_head_mask[None, :].expand_as(omega)
            if faithful_head_mask is None:
                faithful_pool = (
                    torch.zeros_like(omega)
                    if (strict_layers and has_faithful_map)
                    else torch.ones_like(omega)
                )
            else:
                faithful_pool = faithful_head_mask[None, :].expand_as(omega)

            harmful_scores = F.relu(u_centered) * harmful_pool
            faithful_scores = F.relu(v) * faithful_pool

            harmful_selected = self._top_ratio_mask(
                harmful_scores,
                ratio=float(self.config.ais_harmful_top_ratio),
            )  # [B,H]
            faithful_selected = self._top_ratio_mask(
                faithful_scores,
                ratio=float(self.config.ais_faithful_top_ratio),
            )  # [B,H]

            patch_topk = int(self.config.ais_budget_patch_topk)
            harmful_patch_mask = self._topk_patch_binary_mask_3d(
                scores=alpha_img_norm * relu_a[:, None, :],
                image_mask=image_mask,
                topk=patch_topk,
            )  # [B,H,K]
            faithful_patch_mask = self._topk_patch_binary_mask_3d(
                scores=alpha_img_norm * preserve_norm[:, None, :],
                image_mask=image_mask,
                topk=patch_topk,
            )  # [B,H,K]

            n_late = float(self._n_late_layers())
            b_layer = float(max(0.0, self.config.ais_budget_total)) / max(1.0, n_late)
            b_layer = b_layer * float(max(0.0, self.config.ais_gamma))
            # In budget mode, we always apply fixed-budget routing on late layers.
            trigger = torch.ones_like(trigger)
            if arm == "harmful_only":
                b_h = b_layer
                b_f = 0.0
            elif arm == "faithful_only":
                b_h = 0.0
                b_f = b_layer * float(max(0.0, self.config.ais_faithful_boost))
            elif arm == "bipolar":
                rho = float(max(0.0, min(1.0, self.config.ais_bipolar_harmful_ratio)))
                b_h = b_layer * rho
                b_f = b_layer * (1.0 - rho) * float(max(0.0, self.config.ais_faithful_boost))
            else:
                # legacy with budget mode: use harmful suppression only.
                b_h = b_layer
                b_f = 0.0

            if op == "semi_hard":
                dose_h = float(max(0.0, self.config.ais_semihard_penalty)) * float(max(0.0, self.config.ais_gamma))
                dose_f = dose_h * float(max(0.0, self.config.ais_faithful_boost))
                harmful_penalty = (
                    trigger[:, None, None]
                    * float(dose_h)
                    * harmful_selected[:, :, None]
                    * harmful_patch_mask
                )
                faithful_boost = (
                    trigger[:, None, None]
                    * float(dose_f)
                    * faithful_selected[:, :, None]
                    * faithful_patch_mask
                )
            else:
                harmful_penalty = (
                    trigger[:, None, None]
                    * float(b_h)
                    * harmful_selected[:, :, None]
                    * harmful_patch_mask
                )

                faithful_boost = (
                    trigger[:, None, None]
                    * float(b_f)
                    * faithful_selected[:, :, None]
                    * faithful_patch_mask
                )
            harmful_strength = harmful_selected
            harmful_selected_dbg = harmful_selected
            harmful_patch_mask_dbg = harmful_patch_mask
        else:
            if arm == "legacy":
                harmful_strength = omega
            else:
                if harmful_head_mask is None:
                    harmful_strength = torch.zeros_like(omega)
                else:
                    harmful_strength = harmful_head_mask[None, :].expand_as(omega)
                if bool(self.config.ais_use_dynamic_omega):
                    harmful_strength = harmful_strength * omega

            harmful_penalty = (
                float(self.config.ais_gamma)
                * trigger[:, None, None]
                * harmful_strength[:, :, None]
                * relu_a[:, None, :]
            )

            faithful_boost = torch.zeros_like(harmful_penalty)
            if arm in {"faithful_only", "bipolar"}:
                if faithful_head_mask is not None:
                    faithful_strength = faithful_head_mask[None, :].expand_as(omega)
                    faithful_boost = (
                        float(self.config.ais_gamma)
                        * float(max(0.0, self.config.ais_faithful_boost))
                        * trigger[:, None, None]
                        * faithful_strength[:, :, None]
                        * preserve_norm[:, None, :]
                    )
            harmful_selected_dbg = (harmful_strength > 0).to(dtype=omega.dtype)

        if arm == "faithful_only":
            bias_last = -faithful_boost
        elif arm == "bipolar":
            bias_last = harmful_penalty - faithful_boost
        else:
            # legacy + harmful_only
            bias_last = harmful_penalty

        bias_last = bias_last * img_mask_f

        # Optional path-probe diagnostic bias (late layers only).
        # This is intentionally simple and training-free: apply fixed penalty on selected key columns.
        path_probe_mode = str(self.config.path_probe_mode or "none").strip().lower()
        path_probe_penalty = float(max(0.0, self.config.path_probe_penalty))
        path_probe_applied = False
        probe_img_bias = torch.zeros_like(bias_last)
        probe_text_bias = torch.zeros_like(bias_last)
        if (
            path_probe_mode in {"drop_img", "drop_text", "drop_both"}
            and path_probe_penalty > 0.0
            and self._in_late(li)
        ):
            if (not bool(self.config.path_probe_first_step_only)) or int(self._step_index) == 0:
                path_probe_applied = True
                if attention_mask is not None:
                    # attention_mask here is combined additive mask (causal + padding).
                    valid_cols = torch.isfinite(attention_mask[:, 0, -1, :]).to(torch.bool)
                else:
                    valid_cols = torch.ones((bsz, kv_seq_len), dtype=torch.bool, device=bias_last.device)
                text_mask = valid_cols & (~image_mask)

                if path_probe_mode in {"drop_img", "drop_both"}:
                    probe_img_bias = float(path_probe_penalty) * img_mask_f
                if path_probe_mode in {"drop_text", "drop_both"}:
                    probe_text_bias = float(path_probe_penalty) * text_mask[:, None, :].to(dtype=bias_last.dtype)

                bias_last = bias_last + probe_img_bias + probe_text_bias

        bias = torch.zeros_like(attn_logits_masked)
        bias[:, :, -1, :] = bias_last.to(dtype=attn_logits_masked.dtype)

        self._late_seen_in_step += 1
        self._late_triggered_in_step += int((trigger > 0).any().item())

        if bool(self.config.ais_debug_log):
            img_cnt = torch.clamp(image_mask.sum(dim=-1).to(dtype=bias_last.dtype), min=1.0)
            if attention_mask is not None:
                valid_cols_dbg = torch.isfinite(attention_mask[:, 0, -1, :]).to(torch.bool)
            else:
                valid_cols_dbg = torch.ones((bsz, kv_seq_len), dtype=torch.bool, device=bias_last.device)
            text_mask_dbg = valid_cols_dbg & (~image_mask)
            text_cnt = torch.clamp(text_mask_dbg.sum(dim=-1).to(dtype=bias_last.dtype), min=1.0)
            per_b_head = bias_last.sum(dim=-1) / img_cnt[:, None]
            per_b_head_abs = bias_last.abs().sum(dim=-1) / img_cnt[:, None]
            penalty_img_mean = float(per_b_head.mean().item())
            penalty_img_abs_mean = float(per_b_head_abs.mean().item())
            harmful_mean = float((harmful_penalty.sum(dim=-1) / img_cnt[:, None]).mean().item())
            faithful_mean = float((faithful_boost.sum(dim=-1) / img_cnt[:, None]).mean().item())
            probe_img_mean = float((probe_img_bias.sum(dim=-1) / img_cnt[:, None]).mean().item())
            probe_text_mean = float(
                (
                    (
                        probe_text_bias.sum(dim=-1)
                        / text_cnt[:, None]
                    ).mean()
                ).item()
            )
            selected_cells = (harmful_selected_dbg[:, :, None] > 0).to(dtype=harmful_patch_mask_dbg.dtype) * (
                harmful_patch_mask_dbg > 0
            ).to(dtype=harmful_patch_mask_dbg.dtype)
            selected_heads_cnt = torch.clamp(
                (harmful_selected_dbg > 0).to(torch.float32).sum(dim=-1),
                min=1.0,
            )  # [B]
            selected_patch_per_head = (
                selected_cells.to(torch.float32).sum(dim=-1).sum(dim=-1) / selected_heads_cnt
            )  # [B]
            selected_patch_any = selected_cells.to(torch.float32).amax(dim=1)  # [B,K]
            selected_patch_cov = selected_patch_any.sum(dim=-1) / img_cnt  # [B]
            sel_pos = harmful_penalty[harmful_penalty > 0]
            harmful_per_cell_dose = float(sel_pos.mean().item()) if int(sel_pos.numel()) > 0 else 0.0
            head_idx_b0 = torch.nonzero((harmful_strength[0] > 0), as_tuple=False).flatten().tolist() if bsz > 0 else []
            patch_idx_b0 = (
                torch.nonzero((selected_patch_any[0] > 0), as_tuple=False).flatten().tolist() if bsz > 0 else []
            )
            row = {
                "generation_idx": int(self._generation_index),
                "step_idx": int(self._step_index),
                "layer_idx": int(li),
                "ais_arm": arm,
                "ais_mean": float(ais.mean().item()),
                "ais_max": float(ais.max().item()),
                "trigger_frac_batch": float((trigger > 0).to(torch.float32).mean().item()),
                "late_layers_seen_in_step": int(self._late_seen_in_step),
                "late_layers_triggered_in_step": int(self._late_triggered_in_step),
                "late_trigger_fraction_step": float(
                    float(self._late_triggered_in_step) / float(max(1, self._late_seen_in_step))
                ),
                "omega_mean": float(omega.mean().item()),
                "penalty_img_mean": penalty_img_mean,
                "penalty_img_abs_mean": penalty_img_abs_mean,
                "harmful_penalty_img_mean": harmful_mean,
                "faithful_boost_img_mean": faithful_mean,
                "image_cols_per_batch_mean": float(image_mask.to(torch.float32).sum(dim=-1).mean().item()),
                "early_count": int(self._early_count),
                "n_harmful_heads_layer": int(len(self._harmful_heads_map.get(li, set()))),
                "n_faithful_heads_layer": int(len(self._faithful_heads_map.get(li, set()))),
                "strict_headset_layers": bool(self.config.ais_strict_headset_layers),
                "budget_mode": bool(use_budget),
                "ais_operator": op,
                "ais_semihard_penalty": float(self.config.ais_semihard_penalty),
                "budget_layer": (
                    float(max(0.0, self.config.ais_budget_total) / max(1, self._n_late_layers()) * max(0.0, self.config.ais_gamma))
                    if use_budget
                    else 0.0
                ),
                "harmful_selected_heads_mean": float((harmful_strength > 0).to(torch.float32).sum(dim=-1).mean().item()),
                "harmful_selected_patch_per_head_mean": float(selected_patch_per_head.mean().item()),
                "harmful_selected_patch_coverage_mean": float(selected_patch_cov.mean().item()),
                "harmful_per_cell_dose_mean": harmful_per_cell_dose,
                "applied_harmful_head_idx_b0": "|".join(str(int(x)) for x in head_idx_b0),
                "applied_harmful_patch_idx_b0": "|".join(str(int(x)) for x in patch_idx_b0),
                "faithful_selected_heads_mean": float(
                    (
                        (
                            (faithful_boost.abs().sum(dim=-1) > 0).to(torch.float32)
                        ).sum(dim=-1)
                    ).mean().item()
                ),
                "budget_patch_topk": int(self.config.ais_budget_patch_topk),
                "path_probe_mode": path_probe_mode,
                "path_probe_penalty": float(path_probe_penalty),
                "path_probe_first_step_only": bool(self.config.path_probe_first_step_only),
                "path_probe_applied": bool(path_probe_applied),
                "path_probe_img_penalty_mean": probe_img_mean,
                "path_probe_text_penalty_mean": probe_text_mean,
            }
            self._debug_rows.append(row)

        return bias



    def apply_frrs_output_steering(
        self,
        layer_idx: int,
        attn_output: torch.Tensor,  # [B,H,Q,D]
        value_states: torch.Tensor,  # [B,H,K,D]
        attention_mask: Optional[torch.Tensor],
        attn_weights_last: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        li = int(layer_idx)
        if not bool(self._frrs_enabled() and self._in_frrs_late(li)):
            return attn_output
        if attn_output.dim() != 4 or value_states.dim() != 4:
            return attn_output

        bsz, n_heads, q_len, _hd = (
            int(attn_output.size(0)),
            int(attn_output.size(1)),
            int(attn_output.size(2)),
            int(attn_output.size(3)),
        )
        kv_seq_len = int(value_states.size(2))
        self._begin_step_if_needed(kv_seq_len=kv_seq_len, bsz=bsz)

        image_mask = self._resolve_image_mask_for_kv(
            bsz=bsz,
            kv_seq_len=kv_seq_len,
            attention_mask=attention_mask,
            device=attn_output.device,
        )
        if image_mask is None or int(image_mask.sum().item()) == 0:
            return attn_output

        feats = self._resolve_frrs_feats_for_batch(
            bsz=bsz,
            image_mask=image_mask,
            device=attn_output.device,
            dtype=attn_output.dtype,
        )
        if feats is None:
            return attn_output

        frrs_online_dbg: Dict[str, float] = {"frrs_online_feat_used": 0.0, "frrs_online_blend": float(self.config.frrs_online_blend)}

        faithful_head_mask = self._head_mask_for_layer(
            layer_idx=li,
            n_heads=n_heads,
            kind="faithful",
            device=attn_output.device,
            dtype=attn_output.dtype,
        )
        harmful_head_mask = self._head_mask_for_layer(
            layer_idx=li,
            n_heads=n_heads,
            kind="harmful",
            device=attn_output.device,
            dtype=attn_output.dtype,
        )

        if bool(self.config.frrs_online_recompute_feats) and attn_weights_last is not None:
            feats, frrs_online_dbg = self._recompute_frrs_feats_online(
                feats=feats,
                attn_weights_last=attn_weights_last,
                image_mask=image_mask,
                faithful_head_mask=faithful_head_mask,
                harmful_head_mask=harmful_head_mask,
                device=attn_output.device,
                dtype=attn_output.dtype,
            )

        arm = str(self.config.frrs_arm or "supportive").strip().lower()
        head_mode = str(self.config.frrs_head_mode or "dynamic").strip().lower()
        if head_mode not in {"static", "dynamic", "hybrid"}:
            head_mode = "dynamic"
        use_static = head_mode in {"static", "hybrid"}
        use_dynamic = head_mode in {"dynamic", "hybrid"}
        dynamic_available = bool(use_dynamic and attn_weights_last is not None)

        if arm == "bipolar":
            static_available = bool(use_static and (faithful_head_mask is not None or harmful_head_mask is not None))
            if not (dynamic_available or static_available):
                return attn_output
        else:
            static_available = bool(use_static and faithful_head_mask is not None)
            if not (dynamic_available or static_available):
                return attn_output

        frrs = self._get_frrs_module()
        out, dbg = frrs(
            attn_output=attn_output,
            value_states=value_states,
            image_mask=image_mask,
            feats=feats,
            faithful_head_mask=faithful_head_mask,
            harmful_head_mask=harmful_head_mask,
            attn_weights_last=attn_weights_last,
        )

        delta_last = (out[:, :, -1, :] - attn_output[:, :, -1, :]).to(dtype=attn_output.dtype)
        if float(delta_last.abs().sum().item()) == 0.0:
            return attn_output

        self._late_seen_in_step += 1
        self._late_triggered_in_step += int((delta_last.abs().sum(dim=(-1, -2)) > 0).any().item())

        if bool(self.config.frrs_debug_log or self.config.ais_debug_log):
            row = {
                "generation_idx": int(self._generation_index),
                "step_idx": int(self._step_index),
                "layer_idx": int(li),
                "frrs_mode": True,
                "frrs_arm": str(self.config.frrs_arm),
                "frrs_head_mode": str(self.config.frrs_head_mode),
                "frrs_r_percent": float(self.config.frrs_r_percent),
                "frrs_online_recompute_feats": bool(self.config.frrs_online_recompute_feats),
                "frrs_online_blend": float(self.config.frrs_online_blend),
                "frrs_alpha": float(self.config.frrs_alpha),
                "frrs_beta": float(self.config.frrs_beta),
                "frrs_tau_c": float(self.config.frrs_tau_c),
                "frrs_tau_e": float(self.config.frrs_tau_e),
                "frrs_topk_ratio": float(self.config.frrs_topk_ratio),
                "frrs_gate_mean": float(dbg.get("frrs_gate_mean", 0.0)),
                "frrs_gate_min": float(dbg.get("frrs_gate_min", 0.0)),
                "frrs_gate_max": float(dbg.get("frrs_gate_max", 0.0)),
                "frrs_cbar_mean": float(dbg.get("frrs_cbar_mean", 0.0)),
                "frrs_ebar_mean": float(dbg.get("frrs_ebar_mean", 0.0)),
                "frrs_delta_abs_mean": float(dbg.get("frrs_delta_abs_mean", 0.0)),
                "frrs_active_frac": float(dbg.get("frrs_active_frac", 0.0)),
                "frrs_supportive_head_cov": float(dbg.get("frrs_supportive_head_cov", 0.0)),
                "frrs_harmful_head_cov": float(dbg.get("frrs_harmful_head_cov", 0.0)),
                "frrs_dynamic_used": float(dbg.get("frrs_dynamic_used", 0.0)),
                "frrs_dyn_k_heads": float(dbg.get("frrs_dyn_k_heads", 0.0)),
                "frrs_dyn_overlap_frac": float(dbg.get("frrs_dyn_overlap_frac", 0.0)),
                "frrs_online_feat_used": float(frrs_online_dbg.get("frrs_online_feat_used", 0.0)),
                "frrs_online_a_mean": float(frrs_online_dbg.get("frrs_online_a_mean", 0.0)),
                "frrs_online_c_mean": float(frrs_online_dbg.get("frrs_online_c_mean", 0.0)),
                "frrs_online_d_mean": float(frrs_online_dbg.get("frrs_online_d_mean", 0.0)),
                "frrs_online_e_mean": float(frrs_online_dbg.get("frrs_online_e_mean", 0.0)),
                "trigger_frac_batch": float((delta_last.abs().sum(dim=(-1, -2)) > 0).to(torch.float32).mean().item()),
                "late_layers_seen_in_step": int(self._late_seen_in_step),
                "late_layers_triggered_in_step": int(self._late_triggered_in_step),
                "late_trigger_fraction_step": float(
                    float(self._late_triggered_in_step) / float(max(1, self._late_seen_in_step))
                ),
            }
            self._debug_rows.append(row)

        return out


def _manual_llama_attention_forward_with_ais(
    self_attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    ais_runtime: Optional[AISGatingRuntime] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in transformers v4.37+. "
            "Use `attention_mask` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    if self_attn.config.pretraining_tp > 1:
        key_value_slicing = (self_attn.num_key_value_heads * self_attn.head_dim) // self_attn.config.pretraining_tp
        query_slices = self_attn.q_proj.weight.split(
            (self_attn.num_heads * self_attn.head_dim) // self_attn.config.pretraining_tp, dim=0
        )
        key_slices = self_attn.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self_attn.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self_attn.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self_attn.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self_attn.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self_attn.layer_idx is None:
            raise ValueError(
                "layer_idx is required for cache-aware autoregressive decoding in attention module."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self_attn.layer_idx)
    cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self_attn.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
    value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

    attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)

    # Build a 4D additive mask that always includes causal masking.
    attn_mask_4d: Optional[torch.Tensor] = None
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_mask_4d = attention_mask
        elif attention_mask.dim() == 2:
            # 2D key padding mask: [B, K], where True/1 means valid token.
            if attention_mask.size() != (bsz, kv_seq_len):
                raise ValueError(
                    f"2D attention mask should be of size {(bsz, kv_seq_len)}, but is {attention_mask.size()}"
                )
            valid = attention_mask.to(torch.bool)
            attn_mask_4d = torch.zeros(
                (bsz, 1, q_len, kv_seq_len),
                dtype=attn_logits.dtype,
                device=attn_logits.device,
            )
            attn_mask_4d = attn_mask_4d.masked_fill(
                ~valid[:, None, None, :],
                torch.finfo(attn_logits.dtype).min,
            )
        else:
            raise ValueError(
                f"Unsupported attention_mask rank {attention_mask.dim()}; expected 2D or 4D."
            )

    # Always enforce autoregressive causality in this manual path.
    # Query positions correspond to the last q_len positions in key sequence when cache is used.
    q_pos = torch.arange(kv_seq_len - q_len, kv_seq_len, device=attn_logits.device)
    k_pos = torch.arange(kv_seq_len, device=attn_logits.device)
    disallowed = k_pos[None, :] > q_pos[:, None]  # [Q, K]
    causal_mask = torch.zeros(
        (1, 1, q_len, kv_seq_len),
        dtype=attn_logits.dtype,
        device=attn_logits.device,
    ).masked_fill(disallowed[None, None, :, :], torch.finfo(attn_logits.dtype).min)

    if attn_mask_4d is None:
        combined_mask = causal_mask
    else:
        combined_mask = attn_mask_4d + causal_mask

    attn_logits = attn_logits + combined_mask

    if ais_runtime is not None and ais_runtime.should_intercept_layer(getattr(self_attn, "layer_idx", None)):
        bias = ais_runtime.compute_bias(
            layer_idx=int(self_attn.layer_idx),
            attn_logits_masked=attn_logits,
            attention_mask=combined_mask,
        )
        if bias is not None:
            attn_logits = attn_logits - bias.to(dtype=attn_logits.dtype)

    # Upcast to fp32 for stable softmax, then cast back.
    attn_weights = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self_attn.attention_dropout, training=self_attn.training)
    if ais_runtime is not None and ais_runtime.should_intercept_layer(getattr(self_attn, "layer_idx", None)):
        ais_runtime.record_proxy_attn(
            layer_idx=int(self_attn.layer_idx),
            attn_weights_last=attn_weights[:, :, -1, :],
            attention_mask=combined_mask,
        )
    attn_output = torch.matmul(attn_weights, value_states)

    if ais_runtime is not None and ais_runtime.should_intercept_layer(getattr(self_attn, "layer_idx", None)):
        attn_output = ais_runtime.apply_frrs_output_steering(
            layer_idx=int(self_attn.layer_idx),
            attn_output=attn_output,
            value_states=value_states,
            attention_mask=combined_mask,
            attn_weights_last=attn_weights[:, :, -1, :],
        )

    if attn_output.size() != (bsz, self_attn.num_heads, q_len, self_attn.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self_attn.num_heads, q_len, self_attn.head_dim)}, "
            f"but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self_attn.hidden_size)

    if self_attn.config.pretraining_tp > 1:
        attn_output_split = attn_output.split(self_attn.hidden_size // self_attn.config.pretraining_tp, dim=2)
        o_proj_slices = self_attn.o_proj.weight.split(self_attn.hidden_size // self_attn.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output_split[i], o_proj_slices[i]) for i in range(self_attn.config.pretraining_tp)])
    else:
        attn_output = self_attn.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _make_wrapped_forward():
    def _wrapped_forward(
        self_attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        runtime: Optional[AISGatingRuntime] = getattr(self_attn, "_ais_runtime", None)
        if runtime is None or not runtime.should_intercept_layer(getattr(self_attn, "layer_idx", None)):
            return _call_orig_forward(
                self_attn=self_attn,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                kwargs=kwargs,
            )
        return _manual_llama_attention_forward_with_ais(
            self_attn=self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            ais_runtime=runtime,
            **kwargs,
        )

    return _wrapped_forward


def _get_decoder_layers(model: torch.nn.Module) -> Optional[Sequence[torch.nn.Module]]:
    # LlavaLlamaForCausalLM / LlamaForCausalLM: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Raw decoder model
    if hasattr(model, "layers"):
        return model.layers
    return None


def install_ais_gating_hooks(
    model: torch.nn.Module,
    runtime: Optional[AISGatingRuntime] = None,
) -> AISGatingRuntime:
    """
    Patch decoder self-attention modules with a wrapper.
    Wrapper falls back to original forward unless runtime says interception is needed.
    """
    rt = runtime or AISGatingRuntime()
    layers = _get_decoder_layers(model)
    if layers is None:
        raise RuntimeError("Could not find decoder layers for AIS gating hook installation.")

    wrapped_fn = _make_wrapped_forward()
    n_patched = 0
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        if bool(getattr(self_attn, "_ais_patched", False)):
            self_attn._ais_runtime = rt
            continue
        self_attn._ais_orig_forward = self_attn.forward
        self_attn._ais_orig_accepts_kwargs = bool(
            any(p.kind.name == "VAR_KEYWORD" for p in inspect.signature(self_attn.forward).parameters.values())
        )
        self_attn._ais_runtime = rt
        self_attn.forward = types.MethodType(wrapped_fn, self_attn)
        self_attn._ais_patched = True
        n_patched += 1

    setattr(model, "_ais_gating_runtime", rt)
    setattr(model, "_ais_gating_hooks_installed", True)
    setattr(model, "_ais_gating_num_patched", int(n_patched))
    return rt


def apply_ais_column_penalty(
    attn_logits: torch.Tensor,
    image_mask: torch.Tensor,
    penalty_last_query: torch.Tensor,
) -> torch.Tensor:
    """
    Pure helper used by tests:
    - attn_logits: [B,H,Q,K]
    - image_mask: [B,K] bool
    - penalty_last_query: [B,H,K] (non-zero only where image_mask is true)
    """
    out = attn_logits.clone()
    mask_f = image_mask[:, None, :].to(dtype=out.dtype)
    out[:, :, -1, :] = out[:, :, -1, :] - penalty_last_query.to(dtype=out.dtype) * mask_f
    return out
