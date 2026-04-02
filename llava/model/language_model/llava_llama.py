#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..ais_gating import AISGatingConfig, AISGatingRuntime, install_ais_gating_hooks


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self._ais_gating_runtime: Optional[AISGatingRuntime] = None
        self._ais_gating_hooks_installed: bool = False
        self._ais_capture_image_mask: bool = False
        self._ais_image_token_mask: Optional[torch.Tensor] = None

    def get_model(self):
        return self.model

    def _ensure_ais_runtime(self) -> AISGatingRuntime:
        if self._ais_gating_runtime is None:
            self._ais_gating_runtime = AISGatingRuntime(AISGatingConfig())
        return self._ais_gating_runtime

    def _ensure_ais_hooks(self) -> AISGatingRuntime:
        rt = self._ensure_ais_runtime()
        if not bool(self._ais_gating_hooks_installed):
            install_ais_gating_hooks(self.get_model(), runtime=rt)
            self._ais_gating_hooks_installed = True
        return rt

    def configure_ais_gating(
        self,
        enable_ais_gating: Optional[bool] = None,
        ais_early_start: Optional[int] = None,
        ais_early_end: Optional[int] = None,
        ais_late_start: Optional[int] = None,
        ais_late_end: Optional[int] = None,
        ais_topk: Optional[int] = None,
        ais_tau: Optional[float] = None,
        ais_gamma: Optional[float] = None,
        ais_eps: Optional[float] = None,
        ais_debug_log: Optional[bool] = None,
        ais_arm: Optional[str] = None,
        ais_harmful_heads: Optional[str] = None,
        ais_faithful_heads: Optional[str] = None,
        ais_headset_json: Optional[str] = None,
        ais_faithful_boost: Optional[float] = None,
        ais_use_dynamic_omega: Optional[bool] = None,
        ais_use_budget_routing: Optional[bool] = None,
        ais_budget_total: Optional[float] = None,
        ais_harmful_top_ratio: Optional[float] = None,
        ais_faithful_top_ratio: Optional[float] = None,
        ais_bipolar_harmful_ratio: Optional[float] = None,
        ais_budget_patch_topk: Optional[int] = None,
        ais_strict_headset_layers: Optional[bool] = None,
        ais_operator: Optional[str] = None,
        ais_semihard_penalty: Optional[float] = None,
        ais_use_oracle_roles: Optional[bool] = None,
        ais_oracle_role_csv: Optional[str] = None,
        ais_oracle_supportive_topk: Optional[int] = None,
        ais_oracle_assertive_topk: Optional[int] = None,
        ais_oracle_lambda_pos: Optional[float] = None,
        ais_oracle_lambda_neg: Optional[float] = None,
        ais_oracle_bias_clip: Optional[float] = None,
        path_probe_mode: Optional[str] = None,
        path_probe_penalty: Optional[float] = None,
        path_probe_first_step_only: Optional[bool] = None,
        enable_rfhar: Optional[bool] = None,
        rfhar_early_start: Optional[int] = None,
        rfhar_early_end: Optional[int] = None,
        rfhar_late_start: Optional[int] = None,
        rfhar_late_end: Optional[int] = None,
        rfhar_r_percent: Optional[float] = None,
        rfhar_gamma: Optional[float] = None,
        rfhar_lambda_penalty: Optional[float] = None,
        rfhar_eps: Optional[float] = None,
        rfhar_debug_log: Optional[bool] = None,
        enable_frgg: Optional[bool] = None,
        frgg_late_start: Optional[int] = None,
        frgg_late_end: Optional[int] = None,
        frgg_gamma: Optional[float] = None,
        frgg_tau_c: Optional[float] = None,
        frgg_tau_e: Optional[float] = None,
        frgg_k_c: Optional[float] = None,
        frgg_k_e: Optional[float] = None,
        frgg_topk_ratio: Optional[float] = None,
        frgg_eps: Optional[float] = None,
        frgg_debug_log: Optional[bool] = None,
        enable_frrs: Optional[bool] = None,
        frrs_late_start: Optional[int] = None,
        frrs_late_end: Optional[int] = None,
        frrs_alpha: Optional[float] = None,
        frrs_beta: Optional[float] = None,
        frrs_tau_c: Optional[float] = None,
        frrs_tau_e: Optional[float] = None,
        frrs_k_c: Optional[float] = None,
        frrs_k_e: Optional[float] = None,
        frrs_topk_ratio: Optional[float] = None,
        frrs_eps: Optional[float] = None,
        frrs_arm: Optional[str] = None,
        frrs_head_mode: Optional[str] = None,
        frrs_r_percent: Optional[float] = None,
        frrs_online_recompute_feats: Optional[bool] = None,
        frrs_online_blend: Optional[float] = None,
        frrs_debug_log: Optional[bool] = None,
        enable_proxy_trace: Optional[bool] = None,
        proxy_late_start: Optional[int] = None,
        proxy_late_end: Optional[int] = None,
        proxy_eps: Optional[float] = None,
    ) -> Dict[str, Any]:
        rt = self._ensure_ais_runtime()
        rt.configure(
            enable_ais_gating=enable_ais_gating,
            ais_early_start=ais_early_start,
            ais_early_end=ais_early_end,
            ais_late_start=ais_late_start,
            ais_late_end=ais_late_end,
            ais_topk=ais_topk,
            ais_tau=ais_tau,
            ais_gamma=ais_gamma,
            ais_eps=ais_eps,
            ais_debug_log=ais_debug_log,
            ais_arm=ais_arm,
            ais_harmful_heads=ais_harmful_heads,
            ais_faithful_heads=ais_faithful_heads,
            ais_headset_json=ais_headset_json,
            ais_faithful_boost=ais_faithful_boost,
            ais_use_dynamic_omega=ais_use_dynamic_omega,
            ais_use_budget_routing=ais_use_budget_routing,
            ais_budget_total=ais_budget_total,
            ais_harmful_top_ratio=ais_harmful_top_ratio,
            ais_faithful_top_ratio=ais_faithful_top_ratio,
            ais_bipolar_harmful_ratio=ais_bipolar_harmful_ratio,
            ais_budget_patch_topk=ais_budget_patch_topk,
            ais_strict_headset_layers=ais_strict_headset_layers,
            ais_operator=ais_operator,
            ais_semihard_penalty=ais_semihard_penalty,
            ais_use_oracle_roles=ais_use_oracle_roles,
            ais_oracle_role_csv=ais_oracle_role_csv,
            ais_oracle_supportive_topk=ais_oracle_supportive_topk,
            ais_oracle_assertive_topk=ais_oracle_assertive_topk,
            ais_oracle_lambda_pos=ais_oracle_lambda_pos,
            ais_oracle_lambda_neg=ais_oracle_lambda_neg,
            ais_oracle_bias_clip=ais_oracle_bias_clip,
            path_probe_mode=path_probe_mode,
            path_probe_penalty=path_probe_penalty,
            path_probe_first_step_only=path_probe_first_step_only,
            enable_rfhar=enable_rfhar,
            rfhar_early_start=rfhar_early_start,
            rfhar_early_end=rfhar_early_end,
            rfhar_late_start=rfhar_late_start,
            rfhar_late_end=rfhar_late_end,
            rfhar_r_percent=rfhar_r_percent,
            rfhar_gamma=rfhar_gamma,
            rfhar_lambda_penalty=rfhar_lambda_penalty,
            rfhar_eps=rfhar_eps,
            rfhar_debug_log=rfhar_debug_log,
            enable_frgg=enable_frgg,
            frgg_late_start=frgg_late_start,
            frgg_late_end=frgg_late_end,
            frgg_gamma=frgg_gamma,
            frgg_tau_c=frgg_tau_c,
            frgg_tau_e=frgg_tau_e,
            frgg_k_c=frgg_k_c,
            frgg_k_e=frgg_k_e,
            frgg_topk_ratio=frgg_topk_ratio,
            frgg_eps=frgg_eps,
            frgg_debug_log=frgg_debug_log,
            enable_frrs=enable_frrs,
            frrs_late_start=frrs_late_start,
            frrs_late_end=frrs_late_end,
            frrs_alpha=frrs_alpha,
            frrs_beta=frrs_beta,
            frrs_tau_c=frrs_tau_c,
            frrs_tau_e=frrs_tau_e,
            frrs_k_c=frrs_k_c,
            frrs_k_e=frrs_k_e,
            frrs_topk_ratio=frrs_topk_ratio,
            frrs_eps=frrs_eps,
            frrs_arm=frrs_arm,
            frrs_head_mode=frrs_head_mode,
            frrs_r_percent=frrs_r_percent,
            frrs_online_recompute_feats=frrs_online_recompute_feats,
            frrs_online_blend=frrs_online_blend,
            frrs_debug_log=frrs_debug_log,
            enable_proxy_trace=enable_proxy_trace,
            proxy_late_start=proxy_late_start,
            proxy_late_end=proxy_late_end,
            proxy_eps=proxy_eps,
        )
        if rt.is_effective_enabled():
            self._ensure_ais_hooks()
        return {
            "enable_ais_gating": bool(rt.config.enable_ais_gating),
            "ais_early_start": int(rt.config.ais_early_start),
            "ais_early_end": int(rt.config.ais_early_end),
            "ais_late_start": int(rt.config.ais_late_start),
            "ais_late_end": int(rt.config.ais_late_end),
            "ais_topk": int(rt.config.ais_topk),
            "ais_tau": float(rt.config.ais_tau),
            "ais_gamma": float(rt.config.ais_gamma),
            "ais_eps": float(rt.config.ais_eps),
            "ais_debug_log": bool(rt.config.ais_debug_log),
            "ais_arm": str(rt.config.ais_arm),
            "ais_harmful_heads": str(rt.config.ais_harmful_heads),
            "ais_faithful_heads": str(rt.config.ais_faithful_heads),
            "ais_headset_json": str(rt.config.ais_headset_json),
            "ais_faithful_boost": float(rt.config.ais_faithful_boost),
            "ais_use_dynamic_omega": bool(rt.config.ais_use_dynamic_omega),
            "ais_use_budget_routing": bool(rt.config.ais_use_budget_routing),
            "ais_budget_total": float(rt.config.ais_budget_total),
            "ais_harmful_top_ratio": float(rt.config.ais_harmful_top_ratio),
            "ais_faithful_top_ratio": float(rt.config.ais_faithful_top_ratio),
            "ais_bipolar_harmful_ratio": float(rt.config.ais_bipolar_harmful_ratio),
            "ais_budget_patch_topk": int(rt.config.ais_budget_patch_topk),
            "ais_strict_headset_layers": bool(rt.config.ais_strict_headset_layers),
            "ais_operator": str(rt.config.ais_operator),
            "ais_semihard_penalty": float(rt.config.ais_semihard_penalty),
            "ais_use_oracle_roles": bool(rt.config.ais_use_oracle_roles),
            "ais_oracle_role_csv": str(rt.config.ais_oracle_role_csv),
            "ais_oracle_supportive_topk": int(rt.config.ais_oracle_supportive_topk),
            "ais_oracle_assertive_topk": int(rt.config.ais_oracle_assertive_topk),
            "ais_oracle_lambda_pos": float(rt.config.ais_oracle_lambda_pos),
            "ais_oracle_lambda_neg": float(rt.config.ais_oracle_lambda_neg),
            "ais_oracle_bias_clip": float(rt.config.ais_oracle_bias_clip),
            "path_probe_mode": str(rt.config.path_probe_mode),
            "path_probe_penalty": float(rt.config.path_probe_penalty),
            "path_probe_first_step_only": bool(rt.config.path_probe_first_step_only),
            "enable_rfhar": bool(rt.config.enable_rfhar),
            "rfhar_early_start": int(rt.config.rfhar_early_start),
            "rfhar_early_end": int(rt.config.rfhar_early_end),
            "rfhar_late_start": int(rt.config.rfhar_late_start),
            "rfhar_late_end": int(rt.config.rfhar_late_end),
            "rfhar_r_percent": float(rt.config.rfhar_r_percent),
            "rfhar_gamma": float(rt.config.rfhar_gamma),
            "rfhar_lambda_penalty": float(rt.config.rfhar_lambda_penalty),
            "rfhar_eps": float(rt.config.rfhar_eps),
            "rfhar_debug_log": bool(rt.config.rfhar_debug_log),
            "enable_frgg": bool(rt.config.enable_frgg),
            "frgg_late_start": int(rt.config.frgg_late_start),
            "frgg_late_end": int(rt.config.frgg_late_end),
            "frgg_gamma": float(rt.config.frgg_gamma),
            "frgg_tau_c": float(rt.config.frgg_tau_c),
            "frgg_tau_e": float(rt.config.frgg_tau_e),
            "frgg_k_c": float(rt.config.frgg_k_c),
            "frgg_k_e": float(rt.config.frgg_k_e),
            "frgg_topk_ratio": float(rt.config.frgg_topk_ratio),
            "frgg_eps": float(rt.config.frgg_eps),
            "frgg_debug_log": bool(rt.config.frgg_debug_log),
            "enable_frrs": bool(rt.config.enable_frrs),
            "frrs_late_start": int(rt.config.frrs_late_start),
            "frrs_late_end": int(rt.config.frrs_late_end),
            "frrs_alpha": float(rt.config.frrs_alpha),
            "frrs_beta": float(rt.config.frrs_beta),
            "frrs_tau_c": float(rt.config.frrs_tau_c),
            "frrs_tau_e": float(rt.config.frrs_tau_e),
            "frrs_k_c": float(rt.config.frrs_k_c),
            "frrs_k_e": float(rt.config.frrs_k_e),
            "frrs_topk_ratio": float(rt.config.frrs_topk_ratio),
            "frrs_eps": float(rt.config.frrs_eps),
            "frrs_arm": str(rt.config.frrs_arm),
            "frrs_head_mode": str(rt.config.frrs_head_mode),
            "frrs_r_percent": float(rt.config.frrs_r_percent),
            "frrs_online_recompute_feats": bool(rt.config.frrs_online_recompute_feats),
            "frrs_online_blend": float(rt.config.frrs_online_blend),
            "frrs_debug_log": bool(rt.config.frrs_debug_log),
            "enable_proxy_trace": bool(rt.config.enable_proxy_trace),
            "proxy_late_start": int(rt.config.proxy_late_start),
            "proxy_late_end": int(rt.config.proxy_late_end),
            "proxy_eps": float(rt.config.proxy_eps),
        }

    def get_ais_debug_rows(self, reset: bool = False) -> List[Dict[str, Any]]:
        if self._ais_gating_runtime is None:
            return []
        return self._ais_gating_runtime.get_debug_rows(reset=reset)

    def get_proxy_trace_rows(self, reset: bool = False) -> List[Dict[str, Any]]:
        if self._ais_gating_runtime is None:
            return []
        return self._ais_gating_runtime.get_proxy_trace_rows(reset=reset)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        enable_ais_gating: Optional[bool] = None,
        ais_early_start: Optional[int] = None,
        ais_early_end: Optional[int] = None,
        ais_late_start: Optional[int] = None,
        ais_late_end: Optional[int] = None,
        ais_topk: Optional[int] = None,
        ais_tau: Optional[float] = None,
        ais_gamma: Optional[float] = None,
        ais_eps: Optional[float] = None,
        ais_debug_log: Optional[bool] = None,
        ais_arm: Optional[str] = None,
        ais_harmful_heads: Optional[str] = None,
        ais_faithful_heads: Optional[str] = None,
        ais_headset_json: Optional[str] = None,
        ais_faithful_boost: Optional[float] = None,
        ais_use_dynamic_omega: Optional[bool] = None,
        ais_use_budget_routing: Optional[bool] = None,
        ais_budget_total: Optional[float] = None,
        ais_harmful_top_ratio: Optional[float] = None,
        ais_faithful_top_ratio: Optional[float] = None,
        ais_bipolar_harmful_ratio: Optional[float] = None,
        ais_budget_patch_topk: Optional[int] = None,
        ais_strict_headset_layers: Optional[bool] = None,
        ais_operator: Optional[str] = None,
        ais_semihard_penalty: Optional[float] = None,
        ais_use_oracle_roles: Optional[bool] = None,
        ais_oracle_role_csv: Optional[str] = None,
        ais_oracle_supportive_topk: Optional[int] = None,
        ais_oracle_assertive_topk: Optional[int] = None,
        ais_oracle_lambda_pos: Optional[float] = None,
        ais_oracle_lambda_neg: Optional[float] = None,
        ais_oracle_bias_clip: Optional[float] = None,
        path_probe_mode: Optional[str] = None,
        path_probe_penalty: Optional[float] = None,
        path_probe_first_step_only: Optional[bool] = None,
        enable_rfhar: Optional[bool] = None,
        rfhar_early_start: Optional[int] = None,
        rfhar_early_end: Optional[int] = None,
        rfhar_late_start: Optional[int] = None,
        rfhar_late_end: Optional[int] = None,
        rfhar_r_percent: Optional[float] = None,
        rfhar_gamma: Optional[float] = None,
        rfhar_lambda_penalty: Optional[float] = None,
        rfhar_eps: Optional[float] = None,
        rfhar_debug_log: Optional[bool] = None,
        enable_frgg: Optional[bool] = None,
        frgg_late_start: Optional[int] = None,
        frgg_late_end: Optional[int] = None,
        frgg_gamma: Optional[float] = None,
        frgg_tau_c: Optional[float] = None,
        frgg_tau_e: Optional[float] = None,
        frgg_k_c: Optional[float] = None,
        frgg_k_e: Optional[float] = None,
        frgg_topk_ratio: Optional[float] = None,
        frgg_eps: Optional[float] = None,
        frgg_debug_log: Optional[bool] = None,
        enable_frrs: Optional[bool] = None,
        frrs_late_start: Optional[int] = None,
        frrs_late_end: Optional[int] = None,
        frrs_alpha: Optional[float] = None,
        frrs_beta: Optional[float] = None,
        frrs_tau_c: Optional[float] = None,
        frrs_tau_e: Optional[float] = None,
        frrs_k_c: Optional[float] = None,
        frrs_k_e: Optional[float] = None,
        frrs_topk_ratio: Optional[float] = None,
        frrs_eps: Optional[float] = None,
        frrs_arm: Optional[str] = None,
        frrs_head_mode: Optional[str] = None,
        frrs_r_percent: Optional[float] = None,
        frrs_online_recompute_feats: Optional[bool] = None,
        frrs_online_blend: Optional[float] = None,
        frrs_debug_log: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if any(
            v is not None
            for v in [
                enable_ais_gating,
                ais_early_start,
                ais_early_end,
                ais_late_start,
                ais_late_end,
                ais_topk,
                ais_tau,
                ais_gamma,
                ais_eps,
                ais_debug_log,
                ais_arm,
                ais_harmful_heads,
                ais_faithful_heads,
                ais_headset_json,
                ais_faithful_boost,
                ais_use_dynamic_omega,
                ais_use_budget_routing,
                ais_budget_total,
                ais_harmful_top_ratio,
                ais_faithful_top_ratio,
                ais_bipolar_harmful_ratio,
                ais_budget_patch_topk,
                ais_strict_headset_layers,
                ais_operator,
                ais_semihard_penalty,
                ais_use_oracle_roles,
                ais_oracle_role_csv,
                ais_oracle_supportive_topk,
                ais_oracle_assertive_topk,
                ais_oracle_lambda_pos,
                ais_oracle_lambda_neg,
                ais_oracle_bias_clip,
                path_probe_mode,
                path_probe_penalty,
                path_probe_first_step_only,
                enable_rfhar,
                rfhar_early_start,
                rfhar_early_end,
                rfhar_late_start,
                rfhar_late_end,
                rfhar_r_percent,
                rfhar_gamma,
                rfhar_lambda_penalty,
                rfhar_eps,
                rfhar_debug_log,
                enable_frgg,
                frgg_late_start,
                frgg_late_end,
                frgg_gamma,
                frgg_tau_c,
                frgg_tau_e,
                frgg_k_c,
                frgg_k_e,
                frgg_topk_ratio,
                frgg_eps,
                frgg_debug_log,
                enable_frrs,
                frrs_late_start,
                frrs_late_end,
                frrs_alpha,
                frrs_beta,
                frrs_tau_c,
                frrs_tau_e,
                frrs_k_c,
                frrs_k_e,
                frrs_topk_ratio,
                frrs_eps,
                frrs_arm,
                frrs_head_mode,
                frrs_r_percent,
                frrs_online_recompute_feats,
                frrs_online_blend,
                frrs_debug_log,
            ]
        ):
            self.configure_ais_gating(
                enable_ais_gating=enable_ais_gating,
                ais_early_start=ais_early_start,
                ais_early_end=ais_early_end,
                ais_late_start=ais_late_start,
                ais_late_end=ais_late_end,
                ais_topk=ais_topk,
                ais_tau=ais_tau,
                ais_gamma=ais_gamma,
                ais_eps=ais_eps,
                ais_debug_log=ais_debug_log,
                ais_arm=ais_arm,
                ais_harmful_heads=ais_harmful_heads,
                ais_faithful_heads=ais_faithful_heads,
                ais_headset_json=ais_headset_json,
                ais_faithful_boost=ais_faithful_boost,
                ais_use_dynamic_omega=ais_use_dynamic_omega,
                ais_use_budget_routing=ais_use_budget_routing,
                ais_budget_total=ais_budget_total,
                ais_harmful_top_ratio=ais_harmful_top_ratio,
                ais_faithful_top_ratio=ais_faithful_top_ratio,
                ais_bipolar_harmful_ratio=ais_bipolar_harmful_ratio,
                ais_budget_patch_topk=ais_budget_patch_topk,
                ais_strict_headset_layers=ais_strict_headset_layers,
                ais_operator=ais_operator,
                ais_semihard_penalty=ais_semihard_penalty,
                ais_use_oracle_roles=ais_use_oracle_roles,
                ais_oracle_role_csv=ais_oracle_role_csv,
                ais_oracle_supportive_topk=ais_oracle_supportive_topk,
                ais_oracle_assertive_topk=ais_oracle_assertive_topk,
                ais_oracle_lambda_pos=ais_oracle_lambda_pos,
                ais_oracle_lambda_neg=ais_oracle_lambda_neg,
                ais_oracle_bias_clip=ais_oracle_bias_clip,
                path_probe_mode=path_probe_mode,
                path_probe_penalty=path_probe_penalty,
                path_probe_first_step_only=path_probe_first_step_only,
                enable_rfhar=enable_rfhar,
                rfhar_early_start=rfhar_early_start,
                rfhar_early_end=rfhar_early_end,
                rfhar_late_start=rfhar_late_start,
                rfhar_late_end=rfhar_late_end,
                rfhar_r_percent=rfhar_r_percent,
                rfhar_gamma=rfhar_gamma,
                rfhar_lambda_penalty=rfhar_lambda_penalty,
                rfhar_eps=rfhar_eps,
                rfhar_debug_log=rfhar_debug_log,
                enable_frgg=enable_frgg,
                frgg_late_start=frgg_late_start,
                frgg_late_end=frgg_late_end,
                frgg_gamma=frgg_gamma,
                frgg_tau_c=frgg_tau_c,
                frgg_tau_e=frgg_tau_e,
                frgg_k_c=frgg_k_c,
                frgg_k_e=frgg_k_e,
                frgg_topk_ratio=frgg_topk_ratio,
                frgg_eps=frgg_eps,
                frgg_debug_log=frgg_debug_log,
                enable_frrs=enable_frrs,
                frrs_late_start=frrs_late_start,
                frrs_late_end=frrs_late_end,
                frrs_alpha=frrs_alpha,
                frrs_beta=frrs_beta,
                frrs_tau_c=frrs_tau_c,
                frrs_tau_e=frrs_tau_e,
                frrs_k_c=frrs_k_c,
                frrs_k_e=frrs_k_e,
                frrs_topk_ratio=frrs_topk_ratio,
                frrs_eps=frrs_eps,
                frrs_arm=frrs_arm,
                frrs_head_mode=frrs_head_mode,
                frrs_r_percent=frrs_r_percent,
                frrs_online_recompute_feats=frrs_online_recompute_feats,
                frrs_online_blend=frrs_online_blend,
                frrs_debug_log=frrs_debug_log,
            )

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        ais_kwargs = {
            "enable_ais_gating": kwargs.pop("enable_ais_gating", None),
            "ais_early_start": kwargs.pop("ais_early_start", None),
            "ais_early_end": kwargs.pop("ais_early_end", None),
            "ais_late_start": kwargs.pop("ais_late_start", None),
            "ais_late_end": kwargs.pop("ais_late_end", None),
            "ais_topk": kwargs.pop("ais_topk", None),
            "ais_tau": kwargs.pop("ais_tau", None),
            "ais_gamma": kwargs.pop("ais_gamma", None),
            "ais_eps": kwargs.pop("ais_eps", None),
            "ais_debug_log": kwargs.pop("ais_debug_log", None),
            "ais_arm": kwargs.pop("ais_arm", None),
            "ais_harmful_heads": kwargs.pop("ais_harmful_heads", None),
            "ais_faithful_heads": kwargs.pop("ais_faithful_heads", None),
            "ais_headset_json": kwargs.pop("ais_headset_json", None),
            "ais_faithful_boost": kwargs.pop("ais_faithful_boost", None),
            "ais_use_dynamic_omega": kwargs.pop("ais_use_dynamic_omega", None),
            "ais_use_budget_routing": kwargs.pop("ais_use_budget_routing", None),
            "ais_budget_total": kwargs.pop("ais_budget_total", None),
            "ais_harmful_top_ratio": kwargs.pop("ais_harmful_top_ratio", None),
            "ais_faithful_top_ratio": kwargs.pop("ais_faithful_top_ratio", None),
            "ais_bipolar_harmful_ratio": kwargs.pop("ais_bipolar_harmful_ratio", None),
            "ais_budget_patch_topk": kwargs.pop("ais_budget_patch_topk", None),
            "ais_strict_headset_layers": kwargs.pop("ais_strict_headset_layers", None),
            "ais_operator": kwargs.pop("ais_operator", None),
            "ais_semihard_penalty": kwargs.pop("ais_semihard_penalty", None),
            "ais_use_oracle_roles": kwargs.pop("ais_use_oracle_roles", None),
            "ais_oracle_role_csv": kwargs.pop("ais_oracle_role_csv", None),
            "ais_oracle_supportive_topk": kwargs.pop("ais_oracle_supportive_topk", None),
            "ais_oracle_assertive_topk": kwargs.pop("ais_oracle_assertive_topk", None),
            "ais_oracle_lambda_pos": kwargs.pop("ais_oracle_lambda_pos", None),
            "ais_oracle_lambda_neg": kwargs.pop("ais_oracle_lambda_neg", None),
            "ais_oracle_bias_clip": kwargs.pop("ais_oracle_bias_clip", None),
            "path_probe_mode": kwargs.pop("path_probe_mode", None),
            "path_probe_penalty": kwargs.pop("path_probe_penalty", None),
            "path_probe_first_step_only": kwargs.pop("path_probe_first_step_only", None),
            "enable_rfhar": kwargs.pop("enable_rfhar", None),
            "rfhar_early_start": kwargs.pop("rfhar_early_start", None),
            "rfhar_early_end": kwargs.pop("rfhar_early_end", None),
            "rfhar_late_start": kwargs.pop("rfhar_late_start", None),
            "rfhar_late_end": kwargs.pop("rfhar_late_end", None),
            "rfhar_r_percent": kwargs.pop("rfhar_r_percent", None),
            "rfhar_gamma": kwargs.pop("rfhar_gamma", None),
            "rfhar_lambda_penalty": kwargs.pop("rfhar_lambda_penalty", None),
            "rfhar_eps": kwargs.pop("rfhar_eps", None),
            "rfhar_debug_log": kwargs.pop("rfhar_debug_log", None),
            "enable_frgg": kwargs.pop("enable_frgg", None),
            "frgg_late_start": kwargs.pop("frgg_late_start", None),
            "frgg_late_end": kwargs.pop("frgg_late_end", None),
            "frgg_gamma": kwargs.pop("frgg_gamma", None),
            "frgg_tau_c": kwargs.pop("frgg_tau_c", None),
            "frgg_tau_e": kwargs.pop("frgg_tau_e", None),
            "frgg_k_c": kwargs.pop("frgg_k_c", None),
            "frgg_k_e": kwargs.pop("frgg_k_e", None),
            "frgg_topk_ratio": kwargs.pop("frgg_topk_ratio", None),
            "frgg_eps": kwargs.pop("frgg_eps", None),
            "frgg_debug_log": kwargs.pop("frgg_debug_log", None),
            "enable_frrs": kwargs.pop("enable_frrs", None),
            "frrs_late_start": kwargs.pop("frrs_late_start", None),
            "frrs_late_end": kwargs.pop("frrs_late_end", None),
            "frrs_alpha": kwargs.pop("frrs_alpha", None),
            "frrs_beta": kwargs.pop("frrs_beta", None),
            "frrs_tau_c": kwargs.pop("frrs_tau_c", None),
            "frrs_tau_e": kwargs.pop("frrs_tau_e", None),
            "frrs_k_c": kwargs.pop("frrs_k_c", None),
            "frrs_k_e": kwargs.pop("frrs_k_e", None),
            "frrs_topk_ratio": kwargs.pop("frrs_topk_ratio", None),
            "frrs_eps": kwargs.pop("frrs_eps", None),
            "frrs_arm": kwargs.pop("frrs_arm", None),
            "frrs_head_mode": kwargs.pop("frrs_head_mode", None),
            "frrs_r_percent": kwargs.pop("frrs_r_percent", None),
            "frrs_online_recompute_feats": kwargs.pop("frrs_online_recompute_feats", None),
            "frrs_online_blend": kwargs.pop("frrs_online_blend", None),
            "frrs_debug_log": kwargs.pop("frrs_debug_log", None),
            "enable_proxy_trace": kwargs.pop("enable_proxy_trace", None),
            "proxy_late_start": kwargs.pop("proxy_late_start", None),
            "proxy_late_end": kwargs.pop("proxy_late_end", None),
            "proxy_eps": kwargs.pop("proxy_eps", None),
        }
        ais_sample_ids = kwargs.pop("ais_sample_ids", None)
        rfhar_feats = kwargs.pop("rfhar_feats", None)
        frgg_feats = kwargs.pop("frgg_feats", None)
        frrs_feats = kwargs.pop("frrs_feats", None)
        if any(v is not None for v in ais_kwargs.values()):
            self.configure_ais_gating(**ais_kwargs)

        rt = self._ensure_ais_runtime()
        self._ais_capture_image_mask = bool(rt.is_effective_enabled())
        if bool(self._ais_capture_image_mask):
            self._ensure_ais_hooks()
            self._ais_image_token_mask = None

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if bool(self._ais_capture_image_mask):
            rt.begin_generation(
                getattr(self, "_ais_image_token_mask", None),
                sample_ids=ais_sample_ids,
                rfhar_feats=rfhar_feats,
                frgg_feats=frgg_feats,
                frrs_feats=frrs_feats,
            )
        try:
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        finally:
            self._ais_capture_image_mask = False
            if bool(rt.active):
                rt.end_generation()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
