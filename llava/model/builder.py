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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    enable_ais_gating=False,
    ais_early_start=0,
    ais_early_end=15,
    ais_late_start=16,
    ais_late_end=31,
    ais_topk=8,
    ais_tau=2.2,
    ais_gamma=0.2,
    ais_eps=1e-6,
    ais_debug_log=False,
    ais_arm="legacy",
    ais_harmful_heads="",
    ais_faithful_heads="",
    ais_headset_json="",
    ais_faithful_boost=1.0,
    ais_use_dynamic_omega=True,
    ais_use_budget_routing=False,
    ais_budget_total=0.0,
    ais_harmful_top_ratio=0.2,
    ais_faithful_top_ratio=0.2,
    ais_bipolar_harmful_ratio=0.5,
    ais_budget_patch_topk=16,
    ais_strict_headset_layers=False,
    ais_operator="soft",
    ais_semihard_penalty=0.0,
    ais_use_oracle_roles=False,
    ais_oracle_role_csv="",
    ais_oracle_supportive_topk=5,
    ais_oracle_assertive_topk=5,
    ais_oracle_lambda_pos=0.25,
    ais_oracle_lambda_neg=0.25,
    ais_oracle_bias_clip=2.0,
    path_probe_mode="none",
    path_probe_penalty=0.0,
    path_probe_first_step_only=True,
    enable_rfhar=False,
    rfhar_early_start=0,
    rfhar_early_end=15,
    rfhar_late_start=16,
    rfhar_late_end=31,
    rfhar_r_percent=0.2,
    rfhar_gamma=0.3,
    rfhar_lambda_penalty=0.5,
    rfhar_eps=1e-6,
    rfhar_debug_log=False,
    enable_frgg=False,
    frgg_late_start=16,
    frgg_late_end=30,
    frgg_gamma=0.3,
    frgg_tau_c=0.0,
    frgg_tau_e=0.0,
    frgg_k_c=8.0,
    frgg_k_e=8.0,
    frgg_topk_ratio=0.2,
    frgg_eps=1e-6,
    frgg_debug_log=False,
    enable_frrs=False,
    frrs_late_start=18,
    frrs_late_end=21,
    frrs_alpha=0.5,
    frrs_beta=0.5,
    frrs_tau_c=0.0,
    frrs_tau_e=0.0,
    frrs_k_c=8.0,
    frrs_k_e=8.0,
    frrs_topk_ratio=0.2,
    frrs_eps=1e-6,
    frrs_arm="supportive",
    frrs_head_mode="dynamic",
    frrs_r_percent=0.2,
    frrs_online_recompute_feats=False,
    frrs_online_blend=1.0,
    frrs_debug_log=False,
    **kwargs
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor
        if bool(enable_ais_gating or enable_rfhar or enable_frgg or enable_frrs) and hasattr(model, "configure_ais_gating"):
            model.configure_ais_gating(
                enable_ais_gating=bool(enable_ais_gating),
                ais_early_start=int(ais_early_start),
                ais_early_end=int(ais_early_end),
                ais_late_start=int(ais_late_start),
                ais_late_end=int(ais_late_end),
                ais_topk=int(ais_topk),
                ais_tau=float(ais_tau),
                ais_gamma=float(ais_gamma),
                ais_eps=float(ais_eps),
                ais_debug_log=bool(ais_debug_log),
                ais_arm=str(ais_arm),
                ais_harmful_heads=str(ais_harmful_heads),
                ais_faithful_heads=str(ais_faithful_heads),
                ais_headset_json=str(ais_headset_json),
                ais_faithful_boost=float(ais_faithful_boost),
                ais_use_dynamic_omega=bool(ais_use_dynamic_omega),
                ais_use_budget_routing=bool(ais_use_budget_routing),
                ais_budget_total=float(ais_budget_total),
                ais_harmful_top_ratio=float(ais_harmful_top_ratio),
                ais_faithful_top_ratio=float(ais_faithful_top_ratio),
                ais_bipolar_harmful_ratio=float(ais_bipolar_harmful_ratio),
                ais_budget_patch_topk=int(ais_budget_patch_topk),
                ais_strict_headset_layers=bool(ais_strict_headset_layers),
                ais_operator=str(ais_operator),
                ais_semihard_penalty=float(ais_semihard_penalty),
                ais_use_oracle_roles=bool(ais_use_oracle_roles),
                ais_oracle_role_csv=str(ais_oracle_role_csv),
                ais_oracle_supportive_topk=int(ais_oracle_supportive_topk),
                ais_oracle_assertive_topk=int(ais_oracle_assertive_topk),
                ais_oracle_lambda_pos=float(ais_oracle_lambda_pos),
                ais_oracle_lambda_neg=float(ais_oracle_lambda_neg),
                ais_oracle_bias_clip=float(ais_oracle_bias_clip),
                path_probe_mode=str(path_probe_mode),
                path_probe_penalty=float(path_probe_penalty),
                path_probe_first_step_only=bool(path_probe_first_step_only),
                enable_rfhar=bool(enable_rfhar),
                rfhar_early_start=int(rfhar_early_start),
                rfhar_early_end=int(rfhar_early_end),
                rfhar_late_start=int(rfhar_late_start),
                rfhar_late_end=int(rfhar_late_end),
                rfhar_r_percent=float(rfhar_r_percent),
                rfhar_gamma=float(rfhar_gamma),
                rfhar_lambda_penalty=float(rfhar_lambda_penalty),
                rfhar_eps=float(rfhar_eps),
                rfhar_debug_log=bool(rfhar_debug_log),
                enable_frgg=bool(enable_frgg),
                frgg_late_start=int(frgg_late_start),
                frgg_late_end=int(frgg_late_end),
                frgg_gamma=float(frgg_gamma),
                frgg_tau_c=float(frgg_tau_c),
                frgg_tau_e=float(frgg_tau_e),
                frgg_k_c=float(frgg_k_c),
                frgg_k_e=float(frgg_k_e),
                frgg_topk_ratio=float(frgg_topk_ratio),
                frgg_eps=float(frgg_eps),
                frgg_debug_log=bool(frgg_debug_log),
                enable_frrs=bool(enable_frrs),
                frrs_late_start=int(frrs_late_start),
                frrs_late_end=int(frrs_late_end),
                frrs_alpha=float(frrs_alpha),
                frrs_beta=float(frrs_beta),
                frrs_tau_c=float(frrs_tau_c),
                frrs_tau_e=float(frrs_tau_e),
                frrs_k_c=float(frrs_k_c),
                frrs_k_e=float(frrs_k_e),
                frrs_topk_ratio=float(frrs_topk_ratio),
                frrs_eps=float(frrs_eps),
                frrs_arm=str(frrs_arm),
                frrs_head_mode=str(frrs_head_mode),
                frrs_r_percent=float(frrs_r_percent),
                frrs_online_recompute_feats=bool(frrs_online_recompute_feats),
                frrs_online_blend=float(frrs_online_blend),
                frrs_debug_log=bool(frrs_debug_log),
            )

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
