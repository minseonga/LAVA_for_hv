from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from pnp_controller.core.runtime_features import (
    compute_aggregate_probe_scores,
    combine_gmi_with_guidance_mass,
    compute_head_attn_vis_ratio_last_row,
    image_span_from_prompt_input_ids,
    topk_mass,
)
from pnp_controller.core.schemas import ProbeState

from .base import OnlineMethodAdapter


@dataclass
class VGAOnlineConfig:
    vga_root: str
    model_path: str
    image_folder: str
    conv_mode: str = "llava_v1"
    model_base: Optional[str] = None
    device: str = "cuda"
    temperature: float = 1.0
    top_p: float = 1.0
    sampling: bool = False
    max_gen_len: int = 8
    num_beams: int = 1
    cd_alpha: float = 0.02
    attn_coef: float = 0.2
    start_layer: int = 16
    end_layer: int = 24
    head_balancing: str = "simg"
    attn_norm: bool = False
    late_start: int = 16
    late_end: int = 24
    probe_feature_mode: str = "static_headset"
    aggregate_frg_metric: str = "frg_shared_topk"
    aggregate_gmi_metric: str = "e_agg_js"
    aggregate_topk: int = 5
    aggregate_lambda: float = 1.0
    headset_json: str = ""
    use_gmi: bool = True  # Set False to skip GMI (use FRG-only veto)
    seed: int = 42


class VGAOnlineAdapter(OnlineMethodAdapter):
    backend_name = "vga_online"

    def __init__(self, cfg: VGAOnlineConfig) -> None:
        self.cfg = cfg
        self.vga_root = os.path.abspath(cfg.vga_root)
        self.image_folder = os.path.abspath(cfg.image_folder)
        self.device = torch.device(cfg.device)
        self._bootstrap_vga_imports()
        self._load_model()
        self._load_headset()

    def _bootstrap_vga_imports(self) -> None:
        if self.vga_root not in sys.path:
            sys.path.insert(0, self.vga_root)

        from transformers import set_seed  # type: ignore
        from llava.constants import (  # type: ignore
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates  # type: ignore
        from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # type: ignore
        from llava.model.builder import load_pretrained_model  # type: ignore
        from llava.utils import disable_torch_init  # type: ignore
        from vcd_utils.greedy_sample import evolve_greedy_sampling  # type: ignore

        evolve_greedy_sampling()
        set_seed(int(self.cfg.seed))

        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token
        self.get_model_name_from_path = get_model_name_from_path
        self.load_pretrained_model = load_pretrained_model
        self.disable_torch_init = disable_torch_init

    def _load_model(self) -> None:
        self.disable_torch_init()
        model_name = self.get_model_name_from_path(os.path.expanduser(self.cfg.model_path))
        tokenizer, model, image_processor, _ = self.load_pretrained_model(
            os.path.expanduser(self.cfg.model_path),
            self.cfg.model_base,
            model_name,
            device=self.cfg.device,
        )
        tokenizer.padding_side = "right"
        model.model.lm_head = model.lm_head
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.model_name = model_name

    def _load_headset(self) -> None:
        self.faithful_heads_by_layer: Dict[int, List[int]] = {}
        self.harmful_heads_by_layer: Dict[int, List[int]] = {}
        if str(self.cfg.probe_feature_mode) != "static_headset":
            return
        p = str(self.cfg.headset_json or "").strip()
        if p == "":
            raise ValueError("headset_json is required in static_headset probe mode.")
        obj = json.loads(Path(p).read_text())
        faithful = obj.get("faithful_heads_by_layer", {}) if isinstance(obj, dict) else {}
        harmful = obj.get("harmful_heads_by_layer", {}) if isinstance(obj, dict) else {}
        if isinstance(faithful, dict):
            self.faithful_heads_by_layer = {int(k): [int(x) for x in v] for k, v in faithful.items()}
        if isinstance(harmful, dict):
            self.harmful_heads_by_layer = {int(k): [int(x) for x in v] for k, v in harmful.items()}

    def _sample_id(self, sample: Dict[str, Any]) -> str:
        for key in ("question_id", "id", "qid", "image_id"):
            v = sample.get(key, None)
            if v is not None and str(v).strip() != "":
                return str(v)
        return ""

    def _safe_object_list(self, raw_obj: Any) -> List[str]:
        if raw_obj is None:
            return []
        if isinstance(raw_obj, str):
            raw_obj = [raw_obj]
        if not isinstance(raw_obj, list):
            return []
        out: List[str] = []
        for x in raw_obj:
            s = str(x).strip()
            if s != "":
                out.append(s)
        return out

    def _build_prompt(self, question: str) -> Tuple[str, str]:
        qs = str(question).strip()
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs_prompt = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs_prompt = self.DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = self.conv_templates[self.cfg.conv_mode].copy()
        conv.append_message(conv.roles[0], qs_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        return prompt, stop_str

    def _prepare_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = str(sample.get("question", sample.get("text", ""))).strip()
        if question == "":
            raise ValueError("Sample is missing question text.")
        image_file = str(sample.get("image", "")).strip()
        if image_file == "":
            raise ValueError("Sample is missing image filename.")

        prompt, stop_str = self._build_prompt(question)
        input_ids = self.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.to(self.device)
        if self.device.type == "cuda":
            image_tensor = image_tensor.half()
        else:
            image_tensor = image_tensor.float()

        return {
            "sample_id": self._sample_id(sample),
            "question": question,
            "image_file": image_file,
            "prompt": prompt,
            "stop_str": stop_str,
            "input_ids": input_ids,
            "image_tensor": image_tensor,
            "object_list": self._safe_object_list(sample.get("object", None)),
        }

    def _compute_vl_guidance(self, vis_logits: torch.Tensor, object_list: List[str]) -> Tuple[torch.Tensor, str]:
        object_ids = []
        for obj in object_list:
            ids = self.tokenizer(obj, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if ids.numel() > 0:
                object_ids.append(ids.to(vis_logits.device))

        if object_ids:
            grounding = []
            for ids in object_ids:
                vl = vis_logits[:, ids]
                vl = vl[:, 0]
                grounding.append(vl)
            grounding_t = torch.stack(grounding, dim=0).max(0).values
            grounding_t = grounding_t / torch.clamp(grounding_t.sum(), min=1e-8)
            return grounding_t.to(vis_logits.dtype), "object"

        top_k_scores, _ = torch.topk(vis_logits, 10, dim=-1)
        top_k_scores = top_k_scores.float()
        probabilities = -top_k_scores * torch.log(top_k_scores + 1e-8) / torch.log(torch.tensor(10.0, device=top_k_scores.device))
        entropy = probabilities.sum(-1)
        vl_guidance = entropy / torch.clamp(entropy.sum(), min=1e-8)
        return vl_guidance.to(vis_logits.dtype), "entropy_fallback"

    def probe(self, sample: Any) -> ProbeState:
        prepared = self._prepare_sample(sample)
        input_ids = prepared["input_ids"]
        image_tensor = prepared["image_tensor"]

        with torch.inference_mode():
            prefill = self.model(
                input_ids[:, :-1],
                images=image_tensor.unsqueeze(0),
                use_cache=True,
                return_dict=True,
            )
            logits = prefill.logits
            vis_logits = F.softmax(logits[0, 35:611, :], dim=-1).float()
            vl_guidance, guidance_mode = self._compute_vl_guidance(vis_logits, prepared["object_list"])

            probe_last = self.model(
                input_ids[:, -1:],
                attention_mask=torch.ones((1, 1), dtype=torch.long, device=self.device),
                past_key_values=prefill.past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        image_start, image_end = image_span_from_prompt_input_ids(
            input_ids=input_ids,
            image_token_index=int(self.IMAGE_TOKEN_INDEX),
            image_token_count=int(vis_logits.size(0)),
        )
        g_top5_mass = topk_mass(vl_guidance.to(torch.float32), k=5)
        if str(self.cfg.probe_feature_mode) == "aggregate":
            agg_stats = compute_aggregate_probe_scores(
                attentions=probe_last.attentions,
                image_start=image_start,
                image_end=image_end,
                late_start=int(self.cfg.late_start),
                late_end=int(self.cfg.late_end),
                guidance=vl_guidance.to(torch.float32),
                topk=int(self.cfg.aggregate_topk),
                mismatch_lambda=float(self.cfg.aggregate_lambda),
            )
            frg_key = str(self.cfg.aggregate_frg_metric)
            gmi_key = str(self.cfg.aggregate_gmi_metric)
            if frg_key not in agg_stats:
                raise KeyError(f"Unknown aggregate FRG metric: {frg_key}")
            if gmi_key not in agg_stats:
                raise KeyError(f"Unknown aggregate GMI metric: {gmi_key}")
            frg = float(agg_stats[frg_key])
            gmi = float(agg_stats[gmi_key])
            attn_stats = {
                "aggregate_mode": True,
                "alpha_img_mean": agg_stats["alpha_img_mean"].tolist(),
                "late_head_vis_ratio_mean": float(agg_stats["late_head_vis_ratio_mean"]),
                "late_head_vis_ratio_topkmean": float(agg_stats["late_head_vis_ratio_topkmean"]),
                "late_head_count": float(agg_stats["late_head_count"]),
                "frg_shared_mean": float(agg_stats["frg_shared_mean"]),
                "frg_shared_topk": float(agg_stats["frg_shared_topk"]),
                "c_agg_cos": float(agg_stats["c_agg_cos"]),
                "c_agg_ip": float(agg_stats["c_agg_ip"]),
                "e_agg_js": float(agg_stats["e_agg_js"]),
                "e_agg_combo": float(agg_stats["e_agg_combo"]),
                "topk_guidance_coverage": float(agg_stats["topk_guidance_coverage"]),
            }
        else:
            attn_stats = compute_head_attn_vis_ratio_last_row(
                attentions=probe_last.attentions,
                image_start=image_start,
                image_end=image_end,
                late_start=int(self.cfg.late_start),
                late_end=int(self.cfg.late_end),
                faithful_heads_by_layer=self.faithful_heads_by_layer,
                harmful_heads_by_layer=self.harmful_heads_by_layer,
            )
            frg = float(attn_stats["faithful_minus_global_attn"])
            if bool(self.cfg.use_gmi):
                gmi = float(
                    combine_gmi_with_guidance_mass(
                        faithful_head_attn_mean=float(attn_stats["faithful_head_attn_mean"]),
                        harmful_head_attn_mean=float(attn_stats["harmful_head_attn_mean"]),
                        g_top5_mass=float(g_top5_mass),
                    )
                )
            else:
                # FRG-only mode: skip GMI (no harmful heads needed)
                gmi = 0.0

        common_gen_kwargs = dict(
            images=image_tensor.unsqueeze(0),
            past_key_values=prefill.past_key_values,
            vl_guidance=vl_guidance,
            vis_logits=vis_logits,
            cd_alpha=float(self.cfg.cd_alpha),
            add_layer=list(range(int(self.cfg.start_layer), int(self.cfg.end_layer) + 1)),
            attn_coef=float(self.cfg.attn_coef),
            head_balancing=str(self.cfg.head_balancing),
            attn_norm=bool(self.cfg.attn_norm),
            do_sample=True,
            sampling=bool(self.cfg.sampling),
            num_beams=int(self.cfg.num_beams),
            max_new_tokens=int(self.cfg.max_gen_len),
            use_cache=True,
        )
        extras = {
            "prepared": prepared,
            "common_gen_kwargs": common_gen_kwargs,
            "probe_feature_mode": str(self.cfg.probe_feature_mode),
            "guidance_mode": guidance_mode,
            "g_top5_mass": float(g_top5_mass),
            "image_start": int(image_start),
            "image_end": int(image_end),
            "attn_stats": attn_stats,
        }
        return ProbeState(sample_id=prepared["sample_id"], frg=frg, gmi=gmi, extras=extras)

    def _generate(self, sample: Dict[str, Any], probe_state: ProbeState, use_add: bool) -> Dict[str, Any]:
        extras = probe_state.extras
        prepared = extras["prepared"]
        gen_kwargs = dict(extras["common_gen_kwargs"])
        gen_kwargs["use_add"] = bool(use_add)

        with torch.inference_mode():
            output_ids = self.model.generate(
                prepared["input_ids"][:, -1:],
                **gen_kwargs,
            )

        output_text = self.tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)[0]
        output_text = output_text.split("ASSISTANT:")[-1].strip()
        stop_str = prepared["stop_str"]
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)]
        output_text = output_text.strip()
        return {
            "question_id": sample.get("question_id", sample.get("id", sample.get("qid", prepared["sample_id"]))),
            "question": prepared["question"],
            "output": output_text,
            "label": sample.get("label", sample.get("answer", None)),
            "prompt": prepared["prompt"],
            "model_id": self.model_name,
            "image": prepared["image_file"],
            "image_id": sample.get("image_id", None),
            "route_mode": "method" if use_add else "baseline",
        }

    def predict_base(self, sample: Any, probe_state: ProbeState | None = None) -> Any:
        if probe_state is None:
            probe_state = self.probe(sample)
        return self._generate(sample=sample, probe_state=probe_state, use_add=False)

    def predict_method(self, sample: Any, probe_state: ProbeState | None = None) -> Any:
        if probe_state is None:
            probe_state = self.probe(sample)
        return self._generate(sample=sample, probe_state=probe_state, use_add=True)
