from __future__ import annotations

import difflib
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
    compute_aggregate_probe_scores_at_row,
    combine_gmi_with_guidance_mass,
    compute_head_attn_vis_ratio_last_row,
    compute_head_attn_vis_ratio_at_row,
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
    probe_position_mode: str = "prompt_last"
    probe_preview_max_new_tokens: int = 3
    probe_preview_reuse_baseline: bool = True
    probe_preview_fallback_to_prompt_last: bool = True
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

        # VGA_origin still imports several private bloom/opt helpers that were
        # removed in newer transformers releases. We never use those codepaths
        # here, so provide no-op shims before importing llava from VGA_origin.
        _noop = lambda *a, **kw: a[0] if a else None
        import transformers.models.bloom.modeling_bloom as _bloom_mod  # type: ignore
        for _fn in ("_expand_mask", "_make_causal_mask"):
            if not hasattr(_bloom_mod, _fn):
                setattr(_bloom_mod, _fn, _noop)
        import transformers.models.opt.modeling_opt as _opt_mod  # type: ignore
        for _fn in ("_expand_mask", "_make_causal_mask"):
            if not hasattr(_opt_mod, _fn):
                setattr(_opt_mod, _fn, _noop)

        from transformers import set_seed  # type: ignore
        from llava.constants import (  # type: ignore
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IGNORE_INDEX,
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
        self.IGNORE_INDEX = IGNORE_INDEX
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
            "image_size": image.size,
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

    def _normalize_yes_no_anchor(self, text: str) -> Optional[str]:
        s = str(text or "").strip().lower()
        if s == "":
            return None
        first = s.split(".", 1)[0].replace(",", " ").replace("\n", " ")
        words = [w.strip().strip("!?;:") for w in first.split() if w.strip() != ""]
        if "no" in words or "not" in words:
            return "no"
        if "yes" in words:
            return "yes"
        return None

    def _choose_cont_ids(self, text: str) -> List[int]:
        raw = str(text or "")
        if raw.strip() == "":
            return []
        cands = [raw, " " + raw]
        best: List[int] = []
        best_score = -1.0
        target = " ".join(raw.strip().lower().split())
        for s in cands:
            ids = [int(x) for x in self.tokenizer(s, add_special_tokens=False).input_ids]
            dec = self.tokenizer.decode(ids, skip_special_tokens=True)
            score = difflib.SequenceMatcher(None, target, " ".join(dec.strip().lower().split())).ratio()
            if score > best_score:
                best = ids
                best_score = float(score)
        return best

    def _locate_phrase_token_start(self, token_ids: List[int], phrase: str) -> Optional[int]:
        seq = [int(x) for x in token_ids]
        if not seq or str(phrase or "").strip() == "":
            return None

        cand_ids: List[List[int]] = []
        seen = set()
        for raw in (str(phrase), " " + str(phrase)):
            ids = [int(x) for x in self.tokenizer(raw, add_special_tokens=False).input_ids]
            if not ids:
                continue
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            cand_ids.append(ids)

        for ids in cand_ids:
            width = int(len(ids))
            for start in range(0, max(0, len(seq) - width + 1)):
                if seq[start:start + width] == ids:
                    return int(start)

        for idx, tid in enumerate(seq):
            piece = self.tokenizer.decode([int(tid)], skip_special_tokens=True).strip().lower()
            if piece == str(phrase):
                return int(idx)
        return None

    def _decode_generated_text(self, output_ids: torch.Tensor, stop_str: str) -> str:
        output_text = self.tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)[0]
        output_text = output_text.split("ASSISTANT:")[-1].strip()
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)]
        return output_text.strip()

    def _build_prediction_row(
        self,
        sample: Dict[str, Any],
        prepared: Dict[str, Any],
        output_text: str,
        use_add: bool,
    ) -> Dict[str, Any]:
        return {
            "question_id": sample.get("question_id", sample.get("id", sample.get("qid", prepared["sample_id"]))),
            "question": prepared["question"],
            "output": str(output_text).strip(),
            "label": sample.get("label", sample.get("answer", None)),
            "prompt": prepared["prompt"],
            "model_id": self.model_name,
            "image": prepared["image_file"],
            "image_id": sample.get("image_id", None),
            "route_mode": "method" if use_add else "baseline",
        }

    def _run_prompt_prefill(
        self,
        input_ids: torch.Tensor,
        image_tensor: torch.Tensor,
    ) -> Any:
        return self.model(
            input_ids[:, :-1],
            images=image_tensor.unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )

    def _prepare_multimodal_expanded_sequence(
        self,
        full_ids: torch.Tensor,
        image_tensor: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        base_attn = torch.ones_like(full_ids, dtype=torch.long, device=self.device)
        fn = self.model.prepare_inputs_labels_for_multimodal
        try:
            packed = fn(
                full_ids,
                None,
                base_attn,
                None,
                full_ids,
                image_tensor.unsqueeze(0),
                [image_size],
            )
        except TypeError as new_sig_exc:
            try:
                packed = fn(
                    full_ids,
                    base_attn,
                    None,
                    full_ids,
                    image_tensor.unsqueeze(0),
                )
            except TypeError:
                raise new_sig_exc
            if not isinstance(packed, tuple) or len(packed) != 5:
                raise RuntimeError("Unexpected legacy prepare_inputs_labels_for_multimodal return shape.")
            _, attn_mask_e, _, mm_embeds_e, labels_e = packed
            pos_ids_e = None
        else:
            if not isinstance(packed, tuple) or len(packed) != 6:
                raise RuntimeError("Unexpected prepare_inputs_labels_for_multimodal return shape.")
            _, pos_ids_e, attn_mask_e, _, mm_embeds_e, labels_e = packed

        if mm_embeds_e is None or labels_e is None or attn_mask_e is None:
            raise RuntimeError("prepare_inputs_labels_for_multimodal did not return expected multimodal tensors.")
        return pos_ids_e, attn_mask_e, mm_embeds_e, labels_e

    def _find_cont_label_positions(self, labels_expanded: torch.Tensor, cont_ids: List[int]) -> Optional[torch.Tensor]:
        if labels_expanded.ndim != 1:
            return None
        tlen = int(len(cont_ids))
        if tlen <= 0:
            return None
        valid_pos = torch.where(labels_expanded != int(self.IGNORE_INDEX))[0]
        if int(valid_pos.numel()) < tlen:
            return None

        tail_pos = valid_pos[-tlen:]
        tail_ids = labels_expanded[tail_pos].tolist()
        if [int(x) for x in tail_ids] == [int(x) for x in cont_ids]:
            return tail_pos

        valid_ids = labels_expanded[valid_pos].tolist()
        target = [int(x) for x in cont_ids]
        for start in range(0, int(len(valid_ids) - tlen + 1)):
            if [int(x) for x in valid_ids[start : start + tlen]] == target:
                return valid_pos[start : start + tlen]
        return None

    def _run_cached_probe_step(
        self,
        token_ids: torch.Tensor,
        past_key_values: Any,
    ) -> Any:
        with torch.inference_mode():
            return self.model(
                token_ids,
                attention_mask=torch.ones((1, int(token_ids.size(1))), dtype=torch.long, device=self.device),
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

    def _compute_probe_metrics(
        self,
        attentions: Any,
        image_start: int,
        image_end: int,
        guidance: torch.Tensor,
        g_top5_mass: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        if str(self.cfg.probe_feature_mode) == "aggregate":
            agg_stats = compute_aggregate_probe_scores(
                attentions=attentions,
                image_start=image_start,
                image_end=image_end,
                late_start=int(self.cfg.late_start),
                late_end=int(self.cfg.late_end),
                guidance=guidance.to(torch.float32),
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
            return frg, gmi, attn_stats

        attn_stats = compute_head_attn_vis_ratio_last_row(
            attentions=attentions,
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
            gmi = 0.0
        return frg, gmi, attn_stats

    def _compute_probe_metrics_at_decision_row(
        self,
        attentions: Any,
        decision_pos: int,
        vision_positions: torch.Tensor,
        text_positions: torch.Tensor,
        guidance: torch.Tensor,
        g_top5_mass: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        if str(self.cfg.probe_feature_mode) == "aggregate":
            agg_stats = compute_aggregate_probe_scores_at_row(
                attentions=attentions,
                decision_pos=int(decision_pos),
                vision_positions=vision_positions,
                text_positions=text_positions,
                late_start=int(self.cfg.late_start),
                late_end=int(self.cfg.late_end),
                guidance=guidance.to(torch.float32),
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
            return frg, gmi, attn_stats

        attn_stats = compute_head_attn_vis_ratio_at_row(
            attentions=attentions,
            decision_pos=int(decision_pos),
            vision_positions=vision_positions,
            text_positions=text_positions,
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
            gmi = 0.0
        return frg, gmi, attn_stats

    def _probe_prompt_last(
        self,
        input_ids: torch.Tensor,
        prefill: Any,
        image_start: int,
        image_end: int,
        guidance: torch.Tensor,
        g_top5_mass: float,
    ) -> Tuple[float, float, Dict[str, Any], Dict[str, Any]]:
        probe_last = self._run_cached_probe_step(input_ids[:, -1:], prefill.past_key_values)
        frg, gmi, attn_stats = self._compute_probe_metrics(
            attentions=probe_last.attentions,
            image_start=image_start,
            image_end=image_end,
            guidance=guidance,
            g_top5_mass=g_top5_mass,
        )
        debug = {
            "probe_source": "prompt_last",
            "probe_anchor": "",
            "probe_anchor_token_idx": -1,
            "baseline_preview_text": "",
            "baseline_preview_reusable": False,
            "baseline_preview_found_anchor": False,
            "baseline_preview_fallback": False,
        }
        return frg, gmi, attn_stats, debug

    def _run_baseline_preview(
        self,
        prepared: Dict[str, Any],
        common_gen_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], str]:
        preview_kwargs = dict(common_gen_kwargs)
        preview_kwargs["use_add"] = False
        preview_kwargs["max_new_tokens"] = max(1, int(self.cfg.probe_preview_max_new_tokens))
        with torch.inference_mode():
            output_ids = self.model.generate(
                prepared["input_ids"][:, -1:],
                **preview_kwargs,
            )
        preview_ids = [int(x) for x in output_ids[0, 1:].tolist()]
        preview_text = self._decode_generated_text(output_ids, prepared["stop_str"])
        return preview_ids, preview_text

    def _probe_baseline_yesno_preview(
        self,
        prepared: Dict[str, Any],
        input_ids: torch.Tensor,
        prefill: Any,
        image_start: int,
        image_end: int,
        guidance: torch.Tensor,
        g_top5_mass: float,
        common_gen_kwargs: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        first_step = self._run_cached_probe_step(input_ids[:, -1:], prefill.past_key_values)
        preview_ids, preview_text = self._run_baseline_preview(prepared, common_gen_kwargs)
        anchor_phrase = self._normalize_yes_no_anchor(preview_text)
        anchor_idx = self._locate_phrase_token_start(preview_ids, anchor_phrase) if anchor_phrase is not None else None
        debug = {
            "probe_source": "baseline_yesno_preview",
            "probe_anchor": str(anchor_phrase or ""),
            "probe_anchor_token_idx": int(anchor_idx) if anchor_idx is not None else -1,
            "baseline_preview_text": str(preview_text or ""),
            "baseline_preview_reusable": bool(
                bool(self.cfg.probe_preview_reuse_baseline)
                and str(preview_text or "").strip() != ""
                and anchor_phrase is not None
            ),
            "baseline_preview_found_anchor": bool(anchor_idx is not None),
            "baseline_preview_fallback": False,
        }
        if anchor_idx is None:
            debug["baseline_preview_fallback"] = True
            return None, None, {}, debug

        decision_attentions = first_step.attentions
        past_key_values = first_step.past_key_values
        for idx in range(int(anchor_idx)):
            token_tensor = torch.tensor([[int(preview_ids[idx])]], dtype=torch.long, device=self.device)
            step = self._run_cached_probe_step(token_tensor, past_key_values)
            past_key_values = step.past_key_values
            decision_attentions = step.attentions

        frg, gmi, attn_stats = self._compute_probe_metrics(
            attentions=decision_attentions,
            image_start=image_start,
            image_end=image_end,
            guidance=guidance,
            g_top5_mass=g_top5_mass,
        )
        return frg, gmi, attn_stats, debug

    def _probe_baseline_yesno_offline_fullseq(
        self,
        prepared: Dict[str, Any],
        input_ids: torch.Tensor,
        guidance: torch.Tensor,
        g_top5_mass: float,
        common_gen_kwargs: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        preview_ids, preview_text = self._run_baseline_preview(prepared, common_gen_kwargs)
        anchor_phrase = self._normalize_yes_no_anchor(preview_text)
        cont_ids = self._choose_cont_ids(preview_text)
        anchor_idx = self._locate_phrase_token_start(cont_ids, anchor_phrase) if anchor_phrase is not None else None
        preview_reusable = bool(
            bool(self.cfg.probe_preview_reuse_baseline)
            and len(preview_ids) >= int(self.cfg.max_gen_len)
            and str(preview_text or "").strip() != ""
        )
        debug = {
            "probe_source": "baseline_yesno_offline_fullseq",
            "probe_anchor": str(anchor_phrase or ""),
            "probe_anchor_token_idx": int(anchor_idx) if anchor_idx is not None else -1,
            "baseline_preview_text": str(preview_text or ""),
            "baseline_preview_reusable": preview_reusable,
            "baseline_preview_found_anchor": bool(anchor_idx is not None),
            "baseline_preview_fallback": False,
        }
        if anchor_phrase is None or not cont_ids:
            debug["baseline_preview_fallback"] = True
            return None, None, {}, debug
        if anchor_idx is None:
            anchor_idx = 0
            debug["probe_anchor_token_idx"] = 0

        cont_t = torch.tensor([cont_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([input_ids, cont_t], dim=1)

        with torch.inference_mode():
            pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = self._prepare_multimodal_expanded_sequence(
                full_ids=full_ids,
                image_tensor=prepared["image_tensor"],
                image_size=prepared["image_size"],
            )

            labels_exp = labels_e[0]
            cont_label_pos = self._find_cont_label_positions(labels_exp, cont_ids)
            if cont_label_pos is None or int(cont_label_pos.numel()) != int(len(cont_ids)):
                debug["baseline_preview_fallback"] = True
                return None, None, {}, debug
            dec_pos = cont_label_pos - 1
            if int(dec_pos.min().item()) < 0:
                debug["baseline_preview_fallback"] = True
                return None, None, {}, debug

            vision_pos = torch.where(labels_exp == int(self.IGNORE_INDEX))[0]
            text_pos = torch.where(labels_exp != int(self.IGNORE_INDEX))[0]
            out = self.model(
                inputs_embeds=mm_embeds_e,
                attention_mask=attn_mask_e,
                position_ids=pos_ids_e,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )

        decision_pos = int(dec_pos[int(anchor_idx)].item())
        debug["probe_decision_pos"] = int(decision_pos)
        frg, gmi, attn_stats = self._compute_probe_metrics_at_decision_row(
            attentions=out.attentions,
            decision_pos=decision_pos,
            vision_positions=vision_pos,
            text_positions=text_pos,
            guidance=guidance,
            g_top5_mass=g_top5_mass,
        )
        return frg, gmi, attn_stats, debug

    def probe(self, sample: Any) -> ProbeState:
        prepared = self._prepare_sample(sample)
        input_ids = prepared["input_ids"]
        image_tensor = prepared["image_tensor"]

        with torch.inference_mode():
            prefill = self._run_prompt_prefill(input_ids=input_ids, image_tensor=image_tensor)
            logits = prefill.logits
            vis_logits = F.softmax(logits[0, 35:611, :], dim=-1).float()
            vl_guidance, guidance_mode = self._compute_vl_guidance(vis_logits, prepared["object_list"])

        image_start, image_end = image_span_from_prompt_input_ids(
            input_ids=input_ids,
            image_token_index=int(self.IMAGE_TOKEN_INDEX),
            image_token_count=int(vis_logits.size(0)),
        )
        g_top5_mass = topk_mass(vl_guidance.to(torch.float32), k=5)

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
        probe_mode = str(self.cfg.probe_position_mode or "prompt_last").strip().lower()
        if probe_mode == "baseline_yesno_preview":
            frg, gmi, attn_stats, probe_debug = self._probe_baseline_yesno_preview(
                prepared=prepared,
                input_ids=input_ids,
                prefill=prefill,
                image_start=image_start,
                image_end=image_end,
                guidance=vl_guidance,
                g_top5_mass=float(g_top5_mass),
                common_gen_kwargs=common_gen_kwargs,
            )
            if frg is None or gmi is None:
                if bool(self.cfg.probe_preview_fallback_to_prompt_last):
                    frg, gmi, attn_stats, prompt_debug = self._probe_prompt_last(
                        input_ids=input_ids,
                        prefill=prefill,
                        image_start=image_start,
                        image_end=image_end,
                        guidance=vl_guidance,
                        g_top5_mass=float(g_top5_mass),
                    )
                    probe_debug["probe_source"] = "baseline_yesno_preview->prompt_last_fallback"
                    probe_debug["baseline_preview_fallback"] = True
                else:
                    raise RuntimeError("baseline_yesno_preview probe could not find yes/no anchor and fallback is disabled.")
        elif probe_mode == "baseline_yesno_offline_fullseq":
            frg, gmi, attn_stats, probe_debug = self._probe_baseline_yesno_offline_fullseq(
                prepared=prepared,
                input_ids=input_ids,
                guidance=vl_guidance,
                g_top5_mass=float(g_top5_mass),
                common_gen_kwargs=common_gen_kwargs,
            )
            if frg is None or gmi is None:
                if bool(self.cfg.probe_preview_fallback_to_prompt_last):
                    frg, gmi, attn_stats, prompt_debug = self._probe_prompt_last(
                        input_ids=input_ids,
                        prefill=prefill,
                        image_start=image_start,
                        image_end=image_end,
                        guidance=vl_guidance,
                        g_top5_mass=float(g_top5_mass),
                    )
                    probe_debug["probe_source"] = "baseline_yesno_offline_fullseq->prompt_last_fallback"
                    probe_debug["baseline_preview_fallback"] = True
                else:
                    raise RuntimeError("baseline_yesno_offline_fullseq probe failed and fallback is disabled.")
        else:
            frg, gmi, attn_stats, probe_debug = self._probe_prompt_last(
                input_ids=input_ids,
                prefill=prefill,
                image_start=image_start,
                image_end=image_end,
                guidance=vl_guidance,
                g_top5_mass=float(g_top5_mass),
            )
        extras = {
            "prepared": prepared,
            "common_gen_kwargs": common_gen_kwargs,
            "probe_feature_mode": str(self.cfg.probe_feature_mode),
            "probe_position_mode": str(self.cfg.probe_position_mode),
            "guidance_mode": guidance_mode,
            "g_top5_mass": float(g_top5_mass),
            "image_start": int(image_start),
            "image_end": int(image_end),
            "attn_stats": attn_stats,
        }
        extras.update(probe_debug)
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

        output_text = self._decode_generated_text(output_ids, prepared["stop_str"])
        return self._build_prediction_row(sample=sample, prepared=prepared, output_text=output_text, use_add=use_add)

    def predict_base(self, sample: Any, probe_state: ProbeState | None = None) -> Any:
        if probe_state is None:
            probe_state = self.probe(sample)
        extras = probe_state.extras if probe_state is not None else {}
        prepared = extras.get("prepared", {})
        preview_text = str(extras.get("baseline_preview_text", "")).strip()
        if bool(extras.get("baseline_preview_reusable", False)) and preview_text != "" and isinstance(prepared, dict):
            return self._build_prediction_row(sample=sample, prepared=prepared, output_text=preview_text, use_add=False)
        return self._generate(sample=sample, probe_state=probe_state, use_add=False)

    def predict_method(self, sample: Any, probe_state: ProbeState | None = None) -> Any:
        if probe_state is None:
            probe_state = self.probe(sample)
        return self._generate(sample=sample, probe_state=probe_state, use_add=True)
