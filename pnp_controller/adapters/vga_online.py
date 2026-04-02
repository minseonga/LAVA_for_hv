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
    compute_attention_head_probes_at_row,
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
    probe_branch_source: str = "preview"
    probe_force_manual_fullseq: bool = False
    probe_preview_max_new_tokens: int = 3
    probe_preview_reuse_baseline: bool = True
    probe_preview_fallback_to_prompt_last: bool = True
    use_gmi: bool = True  # Set False to skip GMI (use FRG-only veto)
    seed: int = 42
    prefer_local_llava: bool = False
    proxy_trace_enabled: bool = False
    proxy_trace_late_start: int = 16
    proxy_trace_late_end: int = 24
    proxy_trace_last_k: int = 8
    proxy_trace_margin_low: float = 1.0


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
        sys.path = [p for p in sys.path if os.path.abspath(str(p)) != self.vga_root]
        if bool(self.cfg.prefer_local_llava):
            sys.path.append(self.vga_root)
        else:
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

    def _safe_token_to_str(self, token_id: int) -> str:
        try:
            return str(self.tokenizer.convert_ids_to_tokens(int(token_id)))
        except Exception:
            return ""

    def _token_ids_to_token_strs(self, token_ids: List[int]) -> List[str]:
        return [self._safe_token_to_str(int(tid)) for tid in token_ids]

    def _select_content_token_indices(self, token_ids: List[int]) -> List[int]:
        keep: List[int] = []
        for idx, token_id in enumerate(token_ids):
            try:
                piece = self.tokenizer.decode([int(token_id)], skip_special_tokens=True)
            except Exception:
                piece = ""
            if any(ch.isalnum() for ch in str(piece or "").strip()):
                keep.append(int(idx))
        return keep if keep else list(range(int(len(token_ids))))

    def _mean_or_zero(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(float(v) for v in values) / float(len(values)))

    def _std_or_zero(self, values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mu = self._mean_or_zero(values)
        var = sum((float(v) - mu) ** 2 for v in values) / float(len(values))
        return float(var ** 0.5)

    def _min_or_zero(self, values: List[float]) -> float:
        return float(min(values)) if values else 0.0

    def _max_or_zero(self, values: List[float]) -> float:
        return float(max(values)) if values else 0.0

    def _sign_flip_count(self, values: List[float]) -> int:
        if len(values) <= 1:
            return 0
        flips = 0
        prev = 1 if float(values[0]) > 0 else (-1 if float(values[0]) < 0 else 0)
        for value in values[1:]:
            cur = 1 if float(value) > 0 else (-1 if float(value) < 0 else 0)
            if prev != 0 and cur != 0 and cur != prev:
                flips += 1
            if cur != 0:
                prev = cur
        return int(flips)

    def _ratio_below(self, values: List[float], threshold: float) -> float:
        if not values:
            return 0.0
        n = sum(1 for value in values if float(value) <= float(threshold))
        return float(n / float(len(values)))

    def _last_k_slice(self, values: List[float]) -> List[float]:
        if not values:
            return []
        k = max(1, int(self.cfg.proxy_trace_last_k))
        return [float(v) for v in values[-k:]]

    def _summarize_scalar_series(self, values: List[float], prefix: str) -> Dict[str, Any]:
        seq = [float(v) for v in values]
        tail = self._last_k_slice(seq)
        return {
            f"{prefix}_mean": self._mean_or_zero(seq),
            f"{prefix}_std": self._std_or_zero(seq),
            f"{prefix}_min": self._min_or_zero(seq),
            f"{prefix}_max": self._max_or_zero(seq),
            f"{prefix}_lastk_mean": self._mean_or_zero(tail),
            f"{prefix}_lastk_std": self._std_or_zero(tail),
        }

    def _build_decode_time_proxy_row(
        self,
        *,
        sample: Dict[str, Any],
        prepared: Dict[str, Any],
        output_ids: torch.Tensor,
        output_scores: List[torch.Tensor],
        proxy_trace_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        gen_ids = [int(x) for x in output_ids[0, 1:].tolist()]
        n_steps = int(min(len(gen_ids), len(output_scores)))
        gen_ids = gen_ids[:n_steps]
        content_idx = self._select_content_token_indices(gen_ids)

        lp_all: List[float] = []
        margin_all: List[float] = []
        ent_all: List[float] = []
        for step_idx in range(int(n_steps)):
            logits = output_scores[int(step_idx)][0].detach().to(torch.float32)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            target_id = int(gen_ids[int(step_idx)])
            target_lp = float(log_probs[target_id].item())
            top2_vals, top2_idx = torch.topk(logits, k=2, dim=-1)
            top1_id = int(top2_idx[0].item())
            top1_val = float(top2_vals[0].item())
            top2_val = float(top2_vals[1].item())
            target_logit = float(logits[target_id].item())
            best_other = top2_val if top1_id == target_id else top1_val
            lp_all.append(target_lp)
            margin_all.append(float(target_logit - best_other))
            ent_all.append(float((-(probs * log_probs).sum()).item()))

        proxy_rows = sorted(proxy_trace_rows, key=lambda row: int(row.get("step_idx", -1)))
        faithful_all = [float(row.get("proxy_faithful_mean", 0.0)) for row in proxy_rows[:n_steps]]
        harmful_all = [float(row.get("proxy_harmful_mean", 0.0)) for row in proxy_rows[:n_steps]]
        gap_all = [float(row.get("proxy_gap_mean", 0.0)) for row in proxy_rows[:n_steps]]

        def pick(values: List[float], idxs: List[int]) -> List[float]:
            if not values:
                return []
            return [float(values[i]) for i in idxs if 0 <= int(i) < len(values)]

        lp_content = pick(lp_all, content_idx)
        margin_content = pick(margin_all, content_idx)
        ent_content = pick(ent_all, content_idx)
        faithful_content = pick(faithful_all, content_idx)
        harmful_content = pick(harmful_all, content_idx)
        gap_content = pick(gap_all, content_idx)

        row: Dict[str, Any] = {
            "id": prepared["sample_id"],
            "image": prepared["image_file"],
            "question": prepared["question"],
            "output": self._decode_generated_text(output_ids, prepared["stop_str"]),
            "n_generated_tokens": int(n_steps),
            "n_content_tokens": int(len(content_idx)),
            "proxy_content_fraction": float(len(content_idx) / max(1, n_steps)),
            "proxy_gap_sign_flip_count_all": int(self._sign_flip_count(gap_all)),
            "proxy_gap_sign_flip_count_content": int(self._sign_flip_count(gap_content)),
            "proxy_low_gap_ratio_all": float(self._ratio_below(gap_all, 0.0)),
            "proxy_low_gap_ratio_content": float(self._ratio_below(gap_content, 0.0)),
            "proxy_low_margin_ratio_all": float(self._ratio_below(margin_all, float(self.cfg.proxy_trace_margin_low))),
            "proxy_low_margin_ratio_content": float(self._ratio_below(margin_content, float(self.cfg.proxy_trace_margin_low))),
        }
        row.update(self._summarize_scalar_series(lp_all, "proxy_lp_all"))
        row.update(self._summarize_scalar_series(lp_content, "proxy_lp_content"))
        row.update(self._summarize_scalar_series(margin_all, "proxy_margin_all"))
        row.update(self._summarize_scalar_series(margin_content, "proxy_margin_content"))
        row.update(self._summarize_scalar_series(ent_all, "proxy_entropy_all"))
        row.update(self._summarize_scalar_series(ent_content, "proxy_entropy_content"))
        row.update(self._summarize_scalar_series(faithful_all, "proxy_faithful_all"))
        row.update(self._summarize_scalar_series(faithful_content, "proxy_faithful_content"))
        row.update(self._summarize_scalar_series(harmful_all, "proxy_harmful_all"))
        row.update(self._summarize_scalar_series(harmful_content, "proxy_harmful_content"))
        row.update(self._summarize_scalar_series(gap_all, "proxy_gap_all"))
        row.update(self._summarize_scalar_series(gap_content, "proxy_gap_content"))
        return row

    def _build_expanded_window_rows(
        self,
        labels_expanded: torch.Tensor,
        cont_label_pos: torch.Tensor,
        dec_pos: torch.Tensor,
        window_radius: int = 8,
    ) -> List[Dict[str, Any]]:
        labels_cpu = labels_expanded.detach().cpu().tolist()
        cont_pos_set = {int(x) for x in cont_label_pos.detach().cpu().tolist()}
        dec_pos_set = {int(x) for x in dec_pos.detach().cpu().tolist()}
        if not labels_cpu:
            return []

        lo = max(0, min(dec_pos_set | cont_pos_set) - int(window_radius))
        hi = min(len(labels_cpu) - 1, max(dec_pos_set | cont_pos_set) + int(window_radius))
        rows: List[Dict[str, Any]] = []
        for pos in range(int(lo), int(hi) + 1):
            label_id = int(labels_cpu[pos])
            is_vision = label_id == int(self.IGNORE_INDEX)
            rows.append(
                {
                    "pos": int(pos),
                    "kind": "vision" if is_vision else "text",
                    "label_id": None if is_vision else int(label_id),
                    "token_str": "<image>" if is_vision else self._safe_token_to_str(int(label_id)),
                    "is_cont": bool(pos in cont_pos_set),
                    "is_decision": bool(pos in dec_pos_set),
                }
            )
        return rows

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

    def _build_base_gen_kwargs(
        self,
        prefill: Any,
        image_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        return dict(
            images=image_tensor.unsqueeze(0),
            past_key_values=prefill.past_key_values,
            do_sample=bool(self.cfg.sampling),
            sampling=bool(self.cfg.sampling),
            num_beams=int(self.cfg.num_beams),
            max_new_tokens=int(self.cfg.max_gen_len),
            use_cache=True,
        )

    def _build_method_gen_kwargs(
        self,
        base_gen_kwargs: Dict[str, Any],
        vl_guidance: torch.Tensor,
        vis_logits: torch.Tensor,
    ) -> Dict[str, Any]:
        method_gen_kwargs = dict(base_gen_kwargs)
        method_gen_kwargs.update(
            vl_guidance=vl_guidance,
            vis_logits=vis_logits,
            cd_alpha=float(self.cfg.cd_alpha),
            add_layer=list(range(int(self.cfg.start_layer), int(self.cfg.end_layer) + 1)),
            attn_coef=float(self.cfg.attn_coef),
            head_balancing=str(self.cfg.head_balancing),
            attn_norm=bool(self.cfg.attn_norm),
        )
        return method_gen_kwargs

    def _prepare_runtime_context(self, sample: Dict[str, Any]) -> Dict[str, Any]:
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
        base_gen_kwargs = self._build_base_gen_kwargs(
            prefill=prefill,
            image_tensor=image_tensor,
        )
        method_gen_kwargs = self._build_method_gen_kwargs(
            base_gen_kwargs=base_gen_kwargs,
            vl_guidance=vl_guidance,
            vis_logits=vis_logits,
        )
        return {
            "prepared": prepared,
            "input_ids": input_ids,
            "image_tensor": image_tensor,
            "prefill": prefill,
            "vis_logits": vis_logits,
            "vl_guidance": vl_guidance,
            "guidance_mode": guidance_mode,
            "image_start": int(image_start),
            "image_end": int(image_end),
            "g_top5_mass": float(g_top5_mass),
            "base_gen_kwargs": base_gen_kwargs,
            "method_gen_kwargs": method_gen_kwargs,
        }

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

    def _run_full_attention_probe(
        self,
        mm_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: Optional[torch.Tensor],
    ) -> Any:
        backbone = self.model.get_model() if hasattr(self.model, "get_model") else getattr(self.model, "model", None)
        if backbone is None:
            raise RuntimeError("Could not resolve language-model backbone for full attention probe.")
        kwargs = dict(
            inputs_embeds=mm_embeds,
            attention_mask=attn_mask,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        if pos_ids is not None:
            try:
                return backbone(position_ids=pos_ids, **kwargs)
            except TypeError as exc:
                if "position_ids" not in str(exc):
                    raise
        return backbone(**kwargs)

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
            "probe_branch_source": "prompt_last",
            "probe_anchor": "",
            "probe_anchor_token_idx": -1,
            "baseline_preview_text": "",
            "probe_branch_text": "",
            "baseline_preview_reusable": False,
            "baseline_preview_found_anchor": False,
            "baseline_preview_fallback": False,
        }
        return frg, gmi, attn_stats, debug

    def _run_baseline_preview(
        self,
        prepared: Dict[str, Any],
        base_gen_kwargs: Dict[str, Any],
    ) -> Tuple[List[int], str]:
        preview_kwargs = dict(base_gen_kwargs)
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

    def _build_yesno_branch_debug(
        self,
        *,
        branch_text: str,
        anchor_phrase: Optional[str],
        anchor_idx: Optional[int],
        branch_source: str,
        preview_reusable: bool,
    ) -> Dict[str, Any]:
        return {
            "probe_branch_source": str(branch_source),
            "probe_anchor": str(anchor_phrase or ""),
            "probe_anchor_token_idx": int(anchor_idx) if anchor_idx is not None else -1,
            "baseline_preview_text": str(branch_text or ""),
            "probe_branch_text": str(branch_text or ""),
            "baseline_preview_reusable": bool(preview_reusable),
            "baseline_preview_found_anchor": bool(anchor_idx is not None),
            "baseline_preview_fallback": False,
        }

    def _probe_baseline_yesno_preview(
        self,
        prepared: Dict[str, Any],
        input_ids: torch.Tensor,
        prefill: Any,
        image_start: int,
        image_end: int,
        guidance: torch.Tensor,
        g_top5_mass: float,
        base_gen_kwargs: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        first_step = self._run_cached_probe_step(input_ids[:, -1:], prefill.past_key_values)
        preview_ids, preview_text = self._run_baseline_preview(prepared, base_gen_kwargs)
        anchor_phrase = self._normalize_yes_no_anchor(preview_text)
        anchor_idx = self._locate_phrase_token_start(preview_ids, anchor_phrase) if anchor_phrase is not None else None
        debug = {
            "probe_source": "baseline_yesno_preview",
            "probe_branch_source": "preview",
            "probe_anchor": str(anchor_phrase or ""),
            "probe_anchor_token_idx": int(anchor_idx) if anchor_idx is not None else -1,
            "baseline_preview_text": str(preview_text or ""),
            "probe_branch_text": str(preview_text or ""),
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
        base_gen_kwargs: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        preview_ids, preview_text = self._run_baseline_preview(prepared, base_gen_kwargs)
        return self._probe_baseline_yesno_offline_fullseq_from_text(
            prepared=prepared,
            input_ids=input_ids,
            guidance=guidance,
            g_top5_mass=g_top5_mass,
            branch_text=preview_text,
            branch_source="preview",
            preview_reusable=bool(
                bool(self.cfg.probe_preview_reuse_baseline)
                and len(preview_ids) >= int(self.cfg.max_gen_len)
                and str(preview_text or "").strip() != ""
            ),
        )

    def _probe_baseline_yesno_offline_fullseq_from_text(
        self,
        prepared: Dict[str, Any],
        input_ids: torch.Tensor,
        guidance: torch.Tensor,
        g_top5_mass: float,
        branch_text: str,
        branch_source: str,
        preview_reusable: bool,
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any], Dict[str, Any]]:
        anchor_phrase = self._normalize_yes_no_anchor(branch_text)
        cont_ids = self._choose_cont_ids(branch_text)
        anchor_idx = self._locate_phrase_token_start(cont_ids, anchor_phrase) if anchor_phrase is not None else None
        debug = self._build_yesno_branch_debug(
            branch_text=branch_text,
            anchor_phrase=anchor_phrase,
            anchor_idx=anchor_idx,
            branch_source=branch_source,
            preview_reusable=preview_reusable,
        )
        debug["probe_source"] = "baseline_yesno_offline_fullseq"
        if anchor_phrase is None or not cont_ids:
            debug["baseline_preview_fallback"] = True
            return None, None, {}, debug
        if anchor_idx is None:
            anchor_idx = 0
            debug["probe_anchor_token_idx"] = 0

        cont_t = torch.tensor([cont_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([input_ids, cont_t], dim=1)

        with torch.inference_mode():
            probe_helper = getattr(self.model, "offline_style_full_attention_probe", None)
            used_helper = False
            helper_enabled = bool(
                callable(probe_helper) and not bool(getattr(self.cfg, "probe_force_manual_fullseq", False))
            )
            if helper_enabled:
                try:
                    out, pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = probe_helper(
                        full_ids=full_ids,
                        images=prepared["image_tensor"].unsqueeze(0),
                        attention_mask=torch.ones_like(full_ids, dtype=torch.long, device=self.device),
                        labels=full_ids,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    debug["probe_impl"] = "offline_style_helper"
                    used_helper = True
                except (RuntimeError, TypeError, ValueError) as exc:
                    debug["probe_impl"] = "offline_style_helper_fallback"
                    debug["probe_impl_error"] = str(exc)
            elif callable(probe_helper):
                debug["probe_impl"] = "adapter_manual_fullseq_forced"
            else:
                debug["probe_impl"] = "adapter_manual_fullseq_no_helper"

            if not used_helper:
                pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = self._prepare_multimodal_expanded_sequence(
                    full_ids=full_ids,
                    image_tensor=prepared["image_tensor"],
                    image_size=prepared["image_size"],
                )
                debug["probe_impl"] = str(debug.get("probe_impl", "adapter_manual_fullseq"))

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
            if not used_helper:
                out = self._run_full_attention_probe(
                    mm_embeds=mm_embeds_e,
                    attn_mask=attn_mask_e,
                    pos_ids=pos_ids_e,
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

    def debug_probe_baseline_yesno_offline_fullseq(
        self,
        sample: Dict[str, Any],
        head_layer_start: Optional[int] = None,
        head_layer_end: Optional[int] = None,
        branch_text: Optional[str] = None,
        branch_source: str = "preview",
    ) -> Dict[str, Any]:
        ctx = self._prepare_runtime_context(sample)
        prepared = ctx["prepared"]
        input_ids = ctx["input_ids"]
        image_tensor = ctx["image_tensor"]
        prefill = ctx["prefill"]
        vis_logits = ctx["vis_logits"]
        vl_guidance = ctx["vl_guidance"]
        guidance_mode = ctx["guidance_mode"]
        g_top5_mass = float(ctx["g_top5_mass"])
        base_gen_kwargs = ctx["base_gen_kwargs"]

        if branch_text is None:
            preview_ids, preview_text = self._run_baseline_preview(prepared, base_gen_kwargs)
            preview_reusable = bool(
                bool(self.cfg.probe_preview_reuse_baseline)
                and len(preview_ids) >= int(self.cfg.max_gen_len)
                and str(preview_text or "").strip() != ""
            )
            probe_branch_source = "preview"
        else:
            preview_ids = []
            preview_text = str(branch_text)
            preview_reusable = False
            probe_branch_source = str(branch_source or "baseline_output")
        anchor_phrase = self._normalize_yes_no_anchor(preview_text)
        cont_ids = self._choose_cont_ids(preview_text)
        anchor_idx = self._locate_phrase_token_start(cont_ids, anchor_phrase) if anchor_phrase is not None else None
        cont_token_strs = self._token_ids_to_token_strs(cont_ids)
        debug = {
            "probe_source": "baseline_yesno_offline_fullseq",
            "probe_branch_source": probe_branch_source,
            "probe_anchor": str(anchor_phrase or ""),
            "probe_anchor_token_idx": int(anchor_idx) if anchor_idx is not None else -1,
            "baseline_preview_text": str(preview_text or ""),
            "probe_branch_text": str(preview_text or ""),
            "baseline_preview_reusable": preview_reusable,
            "baseline_preview_found_anchor": bool(anchor_idx is not None),
            "baseline_preview_fallback": False,
            "probe_cont_ids": [int(x) for x in cont_ids],
            "probe_cont_token_strs": cont_token_strs,
            "prompt_input_ids_len": int(input_ids.size(1)),
            "probe_cont_len": int(len(cont_ids)),
            "probe_branch_text_len": int(len(str(preview_text or ""))),
        }
        if anchor_phrase is None or not cont_ids:
            raise RuntimeError("Could not determine yes/no anchor phrase for offline fullseq probe.")
        if anchor_idx is None:
            anchor_idx = 0
            debug["probe_anchor_token_idx"] = 0
        if 0 <= int(anchor_idx) < int(len(cont_ids)):
            debug["probe_anchor_token_id"] = int(cont_ids[int(anchor_idx)])
            debug["probe_anchor_token_str"] = self._safe_token_to_str(int(cont_ids[int(anchor_idx)]))
        else:
            debug["probe_anchor_token_id"] = -1
            debug["probe_anchor_token_str"] = ""

        cont_t = torch.tensor([cont_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([input_ids, cont_t], dim=1)
        debug["probe_full_ids_len"] = int(full_ids.size(1))

        with torch.inference_mode():
            probe_helper = getattr(self.model, "offline_style_full_attention_probe", None)
            used_helper = False
            helper_enabled = bool(
                callable(probe_helper) and not bool(getattr(self.cfg, "probe_force_manual_fullseq", False))
            )
            if helper_enabled:
                try:
                    out, pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = probe_helper(
                        full_ids=full_ids,
                        images=prepared["image_tensor"].unsqueeze(0),
                        attention_mask=torch.ones_like(full_ids, dtype=torch.long, device=self.device),
                        labels=full_ids,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    debug["probe_impl"] = "offline_style_helper"
                    used_helper = True
                    debug["mm_embeds_shape"] = [int(x) for x in mm_embeds_e.shape]
                    debug["attn_mask_shape"] = [int(x) for x in attn_mask_e.shape]
                    debug["labels_shape"] = [int(x) for x in labels_e.shape]
                    debug["pos_ids_shape"] = [int(x) for x in pos_ids_e.shape] if pos_ids_e is not None else []
                except (RuntimeError, TypeError, ValueError) as exc:
                    debug["probe_impl"] = "offline_style_helper_fallback"
                    debug["probe_impl_error"] = str(exc)
            elif callable(probe_helper):
                debug["probe_impl"] = "adapter_manual_fullseq_forced"
            else:
                debug["probe_impl"] = "adapter_manual_fullseq_no_helper"

            if not used_helper:
                pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = self._prepare_multimodal_expanded_sequence(
                    full_ids=full_ids,
                    image_tensor=prepared["image_tensor"],
                    image_size=prepared["image_size"],
                )
                debug["probe_impl"] = str(debug.get("probe_impl", "adapter_manual_fullseq"))
                debug["mm_embeds_shape"] = [int(x) for x in mm_embeds_e.shape]
                debug["attn_mask_shape"] = [int(x) for x in attn_mask_e.shape]
                debug["labels_shape"] = [int(x) for x in labels_e.shape]
                debug["pos_ids_shape"] = [int(x) for x in pos_ids_e.shape] if pos_ids_e is not None else []

            labels_exp = labels_e[0]
            cont_label_pos = self._find_cont_label_positions(labels_exp, cont_ids)
            if cont_label_pos is None or int(cont_label_pos.numel()) != int(len(cont_ids)):
                raise RuntimeError("Could not align continuation labels in offline fullseq probe.")
            dec_pos = cont_label_pos - 1
            if int(dec_pos.min().item()) < 0:
                raise RuntimeError("Negative decision position encountered in offline fullseq probe.")
            debug["probe_cont_label_positions"] = [int(x) for x in cont_label_pos.detach().cpu().tolist()]
            debug["probe_decision_positions"] = [int(x) for x in dec_pos.detach().cpu().tolist()]

            vision_pos = torch.where(labels_exp == int(self.IGNORE_INDEX))[0]
            text_pos = torch.where(labels_exp != int(self.IGNORE_INDEX))[0]
            valid_label_ids = [int(x) for x in labels_exp[text_pos].detach().cpu().tolist()]
            debug["vision_positions"] = [int(x) for x in vision_pos.detach().cpu().tolist()]
            debug["text_positions"] = [int(x) for x in text_pos.detach().cpu().tolist()]
            debug["expanded_valid_label_ids"] = valid_label_ids
            debug["expanded_valid_label_token_strs"] = self._token_ids_to_token_strs(valid_label_ids)
            debug["n_vision_positions"] = int(vision_pos.numel())
            debug["n_text_positions"] = int(text_pos.numel())
            debug["expanded_seq_len"] = int(labels_exp.numel())
            debug["expanded_window_rows"] = self._build_expanded_window_rows(
                labels_expanded=labels_exp,
                cont_label_pos=cont_label_pos,
                dec_pos=dec_pos,
            )
            if not used_helper:
                out = self._run_full_attention_probe(
                    mm_embeds=mm_embeds_e,
                    attn_mask=attn_mask_e,
                    pos_ids=pos_ids_e,
                )

        decision_pos = int(dec_pos[int(anchor_idx)].item())
        debug["probe_decision_pos"] = int(decision_pos)
        frg, gmi, attn_stats = self._compute_probe_metrics_at_decision_row(
            attentions=out.attentions,
            decision_pos=decision_pos,
            vision_positions=vision_pos,
            text_positions=text_pos,
            guidance=vl_guidance,
            g_top5_mass=float(g_top5_mass),
        )

        layer_l0 = 0 if head_layer_start is None else max(0, int(head_layer_start))
        layer_l1 = int(len(out.attentions) - 1) if head_layer_end is None else min(int(head_layer_end), int(len(out.attentions) - 1))
        yesno_token_idx = int(anchor_idx)
        yesno_token_str = ""
        if 0 <= int(anchor_idx) < int(len(cont_ids)):
            yesno_token_str = self._safe_token_to_str(int(cont_ids[int(anchor_idx)]))

        per_head_rows: List[Dict[str, Any]] = []
        for block_idx in range(int(layer_l0), int(layer_l1) + 1):
            att_l = out.attentions[int(block_idx)]
            probes = compute_attention_head_probes_at_row(
                att_l=att_l,
                decision_pos=int(decision_pos),
                vision_positions=vision_pos,
                text_positions=text_pos,
            )
            if probes is None:
                continue
            n_heads = int(probes["head_attn_vis_sum"].numel())
            for head_idx in range(n_heads):
                per_head_rows.append(
                    {
                        "id": str(prepared["sample_id"]),
                        "image_id": str(prepared["image_file"]),
                        "question": str(prepared["question"]),
                        "object_phrase": ",".join(prepared["object_list"]),
                        "answer_pred": str(anchor_phrase),
                        "pred_text": str(preview_text),
                        "anchor_phrase": str(anchor_phrase),
                        "yesno_token_idx": int(yesno_token_idx),
                        "yesno_token_str": str(yesno_token_str),
                        "block_layer_idx": int(block_idx),
                        "head_idx": int(head_idx),
                        "head_attn_vis_sum": float(probes["head_attn_vis_sum"][head_idx].item()),
                        "head_attn_vis_ratio": float(probes["head_attn_vis_ratio"][head_idx].item()),
                        "head_attn_vis_peak": float(probes["head_attn_vis_peak"][head_idx].item()),
                        "head_attn_vis_entropy": float(probes["head_attn_vis_entropy"][head_idx].item()),
                        "probe_impl": str(debug.get("probe_impl", "")),
                        "probe_impl_error": str(debug.get("probe_impl_error", "")),
                        "probe_source": str(debug.get("probe_source", "")),
                        "probe_decision_pos": int(decision_pos),
                        "baseline_preview_fallback": int(bool(debug.get("baseline_preview_fallback", False))),
                    }
                )

        return {
            "sample_id": str(prepared["sample_id"]),
            "question": str(prepared["question"]),
            "image_id": str(prepared["image_file"]),
            "object_phrase": ",".join(prepared["object_list"]),
            "guidance_mode": str(guidance_mode),
            "g_top5_mass": float(g_top5_mass),
            "frg": float(frg),
            "gmi": float(gmi),
            "attn_stats": attn_stats,
            "debug": debug,
            "per_head_rows": per_head_rows,
        }

    def _summarize_path_metric(self, values: List[float], prefix: str) -> Dict[str, float]:
        if not values:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_last": 0.0,
                f"{prefix}_mean_plus_std": 0.0,
            }
        vals_t = torch.tensor(values, dtype=torch.float32)
        mean_v = float(vals_t.mean().item())
        std_v = float(vals_t.std(unbiased=False).item())
        min_v = float(vals_t.min().item())
        max_v = float(vals_t.max().item())
        last_v = float(vals_t[-1].item())
        return {
            f"{prefix}_mean": mean_v,
            f"{prefix}_std": std_v,
            f"{prefix}_min": min_v,
            f"{prefix}_max": max_v,
            f"{prefix}_last": last_v,
            f"{prefix}_mean_plus_std": float(mean_v + std_v),
        }

    def _resolve_verify_context(
        self,
        sample: Dict[str, Any],
        probe_state: Optional[ProbeState] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, float]:
        extras = probe_state.extras if probe_state is not None else {}
        prepared = extras.get("prepared", None)
        input_ids = extras.get("input_ids", None)
        vl_guidance = extras.get("vl_guidance", None)
        g_top5_mass = extras.get("g_top5_mass", None)
        if (
            isinstance(prepared, dict)
            and torch.is_tensor(input_ids)
            and torch.is_tensor(vl_guidance)
            and g_top5_mass is not None
        ):
            return prepared, input_ids, vl_guidance, float(g_top5_mass)

        ctx = self._prepare_runtime_context(sample)
        return ctx["prepared"], ctx["input_ids"], ctx["vl_guidance"], float(ctx["g_top5_mass"])

    def verify_candidate_path(
        self,
        sample: Dict[str, Any],
        candidate_text: str,
        probe_state: Optional[ProbeState] = None,
        force_manual_fullseq: Optional[bool] = None,
    ) -> Dict[str, Any]:
        prepared, input_ids, vl_guidance, g_top5_mass = self._resolve_verify_context(
            sample=sample,
            probe_state=probe_state,
        )
        raw_text = str(candidate_text or "").strip()
        debug: Dict[str, Any] = {
            "verify_source": "candidate_path_fullseq",
            "candidate_text": raw_text,
            "verify_impl": "",
            "verify_impl_error": "",
            "force_manual_fullseq": bool(force_manual_fullseq) if force_manual_fullseq is not None else None,
        }
        if raw_text == "":
            return {
                "valid": False,
                "n_tokens": 0,
                "token_rows": [],
                "debug": debug,
            }

        cont_ids = self._choose_cont_ids(raw_text)
        if not cont_ids:
            debug["verify_impl_error"] = "empty_cont_ids"
            return {
                "valid": False,
                "n_tokens": 0,
                "token_rows": [],
                "debug": debug,
            }

        cont_t = torch.tensor([cont_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([input_ids, cont_t], dim=1)
        use_force_manual = bool(self.cfg.probe_force_manual_fullseq) if force_manual_fullseq is None else bool(force_manual_fullseq)

        with torch.inference_mode():
            probe_helper = getattr(self.model, "offline_style_full_attention_probe", None)
            used_helper = False
            helper_enabled = bool(callable(probe_helper) and not use_force_manual)
            if helper_enabled:
                try:
                    out, pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = probe_helper(
                        full_ids=full_ids,
                        images=prepared["image_tensor"].unsqueeze(0),
                        attention_mask=torch.ones_like(full_ids, dtype=torch.long, device=self.device),
                        labels=full_ids,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    debug["verify_impl"] = "offline_style_helper"
                    used_helper = True
                except (RuntimeError, TypeError, ValueError) as exc:
                    debug["verify_impl"] = "offline_style_helper_fallback"
                    debug["verify_impl_error"] = str(exc)
            elif callable(probe_helper):
                debug["verify_impl"] = "candidate_manual_fullseq_forced"
            else:
                debug["verify_impl"] = "candidate_manual_fullseq_no_helper"

            if not used_helper:
                pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = self._prepare_multimodal_expanded_sequence(
                    full_ids=full_ids,
                    image_tensor=prepared["image_tensor"],
                    image_size=prepared["image_size"],
                )

            labels_exp = labels_e[0]
            cont_label_pos = self._find_cont_label_positions(labels_exp, cont_ids)
            if cont_label_pos is None or int(cont_label_pos.numel()) != int(len(cont_ids)):
                debug["verify_impl_error"] = "cont_alignment_failed"
                return {
                    "valid": False,
                    "n_tokens": 0,
                    "token_rows": [],
                    "debug": debug,
                }
            dec_pos = cont_label_pos - 1
            if int(dec_pos.min().item()) < 0:
                debug["verify_impl_error"] = "negative_decision_pos"
                return {
                    "valid": False,
                    "n_tokens": 0,
                    "token_rows": [],
                    "debug": debug,
                }

            vision_pos = torch.where(labels_exp == int(self.IGNORE_INDEX))[0]
            text_pos = torch.where(labels_exp != int(self.IGNORE_INDEX))[0]
            if not used_helper:
                out = self._run_full_attention_probe(
                    mm_embeds=mm_embeds_e,
                    attn_mask=attn_mask_e,
                    pos_ids=pos_ids_e,
                )

        token_rows: List[Dict[str, Any]] = []
        frg_vals: List[float] = []
        gmi_vals: List[float] = []
        cont_pos_list = [int(x) for x in cont_label_pos.detach().cpu().tolist()]
        dec_pos_list = [int(x) for x in dec_pos.detach().cpu().tolist()]
        for idx, decision_pos in enumerate(dec_pos_list):
            frg_t, gmi_t, attn_stats_t = self._compute_probe_metrics_at_decision_row(
                attentions=out.attentions,
                decision_pos=int(decision_pos),
                vision_positions=vision_pos,
                text_positions=text_pos,
                guidance=vl_guidance,
                g_top5_mass=float(g_top5_mass),
            )
            frg_vals.append(float(frg_t))
            gmi_vals.append(float(gmi_t))
            token_id = int(cont_ids[idx])
            token_rows.append(
                {
                    "token_idx": int(idx),
                    "token_id": token_id,
                    "token_str": self._safe_token_to_str(token_id),
                    "label_pos": int(cont_pos_list[idx]),
                    "decision_pos": int(decision_pos),
                    "frg": float(frg_t),
                    "gmi": float(gmi_t),
                    "faithful_head_attn_mean": float(attn_stats_t.get("faithful_head_attn_mean", 0.0)),
                    "harmful_head_attn_mean": float(attn_stats_t.get("harmful_head_attn_mean", 0.0)),
                    "global_late_head_attn_mean": float(attn_stats_t.get("global_late_head_attn_mean", 0.0)),
                }
            )

        out_obj: Dict[str, Any] = {
            "valid": True,
            "candidate_text": raw_text,
            "n_tokens": int(len(token_rows)),
            "token_rows": token_rows,
            "debug": debug,
        }
        out_obj.update(self._summarize_path_metric(frg_vals, "path_frg"))
        out_obj.update(self._summarize_path_metric(gmi_vals, "path_gmi"))
        return out_obj

    def probe(self, sample: Any, branch_text: str | None = None) -> ProbeState:
        ctx = self._prepare_runtime_context(sample)
        prepared = ctx["prepared"]
        input_ids = ctx["input_ids"]
        prefill = ctx["prefill"]
        vl_guidance = ctx["vl_guidance"]
        guidance_mode = ctx["guidance_mode"]
        image_start = int(ctx["image_start"])
        image_end = int(ctx["image_end"])
        g_top5_mass = float(ctx["g_top5_mass"])
        base_gen_kwargs = ctx["base_gen_kwargs"]

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
                base_gen_kwargs=base_gen_kwargs,
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
            if branch_text is not None:
                frg, gmi, attn_stats, probe_debug = self._probe_baseline_yesno_offline_fullseq_from_text(
                    prepared=prepared,
                    input_ids=input_ids,
                    guidance=vl_guidance,
                    g_top5_mass=float(g_top5_mass),
                    branch_text=str(branch_text),
                    branch_source=str(self.cfg.probe_branch_source or "baseline_output"),
                    preview_reusable=False,
                )
            else:
                frg, gmi, attn_stats, probe_debug = self._probe_baseline_yesno_offline_fullseq(
                    prepared=prepared,
                    input_ids=input_ids,
                    guidance=vl_guidance,
                    g_top5_mass=float(g_top5_mass),
                    base_gen_kwargs=base_gen_kwargs,
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
            "input_ids": input_ids,
            "vl_guidance": vl_guidance,
            "base_gen_kwargs": ctx["base_gen_kwargs"],
            "method_gen_kwargs": ctx["method_gen_kwargs"],
            "probe_feature_mode": str(self.cfg.probe_feature_mode),
            "probe_position_mode": str(self.cfg.probe_position_mode),
            "probe_branch_source": str(self.cfg.probe_branch_source),
            "probe_force_manual_fullseq": bool(self.cfg.probe_force_manual_fullseq),
            "guidance_mode": guidance_mode,
            "g_top5_mass": float(g_top5_mass),
            "image_start": int(image_start),
            "image_end": int(image_end),
            "attn_stats": attn_stats,
        }
        extras.update(probe_debug)
        return ProbeState(sample_id=prepared["sample_id"], frg=frg, gmi=gmi, extras=extras)

    def _generate_from_prepared(
        self,
        sample: Dict[str, Any],
        prepared: Dict[str, Any],
        gen_kwargs: Dict[str, Any],
        use_add: bool,
        capture_proxy: bool = False,
    ) -> Dict[str, Any]:
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs["use_add"] = bool(use_add)
        gen_kwargs["enable_proxy_trace"] = bool(capture_proxy and self.cfg.proxy_trace_enabled)
        if bool(capture_proxy and self.cfg.proxy_trace_enabled):
            gen_kwargs["proxy_late_start"] = int(self.cfg.proxy_trace_late_start)
            gen_kwargs["proxy_late_end"] = int(self.cfg.proxy_trace_late_end)
            gen_kwargs["ais_headset_json"] = str(self.cfg.headset_json or "")
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
            gen_kwargs["ais_sample_ids"] = [prepared["sample_id"]]
            _ = getattr(self.model, "get_proxy_trace_rows", lambda reset=False: [])(reset=True)
        with torch.inference_mode():
            output_obj = self.model.generate(
                prepared["input_ids"][:, -1:],
                **gen_kwargs,
            )
        output_ids = output_obj.sequences if hasattr(output_obj, "sequences") else output_obj
        output_text = self._decode_generated_text(output_ids, prepared["stop_str"])
        pred_row = self._build_prediction_row(sample=sample, prepared=prepared, output_text=output_text, use_add=use_add)
        if not bool(capture_proxy and self.cfg.proxy_trace_enabled):
            return pred_row

        score_list = list(getattr(output_obj, "scores", []) or [])
        proxy_trace_rows = list(getattr(self.model, "get_proxy_trace_rows", lambda reset=False: [])(reset=True))
        proxy_row = self._build_decode_time_proxy_row(
            sample=sample,
            prepared=prepared,
            output_ids=output_ids,
            output_scores=score_list,
            proxy_trace_rows=proxy_trace_rows,
        )
        return {
            "prediction": pred_row,
            "proxy": proxy_row,
            "proxy_trace_rows": proxy_trace_rows,
        }

    def predict_base_direct(self, sample: Any) -> Any:
        ctx = self._prepare_runtime_context(sample)
        return self._generate_from_prepared(
            sample=sample,
            prepared=ctx["prepared"],
            gen_kwargs=ctx["base_gen_kwargs"],
            use_add=False,
        )

    def _generate(self, sample: Dict[str, Any], probe_state: ProbeState, use_add: bool) -> Dict[str, Any]:
        extras = probe_state.extras
        prepared = extras["prepared"]
        gen_kwargs = extras["method_gen_kwargs"] if bool(use_add) else extras["base_gen_kwargs"]
        return self._generate_from_prepared(
            sample=sample,
            prepared=prepared,
            gen_kwargs=gen_kwargs,
            use_add=use_add,
        )

    def predict_method_with_proxy(self, sample: Any) -> Dict[str, Any]:
        ctx = self._prepare_runtime_context(sample)
        return self._generate_from_prepared(
            sample=sample,
            prepared=ctx["prepared"],
            gen_kwargs=ctx["method_gen_kwargs"],
            use_add=True,
            capture_proxy=True,
        )

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
