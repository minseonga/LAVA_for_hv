from __future__ import annotations

import csv
import traceback
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


def safe_id(value: object) -> str:
    return str(value or "").strip()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value or "").strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def mean_or_zero(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std_or_zero(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 0.0
    mu = mean_or_zero(seq)
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(math.sqrt(max(0.0, var)))


def normalize_head_map(raw: object) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if not isinstance(raw, Mapping):
        return out
    for key, value in raw.items():
        try:
            layer_idx = int(key)
        except Exception:
            continue
        heads: List[int] = []
        if isinstance(value, str):
            parts = [x.strip() for x in value.split(",") if x.strip()]
            for part in parts:
                try:
                    heads.append(int(part))
                except Exception:
                    continue
        elif isinstance(value, Sequence):
            for item in value:
                try:
                    heads.append(int(item))
                except Exception:
                    continue
        if heads:
            out[layer_idx] = sorted(set(heads))
    return out


@dataclass
class Headset:
    faithful_heads_by_layer: Dict[int, List[int]]
    harmful_heads_by_layer: Dict[int, List[int]]
    late_start: int
    late_end: int


@dataclass
class ForwardPack:
    prompt: str
    candidate_text: str
    full_ids: torch.Tensor
    cont_ids: torch.Tensor
    labels_exp: torch.Tensor
    cont_label_positions: torch.Tensor
    decision_positions: torch.Tensor
    vision_positions: torch.Tensor
    text_positions: torch.Tensor
    logits: torch.Tensor
    attentions: Optional[Tuple[torch.Tensor, ...]]


@dataclass
class ScoreRow:
    sample_id: str
    image: str
    question: str
    intervention_text: str
    baseline_text: str
    gt_label: str
    intervention_label: str
    baseline_label: str
    intervention_correct: Optional[int]
    baseline_correct: Optional[int]
    stage_a_score: float
    stage_a_gap_mean: float
    stage_a_faithful_mean: float
    stage_a_harmful_mean: float
    stage_a_faithful_std: float
    stage_b_score: float
    stage_b_delta_mean: float
    stage_b_delta_std: float
    n_cont_tokens: int
    n_content_tokens: int
    score_error: str = ""

    def to_csv_row(self) -> Dict[str, object]:
        return {
            "id": self.sample_id,
            "image": self.image,
            "question": self.question,
            "intervention_text": self.intervention_text,
            "baseline_text": self.baseline_text,
            "gt_label": self.gt_label,
            "intervention_label": self.intervention_label,
            "baseline_label": self.baseline_label,
            "intervention_correct": self.intervention_correct,
            "baseline_correct": self.baseline_correct,
            "stage_a_score": self.stage_a_score,
            "stage_a_gap_mean": self.stage_a_gap_mean,
            "stage_a_faithful_mean": self.stage_a_faithful_mean,
            "stage_a_harmful_mean": self.stage_a_harmful_mean,
            "stage_a_faithful_std": self.stage_a_faithful_std,
            "stage_b_score": self.stage_b_score,
            "stage_b_delta_mean": self.stage_b_delta_mean,
            "stage_b_delta_std": self.stage_b_delta_std,
            "n_cont_tokens": self.n_cont_tokens,
            "n_content_tokens": self.n_content_tokens,
            "score_error": self.score_error,
        }


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(row.keys()) for row in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(part.strip().lower() for part in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def load_label_map(gt_csv: str, id_col: str = "id", label_col: str = "answer") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_csv_rows(gt_csv):
        sample_id = safe_id(row.get(id_col))
        label = safe_id(row.get(label_col)).lower()
        if sample_id and label in {"yes", "no"}:
            out[sample_id] = label
    return out


def load_prediction_text_map(path: str, text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = safe_id(row.get("question_id", row.get("id")))
        if not sample_id:
            continue
        if text_key == "auto":
            text = str(row.get("text", "")).strip()
            if not text:
                text = str(row.get("output", "")).strip()
            if not text:
                text = str(row.get("answer", "")).strip()
        else:
            text = str(row.get(text_key, "")).strip()
        out[sample_id] = text
    return out


def load_question_rows(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    if limit > 0:
        return rows[: int(limit)]
    return rows


def load_headset(headset_json: str, late_start: int = -1, late_end: int = -1) -> Headset:
    obj = read_json(headset_json)
    faithful = normalize_head_map(obj.get("faithful_heads_by_layer", {}))
    harmful = normalize_head_map(obj.get("harmful_heads_by_layer", {}))
    all_layers = sorted(set(faithful.keys()) | set(harmful.keys()))
    if not all_layers:
        raise ValueError(f"No headset layer maps found in {headset_json}")
    if int(late_start) < 0:
        late_start = min(all_layers)
    if int(late_end) < 0:
        late_end = max(all_layers)
    return Headset(
        faithful_heads_by_layer=faithful,
        harmful_heads_by_layer=harmful,
        late_start=int(late_start),
        late_end=int(late_end),
    )


def threshold_candidates(values: Sequence[float], q_grid: Sequence[float]) -> List[float]:
    finite_vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite_vals:
        return [0.0]
    t = torch.tensor(finite_vals, dtype=torch.float32)
    out = {finite_vals[0] - 1e-6, finite_vals[-1] + 1e-6}
    for q in q_grid:
        qq = min(1.0, max(0.0, float(q)))
        out.add(float(torch.quantile(t, qq).item()))
    return sorted(out)


def select_content_indices(tokenizer: Any, cont_ids: torch.Tensor) -> List[int]:
    keep: List[int] = []
    for idx, token_id in enumerate(cont_ids.tolist()):
        try:
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        except Exception:
            piece = ""
        s = str(piece).strip()
        if any(ch.isalnum() for ch in s):
            keep.append(int(idx))
    if keep:
        return keep
    return list(range(int(cont_ids.numel())))


def build_prompt(question: str, conv_mode: str, mm_use_im_start_end: bool) -> str:
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
    from llava.conversation import conv_templates

    qs = str(question or "").strip()
    if mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


class CleanroomLlavaRuntime:
    def __init__(
        self,
        model_path: str,
        model_base: Optional[str],
        conv_mode: str,
        device: str = "cuda",
        load_8bit: bool = False,
        load_4bit: bool = False,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
        image_processor: Optional[Any] = None,
    ) -> None:
        requested_device = str(device or "cuda")
        fallback_used = False
        if tokenizer is None or model is None or image_processor is None:
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init

            disable_torch_init()
            model_name = get_model_name_from_path(model_path)
            # Avoid the expensive "auto -> fail -> cpu fallback -> to(cuda)"
            # path. This builder later passes the original device_map argument
            # into vision_tower.to(device=...), so it must remain a device
            # string here. Passing device="cuda:0" makes the builder use a
            # single-device HF map internally while keeping vision_tower.to()
            # valid.
            direct_device = "cuda:0" if requested_device == "cuda" else requested_device
            try:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path,
                    model_base,
                    model_name,
                    load_8bit=bool(load_8bit),
                    load_4bit=bool(load_4bit),
                    device_map=direct_device,
                    device=direct_device,
                )
            except ValueError as exc:
                if "does not support `device_map='auto'`" not in str(exc):
                    raise
                if bool(load_8bit or load_4bit):
                    raise
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    model_path,
                    model_base,
                    model_name,
                    load_8bit=False,
                    load_4bit=False,
                    device_map="cpu",
                    device="cpu",
                )
                if requested_device != "cpu":
                    model = model.to(device=requested_device, dtype=torch.float16)
                fallback_used = True
        if tokenizer is None or model is None or image_processor is None:
            raise RuntimeError("Failed to initialize cleanroom runtime components.")
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        self.device = torch.device(requested_device if fallback_used else model.device)
        self.image_preprocess_mode = str(os.environ.get("CLEANROOM_IMAGE_PREPROCESS_MODE", "direct")).strip().lower()
        self.teacher_force_forward_mode = str(os.environ.get("CLEANROOM_TF_FORWARD_MODE", "backbone")).strip().lower()
        self.model.eval()
        try:
            vision_tower = self.model.get_vision_tower()
        except Exception:
            vision_tower = None
        if vision_tower is not None and hasattr(vision_tower, "eval"):
            vision_tower.eval()

    def prompt_text(self, question: str) -> str:
        return build_prompt(
            question=question,
            conv_mode=self.conv_mode,
            mm_use_im_start_end=bool(getattr(self.model.config, "mm_use_im_start_end", False)),
        )

    def _process_image(self, image: Image.Image) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        if self.image_preprocess_mode in {"process", "process_images", "legacy"}:
            from llava.mm_utils import process_images

            image_tensor = process_images([image], self.image_processor, self.model.config)
            if image_tensor is None:
                image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        else:
            # Match VGA_origin's caption/VQA eval path first. The legacy
            # process_images path remains available through
            # CLEANROOM_IMAGE_PREPROCESS_MODE=process_images for exact cached
            # discriminative artifact parity checks.
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            if image_tensor is None:
                from llava.mm_utils import process_images

                image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            if not image_tensor:
                raise RuntimeError("Image preprocessing returned an empty image tensor list.")
            image_tensor = image_tensor[0]
        if image_tensor is None:
            raise RuntimeError("Image preprocessing returned None.")
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        return image_tensor.to(self.device, dtype=torch.float16), [image.size]

    def _prepare_multimodal_expanded_sequence(
        self,
        full_ids: torch.Tensor,
        images_tensor: torch.Tensor,
        image_sizes: Sequence[Tuple[int, int]],
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
                images_tensor.unsqueeze(0),
                image_sizes,
            )
        except TypeError as new_sig_exc:
            try:
                packed = fn(
                    full_ids,
                    base_attn,
                    None,
                    full_ids,
                    images_tensor.unsqueeze(0),
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

    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def make_blur_control(self, image: Image.Image, blur_radius: float) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    def teacher_force_candidate(
        self,
        image: Image.Image,
        question: str,
        candidate_text: str,
        output_attentions: bool,
    ) -> ForwardPack:
        from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        prompt = self.prompt_text(question)
        prompt_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        cont_ids = self.tokenizer(
            str(candidate_text or ""),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].to(self.device)
        if int(cont_ids.numel()) <= 0:
            raise ValueError("Candidate text tokenization is empty.")

        full_ids = torch.cat([prompt_ids[0], cont_ids], dim=0).unsqueeze(0)
        images_tensor, image_sizes = self._process_image(image)

        with torch.no_grad():
            pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = self._prepare_multimodal_expanded_sequence(
                full_ids=full_ids,
                images_tensor=images_tensor,
                image_sizes=image_sizes,
            )

            # Call the language-model backbone directly. VGA_origin's
            # LlavaLlamaForCausalLM.forward() always re-enters
            # prepare_inputs_labels_for_multimodal(..., images), which breaks
            # teacher-forced replay with pre-expanded multimodal embeddings.
            backbone = self.model.get_model() if hasattr(self.model, "get_model") else getattr(self.model, "model", None)
            if backbone is None:
                raise RuntimeError("Could not resolve language-model backbone for teacher-forced replay.")

            forward_kwargs = {
                "inputs_embeds": mm_embeds_e,
                "attention_mask": attn_mask_e,
                "use_cache": False,
                "output_attentions": bool(output_attentions),
                "output_hidden_states": False,
                "return_dict": True,
            }
            if pos_ids_e is not None:
                forward_kwargs["position_ids"] = pos_ids_e
            if self.teacher_force_forward_mode in {"model", "full", "legacy"}:
                try:
                    outputs = self.model(**forward_kwargs)
                except TypeError as exc:
                    if "position_ids" not in str(exc):
                        raise
                    forward_kwargs.pop("position_ids", None)
                    outputs = self.model(**forward_kwargs)
                logits = outputs.logits
            else:
                try:
                    outputs = backbone(**forward_kwargs)
                except TypeError as exc:
                    if "position_ids" not in str(exc):
                        raise
                    forward_kwargs.pop("position_ids", None)
                    outputs = backbone(**forward_kwargs)
                logits = self.model.lm_head(outputs[0])

        labels_exp = labels_e[0]
        text_positions = torch.where(labels_exp != int(IGNORE_INDEX))[0]
        if int(text_positions.numel()) < int(cont_ids.numel()):
            raise RuntimeError("Expanded sequence is shorter than continuation token count.")
        cont_label_positions = text_positions[-int(cont_ids.numel()):]
        decision_positions = cont_label_positions - 1
        if int(decision_positions.min().item()) < 0:
            raise RuntimeError("Invalid decision positions after expansion.")
        vision_positions = torch.where(labels_exp == int(IGNORE_INDEX))[0]
        if int(vision_positions.numel()) <= 0:
            raise RuntimeError("No visual token span found in expanded sequence.")

        return ForwardPack(
            prompt=prompt,
            candidate_text=str(candidate_text or ""),
            full_ids=full_ids.detach().cpu(),
            cont_ids=cont_ids.detach().cpu(),
            labels_exp=labels_exp.detach().cpu(),
            cont_label_positions=cont_label_positions.detach().cpu(),
            decision_positions=decision_positions.detach().cpu(),
            vision_positions=vision_positions.detach().cpu(),
            text_positions=text_positions.detach().cpu(),
            logits=logits[0].detach().cpu(),
            attentions=None if outputs.attentions is None else tuple(att.detach().cpu() for att in outputs.attentions),
        )

    def generate_baseline(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
    ) -> str:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        prompt = self.prompt_text(question)
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        image_tensor, image_sizes = self._process_image(image)
        with torch.inference_mode():
            gen_kwargs = dict(
                images=image_tensor.unsqueeze(0),
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
            )
            try:
                output_ids = self.model.generate(
                    input_ids,
                    **gen_kwargs,
                )
            except Exception as exc:
                if "image_sizes" not in str(exc):
                    raise
                gen_kwargs.pop("image_sizes", None)
                output_ids = self.model.generate(
                    input_ids,
                    **gen_kwargs,
                )
        if int(output_ids.shape[1]) > int(input_ids.shape[1]):
            gen_ids = output_ids[:, input_ids.shape[1]:]
        else:
            gen_ids = output_ids
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def stage_a_score_from_pack(
    pack: ForwardPack,
    headset: Headset,
    beta: float,
    lambda_a: float,
    content_indices: Sequence[int],
    eps: float = 1e-6,
) -> Dict[str, float]:
    if pack.attentions is None:
        raise ValueError("Stage A requires attention outputs.")

    faithful_by_token: List[float] = []
    harmful_by_token: List[float] = []
    gap_by_token: List[float] = []

    vision_positions = pack.vision_positions.long()
    text_positions = pack.text_positions.long()

    for rel_idx in content_indices:
        decision_pos = int(pack.decision_positions[int(rel_idx)].item())
        faithful_vals: List[float] = []
        harmful_vals: List[float] = []
        for layer_idx, attn in enumerate(pack.attentions):
            if layer_idx < headset.late_start or layer_idx > headset.late_end:
                continue
            if attn is None:
                continue
            if attn.dim() != 4 or int(attn.size(0)) != 1:
                continue
            att = attn[0].to(torch.float32)
            if decision_pos < 0 or decision_pos >= int(att.size(1)):
                continue
            row = att[:, decision_pos, :]
            vis_sum = row[:, vision_positions].sum(dim=-1)
            if int(text_positions.numel()) > 0:
                txt_sum = row[:, text_positions].sum(dim=-1)
            else:
                txt_sum = torch.zeros_like(vis_sum)
            ratio = vis_sum / torch.clamp(vis_sum + txt_sum, min=float(eps))
            faithful_heads = headset.faithful_heads_by_layer.get(int(layer_idx), [])
            harmful_heads = headset.harmful_heads_by_layer.get(int(layer_idx), [])
            for head_idx in faithful_heads:
                if 0 <= int(head_idx) < int(ratio.numel()):
                    faithful_vals.append(float(ratio[int(head_idx)].item()))
            for head_idx in harmful_heads:
                if 0 <= int(head_idx) < int(ratio.numel()):
                    harmful_vals.append(float(ratio[int(head_idx)].item()))
        faithful_mean = mean_or_zero(faithful_vals)
        harmful_mean = mean_or_zero(harmful_vals)
        faithful_by_token.append(faithful_mean)
        harmful_by_token.append(harmful_mean)
        gap_by_token.append(float(faithful_mean - float(beta) * harmful_mean))

    faithful_mean = mean_or_zero(faithful_by_token)
    harmful_mean = mean_or_zero(harmful_by_token)
    gap_mean = mean_or_zero(gap_by_token)
    faithful_std = std_or_zero(faithful_by_token)
    score = float(gap_mean - float(lambda_a) * faithful_std)
    return {
        "stage_a_score": score,
        "stage_a_gap_mean": gap_mean,
        "stage_a_faithful_mean": faithful_mean,
        "stage_a_harmful_mean": harmful_mean,
        "stage_a_faithful_std": faithful_std,
    }


def stage_b_score_from_packs(
    real_pack: ForwardPack,
    control_pack: ForwardPack,
    lambda_b: float,
    content_indices: Sequence[int],
) -> Dict[str, float]:
    real_logits = real_pack.logits.to(torch.float32)
    control_logits = control_pack.logits.to(torch.float32)

    real_decision_positions = real_pack.decision_positions.long()
    control_decision_positions = control_pack.decision_positions.long()
    real_target_ids = real_pack.labels_exp[real_pack.cont_label_positions.long()].long()
    control_target_ids = control_pack.labels_exp[control_pack.cont_label_positions.long()].long()

    if int(real_target_ids.numel()) != int(control_target_ids.numel()):
        raise RuntimeError("Real/control continuation lengths do not match.")
    if not torch.equal(real_target_ids, control_target_ids):
        raise RuntimeError("Real/control continuation token ids do not match.")

    real_lp = F.log_softmax(real_logits[real_decision_positions], dim=-1).gather(
        1, real_target_ids.unsqueeze(-1)
    ).squeeze(-1)
    control_lp = F.log_softmax(control_logits[control_decision_positions], dim=-1).gather(
        1, control_target_ids.unsqueeze(-1)
    ).squeeze(-1)
    delta = (real_lp - control_lp).detach().cpu()
    picked = [float(delta[int(idx)].item()) for idx in content_indices]
    delta_mean = mean_or_zero(picked)
    delta_std = std_or_zero(picked)
    score = float(delta_mean - float(lambda_b) * delta_std)
    return {
        "stage_b_score": score,
        "stage_b_delta_mean": delta_mean,
        "stage_b_delta_std": delta_std,
    }
