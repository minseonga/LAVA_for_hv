from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from frgavr_cleanroom.runtime import ForwardPack


TORCH_DTYPE: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


class Qwen25VLRuntime:
    """Minimal Qwen2.5-VL runtime for old compact PCP feature replay."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda",
        torch_type: str = "bf16",
        attn_implementation: str = "eager",
        device_map: str = "cuda",
        min_pixels: int = 14 * 14 * 1280,
        max_pixels: int = 28 * 28 * 1280,
    ) -> None:
        from qwen_vl_utils import process_vision_info  # type: ignore
        from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration  # type: ignore

        self.model_path = str(model_path)
        self.device = torch.device(str(device or "cuda"))
        self.dtype = TORCH_DTYPE[str(torch_type)]
        self.process_vision_info = process_vision_info
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
            trust_remote_code=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            attn_implementation=str(attn_implementation),
            torch_dtype=self.dtype,
            device_map=str(device_map),
        ).eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        self.image_token_ids = self._resolve_image_token_ids()

    def _resolve_image_token_ids(self) -> List[int]:
        ids: List[int] = []
        for attr in ("image_token_id", "video_token_id"):
            value = getattr(getattr(self.model, "config", None), attr, None)
            if value is not None:
                ids.append(int(value))
        for token in ("<|image_pad|>", "<|video_pad|>"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if token_id is not None and int(token_id) >= 0:
                ids.append(int(token_id))
        return sorted(set(ids))

    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _messages(self, image: Image.Image, question: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(question or "").strip()},
                ],
            }
        ]

    def prompt_text(self, image: Image.Image, question: str) -> str:
        text = self.processor.apply_chat_template(
            [self._messages(image, question)],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text[0] if isinstance(text, list) else str(text)

    def _processor_inputs(self, image: Image.Image, text: str) -> Any:
        messages = self._messages(image, "")
        messages[0]["content"][1]["text"] = ""
        image_inputs, video_inputs = self.process_vision_info([messages])
        inputs = self.processor(
            text=[str(text)],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _input_ids_for_text(self, image: Image.Image, text: str) -> torch.Tensor:
        return self._processor_inputs(image, text).input_ids[0]

    def _candidate_token_ids(self, candidate_text: str) -> torch.Tensor:
        ids = self.tokenizer(
            str(candidate_text or ""),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        if int(ids.numel()) <= 0:
            raise ValueError("Candidate text tokenization is empty.")
        return ids

    def teacher_force_candidate(
        self,
        image: Image.Image,
        question: str,
        candidate_text: str,
        output_attentions: bool,
        output_hidden_states: bool = False,
    ) -> ForwardPack:
        prompt = self.prompt_text(image, question)
        full_text = prompt + str(candidate_text or "")
        inputs = self._processor_inputs(image, full_text)
        full_ids = inputs.input_ids
        prompt_ids = self._input_ids_for_text(image, prompt)
        cont_ids = self._candidate_token_ids(candidate_text).to(full_ids.device)
        if int(full_ids.shape[-1]) <= int(prompt_ids.numel()):
            raise RuntimeError("Full Qwen2.5-VL sequence is not longer than prompt sequence.")

        cont_len = int(full_ids.shape[-1] - int(prompt_ids.numel()))
        if cont_len != int(cont_ids.numel()):
            # Tokenizer boundary behavior can differ after a chat prompt. The
            # logits positions are still the final continuation span.
            cont_ids = full_ids[0, -cont_len:].detach()

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                output_attentions=bool(output_attentions),
                output_hidden_states=bool(output_hidden_states),
                return_dict=True,
            )

        seq_ids = full_ids[0]
        labels_exp = seq_ids.detach().clone()
        vision_mask = torch.zeros_like(labels_exp, dtype=torch.bool)
        for token_id in self.image_token_ids:
            vision_mask |= labels_exp == int(token_id)
        labels_exp[vision_mask] = -100

        all_positions = torch.arange(int(labels_exp.numel()), device=labels_exp.device)
        text_positions = all_positions[~vision_mask]
        cont_label_positions = all_positions[-cont_len:]
        decision_positions = cont_label_positions - 1
        if int(decision_positions.min().item()) < 0:
            raise RuntimeError("Invalid Qwen2.5-VL decision positions.")
        vision_positions = all_positions[vision_mask]
        if int(vision_positions.numel()) <= 0:
            raise RuntimeError("No Qwen2.5-VL image token positions found.")

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
            logits=outputs.logits[0].detach().cpu(),
            attentions=None if outputs.attentions is None else tuple(att.detach().cpu() for att in outputs.attentions),
            last_hidden_state=(
                None
                if getattr(outputs, "hidden_states", None) is None
                else outputs.hidden_states[-1][0].detach().cpu()
            ),
        )

    def generate_baseline(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 32,
    ) -> str:
        prompt = self.prompt_text(image, question)
        inputs = self._processor_inputs(image, prompt)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
