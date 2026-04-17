from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from frgavr_cleanroom.runtime import ForwardPack


TORCH_TYPE_ARG = {
    "fp16": "float16",
    "bf16": "bfloat16",
}

TORCH_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _install_llava_next_root(llava_next_root: str) -> None:
    root = Path(llava_next_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"official LLaVA-NeXT repo not found: {root}")

    existing = sys.modules.get("llava")
    existing_file = str(getattr(existing, "__file__", "") or "")
    if existing is not None and not existing_file.startswith(str(root)):
        raise RuntimeError(
            "A non-official `llava` package is already imported in this process. "
            "Run the online script with --runtime_backend llava_next_official before "
            "any local LLaVA runtime is initialized."
        )

    root_s = str(root)
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)


class OfficialLlavaNextRuntime:
    """Runtime adapter exposing the CleanroomLlavaRuntime interface for official LLaVA-NeXT."""

    def __init__(
        self,
        *,
        llava_next_root: str,
        model_path: str,
        model_base: Optional[str],
        conv_mode: str,
        device: str = "cuda",
        torch_type: str = "fp16",
        attn_implementation: str = "eager",
    ) -> None:
        _install_llava_next_root(llava_next_root)

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        torch_type = str(torch_type)
        if torch_type not in TORCH_TYPE_ARG:
            raise ValueError(f"Unsupported torch_type={torch_type!r}; expected one of {sorted(TORCH_TYPE_ARG)}")
        attn_implementation = str(attn_implementation or "eager")
        if attn_implementation == "none":
            attn_implementation = "eager"

        disable_torch_init()
        model_path = os.path.expanduser(str(model_path))
        model_base = None if not str(model_base or "").strip() else os.path.expanduser(str(model_base))
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            device_map=str(device or "cuda"),
            torch_dtype=TORCH_TYPE_ARG[torch_type],
            attn_implementation=attn_implementation,
        )
        self.tokenizer.padding_side = "right"
        self.model.eval()
        self.conv_mode = str(conv_mode)
        self.device = torch.device(str(device or "cuda"))
        self.torch_type = torch_type
        self.dtype = TORCH_DTYPE[torch_type]

    def load_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def prompt_text(self, question: str) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import SeparatorStyle, conv_templates

        conv = conv_templates[self.conv_mode].copy()
        if conv.sep_style == SeparatorStyle.LLAMA_3 and conv.tokenizer is None:
            conv.tokenizer = self.tokenizer
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + str(question or "").strip())
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _to_device_images(self, images: Any) -> Any:
        if isinstance(images, torch.Tensor):
            return images.to(device=self.device, dtype=self.dtype)
        if isinstance(images, list):
            return [item.to(device=self.device, dtype=self.dtype) for item in images]
        raise TypeError(f"Unsupported processed image type: {type(images)!r}")

    def _process_image(self, image: Image.Image) -> Tuple[Any, List[Tuple[int, int]]]:
        from llava.mm_utils import process_images

        image_tensor = process_images([image], self.image_processor, self.model.config)
        return self._to_device_images(image_tensor), [image.size]

    def _prepare_multimodal_expanded_sequence(
        self,
        full_ids: torch.Tensor,
        images_tensor: Any,
        image_sizes: Sequence[Tuple[int, int]],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        base_attn = torch.ones_like(full_ids, dtype=torch.long, device=self.device)
        packed = self.model.prepare_inputs_labels_for_multimodal(
            full_ids,
            None,
            base_attn,
            None,
            full_ids,
            images_tensor,
            modalities=["image"],
            image_sizes=image_sizes,
        )
        if not isinstance(packed, tuple) or len(packed) != 6:
            raise RuntimeError("Unexpected official LLaVA-NeXT prepare_inputs_labels_for_multimodal return shape.")
        _, pos_ids_e, attn_mask_e, _, mm_embeds_e, labels_e = packed
        if mm_embeds_e is None or labels_e is None or attn_mask_e is None:
            raise RuntimeError("Official LLaVA-NeXT multimodal preparation returned incomplete tensors.")
        return pos_ids_e, attn_mask_e, mm_embeds_e, labels_e

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
            backbone = self.model.get_model() if hasattr(self.model, "get_model") else getattr(self.model, "model", None)
            if backbone is None:
                raise RuntimeError("Could not resolve official LLaVA-NeXT language-model backbone.")

            forward_kwargs: Dict[str, Any] = {
                "inputs_embeds": mm_embeds_e,
                "attention_mask": attn_mask_e,
                "use_cache": False,
                "output_attentions": bool(output_attentions),
                "output_hidden_states": False,
                "return_dict": True,
            }
            if pos_ids_e is not None:
                forward_kwargs["position_ids"] = pos_ids_e
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
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                modalities=["image"] * int(input_ids.shape[0]),
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
