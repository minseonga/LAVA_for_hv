import argparse
import torch
import os
import json
import csv
from tqdm import tqdm
try:
    import shortuuid  # type: ignore
except Exception:
    shortuuid = None
import uuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from transformers import LogitsProcessor, LogitsProcessorList

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def write_csv(path, rows):
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


def load_rfhar_feats_map(path: str):
    p = str(path or "").strip()
    if p == "":
        return {}
    if not os.path.isfile(p):
        raise FileNotFoundError(f"RF-HAR feature file not found: {p}")

    def _normalize_obj(obj):
        if not isinstance(obj, dict):
            return None, None
        sid = str(obj.get("id", obj.get("question_id", ""))).strip()
        if sid == "":
            return None, None
        feat = {}
        for k in ("C", "A", "D", "B"):
            if k not in obj:
                return None, None
            feat[k] = obj[k]
        return sid, feat

    out = {}
    if p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                sid, feat = _normalize_obj(obj)
                if sid is not None:
                    out[sid] = feat
    else:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
            for k, v in obj.items():
                sid = str(k).strip()
                if sid == "":
                    continue
                sid2, feat = _normalize_obj({"id": sid, **v})
                if sid2 is not None:
                    out[sid2] = feat
        elif isinstance(obj, list):
            for item in obj:
                sid, feat = _normalize_obj(item)
                if sid is not None:
                    out[sid] = feat
    return out


def build_rfhar_feats_tensor_dict(feat_obj):
    if feat_obj is None:
        return None
    out = {}
    for k in ("C", "A", "D", "B"):
        if k not in feat_obj:
            return None
        t = torch.as_tensor(feat_obj[k], dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() != 2:
            raise ValueError(f"RF-HAR feature {k} must be rank-1 or rank-2, got shape={tuple(t.shape)}")
        out[k] = t
    shape = out["C"].shape
    if not (out["A"].shape == shape and out["D"].shape == shape and out["B"].shape == shape):
        raise ValueError(
            f"RF-HAR feature shapes must match, got "
            f"C={tuple(out['C'].shape)} A={tuple(out['A'].shape)} D={tuple(out['D'].shape)} B={tuple(out['B'].shape)}"
        )
    return out


def load_frgg_feats_map(path: str):
    p = str(path or "").strip()
    if p == "":
        return {}
    if not os.path.isfile(p):
        raise FileNotFoundError(f"FRGG feature file not found: {p}")

    def _normalize_obj(obj):
        if not isinstance(obj, dict):
            return None, None
        sid = str(obj.get("id", obj.get("question_id", ""))).strip()
        if sid == "":
            return None, None
        feat = {}
        for k in ("A", "C", "E"):
            if k not in obj:
                return None, None
            feat[k] = obj[k]
        return sid, feat

    out = {}
    if p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s2 = line.strip()
                if not s2:
                    continue
                obj = json.loads(s2)
                sid, feat = _normalize_obj(obj)
                if sid is not None:
                    out[sid] = feat
    else:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
            for k, v in obj.items():
                sid = str(k).strip()
                if sid == "":
                    continue
                sid2, feat = _normalize_obj({"id": sid, **v})
                if sid2 is not None:
                    out[sid2] = feat
        elif isinstance(obj, list):
            for item in obj:
                sid, feat = _normalize_obj(item)
                if sid is not None:
                    out[sid] = feat
    return out


def build_frgg_feats_tensor_dict(feat_obj):
    if feat_obj is None:
        return None
    out = {}
    for k in ("A", "C", "E"):
        if k not in feat_obj:
            return None
        t = torch.as_tensor(feat_obj[k], dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() != 2:
            raise ValueError(f"FRGG feature {k} must be rank-1 or rank-2, got shape={tuple(t.shape)}")
        out[k] = t
    shape = out["A"].shape
    if not (out["C"].shape == shape and out["E"].shape == shape):
        raise ValueError(
            f"FRGG feature shapes must match, got "
            f"A={tuple(out['A'].shape)} C={tuple(out['C'].shape)} E={tuple(out['E'].shape)}"
        )
    return out


def load_frrs_feats_map(path: str):
    p = str(path or "").strip()
    if p == "":
        return {}
    if not os.path.isfile(p):
        raise FileNotFoundError(f"FRRS feature file not found: {p}")

    def _normalize_obj(obj):
        if not isinstance(obj, dict):
            return None, None
        sid = str(obj.get("id", obj.get("question_id", ""))).strip()
        if sid == "":
            return None, None
        feat = {}
        for k in ("A", "C", "E"):
            if k not in obj:
                return None, None
            feat[k] = obj[k]
        if "D" in obj:
            feat["D"] = obj["D"]
        return sid, feat

    out = {}
    if p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s2 = line.strip()
                if not s2:
                    continue
                obj = json.loads(s2)
                sid, feat = _normalize_obj(obj)
                if sid is not None:
                    out[sid] = feat
    else:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
            for k, v in obj.items():
                sid = str(k).strip()
                if sid == "":
                    continue
                sid2, feat = _normalize_obj({"id": sid, **v})
                if sid2 is not None:
                    out[sid2] = feat
        elif isinstance(obj, list):
            for item in obj:
                sid, feat = _normalize_obj(item)
                if sid is not None:
                    out[sid] = feat
    return out


def build_frrs_feats_tensor_dict(feat_obj):
    if feat_obj is None:
        return None
    out = {}
    for k in ("A", "C", "E"):
        if k not in feat_obj:
            return None
        t = torch.as_tensor(feat_obj[k], dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() != 2:
            raise ValueError(f"FRRS feature {k} must be rank-1 or rank-2, got shape={tuple(t.shape)}")
        out[k] = t
    shape = out["A"].shape
    if not (out["C"].shape == shape and out["E"].shape == shape):
        raise ValueError(
            f"FRRS feature shapes must match, got "
            f"A={tuple(out['A'].shape)} C={tuple(out['C'].shape)} E={tuple(out['E'].shape)}"
        )
    if "D" in feat_obj:
        t = torch.as_tensor(feat_obj["D"], dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() != 2:
            raise ValueError(f"FRRS feature D must be rank-1 or rank-2, got shape={tuple(t.shape)}")
        if t.shape != shape:
            raise ValueError(f"FRRS D shape mismatch: D={tuple(t.shape)} vs A={tuple(shape)}")
        out["D"] = t
    return out


class FirstTokenCaptureAndBiasProcessor(LogitsProcessor):
    """Capture first generated-token yes/no logits in generate() path, with optional bias."""

    def __init__(
        self,
        prompt_len: int,
        yes_id: int,
        no_id: int,
        yes_bias: float = 0.0,
        no_bias: float = 0.0,
        apply_bias: bool = False,
    ):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.yes_id = int(yes_id)
        self.no_id = int(no_id)
        self.yes_bias = float(yes_bias)
        self.no_bias = float(no_bias)
        self.apply_bias = bool(apply_bias)
        self._captured = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # First decode step in autoregressive generation.
        # NOTE: with inputs_embeds-based generation, input_ids length may not equal prompt_len.
        # Capture/apply exactly once on the first logits-processor call.
        if self._captured is None:
            # Scores entering logits_processor are post-model (thus post-AIS), pre-local processor edits.
            yes_pre = float(scores[0, self.yes_id].item())
            no_pre = float(scores[0, self.no_id].item())

            if self.apply_bias:
                if self.yes_bias != 0.0:
                    scores[:, self.yes_id] = scores[:, self.yes_id] + float(self.yes_bias)
                if self.no_bias != 0.0:
                    scores[:, self.no_id] = scores[:, self.no_id] + float(self.no_bias)

            yes_post = float(scores[0, self.yes_id].item())
            no_post = float(scores[0, self.no_id].item())
            self._captured = {
                "yes_logit_pre": yes_pre,
                "no_logit_pre": no_pre,
                "margin_pre": float(yes_pre - no_pre),
                "yes_logit_post": yes_post,
                "no_logit_post": no_post,
                "margin_post": float(yes_post - no_post),
                "delta_margin": float((yes_post - no_post) - (yes_pre - no_pre)),
            }
        return scores

    def get_capture(self):
        return dict(self._captured) if self._captured is not None else None


def make_answer_id():
    if shortuuid is not None:
        return shortuuid.uuid()
    return uuid.uuid4().hex


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        enable_ais_gating=bool(args.enable_ais_gating),
        ais_early_start=int(args.ais_early_start),
        ais_early_end=int(args.ais_early_end),
        ais_late_start=int(args.ais_late_start),
        ais_late_end=int(args.ais_late_end),
        ais_topk=int(args.ais_topk),
        ais_tau=float(args.ais_tau),
        ais_gamma=float(args.ais_gamma),
        ais_eps=float(args.ais_eps),
        ais_debug_log=bool(args.ais_debug_log),
        ais_arm=str(args.ais_arm),
        ais_harmful_heads=str(args.ais_harmful_heads),
        ais_faithful_heads=str(args.ais_faithful_heads),
        ais_headset_json=str(args.ais_headset_json),
        ais_faithful_boost=float(args.ais_faithful_boost),
        ais_use_dynamic_omega=bool(args.ais_use_dynamic_omega),
        ais_use_budget_routing=bool(args.ais_use_budget_routing),
        ais_budget_total=float(args.ais_budget_total),
        ais_harmful_top_ratio=float(args.ais_harmful_top_ratio),
        ais_faithful_top_ratio=float(args.ais_faithful_top_ratio),
        ais_bipolar_harmful_ratio=float(args.ais_bipolar_harmful_ratio),
        ais_budget_patch_topk=int(args.ais_budget_patch_topk),
        ais_strict_headset_layers=bool(args.ais_strict_headset_layers),
        ais_operator=str(args.ais_operator),
        ais_semihard_penalty=float(args.ais_semihard_penalty),
        ais_use_oracle_roles=bool(args.ais_use_oracle_roles),
        ais_oracle_role_csv=str(args.ais_oracle_role_csv),
        ais_oracle_supportive_topk=int(args.ais_oracle_supportive_topk),
        ais_oracle_assertive_topk=int(args.ais_oracle_assertive_topk),
        ais_oracle_lambda_pos=float(args.ais_oracle_lambda_pos),
        ais_oracle_lambda_neg=float(args.ais_oracle_lambda_neg),
        ais_oracle_bias_clip=float(args.ais_oracle_bias_clip),
        path_probe_mode=str(args.path_probe_mode),
        path_probe_penalty=float(args.path_probe_penalty),
        path_probe_first_step_only=bool(args.path_probe_first_step_only),
        enable_rfhar=bool(args.enable_rfhar),
        rfhar_early_start=int(args.rfhar_early_start),
        rfhar_early_end=int(args.rfhar_early_end),
        rfhar_late_start=int(args.rfhar_late_start),
        rfhar_late_end=int(args.rfhar_late_end),
        rfhar_r_percent=float(args.rfhar_r_percent),
        rfhar_gamma=float(args.rfhar_gamma),
        rfhar_lambda_penalty=float(args.rfhar_lambda_penalty),
        rfhar_eps=float(args.rfhar_eps),
        rfhar_debug_log=bool(args.rfhar_debug_log),
        enable_frgg=bool(args.enable_frgg),
        frgg_late_start=int(args.frgg_late_start),
        frgg_late_end=int(args.frgg_late_end),
        frgg_gamma=float(args.frgg_gamma),
        frgg_tau_c=float(args.frgg_tau_c),
        frgg_tau_e=float(args.frgg_tau_e),
        frgg_k_c=float(args.frgg_k_c),
        frgg_k_e=float(args.frgg_k_e),
        frgg_topk_ratio=float(args.frgg_topk_ratio),
        frgg_eps=float(args.frgg_eps),
        frgg_debug_log=bool(args.frgg_debug_log),
        enable_frrs=bool(args.enable_frrs),
        frrs_late_start=int(args.frrs_late_start),
        frrs_late_end=int(args.frrs_late_end),
        frrs_alpha=float(args.frrs_alpha),
        frrs_beta=float(args.frrs_beta),
        frrs_tau_c=float(args.frrs_tau_c),
        frrs_tau_e=float(args.frrs_tau_e),
        frrs_k_c=float(args.frrs_k_c),
        frrs_k_e=float(args.frrs_k_e),
        frrs_topk_ratio=float(args.frrs_topk_ratio),
        frrs_eps=float(args.frrs_eps),
        frrs_arm=str(args.frrs_arm),
        frrs_head_mode=str(args.frrs_head_mode),
        frrs_r_percent=float(args.frrs_r_percent),
        frrs_online_recompute_feats=bool(args.frrs_online_recompute_feats),
        frrs_online_blend=float(args.frrs_online_blend),
        frrs_debug_log=bool(args.frrs_debug_log),
    )
    path_probe_mode_active = str(args.path_probe_mode).strip().lower() != "none"
    need_runtime_cfg = bool(args.enable_ais_gating) or bool(path_probe_mode_active) or bool(args.enable_rfhar) or bool(args.enable_frgg) or bool(args.enable_frrs)
    rfhar_feats_map = {}
    if bool(args.enable_rfhar):
        rfhar_feats_map = load_rfhar_feats_map(args.rfhar_feats_json)
        if len(rfhar_feats_map) == 0:
            raise RuntimeError(
                "RF-HAR is enabled but no features were loaded. "
                "Provide --rfhar-feats-json with per-sample C/A/D/B features."
            )
    frgg_feats_map = {}
    if bool(args.enable_frgg):
        frgg_feats_map = load_frgg_feats_map(args.frgg_feats_json)
        if len(frgg_feats_map) == 0:
            raise RuntimeError(
                "FRGG is enabled but no features were loaded. "
                "Provide --frgg-feats-json with per-sample A/C/E features."
            )
    frrs_feats_map = {}
    if bool(args.enable_frrs):
        frrs_feats_map = load_frrs_feats_map(args.frrs_feats_json)
        if len(frrs_feats_map) == 0:
            raise RuntimeError(
                "FRRS is enabled but no features were loaded. "
                "Provide --frrs-feats-json with per-sample A/C/E (optional D) features."
            )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    ft_rows = []
    yes_token_id = None
    no_token_id = None
    need_yesno_ids = (str(args.dump_first_token_logits).strip() != "") or bool(args.first_token_safeguard)
    if need_yesno_ids:
        def _pick_token_id(text: str):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return int(ids[0])
            ids2 = tokenizer(text.strip(), add_special_tokens=False).input_ids
            if len(ids2) == 1:
                return int(ids2[0])
            return int(ids[-1]) if len(ids) > 0 else None

        yes_token_id = _pick_token_id(" yes")
        no_token_id = _pick_token_id(" no")
        if yes_token_id is None or no_token_id is None:
            raise RuntimeError("Failed to resolve single-token ids for yes/no first-token diagnostics.")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor_cuda = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            gen_kwargs = dict(
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

            # Optional: capture no-AIS first-token logits on the same generate path.
            pre_nogate_ft = None
            if (
                str(args.dump_first_token_logits).strip() != ""
                and bool(args.first_token_logits_include_pre)
                and bool(need_runtime_cfg)
                and hasattr(model, "configure_ais_gating")
            ):
                saved_cfg = model.configure_ais_gating()
                model.configure_ais_gating(
                    enable_ais_gating=False,
                    path_probe_mode="none",
                    path_probe_penalty=0.0,
                )
                pre_cap = FirstTokenCaptureAndBiasProcessor(
                    prompt_len=int(input_ids.shape[1]),
                    yes_id=int(yes_token_id),
                    no_id=int(no_token_id),
                    apply_bias=False,
                )
                pre_kwargs = dict(gen_kwargs)
                pre_kwargs["max_new_tokens"] = 1
                pre_kwargs["logits_processor"] = LogitsProcessorList([pre_cap])
                _ = model.generate(
                    input_ids,
                    images=image_tensor_cuda,
                    image_sizes=image_sizes,
                    **pre_kwargs,
                )
                pre_nogate_ft = pre_cap.get_capture()
                model.configure_ais_gating(**saved_cfg)

            need_ft_capture = str(args.dump_first_token_logits).strip() != "" or bool(args.first_token_safeguard)
            cap_proc = None
            if need_ft_capture:
                cap_proc = FirstTokenCaptureAndBiasProcessor(
                    prompt_len=int(input_ids.shape[1]),
                    yes_id=int(yes_token_id),
                    no_id=int(no_token_id),
                    yes_bias=float(args.first_token_yes_bias),
                    no_bias=float(args.first_token_no_bias),
                    apply_bias=bool(args.first_token_safeguard),
                )
                gen_kwargs["logits_processor"] = LogitsProcessorList([cap_proc])

            if bool(need_runtime_cfg) and hasattr(model, "configure_ais_gating"):
                gen_kwargs.update(
                    dict(
                        enable_ais_gating=bool(args.enable_ais_gating),
                        ais_early_start=int(args.ais_early_start),
                        ais_early_end=int(args.ais_early_end),
                        ais_late_start=int(args.ais_late_start),
                        ais_late_end=int(args.ais_late_end),
                        ais_topk=int(args.ais_topk),
                        ais_tau=float(args.ais_tau),
                        ais_gamma=float(args.ais_gamma),
                        ais_eps=float(args.ais_eps),
                        ais_debug_log=bool(args.ais_debug_log),
                        ais_arm=str(args.ais_arm),
                        ais_harmful_heads=str(args.ais_harmful_heads),
                        ais_faithful_heads=str(args.ais_faithful_heads),
                        ais_headset_json=str(args.ais_headset_json),
                        ais_faithful_boost=float(args.ais_faithful_boost),
                        ais_use_dynamic_omega=bool(args.ais_use_dynamic_omega),
                        ais_use_budget_routing=bool(args.ais_use_budget_routing),
                        ais_budget_total=float(args.ais_budget_total),
                        ais_harmful_top_ratio=float(args.ais_harmful_top_ratio),
                        ais_faithful_top_ratio=float(args.ais_faithful_top_ratio),
                        ais_bipolar_harmful_ratio=float(args.ais_bipolar_harmful_ratio),
                        ais_budget_patch_topk=int(args.ais_budget_patch_topk),
                        ais_strict_headset_layers=bool(args.ais_strict_headset_layers),
                        ais_operator=str(args.ais_operator),
                        ais_semihard_penalty=float(args.ais_semihard_penalty),
                        ais_use_oracle_roles=bool(args.ais_use_oracle_roles),
                        ais_oracle_role_csv=str(args.ais_oracle_role_csv),
                        ais_oracle_supportive_topk=int(args.ais_oracle_supportive_topk),
                        ais_oracle_assertive_topk=int(args.ais_oracle_assertive_topk),
                        ais_oracle_lambda_pos=float(args.ais_oracle_lambda_pos),
                        ais_oracle_lambda_neg=float(args.ais_oracle_lambda_neg),
                        ais_oracle_bias_clip=float(args.ais_oracle_bias_clip),
                        path_probe_mode=str(args.path_probe_mode),
                        path_probe_penalty=float(args.path_probe_penalty),
                        path_probe_first_step_only=bool(args.path_probe_first_step_only),
                        enable_rfhar=bool(args.enable_rfhar),
                        rfhar_early_start=int(args.rfhar_early_start),
                        rfhar_early_end=int(args.rfhar_early_end),
                        rfhar_late_start=int(args.rfhar_late_start),
                        rfhar_late_end=int(args.rfhar_late_end),
                        rfhar_r_percent=float(args.rfhar_r_percent),
                        rfhar_gamma=float(args.rfhar_gamma),
                        rfhar_lambda_penalty=float(args.rfhar_lambda_penalty),
                        rfhar_eps=float(args.rfhar_eps),
                        rfhar_debug_log=bool(args.rfhar_debug_log),
                        enable_frgg=bool(args.enable_frgg),
                        frgg_late_start=int(args.frgg_late_start),
                        frgg_late_end=int(args.frgg_late_end),
                        frgg_gamma=float(args.frgg_gamma),
                        frgg_tau_c=float(args.frgg_tau_c),
                        frgg_tau_e=float(args.frgg_tau_e),
                        frgg_k_c=float(args.frgg_k_c),
                        frgg_k_e=float(args.frgg_k_e),
                        frgg_topk_ratio=float(args.frgg_topk_ratio),
                        frgg_eps=float(args.frgg_eps),
                        frgg_debug_log=bool(args.frgg_debug_log),
                        enable_frrs=bool(args.enable_frrs),
                        frrs_late_start=int(args.frrs_late_start),
                        frrs_late_end=int(args.frrs_late_end),
                        frrs_alpha=float(args.frrs_alpha),
                        frrs_beta=float(args.frrs_beta),
                        frrs_tau_c=float(args.frrs_tau_c),
                        frrs_tau_e=float(args.frrs_tau_e),
                        frrs_k_c=float(args.frrs_k_c),
                        frrs_k_e=float(args.frrs_k_e),
                        frrs_topk_ratio=float(args.frrs_topk_ratio),
                        frrs_eps=float(args.frrs_eps),
                        frrs_arm=str(args.frrs_arm),
                        frrs_head_mode=str(args.frrs_head_mode),
                        frrs_r_percent=float(args.frrs_r_percent),
                        frrs_online_recompute_feats=bool(args.frrs_online_recompute_feats),
                        frrs_online_blend=float(args.frrs_online_blend),
                        frrs_debug_log=bool(args.frrs_debug_log),
                    )
                )
            if bool(args.enable_ais_gating):
                gen_kwargs["ais_sample_ids"] = [str(idx)]
            if bool(args.enable_rfhar):
                feat_obj = rfhar_feats_map.get(str(idx), None)
                if feat_obj is not None:
                    gen_kwargs["rfhar_feats"] = build_rfhar_feats_tensor_dict(feat_obj)
            if bool(args.enable_frgg):
                feat_obj = frgg_feats_map.get(str(idx), None)
                if feat_obj is not None:
                    gen_kwargs["frgg_feats"] = build_frgg_feats_tensor_dict(feat_obj)
            if bool(args.enable_frrs):
                feat_obj = frrs_feats_map.get(str(idx), None)
                if feat_obj is not None:
                    gen_kwargs["frrs_feats"] = build_frrs_feats_tensor_dict(feat_obj)
            output_ids = model.generate(
                input_ids,
                images=image_tensor_cuda,
                image_sizes=image_sizes,
                **gen_kwargs,
            )

            if str(args.dump_first_token_logits).strip() != "":
                ft = {
                    "question_id": str(idx),
                    "yes_token_id": int(yes_token_id),
                    "no_token_id": int(no_token_id),
                }
                cur = cap_proc.get_capture() if cap_proc is not None else None
                if cur is not None:
                    ft.update(cur)
                if pre_nogate_ft is not None:
                    ft["yes_logit_pre_nogate"] = float(pre_nogate_ft["yes_logit_pre"])
                    ft["no_logit_pre_nogate"] = float(pre_nogate_ft["no_logit_pre"])
                    ft["margin_pre_nogate"] = float(pre_nogate_ft["margin_pre"])
                ft_rows.append(ft)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = make_answer_id()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()
    if str(args.dump_first_token_logits).strip() != "":
        dump_path = os.path.abspath(str(args.dump_first_token_logits))
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        write_csv(dump_path, ft_rows)
        print(f"[saved] first-token logits: {dump_path}")

    if str(args.ais_debug_dump).strip() != "" and hasattr(model, "get_ais_debug_rows"):
        rows = model.get_ais_debug_rows(reset=False)
        dump_path = os.path.abspath(str(args.ais_debug_dump))
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        write_csv(dump_path, rows)
        print(f"[saved] AIS debug rows: {dump_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--enable-ais-gating", action="store_true")
    parser.add_argument("--ais-early-start", type=int, default=0)
    parser.add_argument("--ais-early-end", type=int, default=15)
    parser.add_argument("--ais-late-start", type=int, default=16)
    parser.add_argument("--ais-late-end", type=int, default=31)
    parser.add_argument("--ais-topk", type=int, default=8)
    parser.add_argument("--ais-tau", type=float, default=2.2)
    parser.add_argument("--ais-gamma", type=float, default=0.2)
    parser.add_argument("--ais-eps", type=float, default=1e-6)
    parser.add_argument("--ais-debug-log", action="store_true")
    parser.add_argument("--ais-debug-dump", type=str, default="")
    parser.add_argument("--ais-arm", type=str, default="legacy", choices=["legacy", "harmful_only", "faithful_only", "bipolar"])
    parser.add_argument("--ais-harmful-heads", type=str, default="")
    parser.add_argument("--ais-faithful-heads", type=str, default="")
    parser.add_argument("--ais-headset-json", type=str, default="")
    parser.add_argument("--ais-faithful-boost", type=float, default=1.0)
    parser.add_argument("--ais-use-dynamic-omega", dest="ais_use_dynamic_omega", action="store_true")
    parser.add_argument("--no-ais-use-dynamic-omega", dest="ais_use_dynamic_omega", action="store_false")
    parser.add_argument("--ais-use-budget-routing", action="store_true")
    parser.add_argument("--ais-budget-total", type=float, default=0.0)
    parser.add_argument("--ais-harmful-top-ratio", type=float, default=0.2)
    parser.add_argument("--ais-faithful-top-ratio", type=float, default=0.2)
    parser.add_argument("--ais-bipolar-harmful-ratio", type=float, default=0.5)
    parser.add_argument("--ais-budget-patch-topk", type=int, default=16)
    parser.add_argument("--ais-strict-headset-layers", action="store_true")
    parser.add_argument("--ais-operator", type=str, default="soft", choices=["soft", "semi_hard"])
    parser.add_argument("--ais-semihard-penalty", type=float, default=0.0)
    parser.add_argument("--ais-use-oracle-roles", action="store_true")
    parser.add_argument("--ais-oracle-role-csv", type=str, default="")
    parser.add_argument("--ais-oracle-supportive-topk", type=int, default=5)
    parser.add_argument("--ais-oracle-assertive-topk", type=int, default=5)
    parser.add_argument("--ais-oracle-lambda-pos", type=float, default=0.25)
    parser.add_argument("--ais-oracle-lambda-neg", type=float, default=0.25)
    parser.add_argument("--ais-oracle-bias-clip", type=float, default=2.0)
    parser.add_argument("--path-probe-mode", type=str, default="none", choices=["none", "drop_img", "drop_text", "drop_both"])
    parser.add_argument("--path-probe-penalty", type=float, default=0.0)
    parser.add_argument("--path-probe-first-step-only", action="store_true")
    parser.add_argument("--path-probe-all-steps", action="store_true")
    parser.add_argument("--enable-rfhar", action="store_true")
    parser.add_argument("--rfhar-early-start", type=int, default=0)
    parser.add_argument("--rfhar-early-end", type=int, default=15)
    parser.add_argument("--rfhar-late-start", type=int, default=16)
    parser.add_argument("--rfhar-late-end", type=int, default=31)
    parser.add_argument("--rfhar-r-percent", type=float, default=0.2)
    parser.add_argument("--rfhar-gamma", type=float, default=0.3)
    parser.add_argument("--rfhar-lambda-penalty", type=float, default=0.5)
    parser.add_argument("--rfhar-eps", type=float, default=1e-6)
    parser.add_argument("--rfhar-debug-log", action="store_true")
    parser.add_argument("--rfhar-feats-json", type=str, default="")
    parser.add_argument("--enable-frgg", action="store_true")
    parser.add_argument("--frgg-late-start", type=int, default=16)
    parser.add_argument("--frgg-late-end", type=int, default=30)
    parser.add_argument("--frgg-gamma", type=float, default=0.3)
    parser.add_argument("--frgg-tau-c", type=float, default=0.0)
    parser.add_argument("--frgg-tau-e", type=float, default=0.0)
    parser.add_argument("--frgg-k-c", type=float, default=8.0)
    parser.add_argument("--frgg-k-e", type=float, default=8.0)
    parser.add_argument("--frgg-topk-ratio", type=float, default=0.2)
    parser.add_argument("--frgg-eps", type=float, default=1e-6)
    parser.add_argument("--frgg-debug-log", action="store_true")
    parser.add_argument("--frgg-feats-json", type=str, default="")
    parser.add_argument("--enable-frrs", action="store_true")
    parser.add_argument("--frrs-late-start", type=int, default=18)
    parser.add_argument("--frrs-late-end", type=int, default=21)
    parser.add_argument("--frrs-alpha", type=float, default=0.5)
    parser.add_argument("--frrs-beta", type=float, default=0.5)
    parser.add_argument("--frrs-tau-c", type=float, default=0.0)
    parser.add_argument("--frrs-tau-e", type=float, default=0.0)
    parser.add_argument("--frrs-k-c", type=float, default=8.0)
    parser.add_argument("--frrs-k-e", type=float, default=8.0)
    parser.add_argument("--frrs-topk-ratio", type=float, default=0.2)
    parser.add_argument("--frrs-eps", type=float, default=1e-6)
    parser.add_argument("--frrs-arm", type=str, default="supportive", choices=["supportive", "bipolar"])
    parser.add_argument("--frrs-head-mode", type=str, default="dynamic", choices=["static", "dynamic", "hybrid"])
    parser.add_argument("--frrs-r-percent", type=float, default=0.2)
    parser.add_argument("--frrs-online-recompute-feats", action="store_true")
    parser.add_argument("--frrs-online-blend", type=float, default=1.0)
    parser.add_argument("--frrs-debug-log", action="store_true")
    parser.add_argument("--frrs-feats-json", type=str, default="")
    parser.add_argument("--dump-first-token-logits", type=str, default="")
    parser.add_argument("--first-token-logits-include-pre", action="store_true")
    parser.add_argument("--first-token-safeguard", action="store_true")
    parser.add_argument("--first-token-yes-bias", type=float, default=0.0)
    parser.add_argument("--first-token-no-bias", type=float, default=0.0)
    parser.set_defaults(ais_use_dynamic_omega=True)
    parser.set_defaults(path_probe_first_step_only=True)
    args = parser.parse_args()
    if bool(args.path_probe_all_steps):
        args.path_probe_first_step_only = False

    eval_model(args)
