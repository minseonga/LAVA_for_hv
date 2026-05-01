#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

LABEL_KEEP = {
    "baseline_label",
    "intervention_label",
    "baseline_text",
    "intervention_text",
    "baseline_correct",
    "intervention_correct",
    "harm",
    "help",
    "neutral",
    "category",
    "gt_label",
    "answer",
    "label",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def read_jsonl_rows(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
            if int(limit) > 0 and len(rows) >= int(limit):
                break
    return rows


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_text_map(path: str, text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl_rows(path):
        sid = safe_id(row.get("question_id", row.get("id")))
        if not sid:
            continue
        key = str(text_key)
        if key == "auto":
            for cand in ("output", "text", "answer", "caption"):
                value = str(row.get(cand, "")).strip()
                if value:
                    out[sid] = value
                    break
        else:
            out[sid] = str(row.get(key, "")).strip()
    return out


def read_label_rows(path: str) -> Dict[str, Dict[str, str]]:
    if not str(path or "").strip():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sid = safe_id(row.get("id", row.get("question_id", row.get("qid"))))
        if not sid:
            continue
        out[sid] = {key: str(row.get(key, "")) for key in LABEL_KEEP if key in row}
    return out


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(os.path.abspath(path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(max(0.0, sum((x - mu) ** 2 for x in values) / float(len(values)))))


def binary_auroc(xs: Sequence[float], ys: Sequence[int]) -> float:
    pos = [float(x) for x, y in zip(xs, ys) if int(y) == 1]
    neg = [float(x) for x, y in zip(xs, ys) if int(y) == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / float(len(pos) * len(neg)))


def label_margin_from_logits(logits: Any, *, token_ids: Mapping[str, Sequence[int]], candidate_label: str) -> Dict[str, float]:
    import torch
    from run_discriminative_meta_strong_online import logsumexp_ids

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    yes_lp_t = logsumexp_ids(log_probs, token_ids.get("yes", []))
    no_lp_t = logsumexp_ids(log_probs, token_ids.get("no", []))
    yes_lp = float(yes_lp_t.item())
    no_lp = float(no_lp_t.item())
    yes_minus_no = float((yes_lp_t - no_lp_t).item())
    yes_prob = float(torch.sigmoid(yes_lp_t - no_lp_t).item())
    no_prob = float(1.0 - yes_prob)
    if candidate_label == "yes":
        cand_lp = yes_lp
        alt_lp = no_lp
        cand_margin = yes_minus_no
        cand_prob = yes_prob
    elif candidate_label == "no":
        cand_lp = no_lp
        alt_lp = yes_lp
        cand_margin = -yes_minus_no
        cand_prob = no_prob
    else:
        cand_lp = -100.0
        alt_lp = -100.0
        cand_margin = 0.0
        cand_prob = 0.0
    return {
        "yes_lp": yes_lp,
        "no_lp": no_lp,
        "yes_minus_no": yes_minus_no,
        "yes_prob_binary": yes_prob,
        "no_prob_binary": no_prob,
        "candidate_label_lp": float(cand_lp),
        "alt_label_lp": float(alt_lp),
        "candidate_minus_alt": float(cand_margin),
        "candidate_prob_binary": float(cand_prob),
        "margin_abs": float(abs(yes_minus_no)),
    }


def layer_trajectory(
    runtime: Any,
    *,
    image: Any,
    question: str,
    candidate_text: str,
    apply_final_norm: bool,
) -> List[Dict[str, Any]]:
    import torch
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from frgavr_cleanroom.runtime import select_content_indices
    from run_discriminative_meta_strong_online import final_label_from_text, yesno_token_id_sets

    prompt = runtime.prompt_text(question)
    prompt_ids = tokenizer_image_token(prompt, runtime.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
        runtime.device
    )
    cont_ids = runtime.tokenizer(str(candidate_text or ""), add_special_tokens=False, return_tensors="pt").input_ids[0].to(
        runtime.device
    )
    if int(cont_ids.numel()) <= 0:
        raise ValueError("Candidate text tokenization is empty.")

    full_ids = torch.cat([prompt_ids[0], cont_ids], dim=0).unsqueeze(0)
    images_tensor, image_sizes = runtime._process_image(image)
    token_ids = yesno_token_id_sets(runtime.tokenizer)
    candidate_label = final_label_from_text(str(candidate_text or ""))

    with torch.no_grad():
        pos_ids_e, attn_mask_e, mm_embeds_e, labels_e = runtime._prepare_multimodal_expanded_sequence(
            full_ids=full_ids,
            images_tensor=images_tensor,
            image_sizes=image_sizes,
        )
        backbone = runtime.model.get_model() if hasattr(runtime.model, "get_model") else getattr(runtime.model, "model", None)
        if backbone is None:
            raise RuntimeError("Could not resolve language-model backbone.")
        forward_kwargs: Dict[str, Any] = {
            "inputs_embeds": mm_embeds_e,
            "attention_mask": attn_mask_e,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if pos_ids_e is not None:
            forward_kwargs["position_ids"] = pos_ids_e
        if runtime.teacher_force_forward_mode in {"model", "full", "legacy"}:
            try:
                outputs = runtime.model(**forward_kwargs)
            except TypeError as exc:
                if "position_ids" not in str(exc):
                    raise
                forward_kwargs.pop("position_ids", None)
                outputs = runtime.model(**forward_kwargs)
        else:
            try:
                outputs = backbone(**forward_kwargs)
            except TypeError as exc:
                if "position_ids" not in str(exc):
                    raise
                forward_kwargs.pop("position_ids", None)
                outputs = backbone(**forward_kwargs)

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Forward did not return hidden_states.")
        if runtime.teacher_force_forward_mode in {"model", "full", "legacy"}:
            final_logits = getattr(outputs, "logits", None)
            if final_logits is None:
                raise RuntimeError("Model forward did not return logits.")
        else:
            final_logits = runtime.model.lm_head(outputs[0])
        final_logits = final_logits.float()[0]

        labels_exp = labels_e[0]
        text_positions = torch.where(labels_exp != int(IGNORE_INDEX))[0]
        if int(text_positions.numel()) < int(cont_ids.numel()):
            raise RuntimeError("Expanded sequence is shorter than continuation token count.")
        cont_label_positions = text_positions[-int(cont_ids.numel()):]
        decision_positions = cont_label_positions - 1
        if int(decision_positions.min().item()) < 0:
            raise RuntimeError("Invalid decision positions after expansion.")
        first_decision_pos = int(decision_positions[0].item())
        target_ids = labels_exp[cont_label_positions].long()
        content_indices = select_content_indices(runtime.tokenizer, cont_ids.detach().cpu())
        content_indices = [int(i) for i in content_indices if 0 <= int(i) < int(target_ids.numel())]
        if not content_indices:
            content_indices = list(range(int(target_ids.numel())))

        norm = None
        if bool(apply_final_norm):
            norm = getattr(backbone, "norm", None)
            if norm is None and hasattr(backbone, "model"):
                norm = getattr(backbone.model, "norm", None)

        rows: List[Dict[str, Any]] = []
        n_hidden = int(len(hidden_states))
        for idx, hidden in enumerate(hidden_states):
            is_final = idx == n_hidden - 1
            if is_final:
                logits = final_logits[first_decision_pos]
            else:
                h = hidden[:, first_decision_pos, :]
                if norm is not None:
                    h = norm(h)
                logits = runtime.model.lm_head(h).float()[0]
            vals = label_margin_from_logits(logits, token_ids=token_ids, candidate_label=candidate_label)

            if is_final:
                token_logits = final_logits[decision_positions]
            else:
                h_cont = hidden[:, decision_positions, :]
                if norm is not None:
                    h_cont = norm(h_cont)
                token_logits = runtime.model.lm_head(h_cont).float()[0]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            probs = torch.softmax(token_logits, dim=-1)
            token_ent = -(probs * log_probs).sum(dim=-1)
            top2_vals, top2_idx = torch.topk(token_logits, k=2, dim=-1)
            top1_logit = top2_vals[:, 0]
            top2_logit = top2_vals[:, 1]
            top1_id = top2_idx[:, 0]
            target_logit = token_logits.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
            best_other_logit = torch.where(top1_id == target_ids, top2_logit, top1_logit)
            target_gap = target_logit - best_other_logit
            pick = torch.tensor(content_indices, dtype=torch.long, device=target_gap.device)
            target_gap_content = target_gap.index_select(0, pick)
            entropy_content = token_ent.index_select(0, pick)
            rows.append(
                {
                    "layer_index": int(idx),
                    "layer_frac": float(idx / max(1, n_hidden - 1)),
                    "is_final_layer": int(is_final),
                    "candidate_label": candidate_label,
                    **vals,
                    "c_target_gap_content_min": float(target_gap_content.min().item()),
                    "c_entropy_content_mean": float(entropy_content.mean().item()),
                    "c_first_target_gap": float(target_gap[0].item()),
                    "c_n_content_tokens": int(len(content_indices)),
                }
            )
    return rows


SUMMARY_FEATURES = [
    ("candidate_margin", "candidate_minus_alt"),
    ("c_target_gap_content_min", "c_target_gap_content_min"),
    ("c_entropy_content_mean", "c_entropy_content_mean"),
    ("c_first_target_gap", "c_first_target_gap"),
]


def summarize_layers(long_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_layer: Dict[int, Dict[str, Any]] = {}
    for row in long_rows:
        try:
            layer = int(row["layer_index"])
            harm = int(float(row.get("harm", 0) or 0))
            help_ = int(float(row.get("help", 0) or 0))
        except Exception:
            continue
        if harm not in {0, 1} or help_ not in {0, 1}:
            continue
        item = by_layer.setdefault(
            layer,
            {
                "ys": [],
                **{f"{name}_values": [] for name, _ in SUMMARY_FEATURES},
                **{f"{name}_harm": [] for name, _ in SUMMARY_FEATURES},
                **{f"{name}_help": [] for name, _ in SUMMARY_FEATURES},
            },
        )
        if harm == 1 or help_ == 1:
            item["ys"].append(harm)
            for name, key in SUMMARY_FEATURES:
                try:
                    value = float(row[key])
                except Exception:
                    continue
                item[f"{name}_values"].append(value)
                if harm == 1:
                    item[f"{name}_harm"].append(value)
                if help_ == 1:
                    item[f"{name}_help"].append(value)
    out: List[Dict[str, Any]] = []
    for layer in sorted(by_layer):
        item = by_layer[layer]
        ys = [int(x) for x in item["ys"]]
        if not ys:
            continue
        summary: Dict[str, Any] = {
            "layer_index": layer,
            "n": len(ys),
            "n_harm": sum(ys),
            "n_help": len(ys) - sum(ys),
        }
        for name, _ in SUMMARY_FEATURES:
            values = [float(x) for x in item[f"{name}_values"]]
            if len(values) != len(ys):
                continue
            auc_high = binary_auroc(values, ys)
            auc_low = binary_auroc([-x for x in values], ys)
            summary[f"harm_{name}_mean"] = mean(item[f"{name}_harm"])
            summary[f"harm_{name}_std"] = std(item[f"{name}_harm"])
            summary[f"help_{name}_mean"] = mean(item[f"{name}_help"])
            summary[f"help_{name}_std"] = std(item[f"{name}_help"])
            summary[f"{name}_auroc"] = max(auc_high, auc_low)
            summary[f"{name}_direction"] = "high" if auc_high >= auc_low else "low"
            summary[f"{name}_raw_auroc_high"] = auc_high
        out.append(summary)
    return out


def maybe_plot(summary_rows: Sequence[Mapping[str, Any]], out_png: str, title: str) -> None:
    if not str(out_png or "").strip():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable; skipping plot: {exc}", flush=True)
        return
    xs = [int(r["layer_index"]) for r in summary_rows]
    harm = [float(r["harm_candidate_margin_mean"]) for r in summary_rows]
    help_ = [float(r["help_candidate_margin_mean"]) for r in summary_rows]
    auc = [float(r["candidate_margin_auroc"]) for r in summary_rows]
    if not xs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=180)
    axes[0].plot(xs, harm, color="#c0392b", linewidth=2.0, label="Harm")
    axes[0].plot(xs, help_, color="#2468d8", linewidth=2.0, label="Help")
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Candidate yes/no margin")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(xs, auc, color="#222222", linewidth=2.0)
    axes[1].axhline(0.5, color="#777777", linewidth=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Harm AUROC")
    axes[1].set_ylim(0.45, min(1.0, max(0.75, max(auc) + 0.05)))
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(os.path.abspath(out_png), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract layer-wise first-answer yes/no decision margins under teacher forcing.")
    ap.add_argument("--runtime_backend", choices=["llava15_cleanroom", "llava_next_official"], required=True)
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument(
        "--intervention_pred_jsonl",
        default="",
        help="Optional prediction jsonl. If omitted or missing an id, label_rows_csv intervention_text is used.",
    )
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--label_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llava_next_root", default="/home/kms/LLaVA-NeXT")
    ap.add_argument("--llava_next_torch_type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--llava_next_attn_implementation", default="sdpa", choices=["none", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--apply_final_norm", type=parse_bool, default=True)
    ap.add_argument("--only_label_rows", type=parse_bool, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    long_csv = os.path.join(out_dir, "layer_margin_trajectory_long.csv")
    summary_csv = os.path.join(out_dir, "layer_margin_trajectory_summary.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    plot_png = os.path.join(out_dir, "layer_margin_trajectory.png")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json)
        return

    labels = read_label_rows(os.path.abspath(args.label_rows_csv))
    label_ids = set(labels)
    questions = read_jsonl_rows(os.path.abspath(args.question_file), limit=0)
    if bool(args.only_label_rows):
        questions = [row for row in questions if safe_id(row.get("question_id", row.get("id"))) in label_ids]
    if int(args.limit) > 0:
        questions = questions[: int(args.limit)]
    pred_map = (
        read_jsonl_text_map(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_key))
        if str(args.intervention_pred_jsonl or "").strip()
        else {}
    )

    if str(args.runtime_backend) == "llava_next_official":
        from frgavr_cleanroom.llava_next_runtime import OfficialLlavaNextRuntime

        runtime = OfficialLlavaNextRuntime(
            llava_next_root=str(args.llava_next_root),
            model_path=str(args.model_path),
            model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
            conv_mode=str(args.conv_mode),
            device=str(args.device),
            torch_type=str(args.llava_next_torch_type),
            attn_implementation=str(args.llava_next_attn_implementation),
        )
    else:
        from frgavr_cleanroom.runtime import CleanroomLlavaRuntime

        runtime = CleanroomLlavaRuntime(
            model_path=str(args.model_path),
            model_base=(None if not str(args.model_base).strip() else str(args.model_base)),
            conv_mode=str(args.conv_mode),
            device=str(args.device),
        )

    long_rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_missing_intervention = 0
    timings: List[float] = []
    for idx, sample in enumerate(questions):
        sid = safe_id(sample.get("question_id", sample.get("id")))
        image_name = str(sample.get("image", "")).strip()
        question = str(sample.get("text", sample.get("question", ""))).strip()
        label_meta = labels.get(sid, {})
        candidate_text = str(pred_map.get(sid, "") or label_meta.get("intervention_text", "")).strip()
        meta = {
            "id": sid,
            "image": image_name,
            "question": question,
            "intervention_text": candidate_text,
            **label_meta,
        }
        try:
            if not sid:
                raise ValueError("missing sample id")
            if not image_name:
                raise ValueError("missing image")
            if not question:
                raise ValueError("missing question")
            if not candidate_text:
                n_missing_intervention += 1
                raise ValueError("missing intervention prediction")
            image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)
            t0 = time.perf_counter()
            image = runtime.load_image(image_path)
            traj = layer_trajectory(
                runtime,
                image=image,
                question=question,
                candidate_text=candidate_text,
                apply_final_norm=bool(args.apply_final_norm),
            )
            timings.append(time.perf_counter() - t0)
            for row in traj:
                long_rows.append({**meta, **row, "score_error": ""})
        except Exception as exc:
            n_errors += 1
            err = {**meta, "layer_index": "", "score_error": str(exc), "score_error_traceback": traceback.format_exc()}
            long_rows.append(err)
            print(f"[error] id={sid} {exc!r}", flush=True)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[layer-trajectory] {idx + 1}/{len(questions)}", flush=True)

    summary_rows = summarize_layers(long_rows)
    write_csv(long_csv, long_rows)
    write_csv(summary_csv, summary_rows)
    maybe_plot(summary_rows, plot_png, title=str(args.runtime_backend))
    write_json(
        summary_json,
        {
            "mode": "decision_margin_layer_trajectory",
            "inputs": {
                "runtime_backend": str(args.runtime_backend),
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl)
                if str(args.intervention_pred_jsonl or "").strip()
                else "",
                "intervention_pred_key": str(args.intervention_pred_key),
                "label_rows_csv": os.path.abspath(args.label_rows_csv),
                "model_path": str(args.model_path),
                "conv_mode": str(args.conv_mode),
                "apply_final_norm": bool(args.apply_final_norm),
                "only_label_rows": bool(args.only_label_rows),
            },
            "counts": {
                "n_question_rows": int(len(questions)),
                "n_label_rows": int(len(labels)),
                "n_long_rows": int(len(long_rows)),
                "n_errors": int(n_errors),
                "n_missing_intervention": int(n_missing_intervention),
            },
            "timing": {
                "feature_total_sec": float(sum(timings)),
                "feature_mean_ms": float(1000.0 * sum(timings) / max(1, len(timings))),
            },
            "outputs": {
                "long_csv": long_csv,
                "summary_csv": summary_csv,
                "plot_png": plot_png,
            },
        },
    )
    print("[saved]", long_csv)
    print("[saved]", summary_csv)
    print("[saved]", summary_json)
    print("[saved]", plot_png)


if __name__ == "__main__":
    main()
