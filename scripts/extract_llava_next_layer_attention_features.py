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
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

SUMMARY_FEATURES = [
    "attn_first_visual_mass_mean",
    "attn_first_visual_mass_max_head",
    "attn_first_topk_visual_frac_mean",
    "attn_content_visual_mass_mean",
    "attn_content_visual_mass_min_token",
    "attn_content_visual_mass_max_token",
    "attn_content_topk_visual_frac_mean",
    "attn_tail_visual_mass_mean",
    "attn_tail_topk_visual_frac_mean",
]


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


def maybe_float(value: Any) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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
        if text_key == "auto":
            for key in ("output", "text", "answer", "caption"):
                value = str(row.get(key, "")).strip()
                if value:
                    out[sid] = value
                    break
        else:
            out[sid] = str(row.get(text_key, "")).strip()
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


def tensor_indices(values: Any, upper: int) -> List[int]:
    out: List[int] = []
    for item in values.tolist():
        idx = int(item)
        if 0 <= idx < int(upper):
            out.append(idx)
    return out


def topk_visual_fraction(attn_query: Any, vision_mask: Any, top_k: int) -> float:
    import torch

    if int(top_k) <= 0:
        return 0.0
    k = min(int(top_k), int(attn_query.shape[-1]))
    if k <= 0:
        return 0.0
    top_idx = torch.topk(attn_query, k=k, dim=-1).indices
    visual_hits = vision_mask.index_select(0, top_idx.reshape(-1)).reshape(top_idx.shape)
    return float(visual_hits.float().mean().item())


def summarize_attention(attn: Any, query_positions: Sequence[int], vision_positions: Sequence[int], top_k: int) -> Dict[str, float]:
    import torch

    # attn shape: [batch, heads, query_len, key_len]
    if attn.dim() != 4 or int(attn.shape[0]) != 1:
        raise RuntimeError(f"Unexpected attention shape: {tuple(attn.shape)}")
    q_len = int(attn.shape[-2])
    k_len = int(attn.shape[-1])
    q_idx = [int(x) for x in query_positions if 0 <= int(x) < q_len]
    v_idx = [int(x) for x in vision_positions if 0 <= int(x) < k_len]
    if not q_idx or not v_idx:
        return {
            "visual_mass_mean": 0.0,
            "visual_mass_std": 0.0,
            "visual_mass_min_token": 0.0,
            "visual_mass_max_token": 0.0,
            "visual_mass_max_head": 0.0,
            "topk_visual_frac_mean": 0.0,
        }
    q = torch.tensor(q_idx, dtype=torch.long, device=attn.device)
    v = torch.tensor(v_idx, dtype=torch.long, device=attn.device)
    # Runtime stores attentions on CPU in fp16 to save memory. CPU topk does not
    # support half, so cast only the queried rows instead of the full matrix.
    selected = attn[0].index_select(1, q).float()
    visual_mass = selected.index_select(2, v).sum(dim=-1).float()
    # visual_mass: [heads, queries]
    token_mean = visual_mass.mean(dim=0)
    head_mean = visual_mass.mean(dim=1)
    vision_mask = torch.zeros(k_len, dtype=torch.bool, device=attn.device)
    vision_mask[v] = True
    topk_vals = [
        topk_visual_fraction(selected[:, qi, :], vision_mask, int(top_k))
        for qi in range(int(selected.shape[1]))
    ]
    return {
        "visual_mass_mean": float(visual_mass.mean().item()),
        "visual_mass_std": float(visual_mass.std(unbiased=False).item()),
        "visual_mass_min_token": float(token_mean.min().item()),
        "visual_mass_max_token": float(token_mean.max().item()),
        "visual_mass_max_head": float(head_mean.max().item()),
        "topk_visual_frac_mean": mean(topk_vals),
    }


def attention_trajectory(
    runtime: Any,
    *,
    image: Any,
    question: str,
    candidate_text: str,
    top_k: int,
    tail_fraction: float,
) -> List[Dict[str, Any]]:
    from frgavr_cleanroom.runtime import select_content_indices

    pack = runtime.teacher_force_candidate(
        image=image,
        question=question,
        candidate_text=candidate_text,
        output_attentions=True,
        output_hidden_states=False,
    )
    if pack.attentions is None:
        raise RuntimeError("Forward did not return attentions. Use eager attention implementation.")

    decision_positions = [int(x) for x in pack.decision_positions.tolist()]
    if not decision_positions:
        raise RuntimeError("No decision positions found.")
    content_indices = select_content_indices(runtime.tokenizer, pack.cont_ids)
    content_decisions = [
        decision_positions[int(i)]
        for i in content_indices
        if 0 <= int(i) < len(decision_positions)
    ]
    if not content_decisions:
        content_decisions = list(decision_positions)
    n_tail = max(1, int(math.ceil(len(content_decisions) * max(0.0, min(1.0, float(tail_fraction))))))
    first_decisions = [decision_positions[0]]
    tail_decisions = content_decisions[-n_tail:]
    vision_positions = [int(x) for x in pack.vision_positions.tolist()]

    rows: List[Dict[str, Any]] = []
    n_layers = len(pack.attentions)
    for idx, attn in enumerate(pack.attentions):
        first = summarize_attention(attn, first_decisions, vision_positions, top_k)
        content = summarize_attention(attn, content_decisions, vision_positions, top_k)
        tail = summarize_attention(attn, tail_decisions, vision_positions, top_k)
        rows.append(
            {
                "layer_index": int(idx),
                "layer_frac": float(idx / max(1, n_layers - 1)),
                "is_final_layer": int(idx == n_layers - 1),
                "attn_first_visual_mass_mean": first["visual_mass_mean"],
                "attn_first_visual_mass_max_head": first["visual_mass_max_head"],
                "attn_first_visual_mass_std": first["visual_mass_std"],
                "attn_first_topk_visual_frac_mean": first["topk_visual_frac_mean"],
                "attn_content_visual_mass_mean": content["visual_mass_mean"],
                "attn_content_visual_mass_min_token": content["visual_mass_min_token"],
                "attn_content_visual_mass_max_token": content["visual_mass_max_token"],
                "attn_content_visual_mass_max_head": content["visual_mass_max_head"],
                "attn_content_visual_mass_std": content["visual_mass_std"],
                "attn_content_topk_visual_frac_mean": content["topk_visual_frac_mean"],
                "attn_tail_visual_mass_mean": tail["visual_mass_mean"],
                "attn_tail_visual_mass_max_head": tail["visual_mass_max_head"],
                "attn_tail_visual_mass_std": tail["visual_mass_std"],
                "attn_tail_topk_visual_frac_mean": tail["topk_visual_frac_mean"],
                "attn_n_decision_tokens": int(len(decision_positions)),
                "attn_n_content_tokens": int(len(content_decisions)),
                "attn_n_tail_tokens": int(len(tail_decisions)),
                "attn_n_vision_tokens": int(len(vision_positions)),
            }
        )
    return rows


def summarize_layers(long_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_layer: Dict[int, Dict[str, Any]] = {}
    for row in long_rows:
        layer_f = maybe_float(row.get("layer_index"))
        harm_f = maybe_float(row.get("harm"))
        help_f = maybe_float(row.get("help"))
        if layer_f is None or harm_f is None or help_f is None:
            continue
        layer = int(layer_f)
        harm = int(harm_f)
        help_ = int(help_f)
        if harm not in {0, 1} or help_ not in {0, 1} or (harm == 0 and help_ == 0):
            continue
        item = by_layer.setdefault(
            layer,
            {
                "ys": [],
                **{f"{name}_values": [] for name in SUMMARY_FEATURES},
                **{f"{name}_harm": [] for name in SUMMARY_FEATURES},
                **{f"{name}_help": [] for name in SUMMARY_FEATURES},
            },
        )
        item["ys"].append(harm)
        for name in SUMMARY_FEATURES:
            value = maybe_float(row.get(name))
            if value is None:
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
        for name in SUMMARY_FEATURES:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract layer-wise answer-to-vision attention features for LLaVA-NeXT replay.")
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--intervention_pred_key", default="auto")
    ap.add_argument("--label_rows_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default="/home/kms/models/llama3-llava-next-8b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_llama_3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llava_next_root", default="/home/kms/LLaVA-NeXT")
    ap.add_argument("--llava_next_torch_type", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--llava_next_attn_implementation", default="eager", choices=["none", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--only_label_rows", type=parse_bool, default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--tail_fraction", type=float, default=0.25)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    long_csv = os.path.join(out_dir, "layer_attention_trajectory_long.csv")
    summary_csv = os.path.join(out_dir, "layer_attention_trajectory_summary.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json):
        print("[reuse]", summary_json, flush=True)
        return

    labels = read_label_rows(os.path.abspath(args.label_rows_csv))
    label_ids = set(labels)
    questions = read_jsonl_rows(os.path.abspath(args.question_file), limit=0)
    if bool(args.only_label_rows):
        questions = [row for row in questions if safe_id(row.get("question_id", row.get("id"))) in label_ids]
    if int(args.limit) > 0:
        questions = questions[: int(args.limit)]
    pred_map = read_jsonl_text_map(os.path.abspath(args.intervention_pred_jsonl), str(args.intervention_pred_key))

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
            image = runtime.load_image(image_path)
            t0 = time.perf_counter()
            rows = attention_trajectory(
                runtime,
                image=image,
                question=question,
                candidate_text=candidate_text,
                top_k=int(args.top_k),
                tail_fraction=float(args.tail_fraction),
            )
            timings.append(time.perf_counter() - t0)
            for row in rows:
                long_rows.append({**meta, **row, "score_error": ""})
        except Exception as exc:
            n_errors += 1
            err = {**meta, "layer_index": "", "score_error": str(exc), "score_error_traceback": traceback.format_exc()}
            long_rows.append(err)
            print(f"[error] id={sid} {exc!r}", flush=True)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[layer-attn] {idx + 1}/{len(questions)}", flush=True)

    summary_rows = summarize_layers(long_rows)
    write_csv(long_csv, long_rows)
    write_csv(summary_csv, summary_rows)
    write_json(
        summary_json,
        {
            "mode": "llava_next_layer_attention_replay",
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                "intervention_pred_key": str(args.intervention_pred_key),
                "label_rows_csv": os.path.abspath(args.label_rows_csv),
                "model_path": str(args.model_path),
                "conv_mode": str(args.conv_mode),
                "llava_next_root": str(args.llava_next_root),
                "llava_next_attn_implementation": str(args.llava_next_attn_implementation),
                "top_k": int(args.top_k),
                "tail_fraction": float(args.tail_fraction),
                "only_label_rows": bool(args.only_label_rows),
            },
            "counts": {
                "n_rows": int(len(questions)),
                "n_errors": int(n_errors),
                "n_missing_intervention": int(n_missing_intervention),
                "n_long_rows": int(len(long_rows)),
                "n_summary_layers": int(len(summary_rows)),
            },
            "timing": {
                "feature_total_sec": float(sum(timings)),
                "feature_mean_ms": float(1000.0 * sum(timings) / max(1, len(timings))),
            },
            "outputs": {
                "long_csv": long_csv,
                "summary_csv": summary_csv,
            },
        },
    )
    print("[saved]", long_csv)
    print("[saved]", summary_csv)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
