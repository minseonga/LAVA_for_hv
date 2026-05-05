#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import runpy
import sys
import tempfile
from typing import List


PLACEHOLDER_MODEL_PATH = "/path/to/Qwen2.5-VL-7B-Instruct"


def rewrite_arg(args: List[str], key: str, value: str) -> List[str]:
    out: List[str] = []
    i = 0
    replaced = False
    while i < len(args):
        if args[i] == key:
            out.extend([key, value])
            i += 2
            replaced = True
        else:
            out.append(args[i])
            i += 1
    if not replaced:
        out.extend([key, value])
    return out


def drop_arg_with_value(args: List[str], key: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(args):
        if args[i] == key:
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out


def read_gt_metadata(path: str) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    if not path:
        return metadata
    with open(os.path.abspath(path), "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("id", row.get("question_id", ""))).strip()
            answer = str(row.get("answer", row.get("label", ""))).strip()
            image_id = str(row.get("image_id", "")).strip()
            if qid:
                metadata[qid] = {"label": answer, "image_id": image_id}
    return metadata


def infer_image_id(row: dict) -> str:
    image_id = str(row.get("image_id", "")).strip()
    if image_id:
        return image_id
    image_name = os.path.basename(str(row.get("image", "")).strip())
    stem, _ = os.path.splitext(image_name)
    return stem


def materialize_question_jsonl(path: str, limit: int, gt_csv: str = "") -> str:
    fd, out_path = tempfile.mkstemp(prefix="qwen25_vga_limit_", suffix=".jsonl")
    os.close(fd)
    gt_metadata = read_gt_metadata(gt_csv)
    n = 0
    with open(os.path.abspath(path), "r", encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", row.get("id", ""))).strip()
            gt_row = gt_metadata.get(qid, {})
            if gt_row and not str(row.get("label", "")).strip() and gt_row.get("label"):
                row["label"] = gt_row["label"]
            if not str(row.get("image_id", "")).strip():
                row["image_id"] = gt_row.get("image_id") or infer_image_id(row)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if n >= int(limit):
                break
    return out_path


def preload_greedy_sampler(vga_root: str, model_path: str) -> None:
    if vga_root not in sys.path:
        sys.path.insert(0, vga_root)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    original_from_pretrained = AutoTokenizer.from_pretrained

    def patched_from_pretrained(path, *args, **kwargs):
        if str(path) == PLACEHOLDER_MODEL_PATH:
            return tokenizer
        return original_from_pretrained(path, *args, **kwargs)

    AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        importlib.import_module("vcd_utils.greedy_sample_qwen2")
    finally:
        AutoTokenizer.from_pretrained = original_from_pretrained


def patch_generation_cache_position_compat() -> None:
    from transformers.generation.utils import GenerationMixin

    original = GenerationMixin._get_initial_cache_position
    if getattr(original, "_qwen25_vga_compat", False):
        return

    params = list(inspect.signature(original).parameters)
    if len(params) != 4:
        return

    def patched(self, *args, **kwargs):
        if len(args) == 2 and not kwargs:
            input_ids, model_kwargs = args
            seq_length = input_ids.shape[-1]
            device = input_ids.device
            return original(self, seq_length, device, model_kwargs)
        return original(self, *args, **kwargs)

    patched._qwen25_vga_compat = True  # type: ignore[attr-defined]
    GenerationMixin._get_initial_cache_position = patched


def patch_processor_pixel_bounds(min_pixels: int = 0, max_pixels: int = 0):
    if int(min_pixels) <= 0 and int(max_pixels) <= 0:
        return None
    from transformers import AutoProcessor

    original_from_pretrained = AutoProcessor.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        if int(min_pixels) > 0:
            kwargs["min_pixels"] = int(min_pixels)
        if int(max_pixels) > 0:
            kwargs["max_pixels"] = int(max_pixels)
        return original_from_pretrained(*args, **kwargs)

    AutoProcessor.from_pretrained = patched_from_pretrained
    return original_from_pretrained


def normalize_answers_file(path: str) -> None:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("question_id") is None:
                continue
            output = str(row.get("output", "")).strip()
            row.setdefault("text", output)
            row.setdefault("caption", output)
            rows.append(row)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VGA Qwen2.5-VL while injecting the real tokenizer into VGA_origin.")
    ap.add_argument("--vga-root", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--gt-csv", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_pixels", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=0)
    known, _ = ap.parse_known_args()

    vga_root = os.path.abspath(os.path.expanduser(known.vga_root))
    target = os.path.join(vga_root, "eval", "object_hallucination_vqa_qwen25-vl.py")
    if not os.path.isfile(target):
        raise FileNotFoundError(f"missing VGA Qwen2.5 runner: {target}")

    argv = sys.argv[1:]
    tmp_question_file = ""
    if int(known.limit) > 0 or str(known.gt_csv).strip():
        tmp_question_file = materialize_question_jsonl(known.question_file, int(known.limit) or 10**18, known.gt_csv)
        argv = rewrite_arg(argv, "--question-file", tmp_question_file)

    argv = drop_arg_with_value(argv, "--vga-root")
    argv = drop_arg_with_value(argv, "--gt-csv")
    argv = drop_arg_with_value(argv, "--limit")
    argv = drop_arg_with_value(argv, "--min_pixels")
    argv = drop_arg_with_value(argv, "--max_pixels")
    argv = rewrite_arg(argv, "--model-path", os.path.expanduser(known.model_path))

    preload_greedy_sampler(vga_root, os.path.expanduser(known.model_path))
    patch_generation_cache_position_compat()
    original_processor_from_pretrained = patch_processor_pixel_bounds(
        min_pixels=int(known.min_pixels),
        max_pixels=int(known.max_pixels),
    )

    old_argv = sys.argv
    try:
        sys.argv = [target] + argv
        runpy.run_path(target, run_name="__main__")
    finally:
        sys.argv = old_argv
        if original_processor_from_pretrained is not None:
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained = original_processor_from_pretrained
        if tmp_question_file:
            try:
                os.unlink(tmp_question_file)
            except OSError:
                pass

    normalize_answers_file(os.path.expanduser(known.answers_file))


if __name__ == "__main__":
    main()
