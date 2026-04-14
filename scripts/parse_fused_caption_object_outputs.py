#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple


CAPTION_RE = re.compile(r"(?i)\bcaption\s*:\s*")
OBJECTS_RE = re.compile(r"(?i)\bobjects?\s*:\s*")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def get_text(row: Dict[str, Any]) -> str:
    for key in ("output", "text", "answer", "caption"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def clean_field(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\s*\[[^\]]*?here\]\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\[(.*)\]\s*$", r"\1", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_fused_text(raw: str) -> Tuple[str, str, bool, str]:
    raw = str(raw or "").strip()
    if not raw:
        return "", "", False, "empty"

    cap_match = CAPTION_RE.search(raw)
    obj_match = OBJECTS_RE.search(raw)
    if cap_match and obj_match:
        if cap_match.end() <= obj_match.start():
            caption = raw[cap_match.end() : obj_match.start()]
            objects = raw[obj_match.end() :]
        else:
            caption = raw[: obj_match.start()]
            objects = raw[obj_match.end() : cap_match.start()]
        return clean_field(caption), clean_field(objects), True, "caption_objects"

    if obj_match:
        caption = raw[: obj_match.start()]
        objects = raw[obj_match.end() :]
        caption = CAPTION_RE.sub("", caption, count=1)
        return clean_field(caption), clean_field(objects), bool(caption.strip()), "objects_only_marker"

    if cap_match:
        caption = raw[cap_match.end() :]
        return clean_field(caption), "", False, "caption_only_marker"

    # Conservative fallback: keep the whole output as the caption so CHAIR can
    # still reveal whether the fused prompt itself damaged the caption metric.
    return clean_field(raw), "", False, "no_markers"


def passthrough_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "question_id",
        "id",
        "image_id",
        "image",
        "split",
        "category",
        "prompt",
        "question",
    ):
        if key in row:
            out[key] = row[key]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse fused Caption:/Objects: generations into separate jsonl files.")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_caption_jsonl", required=True)
    ap.add_argument("--out_objects_jsonl", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--preview_limit", type=int, default=20)
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    cap_rows: List[Dict[str, Any]] = []
    obj_rows: List[Dict[str, Any]] = []
    parse_modes: Dict[str, int] = {}
    failures: List[Dict[str, Any]] = []

    for row in rows:
        raw = get_text(row)
        caption, objects, ok, mode = parse_fused_text(raw)
        parse_modes[mode] = parse_modes.get(mode, 0) + 1
        meta = passthrough_meta(row)

        cap_row = dict(meta)
        cap_row.update(
            {
                "output": caption,
                "text": caption,
                "raw_output": raw,
                "objects_text": objects,
                "fused_parse_ok": int(bool(ok)),
                "fused_parse_mode": mode,
            }
        )
        cap_rows.append(cap_row)

        obj_row = dict(meta)
        obj_row.update(
            {
                "output": objects,
                "text": objects,
                "caption_text": caption,
                "raw_output": raw,
                "fused_parse_ok": int(bool(ok)),
                "fused_parse_mode": mode,
            }
        )
        obj_rows.append(obj_row)

        if not ok and len(failures) < int(args.preview_limit):
            failures.append(
                {
                    "question_id": str(row.get("question_id", row.get("id", ""))),
                    "image_id": str(row.get("image_id", "")),
                    "mode": mode,
                    "raw_output": raw[:500],
                    "caption": caption[:300],
                    "objects": objects[:300],
                }
            )

    write_jsonl(args.out_caption_jsonl, cap_rows)
    write_jsonl(args.out_objects_jsonl, obj_rows)
    summary = {
        "inputs": {"in_jsonl": os.path.abspath(args.in_jsonl)},
        "counts": {
            "n_rows": len(rows),
            "n_caption_nonempty": sum(1 for row in cap_rows if str(row.get("output", "")).strip()),
            "n_objects_nonempty": sum(1 for row in obj_rows if str(row.get("output", "")).strip()),
            "n_parse_ok": sum(int(row.get("fused_parse_ok", 0)) for row in cap_rows),
            "parse_modes": parse_modes,
        },
        "failures_preview": failures,
        "outputs": {
            "caption_jsonl": os.path.abspath(args.out_caption_jsonl),
            "objects_jsonl": os.path.abspath(args.out_objects_jsonl),
            "summary_json": os.path.abspath(args.out_summary_json),
        },
    }
    write_json(args.out_summary_json, summary)
    print("[saved]", os.path.abspath(args.out_caption_jsonl))
    print("[saved]", os.path.abspath(args.out_objects_jsonl))
    print("[saved]", os.path.abspath(args.out_summary_json))
    print("[parse]", json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
