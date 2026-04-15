#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(os.path.abspath(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_id(row: Dict[str, Any]) -> str:
    raw = str(row.get("question_id") or row.get("image_id") or row.get("id") or "").strip()
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def pick_text(row: Dict[str, Any], key: str) -> str:
    if key != "auto":
        return str(row.get(key, "")).strip()
    for cand in ("output", "text", "caption", "answer", "prediction"):
        text = str(row.get(cand, "")).strip()
        if text:
            return text
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge selected re-caption predictions into a full prediction JSONL.")
    ap.add_argument("--base_pred_jsonl", required=True, help="Default predictions, usually intervention captions.")
    ap.add_argument("--repair_pred_jsonl", required=True, help="Re-caption predictions for selected samples.")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--base_text_key", default="auto")
    ap.add_argument("--repair_text_key", default="auto")
    args = ap.parse_args()

    repairs = {safe_id(row): row for row in read_jsonl(args.repair_pred_jsonl) if safe_id(row)}
    out_rows: List[Dict[str, Any]] = []
    n_repaired = 0
    for row in read_jsonl(args.base_pred_jsonl):
        sid = safe_id(row)
        out = dict(row)
        repair = repairs.get(sid)
        if repair is not None:
            text = pick_text(repair, args.repair_text_key)
            out["text"] = text
            out["output"] = text
            out["caption"] = text
            out["repair_source"] = "negative_object_recaption"
            out["negative_objects"] = repair.get("negative_objects", "")
            n_repaired += 1
        else:
            text = pick_text(row, args.base_text_key)
            out["text"] = text
            out.setdefault("output", text)
            out.setdefault("caption", text)
            out["repair_source"] = "original"
        out_rows.append(out)

    write_jsonl(args.out_jsonl, out_rows)
    print("[saved]", os.path.abspath(args.out_jsonl))
    if str(args.out_summary_json or "").strip():
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "base_pred_jsonl": os.path.abspath(args.base_pred_jsonl),
                    "repair_pred_jsonl": os.path.abspath(args.repair_pred_jsonl),
                },
                "counts": {
                    "n_base_rows": len(out_rows),
                    "n_repair_rows": len(repairs),
                    "n_repaired": n_repaired,
                },
                "outputs": {"out_jsonl": os.path.abspath(args.out_jsonl)},
            },
        )
        print("[saved]", os.path.abspath(args.out_summary_json))


if __name__ == "__main__":
    main()
