#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ID_KEYS = ("question_id", "id", "image_id", "qid")
TEXT_KEYS = ("text", "output", "caption", "answer", "response")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row_id(row)
            if not sid:
                sid = str(idx)
            out[str(sid)] = row
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def row_id(row: Dict[str, Any]) -> str:
    for key in ID_KEYS:
        value = str(row.get(key, "")).strip()
        if value:
            return normalize_id(value)
    image = str(row.get("image", "") or row.get("file_name", "")).strip()
    return normalize_id(image)


def normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(\d{1,12})(?:\.\w+)?$", text)
    if match:
        return str(int(match.group(1)))
    return text


def image_file_from_any(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    name = Path(text).name
    if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return name
    match = re.search(r"(\d{1,12})$", name)
    if match:
        return f"COCO_val2014_{int(match.group(1)):012d}.jpg"
    return name


def caption_text(row: Dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def canonical_object(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[-1]).strip().lower()
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            obj = json.loads(text.replace("'", '"'))
            if isinstance(obj, list) and obj:
                return str(obj[-1]).strip().lower()
        except Exception:
            pass
    return text.lower()


def object_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        if not values.strip():
            return []
        parts = re.split(r"\s*\|\s*|\s*,\s*", values)
        return [canonical_object(p) for p in parts if canonical_object(p)]
    if isinstance(values, Iterable):
        return [canonical_object(v) for v in values if canonical_object(v)]
    return []


def chair_sentence_id(row: Dict[str, Any]) -> str:
    for key in ("question_id", "id", "image_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return normalize_id(value)
    return ""


def load_chair_sentence_map(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = read_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in obj.get("sentences", []):
        sid = chair_sentence_id(row)
        if sid:
            out[sid] = row
    return out


def hallucinated_objects(row: Dict[str, Any]) -> List[str]:
    values = object_list(row.get("mscoco_hallucinated_words"))
    if values:
        return values
    generated = object_list(row.get("mscoco_generated_words"))
    gt = set(object_list(row.get("mscoco_gt_words")))
    return [obj for obj in generated if obj and obj not in gt]


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


def object_in_sentence(sentence: str, obj: str) -> bool:
    obj = str(obj or "").strip().lower()
    if not obj:
        return False
    pattern = r"(?<![A-Za-z])" + re.escape(obj) + r"s?(?![A-Za-z])"
    return re.search(pattern, sentence.lower()) is not None


def sentence_similarity(a: str, b: str) -> float:
    wa = {w for w in re.findall(r"[A-Za-z]+", a.lower()) if len(w) > 2}
    wb = {w for w in re.findall(r"[A-Za-z]+", b.lower()) if len(w) > 2}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / float(len(wa | wb))


def choose_sentence_pair(method_caption: str, repaired_caption: str, removed: Sequence[str]) -> Tuple[str, str]:
    before_sentences = split_sentences(method_caption)
    after_sentences = split_sentences(repaired_caption)
    before = ""
    for obj in removed:
        for sent in before_sentences:
            if object_in_sentence(sent, obj):
                before = sent
                break
        if before:
            break
    if not before and before_sentences:
        before = before_sentences[0]
    after = ""
    if before and after_sentences:
        candidates = [
            sent
            for sent in after_sentences
            if not any(object_in_sentence(sent, obj) for obj in removed)
        ]
        if not candidates:
            candidates = after_sentences
        after = max(candidates, key=lambda sent: sentence_similarity(before, sent))
    return before, after


def markdown(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Rank | Image | Removed hallucination | Before c_M | After c_R |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{image_abs_path}` | {removed_hallucinations} | {before_sentence} | {after_sentence} |".format(
                **{k: str(v).replace("\n", " ").replace("|", "\\|") for k, v in row.items()}
            )
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract CHAIR-improved generative before/after samples for appendix tables."
    )
    ap.add_argument("--method_pred_jsonl", required=True)
    ap.add_argument("--repaired_pred_jsonl", required=True)
    ap.add_argument("--method_chair_json", required=True)
    ap.add_argument("--repaired_chair_json", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_md", default="")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--require_no_added_hallucination", action="store_true")
    args = ap.parse_args()

    method_pred = read_jsonl_map(Path(args.method_pred_jsonl))
    repaired_pred = read_jsonl_map(Path(args.repaired_pred_jsonl))
    method_chair = load_chair_sentence_map(Path(args.method_chair_json))
    repaired_chair = load_chair_sentence_map(Path(args.repaired_chair_json))
    image_folder = Path(args.image_folder)

    rows: List[Dict[str, Any]] = []
    common_ids = sorted(set(method_chair) & set(repaired_chair), key=lambda x: int(x) if x.isdigit() else x)
    for sid in common_ids:
        m_chair = method_chair[sid]
        r_chair = repaired_chair[sid]
        m_hall_list = hallucinated_objects(m_chair)
        r_hall_list = hallucinated_objects(r_chair)
        m_hall = set(m_hall_list)
        r_hall = set(r_hall_list)
        removed = sorted(m_hall - r_hall)
        added = sorted(r_hall - m_hall)
        if not removed:
            continue
        if args.require_no_added_hallucination and added:
            continue
        m_pred = method_pred.get(sid, {})
        r_pred = repaired_pred.get(sid, {})
        method_caption = caption_text(m_pred) or str(m_chair.get("caption", "")).strip()
        repaired_caption = caption_text(r_pred) or str(r_chair.get("caption", "")).strip()
        if method_caption.strip() == repaired_caption.strip():
            continue
        before_sentence, after_sentence = choose_sentence_pair(method_caption, repaired_caption, removed)
        image_name = (
            image_file_from_any(m_pred.get("image"))
            or image_file_from_any(r_pred.get("image"))
            or image_file_from_any(m_chair.get("image_id"))
            or image_file_from_any(sid)
        )
        image_abs = str((image_folder / image_name).resolve()) if image_name else ""
        removed_inst = len([obj for obj in m_hall_list if obj in set(removed)])
        added_inst = len([obj for obj in r_hall_list if obj in set(added)])
        score = 100 * len(removed) + 10 * removed_inst - 50 * len(added) - 5 * added_inst
        rows.append(
            {
                "rank": 0,
                "score": score,
                "question_id": sid,
                "image": image_name,
                "image_abs_path": image_abs,
                "removed_hallucinations": ", ".join(removed),
                "added_hallucinations": ", ".join(added),
                "method_hallucinations": ", ".join(sorted(m_hall)),
                "repaired_hallucinations": ", ".join(sorted(r_hall)),
                "before_sentence": before_sentence,
                "after_sentence": after_sentence,
                "method_caption": method_caption,
                "repaired_caption": repaired_caption,
            }
        )

    rows.sort(key=lambda r: (-int(r["score"]), int(r["question_id"]) if str(r["question_id"]).isdigit() else str(r["question_id"])))
    rows = rows[: max(1, int(args.top_k))]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows)
    print("[saved]", out_csv)
    if args.out_json:
        out_json = Path(args.out_json)
        write_json(out_json, {"n": len(rows), "rows": rows})
        print("[saved]", out_json)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown(rows) + "\n", encoding="utf-8")
        print("[saved]", out_md)


if __name__ == "__main__":
    main()
