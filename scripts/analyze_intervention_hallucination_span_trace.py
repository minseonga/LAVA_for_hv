#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import CleanroomLlavaRuntime
from scripts.extract_vga_generative_mention_features import build_feature_payload


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                cols.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in cols})


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def canonical_object(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[-1]).strip().lower()
    return str(value).strip().lower()


def hallucinated_object(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0]).strip().lower()
    return str(value).strip().lower()


def image_id_from_value(value: Any) -> str:
    raw = str(value or "").strip()
    digits = re.findall(r"(\d+)", raw)
    if digits:
        return str(int(digits[-1]))
    return raw


def safe_id(value: object) -> str:
    return str(value or "").strip()


def pick_text(row: Dict[str, Any], key: str) -> str:
    if key != "auto":
        return str(row.get(key, "")).strip()
    for cand in ("output", "text", "caption", "answer", "prediction"):
        text = str(row.get(cand, "")).strip()
        if text:
            return text
    return ""


def load_prediction_text_map(path: str, text_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl(path):
        sample_id = safe_id(row.get("question_id", row.get("id")))
        if sample_id:
            out[sample_id] = pick_text(row, text_key)
    return out


def load_question_map(path: str, limit: int) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    if int(limit) > 0:
        rows = rows[: int(limit)]
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        iid = image_id_from_value(row.get("image_id", row.get("image", row.get("question_id", row.get("id")))))
        if iid:
            out[iid] = row
    return out


def load_chair_map(path: str) -> Dict[str, Dict[str, Any]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for sent in obj.get("sentences", []):
        iid = image_id_from_value(sent.get("image_id"))
        gen_list = [canonical_object(x) for x in sent.get("mscoco_generated_words", [])]
        gen_list = [x for x in gen_list if x]
        gt_set = {canonical_object(x) for x in sent.get("mscoco_gt_words", []) if canonical_object(x)}
        hall_set = {hallucinated_object(x) for x in sent.get("mscoco_hallucinated_words", []) if hallucinated_object(x)}
        out[iid] = {
            **sent,
            "_generated_set": set(gen_list),
            "_generated_list": gen_list,
            "_gt_set": gt_set,
            "_hall_set": hall_set,
        }
    return out


def words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(str(text or ""))]


def singularize_word(word: str) -> str:
    word = str(word or "").lower()
    irregular = {
        "children": "child",
        "feet": "foot",
        "geese": "goose",
        "knives": "knife",
        "men": "man",
        "mice": "mouse",
        "people": "person",
        "teeth": "tooth",
        "women": "woman",
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ves"):
        return word[:-3] + "f"
    if len(word) > 3 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def word_matches(a: str, b: str) -> bool:
    return singularize_word(a) == singularize_word(b)


def contains_object(mention_text: str, obj: str) -> bool:
    mt = str(mention_text or "").lower()
    oo = str(obj or "").lower()
    if not oo:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(oo) + r"(?![A-Za-z0-9])"
    if re.search(pattern, mt):
        return True
    obj_words = words(oo)
    mention_words = words(mt)
    return bool(obj_words) and all(any(word_matches(word, cand) for cand in mention_words) for word in obj_words)


def match_object_to_mention(obj: str, mention_rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    matches: List[Tuple[int, int, Dict[str, Any]]] = []
    for row in mention_rows:
        text = str(row.get("text", ""))
        if not contains_object(text, obj):
            continue
        is_object = int("object_mention" in str(row.get("kinds", "")).split("|"))
        span_len = len(words(text))
        matches.append((-is_object, span_len, row))
    if not matches:
        return None
    matches.sort(key=lambda x: (x[0], x[1]))
    return matches[0][2]


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / float(len(values))) if values else 0.0


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(float(x) for x in values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def auc_high(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    good = 0.0
    total = 0
    for a in pos:
        for b in neg:
            total += 1
            if a > b:
                good += 1.0
            elif a == b:
                good += 0.5
    return float(good / float(total)) if total else None


def metric_summary(rows: Sequence[Dict[str, Any]], group: str) -> Dict[str, Any]:
    subset = [row for row in rows if row.get("span_group") == group and int(row.get("matched", 0)) == 1]
    out: Dict[str, Any] = {
        "n": int(len(subset)),
        "objects": Counter(str(row.get("object", "")) for row in subset).most_common(20),
    }
    for key in ("lp_min", "gap_min", "ent_max", "first_idx", "last_idx", "n_tokens"):
        vals: List[float] = []
        for row in subset:
            try:
                val = float(row.get(key, ""))
            except Exception:
                continue
            if math.isfinite(val):
                vals.append(val)
        out[key] = {
            "mean": mean(vals),
            "median": median(vals),
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare teacher-forced span trace for intervention-only hallucinated vs supported object spans."
    )
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--baseline_chair_json", type=str, required=True)
    ap.add_argument("--intervention_chair_json", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--pred_text_key", type=str, default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_mentions", type=int, default=128)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    question_map = load_question_map(os.path.abspath(args.question_file), int(args.limit))
    baseline = load_chair_map(os.path.abspath(args.baseline_chair_json))
    intervention = load_chair_map(os.path.abspath(args.intervention_chair_json))
    pred_text = load_prediction_text_map(os.path.abspath(args.intervention_pred_jsonl), str(args.pred_text_key))

    runtime = CleanroomLlavaRuntime(
        model_path=str(args.model_path),
        model_base=str(args.model_base or "") or None,
        conv_mode=str(args.conv_mode),
        device=str(args.device),
    )

    span_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    image_ids = [iid for iid in question_map.keys() if iid in baseline and iid in intervention]

    for iid in tqdm(image_ids):
        qrow = question_map[iid]
        sample_id = safe_id(qrow.get("question_id", qrow.get("id", iid)))
        candidate_text = pred_text.get(sample_id, "")
        if not candidate_text:
            candidate_text = str(intervention[iid].get("caption", "")).strip()
        if not candidate_text:
            continue
        image_name = str(qrow.get("image", "")).strip()
        image_path = os.path.join(os.path.abspath(args.image_folder), image_name)
        try:
            payload = build_feature_payload(
                runtime=runtime,
                image_path=image_path,
                question=str(qrow.get("text", "")),
                candidate_text=candidate_text,
                sample_id=sample_id,
                image_name=image_name,
                max_mentions=int(args.max_mentions),
            )
        except Exception as exc:
            sample_rows.append(
                {
                    "image_id": iid,
                    "id": sample_id,
                    "image": image_name,
                    "error": repr(exc),
                }
            )
            continue

        mention_rows = payload["mention_rows"]
        base_gen = baseline[iid]["_generated_set"]
        int_gen = intervention[iid]["_generated_set"]
        gt = intervention[iid]["_gt_set"]
        int_only_hall = sorted(obj for obj in (int_gen - base_gen) if obj not in gt)
        supported = sorted(obj for obj in int_gen if obj in gt)
        int_only_supported = sorted(obj for obj in (int_gen - base_gen) if obj in gt)
        shared_hall = sorted(obj for obj in intervention[iid]["_hall_set"] if obj in baseline[iid]["_hall_set"])

        sample_rows.append(
            {
                "image_id": iid,
                "id": sample_id,
                "image": image_name,
                "n_int_only_hall": len(int_only_hall),
                "n_supported": len(supported),
                "n_int_only_supported": len(int_only_supported),
                "n_shared_hall": len(shared_hall),
                "int_only_hall_objects": "|".join(int_only_hall),
                "supported_objects": "|".join(supported),
                "int_caption": candidate_text,
            }
        )

        groups = [
            ("int_only_hallucination", int_only_hall),
            ("supported", supported),
            ("int_only_supported", int_only_supported),
            ("shared_hallucination", shared_hall),
        ]
        for span_group, objects in groups:
            for obj in objects:
                match = match_object_to_mention(obj, mention_rows)
                row: Dict[str, Any] = {
                    "image_id": iid,
                    "id": sample_id,
                    "image": image_name,
                    "span_group": span_group,
                    "object": obj,
                    "matched": int(match is not None),
                    "caption": candidate_text,
                }
                if match is not None:
                    for key in (
                        "text",
                        "kinds",
                        "n_tokens",
                        "first_idx",
                        "last_idx",
                        "lp_min",
                        "gap_min",
                        "ent_max",
                        "lp_tail_gap",
                    ):
                        row[key] = match.get(key, "")
                span_rows.append(row)

    span_csv = os.path.join(out_dir, "span_trace_rows.csv")
    sample_csv = os.path.join(out_dir, "sample_rows.csv")
    write_csv(span_csv, span_rows)
    write_csv(sample_csv, sample_rows)

    hall = [row for row in span_rows if row.get("span_group") == "int_only_hallucination" and int(row.get("matched", 0)) == 1]
    supp = [row for row in span_rows if row.get("span_group") == "supported" and int(row.get("matched", 0)) == 1]
    aucs: Dict[str, Any] = {}
    for key in ("lp_min", "gap_min", "ent_max", "lp_tail_gap"):
        hvals = [float(row[key]) for row in hall if str(row.get(key, "")).strip()]
        svals = [float(row[key]) for row in supp if str(row.get(key, "")).strip()]
        au = auc_high(hvals, svals)
        aucs[key] = {
            "auc_hall_high_vs_supported": au,
            "auc_hall_low_vs_supported": None if au is None else float(1.0 - au),
        }

    summary = {
        "inputs": {
            "question_file": os.path.abspath(args.question_file),
            "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
            "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "limit": int(args.limit),
            "max_mentions": int(args.max_mentions),
        },
        "counts": {
            "n_samples": int(len(sample_rows)),
            "n_span_rows": int(len(span_rows)),
            "n_int_only_hall_matched": int(len(hall)),
            "n_supported_matched": int(len(supp)),
            "n_int_only_hall_total": int(sum(1 for row in span_rows if row.get("span_group") == "int_only_hallucination")),
            "n_supported_total": int(sum(1 for row in span_rows if row.get("span_group") == "supported")),
        },
        "groups": {
            "int_only_hallucination": metric_summary(span_rows, "int_only_hallucination"),
            "supported": metric_summary(span_rows, "supported"),
            "int_only_supported": metric_summary(span_rows, "int_only_supported"),
            "shared_hallucination": metric_summary(span_rows, "shared_hallucination"),
        },
        "aucs": aucs,
        "outputs": {
            "span_trace_rows_csv": span_csv,
            "sample_rows_csv": sample_csv,
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
