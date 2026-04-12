#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Sequence

from extract_chair_object_delta_yesno_features import add_prefix_stats, score_objects
from frgavr_cleanroom.runtime import CleanroomLlavaRuntime, load_question_rows, parse_bool, safe_id, write_csv, write_json


COCO_OBJECTS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


OBJECT_ALIASES: Dict[str, List[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
    "car": ["car", "cars", "automobile", "automobiles", "vehicle", "vehicles"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
    "airplane": ["airplane", "airplanes", "plane", "planes", "aircraft"],
    "bus": ["bus", "buses"],
    "train": ["train", "trains"],
    "truck": ["truck", "trucks"],
    "boat": ["boat", "boats", "ship", "ships"],
    "traffic light": ["traffic light", "traffic lights", "stoplight", "stoplights"],
    "fire hydrant": ["fire hydrant", "fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop sign", "stop signs"],
    "parking meter": ["parking meter", "parking meters"],
    "bench": ["bench", "benches"],
    "bird": ["bird", "birds"],
    "cat": ["cat", "cats"],
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "horse": ["horse", "horses"],
    "sheep": ["sheep"],
    "cow": ["cow", "cows"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears"],
    "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"],
    "backpack": ["backpack", "backpacks", "bag", "bags"],
    "umbrella": ["umbrella", "umbrellas"],
    "handbag": ["handbag", "handbags", "purse", "purses"],
    "tie": ["tie", "ties", "necktie", "neckties"],
    "suitcase": ["suitcase", "suitcases", "luggage"],
    "frisbee": ["frisbee", "frisbees"],
    "skis": ["ski", "skis"],
    "snowboard": ["snowboard", "snowboards"],
    "sports ball": ["sports ball", "sports balls", "ball", "balls"],
    "kite": ["kite", "kites"],
    "baseball bat": ["baseball bat", "baseball bats", "bat", "bats"],
    "baseball glove": ["baseball glove", "baseball gloves", "glove", "gloves"],
    "skateboard": ["skateboard", "skateboards"],
    "surfboard": ["surfboard", "surfboards"],
    "tennis racket": ["tennis racket", "tennis rackets", "racket", "rackets", "racquet", "racquets"],
    "bottle": ["bottle", "bottles"],
    "wine glass": ["wine glass", "wine glasses", "glass", "glasses"],
    "cup": ["cup", "cups", "mug", "mugs"],
    "fork": ["fork", "forks"],
    "knife": ["knife", "knives"],
    "spoon": ["spoon", "spoons"],
    "bowl": ["bowl", "bowls"],
    "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"],
    "sandwich": ["sandwich", "sandwiches"],
    "orange": ["orange", "oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "hot dog": ["hot dog", "hot dogs", "hotdog", "hotdogs"],
    "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "donuts", "doughnut", "doughnuts"],
    "cake": ["cake", "cakes"],
    "chair": ["chair", "chairs", "seat", "seats"],
    "couch": ["couch", "couches", "sofa", "sofas"],
    "potted plant": ["potted plant", "potted plants", "plant", "plants"],
    "bed": ["bed", "beds"],
    "dining table": ["dining table", "dining tables", "table", "tables"],
    "toilet": ["toilet", "toilets"],
    "tv": ["tv", "tvs", "television", "televisions", "monitor", "monitors"],
    "laptop": ["laptop", "laptops"],
    "mouse": ["mouse", "mice"],
    "remote": ["remote", "remotes", "remote control", "remote controls"],
    "keyboard": ["keyboard", "keyboards"],
    "cell phone": ["cell phone", "cell phones", "phone", "phones", "mobile phone", "mobile phones"],
    "microwave": ["microwave", "microwaves"],
    "oven": ["oven", "ovens"],
    "toaster": ["toaster", "toasters"],
    "sink": ["sink", "sinks"],
    "refrigerator": ["refrigerator", "refrigerators", "fridge", "fridges"],
    "book": ["book", "books"],
    "clock": ["clock", "clocks"],
    "vase": ["vase", "vases"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy bears", "bear", "bears"],
    "hair drier": ["hair drier", "hair driers", "hair dryer", "hair dryers"],
    "toothbrush": ["toothbrush", "toothbrushes"],
}


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if os.path.splitext(path)[-1].lower() == ".json":
        obj = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(obj, list):
            return [dict(x) for x in obj if isinstance(x, dict)]
        raise ValueError(f"Expected list JSON in {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            item = json.loads(s)
            if isinstance(item, dict):
                rows.append(dict(item))
    return rows


def id_candidates(row: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("id", "image_id", "question_id"):
        raw = str(row.get(key, "")).strip()
        if not raw:
            continue
        values.append(raw)
        digits = re.findall(r"\d+", raw)
        if digits:
            values.append(str(int(digits[-1])))
    image = str(row.get("image", "")).strip()
    if image:
        digits = re.findall(r"\d+", image)
        if digits:
            values.append(str(int(digits[-1])))
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_prediction_map(path: str, text_key: str) -> Dict[str, Dict[str, Any]]:
    rows = read_json_or_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        if text_key == "auto":
            for key in ("output", "text", "caption", "answer", "prediction"):
                if str(item.get(key, "")).strip():
                    item["_pred_text"] = str(item.get(key, ""))
                    break
        else:
            item["_pred_text"] = str(item.get(text_key, ""))
        for sid in id_candidates(item):
            out[sid] = item
    return out


def parse_object_vocab(spec: str) -> List[str]:
    s = str(spec or "").strip()
    if not s or s.lower() == "coco80":
        return list(COCO_OBJECTS)
    if os.path.isfile(s):
        vals: List[str] = []
        for line in open(s, "r", encoding="utf-8"):
            item = line.strip()
            if item:
                vals.append(item)
        return vals
    return [x.strip() for x in s.split(",") if x.strip()]


def alias_spans(text: str, alias: str) -> List[tuple[int, int]]:
    pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text.lower())]


def extract_mentioned_objects(text: str, vocab: Sequence[str]) -> List[str]:
    matches: List[tuple[int, int, int, str]] = []
    for obj in vocab:
        aliases = OBJECT_ALIASES.get(str(obj), [str(obj), f"{obj}s"])
        for alias in aliases:
            for start, end in alias_spans(text, alias):
                matches.append((start, end, len(alias), str(obj)))
    matches.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[3]))
    accepted_spans: List[tuple[int, int]] = []
    found: set[str] = set()
    for start, end, _, obj in matches:
        if any(not (end <= a_start or start >= a_end) for a_start, a_end in accepted_spans):
            continue
        accepted_spans.append((start, end))
        found.add(obj)
    return sorted(found)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(x) for x in values) / float(len(values)))


def sum_vals(values: Sequence[float]) -> float:
    return float(sum(float(x) for x in values))


def topk_sum(values: Sequence[float], k: int) -> float:
    return float(sum(sorted((float(x) for x in values), reverse=True)[: max(0, int(k))]))


def max_or_zero(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def normalized_entropy(values: Sequence[float]) -> float:
    vals = [max(0.0, float(x)) for x in values]
    total = sum(vals)
    if total <= 0.0 or len(vals) <= 1:
        return 0.0
    entropy = 0.0
    for value in vals:
        if value <= 0.0:
            continue
        p = value / total
        entropy -= p * math.log(max(p, 1e-12))
    return float(entropy / math.log(float(len(vals))))


def support_prob(item: Dict[str, Any]) -> float:
    return float(item.get("yesno_prob", 0.0))


def add_inventory_features(
    row: Dict[str, Any],
    *,
    vocab: Sequence[str],
    mentioned: Sequence[str],
    scored: Sequence[Dict[str, Any]],
    support_threshold: float,
    high_support_threshold: float,
    ambiguity_low_threshold: float,
) -> None:
    mentioned_set = set(str(x) for x in mentioned)
    inventory_items = [item for item in scored if support_prob(item) >= float(support_threshold)]
    mentioned_items = [item for item in scored if str(item.get("object")) in mentioned_set]
    unmentioned_items = [item for item in scored if str(item.get("object")) not in mentioned_set]
    missing_items = [item for item in inventory_items if str(item.get("object")) not in mentioned_set]
    missing_hi_items = [item for item in unmentioned_items if support_prob(item) >= float(high_support_threshold)]
    ambiguity_items = [
        item
        for item in unmentioned_items
        if float(ambiguity_low_threshold) <= support_prob(item) < float(high_support_threshold)
    ]
    mentioned_supported = [item for item in mentioned_items if support_prob(item) >= float(support_threshold)]
    mentioned_unsupported = [item for item in mentioned_items if support_prob(item) < float(support_threshold)]

    inventory_probs = [support_prob(item) for item in inventory_items]
    unmentioned_probs = [support_prob(item) for item in unmentioned_items]
    missing_probs = [support_prob(item) for item in missing_items]
    missing_hi_probs = [support_prob(item) for item in missing_hi_items]
    ambiguity_probs = [support_prob(item) for item in ambiguity_items]
    mentioned_probs = [support_prob(item) for item in mentioned_items]
    mentioned_risks = [float(item.get("yesno_risk", 0.0)) for item in mentioned_items]
    mentioned_unsupported_risks = [float(item.get("yesno_risk", 0.0)) for item in mentioned_unsupported]

    inv_sum = sum_vals(inventory_probs)
    unmentioned_sum = sum_vals(unmentioned_probs)
    missing_sum = sum_vals(missing_probs)
    missing_hi_sum = sum_vals(missing_hi_probs)
    ambiguity_sum = sum_vals(ambiguity_probs)
    mentioned_sum = sum_vals(mentioned_probs)
    risk_sum = sum_vals(mentioned_risks)
    unsupported_risk_sum = sum_vals(mentioned_unsupported_risks)
    missing_top1 = topk_sum(missing_probs, 1)
    missing_hi_top1 = topk_sum(missing_hi_probs, 1)
    ambiguity_count = int(len(ambiguity_items))
    ambiguity_entropy = normalized_entropy(ambiguity_probs)
    unmentioned_entropy = normalized_entropy(unmentioned_probs)
    missing_concentration = float(missing_top1 / max(1e-6, missing_sum))
    missing_hi_concentration = float(missing_hi_top1 / max(1e-6, missing_hi_sum))
    unmentioned_concentration = float(max_or_zero(unmentioned_probs) / max(1e-6, unmentioned_sum))
    mentioned_risk_max = max_or_zero(mentioned_risks)
    mentioned_support_max = max_or_zero(mentioned_probs)

    row.update(
        {
            "invyn_object_vocab_size": int(len(vocab)),
            "invyn_mentioned_object_count": int(len(mentioned_set)),
            "invyn_mentioned_object_names": " | ".join(sorted(mentioned_set)),
            "invyn_inventory_supported_count": int(len(inventory_items)),
            "invyn_inventory_supported_names": " | ".join(str(item.get("object", "")) for item in inventory_items),
            "invyn_inventory_support_prob_sum": inv_sum,
            "invyn_inventory_support_prob_mean": mean(inventory_probs),
            "invyn_missing_supported_count": int(len(missing_items)),
            "invyn_missing_supported_names": " | ".join(str(item.get("object", "")) for item in missing_items),
            "invyn_missing_support_prob_sum": missing_sum,
            "invyn_missing_support_prob_mean": mean(missing_probs),
            "invyn_missing_support_top1_prob": missing_top1,
            "invyn_missing_support_top3_sum": topk_sum(missing_probs, 3),
            "invyn_missing_support_top5_sum": topk_sum(missing_probs, 5),
            "invyn_missing_supported_rate": float(len(missing_items) / max(1, len(inventory_items))),
            "invyn_missing_hi_count": int(len(missing_hi_items)),
            "invyn_missing_hi_names": " | ".join(str(item.get("object", "")) for item in missing_hi_items),
            "invyn_missing_hi_mass": missing_hi_sum,
            "invyn_missing_hi_top1_prob": missing_hi_top1,
            "invyn_missing_hi_top3_sum": topk_sum(missing_hi_probs, 3),
            "invyn_missing_concentration": missing_concentration,
            "invyn_missing_hi_concentration": missing_hi_concentration,
            "invyn_mentioned_support_prob_sum": mentioned_sum,
            "invyn_mentioned_support_prob_mean": mean(mentioned_probs),
            "invyn_mentioned_support_max": mentioned_support_max,
            "invyn_mentioned_supported_count": int(len(mentioned_supported)),
            "invyn_mentioned_unsupported_count": int(len(mentioned_unsupported)),
            "invyn_mentioned_unsupported_risk_sum": unsupported_risk_sum,
            "invyn_mentioned_unsupported_risk_mean": mean(mentioned_unsupported_risks),
            "invyn_mentioned_risk_sum": risk_sum,
            "invyn_mentioned_risk_max": mentioned_risk_max,
            "invyn_coverage_by_inventory_count": float(len(mentioned_supported) / max(1, len(inventory_items))),
            "invyn_coverage_by_inventory_mass": float(mentioned_sum / max(1e-6, inv_sum)),
            "invyn_inventory_deficit_count": int(max(0, len(inventory_items) - len(mentioned_supported))),
            "invyn_inventory_deficit_mass": float(max(0.0, inv_sum - mentioned_sum)),
            "invyn_unmentioned_support_prob_sum": unmentioned_sum,
            "invyn_unmentioned_support_top1_prob": max_or_zero(unmentioned_probs),
            "invyn_unmentioned_concentration": unmentioned_concentration,
            "invyn_unmentioned_entropy": unmentioned_entropy,
            "invyn_ambiguity_count": ambiguity_count,
            "invyn_ambiguity_mass": ambiguity_sum,
            "invyn_ambiguity_mean": mean(ambiguity_probs),
            "invyn_ambiguity_entropy": ambiguity_entropy,
            "invyn_ambiguity_rate": float(ambiguity_count / max(1, len(unmentioned_items))),
            "invyn_missing_minus_mentioned_risk": float(missing_sum - risk_sum),
            "invyn_missing_minus_unsupported_risk": float(missing_sum - unsupported_risk_sum),
            "invyn_missing_hi_minus_mentioned_risk": float(missing_hi_sum - risk_sum),
            "invyn_missing_hi_minus_ambiguity_mass": float(missing_hi_sum - ambiguity_sum),
            "invyn_missing_hi_conc_minus_ambiguity_entropy": float(missing_hi_concentration - ambiguity_entropy),
            "invyn_missing_risk_ratio_eps_010": float(missing_sum / (risk_sum + 0.10)),
            "invyn_missing_risk_ratio_eps_100": float(missing_sum / (risk_sum + 1.00)),
            "invyn_missing_x_low_risk": float(missing_sum * max(0.0, 1.0 - min(1.0, risk_sum))),
            "invyn_missing_hi_x_low_risk": float(missing_hi_sum * max(0.0, 1.0 - min(1.0, risk_sum))),
            "invyn_benefit_cost_score_v1": float(missing_hi_sum - risk_sum - ambiguity_sum),
            "invyn_benefit_cost_score_v2": float(missing_top1 * missing_concentration - risk_sum - ambiguity_sum),
            "invyn_substitution_joint_score": float(missing_top1 * mentioned_risk_max),
            "invyn_substitution_margin_vs_mentioned_support": float(missing_top1 - mentioned_support_max),
            "invyn_substitution_score_v1": float(missing_top1 + mentioned_risk_max - ambiguity_entropy),
        }
    )
    add_prefix_stats(row, "invyn_all_objects", scored)
    add_prefix_stats(row, "invyn_mentioned_objects", mentioned_items)
    add_prefix_stats(row, "invyn_missing_supported_objects", missing_items)
    add_prefix_stats(row, "invyn_missing_hi_objects", missing_hi_items)
    add_prefix_stats(row, "invyn_ambiguity_objects", ambiguity_items)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract intervention-only image object-inventory yes/no features for generative rollback routing."
    )
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--intervention_pred_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", default="")
    ap.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", default="")
    ap.add_argument("--conv_mode", default="llava_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--object_vocab", default="coco80")
    ap.add_argument("--support_threshold", type=float, default=0.50)
    ap.add_argument("--high_support_threshold", type=float, default=0.70)
    ap.add_argument("--ambiguity_low_threshold", type=float, default=0.35)
    ap.add_argument("--pred_text_key", default="auto")
    ap.add_argument("--question_template", default="Is there a {object} in the image? Answer yes or no.")
    ap.add_argument("--yes_text", default="Yes")
    ap.add_argument("--no_text", default="No")
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=True)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    if bool(args.reuse_if_exists) and os.path.isfile(args.out_csv):
        print(f"[reuse] {args.out_csv}")
        return

    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    pred_map = load_prediction_map(os.path.abspath(args.intervention_pred_jsonl), str(args.pred_text_key))
    vocab = parse_object_vocab(str(args.object_vocab))
    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(args.model_base or None),
        conv_mode=args.conv_mode,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    n_errors = 0
    n_object_probes = 0
    n_missing_pred = 0
    for idx, sample in enumerate(question_rows):
        sid = safe_id(sample.get("question_id", sample.get("id", sample.get("image_id"))))
        try:
            sid = str(int(sid))
        except Exception:
            pass
        image_name = str(sample.get("image", "")).strip()
        row: Dict[str, Any] = {
            "id": sid,
            "image_id": sid,
            "image": image_name,
            "invyn_error": "",
        }
        try:
            if not sid:
                raise ValueError("Missing sample id.")
            pred_row = None
            for cand in id_candidates({"id": sid, "image": image_name}):
                pred_row = pred_map.get(cand)
                if pred_row is not None:
                    break
            if pred_row is None:
                n_missing_pred += 1
                raise ValueError("Missing intervention prediction row.")
            caption = str(pred_row.get("_pred_text", ""))
            image_path = os.path.join(args.image_folder, image_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = runtime.load_image(image_path)

            mentioned = extract_mentioned_objects(caption, vocab)
            scored = score_objects(
                vocab,
                runtime=runtime,
                image=image,
                question_template=str(args.question_template),
                yes_text=str(args.yes_text),
                no_text=str(args.no_text),
                score_mode="yesno",
                cache={},
            )
            n_object_probes += int(len(scored))
            row["invyn_caption"] = caption
            add_inventory_features(
                row,
                vocab=vocab,
                mentioned=mentioned,
                scored=scored,
                support_threshold=float(args.support_threshold),
                high_support_threshold=float(args.high_support_threshold),
                ambiguity_low_threshold=float(args.ambiguity_low_threshold),
            )
        except Exception as exc:
            n_errors += 1
            row["invyn_error"] = str(exc)
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[inventory-yesno] {idx + 1}/{len(question_rows)} object_probes={n_object_probes}")

    write_csv(args.out_csv, rows)
    print(f"[saved] {args.out_csv}")
    if str(args.out_summary_json or "").strip():
        feature_keys = [key for key in rows[0].keys() if key.startswith("invyn_")] if rows else []
        write_json(
            args.out_summary_json,
            {
                "inputs": {
                    "question_file": os.path.abspath(args.question_file),
                    "image_folder": os.path.abspath(args.image_folder),
                    "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
                    "model_path": str(args.model_path),
                    "model_base": str(args.model_base),
                    "conv_mode": str(args.conv_mode),
                    "device": str(args.device),
                    "object_vocab": str(args.object_vocab),
                    "n_object_vocab": int(len(vocab)),
                    "support_threshold": float(args.support_threshold),
                    "high_support_threshold": float(args.high_support_threshold),
                    "ambiguity_low_threshold": float(args.ambiguity_low_threshold),
                    "pred_text_key": str(args.pred_text_key),
                    "question_template": str(args.question_template),
                    "yes_text": str(args.yes_text),
                    "no_text": str(args.no_text),
                    "limit": int(args.limit),
                },
                "counts": {
                    "n_rows": int(len(rows)),
                    "n_errors": int(n_errors),
                    "n_missing_pred": int(n_missing_pred),
                    "n_object_probes": int(n_object_probes),
                    "n_forward_passes_est": int(n_object_probes * 2),
                    "n_features": int(len(feature_keys)),
                },
                "feature_keys": feature_keys,
                "outputs": {"out_csv": os.path.abspath(args.out_csv)},
            },
        )


if __name__ == "__main__":
    main()
