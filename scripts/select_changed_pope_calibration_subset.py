#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def safe_id(x: Any) -> str:
    return str("" if x is None else x).strip()


def parse_yes_no(text: Any) -> str:
    s = str("" if text is None else text).strip()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip().lower() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    return "yes"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(fieldnames))
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fieldnames})


def load_gt(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            sid = safe_id(row.get("id") or row.get("question_id"))
            ans = safe_id(row.get("answer")).lower()
            if sid and ans in {"yes", "no"}:
                out[sid] = dict(row)
    return out


def load_pred(path: str, key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    mode = (key or "auto").strip().lower()
    for row in read_jsonl(path):
        sid = safe_id(row.get("question_id") or row.get("id"))
        if not sid:
            continue
        if mode == "auto":
            text = row.get("text", "")
            if not str(text).strip():
                text = row.get("output", "")
            if not str(text).strip():
                text = row.get("answer", "")
        else:
            text = row.get(mode, "")
        label = parse_yes_no(text)
        if label in {"yes", "no"}:
            out[sid] = label
    return out


def parse_pair_spec(spec: str) -> Tuple[str, str, str, str, str]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 5:
        raise ValueError(
            "--pred_pair must have 5 comma-separated fields: "
            "name,baseline_jsonl,baseline_key,intervention_jsonl,intervention_key"
        )
    name, base_path, base_key, int_path, int_key = parts
    if not name:
        raise ValueError("pred_pair name is empty")
    return name, base_path, base_key or "auto", int_path, int_key or "auto"


def stratified_take(ids: Sequence[str], gt: Dict[str, Dict[str, Any]], target_n: int, rng: random.Random) -> List[str]:
    by_label: Dict[str, List[str]] = {"yes": [], "no": []}
    other: List[str] = []
    for sid in ids:
        ans = safe_id(gt.get(sid, {}).get("answer")).lower()
        if ans in by_label:
            by_label[ans].append(sid)
        else:
            other.append(sid)
    for bucket in by_label.values():
        rng.shuffle(bucket)
    rng.shuffle(other)

    n_yes = min(len(by_label["yes"]), target_n // 2)
    n_no = min(len(by_label["no"]), target_n - n_yes)
    if n_yes + n_no < target_n:
        n_yes = min(len(by_label["yes"]), target_n - n_no)
        n_no = min(len(by_label["no"]), target_n - n_yes)

    selected = by_label["yes"][:n_yes] + by_label["no"][:n_no]
    if len(selected) < target_n:
        remaining = [sid for sid in ids if sid not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_n - len(selected)])
    rng.shuffle(selected)
    return selected[:target_n]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Select a fixed calibration subset from a POPE-style bank, enriching "
            "for route candidates using only baseline/intervention prediction disagreement."
        )
    )
    ap.add_argument("--source_q_jsonl", required=True)
    ap.add_argument("--source_q_with_object_jsonl", default="")
    ap.add_argument("--source_gt_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument(
        "--pred_pair",
        action="append",
        default=[],
        help=(
            "Prediction pair spec: name,baseline_jsonl,baseline_key,intervention_jsonl,intervention_key. "
            "Can be repeated. changed_any is the union over pairs."
        ),
    )
    ap.add_argument("--qid_prefix", default="")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    q_rows = read_jsonl(os.path.abspath(args.source_q_jsonl))
    q_by_id = {safe_id(r.get("question_id") or r.get("id")): r for r in q_rows}
    if args.source_q_with_object_jsonl.strip():
        q_obj_rows = read_jsonl(os.path.abspath(args.source_q_with_object_jsonl))
        q_obj_by_id = {safe_id(r.get("question_id") or r.get("id")): r for r in q_obj_rows}
    else:
        q_obj_by_id = {}
    gt = load_gt(os.path.abspath(args.source_gt_csv))

    candidate_ids = [sid for sid in q_by_id if sid in gt]
    if not candidate_ids:
        raise RuntimeError("No overlapping ids between source_q_jsonl and source_gt_csv.")

    pair_specs = [parse_pair_spec(s) for s in args.pred_pair]
    pair_preds: Dict[str, Tuple[Dict[str, str], Dict[str, str]]] = {}
    for name, base_path, base_key, int_path, int_key in pair_specs:
        pair_preds[name] = (
            load_pred(os.path.abspath(base_path), base_key),
            load_pred(os.path.abspath(int_path), int_key),
        )

    audit_rows: List[Dict[str, Any]] = []
    changed_ids: List[str] = []
    unchanged_ids: List[str] = []
    pair_counts: Dict[str, Dict[str, int]] = {
        name: defaultdict(int) for name in pair_preds
    }

    for sid in candidate_ids:
        ans = safe_id(gt[sid].get("answer")).lower()
        audit: Dict[str, Any] = {
            "id": sid,
            "answer": ans,
            "image_id": gt[sid].get("image_id", ""),
        }
        changed_any = False
        for name, (base_pred, int_pred) in pair_preds.items():
            b = base_pred.get(sid, "")
            i = int_pred.get(sid, "")
            changed = b in {"yes", "no"} and i in {"yes", "no"} and b != i
            changed_any = changed_any or changed
            audit[f"{name}_baseline_label"] = b
            audit[f"{name}_intervention_label"] = i
            audit[f"{name}_changed"] = int(changed)
            if b not in {"yes", "no"} or i not in {"yes", "no"}:
                pair_counts[name]["missing"] += 1
            elif not changed:
                pair_counts[name]["unchanged"] += 1
            else:
                pair_counts[name]["changed"] += 1
                if b == ans and i != ans:
                    pair_counts[name]["harm"] += 1
                    audit[f"{name}_transition"] = "harm"
                elif b != ans and i == ans:
                    pair_counts[name]["help"] += 1
                    audit[f"{name}_transition"] = "help"
                else:
                    pair_counts[name]["neutral"] += 1
                    audit[f"{name}_transition"] = "neutral"
        audit["changed_any"] = int(changed_any)
        audit_rows.append(audit)
        if changed_any:
            changed_ids.append(sid)
        else:
            unchanged_ids.append(sid)

    if pair_preds:
        selected = stratified_take(changed_ids, gt, int(args.target_n), rng)
        if len(selected) < int(args.target_n):
            selected_set = set(selected)
            fill_pool = [sid for sid in unchanged_ids if sid not in selected_set]
            fill = stratified_take(fill_pool, gt, int(args.target_n) - len(selected), rng)
            selected.extend(fill)
            rng.shuffle(selected)
    else:
        selected = stratified_take(candidate_ids, gt, int(args.target_n), rng)

    selected_set = set(selected)
    out_q_rows = [q_by_id[sid] for sid in selected if sid in q_by_id]
    out_q_obj_rows = []
    for sid in selected:
        row = dict(q_obj_by_id.get(sid) or q_by_id[sid])
        row.setdefault("object", [])
        out_q_obj_rows.append(row)

    out_gt_rows = [gt[sid] for sid in selected if sid in gt]
    out_subset_rows = [{"id": sid, "group": gt[sid].get("group", "calibration")} for sid in selected if sid in gt]

    q_jsonl = os.path.join(out_dir, "gqa_train_pope_q.jsonl")
    q_obj_jsonl = os.path.join(out_dir, "gqa_train_pope_q_with_object.jsonl")
    gt_csv = os.path.join(out_dir, "gqa_train_pope_gt.csv")
    subset_csv = os.path.join(out_dir, "gqa_train_pope_subset_ids.csv")
    audit_csv = os.path.join(out_dir, "route_candidate_audit.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    write_jsonl(q_jsonl, out_q_rows)
    write_jsonl(q_obj_jsonl, out_q_obj_rows)
    gt_fields = ["id", "answer", "category", "image_id", "question", "orig_question_id", "group", "object"]
    extra_gt_fields = sorted({k for row in out_gt_rows for k in row.keys()} - set(gt_fields))
    write_csv(gt_csv, out_gt_rows, gt_fields + extra_gt_fields)
    write_csv(subset_csv, out_subset_rows, ["id", "group"])
    audit_fields = sorted({k for row in audit_rows for k in row.keys()})
    write_csv(audit_csv, audit_rows, audit_fields)

    selected_changed = sum(1 for sid in selected if sid in set(changed_ids))
    selected_by_answer = defaultdict(int)
    for sid in selected:
        selected_by_answer[safe_id(gt[sid].get("answer")).lower()] += 1

    summary = {
        "inputs": {
            "source_q_jsonl": os.path.abspath(args.source_q_jsonl),
            "source_q_with_object_jsonl": os.path.abspath(args.source_q_with_object_jsonl) if args.source_q_with_object_jsonl else "",
            "source_gt_csv": os.path.abspath(args.source_gt_csv),
            "target_n": int(args.target_n),
            "seed": int(args.seed),
            "pred_pair": args.pred_pair,
        },
        "counts": {
            "n_source_questions": int(len(q_rows)),
            "n_candidate_ids": int(len(candidate_ids)),
            "n_changed_any": int(len(changed_ids)),
            "n_unchanged_or_missing": int(len(unchanged_ids)),
            "n_selected": int(len(selected)),
            "n_selected_changed_any": int(selected_changed),
            "n_selected_fill": int(len(selected) - selected_changed),
            "n_selected_yes": int(selected_by_answer["yes"]),
            "n_selected_no": int(selected_by_answer["no"]),
            "pair_counts": {name: dict(counts) for name, counts in pair_counts.items()},
        },
        "outputs": {
            "q_jsonl": q_jsonl,
            "q_with_object_jsonl": q_obj_jsonl,
            "gt_csv": gt_csv,
            "subset_ids_csv": subset_csv,
            "route_candidate_audit_csv": audit_csv,
            "summary_json": summary_json,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
