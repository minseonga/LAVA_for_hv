#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_apply_cvis_layered_d_fusion_controller import (  # noqa: E402
    build_fusion_scores,
    filter_c_rows,
    index_rows_by_id,
    merge_rows,
    read_csv_rows,
    read_json,
    score_c_rows,
    score_d_rows,
    score_object_rows,
)
from build_apply_layered_d_family_controller import is_candidate, maybe_float, maybe_int, write_csv, write_json  # noqa: E402


def percentile_ranks(scores: Mapping[str, float]) -> Dict[str, float]:
    finite = sorted((float(v), str(k)) for k, v in scores.items() if math.isfinite(float(v)))
    n = len(finite)
    if n == 0:
        return {}
    out: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i + 1
        while j < n and finite[j][0] == finite[i][0]:
            j += 1
        rank = float(j / n)
        for k in range(i, j):
            out[finite[k][1]] = rank
        i = j
    return out


def top_ids(scores: Mapping[str, float], frac: float) -> set[str]:
    n = len(scores)
    if n == 0:
        return set()
    k = max(1, int(math.ceil(float(frac) * n)))
    return {sid for sid, _ in sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)[:k]}


def parse_fracs(spec: str) -> List[float]:
    vals: List[float] = []
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value > 1.0:
            value /= 100.0
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"top fraction must be in (0,1], got {token}")
        vals.append(value)
    return vals or [0.1, 0.2, 0.3]


def short_text(value: Any, limit: int = 160) -> str:
    s = " ".join(str(value or "").split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def summarize_bucket(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cats = Counter(str(r.get("category", "") or "unknown") for r in rows)
    transitions = Counter(f"{r.get('baseline_label','')}->{r.get('intervention_label','')}" for r in rows)
    harm = sum(int(maybe_int(r.get("harm")) or 0) for r in rows)
    help_ = sum(int(maybe_int(r.get("help")) or 0) for r in rows)
    return {
        "n": len(rows),
        "harm": harm,
        "help": help_,
        "category_counts": dict(cats),
        "transition_counts": dict(transitions),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit harm missed by fixed C_vis and layered-D scores.")
    ap.add_argument("--c_rows_csv", required=True)
    ap.add_argument("--d_trajectory_long_csv", required=True)
    ap.add_argument("--object_trajectory_long_csv", default="")
    ap.add_argument("--policy_json", required=True, help="C-vis layered-D fusion selected_policy.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_filter", default="", choices=["", "all", "changed_answer", "yes_to_no"])
    ap.add_argument("--top_fracs", default="0.05,0.10,0.20,0.30")
    ap.add_argument("--focus_top_frac", type=float, default=0.10)
    ap.add_argument("--max_examples", type=int, default=80)
    args = ap.parse_args()

    policy = read_json(os.path.abspath(args.policy_json))
    candidate_filter = str(args.candidate_filter or policy.get("candidate_filter") or "changed_answer")
    c_rows = filter_c_rows(read_csv_rows(os.path.abspath(args.c_rows_csv)), str(policy.get("c_layer", "")))
    d_rows = __import__("build_apply_layered_d_family_controller").read_rows(os.path.abspath(args.d_trajectory_long_csv))
    object_policy = policy.get("object_policy")
    if object_policy and not str(args.object_trajectory_long_csv or "").strip():
        raise RuntimeError("--object_trajectory_long_csv is required by this policy.")
    object_rows = (
        read_csv_rows(os.path.abspath(args.object_trajectory_long_csv))
        if str(args.object_trajectory_long_csv or "").strip()
        else []
    )

    c_rows_by_id = index_rows_by_id(c_rows)
    c_scores_all = score_c_rows(c_rows_by_id, policy["c_metric"])
    d_rows_by_id, d_scores_all = score_d_rows(d_rows, policy["d_policy"])
    object_rows_by_id, object_scores_all = score_object_rows(object_rows, object_policy)
    merged_by_id = merge_rows(c_rows_by_id, d_rows_by_id, object_rows_by_id)

    common_ids = set(c_scores_all) & set(d_scores_all)
    valid_ids = [
        sid
        for sid in sorted(common_ids, key=lambda x: (len(str(x)), str(x)))
        if is_candidate(merged_by_id[sid], candidate_filter)
    ]
    c_scores = {sid: float(c_scores_all[sid]) for sid in valid_ids}
    d_scores = {sid: float(d_scores_all[sid]) for sid in valid_ids}
    object_scores = {sid: float(object_scores_all[sid]) for sid in valid_ids if sid in object_scores_all} if object_policy else {}
    fusion_spec = dict(policy["selected_policy"]["fusion"])
    score_maps: List[Mapping[str, float]] = [c_scores, d_scores]
    if object_policy:
        score_maps.append(object_scores)
    fusion_scores = build_fusion_scores(score_maps, fusion_spec, required_streams=2)

    c_pct = percentile_ranks(c_scores)
    d_pct = percentile_ranks(d_scores)
    object_pct = percentile_ranks(object_scores) if object_policy else {}
    f_pct = percentile_ranks(fusion_scores)

    rows: List[Dict[str, Any]] = []
    for sid in valid_ids:
        src = merged_by_id[sid]
        rows.append(
            {
                "id": sid,
                "category": src.get("category", ""),
                "image": src.get("image", src.get("image_id", "")),
                "question": short_text(src.get("question", src.get("text", ""))),
                "gt_label": src.get("gt_label", src.get("answer", "")),
                "baseline_label": src.get("baseline_label", ""),
                "intervention_label": src.get("intervention_label", ""),
                "harm": int(maybe_int(src.get("harm")) or 0),
                "help": int(maybe_int(src.get("help")) or 0),
                "baseline_correct": src.get("baseline_correct", ""),
                "intervention_correct": src.get("intervention_correct", ""),
                "c_score": c_scores[sid],
                "d_score": d_scores[sid],
                "object_score": object_scores.get(sid) if object_policy else None,
                "fusion_score": fusion_scores[sid],
                "c_pct": c_pct.get(sid),
                "d_pct": d_pct.get(sid),
                "object_pct": object_pct.get(sid) if object_policy else None,
                "fusion_pct": f_pct.get(sid),
                "baseline_text": short_text(src.get("baseline_text", "")),
                "intervention_text": short_text(src.get("intervention_text", "")),
            }
        )

    top_fracs = parse_fracs(str(args.top_fracs))
    summary: Dict[str, Any] = {
        "inputs": {
            "c_rows_csv": os.path.abspath(args.c_rows_csv),
            "d_trajectory_long_csv": os.path.abspath(args.d_trajectory_long_csv),
            "object_trajectory_long_csv": os.path.abspath(args.object_trajectory_long_csv)
            if str(args.object_trajectory_long_csv or "").strip()
            else "",
            "policy_json": os.path.abspath(args.policy_json),
            "candidate_filter": candidate_filter,
        },
        "policy": {
            "c_feature": policy.get("c_feature"),
            "c_metric": policy.get("c_metric"),
            "d_policy": policy.get("d_policy"),
            "object_policy": policy.get("object_policy"),
            "selected_policy": policy.get("selected_policy"),
        },
        "counts": summarize_bucket(rows),
        "top_fracs": {},
    }

    for frac in top_fracs:
        c_top = top_ids(c_scores, frac)
        d_top = top_ids(d_scores, frac)
        object_top = top_ids(object_scores, frac) if object_policy else set()
        f_top = top_ids(fusion_scores, frac)
        stream_sets = [c_top, d_top] + ([object_top] if object_policy else [])
        any_stream_top = set().union(*stream_sets)
        all_streams_top = set.intersection(*stream_sets) if stream_sets else set()
        groups = {
            "c_top": c_top,
            "d_top": d_top,
            **({"object_top": object_top} if object_policy else {}),
            "fusion_top": f_top,
            "c_or_d_top": c_top | d_top,
            "c_and_d_top": c_top & d_top,
            "any_stream_top": any_stream_top,
            "all_streams_top": all_streams_top,
            "missed_by_c_or_d_top": set(valid_ids) - (c_top | d_top),
            "missed_by_any_stream_top": set(valid_ids) - any_stream_top,
            "missed_by_fusion_top": set(valid_ids) - f_top,
        }
        summary["top_fracs"][f"top_{frac:.3f}"] = {
            name: summarize_bucket([r for r in rows if str(r["id"]) in ids])
            for name, ids in groups.items()
        }

    focus_frac = float(args.focus_top_frac)
    if focus_frac > 1.0:
        focus_frac /= 100.0
    c_focus = top_ids(c_scores, focus_frac)
    d_focus = top_ids(d_scores, focus_frac)
    object_focus = top_ids(object_scores, focus_frac) if object_policy else set()
    f_focus = top_ids(fusion_scores, focus_frac)
    for row in rows:
        sid = str(row["id"])
        row["c_top_focus"] = int(sid in c_focus)
        row["d_top_focus"] = int(sid in d_focus)
        row["object_top_focus"] = int(sid in object_focus) if object_policy else 0
        row["fusion_top_focus"] = int(sid in f_focus)
        hits = [name for name, ids in [("c", c_focus), ("d", d_focus), ("object", object_focus)] if sid in ids]
        if not hits:
            row["bucket"] = "miss"
        elif not object_policy and len(hits) == 2:
            row["bucket"] = "both"
        elif len(hits) == 1:
            row["bucket"] = f"{hits[0]}_only"
        else:
            row["bucket"] = "+".join(hits)

    harm_miss = [
        row
        for row in rows
        if int(row["harm"]) == 1
        and int(row["c_top_focus"]) == 0
        and int(row["d_top_focus"]) == 0
        and int(row["object_top_focus"]) == 0
    ]
    harm_miss = sorted(harm_miss, key=lambda r: (float(r["fusion_pct"] or 0), float(r["c_pct"] or 0), float(r["d_pct"] or 0)))

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "cvis_layered_d_audit_rows.csv"), rows)
    write_csv(os.path.join(out_dir, "missed_harm_examples.csv"), harm_miss[: int(args.max_examples)])
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print("[saved]", os.path.join(out_dir, "summary.json"))
    print("[saved]", os.path.join(out_dir, "cvis_layered_d_audit_rows.csv"))
    print("[saved]", os.path.join(out_dir, "missed_harm_examples.csv"))


if __name__ == "__main__":
    main()
