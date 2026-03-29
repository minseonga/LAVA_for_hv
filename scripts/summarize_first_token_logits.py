#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List


def sf(x: Any, d: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return d
        return float(x)
    except Exception:
        return d


def si(x: Any, d: int = 0) -> int:
    try:
        if x is None or x == "":
            return d
        return int(float(x))
    except Exception:
        return d


def mean(v: List[float]) -> float:
    return float(sum(v) / len(v)) if v else 0.0


def frac(v: List[bool]) -> float:
    return float(sum(1 for x in v if x) / len(v)) if v else 0.0


def parse_yes_no(text: str) -> str:
    s = (text or "").strip().lower()
    if s == "":
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    ws = set(w.strip().lower() for w in first.split())
    if "no" in ws or "not" in ws:
        return "no"
    if "yes" in ws:
        return "yes"
    return "yes"


def load_pred(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            o = json.loads(s)
            qid = str(o.get("question_id", "")).strip()
            if not qid:
                continue
            out[qid] = parse_yes_no(str(o.get("text", "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize first-token logits pre/post (nogate vs gated).")
    ap.add_argument("--first_token_csv", type=str, required=True)
    ap.add_argument("--pred_jsonl", type=str, default="", help="optional model output jsonl to compare first-token sign vs final pred")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.abspath(args.first_token_csv), "r", encoding="utf-8")))
    if len(rows) == 0:
        raise RuntimeError("No rows in first_token_csv")

    pred = load_pred(os.path.abspath(args.pred_jsonl)) if str(args.pred_jsonl).strip() != "" else {}

    delta_margin = []
    abs_delta_margin = []
    sign_flip = []
    changed_small = []
    changed_med = []
    changed_large = []
    pre_margin = []
    post_margin = []

    mismatch_first_vs_final = []

    per_rows = []

    for r in rows:
        qid = str(r.get("question_id", "")).strip()
        m_pre_nogate = sf(r.get("margin_pre_nogate"), 0.0)
        m_post = sf(r.get("margin_pre"), 0.0)  # from gated run, pre-safeguard
        d = float(m_post - m_pre_nogate)

        delta_margin.append(d)
        abs_delta_margin.append(abs(d))
        sign_flip.append((m_pre_nogate >= 0) != (m_post >= 0))
        changed_small.append(abs(d) > 1e-3)
        changed_med.append(abs(d) > 1e-2)
        changed_large.append(abs(d) > 1e-1)
        pre_margin.append(m_pre_nogate)
        post_margin.append(m_post)

        first_label_post = "yes" if m_post >= 0 else "no"
        final_label = pred.get(qid, "") if pred else ""
        mm = (final_label in {"yes", "no"} and final_label != first_label_post)
        mismatch_first_vs_final.append(mm)

        per_rows.append(
            {
                "question_id": qid,
                "margin_pre_nogate": m_pre_nogate,
                "margin_post_gated": m_post,
                "delta_margin": d,
                "first_token_sign_flip": int(((m_pre_nogate >= 0) != (m_post >= 0))),
                "first_label_post": first_label_post,
                "final_pred": final_label,
                "first_vs_final_mismatch": int(mm),
            }
        )

    summary = {
        "counts": {
            "n": len(rows),
        },
        "margin": {
            "pre_nogate_mean": mean(pre_margin),
            "post_gated_mean": mean(post_margin),
            "delta_mean": mean(delta_margin),
            "abs_delta_mean": mean(abs_delta_margin),
            "abs_delta_p90": sorted(abs_delta_margin)[int(0.9 * (len(abs_delta_margin) - 1))],
            "abs_delta_max": max(abs_delta_margin),
        },
        "change_fractions": {
            "abs_delta_gt_1e-3": frac(changed_small),
            "abs_delta_gt_1e-2": frac(changed_med),
            "abs_delta_gt_1e-1": frac(changed_large),
            "margin_sign_flip": frac(sign_flip),
            "first_vs_final_mismatch": frac(mismatch_first_vs_final) if pred else None,
        },
        "outputs": {
            "out_json": os.path.abspath(args.out_json),
            "per_row_csv": os.path.splitext(os.path.abspath(args.out_json))[0] + "_rows.csv",
        },
    }

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out_rows_csv = os.path.splitext(out_json)[0] + "_rows.csv"
    with open(out_rows_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        wr.writeheader()
        wr.writerows(per_rows)

    print("[saved]", out_json)
    print("[saved]", out_rows_csv)
    print("[summary]", json.dumps(summary["change_fractions"], ensure_ascii=False))


if __name__ == "__main__":
    main()
