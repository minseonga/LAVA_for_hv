#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Any, List


def sf(x: Any, d: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return d
        return float(x)
    except Exception:
        return d


def mean(v: List[float]) -> float:
    return float(sum(v) / len(v)) if v else 0.0


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
    if str(path or "").strip() == "":
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            o = json.loads(s)
            qid = str(o.get("question_id", "")).strip()
            if qid == "":
                continue
            out[qid] = parse_yes_no(str(o.get("text", "")))
    return out


def load_ft(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            qid = str(r.get("question_id", "")).strip()
            if qid == "":
                continue
            out[qid] = {
                "yes": sf(r.get("yes_logit_pre"), 0.0),
                "no": sf(r.get("no_logit_pre"), 0.0),
                "margin": sf(r.get("margin_pre"), 0.0),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare first-token logits/margins between two runs.")
    ap.add_argument("--base_csv", type=str, required=True)
    ap.add_argument("--new_csv", type=str, required=True)
    ap.add_argument("--base_pred_jsonl", type=str, default="")
    ap.add_argument("--new_pred_jsonl", type=str, default="")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_rows_csv", type=str, default="")
    args = ap.parse_args()

    base = load_ft(os.path.abspath(args.base_csv))
    new = load_ft(os.path.abspath(args.new_csv))
    ids = sorted(set(base.keys()) & set(new.keys()), key=lambda x: int(x) if x.isdigit() else x)
    if len(ids) == 0:
        raise RuntimeError("No overlapping ids between base_csv and new_csv")

    pred_base = load_pred(os.path.abspath(args.base_pred_jsonl))
    pred_new = load_pred(os.path.abspath(args.new_pred_jsonl))

    d_margin = []
    abs_d_margin = []
    d_yes = []
    d_no = []
    sign_flip = 0
    argmax_change = 0
    pred_change = 0

    rows = []
    for qid in ids:
        b = base[qid]
        n = new[qid]
        dy = float(n["yes"] - b["yes"])
        dn = float(n["no"] - b["no"])
        dm = float(n["margin"] - b["margin"])
        d_yes.append(dy)
        d_no.append(dn)
        d_margin.append(dm)
        abs_d_margin.append(abs(dm))

        b_sign = (b["margin"] >= 0.0)
        n_sign = (n["margin"] >= 0.0)
        if b_sign != n_sign:
            sign_flip += 1
        if (b["yes"] >= b["no"]) != (n["yes"] >= n["no"]):
            argmax_change += 1

        pb = pred_base.get(qid, "")
        pn = pred_new.get(qid, "")
        if pb in {"yes", "no"} and pn in {"yes", "no"} and pb != pn:
            pred_change += 1

        rows.append(
            {
                "question_id": qid,
                "base_yes": b["yes"],
                "base_no": b["no"],
                "base_margin": b["margin"],
                "new_yes": n["yes"],
                "new_no": n["no"],
                "new_margin": n["margin"],
                "delta_yes": dy,
                "delta_no": dn,
                "delta_margin": dm,
                "margin_sign_flip": int(b_sign != n_sign),
                "argmax_change": int((b["yes"] >= b["no"]) != (n["yes"] >= n["no"])),
                "base_pred": pb,
                "new_pred": pn,
                "pred_change": int(pb in {"yes", "no"} and pn in {"yes", "no"} and pb != pn),
            }
        )

    abs_sorted = sorted(abs_d_margin)
    p90 = abs_sorted[int(0.9 * (len(abs_sorted) - 1))]

    out = {
        "inputs": {
            "base_csv": os.path.abspath(args.base_csv),
            "new_csv": os.path.abspath(args.new_csv),
            "base_pred_jsonl": os.path.abspath(args.base_pred_jsonl) if args.base_pred_jsonl else "",
            "new_pred_jsonl": os.path.abspath(args.new_pred_jsonl) if args.new_pred_jsonl else "",
        },
        "counts": {
            "n_common": int(len(ids)),
        },
        "delta": {
            "yes_mean": mean(d_yes),
            "no_mean": mean(d_no),
            "margin_mean": mean(d_margin),
            "abs_margin_mean": mean(abs_d_margin),
            "abs_margin_p90": float(p90),
            "abs_margin_max": float(max(abs_d_margin)),
        },
        "changes": {
            "margin_sign_flip_count": int(sign_flip),
            "margin_sign_flip_rate": float(sign_flip / len(ids)),
            "argmax_change_count": int(argmax_change),
            "argmax_change_rate": float(argmax_change / len(ids)),
            "pred_change_count": int(pred_change),
            "pred_change_rate": float(pred_change / len(ids)),
        },
    }

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out_rows = (
        os.path.abspath(args.out_rows_csv)
        if str(args.out_rows_csv).strip() != ""
        else os.path.splitext(out_json)[0] + "_rows.csv"
    )
    with open(out_rows, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print("[saved]", out_json)
    print("[saved]", out_rows)
    print("[summary]", json.dumps(out["changes"], ensure_ascii=False))


if __name__ == "__main__":
    main()
