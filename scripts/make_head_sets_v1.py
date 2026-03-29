#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple


RE_FAITHFUL = re.compile(r"^head_attn_vis_(ratio|sum)__layer_(\d+)__head_(\d+)$")
RE_HARMFUL = re.compile(r"^head_contrib_l(\d+)_h(\d+)$")


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def pick_faithful(
    rows: List[Dict[str, str]],
    topk: int,
    auc_min: float,
    prefer_ratio: bool = True,
) -> List[Dict[str, object]]:
    cand = []
    for r in rows:
        m = str(r.get("metric", ""))
        g = RE_FAITHFUL.match(m)
        if g is None:
            continue
        kind = g.group(1)
        li = int(g.group(2))
        hi = int(g.group(3))
        auc = safe_float(r.get("auc_best_dir"), 0.0)
        ks = safe_float(r.get("ks_hall_high"), 0.0)
        direction = str(r.get("direction", ""))
        if direction != "lower_in_hallucination":
            continue
        if auc < float(auc_min):
            continue
        cand.append(
            {
                "layer": li,
                "head": hi,
                "kind": kind,
                "auc": auc,
                "ks": ks,
                "metric": m,
                "direction": direction,
            }
        )

    # Prefer ratio to avoid ratio/sum duplicates dominating the same head.
    if prefer_ratio:
        cand.sort(key=lambda x: (x["kind"] != "ratio", -x["auc"], -x["ks"]))
    else:
        cand.sort(key=lambda x: (-x["auc"], -x["ks"]))

    out = []
    seen = set()
    for c in cand:
        key = (int(c["layer"]), int(c["head"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= int(topk):
            break
    return out


def pick_harmful(rows: List[Dict[str, str]], topk: int, auc_min: float) -> List[Dict[str, object]]:
    cand = []
    for r in rows:
        m = str(r.get("metric", ""))
        g = RE_HARMFUL.match(m)
        if g is None:
            continue
        li = int(g.group(1))
        hi = int(g.group(2))
        auc = safe_float(r.get("auc_best_dir"), 0.0)
        ks = safe_float(r.get("ks_hall_high"), 0.0)
        direction = str(r.get("direction", ""))
        if direction != "higher_in_hallucination":
            continue
        if auc < float(auc_min):
            continue
        cand.append(
            {
                "layer": li,
                "head": hi,
                "auc": auc,
                "ks": ks,
                "metric": m,
                "direction": direction,
            }
        )
    cand.sort(key=lambda x: (-x["auc"], -x["ks"]))
    out = []
    seen = set()
    for c in cand:
        key = (int(c["layer"]), int(c["head"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= int(topk):
            break
    return out


def to_layer_map(items: List[Dict[str, object]]) -> Dict[str, List[int]]:
    mp: Dict[str, List[int]] = {}
    for x in items:
        li = str(int(x["layer"]))
        hi = int(x["head"])
        mp.setdefault(li, []).append(hi)
    for k in list(mp.keys()):
        mp[k] = sorted(set(mp[k]))
    return mp


def to_specs(items: List[Dict[str, object]]) -> List[str]:
    return [f"{int(x['layer'])}:{int(x['head'])}" for x in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--faithful_head_eval_csv",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/pope_visual_disconnect_1000_headscan_l10_24/head_eval_fp_vs_tp_yes.csv",
    )
    ap.add_argument(
        "--harmful_head_eval_csv",
        type=str,
        default="/home/kms/LLaVA_calibration/experiments/pope_ais_decomposition_v1/head_eval_ais_fp_vs_tp_yes.csv",
    )
    ap.add_argument("--faithful_topk", type=int, default=16)
    ap.add_argument("--harmful_topk", type=int, default=16)
    ap.add_argument("--faithful_auc_min", type=float, default=0.70)
    ap.add_argument("--harmful_auc_min", type=float, default=0.70)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    faith_rows = load_csv(args.faithful_head_eval_csv)
    harm_rows = load_csv(args.harmful_head_eval_csv)
    faithful = pick_faithful(
        faith_rows,
        topk=int(args.faithful_topk),
        auc_min=float(args.faithful_auc_min),
        prefer_ratio=True,
    )
    harmful = pick_harmful(
        harm_rows,
        topk=int(args.harmful_topk),
        auc_min=float(args.harmful_auc_min),
    )

    payload = {
        "inputs": {
            "faithful_head_eval_csv": os.path.abspath(args.faithful_head_eval_csv),
            "harmful_head_eval_csv": os.path.abspath(args.harmful_head_eval_csv),
            "faithful_topk": int(args.faithful_topk),
            "harmful_topk": int(args.harmful_topk),
            "faithful_auc_min": float(args.faithful_auc_min),
            "harmful_auc_min": float(args.harmful_auc_min),
        },
        "counts": {
            "n_faithful": int(len(faithful)),
            "n_harmful": int(len(harmful)),
        },
        "faithful_heads": faithful,
        "harmful_heads": harmful,
        "faithful_head_specs": to_specs(faithful),
        "harmful_head_specs": to_specs(harmful),
        "faithful_heads_by_layer": to_layer_map(faithful),
        "harmful_heads_by_layer": to_layer_map(harmful),
    }

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[saved]", out_path)


if __name__ == "__main__":
    main()

