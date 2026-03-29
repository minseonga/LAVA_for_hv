#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_bool(x: Any) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def parse_num_list(s: str, tp=float) -> List[Any]:
    out: List[Any] = []
    for t in str(s).split(","):
        tt = str(t).strip()
        if tt == "":
            continue
        out.append(tp(tt))
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in keys})


def load_candidates(path: str, vpmi_col: str) -> Dict[str, List[Dict[str, Any]]]:
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid == "":
                continue
            cand = {
                "id": sid,
                "cand_idx": int(float(r.get("cand_idx") or 0)),
                "is_champion": parse_bool(r.get("is_champion")),
                "is_correct_eval": parse_bool(r.get("is_correct_eval")),
                "s_full": safe_float(r.get("s_full")),
                "vpmi": safe_float(r.get(vpmi_col)),
            }
            by_id[sid].append(cand)
    return by_id


def pct_rank(vals: Sequence[float], v: float) -> float:
    if len(vals) == 0:
        return 0.5
    return float(sum(1 for x in vals if x <= v) / len(vals))


def eval_cfg(
    by_id: Dict[str, List[Dict[str, Any]]],
    alpha: float,
    beta: float,
    delta: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ids = sorted(by_id.keys())
    gain = 0
    harm = 0
    same = 0
    switch = 0
    detail: List[Dict[str, Any]] = []

    for sid in ids:
        cands = by_id[sid]
        champs = [c for c in cands if bool(c.get("is_champion"))]
        if len(champs) == 0:
            continue
        champ = champs[0]
        if champ.get("s_full") is None:
            continue
        c_pool = [c for c in cands if c.get("s_full") is not None and float(c["s_full"]) >= float(champ["s_full"]) - float(delta)]
        if len(c_pool) == 0:
            c_pool = [champ]

        v_list = [float(c["vpmi"]) for c in c_pool if c.get("vpmi") is not None]
        score_best = -1e30
        best = champ
        for c in c_pool:
            sc = float(c["s_full"])
            if c.get("vpmi") is not None:
                rp = pct_rank(v_list, float(c["vpmi"])) if len(v_list) > 0 else 0.5
                sc += float(alpha) * float(c["vpmi"]) + float(beta) * float(rp - 0.5)
            if sc > score_best:
                score_best = sc
                best = c

        b_ok = bool(champ.get("is_correct_eval"))
        f_ok = bool(best.get("is_correct_eval"))
        did_switch = int(best.get("cand_idx", -1)) != int(champ.get("cand_idx", -1))
        if did_switch:
            switch += 1
        outcome = "same"
        if b_ok != f_ok:
            if (not b_ok) and f_ok:
                gain += 1
                outcome = "gain"
            else:
                harm += 1
                outcome = "harm"
        else:
            same += 1
        detail.append(
            {
                "id": sid,
                "champ_idx": int(champ.get("cand_idx", -1)),
                "final_idx": int(best.get("cand_idx", -1)),
                "switched": bool(did_switch),
                "champ_ok": bool(b_ok),
                "final_ok": bool(f_ok),
                "outcome": outcome,
            }
        )

    n = len(detail)
    base_acc = (None if n == 0 else float(sum(1 for r in detail if bool(r["champ_ok"])) / n))
    final_acc = (None if n == 0 else float(sum(1 for r in detail if bool(r["final_ok"])) / n))
    row = {
        "alpha": float(alpha),
        "beta": float(beta),
        "delta": float(delta),
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": (None if base_acc is None or final_acc is None else float(final_acc - base_acc)),
        "gain": int(gain),
        "harm": int(harm),
        "net": int(gain - harm),
        "switch": int(switch),
        "switch_rate": (None if n == 0 else float(switch / n)),
        "precision_gain": (None if switch == 0 else float(gain / switch)),
    }
    return row, detail


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline APC proxy from beam candidate pool.")
    ap.add_argument("--per_candidate_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--vpmi_col", type=str, default="vpmi_core_mean")
    ap.add_argument("--alpha_list", type=str, default="0.0,0.1,0.2,0.4,0.8")
    ap.add_argument("--beta_list", type=str, default="0.0,0.05,0.1,0.2")
    ap.add_argument("--delta_list", type=str, default="0.0,0.5,1.0,1.5,2.0")
    ap.add_argument("--save_topk", type=int, default=5)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    by_id = load_candidates(os.path.abspath(args.per_candidate_csv), vpmi_col=str(args.vpmi_col))
    alphas = [float(x) for x in parse_num_list(args.alpha_list, float)]
    betas = [float(x) for x in parse_num_list(args.beta_list, float)]
    deltas = [float(x) for x in parse_num_list(args.delta_list, float)]

    rows: List[Dict[str, Any]] = []
    detail_map: Dict[Tuple[float, float, float], List[Dict[str, Any]]] = {}
    for a in alphas:
        for b in betas:
            for d in deltas:
                r, det = eval_cfg(by_id, alpha=a, beta=b, delta=d)
                rows.append(r)
                detail_map[(a, b, d)] = det

    rows.sort(
        key=lambda r: (
            int(r.get("net", -10**9)),
            float(r.get("delta_acc", -1e9) if r.get("delta_acc") is not None else -1e9),
            float(r.get("precision_gain", -1e9) if r.get("precision_gain") is not None else -1e9),
            -float(r.get("switch_rate", 1e9) if r.get("switch_rate") is not None else 1e9),
        ),
        reverse=True,
    )
    write_csv(os.path.join(out_dir, "apc_proxy_sweep.csv"), rows)

    topk = int(max(1, args.save_topk))
    top_rows = rows[:topk]
    write_csv(os.path.join(out_dir, "topk_summary.csv"), top_rows)
    for i, r in enumerate(top_rows, 1):
        key = (float(r["alpha"]), float(r["beta"]), float(r["delta"]))
        det = detail_map.get(key, [])
        write_csv(os.path.join(out_dir, f"top{i}_detail_a{key[0]}_b{key[1]}_d{key[2]}.csv"), det)

    summary = {
        "inputs": {
            "per_candidate_csv": os.path.abspath(args.per_candidate_csv),
            "vpmi_col": str(args.vpmi_col),
            "alpha_list": alphas,
            "beta_list": betas,
            "delta_list": deltas,
            "n_ids": int(len(by_id)),
        },
        "best": (None if len(rows) == 0 else rows[0]),
        "outputs": {
            "sweep_csv": os.path.join(out_dir, "apc_proxy_sweep.csv"),
            "topk_summary_csv": os.path.join(out_dir, "topk_summary.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "apc_proxy_sweep.csv"))
    print("[saved]", os.path.join(out_dir, "topk_summary.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
