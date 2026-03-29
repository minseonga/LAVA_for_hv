#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple


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


def load_map(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid == "":
                continue
            out[sid] = r
    return out


def count_words(text: str) -> int:
    return int(len(re.findall(r"[A-Za-z0-9']+", str(text or ""))))


def infer_len(row: Dict[str, Any], len_col: str, text_col: str) -> Optional[int]:
    if str(len_col) != "auto":
        vv = safe_float(row.get(len_col))
        return None if vv is None else int(vv)

    for k in ["n_gen_tokens"]:
        vv = safe_float(row.get(k))
        if vv is not None:
            return int(vv)

    # Fallback: approximate length from generated text when token count is unavailable.
    text_keys = [text_col] if str(text_col) != "auto" else ["pred_text", "champ_text", "pred_answer_eval", "champ_short_answer"]
    for k in text_keys:
        if k not in row:
            continue
        n = count_words(str(row.get(k) or ""))
        if n > 0:
            return int(n)
    return None


def infer_collapse(row: Dict[str, Any], collapse_col: str) -> Optional[float]:
    if str(collapse_col) != "auto":
        return safe_float(row.get(collapse_col))

    # Prefer "lower is worse" collapse-like signals.
    for k in ["vpmi_collapse_gap_k40", "champ_vpmi_core_tail_min", "champ_vpmi_core_min", "vpmi_suffix_min_k40"]:
        v = safe_float(row.get(k))
        if v is not None:
            return float(v)
    return None


def eval_once(
    ids: List[str],
    greedy: Dict[str, Dict[str, Any]],
    beam: Dict[str, Dict[str, Any]],
    min_len: int,
    tau_collapse: float,
    len_col: str,
    collapse_col: str,
    text_col: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = len(ids)
    gain = 0
    harm = 0
    same = 0
    switched = 0
    gate_true = 0
    per_sample: List[Dict[str, Any]] = []

    for sid in ids:
        g = greedy[sid]
        b = beam[sid]
        g_ok = parse_bool(g.get("is_success"))
        b_ok = parse_bool(b.get("is_success"))

        n_tok = infer_len(g, len_col=len_col, text_col=text_col)
        collapse = infer_collapse(g, collapse_col=collapse_col)
        gate = bool(n_tok is not None and int(n_tok) >= int(min_len) and collapse is not None and float(collapse) <= float(tau_collapse))
        if gate:
            gate_true += 1

        final_ok = b_ok if gate else g_ok
        did_switch = bool(gate)
        if did_switch:
            switched += 1

        outcome = "same"
        if g_ok != final_ok:
            if (not g_ok) and final_ok:
                gain += 1
                outcome = "gain"
            elif g_ok and (not final_ok):
                harm += 1
                outcome = "harm"
        else:
            same += 1

        per_sample.append(
            {
                "id": sid,
                "n_gen_tokens_or_est": n_tok,
                "vpmi_collapse_gap_k40": collapse,
                "gate_true": bool(gate),
                "switched_to_beam": bool(did_switch),
                "greedy_ok": bool(g_ok),
                "beam_ok": bool(b_ok),
                "final_ok": bool(final_ok),
                "outcome": outcome,
            }
        )

    base_acc = float(sum(1 for sid in ids if parse_bool(greedy[sid].get("is_success"))) / max(1, n))
    final_acc = float(sum(1 for r in per_sample if bool(r["final_ok"])) / max(1, n))
    row = {
        "n": int(n),
        "min_len": int(min_len),
        "tau_collapse": float(tau_collapse),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": float(final_acc - base_acc),
        "gain": int(gain),
        "harm": int(harm),
        "net": int(gain - harm),
        "same": int(same),
        "gate_true": int(gate_true),
        "gate_rate": float(gate_true / max(1, n)),
        "switch_rate": float(switched / max(1, n)),
        "precision_gain": (None if switched == 0 else float(gain / switched)),
        "speedup_vs_fixed6": (None if gate_true == 0 else float(6.0 / (1.0 + 5.0 * (gate_true / max(1, n))))),
    }
    return row, per_sample


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline eval: length-conditioned collapse-gap gate (greedy->beam switch).")
    ap.add_argument("--greedy_per_sample", type=str, required=True)
    ap.add_argument("--beam_per_sample", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--min_len_list", type=str, default="8,10,12,15")
    ap.add_argument("--tau_list", type=str, default="-4.5,-4.0,-3.5,-3.0,-2.5,-2.0")
    ap.add_argument("--save_topk", type=int, default=5)
    ap.add_argument("--len_col", type=str, default="auto", help="Length column in greedy CSV (default: auto detect).")
    ap.add_argument(
        "--collapse_col",
        type=str,
        default="auto",
        help="Collapse-like feature column in greedy CSV (default: auto detect).",
    )
    ap.add_argument(
        "--text_col",
        type=str,
        default="auto",
        help="Fallback text column for length estimation when len_col is absent.",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    gmap = load_map(os.path.abspath(args.greedy_per_sample))
    bmap = load_map(os.path.abspath(args.beam_per_sample))
    ids = sorted(set(gmap.keys()) & set(bmap.keys()))
    if len(ids) == 0:
        raise RuntimeError("No overlapping sample ids between greedy and beam per_sample.csv.")

    min_len_list = [int(x) for x in parse_num_list(args.min_len_list, int)]
    tau_list = [float(x) for x in parse_num_list(args.tau_list, float)]

    table: List[Dict[str, Any]] = []
    detail_by_key: Dict[Tuple[int, float], List[Dict[str, Any]]] = {}
    for ml in min_len_list:
        for tau in tau_list:
            row, detail = eval_once(
                ids,
                gmap,
                bmap,
                min_len=ml,
                tau_collapse=tau,
                len_col=str(args.len_col),
                collapse_col=str(args.collapse_col),
                text_col=str(args.text_col),
            )
            table.append(row)
            detail_by_key[(int(ml), float(tau))] = detail

    table.sort(
        key=lambda r: (
            int(r.get("net", -10**9)),
            float(r.get("delta_acc", -1e9)),
            float(r.get("precision_gain", -1e9) if r.get("precision_gain") is not None else -1e9),
            -float(r.get("switch_rate", 1e9)),
        ),
        reverse=True,
    )

    write_csv(os.path.join(out_dir, "sweep_table.csv"), table)
    topk = int(max(1, args.save_topk))
    top_rows = table[:topk]
    write_csv(os.path.join(out_dir, "topk_summary.csv"), top_rows)

    for i, r in enumerate(top_rows, 1):
        ml = int(r["min_len"])
        tau = float(r["tau_collapse"])
        key = (ml, tau)
        if key not in detail_by_key:
            continue
        write_csv(os.path.join(out_dir, f"top{i}_per_sample_minlen{ml}_tau{tau}.csv"), detail_by_key[key])

    summary = {
        "inputs": {
            "greedy_per_sample": os.path.abspath(args.greedy_per_sample),
            "beam_per_sample": os.path.abspath(args.beam_per_sample),
            "n_common_ids": int(len(ids)),
            "min_len_list": min_len_list,
            "tau_list": tau_list,
            "len_col": str(args.len_col),
            "collapse_col": str(args.collapse_col),
            "text_col": str(args.text_col),
        },
        "best": (None if len(table) == 0 else table[0]),
        "outputs": {
            "sweep_table_csv": os.path.join(out_dir, "sweep_table.csv"),
            "topk_summary_csv": os.path.join(out_dir, "topk_summary.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "sweep_table.csv"))
    print("[saved]", os.path.join(out_dir, "topk_summary.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
