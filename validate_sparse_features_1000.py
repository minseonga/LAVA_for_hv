#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple


TRUE_SET = {"1", "true", "t", "yes", "y"}


def as_bool(x: Any) -> bool:
    return str("" if x is None else x).strip().lower() in TRUE_SET


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if len(xs) == 0:
        return None
    qq = max(0.0, min(1.0, float(q)))
    if len(xs) == 1:
        return float(xs[0])
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def ks_distance(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    xa = sorted(float(x) for x in a if math.isfinite(float(x)))
    xb = sorted(float(x) for x in b if math.isfinite(float(x)))
    if len(xa) == 0 or len(xb) == 0:
        return None
    vals = sorted(set(xa + xb))
    ia = 0
    ib = 0
    na = float(len(xa))
    nb = float(len(xb))
    best = 0.0
    for v in vals:
        while ia < len(xa) and xa[ia] <= v:
            ia += 1
        while ib < len(xb) and xb[ib] <= v:
            ib += 1
        best = max(best, abs((ia / na) - (ib / nb)))
    return float(best)


def best_threshold(pos_vals: Sequence[float], neg_vals: Sequence[float]) -> Optional[Dict[str, Any]]:
    pos = [float(x) for x in pos_vals if math.isfinite(float(x))]
    neg = [float(x) for x in neg_vals if math.isfinite(float(x))]
    if len(pos) == 0 or len(neg) == 0:
        return None

    pooled = pos + neg
    cands: List[float] = []
    for q in [i / 100.0 for i in range(1, 100)]:
        v = quantile(pooled, q)
        if v is not None:
            cands.append(float(v))
    if len(cands) == 0:
        return None
    cands = sorted(set(cands))

    npos = float(len(pos))
    nneg = float(len(neg))
    best: Optional[Dict[str, Any]] = None

    def eval_one(direction: str, tau: float) -> Dict[str, Any]:
        if direction == "le":
            tp = sum(1 for x in pos if x <= tau)
            fp = sum(1 for x in neg if x <= tau)
        else:
            tp = sum(1 for x in pos if x >= tau)
            fp = sum(1 for x in neg if x >= tau)
        tpr = float(tp / npos)
        fpr = float(fp / nneg)
        sel = int(tp + fp)
        prec = None if sel == 0 else float(tp / sel)
        return {
            "direction": direction,
            "tau": float(tau),
            "tp": int(tp),
            "fp": int(fp),
            "tpr": tpr,
            "fpr": fpr,
            "youden": float(tpr - fpr),
            "precision": prec,
            "selected": int(sel),
            "selected_rate": float(sel / (len(pos) + len(neg))),
        }

    for tau in cands:
        for direction in ("le", "ge"):
            cur = eval_one(direction, tau)
            if best is None:
                best = cur
                continue
            key_cur = (
                float(cur["youden"]),
                float(-1e9 if cur["precision"] is None else cur["precision"]),
                -float(cur["selected_rate"]),
            )
            key_best = (
                float(best["youden"]),
                float(-1e9 if best["precision"] is None else best["precision"]),
                -float(best["selected_rate"]),
            )
            if key_cur > key_best:
                best = cur
    if best is None:
        return None

    q05 = quantile(pooled, 0.05)
    q95 = quantile(pooled, 0.95)
    if q05 is None or q95 is None or float(q95) <= float(q05):
        near = None
    else:
        eps = float(0.02 * (float(q95) - float(q05)))
        if eps <= 0.0:
            near = None
        else:
            near_cnt = sum(1 for x in pooled if abs(float(x) - float(best["tau"])) <= eps)
            near = float(near_cnt / len(pooled))

    best["near_density_eps2pct_iqr90"] = near
    best["ks"] = ks_distance(pos, neg)
    best["n_pos"] = int(len(pos))
    best["n_neg"] = int(len(neg))
    return best


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
            wr.writerow({k: r.get(k, None) for k in keys})


def parse_json_float_list(x: Any) -> List[float]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: List[float] = []
    for v in obj:
        vv = safe_float(v)
        if vv is not None:
            out.append(float(vv))
    return out


def sign_deadband(v: float, deadband: float) -> int:
    if float(v) > float(deadband):
        return 1
    if float(v) < -float(deadband):
        return -1
    return 0


def flip_count_deadband(vals: Sequence[float], deadband: float) -> int:
    prev = 0
    flips = 0
    for x in vals:
        s = sign_deadband(float(x), float(deadband))
        if s == 0:
            continue
        if prev != 0 and s != prev:
            flips += 1
        prev = s
    return int(flips)


def longest_negative_run(vals: Sequence[float]) -> int:
    best = 0
    cur = 0
    for x in vals:
        if float(x) < 0.0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def mean(vals: Sequence[float]) -> Optional[float]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if len(xs) == 0:
        return None
    return float(sum(xs) / len(xs))


def rank_desc_pct_map(items: Sequence[Dict[str, Any]], key: str) -> Dict[int, Optional[float]]:
    vals: List[Tuple[int, float]] = []
    for c in items:
        idx = int(c["idx"])
        v = safe_float(c.get(key))
        if v is None:
            continue
        vals.append((idx, float(v)))
    vals = sorted(vals, key=lambda x: float(x[1]), reverse=True)
    out: Dict[int, Optional[float]] = {int(c["idx"]): None for c in items}
    n = len(vals)
    if n == 0:
        return out
    if n == 1:
        out[int(vals[0][0])] = 1.0
        return out
    for r, (idx, _) in enumerate(vals):
        out[int(idx)] = float((n - 1 - r) / (n - 1))
    return out


def candidate_ts_features(c: Dict[str, Any], k_prefix: int, k_suffix: int, deadband: float) -> Dict[str, Optional[float]]:
    toks = [float(x) for x in (c.get("core_vpmi_toks") or []) if math.isfinite(float(x))]
    out: Dict[str, Optional[float]] = {
        "vpmi_prefix_mean_k": None,
        "vpmi_suffix_mean_k": None,
        "vpmi_suffix_min_k": None,
        "vpmi_ps_gap_k": None,
        "neg_run_len_suffix_k": None,
        "flip_rate_db": None,
    }
    if len(toks) > 0:
        kp = int(max(1, min(int(k_prefix), len(toks))))
        ks = int(max(1, min(int(k_suffix), len(toks))))
        pref = toks[:kp]
        suff = toks[-ks:]
        pref_m = mean(pref)
        suff_m = mean(suff)
        out["vpmi_prefix_mean_k"] = pref_m
        out["vpmi_suffix_mean_k"] = suff_m
        out["vpmi_suffix_min_k"] = float(min(suff))
        if pref_m is not None and suff_m is not None:
            out["vpmi_ps_gap_k"] = float(pref_m - suff_m)
        out["neg_run_len_suffix_k"] = float(longest_negative_run(suff))
        flips = flip_count_deadband(toks, deadband=float(deadband))
        out["flip_rate_db"] = float(flips / max(1, len(toks) - 1))
        return out

    # Fallback when tokenwise arrays are absent.
    cnt = safe_float(c.get("vpmi_core_sign_flip_count"))
    clen = safe_float(c.get("core_len"))
    if cnt is not None and clen is not None and float(clen) > 1:
        out["flip_rate_db"] = float(float(cnt) / max(1.0, float(clen) - 1.0))
    return out


def detect_eval_mode(row: Dict[str, Any], eval_mode: str) -> str:
    mode = str(eval_mode).strip().lower()
    if mode == "auto":
        mm = str(row.get("eval_match_mode", "")).strip().lower()
        if mm in {"strict", "heuristic"}:
            return mm
        return "heuristic"
    return "heuristic" if mode not in {"strict", "heuristic"} else mode


def load_run(run_dir: str, eval_mode: str) -> Dict[str, Dict[str, Any]]:
    per_sample_path = os.path.join(run_dir, "per_sample.csv")
    per_cand_path = os.path.join(run_dir, "per_candidate.csv")
    per_sample = list(csv.DictReader(open(per_sample_path, encoding="utf-8")))
    per_cand = list(csv.DictReader(open(per_cand_path, encoding="utf-8")))

    cands_by_sid: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_cand:
        sid = str(r.get("id", ""))
        idx_f = safe_float(r.get("cand_idx"))
        if sid == "" or idx_f is None:
            continue
        s_full = safe_float(r.get("s_full"))
        s_q = safe_float(r.get("s_ans_q"))
        s_core = safe_float(r.get("s_core_img"))
        vpmi = None if s_core is None or s_q is None else float(s_core - s_q)
        cands_by_sid.setdefault(sid, []).append(
            {
                "idx": int(idx_f),
                "text": str(r.get("text", "")),
                "short_answer": str(r.get("short_answer", "")),
                "is_champion": as_bool(r.get("is_champion", "")),
                "is_correct_eval": as_bool(r.get("is_correct_eval", "")) if str(r.get("is_correct_eval", "")).strip() != "" else None,
                "is_correct_heuristic": as_bool(r.get("is_correct_heuristic", "")) if str(r.get("is_correct_heuristic", "")).strip() != "" else None,
                "is_correct_strict": as_bool(r.get("is_correct_strict", "")) if str(r.get("is_correct_strict", "")).strip() != "" else None,
                "s_full": s_full,
                "s_ans_q": s_q,
                "s_core_img": s_core,
                "vpmi": vpmi,
                "margin_core_img_min": safe_float(r.get("margin_core_img_min")),
                "vpmi_core_min_pos_norm": safe_float(r.get("vpmi_core_min_pos_norm")),
                "vpmi_core_sign_flip_count": safe_float(r.get("vpmi_core_sign_flip_count")),
                "core_len": safe_float(r.get("core_len")),
                "core_vpmi_toks": parse_json_float_list(r.get("core_vpmi_toks_json")),
            }
        )

    out: Dict[str, Dict[str, Any]] = {}
    for r in per_sample:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", ""))
        if sid == "":
            continue
        cands = cands_by_sid.get(sid, [])
        if len(cands) == 0:
            continue
        champ = next((c for c in cands if bool(c.get("is_champion", False))), None)
        if champ is None:
            finite = [c for c in cands if c.get("s_full") is not None]
            if len(finite) == 0:
                continue
            champ = max(finite, key=lambda x: float(x["s_full"]))

        mode = detect_eval_mode(r, eval_mode=eval_mode)
        if mode == "strict" and str(r.get("is_success_strict", "")).strip() != "":
            base_ok = as_bool(r.get("is_success_strict", ""))
        elif mode == "heuristic" and str(r.get("is_success_heuristic", "")).strip() != "":
            base_ok = as_bool(r.get("is_success_heuristic", ""))
        elif str(r.get("is_success", "")).strip() != "":
            base_ok = as_bool(r.get("is_success", ""))
        else:
            base_ok = False

        pool = [c for c in cands if int(c["idx"]) != int(champ["idx"])]
        for c in pool:
            if c["is_correct_eval"] is None:
                if mode == "strict" and c["is_correct_strict"] is not None:
                    c["is_correct_eval"] = bool(c["is_correct_strict"])
                elif mode == "heuristic" and c["is_correct_heuristic"] is not None:
                    c["is_correct_eval"] = bool(c["is_correct_heuristic"])
                else:
                    c["is_correct_eval"] = False

        out[sid] = {
            "sid": sid,
            "question": str(r.get("question", "")),
            "answer": str(r.get("answer", "")),
            "mode": mode,
            "base_ok": bool(base_ok),
            "champ": champ,
            "pool": pool,
            "champ_illusion_gap": safe_float(r.get("illusion_gap_format_minus_core")),
        }
    return out


def collect_threshold_rows(stage: str, label_pos: str, label_neg: str, feat_map: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for feat, gg in feat_map.items():
        pos_vals = gg["pos"]
        neg_vals = gg["neg"]
        best = best_threshold(pos_vals, neg_vals)
        if best is None:
            continue
        near = best.get("near_density_eps2pct_iqr90")
        ks = best.get("ks")
        sparse_score = None if near is None or ks is None else float(float(ks) * (1.0 - float(near)))
        row = {
            "stage": stage,
            "feature": feat,
            "pos_label": label_pos,
            "neg_label": label_neg,
            "sparse_score": sparse_score,
            **best,
        }
        rows.append(row)
    rows.sort(
        key=lambda r: (
            -(safe_float(r.get("sparse_score")) or -1e18),
            -(safe_float(r.get("ks")) or -1e18),
            -(safe_float(r.get("youden")) or -1e18),
        )
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate sparse/relative features on 1000-shot adaptive pool data.")
    ap.add_argument("--greedy_dir", type=str, required=True)
    ap.add_argument("--beam_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--k_prefix", type=int, default=2)
    ap.add_argument("--k_suffix", type=int, default=2)
    ap.add_argument("--flip_deadband", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    greedy = load_run(args.greedy_dir, eval_mode=str(args.eval_mode))
    beam = load_run(args.beam_dir, eval_mode=str(args.eval_mode))
    ids_all = sorted(set(greedy.keys()) & set(beam.keys()))

    gate_feats: Dict[str, Dict[str, List[float]]] = {}
    sel_feats: Dict[str, Dict[str, List[float]]] = {}

    def add_feat(bucket: Dict[str, Dict[str, List[float]]], name: str, val: Optional[float], is_pos: bool) -> None:
        if val is None or not math.isfinite(float(val)):
            return
        if name not in bucket:
            bucket[name] = {"pos": [], "neg": []}
        bucket[name]["pos" if is_pos else "neg"].append(float(val))

    n_gain = 0
    n_harm = 0
    n_better = 0
    n_worse = 0
    n_gate_tok_missing = 0
    n_sel_tok_missing = 0

    for sid in ids_all:
        gs = greedy[sid]
        bs = beam[sid]

        gain = bool((not gs["base_ok"]) and bs["base_ok"])
        harm = bool(gs["base_ok"] and (not bs["base_ok"]))
        if gain or harm:
            if gain:
                n_gain += 1
            if harm:
                n_harm += 1
            gc = gs["champ"]
            gts = candidate_ts_features(
                gc,
                k_prefix=int(args.k_prefix),
                k_suffix=int(args.k_suffix),
                deadband=float(args.flip_deadband),
            )
            if len(gc.get("core_vpmi_toks") or []) == 0:
                n_gate_tok_missing += 1
            is_pos = bool(gain)
            add_feat(gate_feats, "g_vpmi_prefix_mean_k", gts["vpmi_prefix_mean_k"], is_pos)
            add_feat(gate_feats, "g_vpmi_suffix_mean_k", gts["vpmi_suffix_mean_k"], is_pos)
            add_feat(gate_feats, "g_vpmi_suffix_min_k", gts["vpmi_suffix_min_k"], is_pos)
            add_feat(gate_feats, "g_vpmi_ps_gap_k", gts["vpmi_ps_gap_k"], is_pos)
            add_feat(gate_feats, "g_neg_run_len_suffix_k", gts["neg_run_len_suffix_k"], is_pos)
            add_feat(gate_feats, "g_flip_rate_db", gts["flip_rate_db"], is_pos)
            add_feat(gate_feats, "g_champ_vpmi", safe_float(gc.get("vpmi")), is_pos)
            add_feat(gate_feats, "g_margin_core_img_min", safe_float(gc.get("margin_core_img_min")), is_pos)
            add_feat(gate_feats, "g_vpmi_core_min_pos_norm", safe_float(gc.get("vpmi_core_min_pos_norm")), is_pos)
            add_feat(gate_feats, "g_illusion_gap", safe_float(gs.get("champ_illusion_gap")), is_pos)

        bc = bs["champ"]
        champ_ok = bool(bs["base_ok"])
        all_items = [bc] + list(bs["pool"])
        rank_vpmi = rank_desc_pct_map(all_items, "vpmi")
        rank_margin = rank_desc_pct_map(all_items, "margin_core_img_min")
        bcts = candidate_ts_features(
            bc,
            k_prefix=int(args.k_prefix),
            k_suffix=int(args.k_suffix),
            deadband=float(args.flip_deadband),
        )
        if len(bc.get("core_vpmi_toks") or []) == 0:
            n_sel_tok_missing += 1

        for c in bs["pool"]:
            cand_ok = bool(c.get("is_correct_eval", False))
            better = bool((not champ_ok) and cand_ok)
            worse = bool(champ_ok and (not cand_ok))
            if not better and not worse:
                continue
            if better:
                n_better += 1
            if worse:
                n_worse += 1

            cts = candidate_ts_features(
                c,
                k_prefix=int(args.k_prefix),
                k_suffix=int(args.k_suffix),
                deadband=float(args.flip_deadband),
            )
            is_pos = bool(better)

            cv = safe_float(c.get("vpmi"))
            bv = safe_float(bc.get("vpmi"))
            cm = safe_float(c.get("margin_core_img_min"))
            bm = safe_float(bc.get("margin_core_img_min"))
            d_v = None if cv is None or bv is None else float(cv - bv)
            d_m = None if cm is None or bm is None else float(cm - bm)
            d_ps = (
                None
                if cts["vpmi_ps_gap_k"] is None or bcts["vpmi_ps_gap_k"] is None
                else float(float(cts["vpmi_ps_gap_k"]) - float(bcts["vpmi_ps_gap_k"]))
            )
            d_pref = (
                None
                if cts["vpmi_prefix_mean_k"] is None or bcts["vpmi_prefix_mean_k"] is None
                else float(float(cts["vpmi_prefix_mean_k"]) - float(bcts["vpmi_prefix_mean_k"]))
            )
            d_suff_mean = (
                None
                if cts["vpmi_suffix_mean_k"] is None or bcts["vpmi_suffix_mean_k"] is None
                else float(float(cts["vpmi_suffix_mean_k"]) - float(bcts["vpmi_suffix_mean_k"]))
            )
            d_smin = (
                None
                if cts["vpmi_suffix_min_k"] is None or bcts["vpmi_suffix_min_k"] is None
                else float(float(cts["vpmi_suffix_min_k"]) - float(bcts["vpmi_suffix_min_k"]))
            )
            d_run = (
                None
                if cts["neg_run_len_suffix_k"] is None or bcts["neg_run_len_suffix_k"] is None
                else float(float(cts["neg_run_len_suffix_k"]) - float(bcts["neg_run_len_suffix_k"]))
            )
            d_flip = (
                None
                if cts["flip_rate_db"] is None or bcts["flip_rate_db"] is None
                else float(float(cts["flip_rate_db"]) - float(bcts["flip_rate_db"]))
            )

            cand_rank_v = rank_vpmi.get(int(c["idx"]))
            champ_rank_v = rank_vpmi.get(int(bc["idx"]))
            cand_rank_m = rank_margin.get(int(c["idx"]))
            champ_rank_m = rank_margin.get(int(bc["idx"]))
            d_rank_v = None if cand_rank_v is None or champ_rank_v is None else float(cand_rank_v - champ_rank_v)
            d_rank_m = None if cand_rank_m is None or champ_rank_m is None else float(cand_rank_m - champ_rank_m)

            add_feat(sel_feats, "d_vpmi", d_v, is_pos)
            add_feat(sel_feats, "d_margin_core_img_min", d_m, is_pos)
            add_feat(sel_feats, "cand_rank_vpmi_pct", cand_rank_v, is_pos)
            add_feat(sel_feats, "champ_rank_vpmi_pct", champ_rank_v, is_pos)
            add_feat(sel_feats, "d_rank_vpmi_pct", d_rank_v, is_pos)
            add_feat(sel_feats, "cand_rank_margin_pct", cand_rank_m, is_pos)
            add_feat(sel_feats, "d_rank_margin_pct", d_rank_m, is_pos)
            add_feat(sel_feats, "c_vpmi_prefix_mean_k", cts["vpmi_prefix_mean_k"], is_pos)
            add_feat(sel_feats, "c_vpmi_suffix_mean_k", cts["vpmi_suffix_mean_k"], is_pos)
            add_feat(sel_feats, "c_vpmi_ps_gap_k", cts["vpmi_ps_gap_k"], is_pos)
            add_feat(sel_feats, "c_vpmi_suffix_min_k", cts["vpmi_suffix_min_k"], is_pos)
            add_feat(sel_feats, "c_neg_run_len_suffix_k", cts["neg_run_len_suffix_k"], is_pos)
            add_feat(sel_feats, "c_flip_rate_db", cts["flip_rate_db"], is_pos)
            add_feat(sel_feats, "d_vpmi_prefix_mean_k", d_pref, is_pos)
            add_feat(sel_feats, "d_vpmi_suffix_mean_k", d_suff_mean, is_pos)
            add_feat(sel_feats, "d_vpmi_ps_gap_k", d_ps, is_pos)
            add_feat(sel_feats, "d_vpmi_suffix_min_k", d_smin, is_pos)
            add_feat(sel_feats, "d_neg_run_len_suffix_k", d_run, is_pos)
            add_feat(sel_feats, "d_flip_rate_db", d_flip, is_pos)

    gate_rows = collect_threshold_rows("gate", "gain", "harm", gate_feats)
    sel_rows = collect_threshold_rows("selector", "better", "worse", sel_feats)

    gate_csv = os.path.join(args.out_dir, "gate_feature_ranking.csv")
    sel_csv = os.path.join(args.out_dir, "selector_feature_ranking.csv")
    write_csv(gate_csv, gate_rows)
    write_csv(sel_csv, sel_rows)

    summary = {
        "inputs": {
            "greedy_dir": os.path.abspath(args.greedy_dir),
            "beam_dir": os.path.abspath(args.beam_dir),
            "eval_mode": str(args.eval_mode),
            "k_prefix": int(args.k_prefix),
            "k_suffix": int(args.k_suffix),
            "flip_deadband": float(args.flip_deadband),
        },
        "counts": {
            "n_ids_intersection": int(len(ids_all)),
            "stage1_gain_n": int(n_gain),
            "stage1_harm_n": int(n_harm),
            "stage2_better_n": int(n_better),
            "stage2_worse_n": int(n_worse),
            "stage1_missing_core_vpmi_toks_n": int(n_gate_tok_missing),
            "stage2_champ_missing_core_vpmi_toks_n": int(n_sel_tok_missing),
        },
        "outputs": {
            "gate_csv": os.path.abspath(gate_csv),
            "selector_csv": os.path.abspath(sel_csv),
            "summary_json": os.path.abspath(os.path.join(args.out_dir, "summary.json")),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", gate_csv)
    print("[saved]", sel_csv)
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
