#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def parse_edges(s: str) -> List[float]:
    out: List[float] = []
    for t in [x.strip() for x in str(s).split(",") if x.strip()]:
        tl = t.lower()
        if tl in {"inf", "+inf"}:
            out.append(float("inf"))
        elif tl == "-inf":
            out.append(float("-inf"))
        else:
            out.append(float(t))
    if len(out) < 2:
        raise ValueError("Need >=2 edges for histogram bins.")
    out = sorted(out)
    return out


def bin_label(v: Optional[float], edges: List[float]) -> str:
    if v is None or not math.isfinite(float(v)):
        return "nan"
    x = float(v)
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            if x >= lo and x <= hi:
                return f"[{lo},{hi}]"
        if x >= lo and x < hi:
            return f"[{lo},{hi})"
    return "out_of_range"


def load_eval_module(path: str):
    spec = importlib.util.spec_from_file_location("eval_selector_tradeoff_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_selector_tradeoff_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def rank_desc_map(cands, key_fn) -> Dict[int, int]:
    vals: List[Tuple[int, float]] = []
    for c in cands:
        v = key_fn(c)
        if v is None:
            continue
        vv = safe_float(v)
        if vv is None:
            continue
        vals.append((int(c.idx), float(vv)))
    vals = sorted(vals, key=lambda x: float(x[1]), reverse=True)
    return {int(idx): int(i + 1) for i, (idx, _) in enumerate(vals)}


def lcp_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and int(a[i]) == int(b[i]):
        i += 1
    return int(i)


def contains_subseq(seq: List[int], sub: List[int]) -> bool:
    if len(sub) == 0 or len(seq) < len(sub):
        return False
    m = len(sub)
    for i in range(0, len(seq) - m + 1):
        if seq[i : i + m] == sub:
            return True
    return False


def parse_json_float_list(x: Any) -> List[float]:
    s = str(x or "").strip()
    if s == "":
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            out = []
            for v in arr:
                fv = safe_float(v)
                if fv is None:
                    continue
                out.append(float(fv))
            return out
    except Exception:
        pass
    return []


def std_or_none(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(max(0.0, v)))


def levenshtein_int(a: List[int], b: List[int]) -> int:
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = int(a[i - 1])
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ai == int(b[j - 1]) else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = tmp
    return int(dp[m])


def parse_json_int_list(x: Any) -> List[int]:
    s = str(x or "").strip()
    if s == "":
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            out = []
            for v in arr:
                iv = safe_float(v)
                if iv is None:
                    continue
                out.append(int(iv))
            return out
    except Exception:
        pass
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Coverage/Trigger/Selector bottleneck diagnostics")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["strict", "heuristic"])
    ap.add_argument("--policy", type=str, default="agree_vminpm_wmin_dfull_le:-0.05")
    ap.add_argument("--trigger", type=str, default="P3")
    ap.add_argument("--vpmi_gap_bins", type=str, default="-inf,-2,-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1,2,inf")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    eval_mod = load_eval_module(os.path.join(os.path.dirname(__file__), "eval_selector_tradeoff.py"))
    samples = eval_mod.load_samples(in_dir, str(args.eval_mode))

    per_sample_rows = list(csv.DictReader(open(os.path.join(in_dir, "per_sample.csv"), encoding="utf-8")))
    per_cand_rows = list(csv.DictReader(open(os.path.join(in_dir, "per_candidate.csv"), encoding="utf-8")))
    ps_by_id: Dict[str, Dict[str, Any]] = {str(r.get("id", "")): r for r in per_sample_rows}
    pc_by_sid_idx: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in per_cand_rows:
        sid = str(r.get("id", ""))
        idxf = safe_float(r.get("cand_idx"))
        if idxf is None:
            continue
        pc_by_sid_idx[(sid, int(idxf))] = r

    have_gt_logp = any("correct_logp_img_mean" in r for r in per_sample_rows)
    have_gt_prefix = any("correct_prefix3_logp_img_mean" in r for r in per_sample_rows)
    have_gt_rank = any("correct_first_token_rank_img" in r for r in per_sample_rows)
    have_gt_tok_ids = any("correct_variant_token_ids_json" in r for r in per_sample_rows)
    have_token_ids = any("token_ids_json" in r for r in per_cand_rows)
    have_core_vpmi_tokens = any("core_vpmi_toks_json" in r for r in per_cand_rows)
    have_margin = any("margin_core_img_min" in r for r in per_cand_rows)

    missing_require_regen: List[str] = []
    if not have_gt_logp:
        missing_require_regen.append("correct_token_logprobs (correct_logp_img_mean)")
    if not have_gt_prefix:
        missing_require_regen.append("prefix_logprob_correct (correct_prefix3_logp_img_mean)")
    if not have_gt_rank:
        missing_require_regen.append("beam_ranking_correct first-token rank (correct_first_token_rank_img)")
    if not have_gt_tok_ids:
        missing_require_regen.append("gold token ids for pool presence (correct_variant_token_ids_json)")
    if not have_token_ids:
        missing_require_regen.append("safe_champ_token_overlap (token_ids_json)")
    if not have_core_vpmi_tokens:
        missing_require_regen.append("champ_tokenwise_vpmi (core_vpmi_toks_json)")
    if not have_margin:
        missing_require_regen.append("logit_margin_core (margin_core_img_min)")

    coverage_rows: List[Dict[str, Any]] = []
    trigger_rows: List[Dict[str, Any]] = []
    selector_none_rows: List[Dict[str, Any]] = []
    st_rows: List[Dict[str, Any]] = []

    edges = parse_edges(args.vpmi_gap_bins)
    vpmi_gap_hist: Dict[str, int] = defaultdict(int)

    counts = {
        "n_total": int(len(samples)),
        "base_wrong": 0,
        "recoverable": 0,
        "unrecoverable": 0,
        "gain": 0,
        "miss": 0,
        "trigger_fail_total": 0,
        "trigger_fail_safe_le_champ": 0,
        "trigger_fail_champ_not_fragile": 0,
        "trigger_fail_both": 0,
        "selector_none": 0,
        "selector_wrong_after_switch": 0,
    }

    for s in samples:
        sid = str(s.sid)
        base_ok = bool(s.base_ok)
        if not base_ok:
            counts["base_wrong"] += 1
        has_correct_pool = bool(any(bool(v) for v in s.safe_ok_by_idx.values()))
        if (not base_ok) and has_correct_pool:
            counts["recoverable"] += 1
        if (not base_ok) and (not has_correct_pool):
            counts["unrecoverable"] += 1

        cands = sorted(
            list(s.pool),
            key=lambda c: float(c.s_full if c.s_full is not None and math.isfinite(float(c.s_full)) else -1e18),
            reverse=True,
        )
        idx2rank_sfull = {int(c.idx): int(i + 1) for i, c in enumerate(cands)}
        correct_cands = [c for c in cands if bool(s.safe_ok_by_idx.get(int(c.idx), False))]
        correct_count = int(len(correct_cands))
        correct_best_rank_sfull = (None if correct_count == 0 else int(min(idx2rank_sfull[int(c.idx)] for c in correct_cands)))

        r_vpmi = rank_desc_map(cands, lambda x: x.vpmi)
        r_vminpm = rank_desc_map(cands, lambda x: x.vpmi_core_min_prior_masked)
        r_wmin = rank_desc_map(cands, lambda x: x.vpmi_word_min)
        correct_best_rank_vpmi = (None if correct_count == 0 else int(min(r_vpmi.get(int(c.idx), 10**9) for c in correct_cands)))
        correct_best_rank_vminpm = (None if correct_count == 0 else int(min(r_vminpm.get(int(c.idx), 10**9) for c in correct_cands)))
        correct_best_rank_wmin = (None if correct_count == 0 else int(min(r_wmin.get(int(c.idx), 10**9) for c in correct_cands)))

        # Coverage root-cause diagnostics (token presence vs sequence presence)
        ps = ps_by_id.get(sid, {})
        gt_tok_ids = parse_json_int_list(ps.get("correct_variant_token_ids_json", None))
        pool_tok_lists: List[List[int]] = []
        for c in cands:
            raw_c = pc_by_sid_idx.get((sid, int(c.idx)), {})
            pool_tok_lists.append(parse_json_int_list(raw_c.get("token_ids_json", None)))

        gt_first_in_pool_tokens = None
        gt_subseq_in_pool = None
        gt_subseq_best_rank_sfull = None
        if len(gt_tok_ids) > 0 and len(pool_tok_lists) > 0:
            first_id = int(gt_tok_ids[0])
            all_pool_tokens = set()
            for ids in pool_tok_lists:
                for t in ids:
                    all_pool_tokens.add(int(t))
            gt_first_in_pool_tokens = bool(int(first_id) in all_pool_tokens)
            found_ranks: List[int] = []
            for c in cands:
                raw_c = pc_by_sid_idx.get((sid, int(c.idx)), {})
                ids = parse_json_int_list(raw_c.get("token_ids_json", None))
                if contains_subseq(ids, gt_tok_ids):
                    found_ranks.append(int(idx2rank_sfull.get(int(c.idx), 10**9)))
            gt_subseq_in_pool = bool(len(found_ranks) > 0)
            gt_subseq_best_rank_sfull = (None if len(found_ranks) == 0 else int(min(found_ranks)))

        selected = eval_mod.select_candidate(str(args.policy), s)
        selected_exists = bool(selected is not None)
        trigger_ok = bool(selected_exists and eval_mod.switch_cond(str(args.trigger), s, selected))
        selected_ok = bool(selected_exists and s.safe_ok_by_idx.get(int(selected.idx), False))
        final_ok = bool(base_ok or (trigger_ok and selected_ok))
        if (not base_ok) and final_ok:
            counts["gain"] += 1
        if (not base_ok) and has_correct_pool and (not final_ok):
            counts["miss"] += 1

        coverage_rows.append(
            {
                "id": sid,
                "base_ok": bool(base_ok),
                "has_correct_pool": bool(has_correct_pool),
                "correct_candidate_count": int(correct_count),
                "correct_best_rank_sfull": correct_best_rank_sfull,
                "correct_best_rank_vpmi": correct_best_rank_vpmi,
                "correct_best_rank_vminpm": correct_best_rank_vminpm,
                "correct_best_rank_wmin": correct_best_rank_wmin,
                "correct_token_pool_presence_first_token": gt_first_in_pool_tokens,
                "correct_token_pool_presence_full_subseq": gt_subseq_in_pool,
                "correct_token_pool_subseq_best_rank_sfull": gt_subseq_best_rank_sfull,
                "champ_vpmi": s.champ.vpmi,
                "champ_s_core_img": s.champ.s_core,
                "champ_s_full": s.champ.s_full,
                "selected_exists": bool(selected_exists),
                "selected_ok": bool(selected_ok),
                "trigger_ok": bool(trigger_ok),
                "final_ok": bool(final_ok),
                "correct_logp_img_mean": safe_float(ps.get("correct_logp_img_mean")),
                "correct_prefix3_logp_img_mean": safe_float(ps.get("correct_prefix3_logp_img_mean")),
                "correct_first_token_rank_img": safe_float(ps.get("correct_first_token_rank_img")),
            }
        )

        if (not base_ok) and has_correct_pool and (not final_ok):
            if not selected_exists:
                counts["selector_none"] += 1
                selector_none_rows.append(
                    {
                        "id": sid,
                        "question": s.question,
                        "answer": s.answer,
                        "champ_text": s.champ.text,
                        "champ_vpmi": s.champ.vpmi,
                        "champ_s_core_img": s.champ.s_core,
                        "correct_candidate_count": int(correct_count),
                        "correct_best_rank_sfull": correct_best_rank_sfull,
                        "correct_best_rank_vpmi": correct_best_rank_vpmi,
                        "correct_best_rank_vminpm": correct_best_rank_vminpm,
                        "correct_best_rank_wmin": correct_best_rank_wmin,
                    }
                )
            else:
                if not trigger_ok:
                    counts["trigger_fail_total"] += 1
                    champ_vpmi = safe_float(s.champ.vpmi)
                    safe_vpmi = safe_float(selected.vpmi)
                    vpmi_gap = (None if champ_vpmi is None or safe_vpmi is None else float(safe_vpmi - champ_vpmi))

                    # P3-style factorized fail reason (diagnostic only).
                    cond1 = bool(champ_vpmi is not None and safe_vpmi is not None and float(safe_vpmi) > float(champ_vpmi))
                    # For diagnostic harmonization use P3 boundary on champ side.
                    cond2 = bool(champ_vpmi is not None and float(champ_vpmi) < 0.0)
                    if (not cond1) and cond2:
                        fail_type = "safe_le_champ_vpmi"
                        counts["trigger_fail_safe_le_champ"] += 1
                    elif cond1 and (not cond2):
                        fail_type = "champ_not_fragile"
                        counts["trigger_fail_champ_not_fragile"] += 1
                    else:
                        fail_type = "both"
                        counts["trigger_fail_both"] += 1

                    vpmi_gap_hist[bin_label(vpmi_gap, edges)] += 1

                    raw_champ = pc_by_sid_idx.get((sid, int(s.champ.idx)), {})
                    raw_safe = pc_by_sid_idx.get((sid, int(selected.idx)), {})
                    champ_ids = parse_json_int_list(raw_champ.get("token_ids_json"))
                    safe_ids = parse_json_int_list(raw_safe.get("token_ids_json"))
                    overlap = (None if len(champ_ids) == 0 or len(safe_ids) == 0 else int(lcp_len(champ_ids, safe_ids)))
                    champ_vpmi_toks = parse_json_float_list(raw_champ.get("core_vpmi_toks_json"))
                    safe_vpmi_toks = parse_json_float_list(raw_safe.get("core_vpmi_toks_json"))
                    champ_vpmi_std = std_or_none(champ_vpmi_toks)
                    safe_vpmi_std = std_or_none(safe_vpmi_toks)
                    overlap_ratio = None
                    lev_dist = None
                    lev_norm = None
                    tok_jaccard = None
                    if len(champ_ids) > 0 and len(safe_ids) > 0:
                        overlap_ratio = float(overlap / max(1, min(len(champ_ids), len(safe_ids))))
                        lev_dist = int(levenshtein_int(champ_ids, safe_ids))
                        lev_norm = float(lev_dist / max(1, max(len(champ_ids), len(safe_ids))))
                        aset = set(int(x) for x in champ_ids)
                        bset = set(int(x) for x in safe_ids)
                        inter = len(aset & bset)
                        union = len(aset | bset)
                        tok_jaccard = (None if union == 0 else float(inter / union))

                    trigger_rows.append(
                        {
                            "id": sid,
                            "question": s.question,
                            "answer": s.answer,
                            "fail_type": fail_type,
                            "champ_text": s.champ.text,
                            "safe_text": selected.text,
                            "safe_ok": bool(selected_ok),
                            "champ_vpmi": champ_vpmi,
                            "safe_vpmi": safe_vpmi,
                            "vpmi_gap_safe_minus_champ": vpmi_gap,
                            "vpmi_gap_bin": bin_label(vpmi_gap, edges),
                            "champ_s_core_img": s.champ.s_core,
                            "safe_s_core_img": selected.s_core,
                            "champ_margin_core_img_min": safe_float(raw_champ.get("margin_core_img_min")),
                            "safe_margin_core_img_min": safe_float(raw_safe.get("margin_core_img_min")),
                            "champ_internal_vpmi_std": champ_vpmi_std,
                            "safe_internal_vpmi_std": safe_vpmi_std,
                            "safe_champ_prefix_overlap_len": overlap,
                            "safe_champ_prefix_overlap_ratio": overlap_ratio,
                            "safe_champ_levenshtein_dist": lev_dist,
                            "safe_champ_levenshtein_norm": lev_norm,
                            "safe_champ_token_jaccard": tok_jaccard,
                            "champ_core_vpmi_toks_json": raw_champ.get("core_vpmi_toks_json", None),
                            "safe_core_vpmi_toks_json": raw_safe.get("core_vpmi_toks_json", None),
                        }
                    )
                else:
                    counts["selector_wrong_after_switch"] += 1

        # selector-trigger interaction row (recoverable only)
        if (not base_ok) and has_correct_pool:
            st_rows.append(
                {
                    "id": sid,
                    "question": s.question,
                    "answer": s.answer,
                    "correct_candidate_count": int(correct_count),
                    "correct_best_rank_vpmi": correct_best_rank_vpmi,
                    "correct_best_rank_vminpm": correct_best_rank_vminpm,
                    "correct_best_rank_wmin": correct_best_rank_wmin,
                    "view_rank_disagreement_abs": (
                        None
                        if correct_best_rank_vminpm is None or correct_best_rank_wmin is None
                        else abs(int(correct_best_rank_vminpm) - int(correct_best_rank_wmin))
                    ),
                    "selected_exists": bool(selected_exists),
                    "selected_idx": (None if not selected_exists else int(selected.idx)),
                    "selected_ok": bool(selected_ok),
                    "trigger_ok": bool(trigger_ok),
                    "final_ok": bool(final_ok),
                }
            )

    # Sample-level within-pool correlation (S_full vs VPMI_core) for dfull-gate diagnostics.
    corr_rows: List[Dict[str, Any]] = []
    for s in samples:
        xs: List[float] = []
        ys: List[float] = []
        for c in s.pool:
            if c.s_full is None or c.vpmi_core_mean is None:
                continue
            sf = safe_float(c.s_full)
            vp = safe_float(c.vpmi_core_mean)
            if sf is None or vp is None:
                continue
            xs.append(float(sf))
            ys.append(float(vp))
        if len(xs) < 2:
            continue
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        corr = None if vx <= 0 or vy <= 0 else float(cov / math.sqrt(vx * vy))
        corr_rows.append({"id": s.sid, "n": len(xs), "corr_sfull_vpmi_core_mean": corr})

    vpmi_gap_rows = [{"vpmi_gap_bin": k, "n": int(v)} for k, v in sorted(vpmi_gap_hist.items(), key=lambda x: x[0])]

    summary = {
        "inputs": {
            "in_dir": in_dir,
            "eval_mode": str(args.eval_mode),
            "policy": str(args.policy),
            "trigger": str(args.trigger),
        },
        "coverage": {
            "base_wrong": int(counts["base_wrong"]),
            "recoverable": int(counts["recoverable"]),
            "unrecoverable": int(counts["unrecoverable"]),
            "recoverable_rate_in_base_wrong": (
                None if counts["base_wrong"] == 0 else float(counts["recoverable"] / counts["base_wrong"])
            ),
            "unrecoverable_rate_in_base_wrong": (
                None if counts["base_wrong"] == 0 else float(counts["unrecoverable"] / counts["base_wrong"])
            ),
            "gain": int(counts["gain"]),
            "miss": int(counts["miss"]),
        },
        "trigger_fail": {
            "total": int(counts["trigger_fail_total"]),
            "safe_le_champ_vpmi": int(counts["trigger_fail_safe_le_champ"]),
            "champ_not_fragile": int(counts["trigger_fail_champ_not_fragile"]),
            "both": int(counts["trigger_fail_both"]),
        },
        "selector_none": {
            "n": int(counts["selector_none"]),
        },
        "selector_wrong_after_switch": {
            "n": int(counts["selector_wrong_after_switch"]),
        },
        "available_metrics": {
            "correct_token_logprobs": bool(have_gt_logp),
            "prefix_logprob_correct": bool(have_gt_prefix),
            "beam_ranking_correct_first_token": bool(have_gt_rank),
            "gold_token_ids_for_pool_presence": bool(have_gt_tok_ids),
            "safe_champ_token_overlap": bool(have_token_ids),
            "champ_tokenwise_vpmi": bool(have_core_vpmi_tokens),
            "logit_margin_core": bool(have_margin),
        },
        "missing_require_regeneration": missing_require_regen,
        "outputs": {
            "coverage_csv": os.path.join(out_dir, "coverage_diagnostics.csv"),
            "trigger_fail_csv": os.path.join(out_dir, "trigger_fail_cases.csv"),
            "vpmi_gap_hist_csv": os.path.join(out_dir, "vpmi_gap_histogram.csv"),
            "selector_none_csv": os.path.join(out_dir, "selector_none_cases.csv"),
            "selector_trigger_interaction_csv": os.path.join(out_dir, "selector_trigger_interaction.csv"),
            "sfull_vpmi_corr_csv": os.path.join(out_dir, "sfull_vs_vpmi_core_corr_per_sample.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "coverage_diagnostics.csv"), coverage_rows)
    write_csv(os.path.join(out_dir, "trigger_fail_cases.csv"), trigger_rows)
    write_csv(os.path.join(out_dir, "vpmi_gap_histogram.csv"), vpmi_gap_rows)
    write_csv(os.path.join(out_dir, "selector_none_cases.csv"), selector_none_rows)
    write_csv(os.path.join(out_dir, "selector_trigger_interaction.csv"), st_rows)
    write_csv(os.path.join(out_dir, "sfull_vs_vpmi_core_corr_per_sample.csv"), corr_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "coverage_diagnostics.csv"))
    print("[saved]", os.path.join(out_dir, "trigger_fail_cases.csv"))
    print("[saved]", os.path.join(out_dir, "vpmi_gap_histogram.csv"))
    print("[saved]", os.path.join(out_dir, "selector_none_cases.csv"))
    print("[saved]", os.path.join(out_dir, "selector_trigger_interaction.csv"))
    print("[saved]", os.path.join(out_dir, "sfull_vs_vpmi_core_corr_per_sample.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
