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
    return float(v)


def parse_bool(x: Any) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def parse_json_int_list(x: Any) -> List[int]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    try:
        arr = json.loads(s)
        if not isinstance(arr, list):
            return []
        out: List[int] = []
        for v in arr:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    except Exception:
        return []


def parse_json_float_list(x: Any) -> List[float]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    try:
        arr = json.loads(s)
        if not isinstance(arr, list):
            return []
        out: List[float] = []
        for v in arr:
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                out.append(float(fv))
        return out
    except Exception:
        return []


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = [(int(labels[i]), float(scores[i])) for i in range(len(labels))]
    n_pos = int(sum(1 for y, _ in pairs if y == 1))
    n_neg = int(sum(1 for y, _ in pairs if y == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    idxs = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(idxs):
        j = i + 1
        while j < len(idxs) and pairs[idxs[j]][1] == pairs[idxs[i]][1]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[idxs[k]] = float(avg_rank)
        i = j
    sum_pos = float(sum(ranks[i] for i in range(len(pairs)) if pairs[i][0] == 1))
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def ks_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pos = sorted(float(scores[i]) for i in range(len(scores)) if int(labels[i]) == 1)
    neg = sorted(float(scores[i]) for i in range(len(scores)) if int(labels[i]) == 0)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    support = sorted(set(pos + neg))
    i = 0
    j = 0
    dmax = 0.0
    for v in support:
        while i < n_pos and pos[i] <= v:
            i += 1
        while j < n_neg and neg[j] <= v:
            j += 1
        dmax = max(dmax, abs(float(i / n_pos) - float(j / n_neg)))
    return float(dmax)


def summarize_metric(labels: List[int], scores: List[float], metric: str) -> Dict[str, Any]:
    auc = auc_from_scores(labels, scores)
    ks = ks_from_scores(labels, scores)
    return {
        "metric": metric,
        "n": int(len(scores)),
        "auc_hall_high": auc,
        "auc_best_dir": (None if auc is None else float(max(auc, 1.0 - auc))),
        "direction": (None if auc is None else ("higher_in_hallucination" if auc >= 0.5 else "lower_in_hallucination")),
        "ks_hall_high": ks,
    }


def softmax(vals: List[float]) -> List[float]:
    if len(vals) == 0:
        return []
    m = max(vals)
    ex = [math.exp(float(v) - float(m)) for v in vals]
    s = sum(ex)
    if s <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [float(v / s) for v in ex]


def norm_nonneg(vals: List[float]) -> List[float]:
    if len(vals) == 0:
        return []
    x = [max(0.0, float(v)) for v in vals]
    s = sum(x)
    if s <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [float(v / s) for v in x]


def main() -> None:
    ap = argparse.ArgumentParser(description="AIS decomposition: per-layer / per-head / per-patch / answer-token conditioned.")
    ap.add_argument("--per_layer_trace_csv", type=str, required=True)
    ap.add_argument("--per_head_trace_csv", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--early_start", type=int, default=0)
    ap.add_argument("--early_end", type=int, default=15)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--ers_metric", type=str, default="yes_attn_vis_ratio")
    ap.add_argument("--ais_sim_metric", type=str, default="yes_sim_local_max")
    ap.add_argument("--head_metric", type=str, default="head_attn_vis_ratio")
    ap.add_argument("--head_weight_norm", type=str, default="softmax", choices=["softmax", "nonneg_norm"])
    ap.add_argument("--patch_idx_col", type=str, default="yes_sim_local_topk_idx_json")
    ap.add_argument("--patch_weight_col", type=str, default="yes_sim_local_topk_weight_json")
    ap.add_argument("--patch_weight_norm", type=str, default="softmax", choices=["softmax", "nonneg_norm"])
    ap.add_argument("--patch_grid_size", type=int, default=24, help="Visual patch grid width/height if square.")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows_layer = read_csv(os.path.abspath(args.per_layer_trace_csv))
    if len(rows_layer) == 0:
        raise RuntimeError("No rows in per_layer_trace_csv.")
    rows_head: List[Dict[str, Any]] = []
    if str(args.per_head_trace_csv).strip() != "" and os.path.isfile(os.path.abspath(args.per_head_trace_csv)):
        rows_head = read_csv(os.path.abspath(args.per_head_trace_csv))

    early_lo = int(min(args.early_start, args.early_end))
    early_hi = int(max(args.early_start, args.early_end))
    late_lo = int(min(args.late_start, args.late_end))
    late_hi = int(max(args.late_start, args.late_end))

    # ERS per sample (early range)
    ers_vals_by_id: Dict[str, List[float]] = defaultdict(list)
    label_by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows_layer:
        sid = str(r.get("id") or "").strip()
        li = safe_float(r.get("block_layer_idx"))
        if sid == "" or li is None:
            continue
        layer = int(li)
        if early_lo <= layer <= early_hi:
            v = safe_float(r.get(args.ers_metric))
            if v is not None:
                ers_vals_by_id[sid].append(float(v))
        if sid not in label_by_id:
            label_by_id[sid] = {
                "id": sid,
                "image_id": r.get("image_id"),
                "answer_gt": r.get("answer_gt"),
                "answer_pred": r.get("answer_pred"),
                "is_fp_hallucination": bool(parse_bool(r.get("is_fp_hallucination"))),
                "is_tp_yes": bool(parse_bool(r.get("is_tp_yes"))),
                "yesno_token_idx": safe_float(r.get("yesno_token_idx")),
                "yesno_token_str": r.get("yesno_token_str"),
            }
    ers_mean_by_id: Dict[str, float] = {}
    for sid, vals in ers_vals_by_id.items():
        if len(vals) > 0:
            ers_mean_by_id[sid] = float(sum(vals) / len(vals))

    # Source map to recover full per-layer fields (patch idx/weight json, etc.).
    src_layer_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows_layer:
        sid = str(r.get("id") or "").strip()
        li = safe_float(r.get("block_layer_idx"))
        if sid == "" or li is None:
            continue
        src_layer_map[(sid, int(li))] = r

    # Per-layer AIS rows (late layers only)
    per_layer_ais_rows: List[Dict[str, Any]] = []
    for r in rows_layer:
        sid = str(r.get("id") or "").strip()
        li = safe_float(r.get("block_layer_idx"))
        sim = safe_float(r.get(args.ais_sim_metric))
        ers = ers_mean_by_id.get(sid)
        if sid == "" or li is None or sim is None or ers is None:
            continue
        layer = int(li)
        if not (late_lo <= layer <= late_hi):
            continue
        ais_layer = float(sim / max(float(args.eps), float(ers)))
        base = label_by_id.get(sid, {"id": sid})
        rec = dict(base)
        rec.update(
            {
                "block_layer_idx": int(layer),
                "ers_mean": float(ers),
                "late_sim": float(sim),
                "ais_layer": float(ais_layer),
                "ais_metric": str(args.ais_sim_metric),
            }
        )
        per_layer_ais_rows.append(rec)

    if len(per_layer_ais_rows) == 0:
        raise RuntimeError("No per-layer AIS rows. Check layer range and metric names.")

    # Per-sample AIS summary
    ais_by_id: Dict[str, List[float]] = defaultdict(list)
    for r in per_layer_ais_rows:
        ais_by_id[str(r["id"])].append(float(r["ais_layer"]))
    per_sample_ais_rows: List[Dict[str, Any]] = []
    for sid, vals in ais_by_id.items():
        meta = label_by_id.get(sid, {"id": sid})
        per_sample_ais_rows.append(
            {
                **meta,
                "ers_mean": ers_mean_by_id.get(sid),
                "ais_mean": float(sum(vals) / len(vals)),
                "ais_max": float(max(vals)),
                "ais_min": float(min(vals)),
                "n_late_layers": int(len(vals)),
            }
        )

    # Layer-level separation
    layer_eval_rows: List[Dict[str, Any]] = []
    layers = sorted(set(int(r["block_layer_idx"]) for r in per_layer_ais_rows))
    for layer in layers:
        labels: List[int] = []
        scores: List[float] = []
        for r in per_layer_ais_rows:
            if int(r["block_layer_idx"]) != int(layer):
                continue
            if not (bool(r.get("is_fp_hallucination")) or bool(r.get("is_tp_yes"))):
                continue
            labels.append(1 if bool(r.get("is_fp_hallucination")) else 0)
            scores.append(float(r["ais_layer"]))
        if len(scores) < 20:
            continue
        sm = summarize_metric(labels, scores, metric=f"ais_layer_l{int(layer)}")
        sm["block_layer_idx"] = int(layer)
        sm["comparison"] = "fp_hall_vs_tp_yes"
        layer_eval_rows.append(sm)
    layer_eval_rows = sorted(layer_eval_rows, key=lambda x: float(x.get("auc_best_dir") or -1.0), reverse=True)

    # Token-conditioned (for POPE mostly answer-token yes/no)
    token_group_stats: List[Dict[str, Any]] = []
    by_tok: Dict[str, List[float]] = defaultdict(list)
    by_tok_fp: Dict[str, List[float]] = defaultdict(list)
    by_tok_tp: Dict[str, List[float]] = defaultdict(list)
    for r in per_sample_ais_rows:
        tok = str(r.get("yesno_token_str") or "")
        v = safe_float(r.get("ais_mean"))
        if v is None:
            continue
        by_tok[tok].append(float(v))
        if bool(r.get("is_fp_hallucination")):
            by_tok_fp[tok].append(float(v))
        if bool(r.get("is_tp_yes")):
            by_tok_tp[tok].append(float(v))
    for tok, vals in by_tok.items():
        token_group_stats.append(
            {
                "yesno_token_str": tok,
                "n": int(len(vals)),
                "ais_mean": float(sum(vals) / len(vals)),
                "n_fp": int(len(by_tok_fp.get(tok, []))),
                "n_tp_yes": int(len(by_tok_tp.get(tok, []))),
                "ais_mean_fp": (None if len(by_tok_fp.get(tok, [])) == 0 else float(sum(by_tok_fp[tok]) / len(by_tok_fp[tok]))),
                "ais_mean_tp_yes": (None if len(by_tok_tp.get(tok, [])) == 0 else float(sum(by_tok_tp[tok]) / len(by_tok_tp[tok]))),
            }
        )
    token_group_stats = sorted(token_group_stats, key=lambda x: int(x.get("n", 0)), reverse=True)

    # Patch contribution decomposition
    patch_contrib_rows: List[Dict[str, Any]] = []
    patch_agg: Dict[int, Dict[str, float]] = defaultdict(lambda: {"sum_all": 0.0, "n_all": 0.0, "sum_fp": 0.0, "n_fp": 0.0, "sum_tp": 0.0, "n_tp": 0.0})
    for r in per_layer_ais_rows:
        sid = str(r.get("id") or "").strip()
        layer = int(r.get("block_layer_idx"))
        src = src_layer_map.get((sid, layer), {})
        idxs = parse_json_int_list(src.get(args.patch_idx_col))
        wts = parse_json_float_list(src.get(args.patch_weight_col))
        n = int(min(len(idxs), len(wts)))
        if n <= 0:
            continue
        idxs = idxs[:n]
        wts = wts[:n]
        wn = softmax(wts) if str(args.patch_weight_norm) == "softmax" else norm_nonneg(wts)
        ais = float(r["ais_layer"])
        for pidx, ww in zip(idxs, wn):
            contrib = float(ais * float(ww))
            rec = {
                "id": r["id"],
                "block_layer_idx": r["block_layer_idx"],
                "patch_idx": int(pidx),
                "patch_weight_normed": float(ww),
                "ais_layer": float(ais),
                "patch_contrib": float(contrib),
                "is_fp_hallucination": bool(r.get("is_fp_hallucination")),
                "is_tp_yes": bool(r.get("is_tp_yes")),
            }
            patch_contrib_rows.append(rec)
            ag = patch_agg[int(pidx)]
            ag["sum_all"] += contrib
            ag["n_all"] += 1.0
            if bool(r.get("is_fp_hallucination")):
                ag["sum_fp"] += contrib
                ag["n_fp"] += 1.0
            if bool(r.get("is_tp_yes")):
                ag["sum_tp"] += contrib
                ag["n_tp"] += 1.0

    patch_summary_rows: List[Dict[str, Any]] = []
    g = int(max(1, args.patch_grid_size))
    for pidx, ag in patch_agg.items():
        row = int(int(pidx) // g)
        col = int(int(pidx) % g)
        m_all = float(ag["sum_all"] / ag["n_all"]) if ag["n_all"] > 0 else None
        m_fp = float(ag["sum_fp"] / ag["n_fp"]) if ag["n_fp"] > 0 else None
        m_tp = float(ag["sum_tp"] / ag["n_tp"]) if ag["n_tp"] > 0 else None
        diff = (None if m_fp is None or m_tp is None else float(m_fp - m_tp))
        patch_summary_rows.append(
            {
                "patch_idx": int(pidx),
                "row": int(row),
                "col": int(col),
                "mean_contrib_all": m_all,
                "mean_contrib_fp": m_fp,
                "mean_contrib_tp_yes": m_tp,
                "diff_fp_minus_tp_yes": diff,
                "n_all": int(ag["n_all"]),
                "n_fp": int(ag["n_fp"]),
                "n_tp_yes": int(ag["n_tp"]),
            }
        )
    patch_summary_rows = sorted(
        patch_summary_rows,
        key=lambda x: abs(float(x.get("diff_fp_minus_tp_yes") or 0.0)),
        reverse=True,
    )

    # Per-head contribution decomposition (requires per_head trace)
    per_head_contrib_rows: List[Dict[str, Any]] = []
    head_eval_rows: List[Dict[str, Any]] = []
    head_rank_rows: List[Dict[str, Any]] = []
    if len(rows_head) > 0:
        ais_by_id_layer: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for r in per_layer_ais_rows:
            ais_by_id_layer[(str(r["id"]), int(r["block_layer_idx"]))] = r

        # group heads by (id, layer)
        bucket: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for h in rows_head:
            sid = str(h.get("id") or "").strip()
            li = safe_float(h.get("block_layer_idx"))
            if sid == "" or li is None:
                continue
            layer = int(li)
            if not (late_lo <= layer <= late_hi):
                continue
            bucket[(sid, layer)].append(h)

        for key, hs in bucket.items():
            sid, layer = key
            base = ais_by_id_layer.get((sid, layer))
            if base is None:
                continue
            ais_layer = safe_float(base.get("ais_layer"))
            if ais_layer is None:
                continue
            raw = []
            for h in hs:
                v = safe_float(h.get(args.head_metric))
                raw.append(0.0 if v is None else float(v))
            wn = softmax(raw) if str(args.head_weight_norm) == "softmax" else norm_nonneg(raw)
            for h, ww in zip(hs, wn):
                contrib = float(float(ais_layer) * float(ww))
                rec = {
                    "id": sid,
                    "block_layer_idx": int(layer),
                    "head_idx": int(float(h.get("head_idx"))),
                    "head_weight_normed": float(ww),
                    "head_raw_metric": safe_float(h.get(args.head_metric)),
                    "ais_layer": float(ais_layer),
                    "head_contrib": float(contrib),
                    "is_fp_hallucination": bool(parse_bool(base.get("is_fp_hallucination"))),
                    "is_tp_yes": bool(parse_bool(base.get("is_tp_yes"))),
                }
                per_head_contrib_rows.append(rec)

        # (layer, head) eval
        pair_keys = sorted(set((int(r["block_layer_idx"]), int(r["head_idx"])) for r in per_head_contrib_rows))
        for layer, head in pair_keys:
            labels: List[int] = []
            scores: List[float] = []
            for r in per_head_contrib_rows:
                if int(r["block_layer_idx"]) != int(layer) or int(r["head_idx"]) != int(head):
                    continue
                if not (bool(r.get("is_fp_hallucination")) or bool(r.get("is_tp_yes"))):
                    continue
                labels.append(1 if bool(r.get("is_fp_hallucination")) else 0)
                scores.append(float(r["head_contrib"]))
            if len(scores) < 20:
                continue
            sm = summarize_metric(labels, scores, metric=f"head_contrib_l{int(layer)}_h{int(head)}")
            sm["block_layer_idx"] = int(layer)
            sm["head_idx"] = int(head)
            sm["comparison"] = "fp_hall_vs_tp_yes"
            head_eval_rows.append(sm)
        head_eval_rows = sorted(head_eval_rows, key=lambda x: float(x.get("auc_best_dir") or -1.0), reverse=True)

        # global head rank (aggregate over late layers, per sample)
        by_id_head: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        label_id: Dict[str, Tuple[bool, bool]] = {}
        for r in per_head_contrib_rows:
            key = (str(r["id"]), int(r["head_idx"]))
            by_id_head[key].append(float(r["head_contrib"]))
            if str(r["id"]) not in label_id:
                label_id[str(r["id"])] = (bool(r.get("is_fp_hallucination")), bool(r.get("is_tp_yes")))
        by_head_fp: Dict[int, List[float]] = defaultdict(list)
        by_head_tp: Dict[int, List[float]] = defaultdict(list)
        for (sid, hidx), vals in by_id_head.items():
            v = float(sum(vals) / len(vals))
            lab = label_id.get(sid, (False, False))
            if lab[0]:
                by_head_fp[int(hidx)].append(v)
            if lab[1]:
                by_head_tp[int(hidx)].append(v)
        for hidx in sorted(set(list(by_head_fp.keys()) + list(by_head_tp.keys()))):
            m_fp = (None if len(by_head_fp.get(hidx, [])) == 0 else float(sum(by_head_fp[hidx]) / len(by_head_fp[hidx])))
            m_tp = (None if len(by_head_tp.get(hidx, [])) == 0 else float(sum(by_head_tp[hidx]) / len(by_head_tp[hidx])))
            head_rank_rows.append(
                {
                    "head_idx": int(hidx),
                    "mean_contrib_fp": m_fp,
                    "mean_contrib_tp_yes": m_tp,
                    "diff_fp_minus_tp_yes": (None if m_fp is None or m_tp is None else float(m_fp - m_tp)),
                    "n_fp": int(len(by_head_fp.get(hidx, []))),
                    "n_tp_yes": int(len(by_head_tp.get(hidx, []))),
                }
            )
        head_rank_rows = sorted(head_rank_rows, key=lambda x: abs(float(x.get("diff_fp_minus_tp_yes") or 0.0)), reverse=True)

    # Overall sample-level eval for ais_mean / ais_max
    eval_rows: List[Dict[str, Any]] = []
    for metric in ["ais_mean", "ais_max"]:
        labels: List[int] = []
        scores: List[float] = []
        for r in per_sample_ais_rows:
            if not (bool(r.get("is_fp_hallucination")) or bool(r.get("is_tp_yes"))):
                continue
            v = safe_float(r.get(metric))
            if v is None:
                continue
            labels.append(1 if bool(r.get("is_fp_hallucination")) else 0)
            scores.append(float(v))
        if len(scores) < 20:
            continue
        sm = summarize_metric(labels, scores, metric=metric)
        sm["comparison"] = "fp_hall_vs_tp_yes"
        eval_rows.append(sm)
    eval_rows = sorted(eval_rows, key=lambda x: float(x.get("auc_best_dir") or -1.0), reverse=True)

    summary = {
        "inputs": {
            "per_layer_trace_csv": os.path.abspath(args.per_layer_trace_csv),
            "per_head_trace_csv": (None if str(args.per_head_trace_csv).strip() == "" else os.path.abspath(args.per_head_trace_csv)),
            "early_start": int(args.early_start),
            "early_end": int(args.early_end),
            "late_start": int(args.late_start),
            "late_end": int(args.late_end),
            "ers_metric": str(args.ers_metric),
            "ais_sim_metric": str(args.ais_sim_metric),
            "head_metric": str(args.head_metric),
            "head_weight_norm": str(args.head_weight_norm),
            "patch_idx_col": str(args.patch_idx_col),
            "patch_weight_col": str(args.patch_weight_col),
            "patch_weight_norm": str(args.patch_weight_norm),
            "patch_grid_size": int(args.patch_grid_size),
            "eps": float(args.eps),
        },
        "counts": {
            "n_layer_rows": int(len(rows_layer)),
            "n_head_rows": int(len(rows_head)),
            "n_ids_with_ers": int(len(ers_mean_by_id)),
            "n_per_layer_ais_rows": int(len(per_layer_ais_rows)),
            "n_per_sample_ais_rows": int(len(per_sample_ais_rows)),
            "n_per_head_contrib_rows": int(len(per_head_contrib_rows)),
            "n_patch_contrib_rows": int(len(patch_contrib_rows)),
        },
        "best_eval": (None if len(eval_rows) == 0 else eval_rows[0]),
        "best_layer_eval": (None if len(layer_eval_rows) == 0 else layer_eval_rows[0]),
        "best_head_eval": (None if len(head_eval_rows) == 0 else head_eval_rows[0]),
        "outputs": {
            "per_sample_ais_csv": os.path.join(out_dir, "per_sample_ais.csv"),
            "per_layer_ais_csv": os.path.join(out_dir, "per_layer_ais.csv"),
            "eval_csv": os.path.join(out_dir, "eval_ais.csv"),
            "layer_eval_csv": os.path.join(out_dir, "layer_eval_ais_fp_vs_tp_yes.csv"),
            "token_group_csv": os.path.join(out_dir, "token_group_stats.csv"),
            "per_head_contrib_csv": os.path.join(out_dir, "per_head_ais_contrib.csv"),
            "head_eval_csv": os.path.join(out_dir, "head_eval_ais_fp_vs_tp_yes.csv"),
            "head_rank_csv": os.path.join(out_dir, "head_rank_global.csv"),
            "patch_contrib_csv": os.path.join(out_dir, "per_patch_ais_contrib.csv"),
            "patch_summary_csv": os.path.join(out_dir, "patch_contrib_summary.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_sample_ais.csv"), per_sample_ais_rows)
    write_csv(os.path.join(out_dir, "per_layer_ais.csv"), per_layer_ais_rows)
    write_csv(os.path.join(out_dir, "eval_ais.csv"), eval_rows)
    write_csv(os.path.join(out_dir, "layer_eval_ais_fp_vs_tp_yes.csv"), layer_eval_rows)
    write_csv(os.path.join(out_dir, "token_group_stats.csv"), token_group_stats)
    write_csv(os.path.join(out_dir, "per_head_ais_contrib.csv"), per_head_contrib_rows)
    write_csv(os.path.join(out_dir, "head_eval_ais_fp_vs_tp_yes.csv"), head_eval_rows)
    write_csv(os.path.join(out_dir, "head_rank_global.csv"), head_rank_rows)
    write_csv(os.path.join(out_dir, "per_patch_ais_contrib.csv"), patch_contrib_rows)
    write_csv(os.path.join(out_dir, "patch_contrib_summary.csv"), patch_summary_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample_ais.csv"))
    print("[saved]", os.path.join(out_dir, "per_layer_ais.csv"))
    print("[saved]", os.path.join(out_dir, "eval_ais.csv"))
    print("[saved]", os.path.join(out_dir, "layer_eval_ais_fp_vs_tp_yes.csv"))
    print("[saved]", os.path.join(out_dir, "token_group_stats.csv"))
    print("[saved]", os.path.join(out_dir, "per_head_ais_contrib.csv"))
    print("[saved]", os.path.join(out_dir, "head_eval_ais_fp_vs_tp_yes.csv"))
    print("[saved]", os.path.join(out_dir, "head_rank_global.csv"))
    print("[saved]", os.path.join(out_dir, "per_patch_ais_contrib.csv"))
    print("[saved]", os.path.join(out_dir, "patch_contrib_summary.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
