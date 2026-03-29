#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build POPE feature-screen v1 table for unified utility bottleneck analysis.

Focus:
- A) visual confidence (VSC/G): from VGA-style dump csv (optional), else proxy from trace
- B) early-late temporal dynamics
- C) faithful routing (from faithful head set)
- D) harmful routing (from harmful head set)
- E) guidance mismatch/context composition (proxy with available signals)
- F) oracle role stats (optional; analysis labels)

This script is offline and does not run model inference.

Design notes:
- A/B/C/D are predictor families.
- E is a composite family derived from A/C/D.
- Oracle/add-back artifacts (role/counterfactual) are treated as target/aux only
  and are not used in default unified predictor score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def jloads(s: Any, default):
    try:
        if s in (None, ""):
            return default
        return json.loads(s)
    except Exception:
        return default


def std(vals: Sequence[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    m = sum(vals) / float(len(vals))
    v = sum((x - m) * (x - m) for x in vals) / float(len(vals) - 1)
    return float(math.sqrt(max(v, 1e-12)))


def topk_mean(vals: Sequence[float], top_ratio: float = 0.2) -> float:
    if len(vals) == 0:
        return 0.0
    arr = sorted(float(v) for v in vals)
    k = max(1, int(math.ceil(len(arr) * float(top_ratio))))
    return float(sum(arr[-k:]) / float(k))


def parse_headset_specs(path: str) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    cfg = json.load(open(path, "r", encoding="utf-8"))
    faithful = set()
    harmful = set()

    def add_specs(src: Any, out: set[Tuple[int, int]]) -> None:
        if isinstance(src, list):
            for x in src:
                if isinstance(x, str) and ":" in x:
                    a, b = x.split(":", 1)
                    li, hi = safe_int(a), safe_int(b)
                    if li is not None and hi is not None:
                        out.add((int(li), int(hi)))
                elif isinstance(x, dict):
                    li, hi = safe_int(x.get("layer")), safe_int(x.get("head"))
                    if li is not None and hi is not None:
                        out.add((int(li), int(hi)))

    add_specs(cfg.get("faithful_head_specs", []), faithful)
    add_specs(cfg.get("harmful_head_specs", []), harmful)
    if not faithful:
        add_specs(cfg.get("faithful_heads", []), faithful)
    if not harmful:
        add_specs(cfg.get("harmful_heads", []), harmful)

    return faithful, harmful


def load_subset(subset_ids_csv: str, subset_gt_csv: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    id2group: Dict[str, str] = {}
    for r in read_csv(subset_ids_csv):
        sid = str(r.get("id", "")).strip()
        grp = str(r.get("group", "")).strip()
        if sid:
            id2group[sid] = grp

    id2meta: Dict[str, Dict[str, Any]] = {}
    for r in read_csv(subset_gt_csv):
        sid = str(r.get("id", "")).strip()
        if sid in id2group:
            id2meta[sid] = {
                "id": sid,
                "group": id2group[sid],
                "answer_gt": str(r.get("answer", "")).strip(),
                "question": str(r.get("question", "")).strip(),
                "image_id": str(r.get("image_id", "")).strip(),
                "orig_question_id": str(r.get("orig_question_id", "")).strip(),
            }
    return id2group, id2meta


def load_split_map(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in read_csv(path):
        sid = str(r.get("id", "")).strip()
        sp = str(r.get("split", "")).strip().lower()
        if sid and sp in {"calib", "eval"}:
            out[sid] = sp
    return out


def load_samples_target_map(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in read_csv(path):
        sid = str(r.get("id", "")).strip()
        if not sid:
            continue
        out[sid] = {
            "target_pred_answer_eval": str(r.get("pred_answer_eval", "")).strip().lower(),
            "target_is_correct": safe_int(r.get("is_correct"), 0),
            "target_is_fp_hallucination": safe_int(r.get("is_fp_hallucination"), 0),
            "target_is_tp_yes": safe_int(r.get("is_tp_yes"), 0),
            "target_is_tn_no": safe_int(r.get("is_tn_no"), 0),
            "target_is_fn_miss": safe_int(r.get("is_fn_miss"), 0),
        }
    return out


def build_layer_index(rows: List[Dict[str, Any]], keep_ids: set[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sid = str(r.get("id", "")).strip()
        if sid in keep_ids:
            out[sid].append(r)
    for sid in out.keys():
        out[sid].sort(key=lambda x: safe_int(x.get("block_layer_idx"), -1))
    return out


def build_head_index(rows: List[Dict[str, Any]], keep_ids: set[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sid = str(r.get("id", "")).strip()
        if sid in keep_ids:
            out[sid].append(r)
    return out


def temporal_features(
    rows: List[Dict[str, Any]],
    early_start: int,
    early_end: int,
    late_start: int,
    late_end: int,
    layer_focus: int,
    eps: float,
) -> Dict[str, float]:
    if len(rows) == 0:
        return {}

    # primary temporal signal from yes_attn_vis_sum
    all_layer_vals: List[Tuple[int, float]] = []
    argmax_patch_seq: List[int] = []
    late_rank_seq: List[float] = []

    for r in rows:
        li = safe_int(r.get("block_layer_idx"))
        if li is None:
            continue
        v = safe_float(r.get("yes_attn_vis_sum"), 0.0)
        all_layer_vals.append((int(li), float(v)))

        aidx = safe_int(r.get("yes_attn_vis_argmax_idx"))
        if aidx is not None:
            argmax_patch_seq.append(int(aidx))

        lr = safe_float(r.get("yes_sim_local_argmax_idx"))
        if lr is not None and li >= late_start and li <= late_end:
            late_rank_seq.append(float(lr))

    if len(all_layer_vals) == 0:
        return {}

    early_vals = [v for l, v in all_layer_vals if l >= early_start and l <= early_end]
    late_vals = [v for l, v in all_layer_vals if l >= late_start and l <= late_end]

    early_attn_mean = float(sum(early_vals) / float(max(1, len(early_vals))))
    late_attn_mean = float(sum(late_vals) / float(max(1, len(late_vals))))
    late_uplift = float(math.log((late_attn_mean + eps) / (early_attn_mean + eps)))

    peak_layer, peak_val = max(all_layer_vals, key=lambda x: x[1])
    max_layer = max(l for l, _ in all_layer_vals)
    peak_before_final = 1 if peak_layer < max_layer else 0

    flips = 0
    for i in range(1, len(argmax_patch_seq)):
        if argmax_patch_seq[i] != argmax_patch_seq[i - 1]:
            flips += 1

    # late top-k persistence via most frequent late argmax patch
    late_argmax = [
        safe_int(r.get("yes_attn_vis_argmax_idx"))
        for r in rows
        if (safe_int(r.get("block_layer_idx"), -10**9) >= late_start and safe_int(r.get("block_layer_idx"), -10**9) <= late_end)
    ]
    late_argmax = [x for x in late_argmax if x is not None]
    if len(late_argmax) > 0:
        cnt = defaultdict(int)
        for x in late_argmax:
            cnt[int(x)] += 1
        late_topk_persistence = float(max(cnt.values()) / float(len(late_argmax)))
    else:
        late_topk_persistence = 0.0

    # persistence after peak
    peak_argmax = None
    for r in rows:
        if safe_int(r.get("block_layer_idx")) == peak_layer:
            peak_argmax = safe_int(r.get("yes_attn_vis_argmax_idx"))
            break
    after = [
        safe_int(r.get("yes_attn_vis_argmax_idx"))
        for r in rows
        if safe_int(r.get("block_layer_idx"), -10**9) > peak_layer
    ]
    after = [x for x in after if x is not None]
    if peak_argmax is None or len(after) == 0:
        persistence_after_peak = 0.0
    else:
        persistence_after_peak = float(sum(1 for x in after if x == peak_argmax) / float(len(after)))

    # focus-layer proxy signals (for A fallback)
    focus_rows = [r for r in rows if safe_int(r.get("block_layer_idx")) == layer_focus]
    if len(focus_rows) == 0:
        focus_rows = rows[-1:]
    fr = focus_rows[0]

    return {
        "early_attn_mean": early_attn_mean,
        "late_attn_mean": late_attn_mean,
        "late_uplift": late_uplift,
        "late_topk_persistence": float(late_topk_persistence),
        "peak_layer_idx": float(peak_layer),
        "peak_before_final": float(peak_before_final),
        "late_rank_std": float(std(late_rank_seq)),
        "argmax_patch_flip_count": float(flips),
        "persistence_after_peak": float(persistence_after_peak),
        # proxy values
        "proxy_yes_sim_objpatch_max": float(safe_float(fr.get("yes_sim_objpatch_max"), 0.0)),
        "proxy_yes_sim_objpatch_topk": float(safe_float(fr.get("yes_sim_objpatch_topk"), 0.0)),
        "proxy_yes_sim_local_max": float(safe_float(fr.get("yes_sim_local_max"), 0.0)),
        "proxy_yes_z_local_max": float(safe_float(fr.get("yes_z_local_max"), 0.0)),
    }


def routing_features(
    rows: List[Dict[str, Any]],
    specs: set[Tuple[int, int]],
    late_start: int,
    late_end: int,
) -> Dict[str, float]:
    vals: List[float] = []
    all_late: List[float] = []
    covered = 0

    for r in rows:
        li = safe_int(r.get("block_layer_idx"))
        hi = safe_int(r.get("head_idx"))
        vv = safe_float(r.get("head_attn_vis_ratio"))
        if li is None or hi is None or vv is None:
            continue
        if li < late_start or li > late_end:
            continue
        all_late.append(float(vv))
        if (int(li), int(hi)) in specs:
            vals.append(float(vv))
            if float(vv) > 0.0:
                covered += 1

    m = float(sum(vals) / float(max(1, len(vals))))
    t = float(topk_mean(vals, top_ratio=0.2)) if vals else 0.0
    cov = float(covered / float(max(1, len(vals))))
    g = float(sum(all_late) / float(max(1, len(all_late)))) if all_late else 0.0

    return {
        "attn_mean": m,
        "attn_topkmean": t,
        "coverage": cov,
        "minus_global": float(m - g),
        "global_late_head_attn_mean": g,
        "n_points": float(len(vals)),
    }


def load_vsc_map(vsc_csv: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in read_csv(vsc_csv):
        sid = str(r.get("id", r.get("question_id", ""))).strip()
        if sid:
            out[sid] = r
    return out


def load_counterfactual_map(path: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in read_csv(path):
        sid = str(r.get("id", "")).strip()
        if not sid:
            continue
        base_m = safe_float(r.get("base_margin_yes_minus_no"), 0.0)
        masked_m = safe_float(r.get("masked_margin_yes_minus_no"), 0.0)
        base_g = safe_float(r.get("base_gt_logit"), 0.0)
        masked_g = safe_float(r.get("masked_gt_logit"), 0.0)
        out[sid] = {
            "aux_counterfactual_harmfulness": float((base_m or 0.0) - (masked_m or 0.0)),
            "aux_counterfactual_drop_gt_logit": float((base_g or 0.0) - (masked_g or 0.0)),
        }
    return out


def load_role_map(path: str) -> Dict[str, Dict[str, float]]:
    rows = read_csv(path)
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sid = str(r.get("id", "")).strip()
        if sid:
            by_id[sid].append(r)

    out: Dict[str, Dict[str, float]] = {}
    for sid, arr in by_id.items():
        n = len(arr)
        harm = sum(1 for r in arr if str(r.get("role_label", "")).lower() == "harmful")
        sup = sum(1 for r in arr if str(r.get("role_label", "")).lower() == "supportive")
        neu = max(0, n - harm - sup)
        out[sid] = {
            "target_role_harmful_ratio": float(harm / float(max(1, n))),
            "target_role_supportive_ratio": float(sup / float(max(1, n))),
            "target_role_neutral_ratio": float(neu / float(max(1, n))),
            "target_role_n_candidates": float(n),
        }
    return out


def zscore_map(rows: List[Dict[str, Any]], cols: List[str], eps: float = 1e-6) -> Dict[str, Dict[str, float]]:
    # returns id->z_col map
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for c in cols:
        vals = [safe_float(r.get(c)) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) == 0:
            means[c] = 0.0
            stds[c] = 1.0
        else:
            m = float(sum(vals) / float(len(vals)))
            s = float(std(vals))
            means[c] = m
            stds[c] = max(eps, s)

    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        sid = str(r.get("id", "")).strip()
        if not sid:
            continue
        out[sid] = {}
        for c in cols:
            v = safe_float(r.get(c), 0.0)
            out[sid][f"z_{c}"] = float(((v or 0.0) - means[c]) / stds[c])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build POPE feature_screen_v1 tables on balanced subset.")
    ap.add_argument("--subset_ids_csv", type=str, required=True)
    ap.add_argument("--subset_gt_csv", type=str, required=True)
    ap.add_argument("--per_layer_trace_csv", type=str, required=True)
    ap.add_argument("--per_head_trace_csv", type=str, required=True)
    ap.add_argument("--headset_json", type=str, required=True)
    ap.add_argument("--vsc_csv", type=str, default="")
    ap.add_argument("--samples_csv", type=str, default="", help="Optional id-aligned sample table with target labels.")
    ap.add_argument("--split_csv", type=str, default="", help="Optional id,split(calib/eval) table.")
    ap.add_argument("--use_split", type=str, default="all", choices=["all", "calib", "eval"])
    ap.add_argument("--counterfactual_csv", type=str, default="")
    ap.add_argument("--role_csv", type=str, default="")
    ap.add_argument("--include_aux_targets", action="store_true", help="Merge aux/target fields into unified table.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--early_start", type=int, default=10)
    ap.add_argument("--early_end", type=int, default=15)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--layer_focus", type=int, default=17)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--source_repo_A", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--source_repo_B", type=str, default="/home/kms/VISTA")
    ap.add_argument("--source_repo_C", type=str, default="/home/kms/VHR")
    ap.add_argument("--source_repo_D", type=str, default="/home/kms/EAZY")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    id2group, id2meta = load_subset(args.subset_ids_csv, args.subset_gt_csv)
    keep_ids = set(id2group.keys())
    split_map = load_split_map(args.split_csv) if str(args.split_csv).strip() else {}
    if str(args.use_split) != "all":
        keep_ids = set([sid for sid in keep_ids if split_map.get(sid, "") == str(args.use_split)])

    layer_rows = read_csv(args.per_layer_trace_csv)
    head_rows = read_csv(args.per_head_trace_csv)
    faithful_specs, harmful_specs = parse_headset_specs(args.headset_json)

    id2layers = build_layer_index(layer_rows, keep_ids)
    id2heads = build_head_index(head_rows, keep_ids)

    vsc_map = load_vsc_map(args.vsc_csv) if str(args.vsc_csv).strip() else {}
    target_map = load_samples_target_map(args.samples_csv) if str(args.samples_csv).strip() else {}
    cf_map = load_counterfactual_map(args.counterfactual_csv) if str(args.counterfactual_csv).strip() else {}
    role_map = load_role_map(args.role_csv) if str(args.role_csv).strip() else {}

    # Per-family rows
    rows_A: List[Dict[str, Any]] = []
    rows_B: List[Dict[str, Any]] = []
    rows_C: List[Dict[str, Any]] = []
    rows_D: List[Dict[str, Any]] = []
    rows_E: List[Dict[str, Any]] = []

    merged: List[Dict[str, Any]] = []

    for sid in sorted(keep_ids, key=lambda x: int(x) if str(x).isdigit() else x):
        base = dict(id2meta.get(sid, {"id": sid, "group": id2group.get(sid, "")}))

        # B: temporal
        tf = temporal_features(
            id2layers.get(sid, []),
            early_start=int(args.early_start),
            early_end=int(args.early_end),
            late_start=int(args.late_start),
            late_end=int(args.late_end),
            layer_focus=int(args.layer_focus),
            eps=float(args.eps),
        )

        # C/D: routing
        faith = routing_features(
            id2heads.get(sid, []),
            faithful_specs,
            int(args.late_start),
            int(args.late_end),
        )
        harm = routing_features(
            id2heads.get(sid, []),
            harmful_specs,
            int(args.late_start),
            int(args.late_end),
        )

        # A: visual confidence (from VSC dump if available, else trace proxy)
        vrow = vsc_map.get(sid)
        if vrow is not None:
            A = {
                "id": sid,
                "group": id2group.get(sid, ""),
                "A_source": "vga_vsc",
                "obj_token_prob_max": float(safe_float(vrow.get("obj_token_prob_max"), 0.0)),
                "obj_token_prob_mean": float(safe_float(vrow.get("obj_token_prob_mean"), 0.0)),
                "obj_token_prob_lse": float(safe_float(vrow.get("obj_token_prob_lse"), 0.0)),
                "obj_token_prob_topkmean": float(safe_float(vrow.get("obj_token_prob_topkmean"), 0.0)),
                "entropy_score": float(safe_float(vrow.get("entropy_score"), 0.0)),
                "G_entropy": float(safe_float(vrow.get("G_entropy"), 0.0)),
                "G_top1_mass": float(safe_float(vrow.get("G_top1_mass"), 0.0)),
                "G_top5_mass": float(safe_float(vrow.get("G_top5_mass"), 0.0)),
                "G_effective_support_size": float(safe_float(vrow.get("G_effective_support_size"), 0.0)),
            }
        else:
            A = {
                "id": sid,
                "group": id2group.get(sid, ""),
                "A_source": "trace_proxy",
                "obj_token_prob_max": float(tf.get("proxy_yes_sim_objpatch_max", 0.0)),
                "obj_token_prob_mean": float(tf.get("proxy_yes_sim_objpatch_topk", 0.0)),
                "obj_token_prob_lse": float(tf.get("proxy_yes_sim_local_max", 0.0)),
                "obj_token_prob_topkmean": float(tf.get("proxy_yes_sim_objpatch_topk", 0.0)),
                "entropy_score": float(max(0.0, 1.0 - tf.get("proxy_yes_z_local_max", 0.0))),
                "G_entropy": 0.0,
                "G_top1_mass": 0.0,
                "G_top5_mass": 0.0,
                "G_effective_support_size": 0.0,
            }

        B = {
            "id": sid,
            "group": id2group.get(sid, ""),
            "early_attn_mean": float(tf.get("early_attn_mean", 0.0)),
            "late_attn_mean": float(tf.get("late_attn_mean", 0.0)),
            "late_uplift": float(tf.get("late_uplift", 0.0)),
            "late_topk_persistence": float(tf.get("late_topk_persistence", 0.0)),
            "peak_layer_idx": float(tf.get("peak_layer_idx", 0.0)),
            "peak_before_final": float(tf.get("peak_before_final", 0.0)),
            "late_rank_std": float(tf.get("late_rank_std", 0.0)),
            "argmax_patch_flip_count": float(tf.get("argmax_patch_flip_count", 0.0)),
            "persistence_after_peak": float(tf.get("persistence_after_peak", 0.0)),
        }

        C = {
            "id": sid,
            "group": id2group.get(sid, ""),
            "faithful_head_attn_mean": float(faith.get("attn_mean", 0.0)),
            "faithful_head_attn_topkmean": float(faith.get("attn_topkmean", 0.0)),
            "faithful_head_coverage": float(faith.get("coverage", 0.0)),
            "faithful_minus_global_attn": float(faith.get("minus_global", 0.0)),
            "faithful_n_points": float(faith.get("n_points", 0.0)),
        }

        D = {
            "id": sid,
            "group": id2group.get(sid, ""),
            "harmful_head_attn_mean": float(harm.get("attn_mean", 0.0)),
            "harmful_head_attn_topkmean": float(harm.get("attn_topkmean", 0.0)),
            "harmful_head_coverage": float(harm.get("coverage", 0.0)),
            "harmful_minus_global_attn": float(harm.get("minus_global", 0.0)),
            "harmful_minus_faithful": float(harm.get("attn_mean", 0.0) - faith.get("attn_mean", 0.0)),
            "harmful_n_points": float(harm.get("n_points", 0.0)),
        }

        # E: guidance mismatch/context composition (proxy with available A/C/D)
        g5 = float(A.get("G_top5_mass", 0.0))
        fmean = float(C.get("faithful_head_attn_mean", 0.0))
        hmean = float(D.get("harmful_head_attn_mean", 0.0))

        faithful_on_G = fmean * g5
        faithful_on_nonG = fmean * (1.0 - g5)
        harmful_on_G = hmean * g5
        harmful_on_nonG = hmean * (1.0 - g5)

        E = {
            "id": sid,
            "group": id2group.get(sid, ""),
            "supportive_outside_G": float(faithful_on_nonG / max(float(args.eps), fmean + float(args.eps))),
            "harmful_inside_G": float(harmful_on_G / max(float(args.eps), hmean + float(args.eps))),
            "guidance_mismatch_score": float(harmful_on_G - faithful_on_G),
            "context_need_score": float(faithful_on_nonG - faithful_on_G),
            "G_overfocus": float(max(float(A.get("G_top1_mass", 0.0)), 1.0 - float(A.get("G_entropy", 0.0)))),
            "faithful_on_G_mass": float(faithful_on_G),
            "faithful_on_nonG_mass": float(faithful_on_nonG),
            "harmful_on_G_mass": float(harmful_on_G),
            "harmful_on_nonG_mass": float(harmful_on_nonG),
            "faithful_G_alignment": float(faithful_on_G / max(float(args.eps), faithful_on_nonG + float(args.eps))),
            "harmful_G_alignment": float(harmful_on_G / max(float(args.eps), harmful_on_nonG + float(args.eps))),
        }

        row = dict(base)
        row.update(A)
        row.update(B)
        row.update(C)
        row.update(D)
        row.update(E)
        row["A_source_repo"] = str(args.source_repo_A)
        row["B_source_repo"] = str(args.source_repo_B)
        row["C_source_repo"] = str(args.source_repo_C)
        row["D_source_repo"] = str(args.source_repo_D)
        row["split"] = split_map.get(sid, "")

        if sid in target_map:
            row.update(target_map[sid])

        if bool(args.include_aux_targets):
            if sid in cf_map:
                row.update(cf_map[sid])
            if sid in role_map:
                row.update(role_map[sid])

        rows_A.append({k: row.get(k) for k in [
            "id", "group", "A_source", "obj_token_prob_max", "obj_token_prob_mean", "obj_token_prob_lse",
            "obj_token_prob_topkmean", "entropy_score", "G_entropy", "G_top1_mass", "G_top5_mass",
            "G_effective_support_size"
        ]})
        rows_B.append({k: row.get(k) for k in [
            "id", "group", "early_attn_mean", "late_attn_mean", "late_uplift", "late_topk_persistence",
            "peak_layer_idx", "peak_before_final", "late_rank_std", "argmax_patch_flip_count", "persistence_after_peak"
        ]})
        rows_C.append({k: row.get(k) for k in [
            "id", "group", "faithful_head_attn_mean", "faithful_head_attn_topkmean", "faithful_head_coverage",
            "faithful_minus_global_attn", "faithful_n_points"
        ]})
        rows_D.append({k: row.get(k) for k in [
            "id", "group", "harmful_head_attn_mean", "harmful_head_attn_topkmean", "harmful_head_coverage",
            "harmful_minus_global_attn", "harmful_minus_faithful", "harmful_n_points"
        ]})
        rows_E.append({k: row.get(k) for k in [
            "id", "group", "supportive_outside_G", "harmful_inside_G", "guidance_mismatch_score", "context_need_score",
            "G_overfocus", "faithful_on_G_mass", "faithful_on_nonG_mass", "harmful_on_G_mass", "harmful_on_nonG_mass",
            "faithful_G_alignment", "harmful_G_alignment"
        ]})

        merged.append(row)

    # Unified score U (sample-level z-combined)
    zcols = [
        "obj_token_prob_lse",
        "faithful_head_attn_mean",
        "late_topk_persistence",
        "harmful_head_attn_mean",
        "late_uplift",
    ]
    zmap = zscore_map(merged, zcols, eps=float(args.eps))
    for r in merged:
        sid = str(r.get("id", "")).strip()
        z = zmap.get(sid, {})
        z_good = (
            float(z.get("z_obj_token_prob_lse", 0.0))
            + float(z.get("z_faithful_head_attn_mean", 0.0))
            + float(z.get("z_late_topk_persistence", 0.0))
        )
        z_bad = (
            float(z.get("z_harmful_head_attn_mean", 0.0))
            + float(z.get("z_late_uplift", 0.0))
        )
        r["U_good"] = float(z_good)
        r["U_bad"] = float(z_bad)
        r["U_unified"] = float(z_good - z_bad)

    # Save outputs
    out_A = os.path.join(args.out_dir, "features_visual_confidence.csv")
    out_B = os.path.join(args.out_dir, "features_temporal_dynamics.csv")
    out_C = os.path.join(args.out_dir, "features_faithful_routing.csv")
    out_D = os.path.join(args.out_dir, "features_harmful_routing.csv")
    out_E = os.path.join(args.out_dir, "features_guidance_mismatch.csv")
    out_M = os.path.join(args.out_dir, "features_unified_table.csv")
    out_S = os.path.join(args.out_dir, "summary.json")

    write_csv(out_A, rows_A)
    write_csv(out_B, rows_B)
    write_csv(out_C, rows_C)
    write_csv(out_D, rows_D)
    write_csv(out_E, rows_E)
    write_csv(out_M, merged)

    summary = {
        "inputs": {
            "subset_ids_csv": os.path.abspath(args.subset_ids_csv),
            "subset_gt_csv": os.path.abspath(args.subset_gt_csv),
            "per_layer_trace_csv": os.path.abspath(args.per_layer_trace_csv),
            "per_head_trace_csv": os.path.abspath(args.per_head_trace_csv),
            "headset_json": os.path.abspath(args.headset_json),
            "vsc_csv": (None if str(args.vsc_csv).strip() == "" else os.path.abspath(args.vsc_csv)),
            "samples_csv": (None if str(args.samples_csv).strip() == "" else os.path.abspath(args.samples_csv)),
            "split_csv": (None if str(args.split_csv).strip() == "" else os.path.abspath(args.split_csv)),
            "use_split": str(args.use_split),
            "counterfactual_csv": (None if str(args.counterfactual_csv).strip() == "" else os.path.abspath(args.counterfactual_csv)),
            "role_csv": (None if str(args.role_csv).strip() == "" else os.path.abspath(args.role_csv)),
            "include_aux_targets": bool(args.include_aux_targets),
            "early_start": int(args.early_start),
            "early_end": int(args.early_end),
            "late_start": int(args.late_start),
            "late_end": int(args.late_end),
            "layer_focus": int(args.layer_focus),
            "eps": float(args.eps),
            "feature_source_map": {
                "A_visual_confidence": str(args.source_repo_A),
                "B_temporal_dynamics": str(args.source_repo_B),
                "C_faithful_routing": str(args.source_repo_C),
                "D_harmful_routing": str(args.source_repo_D),
                "E_guidance_mismatch": "composed_from_A_C_D",
                "F_oracle_target_label": "from_addback_results_if_provided",
            },
        },
        "counts": {
            "n_subset_ids": int(len(keep_ids)),
            "n_with_layer_trace": int(sum(1 for sid in keep_ids if sid in id2layers)),
            "n_with_head_trace": int(sum(1 for sid in keep_ids if sid in id2heads)),
            "n_with_vsc": int(sum(1 for sid in keep_ids if sid in vsc_map)),
            "n_with_target_map": int(sum(1 for sid in keep_ids if sid in target_map)),
            "n_with_aux_counterfactual": int(sum(1 for sid in keep_ids if sid in cf_map)),
            "n_with_target_role": int(sum(1 for sid in keep_ids if sid in role_map)),
            "n_merged_rows": int(len(merged)),
            "n_faithful_specs": int(len(faithful_specs)),
            "n_harmful_specs": int(len(harmful_specs)),
        },
        "outputs": {
            "features_visual_confidence_csv": out_A,
            "features_temporal_dynamics_csv": out_B,
            "features_faithful_routing_csv": out_C,
            "features_harmful_routing_csv": out_D,
            "features_guidance_mismatch_csv": out_E,
            "features_unified_table_csv": out_M,
            "summary_json": out_S,
        },
    }

    with open(out_S, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_A)
    print("[saved]", out_B)
    print("[saved]", out_C)
    print("[saved]", out_D)
    print("[saved]", out_E)
    print("[saved]", out_M)
    print("[saved]", out_S)


if __name__ == "__main__":
    main()
