#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


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
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(dict(r))
    return rows


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
    # Tie-safe KS on union support: D = sup_x |F_pos(x) - F_neg(x)|
    support = sorted(set(pos + neg))
    i = 0
    j = 0
    dmax = 0.0
    for v in support:
        while i < n_pos and pos[i] <= v:
            i += 1
        while j < n_neg and neg[j] <= v:
            j += 1
        f_pos = float(i / n_pos)
        f_neg = float(j / n_neg)
        dmax = max(dmax, abs(f_pos - f_neg))
    return float(dmax)


def summarize_metric(labels: List[int], scores: List[float]) -> Dict[str, Any]:
    auc = auc_from_scores(labels, scores)
    ks = ks_from_scores(labels, scores)
    return {
        "n": int(len(scores)),
        "auc_hall_high": auc,
        "auc_best_dir": (None if auc is None else float(max(auc, 1.0 - auc))),
        "direction": (None if auc is None else ("higher_in_hallucination" if auc >= 0.5 else "lower_in_hallucination")),
        "ks_hall_high": ks,
    }


def parse_json_int_list(x: Any) -> List[int]:
    s = str("" if x is None else x).strip()
    if s == "":
        return []
    try:
        arr = json.loads(s)
        if not isinstance(arr, list):
            return []
        out = []
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
        out = []
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


def softmax_probs(vals: Sequence[float]) -> List[float]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if len(xs) == 0:
        return []
    m = max(xs)
    ex = [math.exp(v - m) for v in xs]
    s = sum(ex)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [float(v / s) for v in ex]


def weighted_jaccard_from_maps(a: Dict[int, float], b: Dict[int, float]) -> Optional[float]:
    if len(a) == 0 or len(b) == 0:
        return None
    keys = set(a.keys()) | set(b.keys())
    num = 0.0
    den = 0.0
    for k in keys:
        av = float(max(0.0, a.get(k, 0.0)))
        bv = float(max(0.0, b.get(k, 0.0)))
        num += min(av, bv)
        den += max(av, bv)
    if den <= 0:
        return None
    return float(num / den)


def weighted_overlap_from_maps(a: Dict[int, float], b: Dict[int, float]) -> Optional[float]:
    # overlap mass in [0,1] if both are normalized distributions.
    if len(a) == 0 or len(b) == 0:
        return None
    keys = set(a.keys()) | set(b.keys())
    ov = 0.0
    for k in keys:
        av = float(max(0.0, a.get(k, 0.0)))
        bv = float(max(0.0, b.get(k, 0.0)))
        ov += min(av, bv)
    return float(ov)


def make_weight_map(idx_list: List[int], raw_weight_list: List[float], normalize: str = "softmax") -> Dict[int, float]:
    n = int(min(len(idx_list), len(raw_weight_list)))
    if n <= 0:
        return {}
    idx = [int(v) for v in idx_list[:n]]
    w = [float(v) for v in raw_weight_list[:n]]
    if normalize == "softmax":
        wn = softmax_probs(w)
    else:
        # non-negative normalize fallback
        wp = [max(0.0, float(v)) for v in w]
        s = sum(wp)
        wn = ([v / s for v in wp] if s > 0 else [1.0 / len(wp)] * len(wp))
    out: Dict[int, float] = {}
    for i, ww in zip(idx, wn):
        out[i] = float(out.get(i, 0.0) + float(ww))
    return out


def renorm_map(x: Dict[int, float]) -> Dict[int, float]:
    if len(x) == 0:
        return {}
    y = {int(k): float(max(0.0, v)) for k, v in x.items()}
    s = float(sum(y.values()))
    if s <= 0:
        n = len(y)
        return {k: float(1.0 / n) for k in y.keys()}
    return {k: float(v / s) for k, v in y.items()}


def parse_layer_pairs(s: str) -> List[Tuple[int, int]]:
    txt = str(s or "").strip()
    if txt == "":
        return []
    out: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for part in txt.split(","):
        t = part.strip()
        if t == "" or ":" not in t:
            continue
        a, b = t.split(":", 1)
        try:
            e = int(a.strip())
            l = int(b.strip())
        except Exception:
            continue
        key = (int(e), int(l))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ERS/AIS/PCS from per-layer yes trace.")
    ap.add_argument("--per_layer_trace_csv", type=str, required=True)
    ap.add_argument(
        "--counterfactual_csv",
        type=str,
        default="",
        help="Optional per-sample counterfactual rerun CSV (e.g., eval_pope_objpatch_mask_rerun output) for ERS_drop metrics.",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--early_start", type=int, default=0)
    ap.add_argument("--early_end", type=int, default=15)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--early_attn_metric", type=str, default="yes_attn_vis_ratio")
    ap.add_argument("--late_sim_metric", type=str, default="yes_sim_local_max")
    ap.add_argument(
        "--ers_local_topk",
        type=int,
        default=8,
        help="Top-k patch indices used for ERS_local answer-region / early-attention overlap.",
    )
    ap.add_argument(
        "--early_margin_metric",
        type=str,
        default="yes_sim_local_gap",
        help="Early-phase margin proxy metric for ERS_margin.",
    )
    ap.add_argument(
        "--late_margin_metric",
        type=str,
        default="yes_sim_local_gap",
        help="Late-phase margin proxy metric for ERS_margin delta.",
    )
    ap.add_argument(
        "--pcs_weight_norm",
        type=str,
        default="softmax",
        choices=["softmax", "nonneg_norm"],
        help="How to normalize top-k weights for weighted PCS.",
    )
    ap.add_argument("--pcs_pairwise", action="store_true", help="Enable explicit layer-pair PCS.")
    ap.add_argument("--pcs_pairs", type=str, default="10:17,12:20", help="Comma-separated early:late pairs.")
    ap.add_argument("--pcs_pair_all", action="store_true", help="Use full Cartesian product of early x late ranges.")
    ap.add_argument("--pcs_answer_region_only", type=parse_bool, default=True, help="Restrict pair PCS to answer-conditioned late region.")
    ap.add_argument("--pcs_answer_region_topk", type=int, default=8, help="Top-k late patches used as answer-conditioned region.")
    ap.add_argument("--pcs_pair_early_idx_col", type=str, default="yes_attn_vis_topk_idx_json")
    ap.add_argument("--pcs_pair_early_weight_col", type=str, default="yes_attn_vis_topk_weight_json")
    ap.add_argument("--pcs_pair_late_idx_col", type=str, default="yes_sim_local_topk_idx_json")
    ap.add_argument("--pcs_pair_late_weight_col", type=str, default="yes_sim_local_topk_weight_json")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--pcs_topk", type=int, default=8, help="Used only if top-k patch index json columns exist.")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rows = read_csv(os.path.abspath(args.per_layer_trace_csv))
    if len(rows) == 0:
        raise RuntimeError("No rows in per_layer_trace_csv.")

    cf_map: Dict[str, Dict[str, Any]] = {}
    if str(args.counterfactual_csv).strip() != "":
        cf_rows = read_csv(os.path.abspath(args.counterfactual_csv))
        for r in cf_rows:
            sid = str(r.get("id") or "").strip()
            if sid == "":
                continue
            cf_map[sid] = r

    pair_defs: List[Tuple[int, int]] = []
    if bool(args.pcs_pair_all):
        e0, e1 = int(min(args.early_start, args.early_end)), int(max(args.early_start, args.early_end))
        l0, l1 = int(min(args.late_start, args.late_end)), int(max(args.late_start, args.late_end))
        for e in range(e0, e1 + 1):
            for l in range(l0, l1 + 1):
                pair_defs.append((int(e), int(l)))
    else:
        pair_defs = parse_layer_pairs(str(args.pcs_pairs))

    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = str(r.get("id") or "").strip()
        if sid == "":
            continue
        by_id.setdefault(sid, []).append(r)
    if len(by_id) == 0:
        raise RuntimeError("No valid id rows.")

    per_id: List[Dict[str, Any]] = []
    per_pair_rows: List[Dict[str, Any]] = []
    for sid, grp in by_id.items():
        early = [
            r for r in grp
            if safe_float(r.get("block_layer_idx")) is not None
            and int(float(r.get("block_layer_idx"))) >= int(min(args.early_start, args.early_end))
            and int(float(r.get("block_layer_idx"))) <= int(max(args.early_start, args.early_end))
        ]
        late = [
            r for r in grp
            if safe_float(r.get("block_layer_idx")) is not None
            and int(float(r.get("block_layer_idx"))) >= int(min(args.late_start, args.late_end))
            and int(float(r.get("block_layer_idx"))) <= int(max(args.late_start, args.late_end))
        ]
        if len(early) == 0 or len(late) == 0:
            continue

        ers_vals = [safe_float(r.get(args.early_attn_metric)) for r in early]
        ers_vals = [float(v) for v in ers_vals if v is not None]
        late_vals = [safe_float(r.get(args.late_sim_metric)) for r in late]
        late_vals = [float(v) for v in late_vals if v is not None]
        if len(ers_vals) == 0 or len(late_vals) == 0:
            continue

        ers_mean = float(sum(ers_vals) / len(ers_vals))
        ers_min = float(min(ers_vals))
        ers_max = float(max(ers_vals))
        late_max = float(max(late_vals))
        late_mean = float(sum(late_vals) / len(late_vals))

        ais_max = float(late_max / max(float(args.eps), ers_mean))
        ais_mean = float(late_mean / max(float(args.eps), ers_mean))

        # (a) ERS_local: early evidence weighted by overlap with answer-conditioned late top-k region.
        ers_local_mean = None
        ers_local_overlap_mean = None
        answer_region: Set[int] = set()
        k_loc = int(max(1, args.ers_local_topk))
        for r in late:
            for v in parse_json_int_list(r.get("yes_sim_local_topk_idx_json"))[:k_loc]:
                answer_region.add(int(v))
        if len(answer_region) > 0:
            ers_local_vals: List[float] = []
            ers_local_ov_vals: List[float] = []
            for r in early:
                base = safe_float(r.get(args.early_attn_metric))
                if base is None:
                    continue
                att_idx = parse_json_int_list(r.get("yes_attn_vis_topk_idx_json"))[:k_loc]
                if len(att_idx) == 0:
                    continue
                att_set = set(int(v) for v in att_idx)
                ov = float(len(att_set & answer_region) / max(1, len(att_set)))
                ers_local_ov_vals.append(ov)
                ers_local_vals.append(float(base) * ov)
            if len(ers_local_vals) > 0:
                ers_local_mean = float(sum(ers_local_vals) / len(ers_local_vals))
                ers_local_overlap_mean = float(sum(ers_local_ov_vals) / len(ers_local_ov_vals))

        # AIS variant normalized by ERS_local (if available).
        ais_local_max = None
        ais_local_mean = None
        if ers_local_mean is not None:
            den_local = max(float(args.eps), float(ers_local_mean))
            ais_local_max = float(late_max / den_local)
            ais_local_mean = float(late_mean / den_local)

        # (c) ERS_margin: early/late margin proxy and its phase delta.
        ers_margin_early = None
        ers_margin_late = None
        ers_margin_delta = None
        early_margin_vals = [safe_float(r.get(args.early_margin_metric)) for r in early]
        early_margin_vals = [float(v) for v in early_margin_vals if v is not None]
        late_margin_vals = [safe_float(r.get(args.late_margin_metric)) for r in late]
        late_margin_vals = [float(v) for v in late_margin_vals if v is not None]
        if len(early_margin_vals) > 0 and len(late_margin_vals) > 0:
            ers_margin_early = float(sum(early_margin_vals) / len(early_margin_vals))
            ers_margin_late = float(sum(late_margin_vals) / len(late_margin_vals))
            ers_margin_delta = float(ers_margin_late - ers_margin_early)

        # PCS: requires per-layer patch-index columns. If absent, leave None.
        pcs_iou = None
        pcs_wjaccard = None
        pcs_woverlap = None
        early_set: Set[int] = set()
        late_set: Set[int] = set()
        early_has_col = any(("yes_attn_vis_topk_idx_json" in r) for r in early)
        late_has_col = any(("yes_sim_local_topk_idx_json" in r) for r in late)
        late_has_argmax = any(("yes_sim_local_argmax_idx" in r) for r in late)

        if early_has_col:
            for r in early:
                for v in parse_json_int_list(r.get("yes_attn_vis_topk_idx_json"))[: int(max(1, args.pcs_topk))]:
                    early_set.add(int(v))
        if late_has_col:
            for r in late:
                for v in parse_json_int_list(r.get("yes_sim_local_topk_idx_json"))[: int(max(1, args.pcs_topk))]:
                    late_set.add(int(v))
        elif late_has_argmax:
            for r in late:
                vv = safe_float(r.get("yes_sim_local_argmax_idx"))
                if vv is not None:
                    late_set.add(int(vv))
        if len(early_set) > 0 and len(late_set) > 0:
            inter = len(early_set & late_set)
            union = len(early_set | late_set)
            if union > 0:
                pcs_iou = float(inter / union)

        # Weighted PCS (top-k weighted overlap / weighted Jaccard).
        # We aggregate early/late top-k maps by mean over layers, then compare.
        early_maps: List[Dict[int, float]] = []
        late_maps: List[Dict[int, float]] = []
        for r in early:
            idxs = parse_json_int_list(r.get("yes_attn_vis_topk_idx_json"))[: int(max(1, args.pcs_topk))]
            wts = parse_json_float_list(r.get("yes_attn_vis_topk_weight_json"))[: int(max(1, args.pcs_topk))]
            mp = make_weight_map(
                idx_list=idxs,
                raw_weight_list=wts,
                normalize=("softmax" if str(args.pcs_weight_norm) == "softmax" else "nonneg_norm"),
            )
            if len(mp) > 0:
                early_maps.append(mp)
        for r in late:
            idxs = parse_json_int_list(r.get("yes_sim_local_topk_idx_json"))[: int(max(1, args.pcs_topk))]
            wts = parse_json_float_list(r.get("yes_sim_local_topk_weight_json"))[: int(max(1, args.pcs_topk))]
            mp = make_weight_map(
                idx_list=idxs,
                raw_weight_list=wts,
                normalize=("softmax" if str(args.pcs_weight_norm) == "softmax" else "nonneg_norm"),
            )
            if len(mp) > 0:
                late_maps.append(mp)

        def avg_maps(maps: List[Dict[int, float]]) -> Dict[int, float]:
            if len(maps) == 0:
                return {}
            keys: Set[int] = set()
            for mp in maps:
                keys |= set(mp.keys())
            out: Dict[int, float] = {}
            for k in keys:
                out[k] = float(sum(float(mp.get(k, 0.0)) for mp in maps) / float(len(maps)))
            s = float(sum(out.values()))
            if s > 0:
                out = {k: float(v / s) for k, v in out.items()}
            return out

        e_map = avg_maps(early_maps)
        l_map = avg_maps(late_maps)
        if len(e_map) > 0 and len(l_map) > 0:
            pcs_wjaccard = weighted_jaccard_from_maps(e_map, l_map)
            pcs_woverlap = weighted_overlap_from_maps(e_map, l_map)

        r0 = grp[0]
        rec = {
            "id": sid,
            "image_id": r0.get("image_id"),
            "answer_gt": r0.get("answer_gt"),
            "answer_pred": r0.get("answer_pred"),
            "is_fp_hallucination": bool(parse_bool(r0.get("is_fp_hallucination"))),
            "is_tp_yes": bool(parse_bool(r0.get("is_tp_yes"))),
            "ers_mean": ers_mean,
            "ers_min": ers_min,
            "ers_max": ers_max,
            "late_sim_max": late_max,
            "late_sim_mean": late_mean,
            "ais_max": ais_max,
            "ais_mean": ais_mean,
            "ers_local_mean": ers_local_mean,
            "ers_local_overlap_mean": ers_local_overlap_mean,
            "ais_local_max": ais_local_max,
            "ais_local_mean": ais_local_mean,
            "ers_margin_early": ers_margin_early,
            "ers_margin_late": ers_margin_late,
            "ers_margin_delta": ers_margin_delta,
            "pcs_iou": pcs_iou,
            "pcs_wjaccard": pcs_wjaccard,
            "pcs_woverlap": pcs_woverlap,
            "pcs_pair_iou_mean": None,
            "pcs_pair_wjaccard_mean": None,
            "pcs_pair_woverlap_mean": None,
            "pcs_pair_n": 0,
            "n_early_layers": int(len(early)),
            "n_late_layers": int(len(late)),
        }
        cf = cf_map.get(sid)
        if cf is not None:
            rec["ers_drop_margin"] = safe_float(cf.get("drop_margin_yes_minus_no"))
            rec["ers_drop_gt_logit"] = safe_float(cf.get("drop_gt_logit"))
            b_yes = safe_float(cf.get("base_yes_logit"))
            m_yes = safe_float(cf.get("masked_yes_logit"))
            rec["ers_drop_yes_logit"] = (None if b_yes is None or m_yes is None else float(b_yes - m_yes))
            b_no = safe_float(cf.get("base_no_logit"))
            m_no = safe_float(cf.get("masked_no_logit"))
            rec["ers_drop_no_logit"] = (None if b_no is None or m_no is None else float(b_no - m_no))
        else:
            rec["ers_drop_margin"] = None
            rec["ers_drop_gt_logit"] = None
            rec["ers_drop_yes_logit"] = None
            rec["ers_drop_no_logit"] = None

        if bool(args.pcs_pairwise) and len(pair_defs) > 0:
            layer_map: Dict[int, Dict[str, Any]] = {}
            for rr in grp:
                li = safe_float(rr.get("block_layer_idx"))
                if li is None:
                    continue
                layer_map[int(li)] = rr

            pair_iou_vals: List[float] = []
            pair_wj_vals: List[float] = []
            pair_wo_vals: List[float] = []
            k_pair = int(max(1, args.pcs_topk))
            k_region = int(max(1, args.pcs_answer_region_topk))
            norm_mode = ("softmax" if str(args.pcs_weight_norm) == "softmax" else "nonneg_norm")

            for e_layer, l_layer in pair_defs:
                re = layer_map.get(int(e_layer))
                rl = layer_map.get(int(l_layer))
                if re is None or rl is None:
                    continue

                e_idx = parse_json_int_list(re.get(args.pcs_pair_early_idx_col))[:k_pair]
                e_w = parse_json_float_list(re.get(args.pcs_pair_early_weight_col))[:k_pair]
                l_idx = parse_json_int_list(rl.get(args.pcs_pair_late_idx_col))[:k_pair]
                l_w = parse_json_float_list(rl.get(args.pcs_pair_late_weight_col))[:k_pair]
                e_map = make_weight_map(e_idx, e_w, normalize=norm_mode)
                l_map = make_weight_map(l_idx, l_w, normalize=norm_mode)
                if len(e_map) == 0 or len(l_map) == 0:
                    continue

                answer_region = set(parse_json_int_list(rl.get(args.pcs_pair_late_idx_col))[:k_region])
                if bool(args.pcs_answer_region_only):
                    if len(answer_region) == 0:
                        continue
                    e_map = {k: v for k, v in e_map.items() if int(k) in answer_region}
                    l_map = {k: v for k, v in l_map.items() if int(k) in answer_region}
                    if len(e_map) == 0 or len(l_map) == 0:
                        continue
                    e_map = renorm_map(e_map)
                    l_map = renorm_map(l_map)

                e_set = set(e_map.keys())
                l_set = set(l_map.keys())
                inter = len(e_set & l_set)
                union = len(e_set | l_set)
                p_iou = (None if union <= 0 else float(inter / union))
                p_wj = weighted_jaccard_from_maps(e_map, l_map)
                p_wo = weighted_overlap_from_maps(e_map, l_map)

                per_pair_rows.append(
                    {
                        "id": sid,
                        "image_id": r0.get("image_id"),
                        "answer_gt": r0.get("answer_gt"),
                        "answer_pred": r0.get("answer_pred"),
                        "is_fp_hallucination": bool(parse_bool(r0.get("is_fp_hallucination"))),
                        "is_tp_yes": bool(parse_bool(r0.get("is_tp_yes"))),
                        "early_layer": int(e_layer),
                        "late_layer": int(l_layer),
                        "pair": f"{int(e_layer)}->{int(l_layer)}",
                        "answer_region_only": bool(args.pcs_answer_region_only),
                        "answer_region_size": int(len(answer_region)),
                        "pcs_pair_iou": p_iou,
                        "pcs_pair_wjaccard": p_wj,
                        "pcs_pair_woverlap": p_wo,
                    }
                )

                if p_iou is not None:
                    pair_iou_vals.append(float(p_iou))
                if p_wj is not None:
                    pair_wj_vals.append(float(p_wj))
                if p_wo is not None:
                    pair_wo_vals.append(float(p_wo))

            rec["pcs_pair_iou_mean"] = (None if len(pair_iou_vals) == 0 else float(sum(pair_iou_vals) / len(pair_iou_vals)))
            rec["pcs_pair_wjaccard_mean"] = (None if len(pair_wj_vals) == 0 else float(sum(pair_wj_vals) / len(pair_wj_vals)))
            rec["pcs_pair_woverlap_mean"] = (None if len(pair_wo_vals) == 0 else float(sum(pair_wo_vals) / len(pair_wo_vals)))
            rec["pcs_pair_n"] = int(max(len(pair_iou_vals), len(pair_wj_vals), len(pair_wo_vals)))
        per_id.append(rec)

    if len(per_id) == 0:
        raise RuntimeError("No per-id rows computed.")

    eval_rows: List[Dict[str, Any]] = []
    fp_tp = [r for r in per_id if bool(r.get("is_fp_hallucination")) or bool(r.get("is_tp_yes"))]
    for metric in [
        "ers_mean",
        "ers_local_mean",
        "ers_local_overlap_mean",
        "ers_margin_early",
        "ers_margin_late",
        "ers_margin_delta",
        "ais_max",
        "ais_mean",
        "ais_local_max",
        "ais_local_mean",
        "ers_drop_margin",
        "ers_drop_gt_logit",
        "ers_drop_yes_logit",
        "ers_drop_no_logit",
        "pcs_iou",
        "pcs_wjaccard",
        "pcs_woverlap",
        "pcs_pair_iou_mean",
        "pcs_pair_wjaccard_mean",
        "pcs_pair_woverlap_mean",
    ]:
        labels: List[int] = []
        scores: List[float] = []
        for r in fp_tp:
            v = safe_float(r.get(metric))
            if v is None:
                continue
            labels.append(1 if bool(r.get("is_fp_hallucination")) else 0)
            scores.append(float(v))
        if len(scores) < 20:
            continue
        sm = summarize_metric(labels, scores)
        sm["metric"] = metric
        sm["comparison"] = "fp_hall_vs_tp_yes"
        eval_rows.append(sm)

    pair_eval_rows: List[Dict[str, Any]] = []
    if bool(args.pcs_pairwise) and len(per_pair_rows) > 0:
        pair_keys = sorted(set((int(r["early_layer"]), int(r["late_layer"])) for r in per_pair_rows))
        for e_layer, l_layer in pair_keys:
            sub = [
                r for r in per_pair_rows
                if int(r["early_layer"]) == int(e_layer) and int(r["late_layer"]) == int(l_layer)
            ]
            for metric in ["pcs_pair_iou", "pcs_pair_wjaccard", "pcs_pair_woverlap"]:
                labels: List[int] = []
                scores: List[float] = []
                for r in sub:
                    if not (bool(r.get("is_fp_hallucination")) or bool(r.get("is_tp_yes"))):
                        continue
                    v = safe_float(r.get(metric))
                    if v is None:
                        continue
                    labels.append(1 if bool(r.get("is_fp_hallucination")) else 0)
                    scores.append(float(v))
                if len(scores) < 20:
                    continue
                sm = summarize_metric(labels, scores)
                sm["metric"] = metric
                sm["comparison"] = "fp_hall_vs_tp_yes"
                sm["early_layer"] = int(e_layer)
                sm["late_layer"] = int(l_layer)
                sm["pair"] = f"{int(e_layer)}->{int(l_layer)}"
                pair_eval_rows.append(sm)
    pair_eval_rows = sorted(pair_eval_rows, key=lambda x: float(safe_float(x.get("auc_best_dir")) or -1.0), reverse=True)

    eval_rows = sorted(eval_rows, key=lambda x: float(safe_float(x.get("auc_best_dir")) or -1.0), reverse=True)

    summary = {
        "inputs": {
            "per_layer_trace_csv": os.path.abspath(args.per_layer_trace_csv),
            "counterfactual_csv": (None if str(args.counterfactual_csv).strip() == "" else os.path.abspath(args.counterfactual_csv)),
            "early_start": int(args.early_start),
            "early_end": int(args.early_end),
            "late_start": int(args.late_start),
            "late_end": int(args.late_end),
            "early_attn_metric": str(args.early_attn_metric),
            "late_sim_metric": str(args.late_sim_metric),
            "ers_local_topk": int(args.ers_local_topk),
            "early_margin_metric": str(args.early_margin_metric),
            "late_margin_metric": str(args.late_margin_metric),
            "pcs_weight_norm": str(args.pcs_weight_norm),
            "pcs_pairwise": bool(args.pcs_pairwise),
            "pcs_pairs": str(args.pcs_pairs),
            "pcs_pair_all": bool(args.pcs_pair_all),
            "pcs_answer_region_only": bool(args.pcs_answer_region_only),
            "pcs_answer_region_topk": int(args.pcs_answer_region_topk),
            "pcs_pair_early_idx_col": str(args.pcs_pair_early_idx_col),
            "pcs_pair_early_weight_col": str(args.pcs_pair_early_weight_col),
            "pcs_pair_late_idx_col": str(args.pcs_pair_late_idx_col),
            "pcs_pair_late_weight_col": str(args.pcs_pair_late_weight_col),
            "eps": float(args.eps),
            "pcs_topk": int(args.pcs_topk),
        },
        "counts": {
            "n_trace_rows": int(len(rows)),
            "n_ids": int(len(by_id)),
            "n_per_id_rows": int(len(per_id)),
            "n_fp": int(sum(1 for r in per_id if bool(r.get("is_fp_hallucination")))),
            "n_tp_yes": int(sum(1 for r in per_id if bool(r.get("is_tp_yes")))),
            "n_per_pair_rows": int(len(per_pair_rows)),
        },
        "best_eval": (None if len(eval_rows) == 0 else eval_rows[0]),
        "best_pair_eval": (None if len(pair_eval_rows) == 0 else pair_eval_rows[0]),
        "outputs": {
            "per_id_csv": os.path.join(out_dir, "per_id_ers_ais_pcs.csv"),
            "eval_csv": os.path.join(out_dir, "eval_ers_ais_pcs.csv"),
            "per_pair_csv": os.path.join(out_dir, "per_pair_pcs.csv"),
            "pair_eval_csv": os.path.join(out_dir, "pair_eval_ers_ais_pcs.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_id_ers_ais_pcs.csv"), per_id)
    write_csv(os.path.join(out_dir, "eval_ers_ais_pcs.csv"), eval_rows)
    write_csv(os.path.join(out_dir, "per_pair_pcs.csv"), per_pair_rows)
    write_csv(os.path.join(out_dir, "pair_eval_ers_ais_pcs.csv"), pair_eval_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_id_ers_ais_pcs.csv"))
    print("[saved]", os.path.join(out_dir, "eval_ers_ais_pcs.csv"))
    print("[saved]", os.path.join(out_dir, "per_pair_pcs.csv"))
    print("[saved]", os.path.join(out_dir, "pair_eval_ers_ais_pcs.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
