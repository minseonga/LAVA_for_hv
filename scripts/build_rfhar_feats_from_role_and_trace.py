#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def parse_json_int_list(x: Any) -> List[int]:
    if x is None:
        return []
    s = str(x).strip()
    if s == "":
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            out: List[int] = []
            for v in arr:
                iv = safe_int(v, None)
                if iv is not None:
                    out.append(int(iv))
            return out
    except Exception:
        pass
    return []


def parse_json_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    s = str(x).strip()
    if s == "":
        return []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            out: List[float] = []
            for v in arr:
                fv = safe_float(v, None)
                if fv is not None:
                    out.append(float(fv))
            return out
    except Exception:
        pass
    return []


def parse_label_set(s: str) -> Set[str]:
    out: Set[str] = set()
    for p in str(s or "").split(","):
        t = p.strip().lower()
        if t:
            out.add(t)
    return out


def load_ids(path_csv: str) -> List[str]:
    rows = read_csv(path_csv)
    out: List[str] = []
    seen: Set[str] = set()
    for r in rows:
        sid = str(r.get("id", "")).strip()
        if sid == "" or sid in seen:
            continue
        out.append(sid)
        seen.add(sid)
    return out


def sparse_to_dense(k_img: int, sparse: Dict[int, float]) -> List[float]:
    out = [0.0 for _ in range(int(k_img))]
    for p, v in sparse.items():
        if 0 <= int(p) < int(k_img):
            out[int(p)] = float(v)
    return out


def mean(vs: Iterable[float]) -> float:
    arr = [float(v) for v in vs]
    if len(arr) == 0:
        return 0.0
    return float(sum(arr) / float(len(arr)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strict RF-HAR C/A/D/B token features from role+trace CSV.")
    ap.add_argument("--role_csv", type=str, required=True, help="per_patch_role_effect.csv")
    ap.add_argument("--trace_csv", type=str, required=True, help="per_layer_yes_trace.csv")
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--out_summary", type=str, default="")
    ap.add_argument("--ids_csv", type=str, default="", help="optional CSV with id column to force output set")
    ap.add_argument("--k_img", type=int, default=576)

    ap.add_argument("--a_layer", type=int, default=17)
    ap.add_argument("--early_start", type=int, default=1)
    ap.add_argument("--early_end", type=int, default=4)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)

    ap.add_argument("--attn_topk_idx_col", type=str, default="yes_attn_vis_topk_idx_json")
    ap.add_argument("--sim_topk_idx_col", type=str, default="yes_sim_local_topk_idx_json")
    ap.add_argument("--sim_topk_weight_col", type=str, default="yes_sim_local_topk_weight_json")

    ap.add_argument("--supportive_labels", type=str, default="supportive")
    ap.add_argument("--harmful_labels", type=str, default="harmful,assertive")

    ap.add_argument("--rank_decay", type=float, default=1.0)
    ap.add_argument("--delta_scale", type=float, default=1.0)
    ap.add_argument("--conflict_bonus", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    k_img = int(args.k_img)
    if k_img <= 0:
        raise RuntimeError("--k_img must be > 0")

    role_rows = read_csv(os.path.abspath(args.role_csv))
    trace_rows = read_csv(os.path.abspath(args.trace_csv))
    if len(role_rows) == 0:
        raise RuntimeError("No rows in role_csv")
    if len(trace_rows) == 0:
        raise RuntimeError("No rows in trace_csv")

    forced_ids: Optional[Set[str]] = None
    output_ids: List[str] = []
    if str(args.ids_csv).strip() != "":
        ids = load_ids(os.path.abspath(args.ids_csv))
        forced_ids = set(ids)
        output_ids = list(ids)

    supportive_labels = parse_label_set(args.supportive_labels)
    harmful_labels = parse_label_set(args.harmful_labels)
    if len(supportive_labels) == 0:
        raise RuntimeError("supportive_labels is empty")
    if len(harmful_labels) == 0:
        raise RuntimeError("harmful_labels is empty")

    # Per-id sparse maps.
    A_trace: Dict[str, Dict[int, float]] = defaultdict(dict)
    E_count: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    L_count: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    E_layers: Dict[str, Set[int]] = defaultdict(set)
    L_layers: Dict[str, Set[int]] = defaultdict(set)

    a_layer = int(args.a_layer)
    e0, e1 = int(min(args.early_start, args.early_end)), int(max(args.early_start, args.early_end))
    l0, l1 = int(min(args.late_start, args.late_end)), int(max(args.late_start, args.late_end))

    for r in trace_rows:
        sid = str(r.get("id", "")).strip()
        if sid == "":
            continue
        if forced_ids is not None and sid not in forced_ids:
            continue

        li = safe_int(r.get("block_layer_idx"), None)
        if li is None:
            continue

        if int(li) == int(a_layer):
            idxs = parse_json_int_list(r.get(args.sim_topk_idx_col))
            ws = parse_json_float_list(r.get(args.sim_topk_weight_col))
            for i, p in enumerate(idxs):
                if not (0 <= int(p) < k_img):
                    continue
                w = float(ws[i]) if i < len(ws) else 0.0
                old = float(A_trace[sid].get(int(p), 0.0))
                if w > old:
                    A_trace[sid][int(p)] = float(w)

        idxs_attn = parse_json_int_list(r.get(args.attn_topk_idx_col))
        if e0 <= int(li) <= e1:
            E_layers[sid].add(int(li))
            for p in idxs_attn:
                if 0 <= int(p) < k_img:
                    E_count[sid][int(p)] += 1.0

        if l0 <= int(li) <= l1:
            L_layers[sid].add(int(li))
            for p in idxs_attn:
                if 0 <= int(p) < k_img:
                    L_count[sid][int(p)] += 1.0

    C_map: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    D_map: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    A_role: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    sup_set: Dict[str, Set[int]] = defaultdict(set)
    harm_set: Dict[str, Set[int]] = defaultdict(set)

    for r in role_rows:
        sid = str(r.get("id", "")).strip()
        if sid == "":
            continue
        if forced_ids is not None and sid not in forced_ids:
            continue

        p = safe_int(r.get("candidate_patch_idx"), None)
        if p is None or not (0 <= int(p) < k_img):
            continue
        p = int(p)

        label = str(r.get("role_label", "")).strip().lower()
        rank = max(0, int(safe_int(r.get("candidate_rank"), 0) or 0))
        rank_w = 1.0 / ((1.0 + float(rank)) ** float(args.rank_decay))

        sim = float(max(0.0, safe_float(r.get("candidate_patch_sim_valid"), 0.0) or 0.0))
        d_gt = abs(float(safe_float(r.get("delta_gt_margin"), 0.0) or 0.0))
        d_yn = abs(float(safe_float(r.get("delta_yes_minus_no"), 0.0) or 0.0))
        delta_abs = max(d_gt, d_yn)

        score = float(rank_w * (sim + float(args.delta_scale) * delta_abs))
        a_score = float(max(sim, 0.5 * delta_abs))

        if label in supportive_labels:
            C_map[sid][p] += score
            if a_score > A_role[sid].get(p, 0.0):
                A_role[sid][p] = a_score
            sup_set[sid].add(p)
        elif label in harmful_labels:
            D_map[sid][p] += score
            if a_score > A_role[sid].get(p, 0.0):
                A_role[sid][p] = a_score
            harm_set[sid].add(p)

    # Final id list.
    if forced_ids is None:
        ids_union = set(C_map.keys()) | set(D_map.keys()) | set(A_trace.keys()) | set(E_count.keys()) | set(L_count.keys())
        output_ids = sorted(ids_union, key=lambda x: int(x) if x.isdigit() else x)

    out_jsonl = os.path.abspath(args.out_jsonl)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    n_nonzero_c = []
    n_nonzero_a = []
    n_nonzero_d = []
    n_nonzero_b = []

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for sid in output_ids:
            c_sparse = C_map.get(sid, {})
            d_sparse = D_map.get(sid, {})
            a_sparse = {}
            # A = max(trace grounding, role-local confidence)
            for p, v in A_trace.get(sid, {}).items():
                if 0 <= int(p) < k_img:
                    a_sparse[int(p)] = float(max(float(v), float(a_sparse.get(int(p), 0.0))))
            for p, v in A_role.get(sid, {}).items():
                if 0 <= int(p) < k_img:
                    a_sparse[int(p)] = float(max(float(v), float(a_sparse.get(int(p), 0.0))))

            b_sparse: Dict[int, float] = defaultdict(float)
            e_layers = max(1, len(E_layers.get(sid, set())))
            l_layers = max(1, len(L_layers.get(sid, set())))
            cand = set(E_count.get(sid, {}).keys()) | set(L_count.get(sid, {}).keys())
            for p in cand:
                e_freq = float(E_count.get(sid, {}).get(int(p), 0.0)) / float(e_layers)
                l_freq = float(L_count.get(sid, {}).get(int(p), 0.0)) / float(l_layers)
                b_sparse[int(p)] = max(0.0, e_freq - l_freq)

            # Conflict-based instability bonus.
            overlap = set(sup_set.get(sid, set())) & set(harm_set.get(sid, set()))
            for p in overlap:
                b_sparse[int(p)] += float(args.conflict_bonus)

            C = sparse_to_dense(k_img, c_sparse)
            A = sparse_to_dense(k_img, a_sparse)
            D = sparse_to_dense(k_img, d_sparse)
            B = sparse_to_dense(k_img, b_sparse)

            n_nonzero_c.append(sum(1 for v in C if abs(float(v)) > float(args.eps)))
            n_nonzero_a.append(sum(1 for v in A if abs(float(v)) > float(args.eps)))
            n_nonzero_d.append(sum(1 for v in D if abs(float(v)) > float(args.eps)))
            n_nonzero_b.append(sum(1 for v in B if abs(float(v)) > float(args.eps)))

            rec = {
                "id": str(sid),
                "C": C,
                "A": A,
                "D": D,
                "B": B,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "inputs": {
            "role_csv": os.path.abspath(args.role_csv),
            "trace_csv": os.path.abspath(args.trace_csv),
            "ids_csv": (os.path.abspath(args.ids_csv) if str(args.ids_csv).strip() != "" else ""),
            "k_img": int(k_img),
            "a_layer": int(a_layer),
            "early_start": int(e0),
            "early_end": int(e1),
            "late_start": int(l0),
            "late_end": int(l1),
            "supportive_labels": sorted(list(supportive_labels)),
            "harmful_labels": sorted(list(harmful_labels)),
        },
        "counts": {
            "n_ids": int(len(output_ids)),
            "mean_nonzero_C": float(mean(n_nonzero_c)),
            "mean_nonzero_A": float(mean(n_nonzero_a)),
            "mean_nonzero_D": float(mean(n_nonzero_d)),
            "mean_nonzero_B": float(mean(n_nonzero_b)),
            "max_nonzero_C": int(max(n_nonzero_c) if n_nonzero_c else 0),
            "max_nonzero_A": int(max(n_nonzero_a) if n_nonzero_a else 0),
            "max_nonzero_D": int(max(n_nonzero_d) if n_nonzero_d else 0),
            "max_nonzero_B": int(max(n_nonzero_b) if n_nonzero_b else 0),
        },
        "outputs": {
            "rfhar_feats_jsonl": out_jsonl,
            "summary_json": os.path.abspath(args.out_summary) if str(args.out_summary).strip() != "" else "",
        },
    }

    out_summary = str(args.out_summary).strip()
    if out_summary != "":
        out_summary = os.path.abspath(out_summary)
        os.makedirs(os.path.dirname(out_summary), exist_ok=True)
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[saved]", out_summary)

    print("[saved]", out_jsonl)
    print("[summary]", json.dumps(summary["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
