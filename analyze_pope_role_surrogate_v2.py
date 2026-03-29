#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
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


def jloads(s: Any, default):
    try:
        return json.loads(s) if s not in (None, "") else default
    except Exception:
        return default


def safe_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def z_from_train(train_vals: Sequence[float], x: float, eps: float = 1e-6) -> float:
    if len(train_vals) <= 1:
        return 0.0
    m = sum(train_vals) / float(len(train_vals))
    v = sum((v - m) * (v - m) for v in train_vals) / float(max(1, len(train_vals) - 1))
    s = math.sqrt(max(v, eps))
    return float((x - m) / s)


def mean(vs: Iterable[float]) -> float:
    arr = [float(v) for v in vs]
    if len(arr) == 0:
        return 0.0
    return float(sum(arr) / float(len(arr)))


def auc_roc(labels: Sequence[int], scores: Sequence[float]) -> float:
    arr = [(float(s), int(y)) for s, y in zip(scores, labels)]
    if len(arr) == 0:
        return float("nan")
    npos = sum(1 for _, y in arr if y == 1)
    nneg = sum(1 for _, y in arr if y == 0)
    if npos <= 0 or nneg <= 0:
        return float("nan")
    arr.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    rank = 1.0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[j][0] == arr[i][0]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i) - 1.0))
        cnt_pos = sum(1 for k in range(i, j) if arr[k][1] == 1)
        rank_sum += avg_rank * float(cnt_pos)
        rank += float(j - i)
        i = j
    return float((rank_sum - float(npos) * (float(npos) + 1.0) * 0.5) / (float(npos) * float(nneg)))


def auc_pr(labels: Sequence[int], scores: Sequence[float]) -> float:
    arr = sorted([(float(s), int(y)) for s, y in zip(scores, labels)], key=lambda x: x[0], reverse=True)
    npos = sum(1 for _, y in arr if y == 1)
    if npos <= 0:
        return float("nan")
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for s, y in arr:
        if y == 1:
            tp += 1
        else:
            fp += 1
        recall = float(tp) / float(npos)
        precision = float(tp) / float(max(1, tp + fp))
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return float(ap)


def ks(labels: Sequence[int], scores: Sequence[float]) -> float:
    arr = sorted([(float(s), int(y)) for s, y in zip(scores, labels)], key=lambda x: x[0], reverse=True)
    npos = sum(1 for _, y in arr if y == 1)
    nneg = sum(1 for _, y in arr if y == 0)
    if npos <= 0 or nneg <= 0:
        return float("nan")
    tp = fp = 0
    best = 0.0
    for _, y in arr:
        if y == 1:
            tp += 1
        else:
            fp += 1
        d = abs(float(tp) / float(npos) - float(fp) / float(nneg))
        if d > best:
            best = d
    return float(best)


def spearman(labels: Sequence[int], values: Sequence[float]) -> float:
    # rank correlation between binary label and value; useful as monotonicity signal.
    n = len(labels)
    if n <= 1:
        return float("nan")
    vals = list(values)
    labs = [float(x) for x in labels]
    ord_v = sorted(range(n), key=lambda i: vals[i])
    rank_v = [0.0] * n
    i = 0
    r = 1.0
    while i < n:
        j = i + 1
        while j < n and vals[ord_v[j]] == vals[ord_v[i]]:
            j += 1
        rr = 0.5 * (r + (r + (j - i) - 1.0))
        for k in range(i, j):
            rank_v[ord_v[k]] = rr
        r += float(j - i)
        i = j
    ord_l = sorted(range(n), key=lambda i: labs[i])
    rank_l = [0.0] * n
    i = 0
    r = 1.0
    while i < n:
        j = i + 1
        while j < n and labs[ord_l[j]] == labs[ord_l[i]]:
            j += 1
        rr = 0.5 * (r + (r + (j - i) - 1.0))
        for k in range(i, j):
            rank_l[ord_l[k]] = rr
        r += float(j - i)
        i = j
    mv = sum(rank_v) / float(n)
    ml = sum(rank_l) / float(n)
    cov = sum((rank_v[i] - mv) * (rank_l[i] - ml) for i in range(n))
    sv = math.sqrt(max(1e-12, sum((rank_v[i] - mv) ** 2 for i in range(n))))
    sl = math.sqrt(max(1e-12, sum((rank_l[i] - ml) ** 2 for i in range(n))))
    return float(cov / (sv * sl))


def f1(tp: int, fp: int, fn: int) -> float:
    p = float(tp) / float(max(1, tp + fp))
    r = float(tp) / float(max(1, tp + fn))
    if p + r <= 0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def apply_threshold(scores: Sequence[float], th: float) -> List[int]:
    return [1 if float(s) >= float(th) else 0 for s in scores]


def pick_threshold_by_train_quantile(scores_train: Sequence[float], labels_train: Sequence[int]) -> Tuple[float, float]:
    # one-time calibration: choose threshold by positive prevalence.
    if len(scores_train) == 0:
        return 0.0, 0.0
    prev = float(sum(labels_train)) / float(max(1, len(labels_train)))
    arr = sorted(float(x) for x in scores_train)
    # top-prev are positive => threshold at (1-prev) quantile
    q = max(0.0, min(1.0, 1.0 - prev))
    i = int(round((len(arr) - 1) * q))
    i = max(0, min(len(arr) - 1, i))
    return float(arr[i]), prev


def metrics_from_pred(labels: Sequence[int], pred: Sequence[int]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for y, p in zip(labels, pred):
        y = int(y)
        p = int(p)
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 0 and p == 0:
            tn += 1
        elif y == 1 and p == 0:
            fn += 1
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "acc": float((tp + tn) / float(max(1, tp + fp + tn + fn))),
        "precision": float(tp / float(max(1, tp + fp))),
        "recall": float(tp / float(max(1, tp + fn))),
        "f1": float(f1(tp, fp, fn)),
    }


def parse_headset_specs(headset_json: Dict[str, Any]) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    f = set()
    h = set()
    for k, out in [("faithful_head_specs", f), ("harmful_head_specs", h)]:
        arr = headset_json.get(k, [])
        if isinstance(arr, list):
            for x in arr:
                s = str(x)
                if ":" in s:
                    a, b = s.split(":", 1)
                    ai = safe_int(a)
                    bi = safe_int(b)
                    if ai is not None and bi is not None:
                        out.add((int(ai), int(bi)))
    # fallback old format
    if not f:
        for x in headset_json.get("faithful_heads", []):
            li = safe_int(x.get("layer"))
            hi = safe_int(x.get("head"))
            if li is not None and hi is not None:
                f.add((int(li), int(hi)))
    if not h:
        for x in headset_json.get("harmful_heads", []):
            li = safe_int(x.get("layer"))
            hi = safe_int(x.get("head"))
            if li is not None and hi is not None:
                h.add((int(li), int(hi)))
    return f, h


def main() -> None:
    ap = argparse.ArgumentParser(description="Role-vs-feature evidence + GT-free surrogate v2 (POPE).")
    ap.add_argument("--role_csv", type=str, required=True, help="per_patch_role_effect.csv")
    ap.add_argument("--per_layer_trace_csv", type=str, required=True, help="per_layer_yes_trace.csv")
    ap.add_argument("--per_head_trace_csv", type=str, required=True, help="per_head_yes_trace.csv")
    ap.add_argument("--headset_json", type=str, required=True, help="headset.json")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--early_start", type=int, default=10)
    ap.add_argument("--early_end", type=int, default=15)
    ap.add_argument("--late_start", type=int, default=16)
    ap.add_argument("--late_end", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    role_rows = read_csv(os.path.abspath(args.role_csv))
    lay_rows = read_csv(os.path.abspath(args.per_layer_trace_csv))
    head_rows = read_csv(os.path.abspath(args.per_head_trace_csv))
    if len(role_rows) == 0:
        raise RuntimeError("No rows in role_csv")
    if len(lay_rows) == 0:
        raise RuntimeError("No rows in per_layer_trace_csv")
    if len(head_rows) == 0:
        raise RuntimeError("No rows in per_head_trace_csv")
    headset = json.load(open(os.path.abspath(args.headset_json), "r", encoding="utf-8"))
    faithful_specs, harmful_specs = parse_headset_specs(headset)

    # 1) build layer patch weight maps: id -> layer -> {patch: weight}
    id_layer_patch_w: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in lay_rows:
        sid = str(r.get("id") or "").strip()
        li = safe_int(r.get("block_layer_idx"))
        if sid == "" or li is None:
            continue
        idxs = jloads(r.get("yes_attn_vis_topk_idx_json"), [])
        wts = jloads(r.get("yes_attn_vis_topk_weight_json"), [])
        if not isinstance(idxs, list) or not isinstance(wts, list):
            continue
        k = min(len(idxs), len(wts))
        mp = id_layer_patch_w[sid][int(li)]
        for i in range(k):
            pi = safe_int(idxs[i])
            ww = safe_float(wts[i])
            if pi is None or ww is None:
                continue
            mp[int(pi)] = float(ww)

    # 2) sample-level faithful/harmful head attention
    id_head_harm: Dict[str, List[float]] = defaultdict(list)
    id_head_faith: Dict[str, List[float]] = defaultdict(list)
    for r in head_rows:
        sid = str(r.get("id") or "").strip()
        li = safe_int(r.get("block_layer_idx"))
        hi = safe_int(r.get("head_idx"))
        vv = safe_float(r.get("head_attn_vis_ratio"))
        if sid == "" or li is None or hi is None or vv is None:
            continue
        spec = (int(li), int(hi))
        if spec in harmful_specs:
            id_head_harm[sid].append(float(vv))
        if spec in faithful_specs:
            id_head_faith[sid].append(float(vv))

    # 3) candidate persistence in late layers
    late_layers = [x for x in range(int(args.late_start), int(args.late_end) + 1)]
    early_layers = [x for x in range(int(args.early_start), int(args.early_end) + 1)]

    merged: List[Dict[str, Any]] = []
    for r in role_rows:
        sid = str(r.get("id") or "").strip()
        grp = str(r.get("group") or "").strip()
        pidx = safe_int(r.get("candidate_patch_idx"))
        sim = safe_float(r.get("candidate_patch_sim_valid"))
        rank = safe_int(r.get("candidate_rank"))
        if sid == "" or pidx is None or sim is None or rank is None:
            continue
        role = str(r.get("role_label") or "").strip().lower()
        # teacher labels
        y_assertive = 1 if (grp == "fp_hall" and role == "harmful") else 0
        y_supportive = 1 if (grp == "tp_yes" and role == "supportive") else 0

        early_w = []
        for li in early_layers:
            early_w.append(float(id_layer_patch_w[sid].get(li, {}).get(int(pidx), 0.0)))
        late_w = []
        late_hit = 0
        for li in late_layers:
            w = float(id_layer_patch_w[sid].get(li, {}).get(int(pidx), 0.0))
            late_w.append(w)
            if w > 0.0:
                late_hit += 1
        early_support = mean(early_w)
        late_support = mean(late_w)
        uplift = math.log((late_support + float(args.eps)) / (early_support + float(args.eps)))
        persistence = float(late_hit) / float(max(1, len(late_layers)))

        harmful_head_attn = mean(id_head_harm.get(sid, []))
        faithful_head_attn = mean(id_head_faith.get(sid, []))

        merged.append(
            {
                "id": sid,
                "group": grp,
                "candidate_patch_idx": int(pidx),
                "candidate_rank": int(rank),
                "role_label": role,
                "y_assertive_fp_teacher": int(y_assertive),
                "y_supportive_tp_teacher": int(y_supportive),
                "sim": float(sim),
                "early_support": float(early_support),
                "late_support": float(late_support),
                "uplift": float(uplift),
                "persistence": float(persistence),
                "harmful_head_attn": float(harmful_head_attn),
                "faithful_head_attn": float(faithful_head_attn),
            }
        )

    if len(merged) == 0:
        raise RuntimeError("No merged rows. Check id/patch joins.")

    merged_csv = os.path.join(args.out_dir, "merged_role_feature_table.csv")
    write_csv(merged_csv, merged)

    # 4) Step-3 evidence: role vs feature (sim correlation weakness test)
    eval_rows: List[Dict[str, Any]] = []

    def eval_feature(task_rows: List[Dict[str, Any]], label_key: str, score_key: str, reverse: bool = False) -> Dict[str, float]:
        y = [int(r[label_key]) for r in task_rows]
        s = [float(r[score_key]) for r in task_rows]
        if reverse:
            s = [-x for x in s]
        return {
            "auc": float(auc_roc(y, s)),
            "pr_auc": float(auc_pr(y, s)),
            "ks": float(ks(y, s)),
            "spearman": float(spearman(y, s)),
        }

    fp_rows = [r for r in merged if r["group"] == "fp_hall"]
    tp_rows = [r for r in merged if r["group"] == "tp_yes"]

    feat_list = [
        ("sim", False),
        ("uplift", False),
        ("harmful_head_attn", False),
        ("faithful_head_attn", False),
        ("persistence", False),
        ("early_support", False),
    ]
    for task_name, rows, label_key in [
        ("assertive_from_fp", fp_rows, "y_assertive_fp_teacher"),
        ("supportive_from_tp", tp_rows, "y_supportive_tp_teacher"),
    ]:
        for feat, rev in feat_list:
            m = eval_feature(rows, label_key, feat, reverse=rev)
            eval_rows.append(
                {
                    "task": task_name,
                    "feature": feat,
                    "n": len(rows),
                    "auc": m["auc"],
                    "pr_auc": m["pr_auc"],
                    "ks": m["ks"],
                    "spearman": m["spearman"],
                }
            )

    # 5) Step-4 surrogate scores + calibration portability
    # split by id
    ids = sorted(set(str(r["id"]) for r in merged))
    rng = random.Random(int(args.seed))
    rng.shuffle(ids)
    ntr = int(round(float(len(ids)) * float(args.train_ratio)))
    ntr = max(1, min(len(ids) - 1, ntr)) if len(ids) > 1 else len(ids)
    tr_ids = set(ids[:ntr])

    tr = [r for r in merged if str(r["id"]) in tr_ids]
    va = [r for r in merged if str(r["id"]) not in tr_ids]

    # z-score refs from train split
    z_ref = {}
    for k in ["uplift", "harmful_head_attn", "persistence", "faithful_head_attn", "early_support"]:
        z_ref[k] = [float(x[k]) for x in tr]

    def add_scores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            zu = z_from_train(z_ref["uplift"], float(r["uplift"]))
            zh = z_from_train(z_ref["harmful_head_attn"], float(r["harmful_head_attn"]))
            zp = z_from_train(z_ref["persistence"], float(r["persistence"]))
            zf = z_from_train(z_ref["faithful_head_attn"], float(r["faithful_head_attn"]))
            ze = z_from_train(z_ref["early_support"], float(r["early_support"]))
            s_assertive = float(zu + zh + zp - zf)
            s_supportive = float(zf + ze - zu)
            rr = dict(r)
            rr["S_assertive"] = s_assertive
            rr["S_supportive"] = s_supportive
            out.append(rr)
        return out

    trs = add_scores(tr)
    vas = add_scores(va)
    all_scored = []
    for r in trs:
        rr = dict(r)
        rr["split"] = "train"
        all_scored.append(rr)
    for r in vas:
        rr = dict(r)
        rr["split"] = "val"
        all_scored.append(rr)

    # evaluate continuous score quality
    def eval_score(rows: List[Dict[str, Any]], label_key: str, score_key: str) -> Dict[str, float]:
        y = [int(r[label_key]) for r in rows]
        s = [float(r[score_key]) for r in rows]
        return {
            "auc": float(auc_roc(y, s)),
            "pr_auc": float(auc_pr(y, s)),
            "ks": float(ks(y, s)),
        }

    # task A: fp assertive
    tr_a = [r for r in trs if r["group"] == "fp_hall"]
    va_a = [r for r in vas if r["group"] == "fp_hall"]
    tr_s = eval_score(tr_a, "y_assertive_fp_teacher", "S_assertive")
    va_s = eval_score(va_a, "y_assertive_fp_teacher", "S_assertive")
    tr_sim = eval_score(tr_a, "y_assertive_fp_teacher", "sim")
    va_sim = eval_score(va_a, "y_assertive_fp_teacher", "sim")

    th_a, prev_a = pick_threshold_by_train_quantile(
        [float(r["S_assertive"]) for r in tr_a],
        [int(r["y_assertive_fp_teacher"]) for r in tr_a],
    )
    pred_va_a = apply_threshold([float(r["S_assertive"]) for r in va_a], th_a)
    cls_va_a = metrics_from_pred([int(r["y_assertive_fp_teacher"]) for r in va_a], pred_va_a)

    # task B: tp supportive
    tr_b = [r for r in trs if r["group"] == "tp_yes"]
    va_b = [r for r in vas if r["group"] == "tp_yes"]
    tr_sb = eval_score(tr_b, "y_supportive_tp_teacher", "S_supportive")
    va_sb = eval_score(va_b, "y_supportive_tp_teacher", "S_supportive")
    tr_sim_b = eval_score(tr_b, "y_supportive_tp_teacher", "sim")
    va_sim_b = eval_score(va_b, "y_supportive_tp_teacher", "sim")

    th_b, prev_b = pick_threshold_by_train_quantile(
        [float(r["S_supportive"]) for r in tr_b],
        [int(r["y_supportive_tp_teacher"]) for r in tr_b],
    )
    pred_va_b = apply_threshold([float(r["S_supportive"]) for r in va_b], th_b)
    cls_va_b = metrics_from_pred([int(r["y_supportive_tp_teacher"]) for r in va_b], pred_va_b)

    surrogate_eval_rows = [
        {
            "task": "assertive_from_fp",
            "split": "train",
            "score": "S_assertive",
            "auc": tr_s["auc"],
            "pr_auc": tr_s["pr_auc"],
            "ks": tr_s["ks"],
        },
        {
            "task": "assertive_from_fp",
            "split": "val",
            "score": "S_assertive",
            "auc": va_s["auc"],
            "pr_auc": va_s["pr_auc"],
            "ks": va_s["ks"],
            "threshold_train_quantile": th_a,
            "val_acc_at_th": cls_va_a["acc"],
            "val_f1_at_th": cls_va_a["f1"],
            "val_precision_at_th": cls_va_a["precision"],
            "val_recall_at_th": cls_va_a["recall"],
        },
        {
            "task": "assertive_from_fp",
            "split": "val",
            "score": "sim_baseline",
            "auc": va_sim["auc"],
            "pr_auc": va_sim["pr_auc"],
            "ks": va_sim["ks"],
        },
        {
            "task": "supportive_from_tp",
            "split": "train",
            "score": "S_supportive",
            "auc": tr_sb["auc"],
            "pr_auc": tr_sb["pr_auc"],
            "ks": tr_sb["ks"],
        },
        {
            "task": "supportive_from_tp",
            "split": "val",
            "score": "S_supportive",
            "auc": va_sb["auc"],
            "pr_auc": va_sb["pr_auc"],
            "ks": va_sb["ks"],
            "threshold_train_quantile": th_b,
            "val_acc_at_th": cls_va_b["acc"],
            "val_f1_at_th": cls_va_b["f1"],
            "val_precision_at_th": cls_va_b["precision"],
            "val_recall_at_th": cls_va_b["recall"],
        },
        {
            "task": "supportive_from_tp",
            "split": "val",
            "score": "sim_baseline",
            "auc": va_sim_b["auc"],
            "pr_auc": va_sim_b["pr_auc"],
            "ks": va_sim_b["ks"],
        },
    ]

    eval_csv = os.path.join(args.out_dir, "feature_role_eval.csv")
    write_csv(eval_csv, eval_rows)
    surrogate_csv = os.path.join(args.out_dir, "surrogate_eval.csv")
    write_csv(surrogate_csv, surrogate_eval_rows)
    scored_csv = os.path.join(args.out_dir, "merged_role_feature_scored.csv")
    write_csv(scored_csv, all_scored)

    # rank-bin role ratios for explicit criterion
    rank_bins = [(0, 7), (8, 15), (16, 31)]
    ratio_rows = []
    for grp, rows in [("fp_hall", fp_rows), ("tp_yes", tp_rows)]:
        for lo, hi in rank_bins:
            rr = [r for r in rows if int(r["candidate_rank"]) >= lo and int(r["candidate_rank"]) <= hi]
            n = len(rr)
            if n == 0:
                continue
            harmful_ratio = sum(1 for r in rr if str(r["role_label"]) == "harmful") / float(n)
            supportive_ratio = sum(1 for r in rr if str(r["role_label"]) == "supportive") / float(n)
            ratio_rows.append(
                {
                    "group": grp,
                    "rank_bin": f"{lo}-{hi}",
                    "n": n,
                    "harmful_ratio": harmful_ratio,
                    "supportive_ratio": supportive_ratio,
                }
            )
    ratio_csv = os.path.join(args.out_dir, "role_ratio_by_rank.csv")
    write_csv(ratio_csv, ratio_rows)

    summary = {
        "inputs": {
            "role_csv": os.path.abspath(args.role_csv),
            "per_layer_trace_csv": os.path.abspath(args.per_layer_trace_csv),
            "per_head_trace_csv": os.path.abspath(args.per_head_trace_csv),
            "headset_json": os.path.abspath(args.headset_json),
            "early_start": int(args.early_start),
            "early_end": int(args.early_end),
            "late_start": int(args.late_start),
            "late_end": int(args.late_end),
            "train_ratio": float(args.train_ratio),
            "seed": int(args.seed),
        },
        "counts": {
            "n_role_rows": int(len(role_rows)),
            "n_merged_rows": int(len(merged)),
            "n_fp_rows": int(len(fp_rows)),
            "n_tp_rows": int(len(tp_rows)),
            "n_faithful_specs": int(len(faithful_specs)),
            "n_harmful_specs": int(len(harmful_specs)),
            "n_train_ids": int(len(tr_ids)),
            "n_val_ids": int(len(ids) - len(tr_ids)),
        },
        "outputs": {
            "merged_csv": merged_csv,
            "feature_eval_csv": eval_csv,
            "surrogate_eval_csv": surrogate_csv,
            "scored_csv": scored_csv,
            "ratio_by_rank_csv": ratio_csv,
            "summary_json": os.path.join(args.out_dir, "summary.json"),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", merged_csv)
    print("[saved]", eval_csv)
    print("[saved]", surrogate_csv)
    print("[saved]", scored_csv)
    print("[saved]", ratio_csv)
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
