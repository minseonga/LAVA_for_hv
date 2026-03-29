#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Any, Dict, List, Sequence, Tuple


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


def normalize_group(g: str) -> str:
    x = str(g or "").strip().lower()
    if x in {"fp", "fp_hall"}:
        return "fp_hall"
    if x in {"tp", "tp_yes"}:
        return "tp_yes"
    if x in {"tn", "tn_no"}:
        return "tn_no"
    if x in {"fn", "fn_miss"}:
        return "fn_miss"
    return x


def calc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    arr = [(float(s), int(y)) for s, y in zip(scores, labels)]
    if len(arr) == 0:
        return float("nan")
    npos = sum(1 for _, y in arr if y == 1)
    nneg = sum(1 for _, y in arr if y == 0)
    if npos == 0 or nneg == 0:
        return float("nan")
    arr = sorted(arr, key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    rank = 1.0
    n = len(arr)
    while i < n:
        j = i + 1
        while j < n and arr[j][0] == arr[i][0]:
            j += 1
        avg_rank = 0.5 * (rank + (rank + (j - i) - 1.0))
        cnt_pos = sum(1 for k in range(i, j) if arr[k][1] == 1)
        rank_sum += avg_rank * float(cnt_pos)
        rank += float(j - i)
        i = j
    auc = (rank_sum - float(npos) * (float(npos) + 1.0) * 0.5) / (float(npos) * float(nneg))
    return float(auc)


def calc_ks(labels: Sequence[int], scores: Sequence[float]) -> float:
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


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    p = float(tp) / float(max(1, tp + fp))
    r = float(tp) / float(max(1, tp + fn))
    if p + r <= 0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def eval_rule(rows: Sequence[Dict[str, Any]], sim_thr: float, rank_thr: int, label_key: str) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    labels = []
    scores = []
    for r in rows:
        y = int(r[label_key])
        sim = float(r["sim"])
        rk = int(r["rank"])
        pred = int(sim <= float(sim_thr) and rk <= int(rank_thr))
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        elif pred == 0 and y == 1:
            fn += 1
        labels.append(y)
        # GT-free monotonic score: lower sim and lower rank means higher assertiveness/supportiveness.
        score = (-sim) + 0.02 * float(max(0, 31 - rk))
        scores.append(score)
    auc = calc_auc(labels, scores)
    ks = calc_ks(labels, scores)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "acc": float((tp + tn) / float(max(1, tp + fp + tn + fn))),
        "f1": float(f1_from_counts(tp, fp, fn)),
        "precision": float(tp / float(max(1, tp + fp))),
        "recall": float(tp / float(max(1, tp + fn))),
        "auc": float(auc),
        "ks": float(ks),
    }


def split_by_id(rows: Sequence[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ids = sorted(set(str(r["id"]) for r in rows))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    ntr = int(round(float(len(ids)) * float(train_ratio)))
    ntr = max(1, min(len(ids) - 1, ntr)) if len(ids) > 1 else len(ids)
    tr_ids = set(ids[:ntr])
    tr = [r for r in rows if str(r["id"]) in tr_ids]
    va = [r for r in rows if str(r["id"]) not in tr_ids]
    return tr, va


def quantiles(vals: Sequence[float], qs: Sequence[float]) -> List[float]:
    arr = sorted(float(v) for v in vals)
    if len(arr) == 0:
        return [float("nan") for _ in qs]
    out = []
    n = len(arr)
    for q in qs:
        i = int(round((n - 1) * float(q)))
        i = max(0, min(n - 1, i))
        out.append(float(arr[i]))
    return out


def prepare_rows(role_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fp = []
    tp = []
    for r in role_rows:
        g = normalize_group(r.get("group"))
        sid = str(r.get("id") or "").strip()
        if sid == "":
            continue
        try:
            sim = float(r.get("candidate_patch_sim_valid"))
            rank = int(r.get("candidate_rank"))
        except Exception:
            continue
        role = str(r.get("role_label") or "").strip().lower()
        if g == "fp_hall":
            y = int(role == "harmful")  # assertive proxy
            fp.append({"id": sid, "sim": sim, "rank": rank, "label_assertive": y})
        elif g == "tp_yes":
            y = int(role == "supportive")
            tp.append({"id": sid, "sim": sim, "rank": rank, "label_supportive": y})
    return fp, tp


def best_threshold_grid(rows_train: Sequence[Dict[str, Any]], label_key: str) -> Tuple[float, int, Dict[str, float]]:
    sim_vals = [float(r["sim"]) for r in rows_train]
    sim_cands = quantiles(sim_vals, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7])
    rank_cands = [3, 5, 7, 9, 11, 15, 23, 31]
    best = None
    for st in sim_cands:
        if not math.isfinite(st):
            continue
        for rt in rank_cands:
            m = eval_rule(rows_train, sim_thr=float(st), rank_thr=int(rt), label_key=label_key)
            key = (float(m["f1"]), float(m["acc"]))
            if (best is None) or (key > best[0]):
                best = (key, float(st), int(rt), m)
    if best is None:
        return 0.0, 31, {"f1": 0.0, "acc": 0.0}
    return float(best[1]), int(best[2]), dict(best[3])


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit GT-free surrogate rules from single-patch role labels.")
    ap.add_argument("--role_csv", type=str, required=True, help="per_patch_role_effect.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    role_rows = read_csv(os.path.abspath(args.role_csv))
    if len(role_rows) == 0:
        raise RuntimeError("No rows in role_csv.")

    fp_rows, tp_rows = prepare_rows(role_rows)
    if len(fp_rows) == 0 and len(tp_rows) == 0:
        raise RuntimeError("No usable rows for surrogate fitting.")

    out_eval: List[Dict[str, Any]] = []
    rule_json: Dict[str, Any] = {"assertive_rule": {}, "supportive_rule": {}}

    if len(fp_rows) > 1:
        tr, va = split_by_id(fp_rows, train_ratio=float(args.train_ratio), seed=int(args.seed))
        sim_thr, rank_thr, tr_m = best_threshold_grid(tr, label_key="label_assertive")
        va_m = eval_rule(va, sim_thr=sim_thr, rank_thr=rank_thr, label_key="label_assertive")
        out_eval.append(
            {
                "task": "assertive_from_fp_hall",
                "n_train": len(tr),
                "n_val": len(va),
                "sim_thr": sim_thr,
                "rank_thr": rank_thr,
                "train_f1": tr_m.get("f1"),
                "train_acc": tr_m.get("acc"),
                "val_f1": va_m.get("f1"),
                "val_acc": va_m.get("acc"),
                "val_auc": va_m.get("auc"),
                "val_ks": va_m.get("ks"),
                "val_precision": va_m.get("precision"),
                "val_recall": va_m.get("recall"),
            }
        )
        rule_json["assertive_rule"] = {
            "sim_le": float(sim_thr),
            "rank_le": int(rank_thr),
            "source": "fp_hall_teacher_labels",
        }

    if len(tp_rows) > 1:
        tr, va = split_by_id(tp_rows, train_ratio=float(args.train_ratio), seed=int(args.seed) + 13)
        sim_thr, rank_thr, tr_m = best_threshold_grid(tr, label_key="label_supportive")
        va_m = eval_rule(va, sim_thr=sim_thr, rank_thr=rank_thr, label_key="label_supportive")
        out_eval.append(
            {
                "task": "supportive_from_tp_yes",
                "n_train": len(tr),
                "n_val": len(va),
                "sim_thr": sim_thr,
                "rank_thr": rank_thr,
                "train_f1": tr_m.get("f1"),
                "train_acc": tr_m.get("acc"),
                "val_f1": va_m.get("f1"),
                "val_acc": va_m.get("acc"),
                "val_auc": va_m.get("auc"),
                "val_ks": va_m.get("ks"),
                "val_precision": va_m.get("precision"),
                "val_recall": va_m.get("recall"),
            }
        )
        rule_json["supportive_rule"] = {
            "sim_le": float(sim_thr),
            "rank_le": int(rank_thr),
            "source": "tp_yes_teacher_labels",
        }

    eval_csv = os.path.join(out_dir, "surrogate_eval.csv")
    write_csv(eval_csv, out_eval)

    summary = {
        "inputs": {
            "role_csv": os.path.abspath(args.role_csv),
            "train_ratio": float(args.train_ratio),
            "seed": int(args.seed),
        },
        "counts": {
            "n_role_rows": int(len(role_rows)),
            "n_fp_rows": int(len(fp_rows)),
            "n_tp_rows": int(len(tp_rows)),
        },
        "rules": rule_json,
        "outputs": {
            "eval_csv": eval_csv,
            "rule_json": os.path.join(out_dir, "surrogate_rules.json"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    with open(os.path.join(out_dir, "surrogate_rules.json"), "w", encoding="utf-8") as f:
        json.dump(rule_json, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[saved] {eval_csv}")
    print(f"[saved] {os.path.join(out_dir, 'surrogate_rules.json')}")
    print(f"[saved] {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()

