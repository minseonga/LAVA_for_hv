#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if safe_float(v) is not None)
    if len(xs) == 0:
        return None
    if len(xs) == 1:
        return float(xs[0])
    p = min(1.0, max(0.0, float(q))) * (len(xs) - 1)
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    if lo == hi:
        return float(xs[lo])
    w = p - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def mean(xs: Sequence[float]) -> Optional[float]:
    ys = [float(x) for x in xs if safe_float(x) is not None]
    if len(ys) == 0:
        return None
    return float(sum(ys) / len(ys))


def ks_stat(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    a = sorted(float(x) for x in pos if safe_float(x) is not None)
    b = sorted(float(x) for x in neg if safe_float(x) is not None)
    if len(a) == 0 or len(b) == 0:
        return None
    i = j = 0
    n, m = len(a), len(b)
    d = 0.0
    while i < n or j < m:
        va = a[i] if i < n else float("inf")
        vb = b[j] if j < m else float("inf")
        v = va if va <= vb else vb
        while i < n and a[i] <= v:
            i += 1
        while j < m and b[j] <= v:
            j += 1
        d = max(d, abs(float(i) / n - float(j) / m))
    return float(d)


def auc_pos_gt_neg(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    p = [float(x) for x in pos if safe_float(x) is not None]
    n = [float(x) for x in neg if safe_float(x) is not None]
    if len(p) == 0 or len(n) == 0:
        return None
    arr: List[Tuple[float, int]] = [(v, 1) for v in p] + [(v, 0) for v in n]
    arr.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    N = len(arr)
    while i < N:
        j = i + 1
        while j < N and arr[j][0] == arr[i][0]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # 1-indexed
        for k in range(i, j):
            if arr[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    n_pos = len(p)
    n_neg = len(n)
    u = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / float(n_pos * n_neg))


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


def stable_split(sid: str, mod: int = 5) -> int:
    h = hashlib.md5(str(sid).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(mod)


def zscore(v: np.ndarray) -> np.ndarray:
    mu = float(np.mean(v))
    sd = float(np.std(v))
    if sd <= 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.05, n_iter: int = 1400, l2: float = 1e-4) -> np.ndarray:
    w = np.zeros(X.shape[1], dtype=np.float64)
    n = float(max(1, X.shape[0]))
    for _ in range(int(n_iter)):
        p = sigmoid(X @ w)
        g = (X.T @ (p - y)) / n
        g[1:] += float(l2) * w[1:]
        w -= float(lr) * g
    return w


def pred_logreg(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


def load_rows(run_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ps = os.path.join(run_dir, "per_sample.csv")
    pt = os.path.join(run_dir, "per_token.csv")
    if not os.path.isfile(ps):
        raise FileNotFoundError(f"Missing file: {ps}")
    if not os.path.isfile(pt):
        raise FileNotFoundError(f"Missing file: {pt}")

    sample_rows: List[Dict[str, Any]] = []
    with open(ps, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid == "":
                continue
            ok = parse_bool(r.get("is_success"))
            n_tok = safe_float(r.get("n_gen_tokens"))
            sample_rows.append({"id": sid, "is_success": 1 if ok else 0, "n_gen_tokens": n_tok})

    token_rows: List[Dict[str, Any]] = []
    with open(pt, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid == "":
                continue
            step = safe_float(r.get("step_idx"))
            lq = safe_float(r.get("logit_q"))
            lv = safe_float(r.get("logit_vq"))
            lpq = safe_float(r.get("logp_q"))
            lpv = safe_float(r.get("logp_vq"))
            v1 = safe_float(r.get("vpmi_logit"))
            v2 = safe_float(r.get("vpmi_logp"))
            if step is None or lq is None or lv is None or lpq is None or lpv is None:
                continue
            if v1 is None:
                v1 = float(lv) - float(lq)
            if v2 is None:
                v2 = float(lpv) - float(lpq)
            token_rows.append(
                {
                    "id": sid,
                    "step_idx": int(step),
                    "logit_q": float(lq),
                    "logit_vq": float(lv),
                    "logp_q": float(lpq),
                    "logp_vq": float(lpv),
                    "vpmi_logit": float(v1),
                    "vpmi_logp": float(v2),
                }
            )
    return sample_rows, token_rows


def build_sample_features(sample_rows: List[Dict[str, Any]], token_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in token_rows:
        by_id.setdefault(str(r["id"]), []).append(r)
    sample_map = {str(r["id"]): r for r in sample_rows}

    out: List[Dict[str, Any]] = []
    for sid, toks in by_id.items():
        if sid not in sample_map:
            continue
        s = sample_map[sid]
        toks_s = sorted(toks, key=lambda x: int(x["step_idx"]))
        m = len(toks_s)
        if m == 0:
            continue
        vpmi_logit = [float(r["vpmi_logit"]) for r in toks_s]
        vpmi_logp = [float(r["vpmi_logp"]) for r in toks_s]
        logit_q = [float(r["logit_q"]) for r in toks_s]
        logp_q = [float(r["logp_q"]) for r in toks_s]

        ratio_logit = [float(v / (abs(q) + 1e-6)) for v, q in zip(vpmi_logit, logit_q)]
        ratio_logp = [float(v / (abs(q) + 1e-6)) for v, q in zip(vpmi_logp, logp_q)]

        out.append(
            {
                "id": sid,
                "is_success": int(s["is_success"]),
                "n_gen_tokens": int(safe_float(s.get("n_gen_tokens")) or m),
                "f_vpmi_logit_mean": mean(vpmi_logit),
                "f_vpmi_logit_min": min(vpmi_logit),
                "f_vpmi_logit_max": max(vpmi_logit),
                "f_vpmi_logp_mean": mean(vpmi_logp),
                "f_vpmi_logp_min": min(vpmi_logp),
                "f_vpmi_logp_max": max(vpmi_logp),
                "f_logit_q_mean": mean(logit_q),
                "f_logp_q_mean": mean(logp_q),
                "f_ratio_logit_mean": mean(ratio_logit),
                "f_ratio_logp_mean": mean(ratio_logp),
            }
        )
    return out


def run_univariate(sample_feats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fns = [
        "f_vpmi_logit_mean",
        "f_vpmi_logit_min",
        "f_vpmi_logit_max",
        "f_vpmi_logp_mean",
        "f_vpmi_logp_min",
        "f_vpmi_logp_max",
        "f_ratio_logit_mean",
        "f_ratio_logp_mean",
    ]
    out: List[Dict[str, Any]] = []
    for fn in fns:
        pos = [float(r[fn]) for r in sample_feats if r.get(fn) is not None and int(r["is_success"]) == 1]
        neg = [float(r[fn]) for r in sample_feats if r.get(fn) is not None and int(r["is_success"]) == 0]
        out.append(
            {
                "feature": fn,
                "n_success": int(len(pos)),
                "n_failure": int(len(neg)),
                "mean_success": mean(pos),
                "mean_failure": mean(neg),
                "ks": ks_stat(pos, neg),
                "auc_pos_gt_neg": auc_pos_gt_neg(pos, neg),
            }
        )
    out.sort(key=lambda r: float(r.get("ks") or -1.0), reverse=True)
    return out


def run_diagA(token_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Compare low-prior spike behavior for both contrast definitions.
    out: List[Dict[str, Any]] = []
    cfgs = [
        ("vpmi_logit", "logit_q"),
        ("vpmi_logp", "logp_q"),
    ]
    for vcol, qcol in cfgs:
        q_vals = [float(r[qcol]) for r in token_rows]
        v_vals = [float(r[vcol]) for r in token_rows]
        q10 = quantile(q_vals, 0.10)
        v90 = quantile(v_vals, 0.90)
        if q10 is None or v90 is None:
            continue
        low = [r for r in token_rows if float(r[qcol]) <= float(q10)]
        succ = [r for r in low if int(r.get("is_success", 0)) == 1]
        fail = [r for r in low if int(r.get("is_success", 0)) == 0]

        def sr(xs: List[Dict[str, Any]]) -> Optional[float]:
            if len(xs) == 0:
                return None
            n = sum(1 for r in xs if float(r[vcol]) >= float(v90))
            return float(n / len(xs))

        succ_v = [float(r[vcol]) for r in succ]
        fail_v = [float(r[vcol]) for r in fail]
        out.append(
            {
                "contrast": vcol,
                "prior_col": qcol,
                "low_prior_q10": float(q10),
                "spike_q90": float(v90),
                "n_low_prior": int(len(low)),
                "spike_rate_success_low_prior": sr(succ),
                "spike_rate_failure_low_prior": sr(fail),
                "mean_success_low_prior": mean(succ_v),
                "mean_failure_low_prior": mean(fail_v),
                "ks_success_vs_failure_low_prior": ks_stat(succ_v, fail_v),
                "auc_success_gt_failure_low_prior": auc_pos_gt_neg(succ_v, fail_v),
            }
        )
    return out


def run_diagC(sample_feats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # controls -> + single contrast feature
    # controls: bias, z(f_logit_q_mean), z(n_gen_tokens)
    rr = [r for r in sample_feats if r.get("f_logit_q_mean") is not None and r.get("n_gen_tokens") is not None]
    if len(rr) == 0:
        return []

    tr = [r for r in rr if stable_split(str(r["id"]), mod=5) != 0]
    te = [r for r in rr if stable_split(str(r["id"]), mod=5) == 0]
    if len(tr) == 0 or len(te) == 0:
        return []

    def make_X(rows: List[Dict[str, Any]], feat: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        y = np.array([float(r["is_success"]) for r in rows], dtype=np.float64)
        c1 = np.array([float(r["f_logit_q_mean"]) for r in rows], dtype=np.float64)
        c2 = np.array([float(r["n_gen_tokens"]) for r in rows], dtype=np.float64)
        cols = [np.ones_like(c1), zscore(c1), zscore(c2)]
        if feat is not None:
            f = np.array([float(r[feat]) for r in rows], dtype=np.float64)
            cols.append(zscore(f))
        X = np.stack(cols, axis=1)
        return X, y

    Xtr0, ytr = make_X(tr, feat=None)
    Xte0, yte = make_X(te, feat=None)
    w0 = fit_logreg(Xtr0, ytr, lr=0.06, n_iter=1600, l2=1e-4)
    p0 = pred_logreg(Xte0, w0)
    p0_pos = [float(p0[i]) for i in range(len(p0)) if int(yte[i]) == 1]
    p0_neg = [float(p0[i]) for i in range(len(p0)) if int(yte[i]) == 0]
    auc0 = auc_pos_gt_neg(p0_pos, p0_neg)
    ks0 = ks_stat(p0_pos, p0_neg)

    cand_feats = [
        "f_vpmi_logit_mean",
        "f_vpmi_logp_mean",
        "f_ratio_logit_mean",
        "f_ratio_logp_mean",
    ]
    out: List[Dict[str, Any]] = []
    for fn in cand_feats:
        tr2 = [r for r in tr if r.get(fn) is not None]
        te2 = [r for r in te if r.get(fn) is not None]
        if len(tr2) == 0 or len(te2) == 0:
            continue
        Xtr1, ytr1 = make_X(tr2, feat=fn)
        Xte1, yte1 = make_X(te2, feat=fn)
        w1 = fit_logreg(Xtr1, ytr1, lr=0.06, n_iter=1600, l2=1e-4)
        p1 = pred_logreg(Xte1, w1)
        p1_pos = [float(p1[i]) for i in range(len(p1)) if int(yte1[i]) == 1]
        p1_neg = [float(p1[i]) for i in range(len(p1)) if int(yte1[i]) == 0]
        auc1 = auc_pos_gt_neg(p1_pos, p1_neg)
        ks1 = ks_stat(p1_pos, p1_neg)
        out.append(
            {
                "feature_added": fn,
                "n_train": int(len(tr2)),
                "n_test": int(len(te2)),
                "auc_controls_only": auc0,
                "auc_controls_plus_feat": auc1,
                "delta_auc": (None if auc0 is None or auc1 is None else float(auc1 - auc0)),
                "ks_controls_only": ks0,
                "ks_controls_plus_feat": ks1,
                "delta_ks": (None if ks0 is None or ks1 is None else float(ks1 - ks0)),
            }
        )
    out.sort(key=lambda r: float(r.get("delta_auc") or -1e9), reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline contrast-definition check for V-PMI decomposition.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    sample_rows, token_rows = load_rows(run_dir)
    sample_map = {str(r["id"]): int(r["is_success"]) for r in sample_rows}
    for r in token_rows:
        r["is_success"] = int(sample_map.get(str(r["id"]), 0))

    sample_feats = build_sample_features(sample_rows, token_rows)

    uni = run_univariate(sample_feats)
    da = run_diagA(token_rows)
    dc = run_diagC(sample_feats)

    write_csv(os.path.join(out_dir, "contrast_univariate_sample.csv"), uni)
    write_csv(os.path.join(out_dir, "contrast_diagA_lowprior.csv"), da)
    write_csv(os.path.join(out_dir, "contrast_diagC_incremental.csv"), dc)

    summary = {
        "inputs": {
            "run_dir": run_dir,
            "n_samples": int(len(sample_feats)),
            "n_tokens": int(len(token_rows)),
        },
        "top_univariate": uni[:10],
        "diagA_lowprior": da,
        "diagC_incremental_top": dc[:10],
        "outputs": {
            "univariate_csv": os.path.join(out_dir, "contrast_univariate_sample.csv"),
            "diagA_csv": os.path.join(out_dir, "contrast_diagA_lowprior.csv"),
            "diagC_csv": os.path.join(out_dir, "contrast_diagC_incremental.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "contrast_univariate_sample.csv"))
    print("[saved]", os.path.join(out_dir, "contrast_diagA_lowprior.csv"))
    print("[saved]", os.path.join(out_dir, "contrast_diagC_incremental.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
