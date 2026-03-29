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


def token_is_prefix(tok: str, step_idx: int) -> bool:
    t = str(tok or "")
    if int(step_idx) == 0:
        return True
    # SentencePiece/LLaMA: ▁ ; GPT-BPE fallback: Ġ
    return t.startswith("\u2581") or t.startswith("Ġ")


def zscore(v: np.ndarray) -> np.ndarray:
    mu = float(np.mean(v))
    sd = float(np.std(v))
    if sd <= 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    n_iter: int = 1200,
    l2: float = 1e-4,
) -> np.ndarray:
    w = np.zeros(X.shape[1], dtype=np.float64)
    n = float(max(1, X.shape[0]))
    for _ in range(int(n_iter)):
        p = sigmoid(X @ w)
        g = (X.T @ (p - y)) / n
        g[1:] += float(l2) * w[1:]  # do not penalize bias term
        w -= float(lr) * g
    return w


def predict_logreg(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


def stable_group_split(sid: str, mod: int = 5) -> int:
    h = hashlib.md5(str(sid).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(mod)


def load_token_rows(run_dir: str) -> List[Dict[str, Any]]:
    per_token = os.path.join(run_dir, "per_token.csv")
    if not os.path.isfile(per_token):
        raise FileNotFoundError(f"Missing file: {per_token}")
    rows: List[Dict[str, Any]] = []
    with open(per_token, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "")
            if sid == "":
                continue
            step = safe_float(r.get("step_idx"))
            logit_q = safe_float(r.get("logit_q"))
            vpmi = safe_float(r.get("vpmi_logit"))
            if step is None or logit_q is None or vpmi is None:
                continue
            ok = parse_bool(r.get("is_success"))
            tok = str(r.get("token_str") or "")
            rows.append(
                {
                    "id": sid,
                    "step_idx": int(step),
                    "logit_q": float(logit_q),
                    "vpmi_logit": float(vpmi),
                    "is_success": 1 if ok else 0,
                    "token_str": tok,
                }
            )
    return rows


def attach_position_and_kind(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(str(r["id"]), []).append(r)
    out: List[Dict[str, Any]] = []
    for sid, rr in by_id.items():
        rr_sorted = sorted(rr, key=lambda x: int(x["step_idx"]))
        m = len(rr_sorted)
        for i, r in enumerate(rr_sorted):
            pos_norm = float(i / max(1, m - 1))
            pref = token_is_prefix(str(r["token_str"]), int(r["step_idx"]))
            row = dict(r)
            row["pos_norm"] = pos_norm
            row["is_prefix"] = 1 if pref else 0
            row["is_continuation"] = 0 if pref else 1
            out.append(row)
    return out


def run_diag_a(rows: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    logit_q_vals = [float(r["logit_q"]) for r in rows]
    vpmi_vals = [float(r["vpmi_logit"]) for r in rows]
    q10 = quantile(logit_q_vals, 0.10)
    q90_v = quantile(vpmi_vals, 0.90)
    assert q10 is not None and q90_v is not None

    low_prior = [r for r in rows if float(r["logit_q"]) <= float(q10)]
    succ = [r for r in low_prior if int(r["is_success"]) == 1]
    fail = [r for r in low_prior if int(r["is_success"]) == 0]

    def spike_rate(xs: List[Dict[str, Any]]) -> Optional[float]:
        if len(xs) == 0:
            return None
        n = sum(1 for r in xs if float(r["vpmi_logit"]) >= float(q90_v))
        return float(n / len(xs))

    succ_sr = spike_rate(succ)
    fail_sr = spike_rate(fail)
    succ_v = [float(r["vpmi_logit"]) for r in succ]
    fail_v = [float(r["vpmi_logit"]) for r in fail]

    # decile profile across logit_q
    d_edges: List[float] = []
    for i in range(11):
        q = quantile(logit_q_vals, i / 10.0)
        d_edges.append(float(q if q is not None else 0.0))
    dec_rows: List[Dict[str, Any]] = []
    for di in range(10):
        lo = d_edges[di]
        hi = d_edges[di + 1]
        cell = [r for r in rows if float(r["logit_q"]) >= lo and (float(r["logit_q"]) < hi or (di == 9 and float(r["logit_q"]) <= hi))]
        if len(cell) == 0:
            continue
        cs = [r for r in cell if int(r["is_success"]) == 1]
        cf = [r for r in cell if int(r["is_success"]) == 0]
        dec_rows.append(
            {
                "decile": int(di),
                "logit_q_lo": float(lo),
                "logit_q_hi": float(hi),
                "n": int(len(cell)),
                "n_success": int(len(cs)),
                "n_failure": int(len(cf)),
                "vpmi_mean_success": mean([float(r["vpmi_logit"]) for r in cs]),
                "vpmi_mean_failure": mean([float(r["vpmi_logit"]) for r in cf]),
                "vpmi_spike_rate_success": spike_rate(cs),
                "vpmi_spike_rate_failure": spike_rate(cf),
            }
        )
    write_csv(os.path.join(out_dir, "diagA_logitq_deciles.csv"), dec_rows)

    return {
        "low_prior_q10_logit_q": float(q10),
        "global_vpmi_q90_spike_thr": float(q90_v),
        "n_low_prior": int(len(low_prior)),
        "n_low_prior_success": int(len(succ)),
        "n_low_prior_failure": int(len(fail)),
        "vpmi_spike_rate_success_low_prior": succ_sr,
        "vpmi_spike_rate_failure_low_prior": fail_sr,
        "vpmi_mean_success_low_prior": mean(succ_v),
        "vpmi_mean_failure_low_prior": mean(fail_v),
        "vpmi_auc_success_gt_failure_low_prior": auc_pos_gt_neg(succ_v, fail_v),
        "vpmi_ks_success_vs_failure_low_prior": ks_stat(succ_v, fail_v),
        "deciles_csv": os.path.join(out_dir, "diagA_logitq_deciles.csv"),
    }


def run_diag_b(rows: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    pref = [r for r in rows if int(r["is_prefix"]) == 1]
    cont = [r for r in rows if int(r["is_continuation"]) == 1]

    # Transition-level drop: prefix -> continuation next token.
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(str(r["id"]), []).append(r)
    trans: List[Dict[str, Any]] = []
    for sid, rr in by_id.items():
        s = sorted(rr, key=lambda x: int(x["step_idx"]))
        for i in range(0, len(s) - 1):
            a = s[i]
            b = s[i + 1]
            if int(a["is_prefix"]) == 1 and int(b["is_continuation"]) == 1:
                trans.append(
                    {
                        "id": sid,
                        "is_success": int(a["is_success"]),
                        "delta_vpmi_cont_minus_pref": float(b["vpmi_logit"]) - float(a["vpmi_logit"]),
                        "pref_vpmi": float(a["vpmi_logit"]),
                        "cont_vpmi": float(b["vpmi_logit"]),
                    }
                )

    write_csv(os.path.join(out_dir, "diagB_prefix_to_cont_transitions.csv"), trans)

    def stats(xs: List[float]) -> Dict[str, Any]:
        return {
            "n": int(len(xs)),
            "mean": mean(xs),
            "median": quantile(xs, 0.5),
            "q10": quantile(xs, 0.1),
            "q90": quantile(xs, 0.9),
        }

    pref_s = [float(r["vpmi_logit"]) for r in pref if int(r["is_success"]) == 1]
    pref_f = [float(r["vpmi_logit"]) for r in pref if int(r["is_success"]) == 0]
    cont_s = [float(r["vpmi_logit"]) for r in cont if int(r["is_success"]) == 1]
    cont_f = [float(r["vpmi_logit"]) for r in cont if int(r["is_success"]) == 0]

    td_s = [float(r["delta_vpmi_cont_minus_pref"]) for r in trans if int(r["is_success"]) == 1]
    td_f = [float(r["delta_vpmi_cont_minus_pref"]) for r in trans if int(r["is_success"]) == 0]

    return {
        "prefix_vpmi_success": stats(pref_s),
        "prefix_vpmi_failure": stats(pref_f),
        "continuation_vpmi_success": stats(cont_s),
        "continuation_vpmi_failure": stats(cont_f),
        "transition_delta_cont_minus_pref_success": stats(td_s),
        "transition_delta_cont_minus_pref_failure": stats(td_f),
        "transition_delta_auc_success_gt_failure": auc_pos_gt_neg(td_s, td_f),
        "transition_delta_ks_success_vs_failure": ks_stat(td_s, td_f),
        "transitions_csv": os.path.join(out_dir, "diagB_prefix_to_cont_transitions.csv"),
    }


def run_diag_c(rows: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    # Group split by sample id to avoid leakage.
    tr = [r for r in rows if stable_group_split(str(r["id"]), mod=5) != 0]
    te = [r for r in rows if stable_group_split(str(r["id"]), mod=5) == 0]
    if len(tr) == 0 or len(te) == 0:
        raise RuntimeError("Insufficient train/test rows for diagnostic C.")

    def pack(rr: List[Dict[str, Any]], with_vpmi: bool) -> Tuple[np.ndarray, np.ndarray]:
        y = np.array([float(r["is_success"]) for r in rr], dtype=np.float64)
        lq = np.array([float(r["logit_q"]) for r in rr], dtype=np.float64)
        pos = np.array([float(r["pos_norm"]) for r in rr], dtype=np.float64)
        cont = np.array([float(r["is_continuation"]) for r in rr], dtype=np.float64)
        cols = [np.ones_like(lq), zscore(lq), pos, cont]
        if with_vpmi:
            vp = np.array([float(r["vpmi_logit"]) for r in rr], dtype=np.float64)
            cols.append(zscore(vp))
        X = np.stack(cols, axis=1)
        return X, y

    Xtr0, ytr = pack(tr, with_vpmi=False)
    Xte0, yte = pack(te, with_vpmi=False)
    Xtr1, _ = pack(tr, with_vpmi=True)
    Xte1, _ = pack(te, with_vpmi=True)

    w0 = fit_logreg(Xtr0, ytr, lr=0.06, n_iter=1600, l2=1e-4)
    w1 = fit_logreg(Xtr1, ytr, lr=0.06, n_iter=1600, l2=1e-4)
    p0 = predict_logreg(Xte0, w0)
    p1 = predict_logreg(Xte1, w1)

    p0_pos = [float(p0[i]) for i in range(len(p0)) if int(yte[i]) == 1]
    p0_neg = [float(p0[i]) for i in range(len(p0)) if int(yte[i]) == 0]
    p1_pos = [float(p1[i]) for i in range(len(p1)) if int(yte[i]) == 1]
    p1_neg = [float(p1[i]) for i in range(len(p1)) if int(yte[i]) == 0]

    auc0 = auc_pos_gt_neg(p0_pos, p0_neg)
    auc1 = auc_pos_gt_neg(p1_pos, p1_neg)
    ks0 = ks_stat(p0_pos, p0_neg)
    ks1 = ks_stat(p1_pos, p1_neg)

    coef_rows = []
    for i, w in enumerate(w0.tolist()):
        coef_rows.append({"model": "controls_only", "coef_idx": i, "coef": float(w)})
    for i, w in enumerate(w1.tolist()):
        coef_rows.append({"model": "controls_plus_vpmi", "coef_idx": i, "coef": float(w)})
    write_csv(os.path.join(out_dir, "diagC_logreg_coefficients.csv"), coef_rows)

    return {
        "n_train_tokens": int(len(tr)),
        "n_test_tokens": int(len(te)),
        "controls": ["bias", "z_logit_q", "pos_norm", "is_continuation"],
        "model_controls_only_auc": auc0,
        "model_controls_plus_vpmi_auc": auc1,
        "delta_auc_add_vpmi": (
            None if auc0 is None or auc1 is None else float(auc1 - auc0)
        ),
        "model_controls_only_ks": ks0,
        "model_controls_plus_vpmi_ks": ks1,
        "delta_ks_add_vpmi": (
            None if ks0 is None or ks1 is None else float(ks1 - ks0)
        ),
        "coefficients_csv": os.path.join(out_dir, "diagC_logreg_coefficients.csv"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose core V-PMI failure modes (A/B/C).")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows0 = load_token_rows(run_dir)
    rows = attach_position_and_kind(rows0)
    write_csv(os.path.join(out_dir, "token_rows_enriched.csv"), rows)

    a = run_diag_a(rows, out_dir)
    b = run_diag_b(rows, out_dir)
    c = run_diag_c(rows, out_dir)

    summary = {
        "inputs": {
            "run_dir": run_dir,
            "n_tokens_used": int(len(rows)),
            "n_samples": int(len(set(str(r["id"]) for r in rows))),
        },
        "diagA_logitq_abyss": a,
        "diagB_tokenization_curse": b,
        "diagC_incremental_value_after_controls": c,
        "outputs": {
            "token_rows_enriched_csv": os.path.join(out_dir, "token_rows_enriched.csv"),
            "diagA_deciles_csv": os.path.join(out_dir, "diagA_logitq_deciles.csv"),
            "diagB_transitions_csv": os.path.join(out_dir, "diagB_prefix_to_cont_transitions.csv"),
            "diagC_coefs_csv": os.path.join(out_dir, "diagC_logreg_coefficients.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "token_rows_enriched.csv"))
    print("[saved]", os.path.join(out_dir, "diagA_logitq_deciles.csv"))
    print("[saved]", os.path.join(out_dir, "diagB_prefix_to_cont_transitions.csv"))
    print("[saved]", os.path.join(out_dir, "diagC_logreg_coefficients.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
