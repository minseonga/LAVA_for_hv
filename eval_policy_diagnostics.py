#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval_selector_tradeoff import (
    Candidate,
    Sample,
    load_samples,
    select_candidate,
    switch_cond,
)


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


def mean(xs: Sequence[float]) -> float:
    return float(sum(float(x) for x in xs) / max(1, len(xs)))


def quantile(xs: Sequence[float], q: float) -> Optional[float]:
    ys = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if len(ys) == 0:
        return None
    qq = min(1.0, max(0.0, float(q)))
    if len(ys) == 1:
        return float(ys[0])
    pos = qq * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ys[lo])
    w = float(pos - lo)
    return float((1.0 - w) * ys[lo] + w * ys[hi])


def bootstrap_ci(vals: Sequence[float], n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if len(xs) == 0:
        return None, None
    rng = random.Random(int(seed))
    n = int(len(xs))
    boots: List[float] = []
    for _ in range(int(max(1, n_boot))):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boots.append(float(sum(sample) / n))
    lo = quantile(boots, float(alpha) / 2.0)
    hi = quantile(boots, 1.0 - float(alpha) / 2.0)
    return lo, hi


def mcnemar_exact_p(n01: int, n10: int) -> float:
    # exact two-sided binomial test for discordant pairs
    b = int(n01 + n10)
    if b <= 0:
        return 1.0
    k = int(min(n01, n10))
    # p = 2 * sum_{i=0..k} C(b,i)/2^b
    p_half = 0.0
    denom = float(2 ** b)
    for i in range(k + 1):
        p_half += float(math.comb(b, i)) / denom
    p = float(min(1.0, 2.0 * p_half))
    return p


@dataclass
class RunCfg:
    name: str
    trigger: str
    policy: str


def parse_configs(raw: str) -> List[RunCfg]:
    out: List[RunCfg] = []
    toks = [x.strip() for x in str(raw or "").split(";") if x.strip() != ""]
    for t in toks:
        parts = [p.strip() for p in t.split("|")]
        if len(parts) != 3:
            raise ValueError(f"Invalid config format: {t}. Expected name|trigger|policy")
        out.append(RunCfg(name=str(parts[0]), trigger=str(parts[1]), policy=str(parts[2])))
    if len(out) == 0:
        raise ValueError("No valid --configs provided.")
    return out


def evaluate_one(samples: Sequence[Sample], cfg: RunCfg) -> Dict[str, Any]:
    per_sample: List[Dict[str, Any]] = []
    n = int(len(samples))
    for s in samples:
        safe = select_candidate(cfg.policy, s)
        switched = False
        safe_ok: Optional[bool] = None
        trigger_ok: Optional[bool] = None
        if safe is not None:
            safe_ok = bool(s.safe_ok_by_idx.get(int(safe.idx), False))
            trigger_ok = bool(switch_cond(cfg.trigger, s, safe))
            switched = bool(trigger_ok)
        final_ok = bool(s.base_ok)
        if switched and safe_ok is not None:
            final_ok = bool(safe_ok)

        has_correct_pool = bool(any(bool(v) for v in s.safe_ok_by_idx.values()))
        has_correct_any = bool(s.base_ok or has_correct_pool)

        # Trigger-constrained oracle:
        # if baseline wrong, can we switch to any correct candidate that passes trigger?
        can_trigger_to_correct = False
        if not bool(s.base_ok):
            for c in s.pool:
                if not bool(s.safe_ok_by_idx.get(int(c.idx), False)):
                    continue
                if bool(switch_cond(cfg.trigger, s, c)):
                    can_trigger_to_correct = True
                    break

        # Miss reasons only for recoverable baseline failures.
        miss_reason = ""
        if (not bool(s.base_ok)) and has_correct_pool and (not bool(final_ok)):
            if safe is None:
                miss_reason = "selector_none"
            elif not bool(trigger_ok):
                if bool(safe_ok):
                    miss_reason = "trigger_miss_on_correct_safe"
                elif bool(can_trigger_to_correct):
                    miss_reason = "selector_wrong_and_trigger_miss"
                else:
                    miss_reason = "trigger_block_all_correct"
            else:
                if bool(safe_ok):
                    miss_reason = "unexpected"
                else:
                    miss_reason = "selector_wrong_after_switch"

        # G/M/U over baseline-wrong subset
        gmu = ""
        if not bool(s.base_ok):
            if has_correct_pool and bool(final_ok):
                gmu = "G"
            elif has_correct_pool and (not bool(final_ok)):
                gmu = "M"
            elif (not has_correct_pool) and (not bool(final_ok)):
                gmu = "U"
            else:
                gmu = "other"

        per_sample.append(
            {
                "id": s.sid,
                "config": cfg.name,
                "trigger": cfg.trigger,
                "policy": cfg.policy,
                "base_ok": bool(s.base_ok),
                "final_ok": bool(final_ok),
                "switched": bool(switched),
                "safe_idx": (None if safe is None else int(safe.idx)),
                "safe_ok": safe_ok,
                "trigger_ok": trigger_ok,
                "has_correct_pool": bool(has_correct_pool),
                "has_correct_any": bool(has_correct_any),
                "can_trigger_to_correct": bool(can_trigger_to_correct),
                "gmu": gmu,
                "miss_reason": miss_reason,
            }
        )

    base_vec = [1.0 if bool(r["base_ok"]) else 0.0 for r in per_sample]
    final_vec = [1.0 if bool(r["final_ok"]) else 0.0 for r in per_sample]
    delta_vec = [float(f - b) for b, f in zip(base_vec, final_vec)]
    gain = int(sum(1 for r in per_sample if (not bool(r["base_ok"])) and bool(r["final_ok"])))
    harm = int(sum(1 for r in per_sample if bool(r["base_ok"]) and (not bool(r["final_ok"]))))
    switch_n = int(sum(1 for r in per_sample if bool(r["switched"])))
    base_acc = mean(base_vec)
    final_acc = mean(final_vec)
    delta_acc = float(final_acc - base_acc)
    precision_gain = (None if (gain + harm) == 0 else float(gain / (gain + harm)))

    n10 = int(gain)  # baseline wrong -> final correct
    n01 = int(harm)  # baseline correct -> final wrong
    p_mcnemar = float(mcnemar_exact_p(n01=n01, n10=n10))

    # Oracle ceilings and bottlenecks
    base_wrong_rows = [r for r in per_sample if not bool(r["base_ok"])]
    recoverable_rows = [r for r in base_wrong_rows if bool(r["has_correct_pool"])]
    unrecoverable_rows = [r for r in base_wrong_rows if not bool(r["has_correct_pool"])]
    miss_rows = [r for r in recoverable_rows if not bool(r["final_ok"])]

    # pool-oracle: baseline wrong + has correct pool => always recover
    oracle_pool_final_acc = float(
        sum(
            1
            for r in per_sample
            if bool(r["base_ok"]) or (not bool(r["base_ok"]) and bool(r["has_correct_pool"]))
        )
        / max(1, n)
    )
    oracle_pool_delta = float(oracle_pool_final_acc - base_acc)

    # trigger-constrained oracle: recover only if trigger can fire for at least one correct candidate
    oracle_trigger_final_acc = float(
        sum(
            1
            for r in per_sample
            if bool(r["base_ok"]) or (not bool(r["base_ok"]) and bool(r["can_trigger_to_correct"]))
        )
        / max(1, n)
    )
    oracle_trigger_delta = float(oracle_trigger_final_acc - base_acc)

    return {
        "cfg": cfg,
        "per_sample": per_sample,
        "metrics": {
            "n": int(n),
            "base_acc": float(base_acc),
            "final_acc": float(final_acc),
            "delta_acc": float(delta_acc),
            "gain": int(gain),
            "harm": int(harm),
            "switch_rate": float(switch_n / max(1, n)),
            "precision_gain": precision_gain,
            "mcnemar_n10_gain": int(n10),
            "mcnemar_n01_harm": int(n01),
            "mcnemar_p_exact": float(p_mcnemar),
        },
        "series": {
            "base_vec": base_vec,
            "final_vec": final_vec,
            "delta_vec": delta_vec,
        },
        "gmu": {
            "base_wrong_n": int(len(base_wrong_rows)),
            "G": int(sum(1 for r in base_wrong_rows if str(r["gmu"]) == "G")),
            "M": int(sum(1 for r in base_wrong_rows if str(r["gmu"]) == "M")),
            "U": int(sum(1 for r in base_wrong_rows if str(r["gmu"]) == "U")),
            "recoverable_n": int(len(recoverable_rows)),
            "unrecoverable_n": int(len(unrecoverable_rows)),
        },
        "miss_reason_counts": {
            "selector_none": int(sum(1 for r in miss_rows if str(r["miss_reason"]) == "selector_none")),
            "trigger_miss_on_correct_safe": int(
                sum(1 for r in miss_rows if str(r["miss_reason"]) == "trigger_miss_on_correct_safe")
            ),
            "selector_wrong_and_trigger_miss": int(
                sum(1 for r in miss_rows if str(r["miss_reason"]) == "selector_wrong_and_trigger_miss")
            ),
            "trigger_block_all_correct": int(
                sum(1 for r in miss_rows if str(r["miss_reason"]) == "trigger_block_all_correct")
            ),
            "selector_wrong_after_switch": int(
                sum(1 for r in miss_rows if str(r["miss_reason"]) == "selector_wrong_after_switch")
            ),
        },
        "oracle": {
            "oracle_pool_final_acc": float(oracle_pool_final_acc),
            "oracle_pool_delta": float(oracle_pool_delta),
            "oracle_trigger_final_acc": float(oracle_trigger_final_acc),
            "oracle_trigger_delta": float(oracle_trigger_delta),
            "gap_actual_to_trigger_oracle": float(oracle_trigger_delta - delta_acc),
            "gap_trigger_to_pool_oracle": float(oracle_pool_delta - oracle_trigger_delta),
            "gap_actual_to_pool_oracle": float(oracle_pool_delta - delta_acc),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Policy diagnostics: CI, significance, G/M/U, oracle bottlenecks")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["auto", "strict", "heuristic"])
    ap.add_argument(
        "--configs",
        type=str,
        required=True,
        help="Semicolon-separated: name|trigger|policy;name2|trigger|policy2",
    )
    ap.add_argument("--ref_name", type=str, default="")
    ap.add_argument("--bootstrap_n", type=int, default=3000)
    ap.add_argument("--bootstrap_seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfgs = parse_configs(args.configs)
    samples = load_samples(args.in_dir, eval_mode=str(args.eval_mode))
    if len(samples) == 0:
        raise RuntimeError("No valid samples loaded.")

    evals: List[Dict[str, Any]] = [evaluate_one(samples, c) for c in cfgs]

    # CI + core metrics
    metric_rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []
    gmu_rows: List[Dict[str, Any]] = []
    miss_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []

    for e in evals:
        cfg: RunCfg = e["cfg"]
        m = dict(e["metrics"])
        s = e["series"]
        lo_d, hi_d = bootstrap_ci(s["delta_vec"], n_boot=int(args.bootstrap_n), seed=int(args.bootstrap_seed))
        lo_f, hi_f = bootstrap_ci(s["final_vec"], n_boot=int(args.bootstrap_n), seed=int(args.bootstrap_seed) + 7)
        ci_rows.append(
            {
                "name": cfg.name,
                "trigger": cfg.trigger,
                "policy": cfg.policy,
                "delta_acc": m["delta_acc"],
                "delta_acc_ci95_lo": lo_d,
                "delta_acc_ci95_hi": hi_d,
                "final_acc": m["final_acc"],
                "final_acc_ci95_lo": lo_f,
                "final_acc_ci95_hi": hi_f,
                "mcnemar_n10_gain": m["mcnemar_n10_gain"],
                "mcnemar_n01_harm": m["mcnemar_n01_harm"],
                "mcnemar_p_exact": m["mcnemar_p_exact"],
            }
        )
        metric_rows.append(
            {
                "name": cfg.name,
                "trigger": cfg.trigger,
                "policy": cfg.policy,
                **m,
            }
        )

        g = dict(e["gmu"])
        bw = max(1, int(g["base_wrong_n"]))
        gmu_rows.append(
            {
                "name": cfg.name,
                "trigger": cfg.trigger,
                "policy": cfg.policy,
                **g,
                "G_rate_in_base_wrong": float(g["G"] / bw),
                "M_rate_in_base_wrong": float(g["M"] / bw),
                "U_rate_in_base_wrong": float(g["U"] / bw),
            }
        )
        miss_rows.append(
            {
                "name": cfg.name,
                "trigger": cfg.trigger,
                "policy": cfg.policy,
                **dict(e["miss_reason_counts"]),
            }
        )
        oracle_rows.append(
            {
                "name": cfg.name,
                "trigger": cfg.trigger,
                "policy": cfg.policy,
                **dict(e["oracle"]),
            }
        )
        per_sample_rows.extend(list(e["per_sample"]))

    # Comparison vs ref (optional)
    compare_rows: List[Dict[str, Any]] = []
    ref_name = str(args.ref_name).strip()
    if ref_name != "":
        ref = next((r for r in metric_rows if str(r["name"]) == ref_name), None)
        ref_g = next((r for r in gmu_rows if str(r["name"]) == ref_name), None)
        ref_miss = next((r for r in miss_rows if str(r["name"]) == ref_name), None)
        final_by_name: Dict[str, List[float]] = {
            str(e["cfg"].name): [float(x) for x in e["series"]["final_vec"]]
            for e in evals
        }
        if ref is not None and ref_g is not None and ref_miss is not None:
            for r in metric_rows:
                gg = next((x for x in gmu_rows if str(x["name"]) == str(r["name"])), None)
                mm = next((x for x in miss_rows if str(x["name"]) == str(r["name"])), None)
                if gg is None or mm is None:
                    continue
                # Pairwise significance vs reference (policy-to-policy)
                n10_pair = 0
                n01_pair = 0
                xa = final_by_name.get(str(r["name"]), [])
                xb = final_by_name.get(str(ref_name), [])
                m_pair = int(min(len(xa), len(xb)))
                for i in range(m_pair):
                    a_ok = bool(float(xa[i]) > 0.5)
                    b_ok = bool(float(xb[i]) > 0.5)
                    if a_ok and (not b_ok):
                        n10_pair += 1
                    elif (not a_ok) and b_ok:
                        n01_pair += 1
                p_pair = float(mcnemar_exact_p(n01=n01_pair, n10=n10_pair))
                compare_rows.append(
                    {
                        "name": r["name"],
                        "trigger": r["trigger"],
                        "policy": r["policy"],
                        "d_final_acc_vs_ref": float(r["final_acc"]) - float(ref["final_acc"]),
                        "d_delta_acc_vs_ref": float(r["delta_acc"]) - float(ref["delta_acc"]),
                        "d_gain_vs_ref": int(r["gain"]) - int(ref["gain"]),
                        "d_harm_vs_ref": int(r["harm"]) - int(ref["harm"]),
                        "harm_suppressed_vs_ref": int(ref["harm"]) - int(r["harm"]),
                        "d_G_vs_ref": int(gg["G"]) - int(ref_g["G"]),
                        "d_M_vs_ref": int(gg["M"]) - int(ref_g["M"]),
                        "d_U_vs_ref": int(gg["U"]) - int(ref_g["U"]),
                        "d_trigger_miss_on_correct_safe_vs_ref": int(mm["trigger_miss_on_correct_safe"]) - int(ref_miss["trigger_miss_on_correct_safe"]),
                        "d_selector_wrong_after_switch_vs_ref": int(mm["selector_wrong_after_switch"]) - int(ref_miss["selector_wrong_after_switch"]),
                        "pair_mcnemar_n10_vs_ref": int(n10_pair),
                        "pair_mcnemar_n01_vs_ref": int(n01_pair),
                        "pair_mcnemar_p_exact_vs_ref": float(p_pair),
                    }
                )

    # Sort outputs
    metric_rows = sorted(metric_rows, key=lambda x: float(x["delta_acc"]), reverse=True)
    ci_rows = sorted(ci_rows, key=lambda x: float(x["delta_acc"]), reverse=True)
    gmu_rows = sorted(gmu_rows, key=lambda x: float(x["G_rate_in_base_wrong"]), reverse=True)
    oracle_rows = sorted(oracle_rows, key=lambda x: float(x["oracle_pool_delta"] - x["oracle_trigger_delta"]), reverse=True)
    compare_rows = sorted(compare_rows, key=lambda x: float(x["d_delta_acc_vs_ref"]), reverse=True)

    summary = {
        "inputs": {
            "in_dir": os.path.abspath(args.in_dir),
            "eval_mode": str(args.eval_mode),
            "n_samples": int(len(samples)),
            "configs": [{"name": c.name, "trigger": c.trigger, "policy": c.policy} for c in cfgs],
            "ref_name": ref_name,
            "bootstrap_n": int(args.bootstrap_n),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "best_by_delta": (None if len(metric_rows) == 0 else metric_rows[0]),
        "outputs": {
            "metrics_csv": os.path.join(os.path.abspath(args.out_dir), "metrics.csv"),
            "ci_significance_csv": os.path.join(os.path.abspath(args.out_dir), "ci_significance.csv"),
            "gmu_csv": os.path.join(os.path.abspath(args.out_dir), "gmu_table.csv"),
            "miss_reason_csv": os.path.join(os.path.abspath(args.out_dir), "miss_reason_table.csv"),
            "oracle_csv": os.path.join(os.path.abspath(args.out_dir), "oracle_bottleneck.csv"),
            "compare_vs_ref_csv": os.path.join(os.path.abspath(args.out_dir), "compare_vs_ref.csv"),
            "per_sample_csv": os.path.join(os.path.abspath(args.out_dir), "per_sample_decisions.csv"),
            "summary_json": os.path.join(os.path.abspath(args.out_dir), "summary.json"),
        },
    }

    write_csv(os.path.join(args.out_dir, "metrics.csv"), metric_rows)
    write_csv(os.path.join(args.out_dir, "ci_significance.csv"), ci_rows)
    write_csv(os.path.join(args.out_dir, "gmu_table.csv"), gmu_rows)
    write_csv(os.path.join(args.out_dir, "miss_reason_table.csv"), miss_rows)
    write_csv(os.path.join(args.out_dir, "oracle_bottleneck.csv"), oracle_rows)
    write_csv(os.path.join(args.out_dir, "compare_vs_ref.csv"), compare_rows)
    write_csv(os.path.join(args.out_dir, "per_sample_decisions.csv"), per_sample_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(args.out_dir, "metrics.csv"))
    print("[saved]", os.path.join(args.out_dir, "ci_significance.csv"))
    print("[saved]", os.path.join(args.out_dir, "gmu_table.csv"))
    print("[saved]", os.path.join(args.out_dir, "miss_reason_table.csv"))
    print("[saved]", os.path.join(args.out_dir, "oracle_bottleneck.csv"))
    print("[saved]", os.path.join(args.out_dir, "compare_vs_ref.csv"))
    print("[saved]", os.path.join(args.out_dir, "per_sample_decisions.csv"))
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
