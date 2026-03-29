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

from eval_selector_tradeoff import Candidate, Sample, load_samples


def parse_float_grid(raw: str, default: Optional[List[Optional[float]]] = None) -> List[Optional[float]]:
    s = str(raw or "").strip()
    if s == "":
        return [None] if default is None else list(default)
    out: List[Optional[float]] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in {"none", "null", "na"}:
            out.append(None)
            continue
        out.append(float(tok))
    if len(out) == 0:
        return [None] if default is None else list(default)
    return out


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


def mean_or_none(xs: Sequence[Optional[float]]) -> Optional[float]:
    ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(ys) == 0:
        return None
    return float(sum(ys) / len(ys))


def top_by(pool: Sequence[Candidate], key_fn, tie_fn) -> Optional[Candidate]:
    best: Optional[Tuple[Tuple[float, float], Candidate]] = None
    for c in pool:
        v = key_fn(c)
        if v is None or not math.isfinite(float(v)):
            continue
        tv = tie_fn(c)
        if tv is None or not math.isfinite(float(tv)):
            tv = -1e18
        key = (float(v), float(tv))
        if best is None or key > best[0]:
            best = (key, c)
    return None if best is None else best[1]


def select_agree_vminpm_wmin(sample: Sample, dfull_thr: Optional[float]) -> Optional[Candidate]:
    top_vminpm = top_by(
        sample.pool,
        key_fn=lambda c: c.vpmi_core_min_prior_masked,
        tie_fn=lambda c: (c.vpmi if c.vpmi is not None else -1e18),
    )
    top_wmin = top_by(
        sample.pool,
        key_fn=lambda c: c.vpmi_word_min,
        tie_fn=lambda c: (c.vpmi if c.vpmi is not None else -1e18),
    )
    if top_vminpm is None or top_wmin is None:
        return None
    if int(top_vminpm.idx) != int(top_wmin.idx):
        return None
    if dfull_thr is None:
        return top_vminpm
    if top_vminpm.s_full is None or sample.champ.s_full is None:
        return None
    if float(top_vminpm.s_full) - float(sample.champ.s_full) <= float(dfull_thr):
        return top_vminpm
    return None


@dataclass(frozen=True)
class Combo:
    cvlt: float
    dfull_thr: Optional[float]
    adv_thr: Optional[float]
    qgap_thr: Optional[float]

    def as_key(self) -> str:
        return (
            f"cvlt:{self.cvlt:.6g}|"
            f"dfull:{'none' if self.dfull_thr is None else format(self.dfull_thr, '.6g')}|"
            f"adv:{'none' if self.adv_thr is None else format(self.adv_thr, '.6g')}|"
            f"qgap:{'none' if self.qgap_thr is None else format(self.qgap_thr, '.6g')}"
        )


def switch_cond(sample: Sample, safe: Candidate, combo: Combo) -> bool:
    if sample.champ.vpmi is None or safe.vpmi is None:
        return False
    if not (float(safe.vpmi) > float(sample.champ.vpmi)):
        return False
    if not (float(sample.champ.vpmi) < float(combo.cvlt)):
        return False
    if combo.adv_thr is not None:
        if float(safe.vpmi) - float(sample.champ.vpmi) < float(combo.adv_thr):
            return False
    if combo.qgap_thr is not None:
        if safe.s_q is None or sample.champ.s_q is None:
            return False
        if float(safe.s_q) - float(sample.champ.s_q) < float(combo.qgap_thr):
            return False
    return True


def eval_on_indices(samples: Sequence[Sample], idxs: Sequence[int], combo: Combo) -> Dict[str, Any]:
    n = int(len(idxs))
    if n == 0:
        return {
            "n": 0,
            "base_acc": None,
            "final_acc": None,
            "delta_acc": None,
            "gain": 0,
            "harm": 0,
            "same": 0,
            "switch_rate": None,
            "precision_gain": None,
        }

    base_correct = 0
    final_correct = 0
    gain = 0
    harm = 0
    same = 0
    n_switch = 0

    for i in idxs:
        s = samples[int(i)]
        pred = bool(s.base_ok)
        if s.base_ok:
            base_correct += 1

        safe = select_agree_vminpm_wmin(s, combo.dfull_thr)
        if safe is not None and switch_cond(s, safe, combo):
            n_switch += 1
            pred = bool(s.safe_ok_by_idx.get(int(safe.idx), False))

        if pred:
            final_correct += 1
        if pred and (not s.base_ok):
            gain += 1
        elif (not pred) and s.base_ok:
            harm += 1
        else:
            same += 1

    base_acc = float(base_correct / n)
    final_acc = float(final_correct / n)
    return {
        "n": int(n),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": float(final_acc - base_acc),
        "gain": int(gain),
        "harm": int(harm),
        "same": int(same),
        "switch_rate": float(n_switch / n),
        "precision_gain": (None if (gain + harm) == 0 else float(gain / (gain + harm))),
    }


def stratified_split(
    samples: Sequence[Sample],
    holdout_ratio: float,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    pos = [i for i, s in enumerate(samples) if bool(s.base_ok)]
    neg = [i for i, s in enumerate(samples) if not bool(s.base_ok)]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_h = int(round(float(len(pos)) * float(holdout_ratio)))
    n_neg_h = int(round(float(len(neg)) * float(holdout_ratio)))
    hold = pos[:n_pos_h] + neg[:n_neg_h]
    tune = pos[n_pos_h:] + neg[n_neg_h:]
    rng.shuffle(hold)
    rng.shuffle(tune)
    return tune, hold


def all_combos(
    cvlts: Sequence[float],
    dfull_grid: Sequence[Optional[float]],
    adv_grid: Sequence[Optional[float]],
    qgap_grid: Sequence[Optional[float]],
) -> List[Combo]:
    out: List[Combo] = []
    for cv in cvlts:
        for df in dfull_grid:
            for ad in adv_grid:
                for qg in qgap_grid:
                    out.append(
                        Combo(
                            cvlt=float(cv),
                            dfull_thr=(None if df is None else float(df)),
                            adv_thr=(None if ad is None else float(ad)),
                            qgap_thr=(None if qg is None else float(qg)),
                        )
                    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Split-based post-gate tuning to reduce test leakage")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="heuristic", choices=["auto", "strict", "heuristic"])
    ap.add_argument("--holdout_ratio", type=float, default=0.3)
    ap.add_argument("--num_splits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cvlt_grid", type=str, default="-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0")
    ap.add_argument("--dfull_grid", type=str, default="-0.12,-0.10,-0.08,-0.06,-0.05,-0.04,-0.03,-0.02,0.0")
    ap.add_argument("--adv_grid", type=str, default="none,0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--qgap_grid", type=str, default="none,-1.5,-1.0,-0.75,-0.5,-0.25,0.0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = load_samples(args.in_dir, eval_mode=str(args.eval_mode))
    n = len(samples)
    if n == 0:
        raise RuntimeError("No valid samples loaded.")

    cvlts = [float(x.strip()) for x in str(args.cvlt_grid).split(",") if x.strip() != ""]
    dfull_grid = parse_float_grid(str(args.dfull_grid))
    adv_grid = parse_float_grid(str(args.adv_grid))
    qgap_grid = parse_float_grid(str(args.qgap_grid))
    combos = all_combos(cvlts, dfull_grid, adv_grid, qgap_grid)

    # Full-set optimistic (leaky) best for reference.
    full_rows: List[Dict[str, Any]] = []
    for cb in combos:
        m = eval_on_indices(samples, list(range(n)), cb)
        full_rows.append(
            {
                "combo": cb.as_key(),
                "cvlt": cb.cvlt,
                "dfull_thr": cb.dfull_thr,
                "adv_thr": cb.adv_thr,
                "qgap_thr": cb.qgap_thr,
                **m,
            }
        )
    full_best = max(
        full_rows,
        key=lambda r: (
            float(r["delta_acc"] if r["delta_acc"] is not None else -1e18),
            -int(r["harm"]),
            int(r["gain"]),
        ),
    )

    split_rows: List[Dict[str, Any]] = []
    chosen_rows: List[Dict[str, Any]] = []

    for k in range(int(args.num_splits)):
        rng = random.Random(int(args.seed) + int(k))
        tune_idx, hold_idx = stratified_split(samples, holdout_ratio=float(args.holdout_ratio), rng=rng)
        if len(tune_idx) == 0 or len(hold_idx) == 0:
            continue

        best_tune: Optional[Tuple[Combo, Dict[str, Any]]] = None
        for cb in combos:
            tune_m = eval_on_indices(samples, tune_idx, cb)
            hold_m = eval_on_indices(samples, hold_idx, cb)
            split_rows.append(
                {
                    "split": int(k),
                    "combo": cb.as_key(),
                    "cvlt": cb.cvlt,
                    "dfull_thr": cb.dfull_thr,
                    "adv_thr": cb.adv_thr,
                    "qgap_thr": cb.qgap_thr,
                    "tune_n": tune_m["n"],
                    "tune_delta_acc": tune_m["delta_acc"],
                    "tune_gain": tune_m["gain"],
                    "tune_harm": tune_m["harm"],
                    "holdout_n": hold_m["n"],
                    "holdout_delta_acc": hold_m["delta_acc"],
                    "holdout_gain": hold_m["gain"],
                    "holdout_harm": hold_m["harm"],
                }
            )

            if best_tune is None:
                best_tune = (cb, tune_m)
            else:
                _, b = best_tune
                key_now = (
                    float(tune_m["delta_acc"] if tune_m["delta_acc"] is not None else -1e18),
                    -int(tune_m["harm"]),
                    int(tune_m["gain"]),
                )
                key_old = (
                    float(b["delta_acc"] if b["delta_acc"] is not None else -1e18),
                    -int(b["harm"]),
                    int(b["gain"]),
                )
                if key_now > key_old:
                    best_tune = (cb, tune_m)

        if best_tune is None:
            continue
        cb_star, tune_star = best_tune
        hold_star = eval_on_indices(samples, hold_idx, cb_star)
        chosen_rows.append(
            {
                "split": int(k),
                "combo": cb_star.as_key(),
                "cvlt": cb_star.cvlt,
                "dfull_thr": cb_star.dfull_thr,
                "adv_thr": cb_star.adv_thr,
                "qgap_thr": cb_star.qgap_thr,
                "tune_delta_acc": tune_star["delta_acc"],
                "tune_gain": tune_star["gain"],
                "tune_harm": tune_star["harm"],
                "holdout_delta_acc": hold_star["delta_acc"],
                "holdout_gain": hold_star["gain"],
                "holdout_harm": hold_star["harm"],
                "holdout_final_acc": hold_star["final_acc"],
            }
        )

    combo_agg: Dict[str, Dict[str, Any]] = {}
    for r in split_rows:
        key = str(r["combo"])
        if key not in combo_agg:
            combo_agg[key] = {
                "combo": key,
                "cvlt": r["cvlt"],
                "dfull_thr": r["dfull_thr"],
                "adv_thr": r["adv_thr"],
                "qgap_thr": r["qgap_thr"],
                "tune_delta_acc_list": [],
                "holdout_delta_acc_list": [],
                "holdout_gain_list": [],
                "holdout_harm_list": [],
                "n_splits": 0,
            }
        combo_agg[key]["n_splits"] += 1
        combo_agg[key]["tune_delta_acc_list"].append(r["tune_delta_acc"])
        combo_agg[key]["holdout_delta_acc_list"].append(r["holdout_delta_acc"])
        combo_agg[key]["holdout_gain_list"].append(r["holdout_gain"])
        combo_agg[key]["holdout_harm_list"].append(r["holdout_harm"])

    agg_rows: List[Dict[str, Any]] = []
    for key, v in combo_agg.items():
        agg_rows.append(
            {
                "combo": key,
                "cvlt": v["cvlt"],
                "dfull_thr": v["dfull_thr"],
                "adv_thr": v["adv_thr"],
                "qgap_thr": v["qgap_thr"],
                "n_splits": int(v["n_splits"]),
                "tune_delta_acc_mean": mean_or_none(v["tune_delta_acc_list"]),
                "holdout_delta_acc_mean": mean_or_none(v["holdout_delta_acc_list"]),
                "holdout_gain_mean": mean_or_none(v["holdout_gain_list"]),
                "holdout_harm_mean": mean_or_none(v["holdout_harm_list"]),
            }
        )
    agg_rows = sorted(
        agg_rows,
        key=lambda r: (
            float(r["holdout_delta_acc_mean"] if r["holdout_delta_acc_mean"] is not None else -1e18),
            -float(r["holdout_harm_mean"] if r["holdout_harm_mean"] is not None else 1e18),
            float(r["holdout_gain_mean"] if r["holdout_gain_mean"] is not None else -1e18),
        ),
        reverse=True,
    )

    chosen_mean_holdout_delta = mean_or_none([r.get("holdout_delta_acc") for r in chosen_rows])
    chosen_mean_holdout_final = mean_or_none([r.get("holdout_final_acc") for r in chosen_rows])
    chosen_mean_holdout_gain = mean_or_none([r.get("holdout_gain") for r in chosen_rows])
    chosen_mean_holdout_harm = mean_or_none([r.get("holdout_harm") for r in chosen_rows])

    summary = {
        "inputs": {
            "in_dir": os.path.abspath(args.in_dir),
            "eval_mode": str(args.eval_mode),
            "n_samples": int(n),
            "holdout_ratio": float(args.holdout_ratio),
            "num_splits": int(args.num_splits),
            "seed": int(args.seed),
            "cvlt_grid": cvlts,
            "dfull_grid": dfull_grid,
            "adv_grid": adv_grid,
            "qgap_grid": qgap_grid,
            "n_combos": int(len(combos)),
        },
        "optimistic_fullset_best": full_best,
        "split_tuning": {
            "n_valid_splits": int(len(chosen_rows)),
            "chosen_mean_holdout_delta_acc": chosen_mean_holdout_delta,
            "chosen_mean_holdout_final_acc": chosen_mean_holdout_final,
            "chosen_mean_holdout_gain": chosen_mean_holdout_gain,
            "chosen_mean_holdout_harm": chosen_mean_holdout_harm,
            "best_combo_by_holdout_mean": (None if len(agg_rows) == 0 else agg_rows[0]),
        },
        "outputs": {
            "fullset_grid_csv": os.path.join(os.path.abspath(args.out_dir), "fullset_grid.csv"),
            "split_combo_rows_csv": os.path.join(os.path.abspath(args.out_dir), "split_combo_rows.csv"),
            "split_chosen_csv": os.path.join(os.path.abspath(args.out_dir), "split_chosen.csv"),
            "combo_agg_csv": os.path.join(os.path.abspath(args.out_dir), "combo_agg_by_holdout_mean.csv"),
            "summary_json": os.path.join(os.path.abspath(args.out_dir), "summary.json"),
        },
    }

    write_csv(os.path.join(args.out_dir, "fullset_grid.csv"), full_rows)
    write_csv(os.path.join(args.out_dir, "split_combo_rows.csv"), split_rows)
    write_csv(os.path.join(args.out_dir, "split_chosen.csv"), chosen_rows)
    write_csv(os.path.join(args.out_dir, "combo_agg_by_holdout_mean.csv"), agg_rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(args.out_dir, "fullset_grid.csv"))
    print("[saved]", os.path.join(args.out_dir, "split_combo_rows.csv"))
    print("[saved]", os.path.join(args.out_dir, "split_chosen.csv"))
    print("[saved]", os.path.join(args.out_dir, "combo_agg_by_holdout_mean.csv"))
    print("[saved]", os.path.join(args.out_dir, "summary.json"))
    print("[best_fullset]", full_best)
    print("[best_holdout_mean]", (None if len(agg_rows) == 0 else agg_rows[0]))


if __name__ == "__main__":
    main()

