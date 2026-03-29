#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def to_bool(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def norm_text(x: Any) -> str:
    s = str("" if x is None else x).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return [dict(r) for r in rd]


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


def quantile(vals: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not xs:
        return None
    qq = min(1.0, max(0.0, float(q)))
    if len(xs) == 1:
        return float(xs[0])
    pos = qq * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float((1.0 - w) * xs[lo] + w * xs[hi])


def iqr(vals: Sequence[float]) -> Optional[float]:
    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)
    if q1 is None or q3 is None:
        return None
    return float(q3 - q1)


def stats_of(vals: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "iqr": None, "min": None, "max": None}
    return {
        "n": int(len(xs)),
        "mean": float(sum(xs) / len(xs)),
        "median": quantile(xs, 0.5),
        "iqr": iqr(xs),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


def mean_or_none(vals: Sequence[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def bin_index(v: Optional[float], edges: List[float]) -> Optional[int]:
    if v is None or not math.isfinite(float(v)):
        return None
    x = float(v)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if (x >= lo and x < hi) or (i == len(edges) - 2 and x == hi):
            return i
    return None


def contains_whole(needle: str, hay: str) -> bool:
    if needle == "":
        return False
    pat = rf"(^|\s){re.escape(needle)}(\s|$)"
    return re.search(pat, hay) is not None


def singularize_word(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def map_gender_tokens(s: str) -> str:
    toks = s.split()
    out: List[str] = []
    for t in toks:
        if t in {"man", "male", "boy", "gentleman", "guy"}:
            out.append("male")
        elif t in {"woman", "female", "girl", "lady"}:
            out.append("female")
        else:
            out.append(t)
    return " ".join(out)


def first_polarity(s: str) -> Optional[str]:
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    return None


def is_success_relabel(question: str, answer: str, champ_text: str, champ_short: str) -> bool:
    q = norm_text(question)
    gt = norm_text(answer)
    pt = norm_text(champ_text)
    ps = norm_text(champ_short)

    if gt == "":
        return False

    # 1) strict exact with either short/text clause
    if gt == ps or gt == pt:
        return True

    # 2) yes/no: prioritize first polarity token
    if gt in {"yes", "no"}:
        pol = first_polarity(pt) or first_polarity(ps)
        if pol is not None:
            return pol == gt
        # fallback to word contain when no explicit polarity head
        return contains_whole(gt, pt)

    # 3) normalize gender synonyms
    gt_gender = map_gender_tokens(gt)
    pt_gender = map_gender_tokens(pt)
    ps_gender = map_gender_tokens(ps)
    if gt_gender in {"male", "female"}:
        if contains_whole(gt_gender, pt_gender) or contains_whole(gt_gender, ps_gender):
            return True

    # 4) multiword answer: phrase contain in text/short
    if len(gt.split()) >= 2:
        if contains_whole(gt, pt) or contains_whole(gt, ps):
            return True

    # 5) single-word robust match (singular/plural tolerant)
    if len(gt.split()) == 1:
        g = singularize_word(gt)
        toks_t = [singularize_word(t) for t in pt.split()]
        toks_s = [singularize_word(t) for t in ps.split()]
        if g in toks_t or g in toks_s:
            return True

    # 6) directional/color/object fallback: whole-word containment in full text
    if contains_whole(gt, pt):
        return True

    # 7) lightweight QA-context fallback for "what color" / "which side"
    if "what color" in q or "which color" in q:
        return contains_whole(gt, pt)
    if "which side" in q or "left or right" in q:
        return contains_whole(gt, pt)

    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Relabel ARTrap outputs without rerunning model")
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--tau_gap", type=float, default=0.65)
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir) if str(args.out_dir).strip() else in_dir + "_relabel"
    os.makedirs(out_dir, exist_ok=True)

    per_sample_path = os.path.join(in_dir, "per_sample.csv")
    per_candidate_path = os.path.join(in_dir, "per_candidate.csv")
    scatter_path = os.path.join(in_dir, "scatter_pq_vs_delta_safe.csv")
    summary_path = os.path.join(in_dir, "summary.json")

    per_sample = read_csv(per_sample_path)
    per_candidate = read_csv(per_candidate_path) if os.path.isfile(per_candidate_path) else []
    scatter = read_csv(scatter_path) if os.path.isfile(scatter_path) else []
    old_summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            old_summary = json.load(f)

    relabel_by_id: Dict[str, str] = {}
    valid_rows: List[Dict[str, Any]] = []
    n_changed = 0
    for r in per_sample:
        rid = str(r.get("id", ""))
        err = str(r.get("error", "")).strip()
        if err:
            relabel_by_id[rid] = str(r.get("label", ""))
            continue
        ok = is_success_relabel(
            question=str(r.get("question", "")),
            answer=str(r.get("answer", "")),
            champ_text=str(r.get("champ_text", "")),
            champ_short=str(r.get("champ_short_answer", "")),
        )
        old_ok = to_bool(r.get("is_success", "False"))
        if ok != old_ok:
            n_changed += 1
        r["is_success_relabel"] = str(bool(ok))
        r["label_relabel"] = "success" if ok else "failure"
        # overwrite for downstream compatibility
        r["is_success"] = str(bool(ok))
        r["label"] = "success" if ok else "failure"
        relabel_by_id[rid] = r["label"]
        valid_rows.append(r)

    for r in per_candidate:
        rid = str(r.get("id", ""))
        if rid in relabel_by_id:
            r["label_relabel"] = relabel_by_id[rid]
            r["label"] = relabel_by_id[rid]

    for r in scatter:
        rid = str(r.get("id", ""))
        if rid in relabel_by_id:
            r["label_relabel"] = relabel_by_id[rid]
            r["label"] = relabel_by_id[rid]

    valid = [r for r in per_sample if str(r.get("error", "")).strip() == ""]
    success = [r for r in valid if to_bool(r.get("is_success", "False"))]
    failure = [r for r in valid if not to_bool(r.get("is_success", "False"))]

    high_prior = [
        r for r in valid
        if safe_float(r.get("champ_p_q")) is not None and float(r.get("champ_p_q")) >= 0.8
    ]
    fragile = [
        r for r in valid
        if safe_float(r.get("delta_safe")) is not None and float(r.get("delta_safe")) <= float(args.tau_gap)
    ]

    # recompute binning with relabel
    pq_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]
    d_edges = [0.0, 0.25, 0.5, 1.0, 2.0, 1e9]
    bin_rows: List[Dict[str, Any]] = []
    for pi in range(len(pq_edges) - 1):
        for di in range(len(d_edges) - 1):
            cell: List[Dict[str, Any]] = []
            for r in valid:
                p = safe_float(r.get("champ_p_q"))
                d = safe_float(r.get("delta_safe"))
                if p is None or d is None:
                    continue
                if bin_index(p, pq_edges) == pi and bin_index(d, d_edges) == di:
                    cell.append(r)
            if not cell:
                continue
            n_fail = int(sum(1 for r in cell if not to_bool(r.get("is_success", "False"))))
            n_succ = int(len(cell) - n_fail)
            bin_rows.append({
                "pq_bin": f"[{pq_edges[pi]:.2f},{pq_edges[pi+1]:.2f})",
                "delta_bin": f"[{d_edges[di]:.2f},{d_edges[di+1]:.2f})",
                "n": int(len(cell)),
                "n_failure": int(n_fail),
                "n_success": int(n_succ),
                "failure_rate": float(n_fail / len(cell)),
            })

    summary = {
        "inputs": {
            **(old_summary.get("inputs", {}) if isinstance(old_summary, dict) else {}),
            "relabel_script": "relabel_artrap_outputs.py",
            "relabel_tau_gap": float(args.tau_gap),
        },
        "counts": {
            "n_total": int(len(per_sample)),
            "n_valid": int(len(valid)),
            "n_success": int(len(success)),
            "n_failure": int(len(failure)),
            "accuracy": mean_or_none([1.0 if to_bool(r.get("is_success", "False")) else 0.0 for r in valid]),
            "n_changed_vs_original_label": int(n_changed),
        },
        "proof1_artrap": {
            "champ_s_full_success": stats_of([safe_float(r.get("champ_s_full")) for r in success]),
            "champ_s_full_failure": stats_of([safe_float(r.get("champ_s_full")) for r in failure]),
            "champ_s_format_img_success": stats_of([safe_float(r.get("champ_s_format_img")) for r in success]),
            "champ_s_format_img_failure": stats_of([safe_float(r.get("champ_s_format_img")) for r in failure]),
            "champ_s_core_img_success": stats_of([safe_float(r.get("champ_s_core_img")) for r in success]),
            "champ_s_core_img_failure": stats_of([safe_float(r.get("champ_s_core_img")) for r in failure]),
            "champ_illusion_gap_success": stats_of([safe_float(r.get("illusion_gap_format_minus_core")) for r in success]),
            "champ_illusion_gap_failure": stats_of([safe_float(r.get("illusion_gap_format_minus_core")) for r in failure]),
        },
        "proof2_pairwise_fragility": {
            "champ_p_q_success": stats_of([safe_float(r.get("champ_p_q")) for r in success]),
            "champ_p_q_failure": stats_of([safe_float(r.get("champ_p_q")) for r in failure]),
            "delta_safe_success": stats_of([safe_float(r.get("delta_safe")) for r in success]),
            "delta_safe_failure": stats_of([safe_float(r.get("delta_safe")) for r in failure]),
            "high_prior_region": {
                "threshold": 0.8,
                "n": int(len(high_prior)),
                "n_failure": int(sum(1 for r in high_prior if not to_bool(r.get("is_success", "False")))),
                "failure_rate": mean_or_none([0.0 if to_bool(r.get("is_success", "False")) else 1.0 for r in high_prior]),
            },
            "fragile_region": {
                "tau_gap": float(args.tau_gap),
                "n": int(len(fragile)),
                "n_failure": int(sum(1 for r in fragile if not to_bool(r.get("is_success", "False")))),
                "failure_rate": mean_or_none([0.0 if to_bool(r.get("is_success", "False")) else 1.0 for r in fragile]),
            },
        },
        "outputs": {
            "per_sample_csv": os.path.join(out_dir, "per_sample.csv"),
            "per_candidate_csv": os.path.join(out_dir, "per_candidate.csv"),
            "scatter_csv": os.path.join(out_dir, "scatter_pq_vs_delta_safe.csv"),
            "binning_csv": os.path.join(out_dir, "binning_pq_delta.csv"),
            "summary_json": os.path.join(out_dir, "summary.json"),
        },
    }

    write_csv(os.path.join(out_dir, "per_sample.csv"), per_sample)
    write_csv(os.path.join(out_dir, "per_candidate.csv"), per_candidate)
    write_csv(os.path.join(out_dir, "scatter_pq_vs_delta_safe.csv"), scatter)
    write_csv(os.path.join(out_dir, "binning_pq_delta.csv"), bin_rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved]", os.path.join(out_dir, "per_sample.csv"))
    print("[saved]", os.path.join(out_dir, "per_candidate.csv"))
    print("[saved]", os.path.join(out_dir, "scatter_pq_vs_delta_safe.csv"))
    print("[saved]", os.path.join(out_dir, "binning_pq_delta.csv"))
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
