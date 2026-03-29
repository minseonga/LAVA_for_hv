#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

YESNO_Q_PREFIXES = (
    "is ",
    "are ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
    "has ",
    "have ",
    "had ",
    "was ",
    "were ",
)


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


def as_bool(x: Any) -> bool:
    s = str("" if x is None else x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def mean_or_none(vals: Sequence[float]) -> Optional[float]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if len(xs) == 0:
        return None
    return float(sum(xs) / len(xs))


def norm_text(x: Any) -> str:
    s = str("" if x is None else x).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def is_yesno_question(question: str) -> bool:
    q = norm_text(question)
    return any(q.startswith(p) for p in YESNO_Q_PREFIXES)


def first_clause(text: str) -> str:
    s = str(text or "").strip()
    s = re.split(r"[\n\.!?;]", s)[0].strip()
    return s


def is_success_heuristic(question: str, answer: str, pred_text: str, pred_short: str) -> bool:
    q = norm_text(question)
    gt = norm_text(answer)
    pt = norm_text(pred_text)
    ps = norm_text(pred_short)
    if gt == "":
        return False

    if gt == ps or gt == pt:
        return True

    if gt in {"yes", "no"}:
        pol = first_polarity(pt) or first_polarity(ps)
        if pol is not None:
            return pol == gt
        return contains_whole(gt, pt)

    gt_gender = map_gender_tokens(gt)
    pt_gender = map_gender_tokens(pt)
    ps_gender = map_gender_tokens(ps)
    if gt_gender in {"male", "female"}:
        if contains_whole(gt_gender, pt_gender) or contains_whole(gt_gender, ps_gender):
            return True

    if len(gt.split()) >= 2:
        if contains_whole(gt, pt) or contains_whole(gt, ps):
            return True

    if len(gt.split()) == 1:
        g = singularize_word(gt)
        toks_t = [singularize_word(t) for t in pt.split()]
        toks_s = [singularize_word(t) for t in ps.split()]
        if g in toks_t or g in toks_s:
            return True

    if contains_whole(gt, pt):
        return True

    if "what color" in q or "which color" in q:
        return contains_whole(gt, pt)
    if "which side" in q or "left or right" in q:
        return contains_whole(gt, pt)
    return False


def is_success_strict(answer: str, pred_text: str, pred_short: str) -> bool:
    gt = norm_text(answer)
    if gt == "":
        return False
    pred = norm_text(first_clause(pred_short if str(pred_short).strip() != "" else pred_text))
    return bool(pred == gt)


@dataclass
class Champ:
    idx: int
    text: str
    short: str
    s_full: Optional[float]
    is_champion: bool


def build_champ_map(per_candidate: List[Dict[str, Any]]) -> Dict[str, Champ]:
    by_sid: Dict[str, List[Champ]] = {}
    for r in per_candidate:
        sid = str(r.get("id", ""))
        if sid == "":
            continue
        c = Champ(
            idx=int(float(r.get("cand_idx", 0))),
            text=str(r.get("text", "")),
            short=str(r.get("short_answer", "")),
            s_full=safe_float(r.get("s_full")),
            is_champion=as_bool(r.get("is_champion", "")),
        )
        by_sid.setdefault(sid, []).append(c)

    out: Dict[str, Champ] = {}
    for sid, cands in by_sid.items():
        marked = [c for c in cands if bool(c.is_champion)]
        if len(marked) > 0:
            out[sid] = marked[0]
            continue
        pool = [c for c in cands if c.s_full is not None]
        if len(pool) > 0:
            out[sid] = max(pool, key=lambda x: float(x.s_full))
            continue
        out[sid] = cands[0]
    return out


def pick_eval_mode(per_sample: List[Dict[str, Any]], eval_mode: str) -> str:
    mode = str(eval_mode).strip().lower()
    if mode in {"heuristic", "strict"}:
        return mode
    for r in per_sample:
        if str(r.get("is_success_heuristic", "")).strip() != "":
            return "heuristic"
        if str(r.get("is_success_strict", "")).strip() != "":
            return "strict"
    return "heuristic"


def base_ok_from_columns(r: Dict[str, Any], mode: str) -> Optional[bool]:
    if mode == "strict" and str(r.get("is_success_strict", "")).strip() != "":
        return bool(as_bool(r.get("is_success_strict", "")))
    if mode == "heuristic" and str(r.get("is_success_heuristic", "")).strip() != "":
        return bool(as_bool(r.get("is_success_heuristic", "")))
    if str(r.get("is_success", "")).strip() != "":
        return bool(as_bool(r.get("is_success", "")))
    return None


def evaluate_raw_baseline(in_dir: str, eval_mode: str, force_recompute: bool) -> Dict[str, Any]:
    in_dir_abs = os.path.abspath(in_dir)
    per_sample = read_csv(os.path.join(in_dir_abs, "per_sample.csv"))
    per_candidate = read_csv(os.path.join(in_dir_abs, "per_candidate.csv"))
    mode = pick_eval_mode(per_sample, eval_mode)
    champ_by_sid = build_champ_map(per_candidate)

    rows: List[Dict[str, Any]] = []
    n_total = len(per_sample)
    n_valid = 0
    n_scored = 0
    n_from_columns = 0
    n_from_recompute = 0
    n_col_recompute_disagree = 0
    col_recompute_overlap = 0
    base_vec: List[float] = []

    for r in per_sample:
        sid = str(r.get("id", ""))
        err = str(r.get("error", "")).strip()
        valid = (err == "")
        if valid:
            n_valid += 1

        q = str(r.get("question", ""))
        a = str(r.get("answer", ""))
        champ = champ_by_sid.get(sid)

        col_ok = base_ok_from_columns(r, mode)
        rec_ok: Optional[bool] = None
        if champ is not None and valid:
            if mode == "strict":
                rec_ok = bool(is_success_strict(a, champ.text, champ.short))
            else:
                rec_ok = bool(is_success_heuristic(q, a, champ.text, champ.short))

        if (col_ok is not None) and (rec_ok is not None):
            col_recompute_overlap += 1
            if bool(col_ok) != bool(rec_ok):
                n_col_recompute_disagree += 1

        base_ok: Optional[bool]
        source = "none"
        if (not force_recompute) and (col_ok is not None):
            base_ok = bool(col_ok)
            source = "column"
            n_from_columns += 1
        else:
            base_ok = rec_ok
            if rec_ok is not None:
                source = "recompute"
                n_from_recompute += 1

        scored = (base_ok is not None) and valid
        if scored:
            n_scored += 1
            base_vec.append(1.0 if bool(base_ok) else 0.0)

        rows.append({
            "id": sid,
            "valid": bool(valid),
            "error": err,
            "eval_mode": mode,
            "base_ok": (None if base_ok is None else bool(base_ok)),
            "source": source,
            "col_ok": (None if col_ok is None else bool(col_ok)),
            "recompute_ok": (None if rec_ok is None else bool(rec_ok)),
            "champ_idx": (None if champ is None else int(champ.idx)),
            "champ_s_full": (None if champ is None else champ.s_full),
        })

    base_acc_valid_scored = mean_or_none(base_vec)
    base_acc_total = None if (n_total <= 0 or base_acc_valid_scored is None) else float(sum(base_vec) / n_total)

    return {
        "input": {
            "in_dir": in_dir_abs,
            "eval_mode": mode,
            "force_recompute": bool(force_recompute),
        },
        "counts": {
            "n_total_per_sample": int(n_total),
            "n_valid_rows": int(n_valid),
            "n_scored": int(n_scored),
            "n_from_columns": int(n_from_columns),
            "n_from_recompute": int(n_from_recompute),
            "column_recompute_overlap": int(col_recompute_overlap),
            "column_recompute_disagree": int(n_col_recompute_disagree),
            "column_recompute_disagree_rate": (
                None if col_recompute_overlap == 0 else float(n_col_recompute_disagree / col_recompute_overlap)
            ),
        },
        "metrics": {
            "base_acc_valid_scored": base_acc_valid_scored,
            "base_acc_total_over_per_sample": base_acc_total,
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate raw baseline (no switching) from an in_dir")
    ap.add_argument("--in_dir", type=str, nargs="+", required=True, help="One or more experiment dirs containing per_sample.csv and per_candidate.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "heuristic", "strict"])
    ap.add_argument("--force_recompute", action="store_true", help="Ignore is_success columns and recompute from champion text")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for in_dir in args.in_dir:
        res = evaluate_raw_baseline(in_dir=in_dir, eval_mode=args.eval_mode, force_recompute=bool(args.force_recompute))
        tag = os.path.basename(os.path.abspath(in_dir.rstrip("/")))
        rows_path = os.path.join(out_dir, f"{tag}_raw_baseline_rows.csv")
        json_path = os.path.join(out_dir, f"{tag}_raw_baseline_summary.json")
        write_csv(rows_path, res["rows"])
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in res.items() if k != "rows"}, f, indent=2, ensure_ascii=False)
        summaries.append({
            "tag": tag,
            "in_dir": res["input"]["in_dir"],
            "eval_mode": res["input"]["eval_mode"],
            "force_recompute": bool(res["input"]["force_recompute"]),
            "n_total_per_sample": int(res["counts"]["n_total_per_sample"]),
            "n_valid_rows": int(res["counts"]["n_valid_rows"]),
            "n_scored": int(res["counts"]["n_scored"]),
            "base_acc_valid_scored": res["metrics"]["base_acc_valid_scored"],
            "base_acc_total_over_per_sample": res["metrics"]["base_acc_total_over_per_sample"],
            "column_recompute_overlap": int(res["counts"]["column_recompute_overlap"]),
            "column_recompute_disagree_rate": res["counts"]["column_recompute_disagree_rate"],
            "summary_json": json_path,
            "rows_csv": rows_path,
        })
        print("[saved]", rows_path)
        print("[saved]", json_path)

    table_path = os.path.join(out_dir, "raw_baseline_table.csv")
    write_csv(table_path, summaries)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": summaries, "out_dir": out_dir}, f, indent=2, ensure_ascii=False)
    print("[saved]", table_path)
    print("[saved]", os.path.join(out_dir, "summary.json"))


if __name__ == "__main__":
    main()
