#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


@dataclass
class Gate:
    name: str
    mode: str
    tau_vpmi: Optional[float]
    tau_sfull: Optional[float]


def parse_gates(raw: str) -> List[Gate]:
    txt = str(raw or "").strip()
    if txt == "":
        txt = "gate_and_vpmi_m5.5_and_sfull_m6|and|-5.5|-6"
    rows = [r.strip() for r in txt.split(";") if r.strip() != ""]
    out: List[Gate] = []
    for rr in rows:
        parts = [p.strip() for p in rr.split("|")]
        if len(parts) != 4:
            raise ValueError(f"Invalid gate spec: {rr}")
        name, mode, tv, tf = parts
        mode_l = mode.lower()
        if mode_l not in {"vpmi_only", "and", "or", "never", "always"}:
            raise ValueError(f"Unsupported mode: {mode}")
        tau_v = None if tv.lower() in {"", "none", "na"} else safe_float(tv)
        tau_f = None if tf.lower() in {"", "none", "na"} else safe_float(tf)
        out.append(Gate(name=name, mode=mode_l, tau_vpmi=tau_v, tau_sfull=tau_f))
    return out


def should_expand(gate: Gate, champ_vpmi: Optional[float], champ_sfull: Optional[float]) -> bool:
    if gate.mode == "never":
        return False
    if gate.mode == "always":
        return True
    c_v = False
    c_f = False
    if gate.tau_vpmi is not None and champ_vpmi is not None:
        c_v = bool(float(champ_vpmi) < float(gate.tau_vpmi))
    if gate.tau_sfull is not None and champ_sfull is not None:
        c_f = bool(float(champ_sfull) < float(gate.tau_sfull))
    if gate.mode == "vpmi_only":
        return c_v
    if gate.mode == "and":
        return bool(c_v and c_f)
    if gate.mode == "or":
        return bool(c_v or c_f)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Build gate-pass ID subset from greedy per_sample.csv")
    ap.add_argument("--greedy_dir", type=str, required=True, help="Directory with greedy per_sample.csv")
    ap.add_argument("--gates", type=str, required=True, help="name|mode|tau_vpmi|tau_sfull ; ...")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_debug_csv", type=str, default="")
    args = ap.parse_args()

    p = os.path.join(os.path.abspath(args.greedy_dir), "per_sample.csv")
    if not os.path.isfile(p):
        raise RuntimeError(f"missing per_sample.csv: {p}")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    gates = parse_gates(args.gates)

    kept: List[Dict[str, Any]] = []
    dbg: List[Dict[str, Any]] = []
    n_valid = 0
    for r in rows:
        if str(r.get("error", "")).strip() != "":
            continue
        sid = str(r.get("id", "")).strip()
        if sid == "":
            continue
        n_valid += 1
        champ_sfull = safe_float(r.get("champ_s_full"))
        champ_vpmi = safe_float(r.get("champ_vpmi"))
        if champ_vpmi is None:
            s_core = safe_float(r.get("champ_s_core_img"))
            s_q = safe_float(r.get("champ_s_ans_q"))
            champ_vpmi = (None if s_core is None or s_q is None else float(s_core - s_q))

        expanded = any(should_expand(g, champ_vpmi=champ_vpmi, champ_sfull=champ_sfull) for g in gates)
        if expanded:
            kept.append({"id": sid})
        dbg.append(
            {
                "id": sid,
                "champ_vpmi": champ_vpmi,
                "champ_s_full": champ_sfull,
                "expanded": bool(expanded),
            }
        )

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print("[saved]", out_path)
    print("[meta] n_valid=", int(n_valid), "n_expand=", int(len(kept)), "expand_rate=", (0.0 if n_valid == 0 else float(len(kept) / n_valid)))

    if str(args.out_debug_csv).strip() != "":
        dpath = os.path.abspath(str(args.out_debug_csv).strip())
        os.makedirs(os.path.dirname(dpath), exist_ok=True)
        keys = ["id", "champ_vpmi", "champ_s_full", "expanded"]
        with open(dpath, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for rr in dbg:
                wr.writerow({k: rr.get(k, None) for k in keys})
        print("[saved]", dpath)


if __name__ == "__main__":
    main()

