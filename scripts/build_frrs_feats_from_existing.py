#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional


def read_jsonl_map(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            sid = str(obj.get("id", obj.get("question_id", ""))).strip()
            if sid == "":
                continue
            out[sid] = obj
    return out


def load_ids(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id", "")).strip()
            if sid:
                out.append(sid)
    return out


def ensure_list(x) -> List[float]:
    if isinstance(x, list):
        return [float(v) for v in x]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FRRS feature jsonl from existing FRGG/RFHAR feature files.")
    ap.add_argument("--frgg_feats_json", type=str, required=True, help="jsonl with id,A,C,E")
    ap.add_argument("--rfhar_feats_json", type=str, default="", help="optional jsonl with id,C,A,D,B")
    ap.add_argument("--ids_csv", type=str, default="", help="optional id list CSV (id column)")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    frgg_map = read_jsonl_map(os.path.abspath(args.frgg_feats_json))
    if len(frgg_map) == 0:
        raise RuntimeError("No rows loaded from --frgg_feats_json")

    rfhar_map: Dict[str, dict] = {}
    if str(args.rfhar_feats_json).strip() != "":
        rfhar_map = read_jsonl_map(os.path.abspath(args.rfhar_feats_json))

    if str(args.ids_csv).strip() != "":
        ids = load_ids(os.path.abspath(args.ids_csv))
    else:
        ids = sorted(frgg_map.keys(), key=lambda x: int(x) if x.isdigit() else x)

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = 0
    n_with_d = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid in ids:
            fr = frgg_map.get(str(sid), None)
            if fr is None:
                continue
            A = ensure_list(fr.get("A", []))
            C = ensure_list(fr.get("C", []))
            E = ensure_list(fr.get("E", []))
            if not (len(A) > 0 and len(A) == len(C) == len(E)):
                continue

            D: Optional[List[float]] = None
            rr = rfhar_map.get(str(sid), None)
            if rr is not None:
                d = ensure_list(rr.get("D", []))
                if len(d) == len(A):
                    D = d
                    n_with_d += 1
            if D is None:
                D = [0.0 for _ in range(len(A))]

            rec = {"id": str(sid), "A": A, "C": C, "E": E, "D": D}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(json.dumps({
        "out_json": out_path,
        "n_rows": int(n),
        "n_with_real_D": int(n_with_d),
        "n_with_zero_D": int(max(0, n - n_with_d)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
