#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Set


def read_questions(path: str) -> List[Dict[str, Any]]:
    obj = json.load(open(path, encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        for qid, meta in obj.items():
            if not isinstance(meta, dict):
                continue
            r = dict(meta)
            r["id"] = str(qid)
            rows.append(r)
    elif isinstance(obj, list):
        for i, meta in enumerate(obj):
            if not isinstance(meta, dict):
                continue
            r = dict(meta)
            r.setdefault("id", str(i))
            rows.append(r)
    return rows


def load_ids(path: str) -> Set[str]:
    obj = json.load(open(path, encoding="utf-8"))
    out: Set[str] = set()
    if isinstance(obj, dict):
        for k in obj.keys():
            out.add(str(k))
        for v in obj.values():
            if isinstance(v, dict) and "id" in v:
                out.add(str(v.get("id")))
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict):
                if "id" in v:
                    out.add(str(v.get("id")))
            else:
                out.add(str(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter questions json by ID set")
    ap.add_argument("--questions_json", type=str, required=True)
    ap.add_argument("--ids_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    rows = read_questions(os.path.abspath(args.questions_json))
    keep_ids = load_ids(os.path.abspath(args.ids_json))
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = str(r.get("id", ""))
        if sid == "" or sid not in keep_ids:
            continue
        rr = dict(r)
        rr.pop("id", None)
        out[sid] = rr

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print("[saved]", out_path)
    print("[meta] n_kept=", len(out), "n_input=", len(rows), "n_ids=", len(keep_ids))


if __name__ == "__main__":
    main()

