#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def maybe_int(value: object) -> Optional[int]:
    v = maybe_float(value)
    if v is None:
        return None
    return int(round(v))


def mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 1.0
    mu = mean(seq)
    var = sum((x - mu) ** 2 for x in seq) / float(len(seq))
    return float(math.sqrt(max(0.0, var)))


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    n_pos = sum(int(y) for y in labels)
    n_neg = len(labels) - n_pos
    if len(scores) != len(labels) or n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = sum(rank for rank, y in zip(ranks, labels) if int(y) == 1)
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def binary_average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)]
    if not pairs:
        return None
    n_pos = sum(y for _, y in pairs)
    if n_pos == 0:
        return None
    pairs.sort(key=lambda x: x[0], reverse=True)
    tp = 0
    ap = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            tp += 1
            ap += float(tp) / float(rank)
    return float(ap / float(n_pos))


def load_xy(rows: Sequence[Dict[str, str]], feature: str, target: str) -> Tuple[List[float], List[int]]:
    xs: List[float] = []
    ys: List[int] = []
    for row in rows:
        x = maybe_float(row.get(feature))
        y = maybe_int(row.get(target))
        if x is None or y not in {0, 1}:
            continue
        xs.append(float(x))
        ys.append(int(y))
    return xs, ys


def evaluate_direction(xs: Sequence[float], ys: Sequence[int], direction: str) -> Dict[str, Any]:
    oriented = [float(x) if str(direction) == "high" else -float(x) for x in xs]
    auc = binary_auroc(oriented, ys)
    ap = binary_average_precision(oriented, ys)
    return {
        "direction": str(direction),
        "auroc": None if auc is None else float(auc),
        "average_precision": None if ap is None else float(ap),
        "n": int(len(xs)),
        "n_pos": int(sum(int(y) for y in ys)),
        "positive_rate": float(sum(int(y) for y in ys) / float(max(1, len(ys)))),
    }


def best_direction(xs: Sequence[float], ys: Sequence[int]) -> Dict[str, Any]:
    high = evaluate_direction(xs, ys, "high")
    low = evaluate_direction(xs, ys, "low")
    high_auc = float(high["auroc"] if high["auroc"] is not None else -1.0)
    low_auc = float(low["auroc"] if low["auroc"] is not None else -1.0)
    return high if high_auc >= low_auc else low


def zscore_oriented(values: Sequence[float], direction: str) -> List[float]:
    seq = [float(v) for v in values]
    oriented = seq if str(direction) == "high" else [-float(v) for v in seq]
    mu = mean(oriented)
    sd = std(oriented)
    if not math.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return [(float(v) - mu) / sd for v in oriented]


def build_composite(
    rows: Sequence[Dict[str, str]],
    feature_specs: Sequence[Tuple[str, str]],
    target: str,
) -> Optional[Dict[str, Any]]:
    if not feature_specs:
        return None
    labels: List[int] = []
    matrix: List[List[float]] = []
    for row in rows:
        vals: List[float] = []
        y = maybe_int(row.get(target))
        if y not in {0, 1}:
            continue
        ok = True
        for feature, _direction in feature_specs:
            x = maybe_float(row.get(feature))
            if x is None:
                ok = False
                break
            vals.append(float(x))
        if not ok:
            continue
        labels.append(int(y))
        matrix.append(vals)
    if not matrix:
        return None

    cols = list(zip(*matrix))
    zcols: List[List[float]] = []
    for (feature, direction), col in zip(feature_specs, cols):
        _ = feature
        zcols.append(zscore_oriented(col, direction))
    scores = [mean(zvals) for zvals in zip(*zcols)]
    auc = binary_auroc(scores, labels)
    ap = binary_average_precision(scores, labels)
    return {
        "n": int(len(scores)),
        "n_pos": int(sum(labels)),
        "auroc": None if auc is None else float(auc),
        "average_precision": None if ap is None else float(ap),
    }


def parse_feature_map(pairs: Sequence[str]) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for raw in pairs:
        parts = [x.strip() for x in str(raw).split(":")]
        if len(parts) != 3 or any(not p for p in parts):
            raise ValueError(f"Invalid --feature_map entry: {raw}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Test whether discriminative claim-brittleness features transfer to generative mention features.")
    ap.add_argument("--disc_table_csv", type=str, required=True)
    ap.add_argument("--gen_table_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--feature_map",
        type=str,
        nargs="+",
        default=[
            "claim_lp_min:probe_lp_min_real:probe_mention_lp_min_real",
            "claim_target_gap_min:probe_target_gap_min_real:probe_mention_target_gap_min_real",
            "claim_entropy_peak:probe_entropy_max_real:probe_mention_entropy_max_real",
        ],
    )
    ap.add_argument("--target", type=str, default="harm")
    args = ap.parse_args()

    disc_rows = read_csv_rows(os.path.abspath(args.disc_table_csv))
    gen_rows = read_csv_rows(os.path.abspath(args.gen_table_csv))
    feature_map = parse_feature_map(args.feature_map)

    transfer_rows: List[Dict[str, Any]] = []
    ranked_specs: List[Tuple[str, str, str, float]] = []

    for common_name, disc_feature, gen_feature in feature_map:
        disc_xs, disc_ys = load_xy(disc_rows, disc_feature, args.target)
        gen_xs, gen_ys = load_xy(gen_rows, gen_feature, args.target)
        disc_best = best_direction(disc_xs, disc_ys)
        transfer_dir = str(disc_best["direction"])
        gen_transfer = evaluate_direction(gen_xs, gen_ys, transfer_dir)
        gen_best = best_direction(gen_xs, gen_ys)
        transfer_rows.append(
            {
                "common_feature": common_name,
                "disc_feature": disc_feature,
                "gen_feature": gen_feature,
                "disc_best_direction": transfer_dir,
                "disc_auroc": disc_best["auroc"],
                "disc_average_precision": disc_best["average_precision"],
                "gen_transfer_auroc": gen_transfer["auroc"],
                "gen_transfer_average_precision": gen_transfer["average_precision"],
                "gen_best_direction": gen_best["direction"],
                "gen_best_auroc": gen_best["auroc"],
                "gen_best_average_precision": gen_best["average_precision"],
            }
        )
        if disc_best["auroc"] is not None:
            ranked_specs.append((common_name, gen_feature, transfer_dir, float(disc_best["auroc"])))

    transfer_rows.sort(key=lambda r: (-float(r["disc_auroc"] or 0.0), str(r["common_feature"])))
    ranked_specs.sort(key=lambda x: -float(x[3]))

    composite_rows: List[Dict[str, Any]] = []
    for k in range(1, len(ranked_specs) + 1):
        topk = ranked_specs[:k]
        # Build generative composite from generative mapped features in discriminative-derived order.
        gen_comp = build_composite(
            gen_rows,
            feature_specs=[(gen_feature, direction) for (_name, gen_feature, direction, _auc) in topk],
            target=args.target,
        )
        composite_rows.append(
            {
                "k": int(k),
                "features": ",".join(name for name, _feature, _direction, _auc in topk),
                "gen_features": ",".join(feature for _name, feature, _direction, _auc in topk),
                "directions": ",".join(direction for _name, _feature, direction, _auc in topk),
                "disc_order_auc_mean": mean(float(item[3]) for item in topk),
                "gen_transfer_composite_auroc": None if gen_comp is None else gen_comp["auroc"],
                "gen_transfer_composite_average_precision": None if gen_comp is None else gen_comp["average_precision"],
            }
        )

    best_single = transfer_rows[0] if transfer_rows else None
    best_composite = None
    if composite_rows:
        best_composite = sorted(
            composite_rows,
            key=lambda r: (
                -float(r["gen_transfer_composite_auroc"] or 0.0),
                -float(r["gen_transfer_composite_average_precision"] or 0.0),
                int(r["k"]),
            ),
        )[0]

    transfer_csv = os.path.join(args.out_dir, "feature_transfer.csv")
    composite_csv = os.path.join(args.out_dir, "composite_transfer.csv")
    write_csv(transfer_csv, transfer_rows)
    write_csv(composite_csv, composite_rows)

    summary = {
        "inputs": {
            "disc_table_csv": os.path.abspath(args.disc_table_csv),
            "gen_table_csv": os.path.abspath(args.gen_table_csv),
            "target": str(args.target),
            "feature_map": [
                {
                    "common_feature": common_name,
                    "disc_feature": disc_feature,
                    "gen_feature": gen_feature,
                }
                for common_name, disc_feature, gen_feature in feature_map
            ],
        },
        "best_single_transfer": best_single,
        "best_composite_transfer": best_composite,
        "feature_transfer_rows": transfer_rows,
        "outputs": {
            "feature_transfer_csv": os.path.abspath(transfer_csv),
            "composite_transfer_csv": os.path.abspath(composite_csv),
        },
    }
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("[saved]", transfer_csv)
    print("[saved]", composite_csv)
    print("[saved]", os.path.join(args.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
