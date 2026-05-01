#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: object) -> Optional[int]:
    value_f = maybe_float(value)
    if value_f is None:
        return None
    return int(round(value_f))


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(scores) != len(labels):
        return None
    n_pos = sum(int(y) for y in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if int(pairs[k][1]) == 1:
                rank_sum += avg_rank
        i = j
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def is_candidate(row: Mapping[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    base = str(row.get("baseline_label", "")).strip().lower()
    intervention = str(row.get("intervention_label", "")).strip().lower()
    if mode == "changed_answer":
        return base in {"yes", "no"} and intervention in {"yes", "no"} and base != intervention
    if mode == "yes_to_no":
        return base == "yes" and intervention == "no"
    raise ValueError(f"Unsupported candidate_filter={mode!r}")


def orient_and_best_net(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, Any]:
    auc_high = binary_auroc(scores, labels)
    auc_low = binary_auroc([-float(x) for x in scores], labels)
    if auc_high is None or auc_low is None:
        raise ValueError("Cannot compute AUROC.")
    if float(auc_high) >= float(auc_low):
        direction = "high"
        oriented = [float(x) for x in scores]
        auroc = float(auc_high)
    else:
        direction = "low"
        oriented = [-float(x) for x in scores]
        auroc = float(auc_low)

    best: Optional[Dict[str, Any]] = None
    for tau in sorted(set(oriented)):
        selected_harm = 0
        selected_help = 0
        selected = 0
        for score, label in zip(oriented, labels):
            if float(score) >= float(tau):
                selected += 1
                if int(label) == 1:
                    selected_harm += 1
                else:
                    selected_help += 1
        item = {
            "tau": float(tau),
            "selected_count": int(selected),
            "selected_harm": int(selected_harm),
            "selected_help": int(selected_help),
            "net": int(selected_harm - selected_help),
            "selected_harm_precision": float(selected_harm / max(1, selected)),
            "selected_harm_recall": float(selected_harm / max(1, sum(labels))),
        }
        if best is None or (
            int(item["net"]),
            int(item["selected_harm"]),
            -int(item["selected_help"]),
        ) > (
            int(best["net"]),
            int(best["selected_harm"]),
            -int(best["selected_help"]),
        ):
            best = item

    assert best is not None
    return {"auroc": auroc, "direction": direction, **best}


def trajectory_features(layer_values: Sequence[Tuple[int, float]]) -> Dict[str, float]:
    vals = sorted((int(layer), float(value)) for layer, value in layer_values)
    layers = [x[0] for x in vals]
    margins = [x[1] for x in vals]
    max_layer = max(layers) if layers else 1
    early = [value for layer, value in vals if 1 <= layer <= max(1, max_layer // 3)] or margins
    mid = [value for layer, value in vals if max_layer // 3 < layer <= (2 * max_layer) // 3] or margins
    late = [value for layer, value in vals if layer > (2 * max_layer) // 3] or margins
    min_value = min(margins)
    max_value = max(margins)
    argmin_layer = layers[margins.index(min_value)]
    final_value = margins[-1]
    return {
        "traj_min": min_value,
        "traj_max": max_value,
        "traj_range": max_value - min_value,
        "traj_mean": mean(margins),
        "traj_support_rate": float(sum(1 for x in margins if x > 0.0) / max(1, len(margins))),
        "traj_argmin_frac": float(argmin_layer / max(1, max_layer)),
        "traj_final_minus_min": final_value - min_value,
        "traj_final_minus_early_mean": final_value - mean(early),
        "traj_late_mean_minus_early_mean": mean(late) - mean(early),
        "traj_min_early": min(early),
        "traj_min_mid": min(mid),
        "traj_min_late": min(late),
        "traj_max_early": max(early),
        "traj_max_mid": max(mid),
        "traj_max_late": max(late),
    }


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze scalar shape features from layer-wise decision-margin trajectories.")
    ap.add_argument("--trajectory_long_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--candidate_filter", default="changed_answer", choices=["all", "changed_answer", "yes_to_no"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.abspath(args.trajectory_long_csv), "r", encoding="utf-8")))
    by_id: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    meta: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not is_candidate(row, str(args.candidate_filter)):
            continue
        sid = str(row.get("id", "")).strip()
        layer = maybe_int(row.get("layer_index"))
        margin = maybe_float(row.get("candidate_minus_alt"))
        harm = maybe_int(row.get("harm"))
        help_ = maybe_int(row.get("help"))
        if not sid or layer is None or margin is None or harm not in {0, 1} or help_ not in {0, 1}:
            continue
        if int(harm) == 0 and int(help_) == 0:
            continue
        by_id[sid].append((int(layer), float(margin)))
        meta[sid] = {"id": sid, "harm": int(harm), "help": int(help_)}

    feature_rows: List[Dict[str, Any]] = []
    for sid, vals in by_id.items():
        feature_rows.append({**meta[sid], **trajectory_features(vals)})

    if not feature_rows:
        raise RuntimeError("No rows available after filtering.")
    feature_names = [key for key in feature_rows[0] if key.startswith("traj_")]
    labels = [int(row["harm"]) for row in feature_rows]
    summary_rows: List[Dict[str, Any]] = []
    for feature in feature_names:
        scores = [float(row[feature]) for row in feature_rows]
        metrics = orient_and_best_net(scores, labels)
        summary_rows.append({"feature": feature, **metrics})

    summary_rows.sort(key=lambda r: (int(r["net"]), float(r["auroc"])), reverse=True)
    out_dir = os.path.abspath(args.out_dir)
    write_csv(os.path.join(out_dir, "trajectory_shape_rows.csv"), feature_rows)
    write_csv(os.path.join(out_dir, "trajectory_shape_summary.csv"), summary_rows)
    write_json(
        os.path.join(out_dir, "summary.json"),
        {
            "inputs": {
                "trajectory_long_csv": os.path.abspath(args.trajectory_long_csv),
                "candidate_filter": str(args.candidate_filter),
            },
            "counts": {
                "n": len(feature_rows),
                "n_harm": sum(labels),
                "n_help": len(labels) - sum(labels),
            },
            "best_by_net": summary_rows[0],
            "outputs": {
                "trajectory_shape_rows_csv": os.path.join(out_dir, "trajectory_shape_rows.csv"),
                "trajectory_shape_summary_csv": os.path.join(out_dir, "trajectory_shape_summary.csv"),
            },
        },
    )
    print("[saved]", os.path.join(out_dir, "trajectory_shape_summary.csv"))
    print(json.dumps(summary_rows[0], indent=2))


if __name__ == "__main__":
    main()
