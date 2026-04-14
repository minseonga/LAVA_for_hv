#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def canonical_list(values: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(values, list):
        return out
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            item = item[1]
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def canonical_set(values: Any) -> set[str]:
    return set(canonical_list(values))


def join_items(values: Iterable[str]) -> str:
    return " | ".join(sorted(str(v) for v in values if str(v).strip()))


def ordered_unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def load_sentences(path: str) -> Dict[str, Dict[str, Any]]:
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(payload.get("sentences", [])):
        image_id = str(row.get("image_id", row.get("question_id", idx)))
        out[image_id] = row
    return out


def sample_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    generated = canonical_list(row.get("mscoco_generated_words", []))
    gt = canonical_set(row.get("mscoco_gt_words", []))
    gen_unique = set(generated)
    supported_unique = gen_unique & gt
    hallucinated_unique = gen_unique - gt
    hallucinated_inst = [obj for obj in generated if obj not in gt]

    precision_unique = safe_div(float(len(supported_unique)), float(len(gen_unique)))
    recall = safe_div(float(len(supported_unique)), float(len(gt)))
    return {
        "image_id": str(row.get("image_id", row.get("question_id", ""))),
        "caption": str(row.get("caption", "")),
        "generated_list": generated,
        "generated_unique": gen_unique,
        "generated_unique_ordered": ordered_unique(generated),
        "gt_objects": gt,
        "supported_unique": supported_unique,
        "hallucinated_unique": hallucinated_unique,
        "hallucinated_inst": hallucinated_inst,
        "n_generated_inst": int(len(generated)),
        "n_generated_unique": int(len(gen_unique)),
        "n_duplicate_object_mentions": int(len(generated) - len(gen_unique)),
        "n_gt_objects": int(len(gt)),
        "n_supported_unique": int(len(supported_unique)),
        "n_hallucinated_unique": int(len(hallucinated_unique)),
        "n_hallucinated_inst": int(len(hallucinated_inst)),
        "chair_s": int(bool(hallucinated_inst)),
        "ci_inst": safe_div(float(len(hallucinated_inst)), float(len(generated))),
        "ci_unique": safe_div(float(len(hallucinated_unique)), float(len(gen_unique))),
        "precision_unique": precision_unique,
        "recall": recall,
        "f1_unique": f1(precision_unique, recall),
    }


def aggregate(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(items)
    gen_inst = sum(int(row["n_generated_inst"]) for row in items)
    gen_unique = sum(int(row["n_generated_unique"]) for row in items)
    dup = sum(int(row["n_duplicate_object_mentions"]) for row in items)
    gt = sum(int(row["n_gt_objects"]) for row in items)
    supp = sum(int(row["n_supported_unique"]) for row in items)
    hall_unique = sum(int(row["n_hallucinated_unique"]) for row in items)
    hall_inst = sum(int(row["n_hallucinated_inst"]) for row in items)
    precision_unique = safe_div(float(supp), float(gen_unique))
    recall = safe_div(float(supp), float(gt))
    return {
        "n": float(n),
        "chair_s": safe_div(float(sum(int(row["chair_s"]) for row in items)), float(n)),
        "ci_inst": safe_div(float(hall_inst), float(gen_inst)),
        "ci_unique": safe_div(float(hall_unique), float(gen_unique)),
        "precision_unique": precision_unique,
        "recall": recall,
        "f1_unique": f1(precision_unique, recall),
        "avg_generated_unique": safe_div(float(gen_unique), float(n)),
        "avg_supported_unique": safe_div(float(supp), float(n)),
        "avg_hallucinated_unique": safe_div(float(hall_unique), float(n)),
        "duplicate_object_mention_rate": safe_div(float(dup), float(gen_inst)),
    }


def bool_int(value: bool) -> int:
    return int(bool(value))


def classify_failure(base: Dict[str, Any], intervention: Dict[str, Any]) -> str:
    base_only_supported = base["supported_unique"] - intervention["supported_unique"]
    int_only_supported = intervention["supported_unique"] - base["supported_unique"]
    base_only_hall = base["hallucinated_unique"] - intervention["hallucinated_unique"]
    int_only_hall = intervention["hallucinated_unique"] - base["hallucinated_unique"]

    if base_only_supported and not int_only_hall:
        return "safe_coverage_recovery"
    if base_only_supported and int_only_hall:
        return "coverage_recovery_with_intervention_only_hall"
    if base_only_hall and not int_only_hall:
        return "baseline_hallucination_cost_only"
    if int_only_hall:
        return "intervention_only_hall_cleanup"
    if float(base["f1_unique"]) > float(intervention["f1_unique"]):
        return "balance_or_precision_gain"
    return "other"


def make_row(
    image_id: str,
    base: Dict[str, Any],
    intervention: Dict[str, Any],
    *,
    ci_eps: float,
    chair_s_eps: float,
    f1_eps: float,
    recall_eps: float,
) -> Dict[str, Any]:
    base_only_supported = base["supported_unique"] - intervention["supported_unique"]
    int_only_supported = intervention["supported_unique"] - base["supported_unique"]
    base_only_hall = base["hallucinated_unique"] - intervention["hallucinated_unique"]
    int_only_hall = intervention["hallucinated_unique"] - base["hallucinated_unique"]

    delta_recall = float(base["recall"]) - float(intervention["recall"])
    delta_f1 = float(base["f1_unique"]) - float(intervention["f1_unique"])
    delta_ci_unique = float(base["ci_unique"]) - float(intervention["ci_unique"])
    delta_chair_s = float(base["chair_s"]) - float(intervention["chair_s"])

    ci_safe = float(base["ci_unique"]) <= float(intervention["ci_unique"]) + float(ci_eps)
    chair_s_safe = float(base["chair_s"]) <= float(intervention["chair_s"]) + float(chair_s_eps)
    f1_gain = delta_f1 > float(f1_eps)
    recall_gain = delta_recall > float(recall_eps)
    f1_nondecrease = delta_f1 >= -float(f1_eps)

    out: Dict[str, Any] = {
        "image_id": image_id,
        "failure_type": classify_failure(base, intervention),
        "oracle_max_f1_unique": bool_int(f1_gain),
        "oracle_f1_gain_ci_unique_noworse": bool_int(f1_gain and ci_safe),
        "oracle_f1_gain_ci_unique_chairs_noworse": bool_int(f1_gain and ci_safe and chair_s_safe),
        "oracle_recall_gain_f1_nondecrease_ci_unique_noworse": bool_int(recall_gain and f1_nondecrease and ci_safe),
        "oracle_recall_gain_f1_nondecrease_ci_unique_chairs_noworse": bool_int(
            recall_gain and f1_nondecrease and ci_safe and chair_s_safe
        ),
        "base_caption": base["caption"],
        "int_caption": intervention["caption"],
        "gt_objects": join_items(base["gt_objects"]),
        "base_generated_unique": join_items(base["generated_unique"]),
        "int_generated_unique": join_items(intervention["generated_unique"]),
        "base_supported_unique": join_items(base["supported_unique"]),
        "int_supported_unique": join_items(intervention["supported_unique"]),
        "base_hallucinated_unique": join_items(base["hallucinated_unique"]),
        "int_hallucinated_unique": join_items(intervention["hallucinated_unique"]),
        "base_only_supported_unique": join_items(base_only_supported),
        "int_only_supported_unique": join_items(int_only_supported),
        "base_only_hallucinated_unique": join_items(base_only_hall),
        "int_only_hallucinated_unique": join_items(int_only_hall),
        "n_base_only_supported_unique": len(base_only_supported),
        "n_int_only_supported_unique": len(int_only_supported),
        "n_base_only_hallucinated_unique": len(base_only_hall),
        "n_int_only_hallucinated_unique": len(int_only_hall),
        "delta_recall_base_minus_int": delta_recall,
        "delta_f1_unique_base_minus_int": delta_f1,
        "delta_ci_unique_base_minus_int": delta_ci_unique,
        "delta_chair_s_base_minus_int": delta_chair_s,
    }
    for prefix, metrics in (("base", base), ("int", intervention)):
        for key in (
            "chair_s",
            "ci_inst",
            "ci_unique",
            "precision_unique",
            "recall",
            "f1_unique",
            "n_generated_inst",
            "n_generated_unique",
            "n_duplicate_object_mentions",
            "n_gt_objects",
            "n_supported_unique",
            "n_hallucinated_unique",
            "n_hallucinated_inst",
        ):
            out[f"{prefix}_{key}"] = metrics[key]
    return out


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze CHAIR unique-Ci-safe fallback oracle between baseline and intervention.")
    ap.add_argument("--baseline_chair_json", required=True)
    ap.add_argument("--intervention_chair_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ci_unique_eps", type=float, default=0.0)
    ap.add_argument("--chair_s_eps", type=float, default=0.0)
    ap.add_argument("--f1_eps", type=float, default=0.0)
    ap.add_argument("--recall_eps", type=float, default=0.0)
    ap.add_argument("--main_oracle_col", default="oracle_recall_gain_f1_nondecrease_ci_unique_noworse")
    ap.add_argument("--top_k_examples", type=int, default=40)
    args = ap.parse_args()

    base_rows = load_sentences(args.baseline_chair_json)
    int_rows = load_sentences(args.intervention_chair_json)
    ids = [image_id for image_id in base_rows.keys() if image_id in int_rows]

    rows: List[Dict[str, Any]] = []
    base_metrics: List[Dict[str, Any]] = []
    int_metrics: List[Dict[str, Any]] = []
    for image_id in ids:
        base = sample_metrics(base_rows[image_id])
        intervention = sample_metrics(int_rows[image_id])
        base_metrics.append(base)
        int_metrics.append(intervention)
        rows.append(
            make_row(
                image_id,
                base,
                intervention,
                ci_eps=float(args.ci_unique_eps),
                chair_s_eps=float(args.chair_s_eps),
                f1_eps=float(args.f1_eps),
                recall_eps=float(args.recall_eps),
            )
        )

    oracle_cols = [
        "oracle_max_f1_unique",
        "oracle_f1_gain_ci_unique_noworse",
        "oracle_f1_gain_ci_unique_chairs_noworse",
        "oracle_recall_gain_f1_nondecrease_ci_unique_noworse",
        "oracle_recall_gain_f1_nondecrease_ci_unique_chairs_noworse",
    ]
    summary: Dict[str, Any] = {
        "inputs": {
            "baseline_chair_json": os.path.abspath(args.baseline_chair_json),
            "intervention_chair_json": os.path.abspath(args.intervention_chair_json),
            "ci_unique_eps": float(args.ci_unique_eps),
            "chair_s_eps": float(args.chair_s_eps),
            "f1_eps": float(args.f1_eps),
            "recall_eps": float(args.recall_eps),
            "main_oracle_col": args.main_oracle_col,
        },
        "counts": {"n_rows": len(rows)},
        "baseline": aggregate(base_metrics),
        "intervention": aggregate(int_metrics),
        "oracles": {},
    }

    for col in oracle_cols:
        selected_ids = {row["image_id"] for row in rows if int(row[col]) == 1}
        routed: List[Dict[str, Any]] = []
        for image_id, base, intervention in zip(ids, base_metrics, int_metrics):
            routed.append(base if image_id in selected_ids else intervention)
        selected_rows = [row for row in rows if int(row[col]) == 1]
        failure_counts = collections.Counter(row["failure_type"] for row in selected_rows)
        base_only_supported_counter: collections.Counter[str] = collections.Counter()
        int_only_hall_counter: collections.Counter[str] = collections.Counter()
        for row in selected_rows:
            base_only_supported_counter.update([x for x in str(row["base_only_supported_unique"]).split(" | ") if x])
            int_only_hall_counter.update([x for x in str(row["int_only_hallucinated_unique"]).split(" | ") if x])
        summary["oracles"][col] = {
            "n_selected": len(selected_ids),
            "fallback_rate": safe_div(float(len(selected_ids)), float(len(rows))),
            "metrics": aggregate(routed),
            "failure_counts": dict(failure_counts),
            "top_base_only_supported": base_only_supported_counter.most_common(20),
            "top_int_only_hallucinated": int_only_hall_counter.most_common(20),
        }

    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    rows_csv = os.path.join(args.out_dir, "unique_safe_oracle_rows.csv")
    examples_csv = os.path.join(args.out_dir, f"{args.main_oracle_col}_examples.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")
    findings_md = os.path.join(args.out_dir, "audit_preliminary_findings.md")

    write_csv(rows_csv, rows)
    selected_examples = [row for row in rows if int(row.get(args.main_oracle_col, 0)) == 1]
    selected_examples.sort(
        key=lambda row: (
            float(row.get("delta_f1_unique_base_minus_int", 0.0)),
            float(row.get("delta_recall_base_minus_int", 0.0)),
        ),
        reverse=True,
    )
    write_csv(examples_csv, selected_examples[: int(args.top_k_examples)])
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    main = summary["oracles"].get(args.main_oracle_col, {})
    lines = [
        "# Unique-Ci-Safe Oracle Audit",
        "",
        f"- rows: {len(rows)}",
        f"- main oracle: `{args.main_oracle_col}`",
        f"- selected: {main.get('n_selected', 0)} ({float(main.get('fallback_rate', 0.0)):.3f})",
        "",
        "## Metrics",
        "",
        f"- baseline: {json.dumps(summary['baseline'], ensure_ascii=False)}",
        f"- intervention: {json.dumps(summary['intervention'], ensure_ascii=False)}",
        f"- main_oracle: {json.dumps(main.get('metrics', {}), ensure_ascii=False)}",
        "",
        "## Failure Counts",
        "",
        json.dumps(main.get("failure_counts", {}), ensure_ascii=False, indent=2),
        "",
        "## Top Base-Only Supported Objects",
        "",
        json.dumps(main.get("top_base_only_supported", []), ensure_ascii=False, indent=2),
        "",
        "## Outputs",
        "",
        f"- rows_csv: `{os.path.abspath(rows_csv)}`",
        f"- examples_csv: `{os.path.abspath(examples_csv)}`",
        f"- summary_json: `{os.path.abspath(summary_json)}`",
    ]
    with open(findings_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[saved] {rows_csv}")
    print(f"[saved] {examples_csv}")
    print(f"[saved] {summary_json}")
    print(f"[saved] {findings_md}")


if __name__ == "__main__":
    main()
