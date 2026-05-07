#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def maybe_int(value: Any) -> int:
    try:
        return int(round(float(value or 0)))
    except Exception:
        return 0


def pct(value: Any) -> str:
    x = maybe_float(value)
    return "" if x is None else f"{100.0 * x:.2f}"


def sec(value: Any) -> str:
    x = maybe_float(value)
    return "" if x is None else f"{x:.4f}"


def ratio(value: Any) -> str:
    x = maybe_float(value)
    return "" if x is None else f"{x:.3f}"


def find_summaries(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("summary.json"))
        else:
            candidates = []
        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            try:
                data = read_json(candidate)
            except Exception:
                continue
            if str(data.get("mode", "")) != "llava15_fixed_c3_lazy_deploy":
                continue
            seen.add(key)
            out.append(candidate)
    return out


def label_from_path(path: Path, root: Optional[Path]) -> str:
    try:
        if root is not None:
            return str(path.parent.resolve().relative_to(root.resolve()))
    except Exception:
        pass
    return path.parent.name


def row_from_summary(path: Path, root: Optional[Path]) -> Dict[str, Any]:
    data = read_json(path)
    inputs = data.get("inputs") or {}
    counts = data.get("counts") or {}
    evaluation = data.get("evaluation") or {}
    timing = data.get("timing") or {}
    n_rows = maybe_int(counts.get("n_rows"))
    n_baseline_triggered = maybe_int(counts.get("n_baseline_triggered"))
    n_baseline_skipped = maybe_int(counts.get("n_baseline_skipped"))
    n_route_baseline = maybe_int(counts.get("n_route_baseline"))
    denom = float(max(1, n_rows))
    return {
        "run": label_from_path(path, root),
        "target": inputs.get("target", ""),
        "dataset": inputs.get("dataset", ""),
        "method": inputs.get("method", ""),
        "deployment_order": inputs.get("deployment_order", ""),
        "n_rows": n_rows,
        "n_completed": maybe_int(counts.get("n_completed", n_rows)),
        "n_errors": maybe_int(counts.get("n_errors")),
        "n_baseline_triggered": n_baseline_triggered,
        "n_baseline_skipped": n_baseline_skipped,
        "n_route_baseline": n_route_baseline,
        "n_replay_score_computed": maybe_int(counts.get("n_replay_score_computed")),
        "n_replay_score_skipped": maybe_int(counts.get("n_replay_score_skipped")),
        "n_answer_changed": maybe_int(counts.get("n_answer_changed")),
        "baseline_trigger_rate": maybe_float(timing.get("baseline_trigger_rate", n_baseline_triggered / denom)),
        "baseline_skip_rate": maybe_float(timing.get("baseline_skip_rate", n_baseline_skipped / denom)),
        "replay_score_compute_rate": maybe_float(timing.get("replay_score_compute_rate")),
        "replay_score_skip_rate": maybe_float(timing.get("replay_score_skip_rate")),
        "answer_changed_rate": maybe_float(timing.get("answer_changed_rate")),
        "route_baseline_rate": maybe_float(timing.get("route_baseline_rate", n_route_baseline / denom)),
        "method_acc": maybe_float(evaluation.get("method_acc")),
        "final_acc": maybe_float(evaluation.get("final_acc")),
        "delta_vs_method": maybe_float(evaluation.get("delta_vs_method")),
        "estimated_method_only_mean_sec_per_sample": maybe_float(timing.get("estimated_method_only_mean_sec_per_sample")),
        "estimated_score_only_no_baseline_mean_sec_per_sample": maybe_float(
            timing.get("estimated_score_only_no_baseline_mean_sec_per_sample")
        ),
        "mean_total_sec_per_sample": maybe_float(timing.get("mean_total_sec_per_sample")),
        "mean_method_generated_sec": maybe_float(timing.get("mean_method_generated_sec")),
        "mean_replay_score_sec": maybe_float(timing.get("mean_replay_score_sec")),
        "mean_baseline_generated_sec": maybe_float(timing.get("mean_baseline_generated_sec")),
        "estimated_always_baseline_mean_sec_per_sample": maybe_float(
            timing.get("estimated_always_baseline_mean_sec_per_sample")
        ),
        "estimated_speedup_vs_always_baseline": maybe_float(timing.get("estimated_speedup_vs_always_baseline")),
        "estimated_latency_savings_pct": maybe_float(timing.get("estimated_latency_savings_pct")),
        "estimated_always_replay_mean_sec_per_sample": maybe_float(
            timing.get("estimated_always_replay_mean_sec_per_sample")
        ),
        "estimated_speedup_vs_always_replay": maybe_float(timing.get("estimated_speedup_vs_always_replay")),
        "estimated_replay_skip_savings_pct": maybe_float(timing.get("estimated_replay_skip_savings_pct")),
        "summary_json": str(path.resolve()),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Run | Order | Target | Dataset | n | Trigger % | Baseline skip % | Replay % | Changed % | Route baseline % | Method Acc | Final Acc | Delta | Method-only sec | Score-only sec | Lazy sec | Always baseline sec | Always replay sec | Speedup | Saved latency % |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        speedup = row.get("estimated_speedup_vs_always_replay") or row.get("estimated_speedup_vs_always_baseline")
        saved = row.get("estimated_replay_skip_savings_pct") or row.get("estimated_latency_savings_pct")
        lines.append(
            f"| {row['run']} | {row['deployment_order']} | {row['target']} | {row['dataset']} | {row['n_rows']} | "
            f"{pct(row['baseline_trigger_rate'])} | {pct(row['baseline_skip_rate'])} | "
            f"{pct(row['replay_score_compute_rate'])} | {pct(row['answer_changed_rate'])} | "
            f"{pct(row['route_baseline_rate'])} | {pct(row['method_acc'])} | "
            f"{pct(row['final_acc'])} | {pct(row['delta_vs_method'])} | "
            f"{sec(row['estimated_method_only_mean_sec_per_sample'])} | "
            f"{sec(row['estimated_score_only_no_baseline_mean_sec_per_sample'])} | "
            f"{sec(row['mean_total_sec_per_sample'])} | "
            f"{sec(row['estimated_always_baseline_mean_sec_per_sample'])} | "
            f"{sec(row['estimated_always_replay_mean_sec_per_sample'])} | "
            f"{ratio(speedup)} | "
            f"{ratio(saved)} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Summarize LLaVA-1.5 fixed-C3 lazy deployment latency ablations."
    )
    ap.add_argument("--root", action="append", default=[], help="Directory to search for summary.json files.")
    ap.add_argument("--summary_json", action="append", default=[], help="Explicit lazy deploy summary.json path.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in list(args.root) + list(args.summary_json)]
    summaries = find_summaries(input_paths)
    if not summaries:
        raise SystemExit("no llava15_fixed_c3_lazy_deploy summary.json files found")
    common_root = Path(args.root[0]).resolve() if args.root else None
    rows = [row_from_summary(path, common_root) for path in summaries]
    rows.sort(key=lambda r: (str(r.get("target", "")), str(r.get("dataset", "")), str(r.get("run", ""))))
    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows)
    print("[saved]", out_csv)
    md = markdown(rows)
    print(md)
    if str(args.out_md or "").strip():
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md + "\n", encoding="utf-8")
        print("[saved]", out_md)


if __name__ == "__main__":
    main()
