#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATASETS = ("mscoco", "aokvqa", "gqa")

LABELS = {
    "vga_llava15": "VGA / LLaVA-1.5",
    "vga_llava_next": "VGA / LLaVA-NeXT",
    "vga_qwen25_vl_7b": "VGA / Qwen2.5-VL-7B",
    "llava15_vaf": "VAF / LLaVA-1.5",
    "llava15_pai_attn": "PAI-attn / LLaVA-1.5",
    "llava_next_vaf": "VAF / LLaVA-NeXT",
    "llava_next_pai_attn": "PAI-attn / LLaVA-NeXT",
    "qwen25_vaf": "VAF / Qwen2.5-VL-7B",
    "qwen25_pai_attn": "PAI-attn / Qwen2.5-VL-7B",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
            writer.writerow({key: row.get(key, "") for key in cols})


def maybe_float(value: object) -> Optional[float]:
    try:
        text = str(value if value is not None else "").strip()
        if not text:
            return None
        out = float(text)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: object) -> int:
    value_f = maybe_float(value)
    if value_f is None:
        return 0
    return int(round(value_f))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if float(den) else 0.0


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / float(max(1, len(values))))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mu = mean(values)
    var = sum((float(x) - mu) ** 2 for x in values) / float(len(values))
    return float(max(math.sqrt(max(0.0, var)), 1e-6))


def label_text(target: str) -> str:
    return LABELS.get(target, target)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_deployment_summaries(apply_roots: Sequence[Path]) -> List[Tuple[str, str, Path]]:
    jobs: List[Tuple[str, str, Path]] = []
    seen = set()
    for root in apply_roots:
        if not root.exists():
            continue
        for summary in sorted(root.rglob("deployment_summary.json")):
            app_dir = summary.parent
            dataset = app_dir.name
            if dataset not in DATASETS:
                continue
            target = app_dir.parent.name
            key = (target, dataset, str(app_dir.resolve()))
            if key in seen:
                continue
            seen.add(key)
            jobs.append((target, dataset, app_dir))
    return jobs


def apply_summary_paths(app_dir: Path) -> Tuple[Path, Path]:
    summary = read_json(app_dir / "summary.json")
    inputs = summary.get("inputs") or {}
    rows_csv = Path(str(inputs.get("rows_csv", ""))).expanduser()
    policy_json = Path(str(inputs.get("policy_json", ""))).expanduser()
    if not rows_csv.exists():
        raise FileNotFoundError(f"rows_csv missing for {app_dir}: {rows_csv}")
    if not policy_json.exists():
        raise FileNotFoundError(f"policy_json missing for {app_dir}: {policy_json}")
    return rows_csv.resolve(), policy_json.resolve()


def feature_summary(rows_csv: Path) -> Dict[str, Any]:
    summary = rows_csv.parent / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"feature summary missing next to {rows_csv}: {summary}")
    return read_json(summary)


def discovery_rows_from_policy(policy_json: Path) -> Path:
    bundle = read_json(policy_json)
    rows_csv = Path(str(bundle.get("rows_csv", ""))).expanduser()
    if not rows_csv.exists():
        summary = policy_json.parent / "summary.json"
        if summary.exists():
            rows_csv = Path(str((read_json(summary).get("inputs") or {}).get("rows_csv", ""))).expanduser()
    if not rows_csv.exists():
        raise FileNotFoundError(f"discovery rows_csv missing from policy {policy_json}: {rows_csv}")
    return rows_csv.resolve()


def python_for_backend(inputs: Mapping[str, Any], args: argparse.Namespace) -> str:
    backend = str(inputs.get("runtime_backend", "llava15_cleanroom"))
    if backend == "llava_next_official":
        return str(args.next_py)
    if backend == "qwen25_vl_official":
        return str(args.qwen_py)
    return str(args.cal_py)


def baseline_replay_command(
    *,
    rows_csv: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    summary = feature_summary(rows_csv)
    inputs = summary.get("inputs") or {}
    baseline_pred = str(inputs.get("baseline_pred_jsonl", "")).strip()
    if not baseline_pred:
        raise RuntimeError(f"feature summary has no baseline_pred_jsonl: {rows_csv.parent / 'summary.json'}")

    cmd = [
        python_for_backend(inputs, args),
        str(repo_root() / "scripts" / "run_discriminative_meta_strong_online.py"),
        "--question_file",
        str(inputs["question_file"]),
        "--image_folder",
        str(inputs["image_folder"]),
        "--intervention_pred_jsonl",
        baseline_pred,
        "--intervention_pred_key",
        str(inputs.get("baseline_pred_key", "auto")),
        "--baseline_pred_jsonl",
        baseline_pred,
        "--baseline_pred_key",
        str(inputs.get("baseline_pred_key", "auto")),
        "--headset_json",
        str(inputs.get("headset_json") or args.headset_json),
        "--out_dir",
        str(out_dir),
        "--model_path",
        str(inputs.get("model_path", "")),
        "--model_base",
        str(inputs.get("model_base", "")),
        "--conv_mode",
        str(inputs.get("conv_mode", "llava_v1")),
        "--runtime_backend",
        str(inputs.get("runtime_backend", "llava15_cleanroom")),
        "--extract_only",
        "true",
        "--skip_stage_a",
        "true",
        "--reuse_if_exists",
        "true",
        "--log_every",
        str(args.log_every),
    ]
    gt_csv = str(inputs.get("gt_csv", "")).strip()
    if gt_csv:
        cmd.extend(["--gt_csv", gt_csv])
    backend = str(inputs.get("runtime_backend", "llava15_cleanroom"))
    if backend == "llava_next_official":
        cmd.extend(
            [
                "--llava_next_root",
                str(inputs.get("llava_next_root") or args.llava_next_root),
                "--llava_next_torch_type",
                str(inputs.get("llava_next_torch_type") or "bf16"),
                "--llava_next_attn_implementation",
                str(inputs.get("llava_next_attn_implementation") or "sdpa"),
            ]
        )
    elif backend == "qwen25_vl_official":
        cmd.extend(
            [
                "--qwen25_torch_type",
                str(inputs.get("qwen25_torch_type") or "bf16"),
                "--qwen25_attn_implementation",
                str(inputs.get("qwen25_attn_implementation") or "eager"),
                "--qwen25_device_map",
                str(inputs.get("qwen25_device_map") or "cuda"),
                "--qwen25_min_pixels",
                str(inputs.get("qwen25_min_pixels") or 200704),
                "--qwen25_max_pixels",
                str(inputs.get("qwen25_max_pixels") or 1003520),
            ]
        )
    return cmd


def run_command(cmd: Sequence[str], *, args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.cal}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("[run]", " ".join(str(x) for x in cmd), flush=True)
    if bool(args.dry_run):
        return
    subprocess.run(list(cmd), check=True, env=env, cwd=str(repo_root()))


def ensure_baseline_replay(rows_csv: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    out_csv = out_dir / "online_feature_rows.csv"
    if out_csv.exists() and bool(args.reuse_if_exists):
        print("[reuse]", out_csv, flush=True)
        return out_csv
    out_dir.mkdir(parents=True, exist_ok=True)
    run_command(baseline_replay_command(rows_csv=rows_csv, out_dir=out_dir, args=args), args=args)
    return out_csv


def build_pairwise_delta(
    *,
    method_rows_csv: Path,
    baseline_rows_csv: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    out_csv = out_dir / "pairwise_delta_rows.csv"
    if out_csv.exists() and bool(args.reuse_if_exists):
        print("[reuse]", out_csv, flush=True)
        return out_csv
    cmd = [
        str(args.cal_py),
        str(repo_root() / "scripts" / "build_pairwise_replay_delta_features.py"),
        "--intervention_rows_csv",
        str(method_rows_csv),
        "--baseline_rows_csv",
        str(baseline_rows_csv),
        "--candidate_filter",
        str(args.candidate_filter),
        "--feature_prefixes",
        str(args.feature_prefixes),
        "--min_present_rate",
        str(args.min_present_rate),
        "--out_dir",
        str(out_dir),
    ]
    run_command(cmd, args=args)
    return out_csv


def oriented_feature_values(rows: Sequence[Mapping[str, Any]], feature: str, direction: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = maybe_float(row.get(feature))
        if value is None:
            continue
        values.append(float(value) if direction == "high" else -float(value))
    return values


def score_row(row: Mapping[str, Any], features: Sequence[Mapping[str, Any]]) -> Optional[float]:
    zs: List[float] = []
    for feat in features:
        value = maybe_float(row.get(str(feat["feature"])))
        if value is None:
            return None
        oriented = float(value) if str(feat["direction"]) == "high" else -float(value)
        zs.append((oriented - float(feat["mu"])) / max(float(feat["sd"]), 1e-6))
    return mean(zs) if zs else None


def fit_j_policy(pairwise_dir: Path, *, top_k: int) -> Dict[str, Any]:
    rows = read_csv(pairwise_dir / "pairwise_delta_rows.csv")
    metrics = read_csv(pairwise_dir / "pairwise_delta_feature_metrics.csv")[: int(top_k)]
    features: List[Dict[str, Any]] = []
    for metric in metrics:
        feature = str(metric["feature"])
        direction = str(metric["direction"])
        vals = oriented_feature_values(rows, feature, direction)
        if not vals:
            continue
        features.append(
            {
                "feature": feature,
                "direction": direction,
                "mu": mean(vals),
                "sd": std(vals),
                "auroc": float(metric.get("auroc", 0.0) or 0.0),
            }
        )
    scored: List[Tuple[float, int, int]] = []
    for row in rows:
        score = score_row(row, features)
        if score is None:
            continue
        scored.append((float(score), maybe_int(row.get("harm")), maybe_int(row.get("help"))))
    total_h = sum(h for _, h, _ in scored)
    total_g = sum(g for _, _, g in scored)
    best: Optional[Tuple[float, int, int, float, int, int, int, float, float]] = None
    for tau in sorted({s for s, _, _ in scored}):
        selected = [(s, h, g) for s, h, g in scored if s >= tau]
        h = sum(x[1] for x in selected)
        g = sum(x[2] for x in selected)
        hrec = safe_div(float(h), float(total_h))
        grec = safe_div(float(g), float(total_g))
        j = hrec - grec
        cand = (j, h - g, -len(selected), float(tau), len(selected), h, g, hrec, grec)
        if best is None or cand > best:
            best = cand
    if best is None:
        best = (0.0, 0, 0, float("inf"), 0, 0, 0, 0.0, 0.0)
    j, net, _neg_n, tau, n, h, g, hrec, grec = best
    return {
        "policy_type": "pairwise_replay_delta_j",
        "top_k": int(top_k),
        "features": features,
        "tau": float(tau),
        "fit": {
            "n": int(len(scored)),
            "selected": int(n),
            "h": int(h),
            "g": int(g),
            "net": int(net),
            "hrec": float(hrec),
            "grec": float(grec),
            "j": float(j),
            "total_harm": int(total_h),
            "total_help": int(total_g),
        },
    }


def apply_policy(pairwise_dir: Path, policy: Mapping[str, Any], deploy: Mapping[str, Any]) -> Dict[str, Any]:
    rows = read_csv(pairwise_dir / "pairwise_delta_rows.csv")
    features = list(policy.get("features") or [])
    tau = float(policy.get("tau", 0.0))
    selected: List[Dict[str, str]] = []
    present = 0
    scores: List[float] = []
    for row in rows:
        score = score_row(row, features)
        if score is None:
            continue
        present += 1
        scores.append(float(score))
        if float(score) >= tau:
            selected.append(row)
    h = sum(maybe_int(row.get("harm")) for row in selected)
    g = sum(maybe_int(row.get("help")) for row in selected)
    th = sum(maybe_int(row.get("harm")) for row in rows)
    tg = sum(maybe_int(row.get("help")) for row in rows)
    n = int(deploy.get("n") or 0) or 9000
    method_acc = float(deploy.get("intervention_acc") or 0.0)
    base_acc = float(deploy.get("baseline_acc") or 0.0)
    pair_acc = method_acc + safe_div(float(h - g), float(n))
    return {
        "present": int(present),
        "n_pair_rows": int(len(rows)),
        "selected": int(len(selected)),
        "selected_harm": int(h),
        "selected_help": int(g),
        "net": int(h - g),
        "total_harm": int(th),
        "total_help": int(tg),
        "hrec": safe_div(float(h), float(th)),
        "grec": safe_div(float(g), float(tg)),
        "baseline_acc": base_acc,
        "method_acc": method_acc,
        "pairwise_acc": pair_acc,
        "delta_vs_method": pair_acc - method_acc,
        "delta_vs_base": pair_acc - base_acc,
    }


def format_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run/evaluate pairwise baseline-vs-method replay delta policies across existing RaPiC apply jobs."
    )
    ap.add_argument("--apply_root", action="append", default=[], help="Root containing target/dataset apply summaries.")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--dataset", action="append", choices=DATASETS, default=[])
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--extract", action="store_true", help="Run missing baseline replay feature extraction.")
    ap.add_argument("--build_pairwise", action="store_true", help="Build missing pairwise delta rows.")
    ap.add_argument("--reuse_if_exists", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--cal", default="/home/kms/LLaVA_calibration")
    ap.add_argument("--cal_py", default="/home/kms/miniconda3/envs/vga_base/bin/python")
    ap.add_argument("--next_py", default="/home/kms/miniconda3/envs/llava_next_official/bin/python")
    ap.add_argument("--qwen_py", default="/home/kms/miniconda3/envs/qwen25_vl/bin/python")
    ap.add_argument("--llava_next_root", default="/home/kms/LLaVA-NeXT")
    ap.add_argument("--headset_json", default="/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json")
    ap.add_argument("--candidate_filter", default="changed_answer", choices=["all", "changed_answer", "yes_to_no", "no_to_yes"])
    ap.add_argument("--feature_prefixes", default="cheap_")
    ap.add_argument("--min_present_rate", type=float, default=0.8)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    apply_roots = [Path(x).expanduser().resolve() for x in args.apply_root]
    if not apply_roots:
        cal = Path(args.cal).expanduser()
        apply_roots = [
            cal / "experiments" / "paper_pcp_cd_finalacc_alpha0p025_main" / "apply" / "vga",
            cal / "experiments" / "paper_pcp_cd_finalacc_alpha0p025_pai_vaf_main" / "apply",
            cal / "experiments" / "paper_pcp_cd_finalacc_densealpha_next_pai_vaf_source" / "apply",
        ]
    target_filter = set(args.target or [])
    dataset_filter = set(args.dataset or DATASETS)
    out_root = Path(args.out_root).expanduser().resolve()
    jobs = [
        job
        for job in find_deployment_summaries(apply_roots)
        if (not target_filter or job[0] in target_filter) and job[1] in dataset_filter
    ]
    if not jobs:
        raise SystemExit("No apply jobs found.")

    by_target: Dict[str, Dict[str, Any]] = {}
    for target, dataset, app_dir in jobs:
        method_rows, policy_json = apply_summary_paths(app_dir)
        discovery_rows = discovery_rows_from_policy(policy_json)
        by_target.setdefault(target, {"policy_json": policy_json, "discovery_rows": discovery_rows, "datasets": {}})
        by_target[target]["datasets"][dataset] = {
            "app_dir": app_dir,
            "method_rows": method_rows,
            "deployment": read_json(app_dir / "deployment_summary.json"),
        }

    summary_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for target, info in sorted(by_target.items()):
        try:
            target_out = out_root / target
            discovery_rows = Path(info["discovery_rows"])
            disc_base_dir = target_out / "discovery" / "baseline_replay_features"
            disc_pair_dir = target_out / "discovery" / "pairwise_baseline_vs_method_replay"
            if args.extract:
                disc_base_rows = ensure_baseline_replay(discovery_rows, disc_base_dir, args)
            else:
                disc_base_rows = disc_base_dir / "online_feature_rows.csv"
            if args.build_pairwise:
                build_pairwise_delta(
                    method_rows_csv=discovery_rows,
                    baseline_rows_csv=disc_base_rows,
                    out_dir=disc_pair_dir,
                    args=args,
                )
            if not (disc_pair_dir / "pairwise_delta_rows.csv").exists():
                raise FileNotFoundError(f"missing discovery pairwise rows: {disc_pair_dir}")
            if not (disc_pair_dir / "pairwise_delta_feature_metrics.csv").exists():
                raise FileNotFoundError(f"missing discovery pairwise metrics: {disc_pair_dir}")
            policy = fit_j_policy(disc_pair_dir, top_k=int(args.top_k))
            write_json(target_out / "pairwise_policy.json", policy)

            for dataset, dinfo in sorted(info["datasets"].items()):
                method_rows = Path(dinfo["method_rows"])
                ds_out = target_out / dataset
                base_dir = ds_out / "baseline_replay_features"
                pair_dir = ds_out / "pairwise_baseline_vs_method_replay"
                if args.extract:
                    base_rows = ensure_baseline_replay(method_rows, base_dir, args)
                else:
                    base_rows = base_dir / "online_feature_rows.csv"
                if args.build_pairwise:
                    build_pairwise_delta(
                        method_rows_csv=method_rows,
                        baseline_rows_csv=base_rows,
                        out_dir=pair_dir,
                        args=args,
                    )
                if not (pair_dir / "pairwise_delta_rows.csv").exists():
                    raise FileNotFoundError(f"missing pairwise rows: {pair_dir}")
                result = apply_policy(pair_dir, policy, dinfo["deployment"])
                row = {
                    "target": target,
                    "method_backbone": label_text(target),
                    "dataset": dataset,
                    "top_k": int(args.top_k),
                    "fit_selected": policy["fit"]["selected"],
                    "fit_h_g_net": f"{policy['fit']['h']}/{policy['fit']['g']}/{policy['fit']['net']}",
                    "fit_hrec": policy["fit"]["hrec"],
                    "fit_grec": policy["fit"]["grec"],
                    "base": result["baseline_acc"],
                    "method": result["method_acc"],
                    "pairwise_rapic": result["pairwise_acc"],
                    "delta_vs_method": result["delta_vs_method"],
                    "delta_vs_base": result["delta_vs_base"],
                    "fallback": result["selected"],
                    "h_g_net": f"{result['selected_harm']}/{result['selected_help']}/{result['net']}",
                    "hrec": result["hrec"],
                    "grec": result["grec"],
                    "present": result["present"],
                    "pair_rows": result["n_pair_rows"],
                }
                summary_rows.append(row)
                write_json(ds_out / "pairwise_apply_summary.json", row)
        except Exception as exc:
            errors.append({"target": target, "error": str(exc)})
            print("[warn]", target, exc, file=sys.stderr, flush=True)

    write_csv(out_root / "pairwise_panel_summary.csv", summary_rows)
    write_json(out_root / "pairwise_panel_summary.json", {"rows": summary_rows, "errors": errors})

    lines = [
        "| Method / Backbone | Dataset | Base | Method | Pairwise RaPiC | dMethod | dBase | Fallback | H/G/Net | Hrec | Grec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method_backbone']} | {row['dataset']} | "
            f"{format_pct(row['base'])} | {format_pct(row['method'])} | {format_pct(row['pairwise_rapic'])} | "
            f"{100*float(row['delta_vs_method']):+.2f} | {100*float(row['delta_vs_base']):+.2f} | "
            f"{row['fallback']} | {row['h_g_net']} | {format_pct(row['hrec'])} | {format_pct(row['grec'])} |"
        )
    (out_root / "pairwise_panel_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("[saved]", out_root / "pairwise_panel_summary.md", flush=True)
    if errors:
        print("[warn] errors:", errors, flush=True)


if __name__ == "__main__":
    main()
