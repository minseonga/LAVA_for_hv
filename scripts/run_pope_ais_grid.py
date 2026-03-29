#!/usr/bin/env python
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple


def parse_yes_no(text: str) -> str:
    s = (text or "").strip()
    first = s.split(".", 1)[0].replace(",", " ")
    words = set(w.strip().lower() for w in first.split())
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def load_gt(path_csv: str, id_col: str, label_col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            k = str(r.get(id_col, "")).strip()
            v = str(r.get(label_col, "")).strip().lower()
            if k and v in {"yes", "no"}:
                out[k] = v
    return out


def load_pred(path_jsonl: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = str(r.get("question_id", "")).strip()
            if not qid:
                continue
            out[qid] = parse_yes_no(r.get("text", ""))
    return out


def eval_metrics(gt: Dict[str, str], pred: Dict[str, str]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for k, y in gt.items():
        p = pred.get(k, None)
        if p is None:
            continue
        if y == "yes" and p == "yes":
            tp += 1
        elif y == "no" and p == "yes":
            fp += 1
        elif y == "no" and p == "no":
            tn += 1
        elif y == "yes" and p == "no":
            fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    yes_ratio = (tp + fp) / n if n else 0.0
    return {
        "n": n,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "yes_ratio": yes_ratio,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def run_cmd(cmd: List[str], cwd: str) -> None:
    print("[run]", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=cwd)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def parse_grid(grid: str) -> List[Tuple[float, float]]:
    out = []
    for token in grid.split(","):
        t = token.strip()
        if not t:
            continue
        tau_s, gamma_s = t.split(":")
        out.append((float(tau_s), float(gamma_s)))
    return out


def parse_float_list(s: str) -> List[float]:
    vals = []
    for tok in str(s).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def _safe_float(v, default=0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    q = min(1.0, max(0.0, float(q)))
    idx = int(round((len(s) - 1) * q))
    idx = max(0, min(len(s) - 1, idx))
    return float(s[idx])


def load_debug_rows(path_csv: str) -> List[dict]:
    rows = []
    if not path_csv or not os.path.exists(path_csv):
        return rows
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def _filter_late(rows: List[dict], late_start: int, late_end: int) -> List[dict]:
    out = []
    for r in rows:
        li = _safe_int(r.get("layer_idx"), -10**9)
        if int(late_start) <= li <= int(late_end):
            out.append(r)
    return out


def derive_tau_from_debug_quantiles(path_csv: str, late_start: int, late_end: int, quantiles: List[float]) -> Dict[str, float]:
    rows = load_debug_rows(path_csv)
    late = _filter_late(rows, late_start=late_start, late_end=late_end)
    vals = [_safe_float(r.get("ais_mean"), 0.0) for r in late]
    out: Dict[str, float] = {}
    for q in quantiles:
        out[f"p{int(round(q * 100))}"] = _quantile(vals, q)
    return out


def summarize_debug(path_csv: str, late_start: int, late_end: int) -> Dict[str, float]:
    rows = load_debug_rows(path_csv)
    late = _filter_late(rows, late_start=late_start, late_end=late_end)
    if len(late) == 0:
        return {
            "debug_rows": int(len(rows)),
            "debug_late_rows": 0,
            "ais_late_mean": 0.0,
            "ais_late_p90": 0.0,
            "trigger_frac_batch_mean": 0.0,
            "late_trigger_fraction_step_mean": 0.0,
            "late_trigger_fraction_step_p90": 0.0,
            "penalty_img_mean": 0.0,
        }

    ais = [_safe_float(r.get("ais_mean"), 0.0) for r in late]
    trig_b = [_safe_float(r.get("trigger_frac_batch"), 0.0) for r in late]
    pen = [_safe_float(r.get("penalty_img_mean"), 0.0) for r in late]

    # Step-level trigger fraction: take max within (generation_idx, step_idx) across late layers.
    by_step = {}
    for r in late:
        key = (_safe_int(r.get("generation_idx"), 0), _safe_int(r.get("step_idx"), 0))
        v = _safe_float(r.get("late_trigger_fraction_step"), 0.0)
        if key not in by_step or v > by_step[key]:
            by_step[key] = v
    step_vals = list(by_step.values()) if by_step else [0.0]

    return {
        "debug_rows": int(len(rows)),
        "debug_late_rows": int(len(late)),
        "ais_late_mean": float(sum(ais) / len(ais)),
        "ais_late_p90": float(_quantile(ais, 0.9)),
        "trigger_frac_batch_mean": float(sum(trig_b) / len(trig_b)),
        "late_trigger_fraction_step_mean": float(sum(step_vals) / len(step_vals)),
        "late_trigger_fraction_step_p90": float(_quantile(step_vals, 0.9)),
        "penalty_img_mean": float(sum(pen) / len(pen)),
    }


def compute_ais_reduction_vs_baseline(
    baseline_debug_csv: str,
    run_debug_csv: str,
    late_start: int,
    late_end: int,
) -> Dict[str, float]:
    base_rows = _filter_late(load_debug_rows(baseline_debug_csv), late_start=late_start, late_end=late_end)
    run_rows = _filter_late(load_debug_rows(run_debug_csv), late_start=late_start, late_end=late_end)
    if len(base_rows) == 0 or len(run_rows) == 0:
        return {
            "ais_delta_mean_vs_baseline": 0.0,
            "ais_delta_common_n": 0,
        }

    key_of = lambda r: (_safe_int(r.get("generation_idx"), -1), _safe_int(r.get("step_idx"), -1), _safe_int(r.get("layer_idx"), -1))
    bmap = {key_of(r): _safe_float(r.get("ais_mean"), 0.0) for r in base_rows}
    rmap = {key_of(r): _safe_float(r.get("ais_mean"), 0.0) for r in run_rows}
    common = [k for k in bmap.keys() if k in rmap]
    if len(common) == 0:
        # fallback: mean difference by distribution-level summary
        bmean = sum(bmap.values()) / max(1, len(bmap))
        rmean = sum(rmap.values()) / max(1, len(rmap))
        return {
            "ais_delta_mean_vs_baseline": float(bmean - rmean),
            "ais_delta_common_n": 0,
        }
    diffs = [float(bmap[k] - rmap[k]) for k in common]
    return {
        "ais_delta_mean_vs_baseline": float(sum(diffs) / len(diffs)),
        "ais_delta_common_n": int(len(common)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="/home/kms/LLaVA_calibration")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--label_col", type=str, default="answer")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--grid", type=str, default="2.0:0.1,2.2:0.2,2.4:0.2")
    ap.add_argument("--grid_from_baseline_quantiles", action="store_true")
    ap.add_argument("--baseline_debug_csv", type=str, default="")
    ap.add_argument("--tau_quantiles", type=str, default="0.90,0.95,0.97,0.99")
    ap.add_argument("--gamma_list", type=str, default="0.05,0.1,0.2,0.4")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--ais_early_start", type=int, default=0)
    ap.add_argument("--ais_early_end", type=int, default=15)
    ap.add_argument("--ais_late_start", type=int, default=16)
    ap.add_argument("--ais_late_end", type=int, default=31)
    ap.add_argument("--ais_topk", type=int, default=8)
    ap.add_argument("--ais_eps", type=float, default=1e-6)
    ap.add_argument("--debug_dump", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt = load_gt(args.gt_csv, id_col=args.id_col, label_col=args.label_col)

    py = sys.executable
    base_jsonl = os.path.join(args.out_dir, "baseline.jsonl")
    base_cmd = [
        py, "-m", "llava.eval.model_vqa_loader",
        "--model-path", args.model_path,
        "--image-folder", args.image_folder,
        "--question-file", args.question_file,
        "--answers-file", base_jsonl,
        "--conv-mode", args.conv_mode,
        "--temperature", str(args.temperature),
        "--num_beams", str(args.num_beams),
        "--max_new_tokens", str(args.max_new_tokens),
    ]
    run_cmd(base_cmd, cwd=args.repo_root)
    base_metrics = eval_metrics(gt, load_pred(base_jsonl))

    rows = []
    rows.append({
        "name": "baseline",
        "tau": "",
        "gamma": "",
        **base_metrics,
        "delta_acc_vs_base": 0.0,
        "delta_f1_vs_base": 0.0,
        "answers_file": base_jsonl,
        "debug_file": "",
    })

    grid_pairs: List[Tuple[float, float]]
    tau_from_quantiles = {}
    if bool(args.grid_from_baseline_quantiles):
        if not args.baseline_debug_csv or not os.path.exists(args.baseline_debug_csv):
            raise RuntimeError("--grid_from_baseline_quantiles requires valid --baseline_debug_csv")
        quantiles = parse_float_list(args.tau_quantiles)
        gammas = parse_float_list(args.gamma_list)
        tau_from_quantiles = derive_tau_from_debug_quantiles(
            args.baseline_debug_csv,
            late_start=int(args.ais_late_start),
            late_end=int(args.ais_late_end),
            quantiles=quantiles,
        )
        tau_vals = list(tau_from_quantiles.values())
        grid_pairs = [(float(t), float(g)) for t in tau_vals for g in gammas]
        print("[info] derived tau from baseline quantiles:", tau_from_quantiles)
    else:
        grid_pairs = parse_grid(args.grid)

    for tau, gamma in grid_pairs:
        name = f"ais_tau{tau:g}_g{gamma:g}"
        out_jsonl = os.path.join(args.out_dir, f"{name}.jsonl")
        dbg_csv = os.path.join(args.out_dir, f"{name}_debug.csv")
        cmd = [
            py, "-m", "llava.eval.model_vqa_loader",
            "--model-path", args.model_path,
            "--image-folder", args.image_folder,
            "--question-file", args.question_file,
            "--answers-file", out_jsonl,
            "--conv-mode", args.conv_mode,
            "--temperature", str(args.temperature),
            "--num_beams", str(args.num_beams),
            "--max_new_tokens", str(args.max_new_tokens),
            "--enable-ais-gating",
            "--ais-early-start", str(args.ais_early_start),
            "--ais-early-end", str(args.ais_early_end),
            "--ais-late-start", str(args.ais_late_start),
            "--ais-late-end", str(args.ais_late_end),
            "--ais-topk", str(args.ais_topk),
            "--ais-tau", str(tau),
            "--ais-gamma", str(gamma),
            "--ais-eps", str(args.ais_eps),
        ]
        if args.debug_dump:
            cmd.extend(["--ais-debug-log", "--ais-debug-dump", dbg_csv])
        run_cmd(cmd, cwd=args.repo_root)
        m = eval_metrics(gt, load_pred(out_jsonl))
        dbg_stats = summarize_debug(
            dbg_csv,
            late_start=int(args.ais_late_start),
            late_end=int(args.ais_late_end),
        ) if args.debug_dump else {}
        ais_delta_stats = (
            compute_ais_reduction_vs_baseline(
                baseline_debug_csv=args.baseline_debug_csv,
                run_debug_csv=dbg_csv,
                late_start=int(args.ais_late_start),
                late_end=int(args.ais_late_end),
            )
            if (args.debug_dump and args.baseline_debug_csv and os.path.exists(args.baseline_debug_csv))
            else {}
        )
        tn_zero = int(m["TN"]) == 0
        trig_100 = float(dbg_stats.get("late_trigger_fraction_step_mean", 0.0)) >= 0.999 if args.debug_dump else False
        rows.append({
            "name": name,
            "tau": tau,
            "gamma": gamma,
            **m,
            **dbg_stats,
            **ais_delta_stats,
            "reject_tn_zero": bool(tn_zero),
            "reject_trigger_100": bool(trig_100),
            "reject_any": bool(tn_zero or trig_100),
            "delta_acc_vs_base": float(m["acc"] - base_metrics["acc"]),
            "delta_f1_vs_base": float(m["f1"] - base_metrics["f1"]),
            "answers_file": out_jsonl,
            "debug_file": (dbg_csv if args.debug_dump else ""),
        })

    rows_sorted = sorted(rows[1:], key=lambda x: (float(x["delta_acc_vs_base"]), float(x["delta_f1_vs_base"])), reverse=True)
    best = rows_sorted[0] if rows_sorted else None

    out_csv = os.path.join(args.out_dir, "grid_metrics.csv")
    keys = [
        "name", "tau", "gamma",
        "n", "acc", "f1", "precision", "recall", "yes_ratio",
        "TP", "FP", "TN", "FN",
        "debug_rows", "debug_late_rows",
        "ais_late_mean", "ais_late_p90",
        "trigger_frac_batch_mean", "late_trigger_fraction_step_mean", "late_trigger_fraction_step_p90",
        "penalty_img_mean",
        "ais_delta_mean_vs_baseline", "ais_delta_common_n",
        "reject_tn_zero", "reject_trigger_100", "reject_any",
        "delta_acc_vs_base", "delta_f1_vs_base",
        "answers_file", "debug_file",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in [rows[0]] + rows_sorted:
            wr.writerow({k: r.get(k, "") for k in keys})

    out_summary = os.path.join(args.out_dir, "summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": {
                    "model_path": args.model_path,
                    "image_folder": args.image_folder,
                    "question_file": args.question_file,
                    "gt_csv": args.gt_csv,
                    "grid": args.grid,
                    "grid_from_baseline_quantiles": bool(args.grid_from_baseline_quantiles),
                    "baseline_debug_csv": args.baseline_debug_csv,
                    "tau_quantiles": args.tau_quantiles,
                    "gamma_list": args.gamma_list,
                    "ais_early_start": int(args.ais_early_start),
                    "ais_early_end": int(args.ais_early_end),
                    "ais_late_start": int(args.ais_late_start),
                    "ais_late_end": int(args.ais_late_end),
                    "ais_topk": int(args.ais_topk),
                    "ais_eps": float(args.ais_eps),
                },
                "baseline": rows[0],
                "best_ais": best,
                "derived_tau": tau_from_quantiles,
                "outputs": {"grid_metrics_csv": out_csv, "summary_json": out_summary},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[saved]", out_csv)
    print("[saved]", out_summary)
    if best is not None:
        print("[best]", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
