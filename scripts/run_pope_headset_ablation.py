#!/usr/bin/env python
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List


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
            qid = str(r.get(id_col, "")).strip()
            y = str(r.get(label_col, "")).strip().lower()
            if qid and y in {"yes", "no"}:
                out[qid] = y
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
    for qid, y in gt.items():
        p = pred.get(qid, None)
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


def compare_to_baseline(
    gt: Dict[str, str],
    base_pred: Dict[str, str],
    arm_pred: Dict[str, str],
) -> Dict[str, int]:
    changed = gain = harm = 0
    for qid, y in gt.items():
        b = base_pred.get(qid, None)
        a = arm_pred.get(qid, None)
        if b is None or a is None:
            continue
        if b != a:
            changed += 1
        b_ok = int(b == y)
        a_ok = int(a == y)
        if b_ok == 0 and a_ok == 1:
            gain += 1
        elif b_ok == 1 and a_ok == 0:
            harm += 1
    return {
        "changed_pred": int(changed),
        "gain": int(gain),
        "harm": int(harm),
        "net_gain": int(gain - harm),
    }


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
            "harmful_penalty_img_mean": 0.0,
            "faithful_boost_img_mean": 0.0,
            "harmful_selected_heads_mean": 0.0,
            "harmful_selected_patch_per_head_mean": 0.0,
            "harmful_selected_patch_coverage_mean": 0.0,
            "harmful_per_cell_dose_mean": 0.0,
        }

    ais = [_safe_float(r.get("ais_mean"), 0.0) for r in late]
    trig_b = [_safe_float(r.get("trigger_frac_batch"), 0.0) for r in late]
    pen = [_safe_float(r.get("penalty_img_mean"), 0.0) for r in late]
    harm = [_safe_float(r.get("harmful_penalty_img_mean"), 0.0) for r in late]
    faith = [_safe_float(r.get("faithful_boost_img_mean"), 0.0) for r in late]
    hsel = [_safe_float(r.get("harmful_selected_heads_mean"), 0.0) for r in late]
    psel = [_safe_float(r.get("harmful_selected_patch_per_head_mean"), 0.0) for r in late]
    pcov = [_safe_float(r.get("harmful_selected_patch_coverage_mean"), 0.0) for r in late]
    dose = [_safe_float(r.get("harmful_per_cell_dose_mean"), 0.0) for r in late]

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
        "harmful_penalty_img_mean": float(sum(harm) / len(harm)),
        "faithful_boost_img_mean": float(sum(faith) / len(faith)),
        "harmful_selected_heads_mean": float(sum(hsel) / len(hsel)),
        "harmful_selected_patch_per_head_mean": float(sum(psel) / len(psel)),
        "harmful_selected_patch_coverage_mean": float(sum(pcov) / len(pcov)),
        "harmful_per_cell_dose_mean": float(sum(dose) / len(dose)),
    }


def run_cmd(cmd: List[str], cwd: str) -> None:
    print("[run]", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=cwd)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="/home/kms/LLaVA_calibration")
    ap.add_argument("--python_cmd", type=str, default="python")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--label_col", type=str, default="answer")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--headset_json", type=str, required=True)
    ap.add_argument("--run_legacy", action="store_true")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--ais_early_start", type=int, default=0)
    ap.add_argument("--ais_early_end", type=int, default=15)
    ap.add_argument("--ais_late_start", type=int, default=16)
    ap.add_argument("--ais_late_end", type=int, default=31)
    ap.add_argument("--ais_topk", type=int, default=8)
    ap.add_argument("--ais_tau", type=float, default=2.2)
    ap.add_argument("--ais_gamma", type=float, default=0.2)
    ap.add_argument("--ais_eps", type=float, default=1e-6)
    ap.add_argument("--ais_faithful_boost", type=float, default=1.0)
    ap.add_argument("--ais_use_dynamic_omega", action="store_true")
    ap.add_argument("--no_ais_use_dynamic_omega", dest="ais_use_dynamic_omega", action="store_false")
    ap.set_defaults(ais_use_dynamic_omega=True)
    ap.add_argument("--ais_use_budget_routing", action="store_true")
    ap.add_argument("--ais_budget_total", type=float, default=0.0)
    ap.add_argument("--ais_harmful_top_ratio", type=float, default=0.2)
    ap.add_argument("--ais_faithful_top_ratio", type=float, default=0.2)
    ap.add_argument("--ais_bipolar_harmful_ratio", type=float, default=0.5)
    ap.add_argument("--ais_budget_patch_topk", type=int, default=16)
    ap.add_argument("--ais_strict_headset_layers", action="store_true")
    ap.add_argument("--ais_operator", type=str, default="soft", choices=["soft", "semi_hard"])
    ap.add_argument("--ais_semihard_penalty", type=float, default=0.0)
    ap.add_argument("--debug_dump", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt = load_gt(args.gt_csv, id_col=args.id_col, label_col=args.label_col)
    py = str(args.python_cmd)

    baseline_jsonl = os.path.join(args.out_dir, "baseline.jsonl")
    baseline_cmd = [
        py, "-m", "llava.eval.model_vqa_loader",
        "--model-path", args.model_path,
        "--image-folder", args.image_folder,
        "--question-file", args.question_file,
        "--answers-file", baseline_jsonl,
        "--conv-mode", args.conv_mode,
        "--temperature", str(args.temperature),
        "--num_beams", str(args.num_beams),
        "--max_new_tokens", str(args.max_new_tokens),
    ]
    run_cmd(baseline_cmd, cwd=args.repo_root)
    pred_base = load_pred(baseline_jsonl)
    base_metrics = eval_metrics(gt, pred_base)

    rows = [{
        "name": "baseline",
        "arm": "",
        "tau": "",
        "gamma": "",
        **base_metrics,
        "changed_pred": 0,
        "gain": 0,
        "harm": 0,
        "net_gain": 0,
        "delta_acc_vs_base": 0.0,
        "delta_f1_vs_base": 0.0,
        "answers_file": baseline_jsonl,
        "debug_file": "",
    }]

    arms = ["harmful_only", "faithful_only", "bipolar"]
    if bool(args.run_legacy):
        arms.append("legacy")

    for arm in arms:
        name = f"ais_{arm}_tau{args.ais_tau:g}_g{args.ais_gamma:g}"
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
            "--ais-tau", str(args.ais_tau),
            "--ais-gamma", str(args.ais_gamma),
            "--ais-eps", str(args.ais_eps),
            "--ais-arm", str(arm),
            "--ais-headset-json", os.path.abspath(args.headset_json),
            "--ais-faithful-boost", str(args.ais_faithful_boost),
            "--ais-budget-total", str(args.ais_budget_total),
            "--ais-harmful-top-ratio", str(args.ais_harmful_top_ratio),
            "--ais-faithful-top-ratio", str(args.ais_faithful_top_ratio),
            "--ais-bipolar-harmful-ratio", str(args.ais_bipolar_harmful_ratio),
            "--ais-budget-patch-topk", str(args.ais_budget_patch_topk),
            "--ais-operator", str(args.ais_operator),
            "--ais-semihard-penalty", str(args.ais_semihard_penalty),
        ]
        if bool(args.ais_strict_headset_layers):
            cmd.append("--ais-strict-headset-layers")
        if bool(args.ais_use_dynamic_omega):
            cmd.append("--ais-use-dynamic-omega")
        else:
            cmd.append("--no-ais-use-dynamic-omega")
        if bool(args.ais_use_budget_routing):
            cmd.append("--ais-use-budget-routing")
        if bool(args.debug_dump):
            cmd.extend(["--ais-debug-log", "--ais-debug-dump", dbg_csv])
        run_cmd(cmd, cwd=args.repo_root)

        pred = load_pred(out_jsonl)
        m = eval_metrics(gt, pred)
        cmp = compare_to_baseline(gt, pred_base, pred)
        dbg = summarize_debug(dbg_csv, late_start=int(args.ais_late_start), late_end=int(args.ais_late_end)) if args.debug_dump else {}
        tn_zero = int(m["TN"]) == 0
        # In budget-routing mode, trigger is expected to be always on by design.
        trig_100 = (
            (float(dbg.get("late_trigger_fraction_step_mean", 0.0)) >= 0.999)
            if (args.debug_dump and not bool(args.ais_use_budget_routing))
            else False
        )
        rows.append({
            "name": name,
            "arm": arm,
            "tau": float(args.ais_tau),
            "gamma": float(args.ais_gamma),
            **m,
            **cmp,
            **dbg,
            "reject_tn_zero": bool(tn_zero),
            "reject_trigger_100": bool(trig_100),
            "reject_any": bool(tn_zero or trig_100),
            "delta_acc_vs_base": float(m["acc"] - base_metrics["acc"]),
            "delta_f1_vs_base": float(m["f1"] - base_metrics["f1"]),
            "answers_file": out_jsonl,
            "debug_file": (dbg_csv if args.debug_dump else ""),
        })

    arm_rows = [r for r in rows if str(r.get("name")) != "baseline"]
    arm_rows_sorted = sorted(arm_rows, key=lambda x: (float(x["acc"]), float(x["f1"])), reverse=True)
    best = arm_rows_sorted[0] if arm_rows_sorted else None

    out_csv = os.path.join(args.out_dir, "ablation_metrics.csv")
    keys = [
        "name", "arm", "tau", "gamma",
        "n", "acc", "f1", "precision", "recall", "yes_ratio",
        "TP", "FP", "TN", "FN",
        "changed_pred", "gain", "harm", "net_gain",
        "debug_rows", "debug_late_rows",
        "ais_late_mean", "ais_late_p90",
        "trigger_frac_batch_mean", "late_trigger_fraction_step_mean", "late_trigger_fraction_step_p90",
        "penalty_img_mean", "harmful_penalty_img_mean", "faithful_boost_img_mean",
        "harmful_selected_heads_mean", "harmful_selected_patch_per_head_mean",
        "harmful_selected_patch_coverage_mean", "harmful_per_cell_dose_mean",
        "reject_tn_zero", "reject_trigger_100", "reject_any",
        "delta_acc_vs_base", "delta_f1_vs_base",
        "answers_file", "debug_file",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in [rows[0]] + arm_rows_sorted:
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
                    "headset_json": os.path.abspath(args.headset_json),
                    "run_legacy": bool(args.run_legacy),
                    "conv_mode": args.conv_mode,
                    "temperature": float(args.temperature),
                    "num_beams": int(args.num_beams),
                    "max_new_tokens": int(args.max_new_tokens),
                    "ais_early_start": int(args.ais_early_start),
                    "ais_early_end": int(args.ais_early_end),
                    "ais_late_start": int(args.ais_late_start),
                    "ais_late_end": int(args.ais_late_end),
                    "ais_topk": int(args.ais_topk),
                    "ais_tau": float(args.ais_tau),
                    "ais_gamma": float(args.ais_gamma),
                    "ais_eps": float(args.ais_eps),
                    "ais_faithful_boost": float(args.ais_faithful_boost),
                    "ais_use_dynamic_omega": bool(args.ais_use_dynamic_omega),
                    "ais_use_budget_routing": bool(args.ais_use_budget_routing),
                    "ais_budget_total": float(args.ais_budget_total),
                    "ais_harmful_top_ratio": float(args.ais_harmful_top_ratio),
                    "ais_faithful_top_ratio": float(args.ais_faithful_top_ratio),
                    "ais_bipolar_harmful_ratio": float(args.ais_bipolar_harmful_ratio),
                    "ais_budget_patch_topk": int(args.ais_budget_patch_topk),
                    "ais_strict_headset_layers": bool(args.ais_strict_headset_layers),
                    "ais_operator": str(args.ais_operator),
                    "ais_semihard_penalty": float(args.ais_semihard_penalty),
                    "debug_dump": bool(args.debug_dump),
                },
                "baseline": rows[0],
                "best_arm": best,
                "outputs": {"ablation_metrics_csv": out_csv, "summary_json": out_summary},
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
