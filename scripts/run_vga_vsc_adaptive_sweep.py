#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import load_label_map, load_question_rows, parse_bool, parse_yes_no, safe_id, write_csv, write_json
from pnp_controller.adapters.vga_online import VGAOnlineAdapter, VGAOnlineConfig


def maybe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(round(float(s)))
    except Exception:
        return None


def mean_or_zero(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(seq) / float(len(seq)))


def std_or_one(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    if len(seq) <= 1:
        return 1.0
    mu = mean_or_zero(seq)
    var = sum((float(v) - mu) ** 2 for v in seq) / float(len(seq))
    sd = float(math.sqrt(max(0.0, var)))
    return sd if sd > 1e-8 else 1.0


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def compute_norm_entropy(prob: Sequence[float], eps: float = 1e-12) -> float:
    seq = [max(float(x), eps) for x in prob]
    n = len(seq)
    if n <= 1:
        return 0.0
    z = float(sum(seq))
    if z <= 0:
        return 0.0
    seq = [x / z for x in seq]
    h = -sum(p * math.log(p) for p in seq)
    return float(h / math.log(n))


def compute_effective_support_size(prob: Sequence[float], eps: float = 1e-12) -> float:
    seq = [max(float(x), 0.0) for x in prob]
    z = float(sum(seq))
    if z <= 0:
        return 1.0
    seq = [x / z for x in seq]
    denom = float(sum(p * p for p in seq))
    return float(1.0 / max(eps, denom))


def topk_mass(values: Sequence[float], k: int) -> float:
    seq = sorted((max(float(x), 0.0) for x in values), reverse=True)
    if not seq:
        return 0.0
    z = float(sum(seq))
    if z <= 0:
        return 0.0
    kk = max(1, min(int(k), len(seq)))
    return float(sum(seq[:kk]) / z)


def safe_quantile(values: Sequence[float], q: float) -> float:
    seq = sorted(float(v) for v in values)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    pos = max(0.0, min(1.0, float(q))) * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(seq[lo])
    frac = float(pos - lo)
    return float(seq[lo] * (1.0 - frac) + seq[hi] * frac)


def z_scores(values: Sequence[float]) -> List[float]:
    seq = [float(v) for v in values]
    mu = mean_or_zero(seq)
    sd = std_or_one(seq)
    return [float((float(v) - mu) / sd) for v in seq]


def parse_quantiles(spec: str) -> List[float]:
    out: List[float] = []
    for part in str(spec or "").split(","):
        s = part.strip()
        if s == "":
            continue
        try:
            q = float(s)
        except Exception:
            continue
        if 0.0 <= q <= 1.0:
            out.append(float(q))
    return sorted(set(out))


def build_profile_kwargs(
    ctx: Dict[str, Any],
    *,
    profile: str,
    weak_attn_coef: float,
    weak_cd_alpha: float,
    weak_start_layer: int,
    weak_end_layer: int,
) -> Tuple[Dict[str, Any], bool]:
    if profile == "baseline":
        return dict(ctx["base_gen_kwargs"]), False
    kwargs = dict(ctx["method_gen_kwargs"])
    if profile == "weak":
        kwargs["attn_coef"] = float(weak_attn_coef)
        kwargs["cd_alpha"] = float(weak_cd_alpha)
        kwargs["add_layer"] = list(range(int(weak_start_layer), int(weak_end_layer) + 1))
    return kwargs, True


def compute_branch_correct(text: str, gt_label: str) -> Tuple[str, Optional[int]]:
    label = parse_yes_no(text) if str(text or "").strip() else ""
    if gt_label not in {"yes", "no"} or label == "":
        return label, None
    return label, int(label == gt_label)


def score_from_row(row: Dict[str, Any], score_name: str) -> float:
    return float(row.get(score_name, 0.0) or 0.0)


def evaluate_binary_policy(rows: Sequence[Dict[str, Any]], score_name: str, tau: float) -> Dict[str, Any]:
    decision_rows: List[Dict[str, Any]] = []
    final_correct_sum = 0
    n_eval = 0
    n_baseline = 0
    n_strong = 0
    for row in rows:
        score = score_from_row(row, score_name)
        route = "strong" if score >= float(tau) else "baseline"
        final_text = str(row.get(f"{route}_text", ""))
        final_label = str(row.get(f"{route}_label", ""))
        final_correct = maybe_int(row.get(f"{route}_correct"))
        if route == "baseline":
            n_baseline += 1
        else:
            n_strong += 1
        if final_correct is not None:
            final_correct_sum += int(final_correct)
            n_eval += 1
        out = dict(row)
        out["policy_type"] = "binary"
        out["score_name"] = score_name
        out["tau"] = float(tau)
        out["route"] = route
        out["final_text"] = final_text
        out["final_label"] = final_label
        out["final_correct"] = final_correct
        decision_rows.append(out)
    final_acc = None if n_eval == 0 else float(final_correct_sum / float(n_eval))
    return {
        "policy_type": "binary",
        "score_name": score_name,
        "tau": float(tau),
        "final_acc": final_acc,
        "baseline_rate": float(n_baseline / max(1, len(rows))),
        "strong_rate": float(n_strong / max(1, len(rows))),
        "decision_rows": decision_rows,
    }


def evaluate_ternary_policy(
    rows: Sequence[Dict[str, Any]],
    score_name: str,
    tau_low: float,
    tau_high: float,
) -> Dict[str, Any]:
    decision_rows: List[Dict[str, Any]] = []
    final_correct_sum = 0
    n_eval = 0
    counts = {"baseline": 0, "weak": 0, "strong": 0}
    for row in rows:
        score = score_from_row(row, score_name)
        if score < float(tau_low):
            route = "baseline"
        elif score < float(tau_high):
            route = "weak"
        else:
            route = "strong"
        counts[route] += 1
        final_text = str(row.get(f"{route}_text", ""))
        final_label = str(row.get(f"{route}_label", ""))
        final_correct = maybe_int(row.get(f"{route}_correct"))
        if final_correct is not None:
            final_correct_sum += int(final_correct)
            n_eval += 1
        out = dict(row)
        out["policy_type"] = "ternary"
        out["score_name"] = score_name
        out["tau_low"] = float(tau_low)
        out["tau_high"] = float(tau_high)
        out["route"] = route
        out["final_text"] = final_text
        out["final_label"] = final_label
        out["final_correct"] = final_correct
        decision_rows.append(out)
    final_acc = None if n_eval == 0 else float(final_correct_sum / float(n_eval))
    return {
        "policy_type": "ternary",
        "score_name": score_name,
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "final_acc": final_acc,
        "baseline_rate": float(counts["baseline"] / max(1, len(rows))),
        "weak_rate": float(counts["weak"] / max(1, len(rows))),
        "strong_rate": float(counts["strong"] / max(1, len(rows))),
        "decision_rows": decision_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run discovery-time adaptive VGA-v2 sweep using VSC trust routing.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--gt_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--vga_root", type=str, default="/home/kms/VGA_origin")
    ap.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_gen_len", type=int, default=8)
    ap.add_argument("--sampling", type=parse_bool, default=False)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--head_balancing", type=str, default="simg")
    ap.add_argument("--attn_norm", type=parse_bool, default=False)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--reuse_if_exists", type=parse_bool, default=False)
    ap.add_argument("--strong_attn_coef", type=float, default=0.2)
    ap.add_argument("--strong_cd_alpha", type=float, default=0.02)
    ap.add_argument("--strong_start_layer", type=int, default=2)
    ap.add_argument("--strong_end_layer", type=int, default=15)
    ap.add_argument("--weak_attn_coef", type=float, default=0.1)
    ap.add_argument("--weak_cd_alpha", type=float, default=0.01)
    ap.add_argument("--weak_start_layer", type=int, default=4)
    ap.add_argument("--weak_end_layer", type=int, default=12)
    ap.add_argument("--quantiles", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    branch_rows_csv = os.path.join(out_dir, "branch_rows.csv")
    policy_sweep_csv = os.path.join(out_dir, "policy_sweep.csv")
    selected_policy_json = os.path.join(out_dir, "selected_policy.json")
    selected_rows_csv = os.path.join(out_dir, "selected_decision_rows.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    if bool(args.reuse_if_exists) and os.path.isfile(summary_json) and os.path.isfile(selected_rows_csv):
        print(f"[reuse] {summary_json}")
        return

    gt_map = load_label_map(args.gt_csv)
    samples = load_question_rows(args.question_file, limit=int(args.limit))

    adapter = VGAOnlineAdapter(
        VGAOnlineConfig(
            vga_root=args.vga_root,
            model_path=args.model_path,
            image_folder=args.image_folder,
            conv_mode=args.conv_mode,
            model_base=(args.model_base or None),
            device=args.device,
            sampling=bool(args.sampling),
            max_gen_len=int(args.max_gen_len),
            num_beams=int(args.num_beams),
            cd_alpha=float(args.strong_cd_alpha),
            attn_coef=float(args.strong_attn_coef),
            start_layer=int(args.strong_start_layer),
            end_layer=int(args.strong_end_layer),
            head_balancing=str(args.head_balancing),
            attn_norm=bool(args.attn_norm),
            probe_feature_mode="aggregate",
            use_gmi=False,
            seed=int(args.seed),
            prefer_local_llava=False,
        )
    )

    rows: List[Dict[str, Any]] = []
    baseline_correct_sum = 0
    weak_correct_sum = 0
    strong_correct_sum = 0
    n_eval_baseline = 0
    n_eval_weak = 0
    n_eval_strong = 0
    prefill_secs: List[float] = []
    decode_secs: List[float] = []
    n_errors = 0

    for idx, sample in enumerate(samples):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        gt_label = str(gt_map.get(sample_id, "")).strip().lower()
        row: Dict[str, Any] = {
            "id": sample_id,
            "image": str(sample.get("image", "")).strip(),
            "question": str(sample.get("text", sample.get("question", ""))).strip(),
            "gt_label": gt_label,
            "error": "",
            "error_traceback": "",
        }
        try:
            t0 = time.perf_counter()
            ctx = adapter._prepare_runtime_context(sample)
            prefill_secs.append(float(time.perf_counter() - t0))

            guidance = [float(x) for x in ctx["vl_guidance"].detach().float().cpu().tolist()]
            g_top1 = topk_mass(guidance, 1)
            g_top5 = float(ctx.get("g_top5_mass", topk_mass(guidance, 5)))
            g_entropy = compute_norm_entropy(guidance)
            g_ess = compute_effective_support_size(guidance)
            g_top1_top2_gap = float(g_top1 - topk_mass(guidance, 2))
            mode_object = int(str(ctx.get("guidance_mode", "")).strip() == "object")
            object_count = int(len(ctx["prepared"].get("object_list", [])))

            row.update(
                {
                    "guidance_mode": str(ctx.get("guidance_mode", "")),
                    "mode_object": int(mode_object),
                    "object_count": int(object_count),
                    "guidance_len": int(len(guidance)),
                    "G_top1_mass": float(g_top1),
                    "G_top5_mass": float(g_top5),
                    "G_entropy": float(g_entropy),
                    "G_effective_support_size": float(g_ess),
                    "G_inv_effective_support_size": float(1.0 / max(1e-8, g_ess)),
                    "G_top1_top2_gap": float(g_top1_top2_gap),
                }
            )

            for profile in ("baseline", "weak", "strong"):
                gen_kwargs, use_add = build_profile_kwargs(
                    ctx,
                    profile=profile,
                    weak_attn_coef=float(args.weak_attn_coef),
                    weak_cd_alpha=float(args.weak_cd_alpha),
                    weak_start_layer=int(args.weak_start_layer),
                    weak_end_layer=int(args.weak_end_layer),
                )
                t1 = time.perf_counter()
                pred = adapter._generate_from_prepared(
                    sample=sample,
                    prepared=ctx["prepared"],
                    gen_kwargs=gen_kwargs,
                    use_add=bool(use_add),
                    capture_proxy=False,
                )
                decode_secs.append(float(time.perf_counter() - t1))
                text = str(pred.get("output", "")).strip()
                label, correct = compute_branch_correct(text, gt_label)
                row[f"{profile}_text"] = text
                row[f"{profile}_label"] = label
                row[f"{profile}_correct"] = correct

            if row["baseline_correct"] is not None:
                baseline_correct_sum += int(row["baseline_correct"])
                n_eval_baseline += 1
            if row["weak_correct"] is not None:
                weak_correct_sum += int(row["weak_correct"])
                n_eval_weak += 1
            if row["strong_correct"] is not None:
                strong_correct_sum += int(row["strong_correct"])
                n_eval_strong += 1
        except Exception as exc:
            n_errors += 1
            row["error"] = str(exc)
            row["error_traceback"] = traceback.format_exc()
        rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[vga-vsc-adaptive] {idx + 1}/{len(samples)}")

    if not rows:
        raise RuntimeError("No rows were processed.")

    g_top1_z = z_scores([float(r.get("G_top1_mass", 0.0) or 0.0) for r in rows])
    g_top5_z = z_scores([float(r.get("G_top5_mass", 0.0) or 0.0) for r in rows])
    g_ent_z = z_scores([float(r.get("G_entropy", 0.0) or 0.0) for r in rows])
    g_ess_z = z_scores([float(r.get("G_effective_support_size", 0.0) or 0.0) for r in rows])
    g_gap_z = z_scores([float(r.get("G_top1_top2_gap", 0.0) or 0.0) for r in rows])
    for i, row in enumerate(rows):
        row["trust_g_top1"] = float(row.get("G_top1_mass", 0.0) or 0.0)
        row["trust_g_top5"] = float(row.get("G_top5_mass", 0.0) or 0.0)
        row["trust_neg_entropy"] = float(-(float(row.get("G_entropy", 0.0) or 0.0)))
        row["trust_neg_ess"] = float(-(float(row.get("G_effective_support_size", 0.0) or 0.0)))
        row["trust_top1_gap"] = float(row.get("G_top1_top2_gap", 0.0) or 0.0)
        row["trust_object_mode"] = float(row.get("mode_object", 0) or 0)
        row["trust_composite_v1"] = float(
            g_top1_z[i] + g_top5_z[i] + g_gap_z[i] - g_ent_z[i] - g_ess_z[i] + 0.5 * float(row["trust_object_mode"])
        )

    quantiles = parse_quantiles(args.quantiles)
    if not quantiles:
        quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    score_names = [
        "trust_g_top1",
        "trust_g_top5",
        "trust_neg_entropy",
        "trust_neg_ess",
        "trust_top1_gap",
        "trust_composite_v1",
    ]

    candidate_rows: List[Dict[str, Any]] = []
    best_policy: Optional[Dict[str, Any]] = None
    best_decisions: List[Dict[str, Any]] = []
    for score_name in score_names:
        values = [score_from_row(row, score_name) for row in rows]
        tau_values = [safe_quantile(values, q) for q in quantiles]
        for tau in tau_values:
            result = evaluate_binary_policy(rows, score_name, tau)
            final_acc = result["final_acc"]
            candidate_rows.append(
                {
                    "policy_type": "binary",
                    "score_name": score_name,
                    "tau": float(tau),
                    "final_acc": final_acc,
                    "baseline_rate": result["baseline_rate"],
                    "strong_rate": result["strong_rate"],
                }
            )
            if final_acc is not None and (best_policy is None or float(final_acc) > float(best_policy["final_acc"])):
                best_policy = {k: v for k, v in result.items() if k != "decision_rows"}
                best_decisions = result["decision_rows"]

        for q_low in quantiles:
            for q_high in quantiles:
                if float(q_high) <= float(q_low):
                    continue
                tau_low = safe_quantile(values, q_low)
                tau_high = safe_quantile(values, q_high)
                result = evaluate_ternary_policy(rows, score_name, tau_low, tau_high)
                final_acc = result["final_acc"]
                candidate_rows.append(
                    {
                        "policy_type": "ternary",
                        "score_name": score_name,
                        "tau_low": float(tau_low),
                        "tau_high": float(tau_high),
                        "final_acc": final_acc,
                        "baseline_rate": result["baseline_rate"],
                        "weak_rate": result["weak_rate"],
                        "strong_rate": result["strong_rate"],
                    }
                )
                if final_acc is not None and (best_policy is None or float(final_acc) > float(best_policy["final_acc"])):
                    best_policy = {k: v for k, v in result.items() if k != "decision_rows"}
                    best_decisions = result["decision_rows"]

    baseline_acc = None if n_eval_baseline == 0 else float(baseline_correct_sum / float(n_eval_baseline))
    weak_acc = None if n_eval_weak == 0 else float(weak_correct_sum / float(n_eval_weak))
    strong_acc = None if n_eval_strong == 0 else float(strong_correct_sum / float(n_eval_strong))
    branch_rows = sorted(rows, key=lambda row: str(row.get("id", "")))

    if best_policy is None:
        static_candidates: List[Tuple[str, Optional[float]]] = [
            ("baseline", baseline_acc),
            ("weak", weak_acc),
            ("strong", strong_acc),
        ]
        static_candidates = [(name, acc) for name, acc in static_candidates if acc is not None]
        if static_candidates:
            best_branch, best_branch_acc = max(static_candidates, key=lambda item: float(item[1]))
            best_policy = {
                "policy_type": "static",
                "selected_branch": str(best_branch),
                "final_acc": float(best_branch_acc),
                "baseline_rate": float(1.0 if best_branch == "baseline" else 0.0),
                "weak_rate": float(1.0 if best_branch == "weak" else 0.0),
                "strong_rate": float(1.0 if best_branch == "strong" else 0.0),
                "selection_note": "No valid threshold policy was found; fell back to best static branch.",
            }
            best_decisions = []
            for row in branch_rows:
                chosen = str(best_branch)
                out = dict(row)
                out["policy_type"] = "static"
                out["route"] = chosen
                out["final_text"] = str(row.get(f"{chosen}_text", ""))
                out["final_label"] = str(row.get(f"{chosen}_label", ""))
                out["final_correct"] = maybe_int(row.get(f"{chosen}_correct"))
                best_decisions.append(out)
        else:
            write_csv(branch_rows_csv, branch_rows)
            write_csv(policy_sweep_csv, candidate_rows)
            raise RuntimeError(
                "Failed to select an adaptive VGA policy because no branch produced evaluable correctness. "
                f"n_errors={n_errors}, n_eval_baseline={n_eval_baseline}, n_eval_weak={n_eval_weak}, n_eval_strong={n_eval_strong}, "
                f"partial rows saved to {branch_rows_csv}."
            )

    selected_rows = sorted(best_decisions, key=lambda row: str(row.get("id", "")))
    candidate_rows.sort(key=lambda row: (-float(row.get("final_acc", -1.0) or -1.0), str(row.get("policy_type", "")), str(row.get("score_name", ""))))
    final_acc = float(best_policy["final_acc"])

    write_csv(branch_rows_csv, branch_rows)
    write_csv(policy_sweep_csv, candidate_rows)
    write_csv(selected_rows_csv, selected_rows)
    write_json(selected_policy_json, best_policy)
    write_json(
        summary_json,
        {
            "inputs": {
                "question_file": os.path.abspath(args.question_file),
                "image_folder": os.path.abspath(args.image_folder),
                "gt_csv": os.path.abspath(args.gt_csv),
                "vga_root": os.path.abspath(args.vga_root),
                "model_path": str(args.model_path),
                "model_base": str(args.model_base),
                "conv_mode": str(args.conv_mode),
                "device": str(args.device),
                "max_gen_len": int(args.max_gen_len),
                "sampling": bool(args.sampling),
                "num_beams": int(args.num_beams),
                "head_balancing": str(args.head_balancing),
                "attn_norm": bool(args.attn_norm),
                "strong_profile": {
                    "attn_coef": float(args.strong_attn_coef),
                    "cd_alpha": float(args.strong_cd_alpha),
                    "start_layer": int(args.strong_start_layer),
                    "end_layer": int(args.strong_end_layer),
                },
                "weak_profile": {
                    "attn_coef": float(args.weak_attn_coef),
                    "cd_alpha": float(args.weak_cd_alpha),
                    "start_layer": int(args.weak_start_layer),
                    "end_layer": int(args.weak_end_layer),
                },
                "quantiles": quantiles,
            },
            "evaluation": {
                "n_rows": int(len(rows)),
                "n_eval_baseline": int(n_eval_baseline),
                "n_eval_weak": int(n_eval_weak),
                "n_eval_strong": int(n_eval_strong),
                "n_errors": int(n_errors),
                "baseline_acc": baseline_acc,
                "weak_acc": weak_acc,
                "strong_acc": strong_acc,
                "selected_final_acc": float(final_acc),
                "delta_vs_baseline": (None if baseline_acc is None else float(final_acc - baseline_acc)),
                "delta_vs_weak": (None if weak_acc is None else float(final_acc - weak_acc)),
                "delta_vs_strong": (None if strong_acc is None else float(final_acc - strong_acc)),
            },
            "selected_policy": best_policy,
            "timing": {
                "prefill_mean_ms": float(1000.0 * mean_or_zero(prefill_secs)),
                "decode_mean_ms_per_branch": float(1000.0 * mean_or_zero(decode_secs)),
                "branch_count_per_sample": 3,
            },
            "outputs": {
                "branch_rows_csv": os.path.abspath(branch_rows_csv),
                "policy_sweep_csv": os.path.abspath(policy_sweep_csv),
                "selected_policy_json": os.path.abspath(selected_policy_json),
                "selected_rows_csv": os.path.abspath(selected_rows_csv),
            },
        },
    )
    print(f"[saved] {branch_rows_csv}")
    print(f"[saved] {policy_sweep_csv}")
    print(f"[saved] {selected_policy_json}")
    print(f"[saved] {summary_json}")


if __name__ == "__main__":
    main()
