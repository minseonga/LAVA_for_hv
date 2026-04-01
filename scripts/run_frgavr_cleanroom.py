#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import (
    CleanroomLlavaRuntime,
    ScoreRow,
    load_headset,
    load_label_map,
    load_prediction_text_map,
    load_question_rows,
    parse_bool,
    parse_yes_no,
    safe_id,
    select_content_indices,
    stage_a_score_from_pack,
    stage_b_score_from_packs,
    threshold_candidates,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_q_grid(text: str) -> List[float]:
    out: List[float] = []
    for part in str(text or "").split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    return out or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def load_score_rows_csv(path: str) -> List[ScoreRow]:
    import csv

    rows: List[ScoreRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                ScoreRow(
                    sample_id=str(row.get("id", "")),
                    image=str(row.get("image", "")),
                    question=str(row.get("question", "")),
                    intervention_text=str(row.get("intervention_text", "")),
                    baseline_text=str(row.get("baseline_text", "")),
                    gt_label=str(row.get("gt_label", "")),
                    intervention_label=str(row.get("intervention_label", "")),
                    baseline_label=str(row.get("baseline_label", "")),
                    intervention_correct=_maybe_int(row.get("intervention_correct")),
                    baseline_correct=_maybe_int(row.get("baseline_correct")),
                    stage_a_score=float(row.get("stage_a_score", "nan")),
                    stage_a_gap_mean=float(row.get("stage_a_gap_mean", "nan")),
                    stage_a_faithful_mean=float(row.get("stage_a_faithful_mean", "nan")),
                    stage_a_harmful_mean=float(row.get("stage_a_harmful_mean", "nan")),
                    stage_a_faithful_std=float(row.get("stage_a_faithful_std", "nan")),
                    stage_b_score=float(row.get("stage_b_score", "nan")),
                    stage_b_delta_mean=float(row.get("stage_b_delta_mean", "nan")),
                    stage_b_delta_std=float(row.get("stage_b_delta_std", "nan")),
                    n_cont_tokens=int(float(row.get("n_cont_tokens", "0"))),
                    n_content_tokens=int(float(row.get("n_content_tokens", "0"))),
                    score_error=str(row.get("score_error", "")),
                )
            )
    return rows


def _maybe_int(value: object) -> Optional[int]:
    s = str(value or "").strip()
    if s == "" or s.lower() in {"none", "nan", "null"}:
        return None
    return int(float(s))


def compute_score_rows(args: argparse.Namespace) -> List[ScoreRow]:
    question_rows = load_question_rows(args.question_file, limit=int(args.limit))
    intervention_map = load_prediction_text_map(args.intervention_pred_jsonl, text_key=args.intervention_pred_key)
    baseline_map = load_prediction_text_map(args.baseline_pred_jsonl, text_key=args.baseline_pred_key) if args.baseline_pred_jsonl else {}
    gt_map = load_label_map(args.gt_csv, id_col=args.gt_id_col, label_col=args.gt_label_col) if args.gt_csv else {}
    headset = load_headset(args.headset_json, late_start=int(args.late_start), late_end=int(args.late_end))
    runtime = CleanroomLlavaRuntime(
        model_path=args.model_path,
        model_base=(None if not str(args.model_base).strip() else args.model_base),
        conv_mode=args.conv_mode,
        load_8bit=bool(args.load_8bit),
        load_4bit=bool(args.load_4bit),
    )

    score_rows: List[ScoreRow] = []
    for idx, sample in enumerate(question_rows):
        sample_id = safe_id(sample.get("question_id", sample.get("id")))
        question = str(sample.get("question", sample.get("text", ""))).strip()
        image_name = str(sample.get("image", "")).strip()
        image_path = os.path.join(args.image_folder, image_name)
        intervention_text = intervention_map.get(sample_id, "").strip()
        baseline_text = baseline_map.get(sample_id, "").strip()
        gt_label = gt_map.get(sample_id, "")
        intervention_label = parse_yes_no(intervention_text) if intervention_text else ""
        baseline_label = parse_yes_no(baseline_text) if baseline_text else ""
        intervention_correct = None if not gt_label or not intervention_label else int(intervention_label == gt_label)
        baseline_correct = None if not gt_label or not baseline_label else int(baseline_label == gt_label)

        row = ScoreRow(
            sample_id=sample_id,
            image=image_name,
            question=question,
            intervention_text=intervention_text,
            baseline_text=baseline_text,
            gt_label=gt_label,
            intervention_label=intervention_label,
            baseline_label=baseline_label,
            intervention_correct=intervention_correct,
            baseline_correct=baseline_correct,
            stage_a_score=float("-inf"),
            stage_a_gap_mean=float("nan"),
            stage_a_faithful_mean=float("nan"),
            stage_a_harmful_mean=float("nan"),
            stage_a_faithful_std=float("nan"),
            stage_b_score=float("-inf"),
            stage_b_delta_mean=float("nan"),
            stage_b_delta_std=float("nan"),
            n_cont_tokens=0,
            n_content_tokens=0,
            score_error="",
        )

        try:
            if not sample_id:
                raise ValueError("Missing sample id.")
            if not question:
                raise ValueError("Missing question text.")
            if not image_name:
                raise ValueError("Missing image filename.")
            if not intervention_text:
                raise ValueError("Missing intervention prediction text.")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = runtime.load_image(image_path)
            real_pack = runtime.teacher_force_candidate(
                image=image,
                question=question,
                candidate_text=intervention_text,
                output_attentions=True,
            )
            control_image = runtime.make_blur_control(image, blur_radius=float(args.blur_radius))
            blur_pack = runtime.teacher_force_candidate(
                image=control_image,
                question=question,
                candidate_text=intervention_text,
                output_attentions=False,
            )

            content_indices = select_content_indices(runtime.tokenizer, real_pack.cont_ids)
            stage_a = stage_a_score_from_pack(
                pack=real_pack,
                headset=headset,
                beta=float(args.beta),
                lambda_a=float(args.lambda_a),
                content_indices=content_indices,
            )
            stage_b = stage_b_score_from_packs(
                real_pack=real_pack,
                control_pack=blur_pack,
                lambda_b=float(args.lambda_b),
                content_indices=content_indices,
            )

            row.stage_a_score = float(stage_a["stage_a_score"])
            row.stage_a_gap_mean = float(stage_a["stage_a_gap_mean"])
            row.stage_a_faithful_mean = float(stage_a["stage_a_faithful_mean"])
            row.stage_a_harmful_mean = float(stage_a["stage_a_harmful_mean"])
            row.stage_a_faithful_std = float(stage_a["stage_a_faithful_std"])
            row.stage_b_score = float(stage_b["stage_b_score"])
            row.stage_b_delta_mean = float(stage_b["stage_b_delta_mean"])
            row.stage_b_delta_std = float(stage_b["stage_b_delta_std"])
            row.n_cont_tokens = int(real_pack.cont_ids.numel())
            row.n_content_tokens = int(len(content_indices))
        except Exception as exc:
            row.score_error = str(exc)
        score_rows.append(row)
        if (idx + 1) % max(1, int(args.log_every)) == 0:
            print(f"[score] {idx + 1}/{len(question_rows)}")

    return score_rows


def evaluate_policy(
    rows: Sequence[ScoreRow],
    tau_a: float,
    tau_b: float,
    use_stage_b: bool,
    baseline_fallback: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pred_rows: List[Dict[str, Any]] = []
    n_eval = 0
    n_final_correct = 0
    n_base_correct = 0
    n_int_correct = 0
    n_stage_a = 0
    n_rescue = 0
    n_stage_a_harm = 0
    n_rescue_harm = 0

    for row in rows:
        stage_a_trigger = not math.isfinite(row.stage_a_score) or float(row.stage_a_score) < float(tau_a)
        stage_b_trigger = False
        route = "keep_stage_a"
        final_text = row.intervention_text
        final_label = row.intervention_label
        if stage_a_trigger:
            n_stage_a += 1
            if use_stage_b:
                stage_b_trigger = (not math.isfinite(row.stage_b_score)) or float(row.stage_b_score) < float(tau_b)
                route = "keep_stage_b"
            else:
                stage_b_trigger = True
                route = "rescue_stage_a_only"
            if stage_b_trigger:
                route = "baseline_rescue"
                if baseline_fallback:
                    final_text = row.baseline_text
                    final_label = row.baseline_label
                n_rescue += 1

        final_correct = None
        if row.gt_label and final_label:
            final_correct = int(final_label == row.gt_label)
            n_eval += 1
            n_final_correct += int(final_correct)
            if row.intervention_correct is not None:
                n_int_correct += int(row.intervention_correct)
            if row.baseline_correct is not None:
                n_base_correct += int(row.baseline_correct)
            if stage_a_trigger and row.intervention_correct is not None and int(row.intervention_correct) == 0:
                n_stage_a_harm += 1
            if stage_b_trigger and row.intervention_correct is not None and int(row.intervention_correct) == 0:
                n_rescue_harm += 1

        pred_rows.append(
            {
                "id": row.sample_id,
                "image": row.image,
                "question": row.question,
                "route": route,
                "stage_a_trigger": bool(stage_a_trigger),
                "stage_b_trigger": bool(stage_b_trigger),
                "stage_a_score": float(row.stage_a_score),
                "stage_b_score": float(row.stage_b_score),
                "intervention_text": row.intervention_text,
                "baseline_text": row.baseline_text,
                "final_text": final_text,
                "gt_label": row.gt_label,
                "intervention_label": row.intervention_label,
                "baseline_label": row.baseline_label,
                "final_label": final_label,
                "intervention_correct": row.intervention_correct,
                "baseline_correct": row.baseline_correct,
                "final_correct": final_correct,
                "score_error": row.score_error,
            }
        )

    eval_n = max(1, n_eval)
    intervention_acc = (n_int_correct / eval_n) if n_eval else None
    baseline_acc = (n_base_correct / eval_n) if n_eval else None
    final_acc = (n_final_correct / eval_n) if n_eval else None
    verify_rate = float(n_stage_a / max(1, len(rows)))
    rescue_rate = float(n_rescue / max(1, len(rows)))
    summary = {
        "n_total": int(len(rows)),
        "n_eval": int(n_eval),
        "tau_a": float(tau_a),
        "tau_b": float(tau_b),
        "use_stage_b": bool(use_stage_b),
        "verify_rate": verify_rate,
        "rescue_rate": rescue_rate,
        "policy_pass_proxy": float(1.0 + verify_rate + rescue_rate),
        "prototype_compute_proxy": float(2.0 + verify_rate + rescue_rate),
        "intervention_acc": intervention_acc,
        "baseline_acc": baseline_acc,
        "final_acc": final_acc,
        "delta_vs_intervention": (None if final_acc is None or intervention_acc is None else float(final_acc - intervention_acc)),
        "stage_a_flagged_intervention_harm_rate": float(n_stage_a_harm / max(1, n_stage_a)),
        "rescued_intervention_harm_rate": float(n_rescue_harm / max(1, n_rescue)),
    }
    return pred_rows, summary


def build_threshold_sweep(
    rows: Sequence[ScoreRow],
    q_grid: Sequence[float],
    max_verify_rate: float,
    max_rescue_rate: float,
    use_stage_b: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    tau_a_candidates = threshold_candidates([row.stage_a_score for row in rows], q_grid=q_grid)
    tau_b_candidates = threshold_candidates([row.stage_b_score for row in rows], q_grid=q_grid) if use_stage_b else [0.0]
    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for tau_a in tau_a_candidates:
        for tau_b in tau_b_candidates:
            _, summary = evaluate_policy(
                rows=rows,
                tau_a=float(tau_a),
                tau_b=float(tau_b),
                use_stage_b=bool(use_stage_b),
                baseline_fallback=True,
            )
            sweep_row = {
                "tau_a": float(tau_a),
                "tau_b": float(tau_b),
                **summary,
            }
            sweep_rows.append(sweep_row)
            if summary["final_acc"] is None:
                continue
            if float(summary["verify_rate"]) > float(max_verify_rate):
                continue
            if float(summary["rescue_rate"]) > float(max_rescue_rate):
                continue
            if best is None:
                best = sweep_row
                continue
            cand_key = (
                float(summary["final_acc"]),
                -float(summary["rescue_rate"]),
                -float(summary["verify_rate"]),
            )
            best_key = (
                float(best["final_acc"]),
                -float(best["rescue_rate"]),
                -float(best["verify_rate"]),
            )
            if cand_key > best_key:
                best = sweep_row
    return sweep_rows, best


def oracle_summary(rows: Sequence[ScoreRow]) -> Dict[str, Any]:
    n = 0
    oracle_correct = 0
    for row in rows:
        if row.intervention_correct is None or row.baseline_correct is None:
            continue
        n += 1
        oracle_correct += max(int(row.intervention_correct), int(row.baseline_correct))
    return {
        "n_eval": int(n),
        "oracle_rescue_acc": (None if n == 0 else float(oracle_correct / float(n))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean-room FRG-AVR experiment runner.")
    ap.add_argument("--question_file", type=str, required=True)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl", type=str, required=True)
    ap.add_argument("--intervention_pred_key", type=str, default="output")
    ap.add_argument("--baseline_pred_jsonl", type=str, default="")
    ap.add_argument("--baseline_pred_key", type=str, default="text")
    ap.add_argument("--gt_csv", type=str, default="")
    ap.add_argument("--gt_id_col", type=str, default="id")
    ap.add_argument("--gt_label_col", type=str, default="answer")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model_base", type=str, default="")
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--headset_json", type=str, required=True)
    ap.add_argument("--late_start", type=int, default=-1)
    ap.add_argument("--late_end", type=int, default=-1)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lambda_a", type=float, default=0.5)
    ap.add_argument("--lambda_b", type=float, default=0.5)
    ap.add_argument("--blur_radius", type=float, default=12.0)
    ap.add_argument("--tau_a", type=float, default=float("nan"))
    ap.add_argument("--tau_b", type=float, default=float("nan"))
    ap.add_argument("--search_thresholds", type=lambda x: parse_bool(x), default=False)
    ap.add_argument("--q_grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95")
    ap.add_argument("--max_verify_rate", type=float, default=0.25)
    ap.add_argument("--max_rescue_rate", type=float, default=0.15)
    ap.add_argument("--disable_stage_b", type=lambda x: parse_bool(x), default=False)
    ap.add_argument("--reuse_scores", type=lambda x: parse_bool(x), default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--load_8bit", type=lambda x: parse_bool(x), default=False)
    ap.add_argument("--load_4bit", type=lambda x: parse_bool(x), default=False)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scores_csv = os.path.join(args.out_dir, "sample_scores.csv")

    if bool(args.reuse_scores) and os.path.isfile(scores_csv):
        print(f"[reuse] {scores_csv}")
        score_rows = load_score_rows_csv(scores_csv)
    else:
        score_rows = compute_score_rows(args)
        write_csv(scores_csv, [row.to_csv_row() for row in score_rows])
        print(f"[saved] {scores_csv}")

    q_grid = parse_q_grid(args.q_grid)
    use_stage_b = not bool(args.disable_stage_b)
    sweep_rows, best = build_threshold_sweep(
        rows=score_rows,
        q_grid=q_grid,
        max_verify_rate=float(args.max_verify_rate),
        max_rescue_rate=float(args.max_rescue_rate),
        use_stage_b=use_stage_b,
    )
    sweep_csv = os.path.join(args.out_dir, "threshold_sweep.csv")
    write_csv(sweep_csv, sweep_rows)
    print(f"[saved] {sweep_csv}")

    tau_a = float(args.tau_a)
    tau_b = float(args.tau_b)
    if bool(args.search_thresholds) or (not math.isfinite(tau_a)) or (use_stage_b and not math.isfinite(tau_b)):
        if best is None:
            raise RuntimeError("No feasible threshold pair found under the current constraints.")
        tau_a = float(best["tau_a"])
        tau_b = float(best["tau_b"])

    pred_rows, policy_summary = evaluate_policy(
        rows=score_rows,
        tau_a=tau_a,
        tau_b=tau_b,
        use_stage_b=use_stage_b,
        baseline_fallback=True,
    )
    pred_jsonl = os.path.join(args.out_dir, "pred_frgavr_cleanroom.jsonl")
    write_jsonl(pred_jsonl, pred_rows)
    print(f"[saved] {pred_jsonl}")

    summary = {
        "inputs": {
            "question_file": os.path.abspath(args.question_file),
            "image_folder": os.path.abspath(args.image_folder),
            "intervention_pred_jsonl": os.path.abspath(args.intervention_pred_jsonl),
            "baseline_pred_jsonl": (os.path.abspath(args.baseline_pred_jsonl) if args.baseline_pred_jsonl else ""),
            "gt_csv": (os.path.abspath(args.gt_csv) if args.gt_csv else ""),
            "model_path": args.model_path,
            "model_base": args.model_base,
            "conv_mode": args.conv_mode,
            "headset_json": os.path.abspath(args.headset_json),
            "beta": float(args.beta),
            "lambda_a": float(args.lambda_a),
            "lambda_b": float(args.lambda_b),
            "blur_radius": float(args.blur_radius),
            "late_start": int(args.late_start),
            "late_end": int(args.late_end),
            "use_stage_b": bool(use_stage_b),
            "max_verify_rate": float(args.max_verify_rate),
            "max_rescue_rate": float(args.max_rescue_rate),
        },
        "thresholds": {
            "tau_a": float(tau_a),
            "tau_b": float(tau_b),
            "selected_from_search": bool(args.search_thresholds or not math.isfinite(float(args.tau_a)) or (use_stage_b and not math.isfinite(float(args.tau_b)))),
            "best_feasible": best,
        },
        "scores": {
            "n_total": int(len(score_rows)),
            "n_score_errors": int(sum(1 for row in score_rows if row.score_error)),
        },
        "oracle": oracle_summary(score_rows),
        "policy": policy_summary,
        "outputs": {
            "scores_csv": os.path.abspath(scores_csv),
            "threshold_sweep_csv": os.path.abspath(sweep_csv),
            "pred_jsonl": os.path.abspath(pred_jsonl),
        },
    }
    summary_json = os.path.join(args.out_dir, "summary.json")
    write_json(summary_json, summary)
    print(f"[saved] {summary_json}")


if __name__ == "__main__":
    main()
