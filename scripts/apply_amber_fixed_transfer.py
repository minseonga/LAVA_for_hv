#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from frgavr_cleanroom.runtime import load_prediction_text_map, parse_yes_no, read_jsonl, safe_id, write_json


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
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


def maybe_float(value: object) -> Optional[float]:
    s = str(value or "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    return out


def normalize_disc_response(text: str) -> str:
    return "Yes" if parse_yes_no(text) == "yes" else "No"


def load_feature_map(*paths: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for path in paths:
        for row in load_csv_rows(path):
            sid = safe_id(row.get("id"))
            if sid:
                out[sid] = row
    return out


def score_row(row: Dict[str, Any], policy: Dict[str, Any]) -> Optional[float]:
    policy_type = str(policy.get("policy_type", "single"))
    if policy_type == "single":
        value_a = maybe_float(row.get(policy["feature_a"]))
        if value_a is None:
            return None
        return -float(value_a) if str(policy["direction_a"]) == "low" else float(value_a)
    if policy_type == "pair_sum_z":
        value_a = maybe_float(row.get(policy["feature_a"]))
        value_b = maybe_float(row.get(policy["feature_b"]))
        if value_a is None or value_b is None:
            return None
        oa = -float(value_a) if str(policy["direction_a"]) == "low" else float(value_a)
        ob = -float(value_b) if str(policy["direction_b"]) == "low" else float(value_b)
        mu_a = float(policy["mu_a"])
        mu_b = float(policy["mu_b"])
        sd_a = max(float(policy["sd_a"]), 1e-6)
        sd_b = max(float(policy["sd_b"]), 1e-6)
        return float((oa - mu_a) / sd_a + (ob - mu_b) / sd_b)
    raise ValueError(f"Unsupported policy_type: {policy_type}")


def extract_metric_value(line: str) -> Optional[float]:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*$", line.strip())
    if not m:
        return None
    return float(m.group(1))


def parse_official_eval(stdout: str) -> Dict[str, Any]:
    lines = [line.rstrip() for line in str(stdout).splitlines()]
    out: Dict[str, Any] = {"raw_stdout": stdout}
    section = ""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.endswith(":") and s in {
            "Generative Task:",
            "Descriminative Task:",
            "Exsitence:",
            "Attribute:",
            "State:",
            "Number:",
            "Action:",
            "Relation:",
        }:
            section = s[:-1].lower().replace(" ", "_")
            continue
        if ":" not in s:
            continue
        key = s.split(":", 1)[0].strip().lower()
        value = extract_metric_value(s)
        if value is None:
            continue
        if section:
            out[f"{section}.{key}"] = value
        else:
            out[key] = value
    chair = out.get("generative_task.chair")
    f1 = out.get("descriminative_task.f1")
    if chair is not None and f1 is not None:
        out["amber_score_percent"] = float((100.0 - float(chair) + float(f1)) / 2.0)
        out["amber_score_unit"] = float((1.0 - float(chair) / 100.0 + float(f1) / 100.0) / 2.0)
    return out


def run_official_eval(
    *,
    amber_root: str,
    python_bin: str,
    inference_data: str,
    evaluation_type: str,
) -> Dict[str, Any]:
    cmd = [
        python_bin,
        os.path.join(amber_root, "inference.py"),
        "--inference_data",
        inference_data,
        "--evaluation_type",
        evaluation_type,
    ]
    proc = subprocess.run(
        cmd,
        cwd=amber_root,
        text=True,
        capture_output=True,
    )
    out = {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode == 0:
        out["metrics"] = parse_official_eval(proc.stdout)
    return out


def subset_name_from_type(amber_type: str) -> str:
    return "generative" if str(amber_type) == "generative" else "discriminative"


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a POPE-calibrated fixed cheap-proxy policy to AMBER outputs.")
    ap.add_argument("--question_file_all", type=str, required=True)
    ap.add_argument("--features_csv_generative", type=str, required=True)
    ap.add_argument("--features_csv_discriminative", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl_generative", type=str, required=True)
    ap.add_argument("--baseline_pred_jsonl_discriminative", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl_generative", type=str, required=True)
    ap.add_argument("--intervention_pred_jsonl_discriminative", type=str, required=True)
    ap.add_argument("--policy_json", type=str, required=True)
    ap.add_argument("--amber_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--baseline_pred_text_key", type=str, default="text")
    ap.add_argument("--intervention_pred_text_key", type=str, default="output")
    ap.add_argument("--python_bin", type=str, default=sys.executable)
    ap.add_argument("--run_official_eval", type=str, default="true")
    args = ap.parse_args()

    run_eval = str(args.run_official_eval).strip().lower() in {"1", "true", "yes", "y", "on"}
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    question_rows = read_jsonl(args.question_file_all)
    feature_map = load_feature_map(args.features_csv_generative, args.features_csv_discriminative)
    baseline_map: Dict[str, str] = {}
    baseline_map.update(load_prediction_text_map(args.baseline_pred_jsonl_generative, args.baseline_pred_text_key))
    baseline_map.update(load_prediction_text_map(args.baseline_pred_jsonl_discriminative, args.baseline_pred_text_key))
    intervention_map: Dict[str, str] = {}
    intervention_map.update(load_prediction_text_map(args.intervention_pred_jsonl_generative, args.intervention_pred_text_key))
    intervention_map.update(load_prediction_text_map(args.intervention_pred_jsonl_discriminative, args.intervention_pred_text_key))

    with open(args.policy_json, "r", encoding="utf-8") as f:
        policy = json.load(f)

    decision_rows: List[Dict[str, Any]] = []
    official_responses: Dict[str, List[Dict[str, Any]]] = {
        "baseline_all": [],
        "baseline_generative": [],
        "baseline_discriminative": [],
        "intervention_all": [],
        "intervention_generative": [],
        "intervention_discriminative": [],
        "final_all": [],
        "final_generative": [],
        "final_discriminative": [],
    }
    counts = {
        "n_all": 0,
        "n_generative": 0,
        "n_discriminative": 0,
        "rescue_all": 0,
        "rescue_generative": 0,
        "rescue_discriminative": 0,
    }

    for q in question_rows:
        sid = safe_id(q.get("question_id", q.get("id")))
        amber_type = str(q.get("amber_type", "")).strip()
        subset = subset_name_from_type(amber_type)
        feat_row = feature_map.get(sid, {})
        baseline_text = str(baseline_map.get(sid, "")).strip()
        intervention_text = str(intervention_map.get(sid, "")).strip()
        score = score_row(feat_row, policy)
        rescue = bool(score is not None and float(score) >= float(policy["tau"]))
        final_text = baseline_text if rescue else intervention_text

        if subset == "discriminative":
            baseline_resp = normalize_disc_response(baseline_text)
            intervention_resp = normalize_disc_response(intervention_text)
            final_resp = normalize_disc_response(final_text)
        else:
            baseline_resp = baseline_text
            intervention_resp = intervention_text
            final_resp = final_text

        qid_int = int(sid)
        official_responses[f"baseline_{subset}"].append({"id": qid_int, "response": baseline_resp})
        official_responses[f"intervention_{subset}"].append({"id": qid_int, "response": intervention_resp})
        official_responses[f"final_{subset}"].append({"id": qid_int, "response": final_resp})
        official_responses["baseline_all"].append({"id": qid_int, "response": baseline_resp})
        official_responses["intervention_all"].append({"id": qid_int, "response": intervention_resp})
        official_responses["final_all"].append({"id": qid_int, "response": final_resp})

        counts["n_all"] += 1
        counts[f"n_{subset}"] += 1
        if rescue:
            counts["rescue_all"] += 1
            counts[f"rescue_{subset}"] += 1

        decision_rows.append(
            {
                "id": sid,
                "amber_type": amber_type,
                "subset": subset,
                "feature_a": policy.get("feature_a", ""),
                "feature_a_raw": feat_row.get(policy.get("feature_a", ""), ""),
                "feature_b": policy.get("feature_b", ""),
                "feature_b_raw": feat_row.get(policy.get("feature_b", ""), "") if policy.get("feature_b") else "",
                "proxy_score": score,
                "tau": policy["tau"],
                "rescue": int(rescue),
                "baseline_text": baseline_text,
                "intervention_text": intervention_text,
                "final_text": final_text,
                "baseline_response_official": baseline_resp,
                "intervention_response_official": intervention_resp,
                "final_response_official": final_resp,
            }
        )

    response_dir = os.path.join(out_dir, "official_responses")
    os.makedirs(response_dir, exist_ok=True)
    response_paths: Dict[str, str] = {}
    for name, rows in official_responses.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["id"]))
        path = os.path.join(response_dir, f"{name}.json")
        write_json(path, rows_sorted)
        response_paths[name] = path

    decisions_csv = os.path.join(out_dir, "decision_rows.csv")
    write_csv(decisions_csv, decision_rows)

    eval_results: Dict[str, Any] = {}
    if run_eval:
        eval_plan = [
            ("baseline_generative", "g"),
            ("intervention_generative", "g"),
            ("final_generative", "g"),
            ("baseline_discriminative", "d"),
            ("intervention_discriminative", "d"),
            ("final_discriminative", "d"),
            ("baseline_all", "a"),
            ("intervention_all", "a"),
            ("final_all", "a"),
        ]
        for name, eval_type in eval_plan:
            eval_results[name] = run_official_eval(
                amber_root=os.path.abspath(args.amber_root),
                python_bin=args.python_bin,
                inference_data=response_paths[name],
                evaluation_type=eval_type,
            )

    summary = {
        "inputs": {
            "question_file_all": os.path.abspath(args.question_file_all),
            "features_csv_generative": os.path.abspath(args.features_csv_generative),
            "features_csv_discriminative": os.path.abspath(args.features_csv_discriminative),
            "baseline_pred_jsonl_generative": os.path.abspath(args.baseline_pred_jsonl_generative),
            "baseline_pred_jsonl_discriminative": os.path.abspath(args.baseline_pred_jsonl_discriminative),
            "intervention_pred_jsonl_generative": os.path.abspath(args.intervention_pred_jsonl_generative),
            "intervention_pred_jsonl_discriminative": os.path.abspath(args.intervention_pred_jsonl_discriminative),
            "policy_json": os.path.abspath(args.policy_json),
            "amber_root": os.path.abspath(args.amber_root),
        },
        "policy": policy,
        "counts": {
            **counts,
            "rescue_rate_all": (0.0 if counts["n_all"] == 0 else float(counts["rescue_all"] / counts["n_all"])),
            "rescue_rate_generative": (0.0 if counts["n_generative"] == 0 else float(counts["rescue_generative"] / counts["n_generative"])),
            "rescue_rate_discriminative": (0.0 if counts["n_discriminative"] == 0 else float(counts["rescue_discriminative"] / counts["n_discriminative"])),
        },
        "outputs": {
            "decision_rows_csv": decisions_csv,
            **response_paths,
        },
        "official_eval": eval_results,
    }
    summary_json = os.path.join(out_dir, "summary.json")
    write_json(summary_json, summary)
    print("[saved]", decisions_csv)
    for path in response_paths.values():
        print("[saved]", path)
    print("[saved]", summary_json)


if __name__ == "__main__":
    main()
