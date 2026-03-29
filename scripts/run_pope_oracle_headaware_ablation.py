#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


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


def load_debug_rows(path_csv: str) -> List[dict]:
    rows = []
    if not path_csv or not os.path.exists(path_csv):
        return rows
    with open(path_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def summarize_debug(path_csv: str, late_start: int, late_end: int) -> Dict[str, float]:
    rows = load_debug_rows(path_csv)
    late = []
    for r in rows:
        li = _safe_int(r.get("layer_idx"), -10**9)
        if int(late_start) <= li <= int(late_end):
            late.append(r)
    if not late:
        return {
            "debug_rows": int(len(rows)),
            "debug_late_rows": 0,
            "trigger_frac_batch_mean": 0.0,
            "late_trigger_fraction_step_mean": 0.0,
            "penalty_img_mean": 0.0,
            "harmful_penalty_img_mean": 0.0,
            "faithful_boost_img_mean": 0.0,
            "harmful_selected_heads_mean": 0.0,
            "harmful_selected_patch_coverage_mean": 0.0,
            "harmful_per_cell_dose_mean": 0.0,
            "oracle_supportive_cols_mean": 0.0,
            "oracle_assertive_cols_mean": 0.0,
        }
    trig_b = [_safe_float(r.get("trigger_frac_batch"), 0.0) for r in late]
    pen = [_safe_float(r.get("penalty_img_mean"), 0.0) for r in late]
    harm = [_safe_float(r.get("harmful_penalty_img_mean"), 0.0) for r in late]
    faith = [_safe_float(r.get("faithful_boost_img_mean"), 0.0) for r in late]
    hsel = [_safe_float(r.get("harmful_selected_heads_mean"), 0.0) for r in late]
    pcov = [_safe_float(r.get("harmful_selected_patch_coverage_mean"), 0.0) for r in late]
    dose = [_safe_float(r.get("harmful_per_cell_dose_mean"), 0.0) for r in late]
    supc = [_safe_float(r.get("oracle_supportive_cols_mean"), 0.0) for r in late]
    assc = [_safe_float(r.get("oracle_assertive_cols_mean"), 0.0) for r in late]
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
        "trigger_frac_batch_mean": float(sum(trig_b) / len(trig_b)),
        "late_trigger_fraction_step_mean": float(sum(step_vals) / len(step_vals)),
        "penalty_img_mean": float(sum(pen) / len(pen)),
        "harmful_penalty_img_mean": float(sum(harm) / len(harm)),
        "faithful_boost_img_mean": float(sum(faith) / len(faith)),
        "harmful_selected_heads_mean": float(sum(hsel) / len(hsel)),
        "harmful_selected_patch_coverage_mean": float(sum(pcov) / len(pcov)),
        "harmful_per_cell_dose_mean": float(sum(dose) / len(dose)),
        "oracle_supportive_cols_mean": float(sum(supc) / len(supc)),
        "oracle_assertive_cols_mean": float(sum(assc) / len(assc)),
    }


def run_cmd(cmd: List[str], cwd: str) -> None:
    print("[run]", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=cwd)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def parse_float_list(s: str) -> List[float]:
    out = []
    for x in str(s or "").split(","):
        x = x.strip()
        if x == "":
            continue
        out.append(float(x))
    return out


def parse_controls(s: str) -> List[str]:
    out: List[str] = []
    for x in str(s or "").split(","):
        t = x.strip().lower()
        if t:
            out.append(t)
    return out


def _parse_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if v is None or v == "":
            return default
        return int(v)
    except Exception:
        return default


def load_role_catalog(path_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Builds per-sample role catalog:
      - pool: unique candidate_patch_idx seen in role csv
      - supportive_ranked: unique supportive patch ids sorted by candidate_rank
      - assertive_ranked: unique assertive/harmful patch ids sorted by candidate_rank
    """
    out: Dict[str, Dict[str, Any]] = {}
    with open(path_csv, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sid = str(r.get("id") or "").strip()
            if sid == "":
                continue
            pidx = _parse_int(r.get("candidate_patch_idx"), None)
            if pidx is None:
                continue
            rank = _parse_int(r.get("candidate_rank"), 10**9)
            role = str(r.get("role_label") or "").strip().lower()
            item = out.setdefault(
                sid,
                {
                    "pool": set(),
                    "supportive_pairs": [],
                    "assertive_pairs": [],
                },
            )
            item["pool"].add(int(pidx))
            if role == "supportive":
                item["supportive_pairs"].append((int(rank), int(pidx)))
            elif role in {"harmful", "assertive"}:
                item["assertive_pairs"].append((int(rank), int(pidx)))

    for sid, item in out.items():
        def _ranked_unique(pairs: Sequence[Tuple[int, int]]) -> List[int]:
            arr = sorted(list(pairs), key=lambda x: (int(x[0]), int(x[1])))
            seen: Set[int] = set()
            keep: List[int] = []
            for _, p in arr:
                if int(p) in seen:
                    continue
                keep.append(int(p))
                seen.add(int(p))
            return keep

        item["pool"] = sorted(int(x) for x in set(item["pool"]))
        item["supportive_ranked"] = _ranked_unique(item["supportive_pairs"])
        item["assertive_ranked"] = _ranked_unique(item["assertive_pairs"])
        item.pop("supportive_pairs", None)
        item.pop("assertive_pairs", None)
    return out


def write_role_csv(path_csv: str, rows: List[Dict[str, Any]]) -> None:
    keys = ["id", "candidate_patch_idx", "candidate_rank", "role_label"]
    with open(path_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})


def make_random_patch_role_csv(
    src_role_csv: str,
    out_csv: str,
    supportive_topk: int,
    assertive_topk: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    cat = load_role_catalog(src_role_csv)
    rows: List[Dict[str, Any]] = []
    n_sid = 0
    n_sup = 0
    n_ass = 0
    for sid, item in cat.items():
        pool = list(item.get("pool") or [])
        if len(pool) == 0:
            continue
        n_sid += 1
        sup_ranked = list(item.get("supportive_ranked") or [])
        ass_ranked = list(item.get("assertive_ranked") or [])
        sup_n = len(sup_ranked)
        ass_n = len(ass_ranked)
        if int(supportive_topk) > 0:
            sup_n = min(sup_n, int(supportive_topk))
        if int(assertive_topk) > 0:
            ass_n = min(ass_n, int(assertive_topk))

        rng.shuffle(pool)
        sup_pick = pool[: min(sup_n, len(pool))]
        remaining = [p for p in pool if p not in set(sup_pick)]
        ass_pick = remaining[: min(ass_n, len(remaining))]
        if len(ass_pick) < ass_n:
            # allow overlap if pool is too small
            need = int(ass_n - len(ass_pick))
            cand = [p for p in pool if p not in set(ass_pick)]
            ass_pick.extend(cand[:need])

        for i, p in enumerate(sup_pick):
            rows.append(
                {
                    "id": sid,
                    "candidate_patch_idx": int(p),
                    "candidate_rank": int(i),
                    "role_label": "supportive",
                }
            )
            n_sup += 1
        for i, p in enumerate(ass_pick):
            rows.append(
                {
                    "id": sid,
                    "candidate_patch_idx": int(p),
                    "candidate_rank": int(i),
                    "role_label": "harmful",
                }
            )
            n_ass += 1

    write_role_csv(out_csv, rows)
    return {
        "n_samples": int(n_sid),
        "n_supportive_rows": int(n_sup),
        "n_assertive_rows": int(n_ass),
        "out_csv": out_csv,
    }


def _read_headset_json(path_json: str) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    with open(path_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    harmful: Dict[int, Set[int]] = {}
    faithful: Dict[int, Set[int]] = {}

    def _pull(val: Any) -> Dict[int, Set[int]]:
        out: Dict[int, Set[int]] = {}
        if isinstance(val, dict):
            for k, arr in val.items():
                li = _parse_int(k, None)
                if li is None:
                    continue
                if isinstance(arr, list):
                    for h in arr:
                        hi = _parse_int(h, None)
                        if hi is not None:
                            out.setdefault(int(li), set()).add(int(hi))
            return out
        if isinstance(val, list):
            for x in val:
                if isinstance(x, dict):
                    li = _parse_int(x.get("layer"), None)
                    hi = _parse_int(x.get("head"), None)
                    if li is not None and hi is not None:
                        out.setdefault(int(li), set()).add(int(hi))
            return out
        return out

    if isinstance(obj, dict):
        harmful = _pull(obj.get("harmful_heads"))
        faithful = _pull(obj.get("faithful_heads"))
    return harmful, faithful


def _write_headset_json(path_json: str, harmful: Dict[int, Set[int]], faithful: Dict[int, Set[int]], tag: str) -> None:
    harmful_list = []
    faithful_list = []
    for li in sorted(harmful.keys()):
        for hi in sorted(harmful[li]):
            harmful_list.append(
                {
                    "layer": int(li),
                    "head": int(hi),
                    "kind": str(tag),
                    "metric": str(tag),
                }
            )
    for li in sorted(faithful.keys()):
        for hi in sorted(faithful[li]):
            faithful_list.append(
                {
                    "layer": int(li),
                    "head": int(hi),
                    "kind": str(tag),
                    "metric": str(tag),
                }
            )
    obj = {
        "inputs": {"source": str(tag)},
        "counts": {"n_harmful": int(len(harmful_list)), "n_faithful": int(len(faithful_list))},
        "harmful_heads": harmful_list,
        "faithful_heads": faithful_list,
    }
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_random_headset_json(
    src_headset_json: str,
    out_json: str,
    n_heads: int,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    harmful_src, faithful_src = _read_headset_json(src_headset_json)
    harmful_new: Dict[int, Set[int]] = {}
    faithful_new: Dict[int, Set[int]] = {}
    for li, hs in harmful_src.items():
        k = len(hs)
        if k <= 0:
            continue
        k = min(int(k), int(n_heads))
        picks = rng.sample(range(int(n_heads)), k=k)
        harmful_new[int(li)] = set(int(x) for x in picks)
    for li, hs in faithful_src.items():
        k = len(hs)
        if k <= 0:
            continue
        k = min(int(k), int(n_heads))
        picks = rng.sample(range(int(n_heads)), k=k)
        faithful_new[int(li)] = set(int(x) for x in picks)
    _write_headset_json(out_json, harmful_new, faithful_new, tag="random_head_control")
    return {
        "n_harmful_layers": int(len(harmful_new)),
        "n_faithful_layers": int(len(faithful_new)),
        "n_harmful_total": int(sum(len(v) for v in harmful_new.values())),
        "n_faithful_total": int(sum(len(v) for v in faithful_new.values())),
        "out_json": out_json,
    }


def make_patch_only_headset_json(
    out_json: str,
    n_heads: int,
    late_start: int,
    late_end: int,
) -> Dict[str, Any]:
    lo = int(min(late_start, late_end))
    hi = int(max(late_start, late_end))
    all_heads = set(int(x) for x in range(int(n_heads)))
    harmful = {int(li): set(all_heads) for li in range(lo, hi + 1)}
    faithful = {int(li): set(all_heads) for li in range(lo, hi + 1)}
    _write_headset_json(out_json, harmful, faithful, tag="patch_only_all_heads")
    return {
        "n_layers": int(hi - lo + 1),
        "n_heads_per_layer": int(n_heads),
        "out_json": out_json,
    }


def build_variants(args: argparse.Namespace) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    controls = set(parse_controls(args.controls))
    variants: List[Dict[str, str]] = []
    if not bool(getattr(args, "skip_main_oracle", False)):
        variants.append(
            {
                "variant": "main_oracle",
                "control": "main_oracle",
                "role_csv": str(args.role_csv),
                "headset_json": str(args.headset_json),
            }
        )
    info: Dict[str, Any] = {"generated": {}}
    gen_dir = os.path.join(args.out_dir, "generated_controls")
    os.makedirs(gen_dir, exist_ok=True)

    if "random_patch_oracle" in controls:
        out_csv = os.path.join(gen_dir, "role_random_patch.csv")
        meta = make_random_patch_role_csv(
            src_role_csv=str(args.role_csv),
            out_csv=out_csv,
            supportive_topk=int(args.ais_oracle_supportive_topk),
            assertive_topk=int(args.ais_oracle_assertive_topk),
            seed=int(args.control_seed),
        )
        variants.append(
            {
                "variant": "random_patch_oracle",
                "control": "random_patch_oracle",
                "role_csv": out_csv,
                "headset_json": str(args.headset_json),
            }
        )
        info["generated"]["random_patch_oracle"] = meta

    if "random_head_oracle" in controls:
        out_json = os.path.join(gen_dir, "headset_random_head.json")
        meta = make_random_headset_json(
            src_headset_json=str(args.headset_json),
            out_json=out_json,
            n_heads=int(args.n_heads),
            seed=int(args.control_seed) + 1,
        )
        variants.append(
            {
                "variant": "random_head_oracle",
                "control": "random_head_oracle",
                "role_csv": str(args.role_csv),
                "headset_json": out_json,
            }
        )
        info["generated"]["random_head_oracle"] = meta

    if "patch_only_oracle" in controls:
        out_json = os.path.join(gen_dir, "headset_patch_only_all_heads.json")
        meta = make_patch_only_headset_json(
            out_json=out_json,
            n_heads=int(args.n_heads),
            late_start=int(args.ais_late_start),
            late_end=int(args.ais_late_end),
        )
        variants.append(
            {
                "variant": "patch_only_oracle",
                "control": "patch_only_oracle",
                "role_csv": str(args.role_csv),
                "headset_json": out_json,
            }
        )
        info["generated"]["patch_only_oracle"] = meta

    if "gt_region_guidance_oracle" in controls:
        p = str(args.gt_region_role_csv or "").strip()
        if p != "" and os.path.isfile(p):
            variants.append(
                {
                    "variant": "gt_region_guidance_oracle",
                    "control": "gt_region_guidance_oracle",
                    "role_csv": p,
                    "headset_json": str(args.headset_json),
                }
            )
            info["generated"]["gt_region_guidance_oracle"] = {"role_csv": p}
        else:
            info["generated"]["gt_region_guidance_oracle"] = {
                "skipped": True,
                "reason": "gt_region_role_csv_missing",
                "path": p,
            }

    return variants, info


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
    ap.add_argument("--role_csv", type=str, required=True)
    ap.add_argument("--conv_mode", type=str, default="llava_v1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--ais_late_start", type=int, default=16)
    ap.add_argument("--ais_late_end", type=int, default=24)
    ap.add_argument("--ais_oracle_supportive_topk", type=int, default=5)
    ap.add_argument("--ais_oracle_assertive_topk", type=int, default=5)
    ap.add_argument("--lambda_pos_list", type=str, default="0.25")
    ap.add_argument("--lambda_neg_list", type=str, default="0.25")
    ap.add_argument("--arms", type=str, default="harmful_only,faithful_only,bipolar")
    ap.add_argument("--debug_dump", action="store_true")
    ap.add_argument(
        "--controls",
        type=str,
        default="",
        help="Comma-separated controls: random_patch_oracle,random_head_oracle,patch_only_oracle,gt_region_guidance_oracle",
    )
    ap.add_argument("--gt_region_role_csv", type=str, default="")
    ap.add_argument("--control_seed", type=int, default=42)
    ap.add_argument("--n_heads", type=int, default=32)
    ap.add_argument("--skip_main_oracle", action="store_true")
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
        "lambda_pos": "",
        "lambda_neg": "",
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

    arms = [x.strip() for x in str(args.arms).split(",") if x.strip()]
    lambda_pos_list = parse_float_list(args.lambda_pos_list)
    lambda_neg_list = parse_float_list(args.lambda_neg_list)
    variants, variant_info = build_variants(args)

    for var in variants:
        variant_name = str(var["variant"])
        role_csv_use = str(var["role_csv"])
        headset_json_use = str(var["headset_json"])
        for arm in arms:
            for lp in lambda_pos_list:
                for ln in lambda_neg_list:
                    name = f"{variant_name}__oracle_{arm}_lp{lp:g}_ln{ln:g}"
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
                        "--ais-arm", str(arm),
                        "--ais-headset-json", headset_json_use,
                        "--ais-late-start", str(args.ais_late_start),
                        "--ais-late-end", str(args.ais_late_end),
                        "--ais-use-oracle-roles",
                        "--ais-oracle-role-csv", role_csv_use,
                        "--ais-oracle-supportive-topk", str(args.ais_oracle_supportive_topk),
                        "--ais-oracle-assertive-topk", str(args.ais_oracle_assertive_topk),
                        "--ais-oracle-lambda-pos", str(lp),
                        "--ais-oracle-lambda-neg", str(ln),
                        "--ais-oracle-bias-clip", "2.0",
                    ]
                    if bool(args.debug_dump):
                        cmd.extend(["--ais-debug-log", "--ais-debug-dump", dbg_csv])
                    run_cmd(cmd, cwd=args.repo_root)

                    pred = load_pred(out_jsonl)
                    met = eval_metrics(gt, pred)
                    cmpm = compare_to_baseline(gt, pred_base, pred)
                    dbg = summarize_debug(dbg_csv, late_start=args.ais_late_start, late_end=args.ais_late_end)

                    rows.append({
                        "name": name,
                        "variant": variant_name,
                        "control_type": str(var.get("control", variant_name)),
                        "role_csv_used": role_csv_use,
                        "headset_json_used": headset_json_use,
                        "arm": arm,
                        "lambda_pos": float(lp),
                        "lambda_neg": float(ln),
                        **met,
                        **cmpm,
                        **dbg,
                        "delta_acc_vs_base": float(met["acc"] - base_metrics["acc"]),
                        "delta_f1_vs_base": float(met["f1"] - base_metrics["f1"]),
                        "answers_file": out_jsonl,
                        "debug_file": dbg_csv if bool(args.debug_dump) else "",
                    })

    out_csv = os.path.join(args.out_dir, "oracle_headaware_ablation.csv")
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, None) for k in keys})

    cand = [r for r in rows if str(r.get("name")) != "baseline"]
    cand_ok = [r for r in cand if int(r.get("TN", 0)) > 0]
    if cand_ok:
        best = sorted(cand_ok, key=lambda r: (float(r.get("delta_acc_vs_base", 0.0)), float(r.get("delta_f1_vs_base", 0.0))), reverse=True)[0]
    elif cand:
        best = sorted(cand, key=lambda r: (float(r.get("delta_acc_vs_base", 0.0)), float(r.get("delta_f1_vs_base", 0.0))), reverse=True)[0]
    else:
        best = rows[0]

    summary = {
        "inputs": vars(args),
        "base_metrics": base_metrics,
        "n_runs": int(len(rows) - 1),
        "variants": variants,
        "variant_info": variant_info,
        "best": best,
        "outputs": {
            "table_csv": out_csv,
            "summary_json": os.path.join(args.out_dir, "summary.json"),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[saved]", out_csv)
    print("[saved]", os.path.join(args.out_dir, "summary.json"))
    print("[best]", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
