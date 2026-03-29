#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
PYTHON_BIN="${PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
QUESTION_FILE="${QUESTION_FILE:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/01_subset/pope_strict_1000_q.jsonl}"
GT_CSV="${GT_CSV:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/01_subset/pope_strict_1000_gt.csv}"
IDS_CSV="${IDS_CSV:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/01_subset/pope_strict_1000_ids.csv}"

FRGG_FEATS_JSON="${FRGG_FEATS_JSON:-/home/kms/LLaVA_calibration/experiments/frgg_1000/frgg_feats.jsonl}"
RFHAR_ORACLE_FEATS_JSON="${RFHAR_ORACLE_FEATS_JSON:-/home/kms/LLaVA_calibration/experiments/rfhar_oracle_strict_1000/05_rfhar_feats/rfhar_feats_oracle.jsonl}"
HEADSET_JSON="${HEADSET_JSON:-/home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json}"

OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/frrs_supportive_oracle_vs_gtfree}"

# Strong default setting (based on previous hard run)
FRRS_LATE_START="${FRRS_LATE_START:-16}"
FRRS_LATE_END="${FRRS_LATE_END:-24}"
FRRS_HEAD_MODE="${FRRS_HEAD_MODE:-dynamic}"
FRRS_R_PERCENT="${FRRS_R_PERCENT:-0.2}"
FRRS_ALPHA="${FRRS_ALPHA:-20}"
FRRS_TAU_C="${FRRS_TAU_C:-0.0}"
FRRS_TAU_E="${FRRS_TAU_E:-0.0}"
FRRS_K_C="${FRRS_K_C:-0.0}"
FRRS_K_E="${FRRS_K_E:-0.0}"
FRRS_TOPK_RATIO="${FRRS_TOPK_RATIO:-0.2}"

CONV_MODE="${CONV_MODE:-llava_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

run_case() {
  local CASE_NAME="$1"
  local RFHAR_FEATS="$2"
  local ONLINE_RECOMPUTE="$3"
  local ONLINE_BLEND="$4"

  local OUT_DIR="${OUT_ROOT}/${CASE_NAME}"
  mkdir -p "${OUT_DIR}"

  echo "[case] ${CASE_NAME}"
  echo "[case] rfhar_feats='${RFHAR_FEATS}' online_recompute=${ONLINE_RECOMPUTE} online_blend=${ONLINE_BLEND}"

  if [[ -n "${RFHAR_FEATS}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_frrs_feats_from_existing.py" \
      --frgg_feats_json "${FRGG_FEATS_JSON}" \
      --rfhar_feats_json "${RFHAR_FEATS}" \
      --ids_csv "${IDS_CSV}" \
      --out_json "${OUT_DIR}/frrs_feats.jsonl"
  else
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_frrs_feats_from_existing.py" \
      --frgg_feats_json "${FRGG_FEATS_JSON}" \
      --ids_csv "${IDS_CSV}" \
      --out_json "${OUT_DIR}/frrs_feats.jsonl"
  fi

  "${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --image-folder "${IMAGE_FOLDER}" \
    --question-file "${QUESTION_FILE}" \
    --answers-file "${OUT_DIR}/baseline.jsonl" \
    --conv-mode "${CONV_MODE}" \
    --temperature "${TEMPERATURE}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --dump-first-token-logits "${OUT_DIR}/baseline_first_token.csv"

  local ONLINE_FLAG=""
  if [[ "${ONLINE_RECOMPUTE}" == "1" ]]; then
    ONLINE_FLAG="--frrs-online-recompute-feats"
  fi

  "${PYTHON_BIN}" -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --image-folder "${IMAGE_FOLDER}" \
    --question-file "${QUESTION_FILE}" \
    --answers-file "${OUT_DIR}/frrs_supportive.jsonl" \
    --conv-mode "${CONV_MODE}" \
    --temperature "${TEMPERATURE}" \
    --num_beams "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --enable-frrs \
    --frrs-feats-json "${OUT_DIR}/frrs_feats.jsonl" \
    --frrs-arm supportive \
    --frrs-late-start "${FRRS_LATE_START}" \
    --frrs-late-end "${FRRS_LATE_END}" \
    --frrs-alpha "${FRRS_ALPHA}" \
    --frrs-beta 0.0 \
    --frrs-tau-c "${FRRS_TAU_C}" \
    --frrs-tau-e "${FRRS_TAU_E}" \
    --frrs-k-c "${FRRS_K_C}" \
    --frrs-k-e "${FRRS_K_E}" \
    --frrs-topk-ratio "${FRRS_TOPK_RATIO}" \
    --frrs-head-mode "${FRRS_HEAD_MODE}" \
    --frrs-r-percent "${FRRS_R_PERCENT}" \
    ${ONLINE_FLAG} \
    --frrs-online-blend "${ONLINE_BLEND}" \
    --ais-headset-json "${HEADSET_JSON}" \
    --frrs-debug-log \
    --ais-debug-dump "${OUT_DIR}/frrs_supportive_debug.csv" \
    --dump-first-token-logits "${OUT_DIR}/frrs_supportive_first_token.csv" \
    --first-token-logits-include-pre

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "${GT_CSV}" \
    --pred_jsonl "${OUT_DIR}/baseline.jsonl" \
    --out_json "${OUT_DIR}/metrics_baseline.json"

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "${GT_CSV}" \
    --pred_jsonl "${OUT_DIR}/frrs_supportive.jsonl" \
    --out_json "${OUT_DIR}/metrics_frrs_supportive.json"

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compare_pope_yesno_runs.py" \
    --gt_csv "${GT_CSV}" \
    --base_pred_jsonl "${OUT_DIR}/baseline.jsonl" \
    --new_pred_jsonl "${OUT_DIR}/frrs_supportive.jsonl" \
    --out_json "${OUT_DIR}/compare_baseline_vs_frrs_supportive.json" \
    --out_fail_csv "${OUT_DIR}/fail_cases_baseline_vs_frrs_supportive.csv"

  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/compare_first_token_between_runs.py" \
    --base_csv "${OUT_DIR}/baseline_first_token.csv" \
    --new_csv "${OUT_DIR}/frrs_supportive_first_token.csv" \
    --base_pred_jsonl "${OUT_DIR}/baseline.jsonl" \
    --new_pred_jsonl "${OUT_DIR}/frrs_supportive.jsonl" \
    --out_json "${OUT_DIR}/compare_first_token_baseline_vs_frrs_supportive.json" \
    --out_rows_csv "${OUT_DIR}/compare_first_token_baseline_vs_frrs_supportive_rows.csv"
}

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}"

# 1) Oracle-informed upper-bound / feasibility
run_case "oracle_informed" "${RFHAR_ORACLE_FEATS_JSON}" 0 0.0

# 2) GT-free practical (online recompute)
run_case "gt_free" "" 1 1.0

"${PYTHON_BIN}" - <<'PY'
import json
import os
import pandas as pd
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for name in ["oracle_informed", "gt_free"]:
    d = root / name
    b = json.loads((d / "metrics_baseline.json").read_text())["metrics"]
    n = json.loads((d / "metrics_frrs_supportive.json").read_text())["metrics"]
    c = json.loads((d / "compare_baseline_vs_frrs_supportive.json").read_text())["overall"]
    ft = json.loads((d / "compare_first_token_baseline_vs_frrs_supportive.json").read_text())
    dbg = pd.read_csv(d / "frrs_supportive_debug.csv")
    rows.append({
        "case": name,
        "base_acc": b["acc"],
        "new_acc": n["acc"],
        "delta_acc": c["delta"]["acc"],
        "base_f1": b["f1"],
        "new_f1": n["f1"],
        "delta_f1": c["delta"]["f1"],
        "changed_pred": c["changes"]["changed_pred"],
        "gain": c["changes"]["gain"],
        "harm": c["changes"]["harm"],
        "net_gain": c["changes"]["net_gain"],
        "first_token_margin_mean": ft["delta"]["margin_mean"],
        "first_token_abs_margin_mean": ft["delta"]["abs_margin_mean"],
        "frrs_gate_mean": float(dbg["frrs_gate_mean"].mean()),
        "frrs_delta_abs_mean": float(dbg["frrs_delta_abs_mean"].mean()),
        "frrs_dynamic_used_mean": float(dbg["frrs_dynamic_used"].mean()) if "frrs_dynamic_used" in dbg.columns else 0.0,
        "frrs_online_feat_used_mean": float(dbg["frrs_online_feat_used"].mean()) if "frrs_online_feat_used" in dbg.columns else 0.0,
    })

df = pd.DataFrame(rows)
out_csv = root / "summary_oracle_vs_gtfree.csv"
out_json = root / "summary_oracle_vs_gtfree.json"
df.to_csv(out_csv, index=False)
out_json.write_text(json.dumps({"rows": rows}, indent=2))
print(f"[saved] {out_csv}")
print(f"[saved] {out_json}")
PY

echo "[done] ${OUT_ROOT}"
