#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"

SAMPLES_CSV="${SAMPLES_CSV:-$REPO_ROOT/experiments/pope_fragility_1000_greedy_b1/per_sample.csv}"
SCORED_CSV="${SCORED_CSV:-$REPO_ROOT/experiments/pope_role_surrogate_v2/merged_role_feature_scored.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
CONV_MODE="${CONV_MODE:-llava_v1}"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/experiments/pope_role_surrogate_replay_v1}"

TARGET_GROUPS="${TARGET_GROUPS:-fp_hall,tp_yes}"
USE_SPLIT="${USE_SPLIT:-val}"

ASSERTIVE_RATIO="${ASSERTIVE_RATIO:-0.2}"
SUPPORTIVE_RATIO="${SUPPORTIVE_RATIO:-0.2}"
HARMFUL_GATE_QUANTILE="${HARMFUL_GATE_QUANTILE:-0.5}"
FAITHFUL_GATE_QUANTILE="${FAITHFUL_GATE_QUANTILE:-0.5}"
PROTECT_SUPPORTIVE="${PROTECT_SUPPORTIVE:-true}"
EXCLUDE_PADDING_PATCHES="${EXCLUDE_PADDING_PATCHES:-true}"
MASK_MODE="${MASK_MODE:-black}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
NUM_BEAMS="${NUM_BEAMS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"

mkdir -p "${OUT_ROOT}"
cd "${REPO_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

for OP in patch_only harmful_head_aware bipolar_head_aware; do
  OUT_DIR="${OUT_ROOT}/${OP}"
  mkdir -p "${OUT_DIR}"
  echo "[run] operator=${OP} -> ${OUT_DIR}"
  python eval_pope_role_surrogate_replay.py \
    --samples_csv "${SAMPLES_CSV}" \
    --scored_csv "${SCORED_CSV}" \
    --image_root "${IMAGE_ROOT}" \
    --out_dir "${OUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --conv_mode "${CONV_MODE}" \
    --target_groups "${TARGET_GROUPS}" \
    --use_split "${USE_SPLIT}" \
    --operator "${OP}" \
    --assertive_ratio "${ASSERTIVE_RATIO}" \
    --supportive_ratio "${SUPPORTIVE_RATIO}" \
    --harmful_gate_quantile "${HARMFUL_GATE_QUANTILE}" \
    --faithful_gate_quantile "${FAITHFUL_GATE_QUANTILE}" \
    --protect_supportive "${PROTECT_SUPPORTIVE}" \
    --exclude_padding_patches "${EXCLUDE_PADDING_PATCHES}" \
    --mask_mode "${MASK_MODE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_beams "${NUM_BEAMS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}"
done

python - << 'PY'
import json
import os

out_root = os.environ.get("OUT_ROOT", "/home/kms/LLaVA_calibration/experiments/pope_role_surrogate_replay_v1")
ops = ["patch_only", "harmful_head_aware", "bipolar_head_aware"]
rows = []
for op in ops:
    p = os.path.join(out_root, op, "summary.json")
    if not os.path.isfile(p):
        continue
    s = json.load(open(p, "r", encoding="utf-8"))
    rows.append({
        "operator": op,
        "n": s.get("counts", {}).get("n_valid"),
        "base_acc": s.get("metrics", {}).get("base_acc"),
        "new_acc": s.get("metrics", {}).get("new_acc"),
        "delta_acc": s.get("metrics", {}).get("delta_acc"),
        "changed_pred": s.get("counts", {}).get("n_changed_pred"),
        "gain": s.get("counts", {}).get("gain"),
        "harm": s.get("counts", {}).get("harm"),
        "net_gain": s.get("counts", {}).get("net_gain"),
        "base": s.get("confusion_base", {}),
        "new": s.get("confusion_new", {}),
    })

summary_path = os.path.join(out_root, "operator_compare.json")
json.dump({"rows": rows}, open(summary_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("[saved]", summary_path)
PY

echo "[done] ${OUT_ROOT}"

