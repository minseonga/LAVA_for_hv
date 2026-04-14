#!/usr/bin/env bash
set -euo pipefail

# Candidate-conditioned inventory proxy.
#
# Builds baseline-only content candidates from caption pairs, then asks one
# image-conditioned verifier question per sample:
#   "Which candidate terms are clearly visible?"
# This uses no CHAIR/COCO ontology at inference and avoids per-object forwards.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v62_candidate_inventory_proxy}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-64}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
SEED="${SEED:-17}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

MAX_CANDIDATES="${MAX_CANDIDATES:-16}"
CANDIDATE_MODE="${CANDIDATE_MODE:-token}"
VERIFIER_USE_ADD="${VERIFIER_USE_ADD:-false}"
VERIFIER_NAME="${VERIFIER_NAME:-candidate_inventory_vanilla}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/features"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

BASE_PRED="$SOURCE_OUT/$SPLIT/pred_baseline_caption.jsonl"
INT_PRED="$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl"
BASE_CHAIR="$SOURCE_OUT/$SPLIT/chair_baseline.json"
INT_CHAIR="$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json"
ORACLE_DIR="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg"
ORACLE_ROWS="$ORACLE_DIR/unique_safe_oracle_rows.csv"

Q_CAND="$OUT_ROOT/splits/${SPLIT}_candidate_inventory_q_limited${LIMIT}.jsonl"
Q_CAND_SUMMARY="$OUT_ROOT/splits/${SPLIT}_candidate_inventory_q.summary.json"
VERIFIER_PRED="$OUT_ROOT/$SPLIT/pred_${VERIFIER_NAME}.jsonl"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] verifier_use_add=$VERIFIER_USE_ADD target=$TARGET_COL max_candidates=$MAX_CANDIDATES mode=$CANDIDATE_MODE"

if [[ ! -f "$ORACLE_ROWS" ]]; then
  echo "[prep] build missing unique-safe oracle rows: $ORACLE_ROWS"
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/analyze_chair_unique_safe_oracle.py \
      --baseline_chair_json "$BASE_CHAIR" \
      --intervention_chair_json "$INT_CHAIR" \
      --out_dir "$ORACLE_DIR" \
      --main_oracle_col "$TARGET_COL"
  )
fi

if ! reuse_file "$Q_CAND"; then
  echo "[1/3] build candidate-conditioned inventory questions"
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_generative_candidate_inventory_questions.py \
      --baseline_pred_jsonl "$BASE_PRED" \
      --intervention_pred_jsonl "$INT_PRED" \
      --out_jsonl "$Q_CAND" \
      --out_summary_json "$Q_CAND_SUMMARY" \
      --max_candidates "$MAX_CANDIDATES" \
      --candidate_mode "$CANDIDATE_MODE" \
      --limit "$LIMIT"
  )
else
  echo "[reuse] $Q_CAND"
fi

if ! reuse_file "$VERIFIER_PRED"; then
  echo "[2/3] run candidate inventory verifier"
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_origin_llava_caption_compat.py \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$Q_CAND" \
      --answers-file "$VERIFIER_PRED" \
      --conv-mode "$VGA_CONV_MODE" \
      --max_gen_len "$VGA_MAX_GEN_LEN" \
      --use_add "$VERIFIER_USE_ADD" \
      --attn_coef "$VGA_ATTN_COEF" \
      --cd_alpha "$VGA_CD_ALPHA" \
      --start_layer "$VGA_START_LAYER" \
      --end_layer "$VGA_END_LAYER" \
      --head_balancing simg \
      --sampling "$VGA_SAMPLING" \
      --seed "$SEED"
  )
else
  echo "[reuse] $VERIFIER_PRED"
fi

echo "[3/3] analyze candidate inventory verifier features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_generative_candidate_inventory_proxy.py \
    --candidate_question_jsonl "$Q_CAND" \
    --verifier_pred_jsonl "$VERIFIER_PRED" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_csv "$OUT_ROOT/features/${SPLIT}_candidate_inventory_proxy_features.csv" \
    --out_feature_metrics_csv "$OUT_ROOT/features/${SPLIT}_candidate_inventory_proxy_feature_metrics.csv" \
    --out_summary_json "$OUT_ROOT/features/${SPLIT}_candidate_inventory_proxy_summary.json"
)

echo "[done] $OUT_ROOT"
