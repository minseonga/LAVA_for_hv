#!/usr/bin/env bash
set -euo pipefail

# Strict RF-HAR oracle pipeline (no reuse of prior experiment artifacts).
# Steps:
# 1) build strict 1000 subset from full POPE
# 2) run vanilla baseline on that subset
# 3) build samples_csv from subset_gt + baseline_pred
# 4) extract fresh per-layer trace
# 5) derive fresh patch role labels
# 6) build token-level RF-HAR features C/A/D/B from role+trace
# 7) make strict calib/eval split
# 8) run RF-HAR
# 9) compare baseline vs RF-HAR (overall + split + fail IDs)

REPO_ROOT="${REPO_ROOT:-/home/kms/LLaVA_calibration}"
CONDA_ENV="${CONDA_ENV:-vocot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-6}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"

FULL_GT_CSV="${FULL_GT_CSV:-$REPO_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
FULL_Q_JSONL="${FULL_Q_JSONL:-$REPO_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
if [[ ! -f "$FULL_Q_JSONL" ]]; then
  FULL_Q_JSONL="$REPO_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl"
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/experiments/rfhar_oracle_strict_1000}"
N_TOTAL="${N_TOTAL:-1000}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"

# Trace / role settings
TARGET_LAYER="${TARGET_LAYER:-17}"
TOPK_LOCAL="${TOPK_LOCAL:-16}"
OBJECT_PATCH_TOPK="${OBJECT_PATCH_TOPK:-64}"

# RF-HAR settings
RFHAR_EARLY_START="${RFHAR_EARLY_START:-1}"
RFHAR_EARLY_END="${RFHAR_EARLY_END:-4}"
RFHAR_LATE_START="${RFHAR_LATE_START:-16}"
RFHAR_LATE_END="${RFHAR_LATE_END:-24}"
RFHAR_R_PERCENT="${RFHAR_R_PERCENT:-0.2}"
RFHAR_GAMMA="${RFHAR_GAMMA:-0.3}"
RFHAR_LAMBDA_PENALTY="${RFHAR_LAMBDA_PENALTY:-0.5}"
RFHAR_EPS="${RFHAR_EPS:-1e-6}"

if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR" 2>/dev/null || true)" && "$OVERWRITE" != "1" ]]; then
  echo "[error] OUT_DIR already exists and is not empty: $OUT_DIR"
  echo "        set OVERWRITE=1 or choose a new OUT_DIR"
  exit 2
fi

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

SUBSET_DIR="$OUT_DIR/01_subset"
BASELINE_DIR="$OUT_DIR/02_baseline"
TRACE_DIR="$OUT_DIR/03_trace"
ROLE_DIR="$OUT_DIR/04_role"
FEAT_DIR="$OUT_DIR/05_rfhar_feats"
SPLIT_DIR="$OUT_DIR/06_split"
RFHAR_DIR="$OUT_DIR/07_rfhar"
REPORT_DIR="$OUT_DIR/08_report"

mkdir -p "$SUBSET_DIR" "$BASELINE_DIR" "$TRACE_DIR" "$ROLE_DIR" "$FEAT_DIR" "$SPLIT_DIR" "$RFHAR_DIR" "$REPORT_DIR"

echo "[1/9] build strict subset"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/scripts/build_pope_strict_subset.py" \
  --full_gt_csv "$FULL_GT_CSV" \
  --full_q_jsonl "$FULL_Q_JSONL" \
  --out_dir "$SUBSET_DIR" \
  --n_total "$N_TOTAL" \
  --seed "$SEED" \
  --balance_category true \
  --balance_answer true

SUBSET_GT_CSV="$SUBSET_DIR/pope_strict_${N_TOTAL}_gt.csv"
SUBSET_Q_JSONL="$SUBSET_DIR/pope_strict_${N_TOTAL}_q.jsonl"
SUBSET_IDS_CSV="$SUBSET_DIR/pope_strict_${N_TOTAL}_ids.csv"

echo "[2/9] run vanilla baseline"
BASELINE_JSONL="$BASELINE_DIR/pred_baseline.jsonl"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" -m llava.eval.model_vqa_loader \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_ROOT" \
  --question-file "$SUBSET_Q_JSONL" \
  --answers-file "$BASELINE_JSONL" \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 8

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/scripts/eval_pope_subset_yesno.py" \
  --gt_csv "$SUBSET_GT_CSV" \
  --pred_jsonl "$BASELINE_JSONL" \
  --out_json "$BASELINE_DIR/metrics_baseline.json"

echo "[3/9] build samples_csv from strict subset + baseline pred"
SAMPLES_CSV="$BASELINE_DIR/samples_from_baseline.csv"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/scripts/build_pope_samples_from_gt_and_pred.py" \
  --subset_gt_csv "$SUBSET_GT_CSV" \
  --pred_jsonl "$BASELINE_JSONL" \
  --pred_text_key auto \
  --out_csv "$SAMPLES_CSV" \
  --out_summary "$BASELINE_DIR/samples_from_baseline_summary.json"

echo "[4/9] extract fresh all-layer trace"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/analyze_pope_visual_disconnect.py" \
  --samples_csv "$SAMPLES_CSV" \
  --image_root "$IMAGE_ROOT" \
  --out_dir "$TRACE_DIR" \
  --model_path "$MODEL_PATH" \
  --topk_local "$TOPK_LOCAL" \
  --object_patch_topk "$OBJECT_PATCH_TOPK" \
  --hidden_layer_idx -1 \
  --attn_layer_idx -1 \
  --save_layer_trace \
  --control_modes none \
  --bootstrap 0 \
  --num_samples "$N_TOTAL" \
  --seed "$SEED"

TRACE_CSV="$TRACE_DIR/per_layer_yes_trace.csv"

echo "[5/9] derive fresh patch role labels"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/analyze_pope_patch_role_fast.py" \
  --samples_csv "$SAMPLES_CSV" \
  --per_layer_trace_csv "$TRACE_CSV" \
  --image_root "$IMAGE_ROOT" \
  --out_dir "$ROLE_DIR" \
  --model_path "$MODEL_PATH" \
  --target_layer "$TARGET_LAYER" \
  --target_groups fp_hall,tp_yes \
  --top_n_per_group 0 \
  --sort_metric yes_sim_objpatch_max \
  --sort_desc true \
  --keep_k 5 \
  --candidate_topn 32 \
  --candidate_mode hybrid \
  --hybrid_topm 16 \
  --candidate_pool valid \
  --object_patch_topk "$OBJECT_PATCH_TOPK" \
  --exclude_padding_patches true \
  --mask_mode black \
  --batch_candidates 8 \
  --role_eps 0.05 \
  --seed "$SEED"

ROLE_CSV="$ROLE_DIR/per_patch_role_effect.csv"

echo "[6/9] build strict RF-HAR token features (C/A/D/B)"
RFHAR_FEATS_JSONL="$FEAT_DIR/rfhar_feats_oracle.jsonl"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/scripts/build_rfhar_feats_from_role_and_trace.py" \
  --role_csv "$ROLE_CSV" \
  --trace_csv "$TRACE_CSV" \
  --ids_csv "$SUBSET_IDS_CSV" \
  --out_jsonl "$RFHAR_FEATS_JSONL" \
  --out_summary "$FEAT_DIR/summary.json" \
  --k_img 576 \
  --a_layer "$TARGET_LAYER" \
  --early_start "$RFHAR_EARLY_START" \
  --early_end "$RFHAR_EARLY_END" \
  --late_start "$RFHAR_LATE_START" \
  --late_end "$RFHAR_LATE_END"

echo "[7/9] make strict calib/eval split"
SPLIT_CSV="$SPLIT_DIR/id_split.csv"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/scripts/make_id_split.py" \
  --subset_ids_csv "$SUBSET_IDS_CSV" \
  --calib_ratio 0.5 \
  --seed "$SEED" \
  --out_csv "$SPLIT_CSV"

echo "[8/9] run RF-HAR"
RFHAR_JSONL="$RFHAR_DIR/pred_rfhar.jsonl"
RFHAR_DEBUG_CSV="$RFHAR_DIR/rfhar_debug.csv"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" -m llava.eval.model_vqa_loader \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_ROOT" \
  --question-file "$SUBSET_Q_JSONL" \
  --answers-file "$RFHAR_JSONL" \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 8 \
  --enable-rfhar \
  --rfhar-feats-json "$RFHAR_FEATS_JSONL" \
  --rfhar-early-start "$RFHAR_EARLY_START" \
  --rfhar-early-end "$RFHAR_EARLY_END" \
  --rfhar-late-start "$RFHAR_LATE_START" \
  --rfhar-late-end "$RFHAR_LATE_END" \
  --rfhar-r-percent "$RFHAR_R_PERCENT" \
  --rfhar-gamma "$RFHAR_GAMMA" \
  --rfhar-lambda-penalty "$RFHAR_LAMBDA_PENALTY" \
  --rfhar-eps "$RFHAR_EPS" \
  --rfhar-debug-log \
  --ais-debug-log \
  --ais-debug-dump "$RFHAR_DEBUG_CSV"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/scripts/eval_pope_subset_yesno.py" \
  --gt_csv "$SUBSET_GT_CSV" \
  --pred_jsonl "$RFHAR_JSONL" \
  --out_json "$RFHAR_DIR/metrics_rfhar.json"

echo "[9/9] compare baseline vs RF-HAR (overall + split + fail IDs)"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_pope_yesno_runs.py" \
  --gt_csv "$SUBSET_GT_CSV" \
  --base_pred_jsonl "$BASELINE_JSONL" \
  --new_pred_jsonl "$RFHAR_JSONL" \
  --pred_text_key auto \
  --id_split_csv "$SPLIT_CSV" \
  --out_json "$REPORT_DIR/compare_baseline_vs_rfhar.json" \
  --out_fail_csv "$REPORT_DIR/fail_cases_baseline_vs_rfhar.csv"

cat <<EOF
[done] strict RF-HAR oracle pipeline completed
[saved] subset_gt:      $SUBSET_GT_CSV
[saved] subset_q:       $SUBSET_Q_JSONL
[saved] baseline_pred:  $BASELINE_JSONL
[saved] trace_csv:      $TRACE_CSV
[saved] role_csv:       $ROLE_CSV
[saved] rfhar_feats:    $RFHAR_FEATS_JSONL
[saved] split_csv:      $SPLIT_CSV
[saved] rfhar_pred:     $RFHAR_JSONL
[saved] compare_json:   $REPORT_DIR/compare_baseline_vs_rfhar.json
[saved] fail_case_csv:  $REPORT_DIR/fail_cases_baseline_vs_rfhar.csv
EOF
