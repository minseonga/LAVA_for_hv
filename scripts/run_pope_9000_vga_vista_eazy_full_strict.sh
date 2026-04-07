#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Full 9000 strict pipeline:
# baseline + VGA + VISTA + EAZY
# plus: taxonomy(VV/FV/VF/FF), D1/D2, feature structure, A×C matrix
# ------------------------------------------------------------

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
VISTA_ROOT="${VISTA_ROOT:-/home/kms/VISTA}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
VISTA_PYTHON_BIN="${VISTA_PYTHON_BIN:-/home/kms/miniconda3/envs/vista/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
SEED="${SEED:-1994}"

# If user did not export CUDA_VISIBLE_DEVICES, follow GPU variable.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
Q_NOOBJ="${Q_NOOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl}"
Q_WITHOBJ="${Q_WITHOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"

FEATURES_CSV="${FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_e1_4_l16_24/features/features_unified_table.csv}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict}"
BASELINE_DIR="$OUT_ROOT/baseline"
VGA_DIR="$OUT_ROOT/vga"
VISTA_DIR="$OUT_ROOT/vista"
EAZY_DIR="$OUT_ROOT/eazy"

BASELINE_PRED_JSONL="$BASELINE_DIR/pred_vanilla_9000.jsonl"
BASELINE_METRICS_JSON="$BASELINE_DIR/metrics_vanilla_9000.json"

# run switches
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VGA="${RUN_VGA:-1}"
RUN_VISTA="${RUN_VISTA:-1}"
RUN_EAZY="${RUN_EAZY:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# VGA settings (with-object 9000)
VGA_CONDA_SH="${VGA_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VGA_ENV="${VGA_ENV:-vga}"
VGA_PRED_JSONL="$VGA_DIR/pred_vga_9000.jsonl"
VGA_METRICS_JSON="$VGA_DIR/metrics_vga_9000.json"
VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-8}"
VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"

# VISTA settings
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-pope_9000_vista_eval}"
VISTA_MODEL="${VISTA_MODEL:-llava-1.5}"

# EAZY settings
EAZY_MODEL="${EAZY_MODEL:-llava-1.5}"
EAZY_BEAM="${EAZY_BEAM:-1}"
EAZY_TOPK_K="${EAZY_TOPK_K:-2}"
EAZY_REQUIRE_CHAIR="${EAZY_REQUIRE_CHAIR:-1}"
EAZY_RUNNER_MODE="${EAZY_RUNNER_MODE:-stable}"

mkdir -p "$BASELINE_DIR" "$VGA_DIR" "$VISTA_DIR" "$EAZY_DIR"

if [[ ! -f "$CAL_PYTHON_BIN" ]]; then
  echo "[error] CAL_PYTHON_BIN not found: $CAL_PYTHON_BIN" >&2
  exit 1
fi

for f in "$Q_NOOBJ" "$Q_WITHOBJ" "$GT_CSV" "$FEATURES_CSV"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing file: $f" >&2
    exit 1
  fi
done

run_baseline_if_needed() {
  if [[ "$RUN_BASELINE" != "1" && -f "$BASELINE_PRED_JSONL" ]]; then
    echo "[reuse] baseline prediction: $BASELINE_PRED_JSONL"
  elif [[ -f "$BASELINE_PRED_JSONL" && "$FORCE_RERUN" != "1" ]]; then
    echo "[reuse] baseline prediction: $BASELINE_PRED_JSONL"
  else
    echo "[run] baseline vanilla llava (9000)"
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" -m llava.eval.model_vqa_loader \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$Q_NOOBJ" \
      --answers-file "$BASELINE_PRED_JSONL" \
      --conv-mode llava_v1 \
      --temperature 0 \
      --num_beams 1 \
      --max_new_tokens 8
  fi

  echo "[eval] baseline metrics"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$BASELINE_PRED_JSONL" \
    --pred_text_key text \
    --out_json "$BASELINE_METRICS_JSON"
}

post_analyze_model() {
  local model_name="$1"
  local pred_jsonl="$2"
  local pred_text_key="$3"
  local model_dir="$4"

  local taxonomy_dir="$model_dir/taxonomy"
  local d1d2_dir="$model_dir/d1d2_audit"
  local feature_dir="$model_dir/feature_structure"
  local matrix_dir="$model_dir/cue_usability_matrix"
  local metrics_json="$model_dir/metrics_${model_name}_9000.json"

  mkdir -p "$taxonomy_dir" "$d1d2_dir" "$feature_dir" "$matrix_dir"

  echo "[eval] ${model_name} metrics"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$pred_jsonl" \
    --pred_text_key "$pred_text_key" \
    --out_json "$metrics_json"

  echo "[taxonomy] baseline vs ${model_name}"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_vga_failure_taxonomy.py" \
    --gt_csv "$GT_CSV" \
    --baseline_pred_jsonl "$BASELINE_PRED_JSONL" \
    --vga_pred_jsonl "$pred_jsonl" \
    --baseline_pred_text_key text \
    --vga_pred_text_key "$pred_text_key" \
    --out_dir "$taxonomy_dir"

  echo "[d1d2] ${model_name}"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/analyze_yes_dampening_d1d2.py" \
    --per_case_csv "$taxonomy_dir/per_case_compare.csv" \
    --features_csv "$FEATURES_CSV" \
    --out_dir "$d1d2_dir"

  echo "[feature-structure] ${model_name}"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_model_feature_structure.py" \
    --features_csv "$FEATURES_CSV" \
    --per_case_csv "$taxonomy_dir/per_case_compare.csv" \
    --out_dir "$feature_dir" \
    --python_bin "$CAL_PYTHON_BIN" \
    --repo_root "$CAL_ROOT" \
    --split_filter all

  echo "[matrix] ${model_name}"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/plot_cue_availability_usability_matrix.py" \
    --features_csv "$FEATURES_CSV" \
    --subset_group_csv "$taxonomy_dir/per_case_compare.csv" \
    --id_col id \
    --group_col case_type \
    --a_feature obj_token_prob_lse \
    --c_feature faithful_minus_global_attn \
    --size_feature guidance_mismatch_score \
    --plot_mode pure \
    --balanced_repeats 10 \
    --balanced_groups VV,FV,VF,FF \
    --out_dir "$matrix_dir" \
    --seed 42
}

run_vga_full() {
  if [[ "$RUN_VGA" != "1" && -f "$VGA_PRED_JSONL" ]]; then
    echo "[reuse] VGA prediction: $VGA_PRED_JSONL"
  elif [[ -f "$VGA_PRED_JSONL" && "$FORCE_RERUN" != "1" ]]; then
    echo "[reuse] VGA prediction: $VGA_PRED_JSONL"
  else
    echo "[run] VGA with-object full-9000"
    if [[ ! -f "$VGA_CONDA_SH" ]]; then
      echo "[error] conda.sh not found: $VGA_CONDA_SH" >&2
      exit 1
    fi
    # shellcheck source=/dev/null
    source "$VGA_CONDA_SH"
    conda activate "$VGA_ENV"
    cd "$VGA_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" python eval/object_hallucination_vqa_llava.py \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$Q_WITHOBJ" \
      --answers-file "$VGA_PRED_JSONL" \
      --conv-mode "$VGA_CONV_MODE" \
      --max_gen_len "$VGA_MAX_GEN_LEN" \
      --use_add "$VGA_USE_ADD" \
      --attn_coef "$VGA_ATTN_COEF" \
      --head_balancing "$VGA_HEAD_BALANCING" \
      --sampling "$VGA_SAMPLING" \
      --cd_alpha "$VGA_CD_ALPHA" \
      --seed "$SEED" \
      --start_layer "$VGA_START_LAYER" \
      --end_layer "$VGA_END_LAYER"
    echo "[saved] $VGA_PRED_JSONL"
  fi

  post_analyze_model "vga" "$VGA_PRED_JSONL" "output" "$VGA_DIR"
}

run_vista_full() {
  if [[ "$RUN_VISTA" == "1" || ! -f "$VISTA_DIR/pred_vista_9000.jsonl" || "$FORCE_RERUN" == "1" ]]; then
    echo "[run] VISTA full-9000"
    GPU="$GPU" \
    CAL_ROOT="$CAL_ROOT" \
    VISTA_ROOT="$VISTA_ROOT" \
    VISTA_PYTHON_BIN="$VISTA_PYTHON_BIN" \
    CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
    DATA_PATH="$IMAGE_FOLDER" \
    LLAVA_MODEL_PATH="$MODEL_PATH" \
    GT_CSV="$GT_CSV" \
    BASELINE_JSONL="$BASELINE_PRED_JSONL" \
    FEATURES_CSV="$FEATURES_CSV" \
    OUT_DIR="$VISTA_DIR" \
    TAX_OUT_DIR="$VISTA_DIR/taxonomy" \
    D1D2_OUT_DIR="$VISTA_DIR/d1d2_audit" \
    EXP_FOLDER="$VISTA_EXP_FOLDER" \
    MODEL="$VISTA_MODEL" \
    SEED="$SEED" \
    ADV_SUBSET_SIZE=-1 \
    POP_SUBSET_SIZE=-1 \
    RND_SUBSET_SIZE=-1 \
    CATEGORY_SIZE=3000 \
    bash "$CAL_ROOT/scripts/run_vista_vs_baseline_taxonomy_9000.sh"
  else
    echo "[reuse] VISTA prediction: $VISTA_DIR/pred_vista_9000.jsonl"
  fi

  post_analyze_model "vista" "$VISTA_DIR/pred_vista_9000.jsonl" "output" "$VISTA_DIR"
}

run_eazy_full() {
  if [[ "$RUN_EAZY" == "1" || ! -f "$EAZY_DIR/pred_eazy_9000.jsonl" || "$FORCE_RERUN" == "1" ]]; then
    echo "[run] EAZY full-9000"
    if [[ "$EAZY_RUNNER_MODE" == "stable" ]]; then
      GPU="$GPU" \
      CAL_ROOT="$CAL_ROOT" \
      EAZY_ROOT="$EAZY_ROOT" \
      EAZY_PYTHON_BIN="$EAZY_PYTHON_BIN" \
      CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
      IMAGE_FOLDER="$IMAGE_FOLDER" \
      Q_WITHOBJ="$Q_WITHOBJ" \
      GT_CSV="$GT_CSV" \
      BASELINE_JSONL="$BASELINE_PRED_JSONL" \
      FEATURES_CSV="$FEATURES_CSV" \
      OUT_DIR="$EAZY_DIR" \
      TAX_OUT_DIR="$EAZY_DIR/taxonomy" \
      D1D2_OUT_DIR="$EAZY_DIR/d1d2_audit" \
      MODEL="$EAZY_MODEL" \
      BEAM="$EAZY_BEAM" \
      TOPK_K="$EAZY_TOPK_K" \
      MAX_NEW_TOKENS=8 \
      bash "$CAL_ROOT/scripts/run_eazy_vs_baseline_taxonomy_9000_stable.sh"
    else
      GPU="$GPU" \
      CAL_ROOT="$CAL_ROOT" \
      EAZY_ROOT="$EAZY_ROOT" \
      EAZY_PYTHON_BIN="$EAZY_PYTHON_BIN" \
      CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
      DATA_PATH="$IMAGE_FOLDER" \
      GT_CSV="$GT_CSV" \
      BASELINE_JSONL="$BASELINE_PRED_JSONL" \
      FEATURES_CSV="$FEATURES_CSV" \
      OUT_DIR="$EAZY_DIR" \
      TAX_OUT_DIR="$EAZY_DIR/taxonomy" \
      D1D2_OUT_DIR="$EAZY_DIR/d1d2_audit" \
      MODEL="$EAZY_MODEL" \
      BEAM="$EAZY_BEAM" \
      TOPK_K="$EAZY_TOPK_K" \
      EAZY_REQUIRE_CHAIR="$EAZY_REQUIRE_CHAIR" \
      bash "$CAL_ROOT/scripts/run_eazy_vs_baseline_taxonomy_9000.sh"
    fi
  else
    echo "[reuse] EAZY prediction: $EAZY_DIR/pred_eazy_9000.jsonl"
  fi

  post_analyze_model "eazy" "$EAZY_DIR/pred_eazy_9000.jsonl" "output" "$EAZY_DIR"
}

build_global_summary() {
  "$CAL_PYTHON_BIN" - <<'PY'
import json
import os

out_root = os.environ['OUT_ROOT']


def loadj(p):
    if not os.path.isfile(p):
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

models = {
    'baseline': os.path.join(out_root, 'baseline'),
    'vga': os.path.join(out_root, 'vga'),
    'vista': os.path.join(out_root, 'vista'),
    'eazy': os.path.join(out_root, 'eazy'),
}

summary = {'out_root': out_root, 'models': {}}
for m, d in models.items():
    info = {
        'metrics_json': os.path.join(d, f"metrics_{m}_9000.json") if m != 'baseline' else os.path.join(d, 'metrics_vanilla_9000.json'),
        'taxonomy_summary_json': os.path.join(d, 'taxonomy', 'summary.json'),
        'd1d2_summary_json': os.path.join(d, 'd1d2_audit', 'summary.json'),
        'feature_structure_summary_json': os.path.join(d, 'feature_structure', 'summary.json'),
        'matrix_summary_json': os.path.join(d, 'cue_usability_matrix', 'summary.json'),
    }
    summary['models'][m] = {
        'paths': info,
        'metrics': loadj(info['metrics_json']),
        'taxonomy': loadj(info['taxonomy_summary_json']),
        'd1d2': loadj(info['d1d2_summary_json']),
        'feature_structure': loadj(info['feature_structure_summary_json']),
        'matrix': loadj(info['matrix_summary_json']),
    }

outp = os.path.join(out_root, 'summary_all_models.json')
with open(outp, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print('[saved]', outp)
PY
}

echo "[start] full-9000 strict run: baseline + VGA + VISTA + EAZY"
run_baseline_if_needed
run_vga_full
run_vista_full
run_eazy_full
OUT_ROOT="$OUT_ROOT" build_global_summary

echo "[done] $OUT_ROOT"
echo "[summary] $OUT_ROOT/summary_all_models.json"
