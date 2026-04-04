# Common Harm Miner Setup

## Summary

`VGA`, `VISTA`, `EAZY`는 단일 Python 환경으로 합치기 어렵다.

주요 충돌:
- `VGA`: Python 3.10, `torch==2.7.0`, `transformers==4.31.0`
- `VISTA`: Python 3.10, `torch==2.6.0+cu126`, `transformers==4.37.0`
- `EAZY`: Python 3.9, `torch==2.0.1`, 내부 `transformers-4.29.2` 포크

그래서 환경은 아래 4개로 나눈다.

- `model_base`
  - `LLaVA_calibration` 공통 스크립트, baseline generation, semantic feature extraction, table build, analysis
- `vga_base`
  - `/home/kms/VGA_origin`
- `vista_base`
  - `/home/kms/VISTA`
- `eazy_base`
  - `/home/kms/EAZY_origin`

## Server Env Create

### 1. `model_base`

```bash
conda create -n model_base -y python=3.10
conda activate model_base

cd ~/LLaVA_calibration
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install pandas pycocotools tqdm seaborn statsmodels protobuf sentencepiece
```

### 2. `vga_base`

```bash
conda create -n vga_base -y python=3.10
conda activate vga_base

cd ~/VGA_origin
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 3. `vista_base`

```bash
cd ~/VISTA
conda env create -n vista_base -f environment.yml
```

이미 env가 있으면:

```bash
cd ~/VISTA
conda env update -n vista_base -f environment.yml --prune
```

### 4. `eazy_base`

```bash
cd ~/EAZY_origin
python - <<'PY'
from pathlib import Path
src = Path("environment.yml")
dst = Path("/tmp/eazy_base.environment.yml")
text = src.read_text()
if "\n  - pytorch\n" not in text.split("dependencies:", 1)[0]:
    text = text.replace("channels:\n", "channels:\n  - pytorch\n", 1)
text = text.replace("      - sklearn==0.0.post5\n", "")
text = text.replace("      - spacy==3.7.0.dev0\n", "      - spacy==3.7.0\n")
dst.write_text(text)
print(dst)
PY

conda env create -n eazy_base -f /tmp/eazy_base.environment.yml
```

이미 env가 있으면:

```bash
cd ~/EAZY_origin
python - <<'PY'
from pathlib import Path
src = Path("environment.yml")
dst = Path("/tmp/eazy_base.environment.yml")
text = src.read_text()
if "\n  - pytorch\n" not in text.split("dependencies:", 1)[0]:
    text = text.replace("channels:\n", "channels:\n  - pytorch\n", 1)
text = text.replace("      - sklearn==0.0.post5\n", "")
text = text.replace("      - spacy==3.7.0.dev0\n", "      - spacy==3.7.0\n")
dst.write_text(text)
print(dst)
PY

conda env update -n eazy_base -f /tmp/eazy_base.environment.yml --prune
```

## Local Push

```bash
cd /Users/gangminseong/LAVA_for_hv

git add \
  scripts/build_discovery_caption_questions.py \
  scripts/run_vista_question_subset.py \
  scripts/run_eazy_question_subset.py \
  scripts/build_method_chair_table.py \
  scripts/build_method_harm_table.py \
  scripts/analyze_common_method_harm_miner.py \
  scripts/run_common_pope_harm_miner.sh \
  COMMON_HARM_MINER_SETUP.md

git commit -m "Add discovery common harm miner pipeline"
git push origin main
```

## Server Pull

```bash
cd ~/LLaVA_calibration
git stash push -u -m "server-local-before-pull"
git pull --ff-only origin main
```

## Main Experiment

이 wrapper는 기본으로:
- `POPE discovery`만 사용
- `discriminative`와 `generative` 둘 다 실행
- generative는 discovery image unique set에 대해 `"Please describe this image in detail."` prompt 사용

```bash
cd ~/LLaVA_calibration

CUDA_VISIBLE_DEVICES=5 \
VGA_ROOT=/home/kms/VGA_origin \
VISTA_ROOT=/home/kms/VISTA \
EAZY_ROOT=/home/kms/EAZY_origin \
CAL_PYTHON_BIN=/home/kms/miniconda3/envs/model_base/bin/python \
VISTA_PYTHON_BIN=/home/kms/miniconda3/envs/vista_base/bin/python \
EAZY_PYTHON_BIN=/home/kms/miniconda3/envs/eazy_base/bin/python \
VGA_ENV=vga_base \
IMAGE_FOLDER=/home/kms/data/images/mscoco/images/train2014 \
OUT_ROOT=/home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1 \
REUSE_IF_EXISTS=true \
bash scripts/run_common_pope_harm_miner.sh
```

## Important Outputs

### Discriminative

```bash
cat /home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1/discriminative/tables/vga_table.summary.json
cat /home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1/discriminative/tables/vista_table.summary.json
cat /home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1/discriminative/tables/eazy_table.summary.json
cat /home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1/discriminative/analysis/summary.json
```

### Generative

```bash
cat /home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1/generative/tables/vga_table.summary.json
cat /home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1/generative/tables/vista_table.summary.json
cat /home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1/generative/tables/eazy_table.summary.json
cat /home/kms/LLaVA_calibration/experiments/common_pope_discovery_harm_miner_v1/generative/analysis/summary.json
```

## Notes

- `run_common_pope_harm_miner.sh`는 discovery split을 기본으로 쓴다.
- discovery split 이미지는 `train2014`를 쓴다. `val2014`가 아니다.
- discriminative table은 `help / harm / both_correct / both_wrong / neutral`을 기록한다.
- generative table은 sample-level `CHAIRi` delta로:
  - `help`: intervention CHAIRi < baseline CHAIRi
  - `harm`: intervention CHAIRi > baseline CHAIRi
  - `neutral`: 같음
- 공통 harm miner analysis는 baseline-side semantic family만 사용한다:
  - `base_lp_content_mean`
  - `base_target_argmax_content_mean`
  - `base_target_gap_content_min`
  - `base_entropy_content_mean`
  - `base_conflict_lp_minus_entropy`
