#!/usr/bin/env bash
set -euo pipefail

# Minimal inference environment for the official LLaVA-NeXT repository.
# Do not install the full official requirements.txt: it contains internal
# packages that are not needed for POPE/CHAIR inference.

ENV_NAME="${ENV_NAME:-llava_next_official}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_BIN="${CONDA_BIN:-/home/kms/miniconda3/bin/conda}"
LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-/home/kms/LLaVA-NeXT}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export PYTHONDONTWRITEBYTECODE=1

if [[ ! -d "$LLAVA_NEXT_ROOT" ]]; then
  echo "[error] official LLaVA-NeXT repo not found: $LLAVA_NEXT_ROOT" >&2
  echo "        git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git $LLAVA_NEXT_ROOT" >&2
  exit 2
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "[reuse] conda env: $ENV_NAME"
fi

PY_BIN="$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"

"$PY_BIN" -m pip install --upgrade --no-cache-dir pip setuptools wheel
"$PY_BIN" -m pip install --no-cache-dir \
  torch==2.1.2 torchvision==0.16.2 \
  --index-url "$TORCH_INDEX_URL"

"$PY_BIN" -m pip install --no-cache-dir \
  "transformers @ git+https://github.com/huggingface/transformers.git@1c39974a4c4036fd641bc1191cc32799f85715a4" \
  tokenizers==0.15.2 \
  huggingface-hub==0.22.2 \
  accelerate==0.29.3 \
  safetensors \
  sentencepiece \
  protobuf==3.20.3 \
  numpy==1.26.4 \
  pillow==10.3.0 \
  tqdm \
  einops==0.6.1 \
  einops-exts==0.0.4 \
  shortuuid \
  timm \
  open-clip-torch==2.24.0

"$PY_BIN" -m pip install --no-cache-dir --no-deps -e "$LLAVA_NEXT_ROOT"

echo "[done] env=$ENV_NAME python=$PY_BIN"
echo "[check]"
"$PY_BIN" - <<'PY'
import torch, transformers, tokenizers, huggingface_hub
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
PY
