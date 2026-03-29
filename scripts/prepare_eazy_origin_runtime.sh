#!/usr/bin/env bash
set -euo pipefail

EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy/bin/python}"
POPE_COCO_FALLBACK="${POPE_COCO_FALLBACK:-/home/kms/VISTA/pope_coco}"
RUNTIME_SHIM_ROOT="${RUNTIME_SHIM_ROOT:-/tmp/eazy_origin_runtime_shim}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-$EAZY_ROOT/nltk_data}"
DOWNLOAD_NLTK="${DOWNLOAD_NLTK:-1}"

if [[ ! -d "$EAZY_ROOT" ]]; then
  echo "[error] EAZY_ROOT not found: $EAZY_ROOT" >&2
  exit 1
fi

if [[ ! -x "$EAZY_PYTHON_BIN" ]]; then
  echo "[error] EAZY_PYTHON_BIN not executable: $EAZY_PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$RUNTIME_SHIM_ROOT/Projects/LVLM_hallucination"
printf '' > "$RUNTIME_SHIM_ROOT/Projects/__init__.py"
printf '' > "$RUNTIME_SHIM_ROOT/Projects/LVLM_hallucination/__init__.py"
printf 'from utils.pope_loader import POPEDataSet\n' > "$RUNTIME_SHIM_ROOT/Projects/LVLM_hallucination/pope_loader.py"
echo "[ready] runtime shim: $RUNTIME_SHIM_ROOT/Projects/LVLM_hallucination/pope_loader.py"

if [[ ! -d "$EAZY_ROOT/pope_coco" ]]; then
  if [[ -d "$POPE_COCO_FALLBACK" ]]; then
    ln -sfn "$POPE_COCO_FALLBACK" "$EAZY_ROOT/pope_coco"
    echo "[linked] $EAZY_ROOT/pope_coco -> $POPE_COCO_FALLBACK"
  else
    echo "[error] pope_coco fallback not found: $POPE_COCO_FALLBACK" >&2
    exit 1
  fi
fi

if [[ ! -f "$EAZY_ROOT/chair_coco/synonyms_txt.txt" ]]; then
  EAZY_ROOT="$EAZY_ROOT" "$EAZY_PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import re

root = Path(os.environ["EAZY_ROOT"])
chair_py = root / "eval_script" / "chair.py"
text = chair_py.read_text(encoding="utf-8")
match = re.search(r"synonyms_txt = '''\n(.*?)\n'''", text, re.S)
if match is None:
    raise SystemExit(f"failed to extract synonyms_txt from {chair_py}")
out_path = root / "chair_coco" / "synonyms_txt.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(match.group(1).strip() + "\n", encoding="utf-8")
print(out_path)
PY
  echo "[ready] chair synonyms: $EAZY_ROOT/chair_coco/synonyms_txt.txt"
fi

if [[ "$DOWNLOAD_NLTK" == "1" ]]; then
  mkdir -p "$NLTK_DATA_DIR"
  if ! PYTHONPATH="$RUNTIME_SHIM_ROOT:$EAZY_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
       NLTK_DATA="$NLTK_DATA_DIR${NLTK_DATA:+:$NLTK_DATA}" \
       "$EAZY_PYTHON_BIN" - <<'PY'
import nltk
import sys

needed = {
    "corpora/wordnet": "wordnet",
    "tokenizers/punkt": "punkt",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
}
missing = []
for path, name in needed.items():
    try:
        nltk.data.find(path)
    except LookupError:
        missing.append(name)
if missing:
    print("missing=" + ",".join(missing))
    sys.exit(1)
print("nltk_ok")
PY
  then
    PYTHONPATH="$RUNTIME_SHIM_ROOT:$EAZY_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    NLTK_DATA="$NLTK_DATA_DIR${NLTK_DATA:+:$NLTK_DATA}" \
    "$EAZY_PYTHON_BIN" -m nltk.downloader -d "$NLTK_DATA_DIR" \
      wordnet punkt averaged_perceptron_tagger
  fi
fi

echo "[done] EAZY origin runtime is prepared."
echo "[hint] export PYTHONPATH=\"$RUNTIME_SHIM_ROOT:$EAZY_ROOT:\$PYTHONPATH\""
echo "[hint] export NLTK_DATA=\"$NLTK_DATA_DIR:\$NLTK_DATA\""
