#!/usr/bin/env bash
# One-shot WP5 setup and run. Does not download textbook text.
# Required local files, which are not in git:
#   data/tokenizer/corpus_bpe.json
#   data/packed/class_6.npy
#   data/clean/class_6/chapter_*.txt
# Do not copy artifacts/wp5 from the CPU machine if you want a fresh GPU run.
set -euo pipefail
cd "$(dirname "$0")/.."

missing=0
if [[ ! -f data/tokenizer/corpus_bpe.json ]]; then
  echo "MISSING data/tokenizer/corpus_bpe.json"
  missing=1
fi
if [[ ! -f data/packed/class_6.npy ]]; then
  echo "MISSING data/packed/class_6.npy"
  missing=1
fi
if ! compgen -G "data/clean/class_6/chapter_*.txt" >/dev/null; then
  echo "MISSING data/clean/class_6/chapter_*.txt"
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "Copy those paths from the original machine. Do not commit them."
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install tiktoken tokenizers pyyaml pytest pymupdf numpy
if command -v nvidia-smi >/dev/null 2>&1; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || python -m pip install torch
else
  python -m pip install torch
fi
python - << 'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
python scripts/05_class6_debug.py
echo "PASTE THIS FILE: reports/wp5_gpu_paste.txt"
