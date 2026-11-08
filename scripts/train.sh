#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# o'qitishni ishga tushirish (Python sarlavhalari triton/fla uchun)
P=$ROOT/tools/pydev/root/usr/include
export C_INCLUDE_PATH=$P/python3.14:$P/x86_64-linux-gnu/python3.14:$P
export TRITON_DEFAULT_FP_FUSION=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$ROOT" && exec .venv-train/bin/python -I scripts/train_sft.py "$@"
