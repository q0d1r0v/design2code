#!/bin/bash
# Runs the evaluation (scripts/eval_model.py) with the same environment as train.sh.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Python headers for the triton/fla kernel builds (setup.sh unpacks them into tools/pydev if the system has none)
V=$(.venv-train/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
P=$ROOT/tools/pydev/root/usr/include
export C_INCLUDE_PATH=$P/python$V:$P/x86_64-linux-gnu/python$V:$P
export TRITON_DEFAULT_FP_FUSION=1 PYTORCH_ALLOC_CONF=expandable_segments:True
exec .venv-train/bin/python -I scripts/eval_model.py "$@"
