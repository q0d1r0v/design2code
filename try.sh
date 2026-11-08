#!/bin/bash
# Try the fine-tuned model:  ./try.sh screenshot.png [screenshot2.jpg ...]
# Results: output/<name>.kt and output/<name>_compare.png
ROOT="$(cd "$(dirname "$0")" && pwd)"
[ $# -eq 0 ] && { echo "Usage: $0 screenshot.png [screenshot2.jpg ...]"; exit 1; }
args=(); for f in "$@"; do [[ "$f" == -* ]] && args+=("$f") || args+=("$(realpath -m "$f")"); done
cd "$ROOT"
# Python headers for the triton/fla kernel builds (setup.sh unpacks them into tools/pydev if the system has none)
V=$(.venv-train/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
P=$ROOT/tools/pydev/root/usr/include
export C_INCLUDE_PATH=$P/python$V:$P/x86_64-linux-gnu/python$V:$P
export TRITON_DEFAULT_FP_FUSION=1 PYTORCH_ALLOC_CONF=expandable_segments:True
exec .venv-train/bin/python -I scripts/run_model.py "${args[@]}" 2> >(grep -v -i "warn\|causal_conv1d\|Loading weights\|fast path" >&2)
