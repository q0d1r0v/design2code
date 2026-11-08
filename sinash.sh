#!/bin/bash
# O'qitilgan modelni sinash:  ./sinash.sh rasm.png [rasm2.jpg ...]
# Natija: natija/ papkasida (<nom>.kt va <nom>_solishtirish.png)
ROOT="$(cd "$(dirname "$0")" && pwd)"
P=$ROOT/tools/pydev/root/usr/include
export C_INCLUDE_PATH=$P/python3.14:$P/x86_64-linux-gnu/python3.14:$P
export TRITON_DEFAULT_FP_FUSION=1 PYTORCH_ALLOC_CONF=expandable_segments:True
[ $# -eq 0 ] && { echo "Ishlatish: $0 rasm.png [rasm2.jpg ...]"; exit 1; }
args=(); for f in "$@"; do [[ "$f" == -* ]] && args+=("$f") || args+=("$(realpath -m "$f")"); done
cd "$ROOT" && exec .venv-train/bin/python -I scripts/run_model.py "${args[@]}" 2> >(grep -v -i "warn\|causal_conv1d\|Loading weights\|fast path" >&2)
