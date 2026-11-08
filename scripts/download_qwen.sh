#!/bin/bash
# Downloads the base Qwen3.5-4B weights from Hugging Face (~9.3 GB). If interrupted, run it again to resume.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/models/Qwen3.5-4B"; mkdir -p "$D"
B=https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main
FILES="config.json chat_template.jinja merges.txt vocab.json tokenizer.json tokenizer_config.json preprocessor_config.json video_preprocessor_config.json model.safetensors.index.json LICENSE model.safetensors-00001-of-00002.safetensors model.safetensors-00002-of-00002.safetensors"
for f in $FILES; do
  want=$(curl -sIL "$B/$f" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
  have=$(stat -c %s "$D/$f" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$want" = "$have" ]; then echo "  present: $f"; continue; fi
  echo "  downloading: $f"
  for try in $(seq 1 20); do
    curl -fSL -C - --retry 5 --retry-delay 5 -o "$D/$f" "$B/$f" && break
    echo "  interrupted, retry $try"; sleep 10
  done
  have=$(stat -c %s "$D/$f" 2>/dev/null || echo 0)
  [ -z "$want" ] || [ "$want" = "$have" ] || { echo "ERROR: $f was not fully downloaded"; exit 1; }
done
echo "Qwen3.5-4B ready: $D"
