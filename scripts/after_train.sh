#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# O'qitish tugashini kutadi, keyin imtihon: o'qitilgan model va asl Qwen (200 ekran). Log: logs/eval.log
cd "$ROOT"
until grep -q "TUGADI" logs/train.log; do
  grep -q "TO'XTADI" logs/train.log && { echo "$(date +%T) o'qitish xato bilan to'xtadi, imtihon boshlanmadi" >> logs/eval.log; exit 1; }
  sleep 30
done
sleep 20
for m in "sft5k --adapter runs/sft_5k/final" "qwen_base"; do
  set -- $m; name=$1; shift
  echo "$(date +%T) GEN $name boshlandi" >> logs/eval.log
  scripts/eval.sh gen --name $name --batch 8 "$@" >> logs/eval.log 2>&1 || { echo "$(date +%T) XATO gen $name" >> logs/eval.log; continue; }
  echo "$(date +%T) SCORE $name" >> logs/eval.log
  scripts/eval.sh score --name $name >> logs/eval.log 2>&1 || echo "$(date +%T) XATO score $name" >> logs/eval.log
done
echo "$(date +%T) IMTIHON TUGADI" >> logs/eval.log
