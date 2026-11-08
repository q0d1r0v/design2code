#!/bin/bash
# Waits for training to finish, then evaluates the fine-tuned model and base Qwen (200 screens). Log: logs/eval.log
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; mkdir -p logs
until grep -q "DONE" logs/train.log; do
  grep -q "STOPPED" logs/train.log && { echo "$(date +%T) training stopped with an error, evaluation not started" >> logs/eval.log; exit 1; }
  sleep 30
done
sleep 20
for m in "sft5k --adapter runs/sft_5k/final" "qwen_base"; do
  set -- $m; name=$1; shift
  echo "$(date +%T) GEN $name started" >> logs/eval.log
  scripts/eval.sh gen --name $name --batch 8 "$@" >> logs/eval.log 2>&1 || { echo "$(date +%T) ERROR gen $name" >> logs/eval.log; continue; }
  echo "$(date +%T) SCORE $name" >> logs/eval.log
  scripts/eval.sh score --name $name >> logs/eval.log 2>&1 || echo "$(date +%T) ERROR score $name" >> logs/eval.log
done
echo "$(date +%T) EVALUATION DONE" >> logs/eval.log
