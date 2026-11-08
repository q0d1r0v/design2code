#!/bin/bash
# Runs training and resumes from the last checkpoint after a crash (up to 5 times).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; mkdir -p logs
for i in 1 2 3 4 5; do
  scripts/train.sh --resume "$@" >> logs/train_full.out 2>&1 && exit 0
  echo "$(date +%T) ERROR, retry $i" >> logs/train.log; sleep 30
done
echo "$(date +%T) STOPPED: all 5 attempts failed" >> logs/train.log
