#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# xato bo'lsa checkpoint'dan davom etadi (5 martagacha)
cd "$ROOT"
for i in 1 2 3 4 5; do
  scripts/train.sh --resume "$@" >> logs/train_full.out 2>&1 && exit 0
  echo "$(date +%T) XATO, qayta urinish $i" >> logs/train.log; sleep 30
done
echo "$(date +%T) TO'XTADI: 5 urinish ham xato" >> logs/train.log
