#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# 3-qadam: 81 048 ta toza ekranni 4 ta parallel ishchida kodga aylantirish va rasmga olish.
# Ikki qism: A = 0..40500, B = 40500..81048. Har qismda bitta ishchi boshidan, bittasi oxiridan (--reverse) yuradi.
# Qayta ishga tushirilsa, tayyor guruhlar o'tkazib yuboriladi.
cd "$ROOT"
for d in render1 render2 render3; do [ -d "$d" ] || { cp -r render "$d"; rm -rf "$d"/.gradle "$d"/.kotlin "$d"/screens/build; }; done
R=".venv-train/bin/python -I scripts/run_batch.py --batch-size 250"
$R --start 0     --count 40500 --workdir render            > logs/w0.log 2>&1 &
$R --start 0     --count 40500 --workdir render3 --reverse > logs/w3.log 2>&1 &
$R --start 40500 --count 40548 --workdir render1           > logs/w1.log 2>&1 &
$R --start 40500 --count 40548 --workdir render2 --reverse > logs/w2.log 2>&1 &
wait
echo "HAMMASI TUGADI"; for f in logs/w0.log logs/w1.log logs/w2.log logs/w3.log; do tail -n 1 $f; done
