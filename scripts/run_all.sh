#!/bin/bash
# Step 3: convert the 81,048 clean screens to code and render them with 4 parallel workers.
# Two halves: A = 0..40500, B = 40500..81048. In each half one worker starts at the front, one at the back (--reverse).
# Finished batches are skipped when restarted.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; mkdir -p logs
for d in render1 render2 render3; do [ -d "$d" ] || { cp -r render "$d"; rm -rf "$d"/.gradle "$d"/.kotlin "$d"/screens/build; }; done
R=".venv-train/bin/python -I scripts/run_batch.py --batch-size 250"
$R --start 0     --count 40500 --workdir render            > logs/w0.log 2>&1 &
$R --start 0     --count 40500 --workdir render3 --reverse > logs/w3.log 2>&1 &
$R --start 40500 --count 40548 --workdir render1           > logs/w1.log 2>&1 &
$R --start 40500 --count 40548 --workdir render2 --reverse > logs/w2.log 2>&1 &
wait
echo "ALL DONE"; for f in logs/w0.log logs/w1.log logs/w2.log logs/w3.log; do tail -n 1 $f; done
