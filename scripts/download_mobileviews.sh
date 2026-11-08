#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# MobileViews 400000-522301 parquet (15.85 GB) ni curl bilan yuklaydi.
# To'xtab qolsa, qayta ishga tushirilganda qolgan joyidan davom etadi. Har 30 s da foizni yozadi.
DIR=$ROOT/data/raw/mobileviews/MobileViews_Screenshots_ViewHierarchies/Parquets
OUT=$DIR/MobileViews_400000-522301.parquet
URL=https://huggingface.co/datasets/mllmTeam/MobileViews/resolve/main/MobileViews_Screenshots_ViewHierarchies/Parquets/MobileViews_400000-522301.parquet
TOTAL=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
echo "Hajmi: $TOTAL bayt"
mkdir -p "$DIR"
curl -sL -C - --retry 10 --retry-delay 5 -o "$OUT" "$URL" &
PID=$!
while kill -0 $PID 2>/dev/null; do
  sleep 30
  SZ=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S)  $(awk -v s=$SZ -v t=$TOTAL 'BEGIN{printf "%.1f%%  (%.2f / %.2f GB)", s/t*100, s/1e9, t/1e9}')"
done
wait $PID; RC=$?
SZ=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
echo "TUGADI rc=$RC  hajm=$SZ / $TOTAL"
