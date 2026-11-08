#!/bin/bash
# design2code: hamma narsani o'rnatadi (sudo kerak emas).  Ishlatish:  ./setup.sh
#   1) Python muhiti (.venv-train)        ~6 GB
#   2) Java 17 (tools/)                   ~190 MB
#   3) Android SDK (agar yo'q bo'lsa)     ~200 MB
#   4) Qwen3.5-4B asosiy model            ~9.3 GB
#   5) O'qitilgan adapter (GitHub Release) ~120 MB
#   6) Render loyihasini sinab ko'rish (Gradle kutubxonalari) ~600 MB
# Qayta ishga tushirsa bo'ladi: tayyor qismlar o'tkazib yuboriladi.
# Sozlamalar (ixtiyoriy): PYTHON=python3.14  ANDROID_HOME=...  D2C_GRADLE_HOME=...  SKIP_MODEL=1
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
REPO_URL=https://github.com/q0d1r0v/design2code
ADAPTER_TAG=sft-5k-v1
PY=${PYTHON:-python3}
step() { echo; echo "=== $*"; }
die() { echo "XATO: $*" >&2; exit 1; }

step "0/6 Tekshiruv"
[ "$(uname -s)" = Linux ] || die "faqat Linux (x86_64) qo'llab-quvvatlanadi"
command -v curl >/dev/null || die "curl o'rnatilmagan"
command -v unzip >/dev/null || die "unzip o'rnatilmagan"
command -v "$PY" >/dev/null || die "$PY topilmadi (PYTHON=... bilan ko'rsating)"
PYV=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "Python 3.11+ kerak (sizda $PYV)"
nvidia-smi >/dev/null 2>&1 || echo "  DIQQAT: NVIDIA GPU topilmadi. Model faqat NVIDIA GPU (8 GB+) bilan ishlaydi."
echo "  Python $PYV, papka: $ROOT"

step "1/6 Python muhiti (.venv-train)"
[ -x .venv-train/bin/python ] || "$PY" -m venv .venv-train
.venv-train/bin/pip install -q --upgrade pip
.venv-train/bin/pip install -q -r requirements.txt
.venv-train/bin/python -I -c "import torch, transformers, peft; print('  torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
# triton (GPU yadrolari) Python.h ni talab qiladi. Tizimda bo'lmasa, sudo'siz paketdan ochib olinadi.
INC=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
if [ ! -f "$INC/Python.h" ] && [ ! -f "tools/pydev/root/usr/include/python$PYV/Python.h" ]; then
  echo "  Python.h yo'q -> python$PYV-dev paketidan olinadi"
  command -v apt-get >/dev/null || die "Python.h yo'q. 'python$PYV-dev' paketini o'rnating."
  mkdir -p tools/pydev && (cd tools/pydev && apt-get download "libpython$PYV-dev" "python$PYV-dev" >/dev/null \
    && for d in *.deb; do dpkg -x "$d" root; done && rm -f *.deb) || die "python$PYV-dev yuklanmadi"
fi

step "2/6 Java 17"
if ! ls -d tools/jdk-17* >/dev/null 2>&1; then
  mkdir -p tools
  curl -fsSL "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse" -o tools/jdk17.tar.gz
  tar -xzf tools/jdk17.tar.gz -C tools && rm tools/jdk17.tar.gz
fi
JAVA_HOME=$(ls -d "$ROOT"/tools/jdk-17* | head -1); export JAVA_HOME
echo "  $("$JAVA_HOME/bin/java" -version 2>&1 | head -1)"
# Paparazzi rasmini PNG qilib saqlash uchun java.awt/javax.imageio klasslari (faqat kompilyatsiyaga)
if [ ! -f tools/awtstub/awt-compile-stub.jar ]; then
  mkdir -p tools/awtstub && rm -rf tools/awtstub/x
  "$JAVA_HOME/bin/jimage" extract --dir tools/awtstub/x --include "regex:/java.desktop/(java/awt/.*|javax/imageio/.*)" "$JAVA_HOME/lib/modules"
  (cd tools/awtstub/x/java.desktop && "$JAVA_HOME/bin/jar" cf ../../awt-compile-stub.jar .) && rm -rf tools/awtstub/x
fi

step "3/6 Android SDK (platforms;android-34)"
SDK=${ANDROID_HOME:-$HOME/Android/Sdk}
[ -d "$SDK/platforms/android-34" ] || [ ! -d "$ROOT/tools/android-sdk/platforms/android-34" ] || SDK="$ROOT/tools/android-sdk"
if [ ! -d "$SDK/platforms/android-34" ] || [ ! -d "$SDK/build-tools/34.0.0" ]; then
  [ -d "$SDK/platforms" ] && [ -w "$SDK" ] || SDK="$ROOT/tools/android-sdk"
  if [ ! -x "$SDK/cmdline-tools/latest/bin/sdkmanager" ]; then
    mkdir -p "$SDK/cmdline-tools"
    curl -fsSL https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -o "$SDK/clt.zip"
    unzip -q "$SDK/clt.zip" -d "$SDK/cmdline-tools" && mv "$SDK/cmdline-tools/cmdline-tools" "$SDK/cmdline-tools/latest" && rm "$SDK/clt.zip"
  fi
  yes | "$SDK/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$SDK" --licenses >/dev/null || true
  "$SDK/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$SDK" "platforms;android-34" "build-tools;34.0.0" >/dev/null
fi
export ANDROID_HOME="$SDK"
echo "sdk.dir=$SDK" > render/local.properties
echo "  SDK: $SDK"

step "4/6 Qwen3.5-4B asosiy model (~9.3 GB)"
if [ -n "$SKIP_MODEL" ]; then echo "  SKIP_MODEL=1: o'tkazib yuborildi"; else scripts/download_qwen.sh; fi

step "5/6 O'qitilgan adapter ($ADAPTER_TAG)"
A=runs/sft_5k/final; mkdir -p "$A"
for f in adapter_config.json adapter_model.safetensors; do
  [ -s "$A/$f" ] && { echo "  bor: $f"; continue; }
  curl -fL --progress-bar --retry 5 -C - -o "$A/$f" "$REPO_URL/releases/download/$ADAPTER_TAG/$f" || die "$f yuklanmadi"
done
.venv-train/bin/python -I - "$A" <<'EOF'
import hashlib, json, sys
a = sys.argv[1]
json.load(open(f"{a}/adapter_config.json"))
h = hashlib.sha256(open(f"{a}/adapter_model.safetensors", "rb").read()).hexdigest()
want = "321aa7e6ccd8ed5d20fe115c8590732a62e2b8fe390d62bfb771e442ef32e1ea"
if h == want:
    print("  adapter OK", h[:16])
else:
    sys.exit(f"XATO: adapter sha256 mos emas: {h}")
EOF

step "6/6 Render loyihasini sinash (birinchi marta Gradle kutubxonalarni yuklaydi)"
export GRADLE_USER_HOME=${D2C_GRADLE_HOME:-$ROOT/tools/gradle-home}
GEN=render/screens/src/main/kotlin/uz/d2c/screens/gen; mkdir -p "$GEN"
cat > "$GEN/All.kt" <<'EOF'
package uz.d2c.screens.gen

import androidx.compose.runtime.Composable

val ALL_SCREENS: List<Pair<Long, @Composable () -> Unit>> = listOf()
EOF
(cd render && ./gradlew -q :screens:testDebugUnitTest -Pd2cOut="$ROOT/data/render_tmp/setup" --console=plain) \
  || die "Gradle render loyihasi ishlamadi (yuqoridagi xatoga qarang)"
rm -rf "$GEN" data/render_tmp/setup
(cd render && ./gradlew --stop >/dev/null 2>&1 || true)

echo
mkdir -p logs natija
echo "TAYYOR. Sinab ko'ring:"
echo "  ./sinash.sh sinov_rasmlar/ekran1.jpg"
