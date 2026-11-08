# Muhit o'zgaruvchilari (Gradle/Java'ni qo'lda ishlatish uchun):  source env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JAVA_HOME=$(ls -d "$ROOT"/tools/jdk-17* | head -1)
export ANDROID_HOME=${ANDROID_HOME:-$HOME/Android/Sdk}
export GRADLE_USER_HOME=${D2C_GRADLE_HOME:-$ROOT/tools/gradle-home}
export PATH=$JAVA_HOME/bin:$PATH
