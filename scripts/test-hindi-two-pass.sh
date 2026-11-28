#!/bin/bash
# Host test for HindiTwoPass.kt with Gradle's bundled Kotlin compiler (no Android SDK or JUnit needed).
set -euo pipefail
cd "$(dirname "$0")/.."
T=/Volumes/ExtremeDisk/android-tools
LIB=$T/gradle-9.9.1/lib
JAVA=${JAVA:-$T/jdk21/bin/java}
[ -x "$JAVA" ] || JAVA=$T/jdk21/Contents/Home/bin/java
OUT="${TMPDIR:-/tmp}/tellwell-hindi-two-pass"; mkdir -p "$OUT"
CP="$LIB/kotlin-compiler-embeddable-2.4.21-min.jar:$LIB/kotlin-stdlib-2.4.21.jar:$LIB/kotlin-reflect-2.4.21.jar:$LIB/kotlin-script-runtime-2.4.21.jar:$LIB/kotlin-daemon-embeddable-2.4.21.jar:$LIB/annotations-24.0.1.jar:$LIB/kotlinx-coroutines-core-jvm-1.10.2.jar"
"$JAVA" -cp "$CP" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -classpath "$LIB/kotlin-stdlib-2.4.21.jar" \
  -d "$OUT" app/src/main/java/io/github/ayushpatnaikgit/tellwell/HindiTwoPass.kt tests/HindiTwoPassTest.kt
"$JAVA" -cp "$OUT:$LIB/kotlin-stdlib-2.4.21.jar" HindiTwoPassTestKt "$@"
