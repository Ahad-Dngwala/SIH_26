#!/usr/bin/env bash
# Compile the Kotlin fusion core and replay the committed parity fixture through it.
#
# Deliberately does not use Gradle. The Android project needs the SDK, an emulator
# image and artifact resolution; this gate needs kotlinc, a JVM and Python. Keeping
# it that cheap is the difference between a gate that gets run before every filter
# change and one that gets run once.
#
# Usage:
#   tools/parity/run_kotlin_parity.sh
#
# Requires kotlinc on PATH. Exits non-zero on divergence.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

fusion_src="$repo_root/android_app/app/src/main/java/org/sih26/deadreckoning/fusion"
harness_src="$repo_root/tools/parity/kotlin"

if ! command -v kotlinc >/dev/null 2>&1; then
  echo "kotlinc not found on PATH." >&2
  echo "Install the Kotlin command line compiler, or run the same check from the" >&2
  echo "Android project as a unit test once Gradle is set up." >&2
  exit 127
fi

echo "Flattening the fixture"
python3 -m tools.parity.export_fixture_csv --output "$work_dir/fixture.csv"

echo "Compiling the Kotlin fusion core and harness"
kotlinc \
  "$fusion_src/LinAlg.kt" \
  "$fusion_src/Ukf.kt" \
  "$harness_src/VerifyParity.kt" \
  -include-runtime -d "$work_dir/parity.jar" 2>&1 | grep -v '^warning: ' || true

echo "Replaying"
java -cp "$work_dir/parity.jar" org.sih26.deadreckoning.parity.VerifyParityKt "$work_dir/fixture.csv"
