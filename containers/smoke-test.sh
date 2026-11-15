#!/bin/bash
# A fixed-charge-target CPM test with real charge assertions and fresh inputs.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: smoke-test NEW_OUTPUT_DIRECTORY [cpu|gpu]" >&2
  exit 2
fi
mode=${2:-cpu}
case "$mode" in cpu|gpu) ;; *) echo "Expected cpu or gpu" >&2; exit 2;; esac
out=$1
if [ -e "$out" ]; then
  echo "Output directory already exists; choose a new directory: $out" >&2
  exit 2
fi
mkdir -p "$out"
out=$(cd "$out" && pwd)
cp -R /opt/cpm/examples/. "$out/"
export GMX_BIN=/opt/gromacs-cpm/bin/gmx
export NTOMP=${NTOMP:-4}
export OMP_NUM_THREADS=$NTOMP
export GMX_NO_QUOTES=1
if [ "$mode" = gpu ]; then
  export CPM_FORCE_MODE=gpu
else
  unset CPM_FORCE_MODE
fi
cd "$out/cpm_testsuite"
"$GMX_BIN" --version > "$out/gromacs-version.txt" 2>&1
./run_tests.sh 05 11 | tee "$out/smoke-test.log"
# These two matrix-free cases need no matrix references; fail on skipped tests.
grep -Eq 'passed +2 +failed +0 +skipped +0' "$out/smoke-test.log"
echo "PASS container smoke test ($mode): fixed and time-dependent electrode charge targets"
