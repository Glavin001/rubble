#!/usr/bin/env bash
# What macOS recorded about Metal's shader compiler in the last minutes: for a CI
# step that runs after a pipeline failed to compile. wgpu's error is the whole
# NSError, and for a compiler that gave up that is only "Compilation failed"
# (domain CompilerError, code 2); the reason, when there is one, is in the unified
# log of MTLCompilerService (the process that compiles for every Metal client), in a
# crash report of it, or in the kernel's memory-pressure kills.
#
#   engine/scripts/metal-compiler-log.sh [minutes, default 5]
set -uo pipefail
MIN="${1:-5}"
[[ "$(uname -s)" == Darwin ]] || { echo "macOS only" >&2; exit 0; }

echo "==> Machine"
sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize 2>/dev/null
sw_vers 2>/dev/null
system_profiler SPDisplaysDataType 2>/dev/null | sed -n 1,25p
echo "==> Memory now"
vm_stat | sed -n 1,12p
memory_pressure 2>/dev/null | tail -n 3
sysctl vm.swapusage 2>/dev/null

echo "==> Unified log, last ${MIN} min: the Metal compiler service, Metal, memory kills"
log show --last "${MIN}m" --style compact --info --debug --predicate '
  process == "MTLCompilerService"
  OR senderImagePath CONTAINS[c] "Metal"
  OR senderImagePath CONTAINS[c] "AGX"
  OR senderImagePath CONTAINS[c] "ParavirtualizedGraphics"
  OR eventMessage CONTAINS[c] "MTLCompiler"
  OR eventMessage CONTAINS[c] "jetsam"
  OR eventMessage CONTAINS[c] "memorystatus"
  OR eventMessage CONTAINS[c] "CompilerError"
  OR (process == "hello")
' 2>&1 | tail -n 400

echo "==> Crash reports"
for dir in "$HOME/Library/Logs/DiagnosticReports" /Library/Logs/DiagnosticReports; do
  ls -lt "$dir" 2>/dev/null | head -n 20
  for f in "$dir"/MTLCompilerService* "$dir"/hello*; do
    [[ -f "$f" ]] || continue
    echo "---- $f"
    head -c 20000 "$f"
  done
done
exit 0
