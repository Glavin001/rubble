#!/usr/bin/env bash
# What macOS recorded about Metal's shader compiler in the last minutes: for a CI
# step that runs after a pipeline failed to compile. wgpu's error is the whole
# NSError, and for a compiler that gave up that is only "Compilation failed"
# (domain CompilerError, code 2); the reason, when there is one, is in the unified
# log of MTLCompilerService (the process that compiles for every Metal client), in a
# crash report of it, or in the kernel's memory-pressure kills. On GitHub's macOS
# runners (a VM with Apple's paravirtual GPU, whose host builds the pipeline) a
# pipeline that takes the host more than about 10 s shows here as
#   (Metal) ... Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)
#   (AppleParavirtGPUMetalIOGPUFamily) Failed to get compute pipeline state info
# (engine/docs/measurements/kernel-compile-time.md, 2026-11-16).
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
  process == "MTLCompilerService" AND (senderImagePath CONTAINS[c] "MTLCompiler" OR messageType >= 16)
  OR senderImagePath ENDSWITH "/Metal"
  OR senderImagePath CONTAINS[c] "AGX"
  OR senderImagePath CONTAINS[c] "Paravirt"
  OR senderImagePath CONTAINS[c] "IOGPU"
  OR eventMessage CONTAINS[c] "CompilerError"
  OR (process == "kernel" AND eventMessage CONTAINS[c] "memorystatus" AND eventMessage CONTAINS[c] "kill")
' 2>&1 | tail -n 300

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
