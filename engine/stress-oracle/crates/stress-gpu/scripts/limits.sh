#!/usr/bin/env bash
# Measure this GPU's limits and write limits/<target>.toml (target from the chip name,
# e.g. "Apple M3 Max" -> m3_max). macOS + Xcode only.
#
# Runs tools/limits/build/limits (see tools/limits/limits.m and limits/README.md) once,
# with control conditions recorded around it: chip, GPU cores, OS, power mode, and GPU
# utilization ("Device Utilization %" from ioreg) sampled before and after the run, i.e.
# by other processes. The run is marked contended when that exceeds --threshold percent;
# every number is still the minimum (and median) of N repeats.
#
# The default suite only uses one threadgroup at a time except the brief bandwidth copy
# (~1 ms per repeat), and takes about 30 s. The co-residency and watchdog probes are
# OPT-IN only:
#   --coresidency     bounded-spin co-residency sweep (blocks the whole GPU for a few
#                     ms per point; every wait is bounded, it cannot hang)
#   --watchdog S      command-buffer watchdog: single command buffers of 0.25 s .. S s.
#                     May stall the desktop for up to S seconds. Run it on an idle machine.
#
#   scripts/limits.sh                          default suite -> limits/<target>.toml
#   scripts/limits.sh --repeats 11 --out /tmp/x.toml
#   scripts/limits.sh --probes ops,dispatch    a subset
set -euo pipefail
cd "$(dirname "$0")/.."
repeats=7
threshold=15
probes=""
out=""
extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repeats) repeats=$2; shift ;;
    --threshold) threshold=$2; shift ;;
    --probes) probes=$2; shift ;;
    --out) out=$2; shift ;;
    --coresidency) extra+=(--coresidency) ;;
    --watchdog) extra+=(--watchdog "$2"); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
if [ "$(uname)" != Darwin ]; then
  echo "limits: Metal probes need macOS (see limits/rtx4090.toml for the CUDA target's documented values)" >&2
  exit 2
fi
tools/limits/build.sh > /dev/null

chip=$(sysctl -n machdep.cpu.brand_string)
target=$(echo "${chip#Apple }" | tr 'A-Z ' 'a-z_')
[ -n "$out" ] || out="limits/$target.toml"
cores=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"gpu-core-count" = [0-9]*' | head -1 | grep -o '[0-9]*$' || echo 0)
powermode=$(pmset -g | awk '$1 == "powermode" {print $2}')
lowpower=$(pmset -g | awk '$1 == "lowpowermode" {print $2}')

utilization() {  # one "Device Utilization %" sample
  ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | grep -o '[0-9]*$' || echo -1
}
samples() {  # 5 samples, 0.2 s apart, space-separated
  local s=()
  for _ in 1 2 3 4 5; do s+=("$(utilization)"); sleep 0.2; done
  echo "${s[*]}"
}
median() { tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
maximum() { tr ' ' '\n' | sort -n | tail -1; }

before=$(samples)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
args=(probe --repeats "$repeats" --work "$work/kernels")
[ -n "$probes" ] && args+=(--probes "$probes")
start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
t0=$(date +%s)
tools/limits/build/limits "${args[@]}" ${extra[@]+"${extra[@]}"} > "$work/probes.toml" 2> >(tee "$work/progress.log" >&2)
t1=$(date +%s)
sleep 1
after=$(samples)
bmed=$(echo "$before" | median); amed=$(echo "$after" | median)
bmax=$(echo "$before" | maximum); amax=$(echo "$after" | maximum)
contended=false
if [ "$bmed" -gt "$threshold" ] || [ "$amed" -gt "$threshold" ]; then contended=true; fi
commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

mkdir -p "$(dirname "$out")"
{
  echo "# GPU limits measured by scripts/limits.sh (tools/limits/limits.m). See limits/README.md."
  echo "# Every timed value is the minimum (and median) of 'repeats' runs on a warmed GPU unless stated."
  echo "# Driver statistics use a PRIVATE Metal API (pipeline option 0x40000)."
  echo
  echo "[control]"
  echo "measured = true"
  echo "chip = \"$chip\""
  echo "target = \"$target\""
  echo "gpu_cores = $cores"
  echo "os = \"$(sw_vers -productVersion) ($(sw_vers -buildVersion))\""
  echo "power_mode = \"${powermode:-unknown}\""
  echo "low_power_mode = \"${lowpower:-unknown}\""
  echo "metal_compiler = \"$(xcrun -sdk macosx metal --version 2>/dev/null | head -1)\""
  echo "date_utc = \"$start\""
  echo "duration_s = $((t1 - t0))"
  echo "commit = \"$commit\""
  echo "repeats = $repeats"
  echo "probes = \"${probes:-default}\""
  echo "opt_in = \"${extra[*]+${extra[*]}}\""
  echo "gpu_utilization_before_pct = [$(echo "$before" | sed 's/ /, /g')]"
  echo "gpu_utilization_after_pct = [$(echo "$after" | sed 's/ /, /g')]"
  echo "gpu_utilization_note = \"ioreg IOAccelerator 'Device Utilization %', whole GPU, sampled while this tool was not running (other processes); max before $bmax, after $amax\""
  echo "contention_threshold_pct = $threshold"
  echo "contended = $contended"
  echo
  cat "$work/probes.toml"
} > "$out"
echo "wrote $out (contended = $contended; utilization by others: before median $bmed%, after median $amed%)" >&2
