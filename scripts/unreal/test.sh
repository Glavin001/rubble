#!/usr/bin/env bash
# RubbleDemo's automation tests: the logic tests (Rubble.Clock, Rubble.Mock) headless in a
# commandlet, then the renderer tests (Rubble.Render) in a game process, which they need.
#
#   scripts/unreal/test.sh [--no-build] [--only Rubble.Clock]
#
# Exit status: the number of failed test groups.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UE_ROOT="${UE_ROOT:-/Users/Shared/Epic Games/UE_5.8}"
APP="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS"
CMD="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor-Cmd"
PROJECT="$ROOT/unreal/RubbleDemo/RubbleDemo.uproject"
LOGS="$ROOT/target/unreal-tests"
BUILD=1
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) BUILD=0; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "test: unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$LOGS"
if [[ $BUILD == 1 ]]; then
  "$ROOT/scripts/unreal/build-editor.sh" > "$LOGS/build.log" 2>&1 || { echo "test: build failed; see $LOGS/build.log" >&2; tail -30 "$LOGS/build.log" >&2; exit 2; }
fi

failed=0
run() { # run <name> <filter> <game|cmd>
  local name="$1" filter="$2" mode="$3" log="$LOGS/$1.log"
  rm -f "$log"
  if [[ "$mode" == game ]]; then
    "$APP/UnrealEditor" "$PROJECT" -game -windowed -ResX=1280 -ResY=720 -RubbleNoVolumetrics -notraceserver \
      "-ExecCmds=Automation RunTests $filter; Quit" "-TestExit=Automation Test Queue Empty" -abslog="$log" > /dev/null 2>&1 &
  else
    "$CMD" "$PROJECT" -nullrhi -unattended -nopause -nosplash -NoSound -notraceserver \
      "-ExecCmds=Automation RunTests $filter; Quit" "-TestExit=Automation Test Queue Empty" -abslog="$log" > /dev/null 2>&1 &
  fi
  local pid=$!
  # Stop a hung run after ten minutes; SIGKILL if a graceful exit hangs.
  for _ in $(seq 1 200); do kill -0 "$pid" 2>/dev/null || break; sleep 3; done
  kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
  local passed failures
  passed=$(grep -c "Result={Success}" "$log" 2>/dev/null || true)
  failures=$(grep -c "Result={Fail" "$log" 2>/dev/null || true)
  if [[ "${passed:-0}" -gt 0 && "${failures:-0}" -eq 0 ]]; then
    echo "pass  $name ($passed tests)"
  else
    failed=$((failed + 1))
    echo "FAIL  $name ($passed passed, $failures failed; $log)"
    grep -E "LogAutomationController: Error: |Result=\{Fail" "$log" 2>/dev/null | sed 's/.*Error: /        /' | head -15
  fi
}
if [[ -z "$ONLY" || "$ONLY" == Rubble.Clock* || "$ONLY" == Rubble.Mock* || "$ONLY" == Rubble.Abi* || "$ONLY" == Rubble.Glb* ]]; then
  run logic "${ONLY:-Rubble.Clock+Rubble.Mock+Rubble.Abi+Rubble.Glb}" cmd
fi
if [[ -z "$ONLY" || "$ONLY" == Rubble.Render* ]]; then
  run render "${ONLY:-Rubble.Render}" game
fi
exit "$failed"
