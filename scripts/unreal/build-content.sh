#!/usr/bin/env bash
# Generate RubbleDemo's materials (Content/Rubble/Materials) from Content/Python/build_content.py,
# headless. Generated content, not committed; re-run after editing the script. Without it the game
# still runs on engine stand-in materials (flat colours, no tints).
#
#   scripts/unreal/build-content.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UE_ROOT="${UE_ROOT:-/Users/Shared/Epic Games/UE_5.8}"
EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor-Cmd"
[[ -x "$EDITOR" ]] || EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor-Cmd"
PROJECT="$ROOT/unreal/RubbleDemo/RubbleDemo.uproject"
LOG="$ROOT/target/unreal/build-content.log"
mkdir -p "$(dirname "$LOG")"
"$EDITOR" "$PROJECT" -run=pythonscript -script="$ROOT/unreal/RubbleDemo/Content/Python/build_content.py" \
  -unattended -nopause -nosplash -NoSound -notraceserver -abslog="$LOG" || true
if grep -q "build_content: done" "$LOG"; then
  grep "build_content:" "$LOG" | sed 's/.*build_content:/build_content:/'
else
  echo "build-content: failed; see $LOG" >&2
  grep -E "Error|Traceback|build_content" "$LOG" | tail -20 >&2
  exit 1
fi
