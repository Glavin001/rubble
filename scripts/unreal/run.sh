#!/usr/bin/env bash
# Run RubbleDemo as a game (from the editor binary), optionally on a scene.
#
#   scripts/unreal/run.sh [family:scene?key=value] [more game args...]
#   scripts/unreal/run.sh soft:wall?speed=56
#   scripts/unreal/run.sh fracture:suburb -ResX=1920 -ResY=1080
#
# Families: rigid, vehicles, soft, fracture (the web pages' scenes; Tab in game lists them).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UE_ROOT="${UE_ROOT:-/Users/Shared/Epic Games/UE_5.8}"
EDITOR="$UE_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
PROJECT="$ROOT/unreal/RubbleDemo/RubbleDemo.uproject"
args=(-game -windowed -ResX=1600 -ResY=900 -notraceserver)
if [[ $# -gt 0 && "$1" == *:* ]]; then
  args+=("-RubbleDemo=$1")
  shift
fi
exec "$EDITOR" "$PROJECT" "${args[@]}" "$@"
