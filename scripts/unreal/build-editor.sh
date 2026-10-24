#!/usr/bin/env bash
# Build the RubbleDemo editor target (the game runs from the editor binary with -game).
#
#   scripts/unreal/build-editor.sh [Development|DebugGame]
#
# UE_ROOT defaults to the Launcher install of UE 5.8.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-Development}"
UE_ROOT="${UE_ROOT:-/Users/Shared/Epic Games/UE_5.8}"
PROJECT="$ROOT/unreal/RubbleDemo/RubbleDemo.uproject"

case "$(uname -s)" in
  Darwin) BUILD="$UE_ROOT/Engine/Build/BatchFiles/Mac/Build.sh"; PLATFORM=Mac ;;
  Linux) BUILD="$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh"; PLATFORM=Linux ;;
  *) echo "build-editor: unsupported host" >&2; exit 2 ;;
esac

# -NoUBA: UBA mangles "-rpath @executable_path/..." link arguments with Xcode 15.4 (see unreal/README.md).
# -NoHotReload: other editors running from the same engine make UBT switch to hot-reload
# naming, which breaks linking between new modules.
exec "$BUILD" RubbleDemoEditor "$PLATFORM" "$CONFIG" -Project="$PROJECT" -WaitMutex -NoUBA -NoHotReload
