#!/usr/bin/env bash
# Build the GPU limits probe tool (Objective-C, Metal + Foundation only) into
# tools/limits/build/limits.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
clang -O2 -fobjc-arc -Wall -Wno-unused-function -framework Metal -framework Foundation limits.m -o build/limits
echo "built $(pwd)/build/limits"
