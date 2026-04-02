#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Building WASM package..."
cd "$PROJECT_ROOT"
wasm-pack build crates/rubble-wasm --target web --out-dir ../../web/src/wasm

echo "==> Installing web dependencies..."
cd "$SCRIPT_DIR"
npm ci

echo "==> Building web project..."
npm run build

echo "==> Done! Output in web/dist/"
