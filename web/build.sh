#!/usr/bin/env bash
# Full web build: npm ci + npm run build:all (WASM via wasm-pack + TypeScript + Vite).
# Intended to be the only command CI, Vercel, and docs use for producing web/dist.
# Run from repository root: ./web/build.sh  (or: bash web/build.sh)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${HOME}/.cargo/env" ]]; then
  # shellcheck source=/dev/null
  . "${HOME}/.cargo/env"
fi
cd "$SCRIPT_DIR"

# Ensure the wasm32 target is available (needed by wasm-pack)
if ! rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
  echo "==> Adding wasm32-unknown-unknown target..."
  rustup target add wasm32-unknown-unknown
fi

npm ci
npm run build:all
echo "==> Done. Output: web/dist/"
