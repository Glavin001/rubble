#!/usr/bin/env bash
# Side-by-side videos and stills of the AVBD vehicle study (engine/docs/avbd-vehicles.md):
# explicit 2 kHz (left) against the AVBD mode (right), CPU only. Output:
# ~/Development/rubble-artifacts/avbd-vehicles/ (avbd_video's default).
#
#   cargo build --release -p engine-softbody --example avbd_video
#   engine/tools/avbd-vehicles-videos.sh
set -u
cd "$(dirname "$0")/.."
run() { perl -e 'alarm 1200; exec @ARGV' "$@"; }
while read -r car cfg scene secs; do
  [ -z "$car" ] && continue
  run ./target/release/examples/avbd_video --car "$car" --config "$cfg" --scene "$scene" --seconds "$secs" 2>&1 | tail -1
done <<'EOF'
etk_kc6 120x50 wall 1.0
etk_kc6 240x100 wall 1.0
etk_kc6 120x20 bump 1.5
etk_kc6 240x100 bump 1.5
etk_kc6 120x50 pole 1.2
sedan1 120x50 wall 1.0
sedan1 120x20 bump 1.5
EOF
