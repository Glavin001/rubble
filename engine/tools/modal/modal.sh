#!/usr/bin/env bash
# The Modal CLI in Docker: nothing is installed on the host. The repo is
# mounted at /repo (run from the repo root and pass repo-relative paths), and
# the login token lives in the `rubble-modal-config` Docker volume.
#
#   engine/tools/modal/modal.sh setup                                  # once: open the printed link
#   engine/tools/modal/modal.sh run engine/tools/modal/cuda_tests.py   # T4 correctness run
#   docker volume rm rubble-modal-config                               # forget the login
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
image=rubble-modal-cli:1.6.2
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build -q -t "$image" "$here" >/dev/null
fi
tty=()
if [ -t 0 ] && [ -t 1 ]; then tty=(-it); fi
exec docker run --rm ${tty[@]+"${tty[@]}"} \
  -v "$repo:/repo" \
  -v rubble-modal-config:/config \
  -e MODAL_CONFIG_PATH=/config/modal.toml \
  ${MODAL_TOKEN_ID:+-e MODAL_TOKEN_ID} ${MODAL_TOKEN_SECRET:+-e MODAL_TOKEN_SECRET} \
  "$image" "$@"
