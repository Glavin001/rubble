#!/usr/bin/env bash
# engine-cuda GPU tests with only the GPU seconds on Modal: build the test
# binaries in local Docker (free), then run them on a single-use Modal GPU.
# Arguments go to cuda_tests.py (--gpu, --perf, --filter, --sync, --sanitize,
# --env). Build-time switches (ENGINE_CUDA_LOAD_CACHE, ...) come from the
# environment.
#
#   engine/tools/modal/run_cuda_tests.sh
#   engine/tools/modal/run_cuda_tests.sh --perf --gpu L40S
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../../.."
archive=engine/target/cuda-tests.tar.gz
engine/tools/cuda/build_tests_in_docker.sh "$archive"
exec engine/tools/modal/modal.sh run engine/tools/modal/cuda_tests.py --archive "$archive" "$@"
