#!/usr/bin/env bash
# Linux x86_64 build toolchain for engine-cuda: pinned slangc and nvcc only
# (no driver: PTX is compiled on the build host and JIT-compiled by whatever
# driver runs it). Used by CI and by the Modal build image.
#
#   sudo engine/tools/cuda/install_linux_toolchain.sh
#   export SLANGC=/opt/slang/bin/slangc NVCC=/usr/local/cuda-12.8/bin/nvcc
set -euo pipefail

SLANG_VERSION=2026.19
SLANG_SHA256=bd7afcb0956dee9304f44ed115e9aaa7140704271b5c3294bcb237e3ae67adc7
CUDA_PKG_VERSION=12-8

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg g++

curl -fsSL -o /tmp/slang.tar.gz \
  "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64-glibc-2.28.tar.gz"
echo "${SLANG_SHA256}  /tmp/slang.tar.gz" | sha256sum -c -
mkdir -p /opt/slang
tar -xzf /tmp/slang.tar.gz -C /opt/slang
rm /tmp/slang.tar.gz
/opt/slang/bin/slangc -v

# NVIDIA's apt repo for this distro (ubuntu2204, ubuntu2404, debian12, ...).
. /etc/os-release
case "$ID" in
  ubuntu) repo="ubuntu${VERSION_ID//./}" ;;
  debian) repo="debian${VERSION_ID%%.*}" ;;
  *) echo "unsupported distro $ID" >&2; exit 1 ;;
esac
curl -fsSL -o /tmp/cuda-keyring.deb \
  "https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb
apt-get update
# nvcc plus the runtime headers it includes implicitly; no driver, no libraries.
apt-get install -y --no-install-recommends "cuda-nvcc-${CUDA_PKG_VERSION}" "cuda-cudart-dev-${CUDA_PKG_VERSION}"
rm -rf /var/lib/apt/lists/*
"/usr/local/cuda-${CUDA_PKG_VERSION//-/.}/bin/nvcc" --version
