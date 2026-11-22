# gpu_run's build image on Apple silicon: native arm64 Linux (no Rosetta), cross-compiling
# to x86_64 Linux for Modal's GPU hosts. Build scripts (slangc -> WGSL, naga) run
# natively on arm64; the code and its C parts are compiled for x86_64 with Ubuntu's cross
# toolchain. Ubuntu 22.04's x86_64 glibc (2.35) runs on Modal's Debian bookworm (2.36).
#
# The CUDA route (-p engine-cuda) builds here too, with the x86_64 nvcc: the toolkit of
# engine/tools/cuda's image (the same release and files, copied from the stage below),
# run through OrbStack's Rosetta with the cross toolchain's libraries and its x86_64 g++ as
# the host compiler. NVIDIA's aarch64 (sbsa) nvcc of the same release is NOT used: it
# writes different SASS for some modules (15 of 237 measured 2026-11-21; on aarch64 `char`
# is unsigned), so its device code is not the x86_64 build's. README, "Build speed".
FROM --platform=linux/amd64 ubuntu:22.04 AS cuda-x86
ARG SLANG_VERSION=2026.19
ARG SLANG_X86_SHA256=bd7afcb0956dee9304f44ed115e9aaa7140704271b5c3294bcb237e3ae67adc7
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL -o /tmp/k.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/k.deb && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvcc-12-8=12.8.93-1 cuda-crt-12-8=12.8.93-1 \
      cuda-nvvm-12-8=12.8.93-1 cuda-cudart-dev-12-8=12.8.90-1 \
 && curl -fsSL -o /tmp/slang.tar.gz \
      "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64-glibc-2.28.tar.gz" \
 && echo "${SLANG_X86_SHA256}  /tmp/slang.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/slang-x86 && tar -xzf /tmp/slang.tar.gz -C /opt/slang-x86 include

FROM --platform=linux/arm64 ubuntu:22.04
ARG SLANG_VERSION=2026.19
ARG SLANG_SHA256=f49229eb9606b47b122e1d8789be46a7bcd6bb2ffa94b06fae6d544a4e6df637
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl build-essential pkg-config python3 git \
      gcc-x86-64-linux-gnu g++-x86-64-linux-gnu libc6-dev-amd64-cross binutils-x86-64-linux-gnu \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/slang.tar.gz \
      "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-aarch64-glibc-2.28.tar.gz" \
 && echo "${SLANG_SHA256}  /tmp/slang.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/slang && tar -xzf /tmp/slang.tar.gz -C /opt/slang && rm /tmp/slang.tar.gz \
 && /opt/slang/bin/slangc -v
# The x86_64 nvcc (above) and the Slang CUDA prelude it includes, with their files' times
# (nvcc's -lineinfo records each header's path and time; the slangc here writes the same
# CUDA source as the x86_64 one). Its x86_64 binaries find the cross libraries through
# /lib/x86_64-linux-gnu; NVCC_CCBIN gives it the x86_64 host compiler.
COPY --from=cuda-x86 /usr/local/cuda-12.8 /usr/local/cuda-12.8
RUN rm -rf /opt/slang/include
COPY --from=cuda-x86 /opt/slang-x86/include /opt/slang/include
RUN for f in /usr/x86_64-linux-gnu/lib/*; do [ -e /lib/x86_64-linux-gnu/"${f##*/}" ] || ln -s "$f" /lib/x86_64-linux-gnu/; done \
 && mkdir -p /lib64 && ln -sf /usr/x86_64-linux-gnu/lib/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2 \
 && NVCC_CCBIN=x86_64-linux-gnu-g++ /usr/local/cuda-12.8/bin/nvcc --version | grep -q V12.8.93
# `--fanout-tests` lists each test binary's tests by running it (`--list`, no GPU): the
# x86_64 binary runs here through OrbStack's Rosetta, given the cross toolchain's loader
# and libraries (RUBBLE_X86_LIBS: gpu_run_pack.py sets LD_LIBRARY_PATH to it for that).
ENV RUBBLE_X86_LIBS=/usr/x86_64-linux-gnu/lib
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
 && /opt/cargo/bin/rustup target add x86_64-unknown-linux-gnu
ENV PATH=/opt/cargo/bin:/opt/slang/bin:$PATH \
    RUSTUP_TOOLCHAIN=stable \
    SLANGC=/opt/slang/bin/slangc \
    NVCC=/usr/local/cuda-12.8/bin/nvcc \
    NVCC_CCBIN=x86_64-linux-gnu-g++ \
    CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc \
    CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc \
    CXX_x86_64_unknown_linux_gnu=x86_64-linux-gnu-g++ \
    AR_x86_64_unknown_linux_gnu=x86_64-linux-gnu-ar \
    STRIP=x86_64-linux-gnu-strip
WORKDIR /repo
