# Local, free build of engine-cuda's test binaries (no GPU, no driver): the
# same x86_64 toolchain as CI. Ubuntu 22.04's glibc (2.35) runs on Modal's
# Debian bookworm GPU image. On Apple silicon this runs under emulation.
FROM --platform=linux/amd64 ubuntu:22.04
COPY install_linux_toolchain.sh /opt/rubble/install_linux_toolchain.sh
RUN bash /opt/rubble/install_linux_toolchain.sh
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config python3-pip \
 && rm -rf /var/lib/apt/lists/*
# Only for `cuda_tests.py --package` (the Modal decorators import it); no login.
RUN python3 -m pip install --no-cache-dir modal==1.6.2
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/opt/cargo/bin:/opt/slang/bin:/usr/local/cuda-12.8/bin:$PATH \
    RUSTUP_TOOLCHAIN=stable \
    SLANGC=/opt/slang/bin/slangc \
    NVCC=/usr/local/cuda-12.8/bin/nvcc \
    ENGINE_CUDA_REQUIRE_PTX=1
WORKDIR /repo
