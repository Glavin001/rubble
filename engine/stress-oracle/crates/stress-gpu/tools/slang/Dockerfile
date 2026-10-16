# Pinned Slang shader compiler. It only turns .slang into shader source text (WGSL,
# Metal), which is the same whatever OS compiled it, so it runs here in Linux and the
# output is checked in: building the solver needs no Slang. See scripts/compile_shaders.sh.
FROM ubuntu:24.04
ARG SLANG_VERSION=2026.19
ARG SLANG_SHA256=f49229eb9606b47b122e1d8789be46a7bcd6bb2ffa94b06fae6d544a4e6df637
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/slang.tar.gz \
      "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-aarch64-glibc-2.28.tar.gz" \
 && echo "${SLANG_SHA256}  /tmp/slang.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/slang && tar -xzf /tmp/slang.tar.gz -C /opt/slang && rm /tmp/slang.tar.gz
ENV PATH=/opt/slang/bin:$PATH LD_LIBRARY_PATH=/opt/slang/lib
WORKDIR /work
ENTRYPOINT ["slangc"]
