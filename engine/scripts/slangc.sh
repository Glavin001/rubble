# Sourced, not run: puts the pinned slangc on $SLANGC when none is installed.
# The WebGPU crates' build scripts compile the Slang kernels with it.
# Needs $TOOLS (the download cache); sets $SLANGC.
SLANG_VERSION=2026.19   # engine/TOOLCHAIN.md

if [[ -z "${SLANGC:-}" ]] && ! command -v slangc >/dev/null 2>&1 \
  && [[ ! -x "$HOME/.local/bin/slangc" ]]; then
  slang_ext=tar.gz
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) slang_asset="slang-$SLANG_VERSION-linux-x86_64-glibc-2.27" ;;
    Linux-aarch64) slang_asset="slang-$SLANG_VERSION-linux-aarch64" ;;
    Darwin-arm64) slang_asset="slang-$SLANG_VERSION-macos-aarch64" ;;
    Darwin-x86_64) slang_asset="slang-$SLANG_VERSION-macos-x86_64" ;;
    MINGW*-x86_64 | MSYS*-x86_64 | CYGWIN*-x86_64)
      slang_asset="slang-$SLANG_VERSION-windows-x86_64"
      slang_ext=zip ;;
    *) echo "no slangc release for $(uname -s)-$(uname -m); install slangc $SLANG_VERSION" >&2
       exit 1 ;;
  esac
  slang_dir="$TOOLS/slang-$SLANG_VERSION"
  slang_url="https://github.com/shader-slang/slang/releases/download/v$SLANG_VERSION/$slang_asset.$slang_ext"
  if [[ ! -x "$slang_dir/bin/slangc" && ! -f "$slang_dir/bin/slangc.exe" ]]; then
    echo "==> Downloading $slang_asset.$slang_ext"
    mkdir -p "$slang_dir"
    if [[ "$slang_ext" == zip ]]; then
      curl -fsSL -o "$slang_dir/slang.zip" "$slang_url"
      unzip -q "$slang_dir/slang.zip" -d "$slang_dir"
      rm "$slang_dir/slang.zip"
    else
      curl -fsSL "$slang_url" | tar -xz -C "$slang_dir"
    fi
  fi
  if [[ -f "$slang_dir/bin/slangc.exe" ]]; then
    export SLANGC="$slang_dir/bin/slangc.exe"
  else
    export SLANGC="$slang_dir/bin/slangc"
  fi
fi
echo "==> slangc: ${SLANGC:-$(command -v slangc || echo "$HOME/.local/bin/slangc")}"
