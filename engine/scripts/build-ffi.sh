#!/usr/bin/env bash
# Build Rubble's C library for this platform and package it for a game engine:
#
#   engine/ffi-dist/rubble-<os>-<arch>/
#     include/rubble.h
#     lib/        librubble_ffi.{a,dylib,so} or rubble_ffi.{lib,dll,dll.lib}
#                 static-link-libs.txt (what the static library needs)
#     examples/hello.c
#     README.md   (engine/crates/engine-ffi/README.md)
#
# Then it compiles examples/hello.c against the package and, with RUN_EXAMPLE=1
# (the default), runs it. The example needs a GPU adapter (Metal, Vulkan or GL;
# lavapipe works); RUN_EXAMPLE=0 only links it.
#
# Needs Rust stable and Git LFS assets. slangc is used from PATH (or $SLANGC)
# when present, otherwise the pinned release is downloaded (scripts/slangc.sh).
# On Windows run it from Git Bash in an MSVC developer shell (cl on PATH).
set -euo pipefail

ENGINE="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="${RUBBLE_TOOLS:-$HOME/.cache/rubble-tools}"
RUN_EXAMPLE="${RUN_EXAMPLE:-1}"
cd "$ENGINE"

. "$ENGINE/scripts/slangc.sh"

echo "==> cargo build -p engine-ffi --release"
cargo build -p engine-ffi --release
echo "==> The committed header matches the ABI"
cargo test -p engine-ffi --release --test header
# The system libraries a host must add when it links the static library.
NATIVE_LIBS="$(cargo rustc -p engine-ffi --release --lib -- --print native-static-libs 2>&1 \
  | sed -n 's/.*native-static-libs: //p' | tail -n 1)"

TARGET="${CARGO_TARGET_DIR:-$ENGINE/target}/release"
case "$(uname -s)" in
  Darwin) os=macos ;;
  Linux) os=linux ;;
  MINGW* | MSYS* | CYGWIN*) os=windows ;;
  *) os="$(uname -s | tr '[:upper:]' '[:lower:]')" ;;
esac
arch="$(uname -m)"
case "$arch" in arm64) arch=aarch64 ;; esac
PKG="$ENGINE/ffi-dist/rubble-$os-$arch"
rm -rf "$PKG"
mkdir -p "$PKG/include" "$PKG/lib" "$PKG/examples"
cp crates/engine-ffi/include/rubble.h "$PKG/include/"
cp crates/engine-ffi/examples/hello.c "$PKG/examples/"
cp crates/engine-ffi/README.md "$PKG/README.md"
printf 'Link the static library with these system libraries:\n%s\n' "$NATIVE_LIBS" \
  > "$PKG/lib/static-link-libs.txt"

echo "==> Package $PKG"
case "$os" in
  macos)
    cp "$TARGET/librubble_ffi.a" "$TARGET/librubble_ffi.dylib" "$PKG/lib/"
    # Load the dylib from next to the executable that links it.
    install_name_tool -id @rpath/librubble_ffi.dylib "$PKG/lib/librubble_ffi.dylib"
    cc -std=c11 -Wall -Wextra -Werror -I "$PKG/include" "$PKG/examples/hello.c" \
      -L "$PKG/lib" -lrubble_ffi -Wl,-rpath,"$PKG/lib" -o "$PKG/examples/hello"
    ;;
  linux)
    cp "$TARGET/librubble_ffi.a" "$TARGET/librubble_ffi.so" "$PKG/lib/"
    cc -std=c11 -Wall -Wextra -Werror -I "$PKG/include" "$PKG/examples/hello.c" \
      -L "$PKG/lib" -lrubble_ffi -lm -Wl,-rpath,"$PKG/lib" -o "$PKG/examples/hello"
    ;;
  windows)
    cp "$TARGET/rubble_ffi.lib" "$TARGET/rubble_ffi.dll" "$TARGET/rubble_ffi.dll.lib" "$PKG/lib/"
    # Link the import library; the DLL sits next to the executable.
    cp "$PKG/lib/rubble_ffi.dll" "$PKG/examples/"
    (cd "$PKG/examples" && cl -nologo -W3 -WX -std:c11 -I ../include hello.c \
      ../lib/rubble_ffi.dll.lib -Fe:hello.exe)
    ;;
esac

if [[ "$RUN_EXAMPLE" == 1 ]]; then
  echo "==> Run examples/hello"
  "$PKG/examples/hello"
fi
ls -l "$PKG/lib"
echo "==> $PKG"
