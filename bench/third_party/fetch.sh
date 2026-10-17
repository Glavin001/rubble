#!/bin/sh
# Fetch the CPU engines the benchmark builds from source, at pinned revisions.
# Sources live in bench/third_party/<name> (gitignored); adapters' build.rs files build them.
set -e
cd "$(dirname "$0")"
fetch() { # name url commit
  [ -d "$1/.git" ] || git clone -q "$2" "$1"
  git -C "$1" fetch -q origin "$3" 2>/dev/null || git -C "$1" fetch -q origin
  git -C "$1" checkout -q "$3"
  echo "$1 @ $(git -C "$1" rev-parse --short=12 HEAD)"
}
fetch box3d https://github.com/erincatto/box3d.git efca0c184d40fd6a82016983f1601c4d23965867
fetch JoltPhysics https://github.com/jrouwe/JoltPhysics.git e77f175595e64cb44218cc9d9d56fc365ad0e36a
