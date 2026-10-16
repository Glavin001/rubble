#!/usr/bin/env bash
# Fail when a commit range adds a file over the limit that is not a Git LFS pointer,
# or when a tracked *.glb is stored as a raw blob. Large assets belong in LFS
# (.gitattributes); raw blobs stay in history forever once they reach main.
#
#   check-large-files.sh BASE HEAD   # every blob the commits BASE..HEAD introduce
#   check-large-files.sh "" HEAD     # only the tree at HEAD (no usable base)
set -euo pipefail

limit_bytes=${LARGE_FILE_LIMIT_BYTES:-5242880}
base=${1:-}
head=${2:-HEAD}
status=0

if [ -n "$base" ] && git cat-file -e "$base^{commit}" 2>/dev/null; then
  objects=$(git rev-list --objects "$base..$head")
else
  objects=$(git ls-tree -r "$head" | awk '{print $3 " " $4}')
fi

while read -r type size path; do
  if [ "$type" = blob ] && [ "$size" -gt "$limit_bytes" ]; then
    echo "::error file=$path::$path is $size bytes (limit $limit_bytes); track it with Git LFS instead"
    status=1
  fi
done < <(printf '%s\n' "$objects" | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)')

while read -r path; do
  # LFS pointers are ~130 bytes; anything larger is the raw asset.
  if [ "$(git cat-file -s "$head:$path")" -gt 1024 ] ||
    ! git cat-file -p "$head:$path" | grep -q '^version https://git-lfs'; then
    echo "::error file=$path::$path is not a Git LFS pointer (*.glb must be stored in LFS)"
    status=1
  fi
done < <(git ls-tree -r --name-only "$head" | grep -E '\.glb$' || true)

if [ "$status" -eq 0 ]; then
  echo "No raw files over $limit_bytes bytes; every *.glb is an LFS pointer."
fi
exit "$status"
