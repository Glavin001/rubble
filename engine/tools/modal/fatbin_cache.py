#!/usr/bin/env python3
"""Keep the local nvcc fatbin cache (ENGINE_CUDA_FATBIN_CACHE, engine-slang-build) bounded.

    python3 fatbin_cache.py DIR [--max-gib N] [--max-days D]

The cache keeps each fatbin by the content of what made it (the CUDA source's bytes,
nvcc's version and arguments): an entry is three files with one stem, `.cu`, `.args`,
`.fatbin`, and a hit touches all three. Entries untouched for more than D days (default
7) go, then the least recently used ones until the cache holds at most N GiB (default
12). Run before a build by gpu_run.sh and build_ref.sh; a removed entry only costs its
nvcc compile again.
"""
import os
import sys
import time


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    root, max_gib, max_days = argv[0], 12.0, 7.0
    rest = argv[1:]
    while rest:
        flag, value, rest = rest[0], float(rest[1]), rest[2:]
        if flag == "--max-gib":
            max_gib = value
        elif flag == "--max-days":
            max_days = value
        else:
            raise SystemExit(f"fatbin_cache: unknown option {flag}")
    if not os.path.isdir(root):
        return
    stems: dict[str, list] = {}  # stem -> [newest mtime, bytes, file names]
    for name in os.listdir(root):
        stem = name.split(".", 1)[0]
        try:
            st = os.stat(os.path.join(root, name))
        except FileNotFoundError:
            continue
        s = stems.setdefault(stem, [0.0, 0, []])
        s[0] = max(s[0], st.st_mtime)
        s[1] += st.st_size
        s[2].append(name)
    now = time.time()
    total = sum(s[1] for s in stems.values())
    removed = 0
    budget = max_gib * (1 << 30)
    for mtime, size, names in sorted(stems.values(), key=lambda s: s[0]):
        if now - mtime <= max_days * 86400 and total <= budget:
            break
        for name in names:
            try:
                os.remove(os.path.join(root, name))
            except FileNotFoundError:
                pass
        total -= size
        removed += 1
    if removed:
        print(f"fatbin_cache: removed {removed} old entries, {total / (1 << 30):.1f} GiB kept", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
