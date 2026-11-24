#!/usr/bin/env python3
"""How many kernel compiles an edit of each Slang file invalidates, and a ratchet on it.

    cargo run -q --profile fast -p engine-webgpu --bin kernels -- --closures CLOSURES.txt
    python3 tools/build-time/fanout.py CLOSURES.txt [--check | --init] [--top N]   (from engine/)

A compile's cache key is the text of every file its source reaches by `#include` (under
the compile's defines: engine-slang-build compile_key), so an edit of a file reruns every
compile whose closure holds it: a file's fan-out. `kernels --closures` logs each
compile's closure (RUBBLE_SLANG_CLOSURE_LOG); this counts them.

--check: fails when a file's fan-out is past its line in fanout-allowlist.txt (lines only
come down), or a file is not listed (add it with --init: the new number shows in review).
--init: rewrites the list from today's closures, keeping the lower of each old line and
today's count. Without either, the report: the files by fan-out, and by lines x fan-out
(what splitting a file would save: engine/docs/build-architecture.md, kernel rules).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOW = os.path.join(HERE, "fanout-allowlist.txt")
ENGINE = os.path.abspath(os.path.join(HERE, "..", ".."))


def norm(p):
    # Relative to the Slang root (`kernels/...`, `physics/...`), as compile_key logs it.
    i = p.find("/slang/")
    return p[i + len("/slang/"):] if i >= 0 else p


def fanout(path):
    units = set()
    count = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 5:
            continue
        unit = tuple(f[:4])
        if unit in units:
            continue
        units.add(unit)
        for p in set(norm(x) for x in f[4:]):
            count[p] = count.get(p, 0) + 1
    return len(units), count


def lines_of(rel):
    try:
        return sum(1 for _ in open(os.path.join(ENGINE, "slang", rel)))
    except OSError:
        return 0


def load():
    allow = {}
    if os.path.exists(ALLOW):
        for line in open(ALLOW):
            line = line.split("#")[0].split()
            if len(line) == 2:
                allow[line[0]] = int(line[1])
    return allow


def main():
    args = sys.argv[1:]
    top = 15
    if "--top" in args:
        i = args.index("--top")
        top = int(args.pop(i + 1))
        args.pop(i)
    flags = {a for a in args if a.startswith("--")}
    rest = [a for a in args if not a.startswith("--")]
    if not rest:
        print(__doc__, file=sys.stderr)
        return 2
    units, count = fanout(rest[0])
    allow = load()
    if "--init" in flags:
        merged = {f: min(n, allow.get(f, n)) for f, n in count.items()}
        with open(ALLOW, "w") as out:
            out.write("# Kernel compiles an edit of each Slang file invalidates (tools/build-time/fanout.py).\n")
            out.write("# Lines only come down: --init keeps the lower of the old line and today's count.\n")
            for f, n in sorted(merged.items(), key=lambda x: (-x[1], x[0])):
                out.write(f"{f} {n}\n")
        print(f"{len(merged)} files -> {ALLOW}")
        return 0
    print(f"{units} compiles, {len(count)} Slang files")
    print("Fan-out (compiles an edit reruns):")
    for f, n in sorted(count.items(), key=lambda x: -x[1])[:top]:
        print(f"  {n:5d}  {f}")
    print("Lines x fan-out (where a split saves most):")
    mass = sorted(((lines_of(f) * n, f, n) for f, n in count.items()), reverse=True)
    for m, f, n in mass[:top]:
        print(f"  {m:9d}  {f} ({lines_of(f)} lines x {n})")
    if "--check" in flags:
        bad = []
        for f, n in sorted(count.items()):
            if f not in allow:
                bad.append(f"{f}: not in fanout-allowlist.txt ({n} compiles): add it with fanout.py --init")
            elif n > allow[f]:
                bad.append(f"{f}: an edit reruns {n} compiles, more than its {allow[f]} (fanout-allowlist.txt): "
                           f"include it in fewer modules, or split it so they include less")
        if bad:
            print("\n".join(bad), file=sys.stderr)
            return 1
        print(f"fan-out: {len(count)} files within their lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
