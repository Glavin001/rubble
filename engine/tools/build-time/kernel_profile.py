#!/usr/bin/env python3
"""What a GPU kernel costs its compiler, and which source functions it comes from:
the kernel inlined and optimized by Apple's own offline tools, without a GPU or a
browser. macOS with Xcode only (xcrun metal, air-opt).

    python3 tools/build-time/kernel_profile.py MODULE.wgsl|DIR ... [--attr N [--who FN ...]] [--jobs N]
                                                                    (from engine/)

Per module: the entry point's instructions after every call is inlined and the code is
optimized (constants folded, dead branches dropped), and the first-load compile in
Chrome on Metal that predicts (seconds = K * instructions ^ B, fitted 2026-11-22 on the
vehicle lab's 28 modules, M3 Max: within 1.3x on average). With --attr N, the source
functions of the largest module's optimized code, from its line tables: each function's
instructions with all it inlined, how many copies of it survived optimization, and the
instructions those copies duplicate (what one call site would save, an upper bound). --who FN: FN's instructions by the call path
that made each copy.

Read it for duplicated copies (a heavy function inlined at several call sites): removing
those from the source shortens Chrome's compile (fleetLevelsAny's four copies: 49 -> 15 s).
Code the optimizer only drops by folding a constant (a constant argument, a masked
uniform) shrinks this count but not Chrome's compile (measured twice: within noise), so
judge a fix by a cold compile in Chrome (explicit-wgsl-check.mjs, EXPLICIT_WGSL_COLD=1).

Why not count the WGSL: a call that passes a constant (`part = 2u`) is folded down to
the branch it takes, so a static count overstates it (tools/build-time/compile_cost.py,
the CI gate, counts statements and cannot see this); and the driver's compile time
follows the code that is left. Measured: splitting such a function by its constant
changed the optimized kernel by 0 instructions and Chrome's cold compile within noise,
while the static count fell 24 %.

The WGSL is translated by naga (wgpu's own MSL, `--example wgsl_to_msl`), not Tint
(Chrome's), so the sizes are naga's; the ranking and the fit hold for both.
"""
import collections
import concurrent.futures as cf
import json
import math
import os
import re
import subprocess
import sys
import tempfile

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
K, B = 1.07e-08, 1.914
CAP = 10.0
INSTR = re.compile(r"^\s+(%[^ ]+ = )?[a-z]")


def to_msl_tool():
    subprocess.run(["cargo", "build", "-q", "--profile", "fast", "-p", "engine-webgpu-lab", "--example", "wgsl_to_msl"],
                   cwd=ENGINE, check=True, stderr=subprocess.DEVNULL)
    return os.path.join(ENGINE, "target", "fast", "examples", "wgsl_to_msl")


def optimized(tool, wgsl, work, debug):
    """The module's optimized LLVM IR (text), inlined as the driver does."""
    base = os.path.join(work, f"{abs(hash(os.path.abspath(wgsl))):x}_{os.path.basename(wgsl)[:-5]}")
    subprocess.run([tool, wgsl, base + ".metal"], check=True)
    flags = ["-gline-tables-only"] if debug else []
    subprocess.run(["xcrun", "metal", "-O2", "-w", *flags, "-S", "-emit-llvm", base + ".metal", "-o", base + ".ll"], check=True)
    subprocess.run(["xcrun", "air-opt", "-S", "-passes=default<O2>", "-inline-threshold=1000000",
                    base + ".ll", "-o", base + ".opt.ll"], check=True)
    return base + ".opt.ll"


def bodies(path):
    """function -> its instructions' !dbg ids (None without one)."""
    cur, out = None, collections.defaultdict(list)
    for line in open(path):
        if line.startswith("define"):
            cur = re.search(r'@"?([\w.$]+)', line).group(1)
        elif line.startswith("}"):
            cur = None
        elif cur and INSTR.match(line):
            m = re.search(r"!dbg !(\d+)", line)
            out[cur].append(m.group(1) if m else None)
    return out


def entry_of(b):
    return max(b, key=lambda k: len(b[k]))


def attribute(path, top, who=()):
    text = open(path).read().split("\n")
    loc, sub, blk = {}, {}, {}
    rl = re.compile(r"^!(\d+) = (?:distinct )?!DILocation\(line: \d+, column: \d+, scope: !(\d+)(?:, inlinedAt: !(\d+))?")
    rs = re.compile(r'^!(\d+) = (?:distinct )?!DISubprogram\(name: "([^"]+)"')
    rb = re.compile(r"^!(\d+) = (?:distinct )?!DILexicalBlock(?:File)?\(scope: !(\d+)")
    for line in text:
        m = rl.match(line)
        if m:
            loc[m.group(1)] = (m.group(2), m.group(3))
            continue
        m = rs.match(line)
        if m:
            sub[m.group(1)] = m.group(2)
            continue
        m = rb.match(line)
        if m:
            blk[m.group(1)] = m.group(2)

    def name(scope):
        while scope in blk:
            scope = blk[scope]
        return re.sub(r"_\d+_?$", "", sub.get(scope, "?"))

    memo = {}

    def chain(d):
        if d not in memo:
            scope, at = loc[d]
            memo[d] = [(name(scope), at)] + (chain(at) if at else [])
        return memo[d]

    b = bodies(path)
    e = entry_of(b)
    incl, own, copies = collections.Counter(), collections.Counter(), collections.defaultdict(set)
    for d in b[e]:
        if d is None or d not in loc:
            own["(no location)"] += 1
            continue
        ch = chain(d)
        own[ch[0][0]] += 1
        seen = set()
        for n, at in ch:
            if n not in seen:
                seen.add(n)
                incl[n] += 1
                copies[n].add(at)
    total = len(b[e])
    for w in who:
        paths = collections.Counter()
        for d in b[e]:
            if d is None or d not in loc:
                continue
            ch = chain(d)
            names = [n for n, _ in ch]
            if w in names:
                i = names.index(w)
                paths[" <- ".join(names[i:i + 6])] += 1
        print(f"\n{w}: {sum(paths.values())} instructions, by call path (innermost first)")
        for p_, c in paths.most_common(12):
            print(f"  {c:7d}  {p_}")
    print(f"\n{e}: {total} instructions after inlining and optimization")
    print("Heaviest duplication (functions of 50+ instructions a copy; elementary arithmetic is "
          "not duplication): instructions with all it inlined, copies that survived, what one "
          "call site would save (upper bound; nested items overlap)")
    rows = [(c - c / len(copies[n]), c, len(copies[n]), n) for n, c in incl.items()
            if len(copies[n]) > 1 and c / len(copies[n]) >= 50]
    for dup, c, k, n in sorted(rows, reverse=True)[:top]:
        print(f"  {int(dup):7d} saved  {c:7d} in {k:4d} copies ({c // k:5d} each)  {n}")
    print("Largest functions (all copies, inclusive):")
    for n, c in incl.most_common(top):
        print(f"  {c:7d} {100 * c / total:5.1f}%  own {own[n]:6d}  copies {len(copies[n]):5d}  {n}")


def main():
    args = sys.argv[1:]
    top, jobs = 0, os.cpu_count() // 2 or 1
    if "--attr" in args:
        i = args.index("--attr")
        top = int(args.pop(i + 1))
        args.pop(i)
    who = []
    while "--who" in args:
        i = args.index("--who")
        who.append(args.pop(i + 1))
        args.pop(i)
    if "--jobs" in args:
        i = args.index("--jobs")
        jobs = int(args.pop(i + 1))
        args.pop(i)
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(os.path.join(a, f) for f in os.listdir(a) if f.endswith(".wgsl"))
        else:
            paths.append(a)
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    tool = to_msl_tool()
    with tempfile.TemporaryDirectory() as work:
        def one(p):
            ir = optimized(tool, p, work, debug=bool(top) and len(paths) == 1)
            n = len(bodies(ir)[entry_of(bodies(ir))])
            return p, n, ir

        with cf.ThreadPoolExecutor(jobs) as ex:
            rows = list(ex.map(one, paths))
        print(f"{"module":60s} {'instructions':>12s} {'predicted s':>12s}")
        for p, n, _ in sorted(rows, key=lambda r: -r[1]):
            s = K * n ** B
            print(f"{os.path.relpath(p, ENGINE)[-60:]:60s} {n:12d} {s:12.1f}{'  OVER' if s > CAP else ''}")
        if top:
            p = max(rows, key=lambda r: r[1])[0]
            attribute(optimized(tool, p, work, debug=True), top, who)
    return 0


if __name__ == "__main__":
    sys.exit(main())
