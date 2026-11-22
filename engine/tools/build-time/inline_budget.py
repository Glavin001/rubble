#!/usr/bin/env python3
"""Each WGSL module's inlined size against its budget (inline-budget.txt).

    python3 tools/build-time/inline_budget.py [WGSL_DIR]          (from engine/)
    python3 tools/build-time/inline_budget.py --init [WGSL_DIR]

A kernel's compile time in the browser (and on Metal) follows the size of its entry
point once every call is inlined, faster than linearly
(docs/measurements/kernel-compile-time.md; scripts/wgsl-inline-size.py measures it).
This keeps it from growing unseen: every explicit module (`explicit_*.wgsl`, the
kernels whose compiles take tens of seconds) has a budget of inlined lines, the size
when the budget was set plus 10 %. Past it the check fails: shrink the module (one call
site per heavy helper, a split by dispatch kind), or raise its line in the same change,
where review sees the cost.

WGSL_DIR: the build's WGSL (default: the newest engine-webgpu OUT_DIR/wgsl under
target/). --init writes the budget from the modules there.
"""
import glob
import importlib.util
import os
import sys

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUDGET = os.path.join(ENGINE, "tools", "build-time", "inline-budget.txt")
HEADROOM = 1.10


def load_tool():
    spec = importlib.util.spec_from_file_location("wgsl_inline_size", os.path.join(ENGINE, "scripts", "wgsl-inline-size.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wgsl_dir(args):
    if args:
        return args[0]
    dirs = glob.glob(os.path.join(ENGINE, "target", "*", "build", "engine-webgpu-*", "out", "wgsl"))
    dirs = [d for d in dirs if glob.glob(os.path.join(d, "explicit_*.wgsl"))]
    if not dirs:
        sys.exit("no engine-webgpu OUT_DIR/wgsl with explicit modules under target/: build engine-webgpu first")
    return max(dirs, key=os.path.getmtime)


def sizes(d):
    """module -> its entry points' inlined lines (the largest entry), from the tool's own analysis."""
    import contextlib
    import io
    import re
    tool = load_tool()
    out = {}
    row = re.compile(r"^(\S+)\s+\d+\s+\d+\s+(\d+)\s+\d+\s+\d+\s+\d+\s*$")
    files = sorted(glob.glob(os.path.join(d, "explicit_*.wgsl")))
    buf = io.StringIO()
    saved = sys.argv
    try:
        sys.argv = ["wgsl-inline-size.py", *files, "--top", "0"]
        with contextlib.redirect_stdout(buf):
            tool.main()
    finally:
        sys.argv = saved
    for line in buf.getvalue().splitlines():
        m = row.match(line)
        if m:
            out[m.group(1)] = max(out.get(m.group(1), 0), int(m.group(2)))
    return out


def main():
    args = sys.argv[1:]
    init = "--init" in args
    args = [a for a in args if a != "--init"]
    found = sizes(wgsl_dir(args))
    if not found:
        sys.exit("no explicit modules measured")
    if init:
        with open(BUDGET, "w") as fh:
            fh.write(
                "# Inlined lines a WGSL module's entry point may have (tools/build-time/inline_budget.py):\n"
                "# the size when set plus 10 %. Past it: shrink the module, or raise its line here in\n"
                "# the same change, where review sees the compile-time cost.\n"
            )
            for m in sorted(found):
                fh.write(f"{m} {int(found[m] * HEADROOM) + 1}\n")
        print(f"{len(found)} modules -> {BUDGET}")
        return 0
    budget = {}
    for line in open(BUDGET):
        line = line.split("#", 1)[0].strip()
        if line:
            m, n = line.rsplit(None, 1)
            budget[m] = int(n)
    bad = []
    for m, n in sorted(found.items()):
        if m not in budget:
            bad.append(f"{m}: {n} inlined lines, no budget (add it: tools/build-time/inline-budget.txt)")
        elif n > budget[m]:
            bad.append(f"{m}: {n} inlined lines, past its budget of {budget[m]} (shrink it, or raise the line)")
    for b in bad:
        print(b)
    print(f"inline budget: {len(found)} modules, {len(bad)} past their budget")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
