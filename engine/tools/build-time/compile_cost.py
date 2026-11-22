#!/usr/bin/env python3
"""Each GPU kernel's first-load compile time, predicted from its WGSL, and the
burn-down list of what to fix. No browser, no GPU, about a second.

    python3 tools/build-time/compile_cost.py [WGSL_DIR]               (from engine/)
    python3 tools/build-time/compile_cost.py --burndown 20 [WGSL_DIR]
    python3 tools/build-time/compile_cost.py --check [WGSL_DIR]       (CI)
    python3 tools/build-time/compile_cost.py --calibrate TIMES.txt [WGSL_DIR]
    python3 tools/build-time/compile_cost.py --init [WGSL_DIR]

Why a prediction and not a line count: GPU compilers (Apple's Metal back end under
Chrome, NVIDIA's) inline every call into the kernel's entry point and their time grows
faster than linearly with that one function's size. A helper of 30 statements called
from 450 places is 13,500 statements for the compiler, however small the source looks.
The model counts the statements of the entry point with every call inlined, skipping
one-statement helpers (`nf(x)` is `return x;` in the fast setting: the compiler folds
it away), and maps them to seconds with a power law fitted to measured cold compiles:

    seconds = K * statements ^ B          (compile-cost-model.txt)

Calibrated 2026-11-22 on an M3 Max, Chrome (Metal), 28 vehicle-lab modules from 0.01 s
to 119 s: within 1.15x on the modules over 10 s, 1.5x on the 1-3 s ones. It ranks and
gates; a number for a doc still comes from a measured cold compile
(`explicit-wgsl-check.mjs` with EXPLICIT_WGSL_COLD=1, or natively and without a browser
`EXPLICIT_WGSL_DIR=DIR cargo run -p engine-webgpu-lab --example explicit_compile_time`,
which agrees with Chrome within about 15 %). --calibrate refits the model from either's
output.

--burndown N: per helper, the seconds the kernels over the cap would lose if that
helper's copies were one (a loop over its cases, or a select of its arguments, instead
of a call per case: docs/measurements/kernel-compile-time.md), summed over those
kernels, with the callers that multiply it. An upper bound per item (items overlap: a
helper inside another counts in both), for ranking.

--check: fails when a kernel is predicted past CAP_SECONDS and is not in
compile-cost-allowlist.txt, or an allowlisted one is predicted slower than its line
(the line may only come down: --init rewrites it from today's kernels).

WGSL_DIR: the build's WGSL (default: the fast loop's kernel store, as inline_budget.py
reads it in CI: `cargo run -q --release -p engine-webgpu --no-default-features --bin
kernels`). Modules are taken in the fast setting (bit-exact results off, the pages'
default; engine-webgpu's fast_float.rs) unless --strict.
"""
import glob
import math
import os
import re
import sys

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.join(ENGINE, "tools", "build-time")
MODEL = os.path.join(HERE, "compile-cost-model.txt")
ALLOW = os.path.join(HERE, "compile-cost-allowlist.txt")
CAP_SECONDS = 10.0
HEADROOM = 1.10

WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
FN = re.compile(r"^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
ENTRY = re.compile(r"@compute[^\n]*\n(?:@[^\n]*\n)*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# engine-webgpu's fast_float.rs: the fast setting's bodies of the strict helpers.
FAST = {"nf_0": lambda p: f"return {p[0]};", "sdiv_0": lambda p: f"return {p[0]} / {p[1]};",
        "ssqrt_0": lambda p: f"return sqrt({p[0]});"}


def functions(src):
    out = {}
    for m in FN.finditer(src):
        i = src.index("{", m.end())
        depth, j = 0, i
        while j < len(src):
            c = src[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[m.group(1)] = (m.start(), src[m.end():j + 1])
    return out


def fast(src):
    for name, rule in FAST.items():
        fns = functions(src)  # (offsets move with each rewrite)
        if name not in fns:
            continue
        start, body = fns[name]
        head = src[start:src.index("{", start)]
        params = [p.split(":")[0].strip() for p in head[head.index("(") + 1:head.rindex(")")].split(",")]
        src = src.replace(head + body[body.index("{"):], head + "{\n    " + rule(params) + "\n}", 1)
    return src


class Module:
    """One module's call graph: each function's own statements and its calls."""

    def __init__(self, path, strict=False):
        src = open(path).read()
        if not strict:
            src = fast(src)
        fns = functions(src)
        names = set(fns)
        self.own, self.calls = {}, {}
        for name, (_, body) in fns.items():
            calls = {}
            for w in WORD.findall(body):
                if w in names and w != name:
                    calls[w] = calls.get(w, 0) + 1
            self.calls[name] = calls
            stm = body.count(";")
            trivial = stm <= 1 and not re.search(r"\b(?:loop|for|while)\b", body)
            self.own[name] = 0 if trivial else stm
        self.entries = [m.group(1) for m in ENTRY.finditer(src)]
        self._inl = {}

    def inlined(self, f):
        if f not in self._inl:
            self._inl[f] = self.own[f] + sum(n * self.inlined(c) for c, n in self.calls[f].items())
        return self._inl[f]

    def copies(self, root):
        count = {}

        def walk(f, mult):
            count[f] = count.get(f, 0) + mult
            for c, n in self.calls[f].items():
                walk(c, mult * n)
        walk(root, 1)
        return count


def model():
    k, b = 2.327e-07, 1.758
    if os.path.exists(MODEL):
        for line in open(MODEL):
            line = line.split("#", 1)[0].split()
            if len(line) == 2 and line[0] == "K":
                k = float(line[1])
            elif len(line) == 2 and line[0] == "B":
                b = float(line[1])
    return k, b


def seconds(stm, kb):
    return kb[0] * max(stm, 1) ** kb[1]


def wgsl_dir(args):
    if args:
        return args[0]
    dirs = glob.glob(os.path.join(ENGINE, "target", "*", "kernel-store", "*", "wgsl"))
    dirs += glob.glob(os.path.join(ENGINE, "target", "*", "build", "engine-webgpu-*", "out", "wgsl"))
    dirs = [d for d in dirs if glob.glob(os.path.join(d, "explicit_*.wgsl"))]
    if not dirs:
        sys.exit("no WGSL with explicit modules under target/: "
                 "cargo run -q --profile fast -p engine-webgpu --bin kernels")
    return max(dirs, key=os.path.getmtime)


def measure(d, strict):
    """module -> (Module, its largest entry, inlined statements)."""
    sys.setrecursionlimit(20000)
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "explicit_*.wgsl"))):
        m = Module(p, strict)
        if not m.entries:
            continue
        e = max(m.entries, key=m.inlined)
        out[os.path.basename(p)[:-5]] = (m, e, m.inlined(e))
    return out


def burndown(found, kb, top):
    over = {n: v for n, v in found.items() if seconds(v[2], kb) > CAP_SECONDS}
    gain, where = {}, {}
    for name, (m, e, stm) in over.items():
        base = seconds(stm, kb)
        copies = m.copies(e)
        for f, n in copies.items():
            if n < 2 or f == e:
                continue
            sub = m.inlined(f)
            if sub == 0:
                continue
            s = base - seconds(stm - (n - 1) * sub, kb)
            key = re.sub(r"_\d+$", "", f)
            gain[key] = gain.get(key, 0.0) + s
            who = sorted(((m.calls[c][f] * copies[c], m.calls[c][f], copies[c], c)
                          for c in copies if f in m.calls[c]), reverse=True)[:3]
            where.setdefault(key, []).append((s, name, n, sub, who))
    print(f"burn-down: {len(over)} kernels predicted past {CAP_SECONDS:.0f} s; "
          f"seconds saved if each helper had one copy (upper bound, items overlap)")
    for key, s in sorted(gain.items(), key=lambda x: -x[1])[:top]:
        rows = sorted(where[key], reverse=True)
        s0, mod, n, sub, who = rows[0]
        why = ", ".join(f"{re.sub(r'_[0-9]+$', '', c)} {k} sites x {cp}" for _, k, cp, c in who)
        print(f"  {s:8.1f} s  {key:28s} {sub:6d} stm x {n:5d} copies in {mod} ({len(rows)} kernels); from {why}")


def calibrate(found, times_path):
    t = {}
    for line in open(times_path):
        # explicit-wgsl-check.mjs: "ok   NAME 541 KB 41.14 s"; explicit_compile_time: "NAME  541 KB  41.14 s"
        m = re.match(r"^(?:ok\s+)?(explicit_\S+)\s+\d+ KB\s+([\d.]+) s", line.strip())
        if m:
            t[m.group(1)] = float(m.group(2))
    pts = [(math.log(found[n][2]), math.log(s)) for n, s in t.items() if n in found and s >= 1.0 and found[n][2] > 0]
    if len(pts) < 3:
        sys.exit(f"need 3+ modules of 1 s or more in both ({len(pts)})")
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    k = math.exp(my - b * mx)
    print(f"K {k:.4g}\nB {b:.4f}\n# from {len(pts)} modules of {times_path}")
    for n, s in sorted(t.items(), key=lambda x: -x[1]):
        if n in found:
            print(f"  {n:44s} measured {s:7.2f} s  predicted {k * found[n][2] ** b:7.2f} s")


def main():
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    top = 20
    if "--burndown" in args:
        i = args.index("--burndown")
        if i + 1 < len(args) and args[i + 1].isdigit():
            top = int(args.pop(i + 1))
    times = None
    if "--calibrate" in args:
        i = args.index("--calibrate")
        times = args.pop(i + 1)
    rest = [a for a in args if not a.startswith("--")]
    d = wgsl_dir(rest)
    found = measure(d, "--strict" in flags)
    kb = model()
    if times:
        return calibrate(found, times) or 0
    pred = {n: seconds(v[2], kb) for n, v in found.items()}
    if "--init" in flags:
        with open(ALLOW, "w") as fh:
            fh.write(f"# Kernels predicted past {CAP_SECONDS:.0f} s of first-load compile "
                     "(tools/build-time/compile_cost.py), each with the seconds it may not exceed:\n"
                     "# today's prediction plus 10 %. Lines may only come down; a kernel leaves the list\n"
                     "# when it is under the cap. --burndown says what to fix first.\n")
            for n in sorted(pred, key=lambda n: -pred[n]):
                if pred[n] > CAP_SECONDS:
                    fh.write(f"{n} {pred[n] * HEADROOM:.1f}\n")
        print(f"{sum(p > CAP_SECONDS for p in pred.values())} kernels -> {ALLOW}")
        return 0
    if "--check" in flags:
        allow = {}
        if os.path.exists(ALLOW):
            for line in open(ALLOW):
                line = line.split("#", 1)[0].split()
                if len(line) == 2:
                    allow[line[0]] = float(line[1])
        bad = []
        for n, s in sorted(pred.items()):
            if n in allow and s > allow[n]:
                bad.append(f"{n}: predicted {s:.1f} s, past its line of {allow[n]:.1f} s (shrink it: --burndown)")
            elif n not in allow and s > CAP_SECONDS:
                bad.append(f"{n}: predicted {s:.1f} s, past the {CAP_SECONDS:.0f} s cap (shrink it: --burndown)")
        stale = [n for n in allow if n in pred and pred[n] <= CAP_SECONDS]
        for b in bad:
            print(b)
        for n in stale:
            print(f"note: {n} is under the cap now: remove its line from {os.path.basename(ALLOW)}")
        print(f"compile cost: {len(pred)} kernels, {len(allow)} allowlisted past the cap, {len(bad)} failing")
        return 1 if bad else 0
    print(f"{'kernel':44s} {'statements':>11s} {'predicted s':>12s}   (inlined; {d})")
    for n in sorted(pred, key=lambda n: -pred[n]):
        print(f"{n:44s} {found[n][2]:11d} {pred[n]:12.1f}{'  OVER' if pred[n] > CAP_SECONDS else ''}")
    if "--burndown" in flags:
        print()
        burndown(found, kb, top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
