#!/usr/bin/env python3
"""Checks the build-architecture rules (engine/docs/build-architecture.md).

    python3 tools/build-time/check.py            (from engine/; CI runs it)
    python3 tools/build-time/check.py --shrink   also lowers each allowlisted size to
                                                 the file's size now (after a split)

Exits 1 with one line per broken rule:

- engine-kernel-manifest depends on nothing.
- The kernel build scripts (engine-webgpu, engine-metal, engine-cuda, engine-viewer)
  build-depend on no engine crate but engine-slang-build and engine-kernel-manifest.
- The runtime (engine-exec, the executors, the World, the FFI) does not depend on
  engine-softbody-authoring (dev-dependencies may).
- No build script or build tool spawns slangc or nvcc itself: engine-slang-build does
  (`Command::new(find_slangc` / `find_nvcc` anywhere else).
- Every loop cap in slang/kernels names its loop by an id of its own
  (`EB_END(ID, ..)`, `EB_PASS(ID, ..)`, `ebLoopCapHit(ID)`; explicit_loop_cap.slang),
  each used once (engine-webgpu's tests/explicit_loop_cap.rs checks the same).
- Function size: no Rust function over 200 lines, except those in
  fn-size-allowlist.txt, each at most its listed size (they may shrink, not grow).
  rustc compiles a function as one unit: an edit inside one of 3000 lines rebuilt the
  crate 0.7 s slower than one inside a small one (engine/docs/measurements/build-time.md),
  and the reader carries all of it.
- Size: no Rust or Slang source over 1000 lines (the target is about 500), except the
  files in size-allowlist.txt, each at most its listed size (they may shrink, not
  grow). A file that must grow past its line is split, or its line raised in the same
  change, where review sees it. A Slang file is split into parts it `#include`s, proved
  lossless with tools/build-time/kernel_digest.py (STRIP_LINES=1).
"""
import json
import os
import re
import subprocess
import sys

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ALLOWLIST = os.path.join(ENGINE, "tools", "build-time", "size-allowlist.txt")
FN_ALLOWLIST = os.path.join(ENGINE, "tools", "build-time", "fn-size-allowlist.txt")
FN_LIMIT = 200
LIMITS = {".rs": 1000, ".slang": 1000}
# The size a file aims for (reported, not enforced).
TARGET = 500
SCAN = ["crates", "slang", "experiments"]

KERNEL_BUILDS = ["engine-webgpu", "engine-metal", "engine-cuda", "engine-viewer"]
BUILD_DEPS_ALLOWED = {"engine-slang-build", "engine-kernel-manifest"}
RUNTIME = ["engine-exec", "engine-webgpu", "engine-metal", "engine-cuda", "engine-world", "engine-ffi"]


def metadata():
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ENGINE, check=True, capture_output=True, text=True,
    ).stdout
    return {p["name"]: p for p in json.loads(out)["packages"]}


def deps(pkg, kind):
    return {d["name"] for d in pkg["dependencies"] if d.get("kind") == kind}


def check_crates(problems):
    pkgs = metadata()
    manifest = pkgs.get("engine-kernel-manifest")
    if manifest is None:
        problems.append("engine-kernel-manifest is not in the workspace")
    elif manifest["dependencies"]:
        problems.append(f"engine-kernel-manifest depends on {sorted(d['name'] for d in manifest['dependencies'])}: it must depend on nothing")
    for name in KERNEL_BUILDS:
        engine = {d for d in deps(pkgs[name], "build") if d.startswith("engine-") or d.startswith("stress-")}
        extra = engine - BUILD_DEPS_ALLOWED
        if extra:
            problems.append(f"{name}'s build script depends on {sorted(extra)}: kernel build scripts read engine-kernel-manifest only")
    for name in RUNTIME:
        if "engine-softbody-authoring" in deps(pkgs[name], None):
            problems.append(f"{name} depends on engine-softbody-authoring: the runtime builds on engine-softbody (a dev-dependency may)")


def check_spawns(problems):
    pattern = re.compile(r"Command::new\(\s*(?:&?\s*)?(?:engine_slang_build::)?find_(slangc|nvcc)\b")
    for root, _, files in os.walk(os.path.join(ENGINE, "crates")):
        if "/target" in root or "engine-slang-build" in root:
            continue
        for f in files:
            if not f.endswith(".rs"):
                continue
            path = os.path.join(root, f)
            for n, line in enumerate(open(path, errors="replace"), 1):
                if pattern.search(line):
                    rel = os.path.relpath(path, ENGINE)
                    problems.append(f"{rel}:{n} spawns {pattern.search(line).group(1)} itself: compile through engine-slang-build (cached, batched)")


def check_loop_ids(all_problems):
    problems = []
    kernels = os.path.join(ENGINE, "slang", "kernels")
    seen = {}
    call = re.compile(r"(?<![A-Za-z0-9_])(EB_END_UNIFORM|EB_END|EB_PASS_UNIFORM|EB_PASS|ebLoopCapHit)\(\s*([^,)]*)")
    for name in sorted(os.listdir(kernels)):
        if not name.endswith(".slang") or name == "explicit_loop_cap.slang":
            continue
        for n, line in enumerate(open(os.path.join(kernels, name)), 1):
            code = line.split("//", 1)[0]
            for m in call.finditer(code):
                arg = m.group(2).strip()
                if not arg.isdigit() or int(arg) == 0:
                    problems.append(f"slang/kernels/{name}:{n}: {m.group(1)}({arg}, ..) has no id (the loop's own number)")
                elif arg in seen:
                    problems.append(f"slang/kernels/{name}:{n}: loop-cap id {arg} is also {seen[arg]}'s")
                else:
                    seen[arg] = f"{name}:{n}"
    if problems and seen:
        problems.append(f"(the next free loop-cap id: {max(int(k) for k in seen) + 1})")
    all_problems.extend(problems)


FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|async\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*fn\s+([A-Za-z_][A-Za-z0-9_]*)")


def fn_sizes():
    """`path::fn` -> lines, for every Rust function (brace matching from the first `{`
    after its name; strings, chars and comments are skipped)."""
    found = {}
    for top in SCAN:
        for root, dirs, files in os.walk(os.path.join(ENGINE, top)):
            dirs[:] = [d for d in dirs if d not in ("target", "node_modules", "generated")]
            for f in files:
                if not f.endswith(".rs"):
                    continue
                path = os.path.join(root, f)
                rel = os.path.relpath(path, ENGINE)
                lines = open(path, errors="replace").read().split("\n")
                for i, line in enumerate(lines):
                    m = FN_RE.match(line)
                    if not m:
                        continue
                    n = body_lines(lines, i)
                    if n is not None:
                        key = f"{rel}::{m.group(1)}"
                        found[key] = max(found.get(key, 0), n)
    return found


def body_lines(lines, start):
    depth, opened = 0, False
    for j in range(start, min(len(lines), start + 20000)):
        text = re.sub(r'"(?:\\.|[^"\\])*"', '""', lines[j])
        text = re.sub(r"'(?:\\.|[^'\\])'", "''", text)
        text = text.split("//", 1)[0]
        for c in text:
            if c == "{":
                depth += 1
                opened = True
            elif c == "}":
                depth -= 1
                if opened and depth == 0:
                    return j - start + 1
            elif c == ";" and not opened:
                return None  # a declaration without a body (trait item, extern fn)
    return None


def load_fn_allowlist():
    allowed = {}
    if os.path.exists(FN_ALLOWLIST):
        for line in open(FN_ALLOWLIST):
            line = line.split("#", 1)[0].strip()
            if line:
                key, n = line.rsplit(None, 1)
                allowed[key] = int(n)
    return allowed


def check_fn_sizes(problems, shrink, init):
    found = fn_sizes()
    allowed = load_fn_allowlist()
    if init:
        allowed = {k: n for k, n in found.items() if n > FN_LIMIT}
    for key, n in sorted(found.items()):
        if key in allowed:
            if n > allowed[key]:
                problems.append(f"{key}: {n} lines, more than its {allowed[key]} in fn-size-allowlist.txt (split it, or raise the line)")
        elif n > FN_LIMIT:
            problems.append(f"{key}: {n} lines, past the {FN_LIMIT}-line function limit (split it)")
    gone = [k for k in allowed if found.get(k, 0) <= FN_LIMIT]
    if shrink or init:
        for k in gone:
            del allowed[k]
        for k in allowed:
            allowed[k] = min(allowed[k], found[k])
        with open(FN_ALLOWLIST, "w") as fh:
            fh.write(
                "# Rust functions over 200 lines (tools/build-time/check.py) when the limit was set,\n"
                "# each with the most lines it may have. They may shrink (check.py --shrink lowers the\n"
                "# numbers), not grow: split one instead, or raise its line in the same change.\n"
            )
            for k in sorted(allowed):
                fh.write(f"{k} {allowed[k]}\n")
    elif gone:
        problems.append(f"fn-size-allowlist.txt lists functions now under the limit or gone ({', '.join(gone[:5])}): run check.py --shrink")


def sizes():
    found = {}
    for top in SCAN:
        for root, dirs, files in os.walk(os.path.join(ENGINE, top)):
            dirs[:] = [d for d in dirs if d not in ("target", "node_modules", "generated")]
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext in LIMITS:
                    path = os.path.join(root, f)
                    with open(path, errors="replace") as fh:
                        found[os.path.relpath(path, ENGINE)] = sum(1 for _ in fh)
    return found


def load_allowlist():
    allowed = {}
    if os.path.exists(ALLOWLIST):
        for line in open(ALLOWLIST):
            line = line.split("#", 1)[0].strip()
            if line:
                path, n = line.rsplit(None, 1)
                allowed[path] = int(n)
    return allowed


def write_allowlist(allowed):
    with open(ALLOWLIST, "w") as fh:
        fh.write(
            "# Sources over the size limit (tools/build-time/check.py: 1000 lines, Rust and\n"
            "# Slang; about 500 the target) when the limit was set, each with the most lines it may have.\n"
            "# They may shrink (check.py --shrink lowers the numbers), not grow: split a\n"
            "# file instead, or raise its line in the same change, where review sees it.\n"
        )
        for path in sorted(allowed):
            fh.write(f"{path} {allowed[path]}\n")


def check_sizes(problems, shrink):
    found = sizes()
    allowed = load_allowlist()
    for path, n in sorted(found.items()):
        limit = LIMITS[os.path.splitext(path)[1]]
        if path in allowed:
            if n > allowed[path]:
                problems.append(f"{path}: {n} lines, more than its {allowed[path]} in size-allowlist.txt (split it, or raise the line)")
        elif n > limit:
            problems.append(f"{path}: {n} lines, past the {limit}-line limit (split it)")
    gone = [p for p in allowed if p not in found or found[p] <= LIMITS[os.path.splitext(p)[1]]]
    if shrink:
        for p in gone:
            del allowed[p]
        for p in allowed:
            allowed[p] = min(allowed[p], found[p])
        write_allowlist(allowed)
    elif gone:
        problems.append(f"size-allowlist.txt lists files now under the limit or gone ({', '.join(gone[:5])}): run check.py --shrink")


def main():
    shrink = "--shrink" in sys.argv[1:]
    if "--init" in sys.argv[1:]:
        found = sizes()
        write_allowlist({p: n for p, n in found.items() if n > LIMITS[os.path.splitext(p)[1]]})
        check_fn_sizes([], False, True)
        return 0
    if "--init-fns" in sys.argv[1:]:
        check_fn_sizes([], False, True)
        return 0
    problems = []
    check_crates(problems)
    check_spawns(problems)
    check_loop_ids(problems)
    check_sizes(problems, shrink)
    check_fn_sizes(problems, shrink, False)
    for p in problems:
        print(p)
    over = sum(1 for n in sizes().values() if n > TARGET)
    if not problems:
        print(f"build architecture: ok ({over} sources past the {TARGET}-line target)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
