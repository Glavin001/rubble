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
- Size: no Rust or Slang source over its limit (Rust 1500 lines, Slang 3000), except
  the files in size-allowlist.txt, each at most its listed size (they may shrink, not
  grow). A file that must grow past its line is split, or its line raised in the same
  change, where review sees it.
"""
import json
import os
import re
import subprocess
import sys

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ALLOWLIST = os.path.join(ENGINE, "tools", "build-time", "size-allowlist.txt")
LIMITS = {".rs": 1500, ".slang": 3000}
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
            "# Sources over the size limit (tools/build-time/check.py: Rust 1500 lines,\n"
            "# Slang 3000) when the limit was set, each with the most lines it may have.\n"
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
        return 0
    problems = []
    check_crates(problems)
    check_spawns(problems)
    check_sizes(problems, shrink)
    for p in problems:
        print(p)
    if not problems:
        print("build architecture: ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
