#!/usr/bin/env python3
"""Every generated kernel file's hash, to prove a build change is lossless.

    python3 tools/build-time/kernel_digest.py OUT.txt            (from engine/)
    python3 tools/build-time/kernel_digest.py --compare A.txt B.txt

The first form builds the libraries of the crates whose build scripts compile Slang
(engine-webgpu, engine-metal, engine-cuda, engine-viewer) twice, the shipped kernel set
and the set with engine-webgpu's test kernels, and writes one line per file their build
scripts wrote: `variant crate path sha256`. The second form prints what differs and
exits 1 when anything does.

A change to how kernels are built (caches, crate boundaries, where the lists live)
must leave this digest identical. A change to kernel source changes it on purpose and
is checked by the ladder in AGENTS.md instead.

PROFILE (default fast) picks the cargo profile; the kernels do not depend on it.
"""
import hashlib
import json
import os
import re
import subprocess
import sys

CRATES = ["engine-webgpu", "engine-metal", "engine-cuda", "engine-viewer"]
# (variant: the crates built and cargo's extra arguments). The test kernels are
# engine-webgpu's alone, and cargo applies `--features` only with that package selected
# by itself (`-p a -p b --features a/x` builds without it).
VARIANTS = {
    "ship": (CRATES, []),
    "test": (["engine-webgpu"], ["--features", "test-kernels"]),
}
# Files a build script writes that are not kernels or kernel tables: logs and stamps.
SKIP_SUFFIXES = (".log", ".stamp", ".tmp")
# MSL and CUDA name their source files in `#line` directives. Where a checkout lives
# (and whether slangc was given absolute or relative paths) changes only those names,
# not the code: they are compared from the Slang tree's root, `kernels/x.slang`.
LINE_SUFFIXES = (".metal", ".cu")
LINE_RE = re.compile(rb'^(#line \d+ ")([^"]*)(")', re.M)


# The CUDA tables record each `.cu` file's length (`cu_bytes`, read by nothing), which
# its `#line` names change; the `.cu` files themselves are compared (normalised).
CU_BYTES_RE = re.compile(rb"cu_bytes: \d+")


def normalise_line(m):
    path = m.group(2)
    at = path.rfind(b"/slang/")
    if at >= 0:
        path = path[at + len(b"/slang/"):]
    return m.group(1) + path + m.group(3)


def out_dirs(profile, crates, extra):
    """The OUT_DIR of each kernel crate's build script, from cargo's own messages."""
    cmd = ["cargo", "build", "--profile", profile, "--lib", "--message-format=json"]
    for c in crates:
        cmd += ["-p", c]
    cmd += extra
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.exit(f"build failed: {' '.join(cmd)}")
    found = {}
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "build-script-executed":
            continue
        pkg = msg["package_id"]
        for c in crates:
            # package_id: "path+file:///…/crates/engine-webgpu#0.1.0" (or the older
            # "engine-webgpu 0.1.0 (path+file://…)").
            if f"/{c}#" in pkg or pkg.startswith(f"{c} "):
                found[c] = msg["out_dir"]
    missing = [c for c in crates if c not in found]
    if missing:
        sys.exit(f"no build-script output for {missing}")
    return found


def digest_dir(root):
    rows = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(SKIP_SUFFIXES):
                continue
            p = os.path.join(d, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                data = fh.read()
            if f.endswith(LINE_SUFFIXES):
                data = LINE_RE.sub(normalise_line, data)
            if f.startswith("cuda_") and f.endswith(".rs"):
                data = CU_BYTES_RE.sub(b"cu_bytes: _", data)
            h.update(data)
            rows.append((os.path.relpath(p, root), h.hexdigest()))
    return sorted(rows)


def build_digest(out_path):
    profile = os.environ.get("PROFILE", "fast")
    lines = []
    for variant, (crates, extra) in VARIANTS.items():
        for crate, out in sorted(out_dirs(profile, crates, extra).items()):
            for rel, h in digest_dir(out):
                lines.append(f"{variant} {crate} {rel} {h}")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(lines)} files -> {out_path}")


def load(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 4:
                rows[" ".join(parts[:3])] = parts[3]
    return rows


def compare(a_path, b_path):
    a, b = load(a_path), load(b_path)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    for k in only_a:
        print(f"- {k}")
    for k in only_b:
        print(f"+ {k}")
    for k in changed:
        print(f"~ {k}")
    same = len(set(a) & set(b)) - len(changed)
    print(f"{same} identical, {len(changed)} changed, {len(only_a)} removed, {len(only_b)} added")
    return 1 if (only_a or only_b or changed) else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    if len(sys.argv) == 2:
        build_digest(sys.argv[1])
        sys.exit(0)
    sys.exit(__doc__)
