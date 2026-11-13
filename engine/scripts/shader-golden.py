#!/usr/bin/env python3
"""Generated-shader goldens: the kernels a build wrote, against a recorded list.

    shader-golden.py record OUT_DIR... > golden.tsv
    shader-golden.py check golden.tsv [--except exceptions.tsv] OUT_DIR...

OUT_DIR is a build script's output (target/<profile>/build/engine-webgpu-*/out or
engine-metal-*/out): its `wgsl`, `spirv`, `rigid_msl`, `terrain_msl`, `msl` and
`cuda` subdirectories are listed as `<subdir>/<file>` with an FNV-1a 64 hash of the
bytes (the same hash `engine-webgpu/tests/main_wgsl_golden.rs` uses); `#line`
directives (the checkout's path and source line numbers) are left out of it.

`check` fails (exit 1) if a golden file is missing or its bytes changed, unless a
line of the exceptions file (`<fnmatch pattern>\t<reason>`) covers it; files the
golden does not list are new and fine. An exception that no longer applies (the file
is identical again) is reported so the list stays honest. CPU only: nothing runs.
"""
import fnmatch
import os
import sys

SUBDIRS = ("wgsl", "spirv", "rigid_msl", "terrain_msl", "msl", "cuda")


def fnv(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def listing(dirs):
    out = {}
    for d in dirs:
        for sub in SUBDIRS:
            root = os.path.join(d, sub)
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                # Reflection JSON carries no code; hash the kernels only.
                if os.path.isfile(path) and not name.endswith(".json"):
                    with open(path, "rb") as f:
                        data = f.read()
                    # `#line N "path"` names the checkout and the source line: not code.
                    if sub != "spirv":
                        data = b"\n".join(l for l in data.split(b"\n") if not l.startswith(b"#line "))
                    out[f"{sub}/{name}"] = fnv(data)
    return out


def main(argv):
    if len(argv) >= 3 and argv[1] == "record":
        for k, h in sorted(listing(argv[2:]).items()):
            print(f"{k}\t{h}")
        return 0
    if len(argv) >= 4 and argv[1] == "check":
        args = argv[3:]
        exceptions = []
        if args[0] == "--except":
            with open(args[1]) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line and not line.startswith("#"):
                        pattern, _, reason = line.partition("\t")
                        exceptions.append((pattern, reason))
            args = args[2:]
        now = listing(args)
        subs = {k.split("/")[0] for k in now}
        bad, excused, stale, same = [], [], set(p for p, _ in exceptions), 0
        with open(argv[2]) as f:
            for line in f:
                key, _, want = line.rstrip("\n").partition("\t")
                if not key or key.split("/")[0] not in subs:
                    continue
                got = now.get(key)
                hit = next((p for p, _ in exceptions if fnmatch.fnmatch(key, p)), None)
                if got == want:
                    same += 1
                    continue
                if hit is not None:
                    excused.append(key)
                    stale.discard(hit)
                else:
                    bad.append(f"{key}: {'missing' if got is None else 'differs'}")
        print(f"{same} identical, {len(excused)} differ as listed, {len(bad)} differ unlisted, {len(now)} files now")
        for k in bad:
            print("  " + k)
        for p in sorted(stale):
            print(f"  exception `{p}` matches no changed file: remove it")
        return 1 if bad or stale else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
