#!/usr/bin/env python3
"""Splits one package's `cargo test` over parallel CI jobs without leaving anything out.

Usage (from engine/): host-test-shards.py PACKAGE SHARD TOTAL

Prints what shard SHARD (1-based) of TOTAL runs, one word per line:
  test <cargo arg>      an argument of `cargo test` (--lib, --bin=NAME, --test=NAME)
  example <NAME>        an example `cargo build` compiles (what a plain `cargo test`
                        does with examples: it builds them and runs nothing)
  doc                   this shard also runs the package's doc tests

The targets come from `cargo metadata`, so a new test file or example is in a shard
without anyone listing it, and the shards together are exactly what
`cargo test -p PACKAGE` builds and runs. Only the balance between shards depends on
the seconds below.
"""
import json
import subprocess
import sys

package, shard, total = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
assert 1 <= shard <= total, "SHARD is 1..TOTAL"

# Seconds a test binary ran on a GitHub runner (run 43824526694, 2026-11-16); the
# others ran under 6 s each. Add a test here when it grows past about 10 s.
RUN_SECONDS = {
    ("engine-softbody", "test", "showcase_car"): 231,
    ("engine-softbody", "test", "crash_invariants"): 134,
    ("engine-softbody", "test", "gen3_rig"): 119,
    ("engine-softbody", "test", "skin_lint"): 95,
    ("engine-softbody", "test", "sedan_energy"): 80,
    ("engine-softbody", "test", "showcase_skin_integrity"): 48,
    ("engine-softbody", "lib", "engine_softbody"): 25,
    ("engine-softbody", "test", "raycast_wheel_visual"): 25,
    ("engine-softbody", "test", "explicit_drive"): 16,
    ("engine-softbody", "test", "sedan_jello"): 14,
    ("engine-softbody", "test", "rig_export"): 14,
    ("engine-softbody", "test", "energy_ledger"): 14,
}
DEFAULT_RUN_SECONDS = 2
# Compiling and linking one target (release, thin LTO), on four cores: an estimate
# from the whole build (about 130 targets in 19.6 min).
BUILD_SECONDS = 9

meta = json.loads(
    subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
)
pkgs = [p for p in meta["packages"] if p["name"] == package]
assert len(pkgs) == 1, f"package {package} not in the workspace"

units = []  # (seconds, output line)
doc = False
for t in pkgs[0]["targets"]:
    kind, name = t["kind"][0], t["name"]
    assert not t.get("required-features"), (
        f"{name}: a target with required-features needs its own rule here"
    )
    if kind in ("lib", "rlib", "cdylib", "staticlib", "dylib", "proc-macro"):
        doc = doc or t.get("doctest", False)
        if t["test"]:
            units.append((BUILD_SECONDS + RUN_SECONDS.get((package, "lib", name), DEFAULT_RUN_SECONDS), "test --lib"))
    elif kind == "bin":
        if t["test"]:
            units.append((BUILD_SECONDS + RUN_SECONDS.get((package, "bin", name), DEFAULT_RUN_SECONDS), f"test --bin={name}"))
    elif kind == "test":
        if t["test"]:
            units.append((BUILD_SECONDS + RUN_SECONDS.get((package, "test", name), DEFAULT_RUN_SECONDS), f"test --test={name}"))
    elif kind == "example":
        # A plain `cargo test` runs an example's tests only with `test = true`.
        assert not t["test"], f"example {name} has test = true: it needs its own rule here"
        units.append((BUILD_SECONDS, f"example {name}"))
    elif kind in ("bench", "custom-build"):
        pass  # `cargo test` builds no benches; the build script is not a target to pick
    else:
        raise SystemExit(f"{name}: unknown target kind {kind}")

# Longest first into the lightest shard: deterministic, so every job of a run agrees.
loads = [0] * total
mine = []
for seconds, line in sorted(units, key=lambda u: (-u[0], u[1])):
    i = loads.index(min(loads))
    loads[i] += seconds
    if i == shard - 1:
        mine.append(line)
print(f"shard {shard}/{total}: {len(mine)} of {len(units)} targets, about {loads[shard - 1]} s of {sum(loads)} s", file=sys.stderr)
for line in mine:
    print(line)
# The doc tests go with the first shard.
if doc and shard == 1:
    print("doc")
