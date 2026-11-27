#!/usr/bin/env python3
"""Splits one package's `cargo test` over parallel CI jobs without leaving anything out.

Usage (from engine/): host-test-shards.py PACKAGE[,PACKAGE...] SHARD TOTAL

Prints what shard SHARD (1-based) of TOTAL runs, one line per target:
  <package> test <cargo arg>   an argument of `cargo test` (--lib, --bin=NAME, --test=NAME)
  <package> example <NAME>     an example to `cargo check` (a plain `cargo test` builds
                               examples and runs nothing; checking them finds the same
                               compile errors without generating and linking code)
  <package> doc                this shard also runs the package's doc tests

The targets come from `cargo metadata`, so a new test file or example is in a shard
without anyone listing it, and the shards together are exactly what
`cargo test -p PACKAGE ...` builds and runs. Only the balance between shards depends
on the seconds below.
"""
import json
import subprocess
import sys

packages, shard, total = sys.argv[1].split(","), int(sys.argv[2]), int(sys.argv[3])
assert 1 <= shard <= total, "SHARD is 1..TOTAL"

# Seconds a test binary ran on a GitHub runner (run 43824526694, 2026-11-16); the
# others ran under 6 s each. Add a test here when it grows past about 10 s.
RUN_SECONDS = {
    ("engine-softbody-authoring", "test", "showcase_car"): 231,
    ("engine-softbody-authoring", "test", "crash_invariants"): 134,
    ("engine-softbody-authoring", "test", "gen3_rig"): 119,
    ("engine-softbody-authoring", "test", "skin_lint"): 95,
    ("engine-softbody-authoring", "test", "sedan_energy"): 80,
    ("engine-softbody-authoring", "test", "showcase_skin_integrity"): 48,
    # Measured locally (15 s on an M-series Mac); a runner's estimate.
    ("engine-softbody-authoring", "test", "gen_sedan_contact"): 65,
    ("engine-softbody-authoring", "test", "gen_sedan_paint"): 480,
    # Continuous contact's acceptance crashes (eleven scenes and three energy tests, 8 or 2
    # runs each: 20-60 s each on an M-series Mac); a runner's estimate. It had no line and
    # counted 2 s, so its shard ran past the job's 25 minutes (2026-11-27).
    # Measured on a runner (2026-11-27): the one binary ran past 25 minutes, nine of its fifteen
    # tests done at 22; split in two (continuous_contact_hard: the six longest crashes).
    ("engine-softbody-authoring", "test", "continuous_contact"): 1000,
    ("engine-softbody-authoring", "test", "continuous_contact_hard"): 1100,
    # The two pressed crashes (2 s each): side 60 pressed alone did not finish in 23 minutes
    # beside the others.
    ("engine-softbody-authoring", "test", "continuous_contact_pressed"): 1100,
    # Velocity contact on three short crashes, 2 runs each (37 s on an M3, 2026-11-27).
    ("engine-softbody-authoring", "test", "continuous_contact_velocity"): 150,
    ("engine-softbody-authoring", "test", "generated_vehicle_lint"): 150,
    ("engine-softbody-authoring", "test", "every_design"): 120,
    ("engine-softbody", "lib", "engine_softbody"): 25,
    ("engine-softbody-authoring", "test", "raycast_wheel_visual"): 25,
    ("engine-softbody-authoring", "test", "explicit_drive"): 16,
    ("engine-softbody-authoring", "test", "sedan_jello"): 14,
    ("engine-softbody-authoring", "test", "rig_export"): 14,
    ("engine-softbody-authoring", "test", "energy_ledger"): 14,
}
DEFAULT_RUN_SECONDS = 2
# Compiling and linking one target (release, thin LTO), on four cores: an estimate
# from the whole build (about 130 targets in 19.6 min).
BUILD_SECONDS = 9
# Checking one example (no code generation, no link).
CHECK_SECONDS = 2

meta = json.loads(
    subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
)
units = []  # (seconds, output line)
docs = []
for package in packages:
    pkgs = [p for p in meta["packages"] if p["name"] == package]
    assert len(pkgs) == 1, f"package {package} not in the workspace"
    doc = False
    for t in pkgs[0]["targets"]:
        kind, name = t["kind"][0], t["name"]
        assert not t.get("required-features"), (
            f"{name}: a target with required-features needs its own rule here"
        )
        if kind in ("lib", "rlib", "cdylib", "staticlib", "dylib", "proc-macro"):
            doc = doc or t.get("doctest", False)
            if t["test"]:
                units.append((BUILD_SECONDS + RUN_SECONDS.get((package, "lib", name), DEFAULT_RUN_SECONDS), f"{package} test --lib"))
        elif kind == "bin":
            if t["test"]:
                units.append((BUILD_SECONDS + RUN_SECONDS.get((package, "bin", name), DEFAULT_RUN_SECONDS), f"{package} test --bin={name}"))
        elif kind == "test":
            if t["test"]:
                units.append((BUILD_SECONDS + RUN_SECONDS.get((package, "test", name), DEFAULT_RUN_SECONDS), f"{package} test --test={name}"))
        elif kind == "example":
            # A plain `cargo test` runs an example's tests only with `test = true`.
            assert not t["test"], f"example {name} has test = true: it needs its own rule here"
            units.append((CHECK_SECONDS, f"{package} example {name}"))
        elif kind in ("bench", "custom-build"):
            pass  # `cargo test` builds no benches; the build script is not a target to pick
        else:
            raise SystemExit(f"{name}: unknown target kind {kind}")
    if doc:
        docs.append(package)

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
if shard == 1:
    for package in docs:
        print(f"{package} doc")
