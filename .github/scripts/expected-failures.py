#!/usr/bin/env python3
"""Checks a `cargo test --no-fail-fast` log against engine/tests-expected-failures.txt.

Usage: expected-failures.py [--partial] LOG LIST CARGO_EXIT_CODE

Passes (exit 0) when every failing test is listed and the run otherwise completed:
no build error, no crashed test binary. Fails (exit 1) when a test not on the list
fails, when a listed test passes (it is fixed: take it off the list), or when a
listed test did not run at all (renamed or removed: update the list).

--partial: LOG is one shard of the run (CI splits the host tests over parallel jobs,
.github/scripts/host-tests.sh). A listed test that is not in it is not a problem
there; the job that joins the shards' logs runs the full check.
"""
import re
import sys

args = sys.argv[1:]
partial = "--partial" in args
args = [a for a in args if a != "--partial"]
log_path, list_path, code = args[0], args[1], int(args[2])
log = open(log_path, encoding="utf-8", errors="replace").read()
log = re.sub(r"\x1b\[[0-9;]*m", "", log)

expected = set()
for line in open(list_path, encoding="utf-8"):
    line = line.split("#", 1)[0].strip()
    if line:
        expected.add(line)

failed, passed = set(), set()
stem = None
for line in log.splitlines():
    m = re.search(r"Running (?:tests|src|benches|examples)/(\S+?)\.rs\b", line)
    if m:
        stem = m.group(1).replace("/", "::")
        continue
    m = re.search(r"Running unittests (\S+)", line)
    if m:
        stem = "unittests:" + m.group(1)
        continue
    m = re.match(r"\s*test (\S+) \.\.\. (FAILED|ok)\s*$", line)
    if m and stem:
        name = f"{stem}::{m.group(1)}"
        (failed if m.group(2) == "FAILED" else passed).add(name)

problems = []
if re.search(r"error(\[E\d+\])?: could not compile|error: could not compile", log):
    problems.append("a crate failed to build")
if re.search(r"\(signal: \d+", log):
    problems.append("a test binary was killed by a signal (crash)")
if code != 0 and not failed:
    problems.append(f"cargo exited {code} with no failing test reported")

unexpected = sorted(failed - expected)
fixed = sorted(expected & passed)
missing = [] if partial else sorted(expected - failed - passed)
if unexpected:
    problems.append("failing tests not on the expected list (new failures):\n  " + "\n  ".join(unexpected))
if fixed:
    problems.append("listed tests now pass (remove them from the list):\n  " + "\n  ".join(fixed))
if missing:
    problems.append("listed tests that did not run (renamed or removed? update the list):\n  " + "\n  ".join(missing))

print(f"tests passed {len(passed)}, failed {len(failed)} ({len(failed & expected)} expected, listed {len(expected)})")
if problems:
    print("FAIL:\n" + "\n".join("- " + p for p in problems))
    sys.exit(1)
if partial:
    print("OK (this shard): every failure is a listed, tracked expected failure (engine/tests-expected-failures.txt)")
else:
    print("OK: every failure is a listed, tracked expected failure (engine/tests-expected-failures.txt)")
