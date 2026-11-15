#!/usr/bin/env python3
"""Checks a `cargo test --no-fail-fast` log against engine/tests-expected-failures.txt.

Usage: expected-failures.py LOG LIST CARGO_EXIT_CODE

Passes (exit 0) when every failing test is listed and the run otherwise completed:
no build error, no crashed test binary. Fails (exit 1) when a test not on the list
fails, when a listed test passes (it is fixed: take it off the list), or when a
listed test did not run at all (renamed or removed: update the list).
"""
import re
import sys

log_path, list_path, code = sys.argv[1], sys.argv[2], int(sys.argv[3])
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
missing = sorted(expected - failed - passed)
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
print("OK: every failure is a listed, tracked expected failure (engine/tests-expected-failures.txt)")
