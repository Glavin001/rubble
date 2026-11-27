#!/usr/bin/env python3
"""One table of `crash_verdict` results for front-rules variants: per variant and scenario/setting
the new failures (not known), and the known failures that no longer occur.

    python3 -I tools/sections/front_verdicts.py off cell cr5 call ...
"""
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", ".."))


def main():
    print("| variant | settings judged | new failures (scenario, setting: criterion, runs) | known failures that now pass in every run |")
    print("|---|---|---|---|")
    for tag in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(ENGINE, "out", "gen", f"hard_{tag}", "S*.jsonl")))
        if not files:
            continue
        out = subprocess.run([os.path.join(ENGINE, "target/release/examples/crash_verdict"), *files], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL).stdout
        new, gone = {}, set()
        for line in out.splitlines():
            m = re.match(r"\s+FAILED (S\d\d) run (\d+) \((\w+)\): (.*?) = ", line)
            if m:
                k = (m.group(1), m.group(3), m.group(4))
                new.setdefault(k, set()).add(int(m.group(2)))
                continue
            m = re.match(r"\s+FAILED (S\d\d) \((\w+)\): (.*?): known failure now passes", line)
            if m:
                gone.add(f"{m.group(1)} {m.group(2)}: {m.group(3)}")
        n = ", ".join(f"{s} {c}: {crit} {len(r)}/8" for (s, c, crit), r in sorted(new.items())) or "none"
        print(f"| {tag} | {len(files)} | {n} | {'; '.join(sorted(gone)) or '–'} |")


if __name__ == "__main__":
    main()
