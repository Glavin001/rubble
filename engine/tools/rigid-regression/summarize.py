#!/usr/bin/env python3
"""Summarise driver.py's CSV: per scene and build, the median over rounds of each run's
p50 wall and p50 GPU step time with the spread (min..max over rounds), the head's delta
against each other build, and whether each build's final-state hashes agree.

usage: summarize.py RESULTS_CSV [HEAD_LABEL]
"""
import collections, statistics, sys

path = sys.argv[1]
head = sys.argv[2] if len(sys.argv) > 2 else "head"
rows = collections.defaultdict(lambda: collections.defaultdict(list))
hashes = collections.defaultdict(lambda: collections.defaultdict(set))
errors = []
order = []
for line in open(path):
    f = line.rstrip("\n").split(",")
    key, label = f[1], f[2]
    if "ERROR" in line:
        errors.append(line.strip())
        continue
    if key not in order:
        order.append(key)
    if f[3] == "city":
        wall, gpu = float(f[6]), float(f[8])
    else:
        wall, gpu = float(f[6]), float(f[8])
        hashes[key][label].add(f[12])
    rows[key][label].append((wall, gpu))

labels = []
for k in order:
    for l in rows[k]:
        if l not in labels:
            labels.append(l)
others = [l for l in labels if l != head]


def med(v):
    return statistics.median(v) if v else float("nan")


def cell(v):
    if not v:
        return "-"
    return f"{med(v):.3f} ({min(v):.3f}..{max(v):.3f})"


def delta(a, b):
    if not a or not b or med(b) == 0 or med(b) != med(b):
        return "-"
    return f"{100 * (med(a) - med(b)) / med(b):+.1f}%"


for metric, idx in (("wall p50 ms (step + poll)", 0), ("GPU p50 ms (timestamp, steps that ran)", 1)):
    print(f"\n### {metric}: median over rounds (min..max)\n")
    print("| scene | n | " + " | ".join(labels) + " | " + " | ".join(f"{head} vs {o}" for o in others) + " |")
    print("|---|---|" + "---|" * (len(labels) + len(others)))
    for k in order:
        vals = {l: [x[idx] for x in rows[k][l] if x[idx] == x[idx]] for l in labels}
        n = min(len(rows[k][l]) for l in labels if rows[k][l])
        print(f"| {k} | {n} | " + " | ".join(cell(vals[l]) for l in labels) + " | "
              + " | ".join(delta(vals[head], vals[o]) for o in others) + " |")

print("\n### Final-state hashes (bench scenes)\n")
print("| scene | " + " | ".join(labels) + " | all builds equal |")
print("|---|" + "---|" * (len(labels) + 1))
for k in order:
    if k not in hashes:
        continue
    hs = hashes[k]
    allh = set().union(*hs.values())
    print(f"| {k} | " + " | ".join(
        (next(iter(hs[l])) if len(hs[l]) == 1 else f"{len(hs[l])} distinct") for l in labels)
        + f" | {'yes' if len(allh) == 1 else 'no'} |")
if errors:
    print("\n### Errors\n")
    for e in errors:
        print("- " + e)
