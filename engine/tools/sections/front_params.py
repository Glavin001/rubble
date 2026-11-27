#!/usr/bin/env python3
"""A design's parameters with a set of front rules (`Physics::front_rules`, lessons §25).

    python3 -I tools/sections/front_params.py DESIGN OUT.json [rule ...] [all] [crash=PRESET]

From engine/: reads the design's own defaults (`vehicle_gen_rig DESIGN --defaults`, the
physics block whole, so a design whose physics differs from the default block, the 4x4's,
keeps it) and writes them with `physics.front_rules` set (the rules named on, every other off)
and optionally `physics.crash`. The file is a `--params` file for `vehicle_gen_rig` and
`crash_ref --params DESIGN=FILE`.
"""
import json
import subprocess
import sys

RULES = ["subframe_mounts_off_crash_box", "crash_box_bay_alone", "crash_box_stroke", "yielding_stops", "hood_fold_row", "crush_from_section"]


def main():
    design, out, *args = sys.argv[1:]
    p = json.loads(subprocess.run(["target/release/examples/vehicle_gen_rig", design, "--defaults"], check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120).stdout)
    rules = {r: False for r in RULES}
    for a in args:
        if a.startswith("crash="):
            p["physics"]["crash"] = a.split("=", 1)[1]
        elif a == "all":
            rules = {r: True for r in RULES}
        elif a in rules:
            rules[a] = True
        else:
            sys.exit(f"unknown rule {a} (known: {', '.join(RULES)})")
    p["physics"]["front_rules"] = rules
    with open(out, "w") as f:
        json.dump(p, f)


if __name__ == "__main__":
    main()
