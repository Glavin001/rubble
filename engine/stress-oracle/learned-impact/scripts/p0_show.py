#!/usr/bin/env python3
"""One summary line per learned_p0 result file."""
import json, sys
for p in sys.argv[1:]:
    try:
        d = json.load(open(p))
    except Exception as e:
        print(p, "unreadable", e); continue
    e = d["end"]; en = e["energy"]
    res = (en["mechanical"] + en["dissipated"] - d["e0"]) / d["e0"]
    fr = [f"{f['ram_vy']:.1f}/{f['fragments']}" + ("*" if f.get("islands") else "") for f in d["frames"]]
    print(f"{p.split('/')[-1]:40s} {d.get('variant','oracle'):15s} ram {e['ram_vy']:7.3f} frag {e['fragments']:4d} broken {e['broken_bonds']:5d} largest {e['largest_fraction']:.3f} calls {d.get('calls','-')} resid {res:+.4f} {d['total_s']:.1f}s | {' '.join(fr)}")
